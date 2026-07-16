package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Post-processing windows for the per-table revision history.
const (
	sameUserGroupWindow = 1 * time.Hour      // timespan to ignore/summarize edits of same user
	rollbackWindow      = 7 * 24 * time.Hour // timespan for ignoring rollbacks
)

// matchColumns assigns a stable UUID to each column of the current table version
// by matching its columns to the previous version's columns. First by equal,
// non-empty header, then by positional proximity for whatever remains.
// Matched columns inherit the previous UUID, unmatched current columns get a
// fresh one. Columns present only in the previous version simply drop out. This
// keeps a column's identity stable across insertions, deletions, and reorderings
// as long as headers are present; header-less tables fall back to index-based positional matching.
func matchColumns(prevHeaders, prevUUIDs, curHeaders []string) []string {
	nCur := len(curHeaders)
	nPrev := len(prevHeaders)
	curUUIDs := make([]string, nCur)

	matchedPrev := make([]bool, nPrev)
	matchedCur := make([]bool, nCur)

	// stage runs one bipartite matching over the still-unmatched columns.
	// headerOnly restricts edges to equal, non-empty headers; otherwise every
	// remaining pair is a candidate (positional fallback). Within a stage, ties
	// prefer columns at nearby positions.
	stage := func(headerOnly bool) {
		var rp, rc []int
		for p := 0; p < nPrev; p++ {
			if !matchedPrev[p] {
				rp = append(rp, p)
			}
		}
		for c := 0; c < nCur; c++ {
			if !matchedCur[c] {
				rc = append(rc, c)
			}
		}
		if len(rp) == 0 || len(rc) == 0 {
			return
		}
		w := make([][]float64, len(rp))
		any := false
		for i, p := range rp {
			w[i] = make([]float64, len(rc))
			for j, c := range rc {
				if headerOnly && (prevHeaders[p] == "" || prevHeaders[p] != curHeaders[c]) {
					continue // no edge
				}
				d := p - c
				if d < 0 {
					d = -d
				}
				w[i][j] = 1.0 / (1.0 + float64(d)) // positive edge, prefers nearby columns
				any = true
			}
		}
		if !any {
			return
		}
		for i, jj := range maxWeightMatching(w) {
			if jj < 0 {
				continue
			}
			p, c := rp[i], rc[jj]
			curUUIDs[c] = prevUUIDs[p]
			matchedPrev[p] = true
			matchedCur[c] = true
		}
	}

	stage(true)  // header match
	stage(false) // positional fallback for the rest

	for c := 0; c < nCur; c++ {
		if !matchedCur[c] {
			curUUIDs[c] = uuid.NewString()
		}
	}
	return curUUIDs
}

func intOrStr(s string) any {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return s
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	cp := strings.Clone(s)
	return &cp
}

type tableInfo struct {
	origIdx     int
	bow         map[string]int // content bag-of-words used by the matcher
	fingerprint [16]byte
	maxCols     int
	headers     []string // per-column header text (len == maxCols), for column matching
	empty       bool
}

// computeTableInfo parses one table string and extracts bow/fp/maxCols/headers.
// The VisualEditor placeholder table is treated as empty so it is never tracked.
func computeTableInfo(origIdx int, ttext string) tableInfo {
	rawRows := tableToRawRows(ttext)
	if len(rawRows) == 0 || isEditorBoilerplate(rawRows) {
		return tableInfo{empty: true}
	}
	maxCols := maxColsRaw(rawRows)
	return tableInfo{
		origIdx:     origIdx,
		bow:         bagOfWords(rawRows),
		fingerprint: fingerprintRaw(rawRows),
		maxCols:     maxCols,
		headers:     columnHeaders(rawRows, maxCols),
	}
}

// revMeta is the lightweight per-revision record kept in memory for each table.
// Entries align 1:1 with the JSONL lines spooled to the table's temp file, so the
// rollback / same-user-grouping post-processing can decide which lines to emit
// without re-reading the (potentially large) cell payloads.
type revMeta struct {
	fp       [16]byte  // raw fingerprint of this revision's table
	userKey  string    // username, else ip, else contributor id, else ""
	ts       time.Time // revision timestamp (UTC), for the time-window filters
	revIDAny any       // revision id
	revDate  string    // pretty revision date
}

// tableState tracks one logical table across all revisions of a page
type tableState struct {
	tableID    string
	colUUIDs   []string // stable UUID per column of the last written version
	colHeaders []string // header text per column of the last written version (for column matching)
	lastFP     [16]byte // fingerprint of the last written revision; used to skip duplicates
	tmpFile    *os.File // revision entries written here as JSONL, one per line
	tmpPath    string

	// Matcher state (Bleifuß et al.): the rearview window of recent content
	// bag-of-words, the last position-rank, and the number of versions so far.
	bowHist  []map[string]int
	lastPos  int
	lifetime int

	// Per-written-revision metadata, aligned 1:1 with the temp-file lines.
	revs []revMeta

	//both get cleared if table reappears later
	deletedInRevID   any
	deletedInRevDate string
}

// pushBOW appends a content vector to the rear-view history, capped at kWindow.
func (st *tableState) pushBOW(bow map[string]int) {
	st.bowHist = append(st.bowHist, bow)
	if len(st.bowHist) > kWindow {
		st.bowHist = st.bowHist[len(st.bowHist)-kWindow:]
	}
}

func (st *tableState) close() {
	if st.tmpFile != nil {
		st.tmpFile.Close()
		st.tmpFile = nil
	}
}

func (st *tableState) remove() {
	if st.tmpPath != "" {
		os.Remove(st.tmpPath)
		st.tmpPath = ""
	}
}

type pageState struct { // mutable state accumulated while processing one <page>
	id     string // raw string form of the page ID (may be non-numeric)
	idAny  any    // int or string depending on intOrStr result; used in JSON output
	title  string
	tables []*tableState
	tmpDir string // directory for temp spool files; "" means the OS default
}

// reset clears all fields and closes/removes any open spool files so the
// struct can be reused for the next page without a heap allocation.
func (ps *pageState) reset() {
	ps.id = ""
	ps.idAny = nil
	ps.title = ""
	for i, st := range ps.tables {
		st.close()
		st.remove()
		ps.tables[i] = nil
	}
	ps.tables = ps.tables[:0]
}

func tableFormat(raw string) string {
	if strings.HasPrefix(strings.TrimSpace(raw), "{|") {
		return "wikitable"
	}
	return "html"
}

func updateDeletionMarkers(ps *pageState, nTracked int, matchedTracked map[int]bool, revID, revDate string) {
	var zero [16]byte
	for ti := 0; ti < nTracked; ti++ {
		state := ps.tables[ti]
		if matchedTracked[ti] {
			// Table is present in this revision — clear any pending deletion marker.
			state.deletedInRevID = nil
			state.deletedInRevDate = ""
		} else if state.deletedInRevID == nil && state.lastFP != zero {
			// Table was seen before but absent in this revision — record the
			// first revision in which it went missing.
			state.deletedInRevID = intOrStr(revID)
			state.deletedInRevDate = revDate
		}
	}
}

// contributorKey reduces a contributor to a stable identity key for the
// same-user grouping filter: username, else IP, else contributor id, else "".
func contributorKey(c Contributor) string {
	if c.Username != nil && *c.Username != "" {
		return "u:" + *c.Username
	}
	if c.IP != nil && *c.IP != "" {
		return "ip:" + *c.IP
	}
	if c.ID != nil && *c.ID != "" {
		return "id:" + *c.ID
	}
	return ""
}

func processRevision(ps *pageState, revID, revDate string, revTime time.Time, contrib Contributor, text string) {
	// nTracked captures how many tables were tracked before this revision so
	// that updateDeletionMarkers only considers pre-existing tables and never
	// mistakes a newly created table for a deleted one.
	nTracked := len(ps.tables)

	rawTables := extractTables(text)
	if len(rawTables) == 0 {
		updateDeletionMarkers(ps, nTracked, nil, revID, revDate)
		return
	}

	// Pass 1: compute bow/fp/maxCols for each table (boilerplate -> empty).
	infos := make([]tableInfo, len(rawTables))
	anyValid := false
	for i, ttext := range rawTables {
		infos[i] = computeTableInfo(i, ttext)
		if !infos[i].empty {
			anyValid = true
		}
	}
	if !anyValid {
		updateDeletionMarkers(ps, nTracked, nil, revID, revDate)
		return
	}

	// Build the current object list for matching (position = source-order rank among the valid tables)
	cur := make([]curObj, 0, len(rawTables))
	validIdx := make([]int, 0, len(rawTables)) // maps cur index to rawTables index
	for i, info := range infos {
		if !info.empty {
			cur = append(cur, curObj{bow: info.bow, pos: len(validIdx)})
			validIdx = append(validIdx, i)
		}
	}
	matches := matchTables(ps.tables, cur)

	revIDany := intOrStr(revID)
	userKey := contributorKey(contrib)

	// Pass 2: reparse each valid table, marshal immediately, free cells
	for si, rawIdx := range validIdx {
		info := infos[rawIdx]
		m := matches[si]
		pos := cur[si].pos

		var state *tableState
		var matchScore *float64
		if m.ti >= 0 {
			state = ps.tables[m.ti]
			// Position can change even when content is identical; keep it fresh
			// so the next revision's local search stays accurate.
			state.lastPos = pos
			if info.fingerprint == state.lastFP {
				continue // unchanged content: dedup, do not spool a new line
			}
			s := m.score
			matchScore = &s
			state.pushBOW(info.bow)
			state.lifetime++
		} else {
			f, err := os.CreateTemp(ps.tmpDir, fmt.Sprintf("wt-%v-*.jsonl", ps.id))
			if err != nil {
				fmt.Fprintf(os.Stderr, "temp file error: %v\n", err)
				continue
			}
			state = &tableState{
				tableID:  fmt.Sprintf("%s-%d", revID, info.origIdx),
				tmpFile:  f,
				tmpPath:  f.Name(),
				lastPos:  pos,
				lifetime: 1,
			}
			state.pushBOW(info.bow)
			ps.tables = append(ps.tables, state)
		}

		state.lastFP = info.fingerprint
		// Assign a stable UUID to each column by matching this version's columns
		// to the previous version's (by header, then position); new columns get a
		// fresh UUID. For a new table (colHeaders nil) every column is new.
		state.colUUIDs = matchColumns(state.colHeaders, state.colUUIDs, info.headers)
		state.colHeaders = info.headers

		hdrBytes, hdrErr := json.Marshal(revisionHeader{
			RevisionID:              revIDany,
			RevisionDate:            revDate,
			Contributor:             contrib,
			ArtificialColumnHeaders: append([]string(nil), state.colUUIDs...),
			Format:                  tableFormat(rawTables[rawIdx]),
			MatchScore:              matchScore,
		})

		if hdrErr != nil {
			fmt.Fprintf(os.Stderr, "json marshal error: %v\n", hdrErr)
			continue
		}

		// json.Marshal produces a complete JSON object ending with '}'.
		// Strip that closing brace so we can inject the "cells" field and
		// then close the object ourselves, forming one complete JSONL record.
		state.tmpFile.Write(hdrBytes[:len(hdrBytes)-1])
		state.tmpFile.WriteString(`,"cells":`)
		if cellErr := tableCellsToJSON(rawTables[rawIdx], state.tmpFile); cellErr != nil {
			fmt.Fprintf(os.Stderr, "json cells error: %v\n", cellErr)
			continue
		}
		state.tmpFile.WriteString("}\n")

		// Record metadata aligned 1:1 with the line just spooled.
		state.revs = append(state.revs, revMeta{
			fp:       info.fingerprint,
			userKey:  userKey,
			ts:       revTime,
			revIDAny: revIDany,
			revDate:  revDate,
		})
	}

	// Build the set of pre-existing tracked tables that were matched in this
	// revision, then update deletion markers for all pre-existing tables.
	matchedTracked := make(map[int]bool, len(matches))
	for _, m := range matches {
		if m.ti >= 0 {
			matchedTracked[m.ti] = true
		}
	}
	updateDeletionMarkers(ps, nTracked, matchedTracked, revID, revDate)
}

// computeKeep decides which of a table's spooled revisions survive the two
// history filters, returning a mask aligned 1:1 with revs:
//
//  1. same-user grouping: within sameUserGroupWindow, edits of the same user collapse to its last
//  2. rollback exclusion: within rollbackWindow, the intervening excursion (and the returning revision)
//     are dropped; a return after the window is a genuine reappearance and is kept.
func computeKeep(revs []revMeta) []bool {
	n := len(revs)
	keep := make([]bool, n)
	for i := range keep {
		keep[i] = true
	}

	// Pass 1: same-user grouping.
	for i := 0; i < n; {
		j := i
		for j+1 < n &&
			revs[j+1].userKey != "" && revs[j+1].userKey == revs[i].userKey &&
			revs[j+1].ts.Sub(revs[j].ts) >= 0 &&
			revs[j+1].ts.Sub(revs[j].ts) <= sameUserGroupWindow {
			j++
		}
		if revs[i].userKey != "" {
			for k := i; k < j; k++ { // keep only the last of the run (index j)
				keep[k] = false
			}
		}
		i = j + 1
	}

	// Pass 2: rollback exclusion over the survivors.
	fpAt := map[[16]byte]int{} // fingerprint → index into result
	var result []int           // surviving original indices, in order
	for oi := 0; oi < n; oi++ {
		if !keep[oi] {
			continue
		}
		fp := revs[oi].fp
		if pos, ok := fpAt[fp]; ok {
			if pos+1 >= len(result) {
				// The matching state is the immediately preceding survivor, so
				// the grouping pass already removed whatever excursion sat
				// between them: this revision is just a duplicate of what is
				// recorded, with no excursion to time. Drop it.
				keep[oi] = false
				continue
			}
			// Measure the excursion's own duration, from the first revision that
			// departed the earlier state until this return. Not the time since the
			// earlier state itself. Otherwise a page stable for >X days that is
			// vandalized and reverted within minutes would not register as a
			// rollback.
			if revs[oi].ts.Sub(revs[result[pos+1]].ts) <= rollbackWindow {
				// Rollback: drop everything after the earlier matching state, plus
				// this returning revision.
				for p := pos + 1; p < len(result); p++ {
					keep[result[p]] = false
					delete(fpAt, revs[result[p]].fp)
				}
				keep[oi] = false
				result = result[:pos+1]
				continue
			}
			// Returned after the window: a genuine reappearance, keep it.
		}
		result = append(result, oi)
		fpAt[fp] = len(result) - 1
	}
	return keep
}

// flushPage reads each table's spool file and writes one JSONL record per tracked table to out
func flushPage(ps *pageState, out *bufio.Writer) error {
	for _, tableState := range ps.tables {
		if tableState.tmpFile == nil {
			continue
		}
		tableState.tmpFile.Sync()
		tableState.tmpFile.Close()
		tableState.tmpFile = nil

		f, err := os.Open(tableState.tmpPath)
		if err != nil {
			os.Remove(tableState.tmpPath)
			continue
		}

		keep := computeKeep(tableState.revs)

		var deletedAt *deletionInfo
		if tableState.deletedInRevID != nil {
			deletedAt = &deletionInfo{
				RevisionID:   tableState.deletedInRevID,
				RevisionDate: tableState.deletedInRevDate,
			}
		}

		prefix, err := json.Marshal(struct {
			PageID    any           `json:"pageID"`
			PageTitle string        `json:"pageTitle"`
			TableID   string        `json:"tableID"`
			DeletedAt *deletionInfo `json:"deletedAt,omitempty"`
		}{ps.idAny, ps.title, tableState.tableID, deletedAt})

		if err != nil {
			f.Close()
			os.Remove(tableState.tmpPath)
			continue
		}

		br := bufio.NewReaderSize(f, 4*1024*1024)
		first := true
		wroteAny := false
		lineIdx := 0 // index into tableState.revs / keep
		for {
			line, err := br.ReadBytes('\n')
			if len(line) > 0 && line[len(line)-1] == '\n' {
				line = line[:len(line)-1]
			}
			if len(line) > 0 {
				kept := lineIdx >= len(keep) || keep[lineIdx]
				if kept {
					if first {
						out.Write(prefix[:len(prefix)-1]) // open the output object
						out.WriteString(`,"tables":[`)
						first = false
					} else {
						out.WriteByte(',')
					}
					out.Write(line)
					wroteAny = true
				}
				lineIdx++
			}
			if err != nil {
				break
			}
		}
		if wroteAny {
			out.WriteString("]}\n")
		}
		f.Close()
		os.Remove(tableState.tmpPath)
		tableState.tmpPath = ""
	}
	return nil
}
