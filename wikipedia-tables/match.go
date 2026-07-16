package main

import (
	"crypto/md5"
	"encoding/binary"
	"math"
	"regexp"
	"strings"
)

var reHTMLTag = regexp.MustCompile(`<[^>]+>`)
var reWhitespace = regexp.MustCompile(`\s+`)

// fingerprintRaw hashes unexpanded raw rows for cheap revision deduplication
func fingerprintRaw(rawRows [][]rawCell) [16]byte {
	h := md5.New()
	var spans [8]byte
	for _, row := range rawRows {
		for _, c := range row {
			h.Write([]byte(strings.TrimSpace(c.content)))
			// Fixed-width, untruncated spans: parseSpan admits values far above
			// 255, so a single byte each would alias (colspan 257 == colspan 1)
			// and silently dedup a real structural change away.
			binary.BigEndian.PutUint32(spans[0:4], uint32(c.colspan))
			binary.BigEndian.PutUint32(spans[4:8], uint32(c.rowspan))
			h.Write(spans[:])
		}
		h.Write([]byte{0xff}) //0xff between rows to distinguish cell and row boundary shift
	}
	var out [16]byte
	copy(out[:], h.Sum(nil))
	return out
}

func maxColsRaw(rawRows [][]rawCell) int {
	maxCols := 0
	for _, row := range rawRows {
		rowWidth := 0
		for _, cell := range row {
			rowWidth += cell.colspan
		}
		if rowWidth > maxCols {
			maxCols = rowWidth
		}
	}
	return maxCols
}

// Three-stage temporal object matching (Bleifuß et al., "Structured Object
// Matching across Web Page Revisions", 2021).
//
// Matching runs in three stages of increasing permissiveness, each computing a
// maximum-weight bipartite matching over the still-unmatched objects:
//
//  1. local  — only position-neighbours, sim_strict (Ruzicka)
//  2. strict — all remaining pairs, sim_strict (Ruzicka)
//  3. relaxed— all remaining pairs, sim_relaxed (containment)
//
// A new instance is compared to the lastkWindow versions of a tracked object,
// weighted by decayFactor^age, so that short-lived vandalism does not break a
// table's identity. Token frequencies areweighted by IDF-like factor.

const (
	kWindow      = 5   // rearview window: number of past object versions considered
	posWindow    = 2   // local-search position neighborhood (θ_POS)
	decayFactor  = 0.9 // exponential decay applied to older versions
	thetaStrict  = 0.6 // minimum sim_strict to accept a match (stages 1 & 2)
	thetaRelaxed = 0.3 // minimum sim_relaxed to accept a match (stage 3)

	// Tie-breaks. Position difference dominates lifetime, realizing the paper's ↓POS-then-↓LT ordering.
	// The magnitudes are chosen so a single position unit (tiePosCoef) strictly outweighs the
	// largest possible lifetime term (tieLTCoef·tieLTCap)
	tiePosCoef = 1e-7
	tieLTCoef  = 1e-14
	tieLTCap   = 1e6
)

// curObj is one current table instance presented to the matcher.
type curObj struct {
	bow map[string]int // content bag-of-words
	pos int            // 0-based position-rank among valid tables in source order
}

// matchResult holds the outcome for one current table.
// ti is the index of the matched tracked table, or -1 for a new table.
type matchResult struct {
	ti    int     // index of the matched tracked table (or -1)
	score float64 // similarity score of the accepted match, 0 when unmatched
}

// tokenWeights computes the IDF-like weight for every token occurring in any
// tracked (most-recent version) or current object:
//
//	w_t = 1 / max(df_prev_t, df_new_t)
//
// where df_* is the number of objects (floored at 1) containing the token. A
// token appearing in ≤1 previous and ≤1 current object keeps weight 1.
func tokenWeights(tracked []*tableState, cur []curObj) map[string]float64 {
	dfPrev := map[string]int{}
	for _, t := range tracked {
		if len(t.bowHist) == 0 {
			continue
		}
		recent := t.bowHist[len(t.bowHist)-1]
		for tok := range recent {
			dfPrev[tok]++
		}
	}
	dfNew := map[string]int{}
	for _, o := range cur {
		for tok := range o.bow {
			dfNew[tok]++
		}
	}
	w := make(map[string]float64, len(dfPrev)+len(dfNew))
	assign := func(tok string) {
		if _, ok := w[tok]; ok {
			return
		}
		dPrev, dNew := dfPrev[tok], dfNew[tok]
		if dPrev < 1 {
			dPrev = 1
		}
		if dNew < 1 {
			dNew = 1
		}
		m := dPrev
		if dNew > m {
			m = dNew
		}
		w[tok] = 1.0 / float64(m)
	}
	for tok := range dfPrev {
		assign(tok)
	}
	for tok := range dfNew {
		assign(tok)
	}
	return w
}

func weightOf(w map[string]float64, tok string) float64 {
	if v, ok := w[tok]; ok {
		return v
	}
	return 1.0
}

// simStrict is the weighted Ruzicka (generalized Jaccard) similarity
func simStrict(a, b map[string]int, w map[string]float64) float64 {
	var num, den float64
	seen := make(map[string]bool, len(a)+len(b))
	acc := func(tok string) {
		if seen[tok] {
			return
		}
		seen[tok] = true
		va := float64(a[tok]) * weightOf(w, tok)
		vb := float64(b[tok]) * weightOf(w, tok)
		num += math.Min(va, vb)
		den += math.Max(va, vb)
	}
	for tok := range a {
		acc(tok)
	}
	for tok := range b {
		acc(tok)
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// simRelaxed is the weighted element-wise containment similarity:
// Σ min(v,w) / min(Σ v, Σ w), tolerant of growth/shrinkage.
func simRelaxed(a, b map[string]int, w map[string]float64) float64 {
	var num, sumA, sumB float64
	seen := make(map[string]bool, len(a)+len(b))
	acc := func(tok string) {
		if seen[tok] {
			return
		}
		seen[tok] = true
		va := float64(a[tok]) * weightOf(w, tok)
		vb := float64(b[tok]) * weightOf(w, tok)
		num += math.Min(va, vb)
	}
	for tok := range a {
		acc(tok)
	}
	for tok := range b {
		acc(tok)
	}
	for tok, c := range a {
		sumA += float64(c) * weightOf(w, tok)
	}
	for tok, c := range b {
		sumB += float64(c) * weightOf(w, tok)
	}
	den := math.Min(sumA, sumB)
	if den == 0 {
		return 0
	}
	return num / den
}

// simHat applies the rearview mirror: the maximum over the tracked object's
// last kWindow versions of decayFactor^age * sim(version, o).
func simHat(tr *tableState, o map[string]int, w map[string]float64, relaxed bool) float64 {
	n := len(tr.bowHist)
	if n == 0 {
		return 0
	}
	limit := n
	if limit > kWindow {
		limit = kWindow
	}
	best := 0.0
	decay := 1.0
	for i := 0; i < limit; i++ {
		ver := tr.bowHist[n-1-i] // i=0 is the most recent version
		var s float64
		if relaxed {
			s = simRelaxed(ver, o, w)
		} else {
			s = simStrict(ver, o, w)
		}
		if v := decay * s; v > best {
			best = v
		}
		decay *= decayFactor
	}
	return best
}

// matchTables assigns each current table to a tracked table or marks it new,
// running the three matching stages described above.
func matchTables(tracked []*tableState, cur []curObj) []matchResult {
	result := make([]matchResult, len(cur))
	for i := range result {
		result[i] = matchResult{ti: -1}
	}
	if len(tracked) == 0 || len(cur) == 0 {
		return result
	}

	w := tokenWeights(tracked, cur)

	matchedT := make([]bool, len(tracked))
	matchedC := make([]bool, len(cur))

	stages := []struct {
		local   bool
		relaxed bool
		theta   float64
	}{
		{local: true, relaxed: false, theta: thetaStrict},
		{local: false, relaxed: false, theta: thetaStrict},
		{local: false, relaxed: true, theta: thetaRelaxed},
	}

	for _, st := range stages {
		// Gather still-unmatched tracked / current indices.
		var rt, rc []int
		for ti := range tracked {
			if !matchedT[ti] {
				rt = append(rt, ti)
			}
		}
		for ci := range cur {
			if !matchedC[ci] {
				rc = append(rc, ci)
			}
		}
		if len(rt) == 0 || len(rc) == 0 {
			continue
		}

		// Build real-score and perturbed-weight matrices over rt * rc.
		real := make([][]float64, len(rt))
		pert := make([][]float64, len(rt))
		anyEdge := false
		for ri, ti := range rt {
			real[ri] = make([]float64, len(rc))
			pert[ri] = make([]float64, len(rc))
			for ci, cj := range rc {
				posDiff := tracked[ti].lastPos - cur[cj].pos
				if posDiff < 0 {
					posDiff = -posDiff
				}
				if st.local && posDiff > posWindow {
					continue // outside local neighborhood: no edge
				}
				s := simHat(tracked[ti], cur[cj].bow, w, st.relaxed)
				if s < st.theta {
					continue // below threshold: no edge
				}
				real[ri][ci] = s
				lt := float64(tracked[ti].lifetime)
				if lt > tieLTCap {
					lt = tieLTCap
				}
				pert[ri][ci] = s - tiePosCoef*float64(posDiff) + tieLTCoef*lt
				anyEdge = true
			}
		}
		if !anyEdge {
			continue
		}

		assign := maxWeightMatching(pert)
		for ri, ci := range assign {
			if ci < 0 {
				continue
			}
			if real[ri][ci] < st.theta { // padding / non-edge assignment
				continue
			}
			ti, cj := rt[ri], rc[ci]
			matchedT[ti] = true
			matchedC[cj] = true
			result[cj] = matchResult{ti: ti, score: real[ri][ci]}
		}
	}
	return result
}

// maxWeightMatching returns, for each row, the column it is matched to (or -1),
// maximizing the total weight of selected cells. Weights ≤ 0 are treated as
// "no edge" and never produce a real match.
//
// The Hungarian solver runs on a cost matrix (cost = −weight) oriented so the
// SMALLER dimension forms the rows. This matters enormously when a page
// has accumulated thousands of tracked tables but a revision contains only a
// few: matching 1 current table against 5000 tracked is then ~5000 ops.
func maxWeightMatching(weight [][]float64) []int {
	n := len(weight)
	if n == 0 {
		return nil
	}
	m := len(weight[0])
	rowAssign := make([]int, n)
	for i := range rowAssign {
		rowAssign[i] = -1
	}
	if m == 0 {
		return rowAssign
	}

	if n <= m {
		// rows = original rows; cost[i][j] = −weight (non-edges 0).
		cost := make([][]float64, n)
		for i := 0; i < n; i++ {
			cost[i] = make([]float64, m)
			for j := 0; j < m; j++ {
				if weight[i][j] > 0 {
					cost[i][j] = -weight[i][j]
				}
			}
		}
		rowToCol := hungarianAssign(cost)
		for i, j := range rowToCol {
			if j >= 0 && weight[i][j] > 0 {
				rowAssign[i] = j
			}
		}
		return rowAssign
	}

	// n > m: transpose so the (smaller) column side becomes the rows.
	cost := make([][]float64, m)
	for i := 0; i < m; i++ {
		cost[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if weight[j][i] > 0 {
				cost[i][j] = -weight[j][i]
			}
		}
	}
	colToRow := hungarianAssign(cost) // index = original col, value = original row (or -1)
	for origCol, origRow := range colToRow {
		if origRow >= 0 && weight[origRow][origCol] > 0 {
			rowAssign[origRow] = origCol
		}
	}
	return rowAssign
}

// hungarianAssign solves the rectangular assignment problem (minimum total cost)
// for a cost matrix with R rows ≤ C columns, matching every row to a distinct
// column. Returns rowToCol where rowToCol[i] is the column matched to row i, or -1.
func hungarianAssign(cost [][]float64) []int {
	R := len(cost)
	if R == 0 {
		return nil
	}
	C := len(cost[0])
	const inf = math.MaxFloat64

	u := make([]float64, R+1)
	v := make([]float64, C+1)
	p := make([]int, C+1)   // p[j] = row assigned to column j (1-indexed), 0 = none
	way := make([]int, C+1) // augmenting-path bookkeeping
	for i := 1; i <= R; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, C+1)
		used := make([]bool, C+1)
		for j := 0; j <= C; j++ {
			minv[j] = inf
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := inf
			j1 := -1
			for j := 1; j <= C; j++ {
				if used[j] {
					continue
				}
				cur := cost[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= C; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for j0 != 0 {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
		}
	}

	rowToCol := make([]int, R)
	for i := range rowToCol {
		rowToCol[i] = -1
	}
	for j := 1; j <= C; j++ {
		if p[j] > 0 {
			rowToCol[p[j]-1] = j - 1
		}
	}
	return rowToCol
}
