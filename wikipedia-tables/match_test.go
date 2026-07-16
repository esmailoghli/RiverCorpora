package main

import "testing"

// Spans must be hashed without truncation: a single byte each would alias
// colspan 257 onto colspan 1 and silently dedup a real structural change.
func TestFingerprintSpansNotTruncated(t *testing.T) {
	one := [][]rawCell{{{content: "x", colspan: 1, rowspan: 1}}}
	big := [][]rawCell{{{content: "x", colspan: 257, rowspan: 1}}}
	if fingerprintRaw(one) == fingerprintRaw(big) {
		t.Errorf("colspan 1 and 257 must not share a fingerprint")
	}
	tallOne := [][]rawCell{{{content: "x", colspan: 1, rowspan: 1}}}
	tallBig := [][]rawCell{{{content: "x", colspan: 1, rowspan: 300}}}
	if fingerprintRaw(tallOne) == fingerprintRaw(tallBig) {
		t.Errorf("rowspan 1 and 300 must not share a fingerprint")
	}
}

// A case where greedy-by-largest-weight is suboptimal: picking the single
// largest cell (0,0)=0.9 forces row1 onto a non-edge. The maximum-weight
// matching must instead take (0,1)+(1,0)=1.65.
func TestMaxWeightMatchingBeatsGreedy(t *testing.T) {
	w := [][]float64{
		{0.90, 0.80},
		{0.85, 0.00}, // row1 has no edge to col1
	}
	got := maxWeightMatching(w)
	if len(got) != 2 || got[0] != 1 || got[1] != 0 {
		t.Fatalf("expected [1 0] (max-weight), got %v", got)
	}
}

// Rectangular: more current tables than tracked. Each tracked row matches its
// best column; the extra column is left unmatched (becomes a new table).
func TestMaxWeightMatchingRectangular(t *testing.T) {
	w := [][]float64{
		{0.10, 0.95, 0.20}, // 1 tracked row, 3 current cols
	}
	got := maxWeightMatching(w)
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("expected row 0 -> col 1, got %v", got)
	}
}

// Degenerate shape that previously hung: many rows (tracked tables), one column
// (current table). With square padding this was O(N³); it must now be instant
// and pick the single best row for the lone column.
func TestMaxWeightMatchingManyRowsOneCol(t *testing.T) {
	const N = 6000
	w := make([][]float64, N)
	for i := 0; i < N; i++ {
		w[i] = []float64{0.0}
	}
	w[4321][0] = 0.95 // the only real edge
	got := maxWeightMatching(w)
	if len(got) != N {
		t.Fatalf("expected %d rows, got %d", N, len(got))
	}
	for i, c := range got {
		if i == 4321 {
			if c != 0 {
				t.Fatalf("row 4321 should match col 0, got %d", c)
			}
		} else if c != -1 {
			t.Fatalf("row %d should be unmatched, got %d", i, c)
		}
	}
}

// Symmetric degenerate shape: one row, many columns.
func TestMaxWeightMatchingOneRowManyCols(t *testing.T) {
	const N = 6000
	row := make([]float64, N)
	row[1234] = 0.8
	got := maxWeightMatching([][]float64{row})
	if len(got) != 1 || got[0] != 1234 {
		t.Fatalf("expected row 0 -> col 1234, got %v", got[:1])
	}
}

// No edges at all (all weights 0): nothing matches.
func TestMaxWeightMatchingNoEdges(t *testing.T) {
	w := [][]float64{
		{0, 0},
		{0, 0},
	}
	got := maxWeightMatching(w)
	for i, c := range got {
		if c != -1 {
			t.Errorf("row %d should be unmatched, got col %d", i, c)
		}
	}
}

// End-to-end matcher: two tracked tables whose positions have swapped in the
// new revision. Identical content means the position tiebreaker (↓POS) decides,
// keeping each table matched to the instance nearest its previous position.
func TestMatchTablesPositionTiebreak(t *testing.T) {
	cities := map[string]int{"berlin": 1, "germany": 2, "munich": 1, "city": 1, "country": 1, "population": 1}
	animals := map[string]int{"lion": 1, "snake": 1, "legs": 1, "sound": 1, "animal": 1, "roar": 1}

	tracked := []*tableState{
		{bowHist: []map[string]int{cities}, lastPos: 0, lifetime: 1},
		{bowHist: []map[string]int{animals}, lastPos: 1, lifetime: 1},
	}
	cur := []curObj{
		{bow: cities, pos: 0},
		{bow: animals, pos: 1},
	}
	res := matchTables(tracked, cur)
	if res[0].ti != 0 || res[1].ti != 1 {
		t.Fatalf("expected cur0->t0, cur1->t1, got %+v", res)
	}
}

// Position must strictly dominate lifetime in the tiebreak: with two tracked
// tables of identical content and one returning instance, the position-closer
// table wins even if the other has a vastly larger lifetime.
func TestMatchTablesPositionDominatesLifetime(t *testing.T) {
	content := map[string]int{"berlin": 1, "germany": 2, "munich": 1, "city": 1, "country": 1, "population": 1}

	// tracked[0]: position 0, lifetime 1 (just one position-unit closer)
	// tracked[1]: position 1, lifetime 1,000,000 (max lifetime term)
	// Under the old constants the lifetime term (1e-6) exactly canceled the
	// one-unit position term (1e-6), producing an arbitrary tie.
	tracked := []*tableState{
		{bowHist: []map[string]int{content}, lastPos: 0, lifetime: 1},
		{bowHist: []map[string]int{content}, lastPos: 1, lifetime: 1000000},
	}
	cur := []curObj{{bow: content, pos: 0}} // one instance at position 0

	res := matchTables(tracked, cur)
	if res[0].ti != 0 {
		t.Fatalf("position-closer table (ti=0) should win the tiebreak despite the "+
			"other's huge lifetime, got ti=%d", res[0].ti)
	}
}
