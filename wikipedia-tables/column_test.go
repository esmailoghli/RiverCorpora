package main

import "testing"

// uuids builds a fake previous-version UUID list, one per header.
func fakeUUIDs(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = string(rune('A' + i)) // "A","B","C",… stand in for UUIDs
	}
	return out
}

func TestMatchColumnsInsertMiddle(t *testing.T) {
	prevH := []string{"city", "country", "population"}
	prevU := fakeUUIDs(3)                                       // A,B,C
	curH := []string{"city", "country", "region", "population"} // "region" inserted at idx 2
	got := matchColumns(prevH, prevU, curH)
	// city→A, country→B, region→new (not A/B/C), population→C
	if got[0] != "A" || got[1] != "B" || got[3] != "C" {
		t.Fatalf("stable columns lost their UUIDs: %v", got)
	}
	for _, u := range []string{"A", "B", "C"} {
		if got[2] == u {
			t.Fatalf("inserted column reused an existing UUID %q: %v", u, got)
		}
	}
}

func TestMatchColumnsRemoveMiddle(t *testing.T) {
	prevH := []string{"city", "country", "population"}
	prevU := fakeUUIDs(3)
	curH := []string{"city", "population"} // "country" removed
	got := matchColumns(prevH, prevU, curH)
	if got[0] != "A" || got[1] != "C" {
		t.Fatalf("expected city→A, population→C, got %v", got)
	}
}

func TestMatchColumnsReorder(t *testing.T) {
	prevH := []string{"city", "country", "population"}
	prevU := fakeUUIDs(3)
	curH := []string{"population", "city", "country"} // fully reordered
	got := matchColumns(prevH, prevU, curH)
	if got[0] != "C" || got[1] != "A" || got[2] != "B" {
		t.Fatalf("reordered columns should carry their UUIDs: %v", got)
	}
}

// Header-less tables must keep the previous index-based behaviour: existing
// columns retain their UUIDs and an appended column gets a fresh one.
func TestMatchColumnsHeaderlessFallback(t *testing.T) {
	prevH := []string{"", "", ""}
	prevU := fakeUUIDs(3)
	curH := []string{"", "", "", ""} // grew by one at the end
	got := matchColumns(prevH, prevU, curH)
	if got[0] != "A" || got[1] != "B" || got[2] != "C" {
		t.Fatalf("header-less columns should stay index-stable: %v", got)
	}
	if got[3] == "A" || got[3] == "B" || got[3] == "C" || got[3] == "" {
		t.Fatalf("appended column should get a fresh UUID: %v", got)
	}
}

// Duplicate headers are disambiguated by position.
func TestMatchColumnsDuplicateHeaders(t *testing.T) {
	prevH := []string{"year", "year", "value"}
	prevU := fakeUUIDs(3) // A,B,C
	curH := []string{"year", "year", "value"}
	got := matchColumns(prevH, prevU, curH)
	if got[0] != "A" || got[1] != "B" || got[2] != "C" {
		t.Fatalf("duplicate headers should match by position: %v", got)
	}
}

func TestColumnHeadersFromRawRows(t *testing.T) {
	rows := [][]rawCell{
		{{content: "City", isHeader: true}, {content: "Pop", isHeader: true}},
		{{content: "Berlin"}, {content: "3500000"}},
	}
	got := columnHeaders(rows, 2)
	if got[0] != "city" || got[1] != "pop" {
		t.Fatalf("expected normalised headers [city pop], got %v", got)
	}
}

// A colspan header should fill each expanded column it covers.
func TestColumnHeadersColspan(t *testing.T) {
	rows := [][]rawCell{
		{{content: "Group", isHeader: true, colspan: 2}, {content: "Total", isHeader: true}},
	}
	got := columnHeaders(rows, 3)
	if got[0] != "group" || got[1] != "group" || got[2] != "total" {
		t.Fatalf("colspan header not expanded across columns: %v", got)
	}
}
