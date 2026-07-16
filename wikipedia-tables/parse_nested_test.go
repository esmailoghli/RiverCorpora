package main

import (
	"strings"
	"testing"
)

// rowsText joins each parsed row's cell contents with " | " for easy assertions.
func rowsText(rows [][]rawCell) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		cells := make([]string, len(r))
		for j, c := range r {
			cells[j] = strings.ReplaceAll(c.content, "\n", "\\n")
		}
		out[i] = strings.Join(cells, " | ")
	}
	return out
}

// Nested table that begins on a NEW LINE after a cell has started. This is the
// standard MediaWiki nesting form; its content must be folded into the parent
// cell, not dropped.
func TestNestedTableNewLine(t *testing.T) {
	table := `{| class="wikitable"
| outer cell
{| class="wikitable"
| inner cell
|}
| outer cell 2
|}`
	rows := parseWikiRawRows(table)
	got := rowsText(rows)
	t.Logf("new-line nested -> %#v", got)
	joined := strings.Join(got, " || ")
	if !strings.Contains(joined, "inner cell") {
		t.Errorf("nested-table content was dropped: %#v", got)
	}
	if !strings.Contains(joined, "outer cell 2") {
		t.Errorf("cell after nested table was lost: %#v", got)
	}
}

// Two levels of new-line nesting must all be folded into the outermost cell.
func TestNestedTableTwoLevels(t *testing.T) {
	table := `{| class="wikitable"
| outer cell
{| class="wikitable"
| middle cell
{| class="wikitable"
| deep cell
|}
|}
| outer cell 2
|}`
	rows := parseWikiRawRows(table)
	got := rowsText(rows)
	t.Logf("two-level nested -> %#v", got)
	joined := strings.Join(got, " || ")
	for _, want := range []string{"deep cell", "middle cell", "outer cell 2"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %#v", want, got)
		}
	}
}

// A {| that appears with no cell open (immediately after a row marker) has
// nothing to attach to; it must not panic and is skipped as block-level.
func TestNestedTableNoOpenCell(t *testing.T) {
	table := `{| class="wikitable"
|-
{| class="wikitable"
| stray
|}
|-
| real cell
|}`
	rows := parseWikiRawRows(table) // must not panic
	got := rowsText(rows)
	t.Logf("no-open-cell -> %#v", got)
	if !strings.Contains(strings.Join(got, " || "), "real cell") {
		t.Errorf("real cell after block-level nested table was lost: %#v", got)
	}
}

// Nested table opened INLINE on the same line as the cell pipe.
func TestNestedTableInline(t *testing.T) {
	table := `{| class="wikitable"
| {| class="wikitable"
| inner cell
|}
| outer cell 2
|}`
	rows := parseWikiRawRows(table)
	got := rowsText(rows)
	t.Logf("inline nested -> %#v", got)
	joined := strings.Join(got, " || ")
	if !strings.Contains(joined, "inner cell") {
		t.Errorf("nested-table content was dropped: %#v", got)
	}
	if !strings.Contains(joined, "outer cell 2") {
		t.Errorf("cell after nested table was lost: %#v", got)
	}
}
