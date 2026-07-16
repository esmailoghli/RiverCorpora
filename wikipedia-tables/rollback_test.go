package main

import (
	"testing"
	"time"
)

func md5fp(b byte) [16]byte { var f [16]byte; f[0] = b; return f }

func day(d int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, d)
}

// keptFPs returns the fingerprint tags (first byte) of the surviving revisions.
func keptFPs(revs []revMeta) []byte {
	keep := computeKeep(revs)
	var out []byte
	for i, k := range keep {
		if k {
			out = append(out, revs[i].fp[0])
		}
	}
	return out
}

func eq(a []byte, b ...byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// A page stable for >7 days, then vandalized and reverted within minutes: the
// excursion is short, so it must be collapsed even though the return is long
// after the original A.
func TestRollbackLongStableThenQuickRevert(t *testing.T) {
	revs := []revMeta{
		{fp: md5fp('A'), ts: day(0), userKey: "u:alice"},                       // stable A
		{fp: md5fp('B'), ts: day(100), userKey: "u:vandal"},                    // vandalism, day 100
		{fp: md5fp('A'), ts: day(100).Add(10 * time.Minute), userKey: "u:bob"}, // reverted 10 min later
	}
	got := keptFPs(revs)
	if !eq(got, 'A') {
		t.Errorf("quick revert after long stability should collapse to A, got %q", got)
	}
}

// A genuine reappearance after a long excursion (>7 days away) must be kept.
func TestRollbackLongExcursionKept(t *testing.T) {
	revs := []revMeta{
		{fp: md5fp('A'), ts: day(0), userKey: "u:alice"},
		{fp: md5fp('B'), ts: day(1), userKey: "u:vandal"},
		{fp: md5fp('A'), ts: day(30), userKey: "u:bob"}, // returns 29 days after B
	}
	got := keptFPs(revs)
	if !eq(got, 'A', 'B', 'A') {
		t.Errorf("long excursion should be kept, got %q", got)
	}
}

// The grouping pass can delete the excursion (here bob reverts himself inside
// his own 1h editing streak), leaving two identical states adjacent among the
// survivors. Bob's session has no net effect, so no second version may survive.
func TestGroupingExcursionRemovedLeavesNoDuplicate(t *testing.T) {
	base := day(0)
	revs := []revMeta{
		{fp: md5fp('A'), ts: base, userKey: "u:alice"},
		{fp: md5fp('B'), ts: base.Add(10 * time.Minute), userKey: "u:bob"},
		{fp: md5fp('A'), ts: base.Add(20 * time.Minute), userKey: "u:bob"},
	}
	if got := keptFPs(revs); !eq(got, 'A') {
		t.Errorf("a self-reverted same-user streak must leave a single A, got %q", got)
	}
}

// Classic quick rollback A->B->A within the window collapses to A.
func TestRollbackQuick(t *testing.T) {
	revs := []revMeta{
		{fp: md5fp('A'), ts: day(0), userKey: "u:alice"},
		{fp: md5fp('B'), ts: day(1), userKey: "u:vandal"},
		{fp: md5fp('A'), ts: day(2), userKey: "u:bob"},
	}
	if got := keptFPs(revs); !eq(got, 'A') {
		t.Errorf("A->B->A within 7d should collapse to A, got %q", got)
	}
}
