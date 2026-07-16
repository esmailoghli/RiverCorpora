package main

import "testing"

func TestBagOfWordsTruncates(t *testing.T) {
	rows := [][]rawCell{{{content: "one two three four five six seven eight nine ten eleven twelve"}}}
	bow := bagOfWords(rows)
	if _, ok := bow["eleven"]; ok {
		t.Errorf("word past the 10-word cell limit should be dropped: %v", bow)
	}
	if bow["ten"] != 1 {
		t.Errorf("expected 'ten' within the 10-word limit, got %v", bow)
	}
}

func TestIsEditorBoilerplate(t *testing.T) {
	boiler := [][]rawCell{
		{{content: "Header text"}, {content: "Header text"}, {content: "Header text"}},
		{{content: "Example"}, {content: "Example"}, {content: "Example"}},
	}
	if !isEditorBoilerplate(boiler) {
		t.Errorf("pristine placeholder table should be detected")
	}
	// Headers filled in with real data -> not boilerplate.
	real := [][]rawCell{
		{{content: "City"}, {content: "Country"}},
		{{content: "Berlin"}, {content: "Germany"}},
	}
	if isEditorBoilerplate(real) {
		t.Errorf("real table must not be treated as boilerplate")
	}
}
