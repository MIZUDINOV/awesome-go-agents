package context

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestMeterEstimate(t *testing.T) {
	m := NewMeter()
	// 120 runes / 4 = 30 + 1 structural.
	if got := m.Estimate("abcdefghij"); got == 0 {
		t.Error("estimate = 0")
	}
}

func TestPruneResult(t *testing.T) {
	caps := PruneCaps{ThresholdChars: 10, HeadChars: 4, TailChars: 2}
	big := strings.Repeat("x", 100)
	short := caps.PruneResult(big)
	if len(short) >= len(big) {
		t.Errorf("prune did not shorten: original=%d pruned=%d", len(big), len(short))
	}
	if !strings.Contains(short, "... pruned ...") {
		t.Errorf("missing prune marker: %q", short)
	}
	small := caps.PruneResult("tiny")
	if small != "tiny" {
		t.Errorf("small should be unchanged")
	}
}

func TestPruneResultUnicodeSafe(t *testing.T) {
	caps := PruneCaps{ThresholdChars: 10, HeadChars: 4, TailChars: 2}
	// 4-byte runes: byte slicing would cut a rune in half.
	big := ""
	for i := 0; i < 40; i++ {
		big += "😀"
	}
	short := caps.PruneResult(big)
	if !utf8.ValidString(short) {
		t.Error("pruned result is not valid UTF-8")
	}
	if !strings.Contains(short, "... pruned ...") {
		t.Errorf("missing marker: %q", short)
	}
	// Mixed ASCII + multibyte.
	mixed := strings.Repeat("a", 20) + "привет мир" + strings.Repeat("b", 20)
	mixedShort := caps.PruneResult(mixed)
	if !utf8.ValidString(mixedShort) {
		t.Error("mixed result not valid UTF-8")
	}
}

func TestSpillUnicodeSafe(t *testing.T) {
	full := strings.Repeat("😀", 20)
	inline, ref := Spill(5, 3, full, "artifact/1.log")
	if ref == nil {
		t.Fatal("expected spill")
	}
	if !utf8.ValidString(inline) {
		t.Error("inline not valid UTF-8")
	}
	if ref.Preview != string([]rune(full)[17:]) {
		t.Errorf("preview = %q", ref.Preview)
	}
}

func TestSpill(t *testing.T) {
	inline, ref := Spill(10, 3, "abcdefghijklmnop", "artifact/1.log")
	if ref == nil {
		t.Fatal("expected spill ref")
	}
	if ref.Artifact != "artifact/1.log" {
		t.Errorf("artifact = %q", ref.Artifact)
	}
	if len(inline) >= 16 {
		t.Errorf("inline should be bounded: %q", inline)
	}
	inline2, ref2 := Spill(100, 3, "small", "a")
	if ref2 != nil || inline2 != "small" {
		t.Errorf("small should stay inline")
	}
}
