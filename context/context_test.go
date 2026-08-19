package context

import (
	"strings"
	"testing"
)

func TestMeterEstimate(t *testing.T) {
	m := NewMeter()
	// 120 runes / 4 = 30 + 1 structural.
	if got := m.Estimate("abcdefghij"); got == 0 {
		t.Error("estimate = 0")
	}
}

func TestDefaultCompactionPolicy(t *testing.T) {
	p := DefaultCompactionPolicy()
	window := int64(60000)
	if p.ThresholdTokens(window) != 48000 {
		t.Errorf("threshold = %d", p.ThresholdTokens(window))
	}
	if p.TargetTokens(window) != 30000 {
		t.Errorf("target = %d", p.TargetTokens(window))
	}
	// Pressure counts overhead against the window.
	if !p.Pressure(window, 4000, 45000) { // 49000 >= 48000
		t.Error("expected pressure")
	}
	if p.Pressure(window, 4000, 1000) {
		t.Error("expected no pressure")
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

func TestCheckpointText(t *testing.T) {
	c := &Checkpoint{
		PrimaryRequest: "Build a landing page",
		KeyTechnicalConcepts: []string{"Tailwind", "Angular"},
		NextStep: "Write index.html",
	}
	text := c.Text()
	for _, want := range []string{"Primary Request and Intent", "Key Technical Concepts", "Tailwind", "Next Step"} {
		if !strings.Contains(text, want) {
			t.Errorf("checkpoint missing %q: %s", want, text)
		}
	}
}

func TestOverflowRecovery(t *testing.T) {
	r := DefaultOverflowRecovery()
	if plan := r.Evaluate(5000, 60000); !plan.Possible {
		t.Error("under window should be possible with no changes")
	}
	if plan := r.Evaluate(200000, 60000); !plan.Possible || !plan.PruneToolResults {
		t.Errorf("overflow plan should prune: %+v", plan)
	}
}

func TestAssembler(t *testing.T) {
	a := NewAssembler().WithDefaultPersona("You are an AI agent.")
	a.WithSections(func() []Section {
		return []Section{RenderSection("tools", "Use read before edit.")}
	})
	assembly, err := a.Assemble(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(assembly.System, "You are an AI agent.") {
		t.Errorf("system missing persona: %q", assembly.System)
	}
	if !strings.Contains(assembly.System, "Use read before edit.") {
		t.Errorf("system missing section: %q", assembly.System)
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
