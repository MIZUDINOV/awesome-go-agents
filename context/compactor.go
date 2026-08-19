package context

// CompactionPolicy controls when and how aggressively the context is
// summarised. Capacity belongs to (provider, model), not the agent, so the
// trigger and target are expressed as fractions of the model's context window.
type CompactionPolicy struct {
	// ThresholdRatio triggers compaction at this fraction of context window.
	ThresholdRatio float64
	// TargetRatio is the fraction of context window to keep as the recent tail
	// after compaction (the summarizable region above it is replaced).
	TargetRatio float64
	// TailTurns / TailTokens bound the verbatim recent tail kept intact.
	TailTurns  int
	TailTokens int64
}

// DefaultCompactionPolicy mirrors DSH defaults on a per-model basis.
func DefaultCompactionPolicy() CompactionPolicy {
	return CompactionPolicy{
		ThresholdRatio: DefaultThresholdRatio,
		TargetRatio:    DefaultTargetRatio,
		TailTokens:     16000,
	}
}

// ThresholdTokens returns the trigger threshold for a given context window.
func (p CompactionPolicy) ThresholdTokens(window int64) int64 {
	return int64(float64(window) * p.ThresholdRatio)
}

// TargetTokens returns how many tokens to keep verbatim after compaction.
func (p CompactionPolicy) TargetTokens(window int64) int64 {
	return int64(float64(window) * p.TargetRatio)
}

// Pressure reports whether a current context size warrants compaction.
// system/tools/other are non-history overhead that must count against the
// window, not just historyTokens.
func (p CompactionPolicy) Pressure(window, historyTokens, overheadTokens int64) bool {
	total := historyTokens + overheadTokens
	return total >= p.ThresholdTokens(window)
}

// Checkpoint is the structured compaction summary a summarizer produces. It is
// the durable "what the next step needs to continue" rather than a casual
// "summarize the conversation".
type Checkpoint struct {
	// PrimaryRequest is the user's original request and intent.
	PrimaryRequest string `json:"primary_request"`
	// KeyTechnicalConcepts are concepts established that matter going forward.
	KeyTechnicalConcepts []string `json:"key_technical_concepts,omitempty"`
	// FilesAndCode are relevant file paths / code regions.
	FilesAndCode []string `json:"files_and_code,omitempty"`
	// ErrorsAndFixes tracks failures and their resolutions.
	ErrorsAndFixes []string `json:"errors_and_fixes,omitempty"`
	// PendingJobs are incomplete operations.
	PendingJobs []string `json:"pending_jobs,omitempty"`
	// CurrentWork is what is being worked on right now.
	CurrentWork string `json:"current_work,omitempty"`
	// NextStep is what the next agent step should do.
	NextStep string `json:"next_step,omitempty"`
	// CriticalContext captures anything else essential to continuation.
	CriticalContext []string `json:"critical_context,omitempty"`
}

// Text renders the checkpoint as a compact, model-friendly block.
func (c *Checkpoint) Text() string {
	var b []byte
	b = appendStringSection(b, "Primary Request and Intent", c.PrimaryRequest)
	b = appendListSection(b, "Key Technical Concepts", c.KeyTechnicalConcepts)
	b = appendListSection(b, "Files and Code", c.FilesAndCode)
	b = appendListSection(b, "Errors and Fixes", c.ErrorsAndFixes)
	b = appendListSection(b, "Pending Jobs", c.PendingJobs)
	b = appendStringSection(b, "Current Work", c.CurrentWork)
	b = appendStringSection(b, "Next Step", c.NextStep)
	b = appendListSection(b, "Critical Context", c.CriticalContext)
	return string(b)
}

func appendStringSection(b []byte, heading, value string) []byte {
	if value == "" {
		return b
	}
	if len(b) > 0 {
		b = append(b, '\n')
	}
	b = append(b, "## "...)
	b = append(b, heading...)
	b = append(b, "\n\n"...)
	b = append(b, value...)
	b = append(b, '\n')
	return b
}

func appendListSection(b []byte, heading string, items []string) []byte {
	if len(items) == 0 {
		return b
	}
	if len(b) > 0 {
		b = append(b, '\n')
	}
	b = append(b, "## "...)
	b = append(b, heading...)
	b = append(b, "\n"...)
	for _, item := range items {
		if item == "" {
			continue
		}
		b = append(b, "- "...)
		b = append(b, item...)
		b = append(b, '\n')
	}
	return b
}
