package context

// OverflowRecovery is the emergency path when the provider actually returned a
// context-exceeded error. It differs from proactive compaction: it prunes tool
// results and drops the oldest attributable blocks to make the request fit,
// then allows one retry.
type OverflowRecovery struct {
	// MaxPruneDepth is how many messages to inspect when removing oversized
	// tool results and old turns.
	MaxPruneDepth int
	// KeepRecentBlocks is how many recent conversation blocks are always kept.
	KeepRecentBlocks int
}

// DefaultOverflowRecovery returns a sane default.
func DefaultOverflowRecovery() OverflowRecovery {
	return OverflowRecovery{MaxPruneDepth: 64, KeepRecentBlocks: 4}
}

// Plan is the outcome of OverflowRecovery.Evaluate: what to prune/drop so a
// request fits, plus whether any change was possible.
type Plan struct {
	// PruneToolResults is true if the recoverer should rewrite oversized tool
	// results (caller applies PruneCaps).
	PruneToolResults bool
	// DropOldestBlocks > 0 instructs dropping that many oldest turns.
	DropOldestBlocks int
	// Possible is false when even aggressive pruning cannot shrink enough
	// (caller should fail rather than loop).
	Possible bool
}

// Evaluate computes an overflow recovery plan given an estimated current size
// and the model's hard context window.
func (r OverflowRecovery) Evaluate(estimatedTokens, contextWindow int64) Plan {
	if estimatedTokens <= contextWindow {
		return Plan{Possible: true}
	}
	// First pass: pruning oversized tool results may recover a meaningful
	// chunk; we always allow it and assume it helps.
	plan := Plan{PruneToolResults: true, Possible: true}
	// If the estimated size is grossly over the window, also drop oldest
	// blocks. One block is allowed as a starting point; the caller retries the
	// build and re-evaluates.
	if estimatedTokens > contextWindow*2 {
		plan.DropOldestBlocks = r.keepBlocks()
	}
	return plan
}

func (r OverflowRecovery) keepBlocks() int {
	if r.KeepRecentBlocks <= 0 {
		return 1
	}
	return r.KeepRecentBlocks - 1
}
