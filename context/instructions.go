package context

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// InstructionFile names recognised in discovery order (project config).
var InstructionFile = []string{"AGENTS.md", "CLAUDE.md", "AGENTS.local.md", "CLAUDE.local.md"}

// DiscoverInstructions walks from dir up to root collecting nearby instruction
// files (deepest first). It is a lightweight, dependency-free analogue of the
// DSH AGENTS.md hierarchy discovery. Returns file paths in hierarchy order.
func DiscoverInstructions(dir string) []string {
	var found []string
	current := dir
	for {
		for _, name := range InstructionFile {
			candidate := filepath.Join(current, name)
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				found = append(found, candidate)
			}
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}
	// Deepest first (project-specific overrides global).
	sort.SliceStable(found, func(i, j int) bool {
		return strings.Count(filepath.Dir(found[i]), string(filepath.Separator)) <
			strings.Count(filepath.Dir(found[j]), string(filepath.Separator))
	})
	return found
}

// LoadInstructions reads discovered files and concatenates their content in
// hierarchy order (project-local last).
func LoadInstructions(dir string) string {
	var builder strings.Builder
	for _, path := range DiscoverInstructions(dir) {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString("\n\n")
		}
		builder.WriteString("### " + filepath.Base(path) + "\n\n")
		builder.Write(data)
	}
	return builder.String()
}
