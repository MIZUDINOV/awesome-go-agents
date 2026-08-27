package skill

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFilesystemProviderDirectDiscovery(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	bundle := filepath.Join(root, "bundle")
	if err := os.Mkdir(bundle, 0o755); err != nil {
		t.Fatal(err)
	}
	writeSkillFile(t, filepath.Join(bundle, "SKILL.md"), "bundle")
	writeSkillFile(t, filepath.Join(root, "flat.md"), "flat")
	nested := filepath.Join(root, "nested", "too-deep")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	writeSkillFile(t, filepath.Join(nested, "SKILL.md"), "too-deep")

	provider, err := NewFilesystemProvider(FilesystemOptions{
		Name:  "filesystem",
		Roots: []FilesystemRoot{{Path: root, Rank: 100, Source: "project-agents"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	observation, err := provider.List(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(observation.Candidates) != 2 {
		t.Fatalf("candidates = %+v", observation.Candidates)
	}
	if observation.Candidates[0].Name != "bundle" || observation.Candidates[1].Name != "flat" {
		t.Fatalf("unexpected discovery = %+v", observation.Candidates)
	}
}

func TestFilesystemProviderFailsClosedForPinnedRun(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeSkillFile(t, filepath.Join(root, "mutable.md"), "mutable")
	provider, err := NewFilesystemProvider(FilesystemOptions{Name: "filesystem", Roots: []FilesystemRoot{{Path: root, Source: "project-agents"}}})
	if err != nil {
		t.Fatal(err)
	}
	registry := NewRegistry(8)
	mustRegisterProvider(t, registry, ProviderOptions{Name: "filesystem", Provider: provider})
	snapshot, err := registry.Snapshot(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Skills) != 1 {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	if _, err := registry.LoadPinned(t.Context(), ListRequest{}, snapshot.Skills[0]); !errors.Is(err, ErrUnsupportedMutable) {
		t.Fatalf("LoadPinned() error = %v, want ErrUnsupportedMutable", err)
	}
}

func TestFilesystemProviderWatcherInvalidatesAndStops(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeSkillFile(t, filepath.Join(root, "initial.md"), "initial")
	provider, err := NewFilesystemProvider(FilesystemOptions{
		Name: "filesystem", Roots: []FilesystemRoot{{Path: root, Source: "project-agents"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	changed := make(chan string, 1)
	done := make(chan error, 1)
	go func() {
		done <- provider.Watch(ctx, 5*time.Millisecond, func(reason string) {
			select {
			case changed <- reason:
			default:
			}
		})
	}()

	time.Sleep(15 * time.Millisecond)
	writeSkillFile(t, filepath.Join(root, "added.md"), "added")
	select {
	case reason := <-changed:
		if reason != "filesystem catalog changed" {
			t.Fatalf("reason = %q", reason)
		}
	case <-time.After(time.Second):
		t.Fatal("watcher did not invalidate the catalog")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Watch() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("watcher did not stop after cancellation")
	}
}

func writeSkillFile(t *testing.T, path, name string) {
	t.Helper()
	content := "---\nname: " + name + "\ndescription: " + name + " description\n---\nInstructions"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
