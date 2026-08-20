package pgstore

import (
	"context"
	"errors"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/session"
)

func TestNativeStoreFailsClosedWithoutTenant(t *testing.T) {
	store := New(nil)
	ctx := context.Background()
	lease := session.Lease{SessionID: "session-1", Token: "token", Fence: 1}
	tests := []struct {
		name string
		run  func() error
	}{
		{name: "append", run: func() error {
			_, err := store.Append(ctx, "session-1", []session.Event{{Type: session.EventUserMessage, Data: session.UserText("hello")}})
			return err
		}},
		{name: "append committed", run: func() error { _, err := store.AppendCommitted(ctx, "session-1", nil); return err }},
		{name: "append fenced", run: func() error { _, err := store.AppendFenced(ctx, lease, nil); return err }},
		{name: "append fenced committed", run: func() error { _, err := store.AppendFencedCommitted(ctx, lease, nil); return err }},
		{name: "load", run: func() error { _, err := store.Load(ctx, "session-1", 0, 0); return err }},
		{name: "tail", run: func() error { _, err := store.Tail(ctx, "session-1", 10); return err }},
		{name: "sequence", run: func() error { _, err := store.Sequence(ctx, "session-1"); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); !errors.Is(err, ErrNoTenant) {
				t.Fatalf("error = %v, want ErrNoTenant", err)
			}
		})
	}
}

func TestWithTenantTrimsWhitespace(t *testing.T) {
	store := New(nil).WithTenant("  tenant-a  ")
	if store.tenant != "tenant-a" {
		t.Fatalf("tenant = %q", store.tenant)
	}
	if err := store.requireNativeAndTenant(); err != nil {
		t.Fatalf("configured tenant rejected: %v", err)
	}
}
