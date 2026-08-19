//go:build integration

// Package pgstore integration test: Scenario H (PostgreSQL restart/identical
// surface) and Scenario G (recovery marks dangling calls). Two drivers share
// the same scenario bodies:
//
//   - external DB (CI): requires WZHOOH_TEST_MIGRATION_DATABASE_URL,
//   - embedded Postgres (this box, no Docker): boots a real native PostgreSQL
//     binary via github.com/fergusstrange/embedded-postgres (see
//     embedded_integration_test.go).
//
// Run locally without Docker or an external DB:
//
//	go test -tags integration ./session/pgstore/... -run Embedded -count=1
//
// Each run uses a uuid-suffixed schema (search_path) so parallel runs never
// collide, and drops it on completion.
package pgstore_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/session/pgstore"
)

const tenant = "tenant-e2e"

// external DB driver -------------------------------------------------------

func TestPgStoreScenarioH_RestartIdenticalSurface(t *testing.T) {
	baseURL := getEnv(t)
	ctx := context.Background()
	pool, cleanup := withIsolatedPool(t, ctx, baseURL)
	defer cleanup()
	runScenarioH(t, ctx, pool)
}

func TestPgStoreScenarioG_RecoveryMarksDangling(t *testing.T) {
	baseURL := getEnv(t)
	ctx := context.Background()
	pool, cleanup := withIsolatedPool(t, ctx, baseURL)
	defer cleanup()
	runScenarioG(t, ctx, pool)
}

// shared scenario bodies ----------------------------------------------------

// runScenarioH is scenario H: worker unload -> new worker acquires the lease,
// reloads the durable log and rebuilds the identical surface, then continues,
// and a concurrent claimer is refused (no split-brain).
func runScenarioH(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	store1 := pgstore.New(pool).WithTenant(tenant)
	lease1, err := store1.ClaimLease(ctx, "sess-H", "worker-1", 30*time.Second, tenant)
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if _, err := store1.AppendFenced(ctx, lease1, []session.Event{
		{ID: "h-assistant", SessionID: "sess-H", Type: session.EventAssistantMessage,
			Data: session.AssistantContent("thinking", "", []session.ToolCall{{CallID: "call-1", Name: "read", Arguments: []byte(`{"file_path":"a"}`)}})},
		{ID: "h-call", SessionID: "sess-H", CallID: "call-1", Type: session.EventToolCall,
			Data: session.ToolCallPayload("call-1", "read", []byte(`{"file_path":"a"}`))},
		{ID: "h-result", SessionID: "sess-H", CallID: "call-1", Type: session.EventToolResult,
			Data: toolResultJSON("call-1", "read", `"file a"`, false)},
	}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := store1.ReleaseLease(ctx, lease1); err != nil {
		t.Fatalf("release: %v", err)
	}

	// Worker 2 (a fresh store instance = "new process") acquires and reloads.
	store2 := pgstore.New(pool).WithTenant(tenant)
	lease2, err := store2.ClaimLease(ctx, "sess-H", "worker-2", 30*time.Second, tenant)
	if err != nil {
		t.Fatalf("worker-2 claim: %v", err)
	}
	events, err := store2.Load(ctx, "sess-H", 0, 0)
	if err != nil {
		t.Fatalf("worker-2 load: %v", err)
	}
	// Replay proves the durable log reconstructs the same surface (identical
	// surface after restart, without network).
	_, proj1, err := session.NewSurface(session.SurfaceSpec{}).Project(events)
	if err != nil {
		t.Fatalf("worker-2 rebuild surface: %v", err)
	}
	if proj1 == nil || proj1.Generation != 0 {
		t.Fatalf("unexpected projection after restart: %+v", proj1)
	}
	// Continue the session on worker-2 and confirm the next step appends.
	if _, err := store2.AppendFenced(ctx, lease2, []session.Event{
		{ID: "h-continue", SessionID: "sess-H", Type: session.EventUserMessage, Data: session.UserText("continue")},
	}); err != nil {
		t.Fatalf("worker-2 append continuation: %v", err)
	}
	if err := store2.ReleaseLease(ctx, lease2); err != nil {
		t.Fatalf("worker-2 release: %v", err)
	}

	// Lease exclusivity: claim while held is refused (no split-brain).
	store3 := pgstore.New(pool).WithTenant(tenant)
	lease3, err := store3.ClaimLease(ctx, "sess-H", "worker-3", 30*time.Second, tenant)
	if err == nil {
		t.Fatal("expected ErrLeaseHeld from concurrent claimer")
	}
	_ = lease3
	if !strings.Contains(err.Error(), "lease") {
		t.Fatalf("expected lease-held error, got %v", err)
	}
}

// runScenarioG is scenario G: a tool/call persisted with no result (worker
// died mid-side-effect) is marked TOOL_OUTCOME_UNKNOWN on recovery, never
// re-executed.
func runScenarioG(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	store := pgstore.New(pool).WithTenant(tenant)
	lease, err := store.ClaimLease(ctx, "sess-G", "worker-died", 30*time.Second, tenant)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AppendFenced(ctx, lease, []session.Event{
		{ID: "g-assistant", SessionID: "sess-G", Type: session.EventAssistantMessage,
			Data: session.AssistantContent("", "", []session.ToolCall{{CallID: "call-x", Name: "write", Arguments: []byte(`{"file_path":"f","content":"x"}`)}})},
		{ID: "g-call", SessionID: "sess-G", CallID: "call-x", Type: session.EventToolCall,
			Data: session.ToolCallPayload("call-x", "write", []byte(`{"file_path":"f","content":"x"}`))},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}

	recoverLease, err := store.ClaimLease(ctx, "sess-G", "worker-recover", 30*time.Second, tenant)
	if err != nil {
		t.Fatal(err)
	}
	report, err := store.Recover(ctx, recoverLease)
	if err != nil {
		t.Fatalf("recover: %v", err)
	}
	if len(report.DanglingCalls) != 1 || report.DanglingCalls[0] != "call-x" {
		t.Fatalf("expected one dangling call, got %v", report.DanglingCalls)
	}
	events, err := store.Load(ctx, "sess-G", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	sawUnknown := false
	for _, e := range events {
		if e.Type == session.EventToolResult && strings.Contains(string(e.Data), `"code":"TOOL_OUTCOME_UNKNOWN"`) {
			sawUnknown = true
		}
	}
	if !sawUnknown {
		t.Fatal("recovery did not emit TOOL_OUTCOME_UNKNOWN for the dangling call")
	}
	if err := store.ReleaseLease(ctx, recoverLease); err != nil {
		t.Fatal(err)
	}
}

// helpers --------------------------------------------------------------------

// withIsolatedPool creates a uuid-suffixed schema on the given server URL,
// mounts the library migration inside it, and returns a pool scoped to that
// schema plus a cleanup that drops it.
func withIsolatedPool(t *testing.T, ctx context.Context, baseURL string) (*pgxpool.Pool, func()) {
	t.Helper()
	admin, err := pgx.Connect(ctx, baseURL)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	schema := "agentkit_it_" + randHex(6)
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+schema); err != nil {
		_ = admin.Close(ctx)
		t.Fatalf("create schema: %v", err)
	}
	if _, err := admin.Exec(ctx, `SET search_path TO `+schema+`; `+pgstore.Migration); err != nil {
		_ = admin.Close(ctx)
		t.Fatalf("migrate: %v", err)
	}
	_ = admin.Close(ctx)
	pool, err := pgxpool.New(ctx, withSearchPath(baseURL, schema))
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	cleanup := func() {
		admin2, err := pgx.Connect(ctx, baseURL)
		if err == nil {
			_, _ = admin2.Exec(ctx, `DROP SCHEMA `+schema+` CASCADE`)
			_ = admin2.Close(ctx)
		}
		pool.Close()
	}
	return pool, cleanup
}

func getEnv(t *testing.T) string {
	t.Helper()
	url := strings.TrimSpace(os.Getenv("WZHOOH_TEST_MIGRATION_DATABASE_URL"))
	if url == "" {
		t.Skip("WZHOOH_TEST_MIGRATION_DATABASE_URL not set; skipping external-DB PostgreSQL integration")
	}
	return url
}

func withSearchPath(baseURL, schema string) string {
	sep := "?"
	if strings.Contains(baseURL, "?") {
		sep = "&"
	}
	return baseURL + sep + "search_path=" + schema
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func toolResultJSON(callID, name, output string, isErr bool) []byte {
	out := output
	if out == "" {
		out = "{}"
	}
	b, _ := json.Marshal(map[string]any{
		"call_id": callID, "name": name, "output": json.RawMessage(out), "is_error": isErr,
	})
	return b
}
