//go:build integration

// Embedded-Postgres driver for the pgstore integration scenarios. Boots a
// REAL native PostgreSQL binary (downloaded by
// github.com/fergusstrange/embedded-postgres on first run) so the
// durability/restart gate can be executed on a box with no Docker daemon and
// no external Postgres. Run:
//
//	go test -tags integration ./session/pgstore/... -run Embedded -count=1
package pgstore_test

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	embedded "github.com/fergusstrange/embedded-postgres"

	"github.com/jackc/pgx/v5/pgxpool"
)

const embeddedPort = 55439

func TestEmbeddedPgScenarioH_RestartIdenticalSurface(t *testing.T) {
	ctx := context.Background()
	pool, cleanup := bootEmbedded(t, ctx)
	defer cleanup()
	runScenarioH(t, ctx, pool)
}

func TestEmbeddedPgScenarioG_RecoveryMarksDangling(t *testing.T) {
	ctx := context.Background()
	pool, cleanup := bootEmbedded(t, ctx)
	defer cleanup()
	runScenarioG(t, ctx, pool)
}

// bootEmbedded starts a disposable native PostgreSQL and returns a pool scoped
// to an isolated, migrated schema (same setup as the external DB driver).
func bootEmbedded(t *testing.T, ctx context.Context) (*pgxpool.Pool, func()) {
	t.Helper()
	cacheDir := filepath.Join(os.TempDir(), "embedded-pg-cache-wz")
	_ = os.MkdirAll(cacheDir, 0o755)
	pg := embedded.NewDatabase(embedded.DefaultConfig().
		Version(embedded.V17).
		Port(embeddedPort).
		Username("wz").Password("wz").
		Database("postgres").
		DataPath(t.TempDir()).
		RuntimePath(t.TempDir()).
		CachePath(cacheDir).
		StartTimeout(120 * time.Second).
		Logger(io.Discard))

	if err := pg.Start(); err != nil {
		t.Fatalf("embedded postgres start (is a native PG binary downloadable here?): %v", err)
	}

	url := fmt.Sprintf("postgres://wz:wz@localhost:%d/postgres?sslmode=disable", embeddedPort)
	// Probe readiness (Start already waits, but be lenient if the binary
	// download/start raced).
	deadline := time.Now().Add(90 * time.Second)
	for {
		pool, err := pgxpool.New(ctx, url)
		if err == nil && pool.Ping(ctx) == nil {
			pool.Close()
			break
		}
		if time.Now().After(deadline) {
			_ = pg.Stop()
			t.Fatalf("embedded postgres did not become ready")
		}
		time.Sleep(500 * time.Millisecond)
	}

	pool, cleanup := withIsolatedPool(t, ctx, url)
	t.Cleanup(func() { _ = pg.Stop() })
	return pool, cleanup
}
