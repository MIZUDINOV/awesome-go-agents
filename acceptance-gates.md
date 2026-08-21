# Agent Harness — Acceptance Gates & E2E trace map

This is the executable acceptance matrix for the normative
[`CONTRACT.md`](CONTRACT.md). It maps the required DeepSeek-style harness
semantics onto concrete tests and mechanisms, so a reviewer can verify each
item quickly. The historical review and exploration documents are evidence,
not additional normative sources.

Scope: the durable provider-neutral kit under `agentkit/`. Wzhooh production
cutover is blocked until its host adapter writes the existing canonical
`chat_runs`/`chat_run_items`/`chat_tool_calls` ledger under the worker-owned
execution guard; no `agentkit_*` ledger is acceptable for production runs.

## How to run

```bash
# agentkit library (all non-DB tests)
cd agentkit
go test ./...
go vet ./...

# PostgreSQL durability integration (scenario H / G)
cd agentkit
WZHOOH_TEST_MIGRATION_DATABASE_URL=postgres://... \
  go test -tags integration ./session/pgstore/... -count=1   # external DB (CI)

# Or embedded native Postgres (no Docker / no external DB): boots a REAL
# PostgreSQL binary on first run (cached under $TEMP/embedded-pg-cache-wz):
cd agentkit
go test -tags integration ./session/pgstore/... -run Embedded -count=1

# legacy compatibility adapter (never a production executor)
cd wzhooh-back   # uses `replace` -> ../agentkit during development
go test ./internal/agents/... ./internal/ai/...
```

`go test -race` is a CI-only gate on this Windows box (no cgo/gcc locally).

## Blocking gate (§23.1) — P0 areas and where they are enforced

| P0 area | Mechanism | Enforcing test |
|---|---|---|
| Single ownership Agent Loop | Standalone `agent.Loop` claims and renews a session lease before writing; every append is fenced; host-owned runs use `Config.ClaimedLease` and never claim/renew/release a competing lease | `agent/agent_test.go`, `agent/e2e_test.go` |
| Append-only session log | `session.Store`/`FencedStore` append-only; compaction shadows but never deletes (H-COMPACT-001) | `session/fenced_test.go`, `session/surface_test.go` |
| Surface reconstruction | `session.Surface.Project` folds compaction and validates tool/result pairing (H-SURFACE-008) | `session/surface_test.go`, `agent/e2e_test.go` (scenario C) |
| PostgreSQL single-writer/durability | `pgstore` per-session counter + lease token + monotonic `execution_fence` + idempotent `event_id` (`ON CONFLICT`); per-session reserve in one tx | `session/pgstore/integration_test.go` (scenario H) |
| tool/call before side-effect | Loop persists `EventToolCall` BEFORE executing the executor; `Recover` marks any call without a result `TOOL_OUTCOME_UNKNOWN` (no blind retry) | `agent/e2e_test.go` (scenario G) |
| read-before-edit + CAS | `LocalFileSystem` observation registry + content-hash version CAS (`ErrStaleVersion`/`ErrNotObserved`) | `integration/local/local_test.go`, `tools/core/core_test.go`, `agent/e2e_test.go` (scenarios A, B) |
| sandbox containment | `LocalSandbox` same boundary for file and command admission; stable `SANDBOX_DENIED_*` codes; never suggests a bash bypass | `integration/local/local_test.go`, `agent/e2e_test.go` (scenario I) |
| crash recovery | `FencedStore.Recover` (turn close + `TOOL_OUTCOME_UNKNOWN`), run by the loop before any new work | `agent/agent_test.go` (recovery), pgstore integration (G) |
| context overflow/compaction integrity | `preflight` prune->compact->bounded retry; durable `compaction/start|summary|end` + ledger checkpoint | `agent/agent_test.go`, `agent/e2e_test.go` (scenarios E, F) |
| tenant isolation | `pgstore.WithTenant`; fenced ops require a configured tenant (`ErrNoTenant`) | pgstore integration (uses explicit tenant) |
| stateful public Agent | durable ordered inbox, next-step steering/injection, approval pause/resume, idle/dispose lifecycle | `agent/agent_test.go` |
| commit-before-publish events | replay/live handoff, sequence ordering under concurrent commits, bounded lag termination and cursor recovery | `agent/events.go`, `agent/agent_test.go` |
| typed content and streaming recovery | block codecs, durable assistant chunks, assembled interrupted drafts and media | `session/blocks.go`, `session/recovery.go`, `agent/loop.go` |
| scoped tool authority | local shadowing/restrictions, native/code/both presentation, policy/guard/finalizer/observer pipeline | `tools/tools_test.go`, `tools/registry.go` |

## Production-ready gate (§23.2)

- **no P0** — see table above.
- **≥90% P1 core points** — loop, session, surface, fenced store, compaction,
  sandbox, CAS, recovery, pruner, typed definitions, `oneOf`, and ordered
  bounded scheduling implemented with sentinels and deterministic schemas.
- **core tools have typed schemas and contract tests** — `tools/core` input and
  output schemas are generated from the typed `DefineTool[I,O]` contracts and
  exercised by `tools/core/core_test.go` and the parity suite
  `wzhooh-back/internal/agentkitruntime/e2b/e2b_test.go`.
- **DB replay rebuilds identical surface** — `session/pgstore/integration_test.go`
  loads the log on a fresh store and rebuilds the projection (scenarios H, G).
- **fake provider runs the full Agent Loop without network** — `agent/agent_test.go`
  and `agent/e2e_test.go` use a scripted `Chat`; the host adapter exercises the
  same wiring via `internal/agents/adapter_test.go`.
- **`go test -race` passes** — CI-only (not runnable without cgo on this box).
- **E2E A–J confirmed** — see below.

## E2E trace map (§22)

| # | Scenario | Enforcing test |
|---|---|---|
| A | Safe edit (read V1 → edit commits vs guard) | `agent/e2e_test.go` → `TestE2E_ScenarioA_SafeEdit` |
| B | Stale file (external write → FS_STALE_VERSION, no clobber) | `agent/e2e_test.go` → `TestE2E_ScenarioB_StaleFile` |
| C | Parallel reads (order preserved, valid pairing) | `tools/tools_test.go` → `TestRunBatchCommitsModelOrderAcrossParallelCalls`; `agent/e2e_test.go` (surface pairing) |
| D | Long output (bounded, truncation marker) | `agent/e2e_test.go` → `TestE2E_ScenarioD_LongOutput` |
| E | Proactive compaction (pressure → compaction events → below threshold) | `agent/agent_test.go` → `TestCompactionOnOverflow` |
| F | Provider overflow (compact → bounded retry → continues) | `agent/e2e_test.go` → `TestE2E_ScenarioF_ProviderOverflow` |
| G | Crash after side-effect start (no re-execution, TOOL_OUTCOME_UNKNOWN) | `agent/agent_test.go` (`TestNoReExecutionOnRecovery`), `agent/e2e_test.go` (G), pgstore integration (G) |
| H | PostgreSQL restart (new worker, identical surface, same turn) | `session/pgstore/integration_test.go` (external DB) + `embedded_integration_test.go` (embedded native PG) |
| I | Sandbox denial (stable code; bash obeys same boundary) | `agent/e2e_test.go` → `TestE2E_ScenarioI_SandboxDenial` |
| J | Cancellation (no new starts; nothing durable after stop) | `agent/agent_test.go` (`TestCancellationStopsBeforeWork`), `agent/e2e_test.go` (J) |

## Safety invariants carried through

- The loop advertises parallel tool-call capability, persists every call before
  any body, and the scheduler overlaps only explicitly safe definitions while
  committing results in model order.
- Raw history is never deleted on compaction.
- Wzhooh production integration is not accepted until it uses the host-owned
  execution guard and canonical ledger. The legacy MemoryStore adapter is a
  compatibility/demo path, never a production executor.
