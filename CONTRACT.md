# AgentKit Harness Contract

AgentKit is a provider-neutral embedded Go harness. It owns durable-session
projection, model request assembly, tool execution ordering, and execution
policy. It does not own a product database, queue, UI, or sandbox lifecycle.

## Non-negotiable boundaries

- Durable history is distinct from model surface, canonical tool value, UI
  presentation, provider protocol, and execution world.
- A model tool call is persisted before its body starts. Results are appended
  in model order, including structured failures.
- Tool admission is central and fail-closed. Execution-world adapters recheck
  path and command containment.
- A turn has ordered model steps. Every step stores a request header before
  provider dispatch, then an assistant message/chunks and paired tool results.
- Compaction is an immutable surface replacement: raw history is retained and
  the newest model-visible tail is never compacted away.

## Public runtime surface

- Applications construct `agent.NewAgent(loop)` and interact through the
  stateful handle (`FollowUp`, `Steer`, `Inject`, `Cancel`, approval decisions,
  `WhenIdle`, and `Dispose`). `Loop.Run` remains a migration wrapper only.
- Durable facts are consumed with `Agent.Subscribe(afterSeq, filter)`: the
  store commits canonical sequences before the event hub publishes them, the
  hub replays then hands off to live delivery in sequence order, and a lagged
  subscriber reconnects from its last acknowledged cursor. Lifecycle
  notifications are deliberately live-only.
- A tool is registered as a typed `DefineTool[I,O]` (or an erased
  `Definition`) and executes through one ordered pipeline: snapshot and
  validate input, policy/guard, sandbox, timeout/cancellation, execution,
  canonical output validation, model/UI rendering, post-policy replacement,
  finalization, freeze, and observation.
- Canonical values, model-facing blocks, and UI metadata are separate. The
  durable block vocabulary is text, reasoning, media, tool-call, tool-result,
  and namespaced extension blocks; streamed chunks are committed before they
  are exposed to subscribers.

## Wzhooh host rule

Wzhooh PostgreSQL rows and `chat_events` remain the canonical run and event
ledger. A host adapter maps a claimed fenced run into these public AgentKit
ports; it must not create a second `agentkit_*` history or lease.

## Acceptance

`acceptance-gates.md` is the executable verification matrix. Exploratory
DeepSeek and review documents are evidence only and cannot override this
contract.
