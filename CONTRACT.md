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
- Tool schemas are object-rooted and typed definitions are the source of
  truth. Tool-owned `Guidance` is separate from the model-facing description;
  `ToolRunContext` carries call/root/parent identity and a cancellation signal,
  while `DeferContext` and `ConcludeTurn` report turn lifecycle explicitly.
- `PresentCall`, `PresentResult`, and content renderers are pure projections:
  canonical output stays runtime-owned, model content uses AgentKit blocks, and
  UI metadata remains compact. Registrations, restrictions, policies, and
  guards can be disposed through scoped handles.
- Canonical values, model-facing blocks, and UI metadata are separate. The
  durable block vocabulary is text, reasoning, media, tool-call, tool-result,
  and namespaced extension blocks; streamed chunks are committed before they
  are exposed to subscribers.

## Wzhooh host rule

Wzhooh PostgreSQL rows and `chat_events` remain the canonical run and event
ledger. A host adapter maps a claimed fenced run into these public AgentKit
ports; it must not create a second `agentkit_*` history or lease.

## Tool ownership

AgentKit contains no product tool implementations. Hosts define concrete
typed tools with `DefineTool[I,O]`, provide their adapters and side effects,
and register those definitions in a `Registry`. AgentKit owns only the
provider-neutral schema, validation, policy, execution pipeline, continuation,
and observation machinery. The only built-in exception is the `run_code`
transport bridge used by Code Mode; it is not a product tool. Reference or
test adapters belong outside the library's production tool surface.

## Skill ownership

AgentKit owns only the provider-neutral skill mechanism: registry and provider
interfaces, parsing, deterministic discovery, immutable snapshots, policy
checks, durable catalog projection, explicit invocation handling, and the
generic model-facing `skill` loader. It does not ship product skill names,
descriptions, bodies, visual references, publication storage, or admin APIs.
Pinned runs load only from providers that explicitly guarantee immutable
lookup; mutable filesystem discovery remains live-only.

Skill activation loads the complete `SKILL.md` instructions only. A host may
provide pinned manifest resources and contained relative-path resources through
the resolver interfaces; the model reads exactly one referenced resource with
a later `skill` call. Resource loading is denied until the durable skill
activation has been restored into the runtime.

Hosts such as Wzhooh define and publish concrete skills through the AgentKit
API and register every real product tool in their own backend composition root.
The `allowed-tools` frontmatter field is advisory metadata only: it never
registers a tool, grants a capability, or widens the active tool policy.
Skill catalogs and load metadata use the existing host session ledger; they
must not create a second durable history.

## Acceptance

`acceptance-gates.md` is the executable verification matrix. Exploratory
DeepSeek and review documents are evidence only and cannot override this
contract.
