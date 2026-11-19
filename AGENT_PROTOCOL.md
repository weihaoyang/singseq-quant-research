# Quant Research Protocol 2.0

## Purpose

The Quant Research Protocol lets any compatible Agent use a user-controlled
research runtime without moving the user's data, source code, or computation
onto an operator-controlled server.

The protocol separates four roles:

- The Agent plans, writes code, and explains results.
- The local runtime exposes tools, data access, and compute.
- Data providers retain custody of their data.
- Signed artifacts and receipts make results portable and independently verifiable.

Protocol identity, client identity and competition identity are separate:
`quant-research-protocol`, `quant-research-client` and `quant-arena`. A platform
product code used for the existing OAuth/catalog registration must not be
treated as a prerequisite for local research.

## Discovery

A conforming local runtime exposes:

- `app.info`: installation, launcher, data root, toolkit root, and platform.
- `agent.manifest`: transports, MCP profiles, tools, prompts, and resources.
- `agent.handshake`: the current readiness and security snapshot.
- `agent.integrations`: copy-ready local MCP configuration for supported Agents.
- `protocol.conformance`: the same read-only suite through the local runtime.

Runtimes declare one of three tool-surface profiles in
`quant_agent_handshake`:

- `agent` is the default local research profile. It contains 25 stable
  first-class tools for runtime discovery, local data selection and inspection,
  factor discovery and screening, Alpha Graph execution, canonical local
  backtests, durable job inspection and portable Evidence Packs. It contains no
  competition submission, publication, account-write, executable pack
  installation or generic command bridge.
- `core` is the portable protocol baseline for dependency-thin runtimes. It
  exposes local data inspection, capability management, Alpha Graph, factor
  screening, candidate-family audits, Evidence Packs, connectors and benchmark
  evaluation. It may use a native confirmation dialog instead of the desktop
  operation-grant flow.
- `full` adds account-backed data, competition and explicit official-evaluation
  tools plus advanced private evaluation and capability management.

`agent` is the default profile for stdio MCP and loopback nodes. A runtime
MUST expose account-backed data, competition or external-write tools only when
the caller explicitly selects `full`; omitting a profile MUST NOT inherit the
maximum surface.

Profiles are role presets, not authorization grants. The default `agent`
surface is the recommended Agent entry; `core` is the portable independent
runtime surface and `full` is an explicit advanced surface. A caller that needs
an external action must select `full` and still satisfy the tool's separate
authorization rule.

A runtime MUST NOT claim a profile unless every tool assigned to that profile is
present. The tool assignment and resulting counts are machine-readable in
`agent.manifest`, so documentation MUST NOT maintain a separate hand-written
tool inventory.

The canonical profile membership is declared once in
`protocol/profiles.json`. Desktop manifests and the standalone verifier are
generated from that file; implementations MUST NOT maintain independent
hand-written copies of the profile sets.

The client may be installed at any filesystem location. Implementations MUST
derive paths from the running installation and MUST NOT require a fixed drive,
username, workspace, or package directory.

`app.info` and the handshake security block SHOULD expose the separately
distributed `.qalpha` verifier path, byte size, and SHA-256 digest. The verifier
is a local executable, never a member executed from the artifact being checked.

## Transports

A conforming runtime MUST expose the same command and tool semantics over:

- stdio MCP for a process launched by the Agent.
- Loopback Streamable HTTP for a persistent local research node.

The loopback transport MUST bind only to `127.0.0.1` or `::1`, require a local
bearer token, and refuse control commands that would recursively start or stop
the daemon through itself.

Normal discovery manifests MUST redact the daemon bearer token. A token may be
returned only by an explicit local configuration-generation call after the user
requests a copy; it MUST NOT be embedded in `agent.manifest`, handshakes,
resources, logs, or ordinary daemon status.

## MCP Handshake

The MCP initialize response advertises tools, prompts, and resources. Agents
SHOULD start with `quant_agent_handshake` or `quant://handshake`.

`quant-agent-handshake/1` contains:

- protocol name and version.
- local client version and command count.
- identity, compute, data, and persistent-node readiness.
- a redacted daemon status.
- local execution and disclosure guarantees.
- ordered `nextActions` that do not mutate external state by themselves.

Credentials and bearer tokens MUST NOT appear in the handshake.
Every `nextActions` command MUST be callable through a first-class tool in the
advertised profile; a runtime MUST NOT recommend a generic-bridge-only or
otherwise unreachable command.

## Resources

A conforming runtime MUST expose these stable resource URIs:

- `quant://handshake`: live machine-readable readiness.
- `quant://playbook`: canonical research workflow.
- `quant://capabilities`: current command surface.
- `quant://security`: execution, disclosure, and publication invariants.
- `quant://data/status`: the selected local panel binding.
- `quant://operators`: the current causal operator catalogue accepted by
  Alpha Graph.

Resource reads MUST NOT upload local content or change research state.

## Prompts

A conforming runtime SHOULD expose task recipes that are versioned with the
server rather than copied from a website:

- `quant_research`: bootstrap a local Alpha study.
- `quant_reproduce_alpha`: verify and reproduce a `.qalpha` artifact.
- `quant_private_evaluate`: evaluate at a custodian's data boundary.

Prompts are instructions, not permissions. A prompt MUST NOT authorize a
submission, publication, purchase, source disclosure, or permission change.

## Tool Contracts

Every tool:

- has a JSON Schema input.
- validates the supplied arguments against that schema before execution.
- returns a JSON envelope with `ok`, `data`, or an error object containing a
  stable `code`, `message`, and optional `stage`, `status`, and `details`.
- declares an output schema that separates the success and error branches and,
  for the default `agent` profile, constrains the success `data` shape.
- declares whether it is read-only and whether it accesses an open world.
- uses the same implementation as the CLI and desktop client.

Profiles are role presets, not authorization grants. The default `agent`
profile MUST NOT expose `quant_command`, executable capability installation,
account-backed writes, competition submission or publication. A runtime MUST
generate prompts and the `quant://playbook` resource from the tools actually
advertised by the active profile; a playbook MUST NOT mention a command name or
tool absent from that surface.

`capabilities` exposes `command_contracts`, and command-level
`--help --json` returns the same schema and annotations used by MCP. Legacy
desktop commands without a declared contract remain CLI implementation details
and are excluded from `agent_commands`. An Agent MUST use the advertised MCP
tools or contracted CLI commands directly; there is no generic command bridge.
The transport entrypoints `mcp.serve` and `agent.serve` remain listed so
discovery can start the protocol surface; they are not callable as research
tools.

First-class tools are provided for discovery, data inspection, Alpha Graph, factor screening,
single-factor audits, aligned candidate-family audits, Evidence Packs,
benchmarks, connectors, private evaluation, local runtime
bootstrap, local research checks and runs, factor mining and batching,
robustness diagnosis, local portfolios, runtime/data status, job listing,
result inspection, explicit local data selection, explicit project data binding,
explicit job cancellation and local daemon lifecycle control. Local data
selection must accept an explicit local path and must not require a platform
account or upload the selected file.
Competition submission MUST be its own destructive tool and MUST require a
one-time operation grant after the user explicitly approves the external
action. Listing completed local candidates is a separate read-only operation
and MUST NOT submit them. The Agent may request and inspect a grant, but the
approval operation MUST NOT be exposed through CLI or MCP. Approval binds the
action, owner, resource identity and expiry and is single use. A literal
confirmation value is not user authorization.

Alpha Graph nodes SHOULD expose the same operator catalogue returned by
`factor.operators`. An operator node binds one or more graph inputs to a
declared causal operator, resolves panel context such as `stocks` and `dates`,
and carries the operator parameters in the graph artifact. Implementations
SHOULD reject unknown operators and cyclic graphs before reading the panel.
The catalogue MUST distinguish canonical operators from compatibility aliases,
report both counts, and give every alias an `aliasOf` target. Aliases resolve to
the same implementation and MUST NOT be counted as additional capabilities.
The graph tool accepts `backend=auto|cpu|gpu`. A runtime that reports CUDA
acceleration MUST identify which stages ran on the device and MUST report
`cuda-diagnostics` rather than claiming the entire DAG ran on GPU when graph
arithmetic or unsupported operators stayed on CPU. OOM or device failure falls
back to the same deterministic CPU implementation and records the reason.
For large result sets, the cross-output correlation matrix MAY use a
deterministic row sample to bound memory and runtime. The result MUST report the
sampled row count and sampling method; an implementation MUST NOT present a
sampled matrix as a full-panel calculation.

Candidate-family evidence MUST be derived from a caller-supplied set of local
candidate definitions evaluated on one bound data slice. It MUST NOT be
assembled from historical server-side Sandbox tasks. A conforming local audit
aligns after-cost held-out returns, computes CSCV/PBO and multiple-testing
evidence, and records the selected candidate. A `.qalpha` Evidence Pack MAY
embed this report as `evidence/family-audit.json`; a publication-grade pack
requires the family matrix to be sealed, PBO to satisfy the declared gate, and
the selected candidate to survive the declared multiple-testing correction.

### Capability Packs

Local extensions are distributed as `qcap/1` capability packs rather than
requiring a replacement client build. Factor Packs declare namespaced factors,
required fields, minimum history and runtime compatibility. Operator Packs
declare namespaced causal operators and their entries. Benchmark Packs are
declarative `qbench/1` rules that bind a namespaced benchmark id to a data
contract and an open-to-open evaluation rule without shipping executable code.
Connector Packs declare namespaced provider ids, credential environment names,
field capabilities, point-in-time semantics, an exact `allowedHosts` allowlist
and one controlled fetch entry. The host supplies only bounded, allowlisted
`http_json` and credential values to that entry; credential-bearing URLs and
cross-origin redirects are rejected. The returned frame MUST be normalized into
the standard local panel contract.
Installation MUST validate paths and digests and apply the factor AST contract
before executable pack code is activated.

Discovery MUST read metadata only and MUST NOT execute pack code. Installed
Factor Packs become part of the same `factor.names`, Alpha Graph, batch and
backtest surfaces as built-in factors. Operator Packs become part of
`factor.operators` and the Alpha Graph runtime. Pack installation or removal is
an explicit local state change and MUST NOT be reachable through a generic
read-only command bridge.

Activating a new pack version MUST retain already installed versions. Only one
version of a pack id is active at a time, and discovery reads the active
version. Users may explicitly activate a retained version to reproduce an
older Evidence Pack. Reproduction MUST fail closed with a capability mismatch
when the required pack version, content digest or referenced factor/operator
entry is not active instead of silently calculating with newer pack code.

Built-in discovery SHOULD use a generated capability manifest filtered by the
local panel schema. Discovery MUST NOT import optional execution libraries or
evaluate factor code merely to list candidate names. Installed capability
metadata is merged live so adding or removing a pack does not require a client
release.

Because Factor Packs and Operator Packs can add executable local code, the MCP
install and removal tools MUST be marked destructive and require a user
approval surface outside model control. Desktop hosts use a one-time operation
grant; the dependency-thin Python host uses a native confirmation dialog and
fails closed when no interactive desktop is available.

Capability manifests MAY carry an Ed25519 publisher signature over their
canonical unsigned contents. Installation MUST reject an invalid signature.
Supplying a trusted publisher key MUST also reject a missing or different key
and may set `publisherVerified=true`. Signature validity and publisher trust
remain separate claims.

## Local Execution Boundary

Factor mining, sweeps, training, diagnosis, and backtests MUST execute on the
user's computer or inside a data custodian's boundary.

The production service MUST NOT start a research job for the desktop client.
It may publish data, rules, signed results, and separately produced evaluator
receipts.

Legacy market and catalog research routes MUST fail closed with
`LOCAL_COMPUTE_REQUIRED`; they MUST NOT execute portfolio optimization,
alphalens diagnostics or a backtest in the web process. The equivalent local
surfaces are `portfolio.local` and `ml.alphalens`.

Competition registration binds an owner-scoped factor version and a previously
registered local result artifact. The explicit official-evaluation action takes
`factorId`, `versionId` and `artifactId`; a successful Sandbox task is not a
prerequisite for newly created work. Historical source-task requests may remain
readable, but no client path may create a new ordinary Sandbox research task.
The reference evaluator is a separate, network-disabled worker container. Third
parties may operate an independent evaluator, but a process boundary alone MUST
NOT be described as an independent operator or hardware attestation.

A factor run is inert with respect to publication. Submission, ranking,
publication, source disclosure, purchase, and permission changes require a
separate explicit user action.

Submission and official-evaluation requests each consume a separate one-time
grant. A grant for one job or artifact cannot authorize another operation,
another account, or a later request.

A saved project binds the validated identity of the data panel used for its
first run. A runtime MUST reject a later run when the selected data identity
changes unless the user explicitly rebinds the project or creates a new
project. This prevents a stable project name from silently changing its
research input.

## Data Federation

Connector discovery reports SDK and credential readiness without returning
credentials. Connector pulls write normalized panels to a user-selected local
path.

`qconnector/1` advertises the `qpanel/2` metadata contract. `qpanel/1` remains
the executable bar-panel wire format, while `qpanel/2` adds explicit
`kind`, `identity`, `eventTime`, `availableAt` and `pitRule` fields for bars,
reference, fundamental, event and flow tables. Multi-table bundles use an
as-of join contract and MUST reject future availability rather than silently
forward-filling provider data.

Connector pulls for local files and warehouses MAY request a non-bar
`tableKind`, explicit identity, `eventTime` and `availableAt`; the runtime writes
the normalized table plus a `qpanel/2` sidecar manifest. `connector.join` MUST
join on availability time, never event time, and MUST write a `qpanel-join/1`
receipt. Provider endpoints that only expose bars MUST reject a non-bar request
rather than relabeling OHLCV as fundamentals or events.

Benchmark evaluation may bind a second, explicitly supplied market-index or ETF
series. The runtime computes forward returns from that series and records its
source, symbol, coverage and price column in the Evidence Pack. An equal-weight
universe proxy remains clearly labelled as a diagnostic approximation and MUST
not be presented as a third-party index.

All factor diagnostics MUST use the same forward label: observe the factor at
session close ``t``, enter at ``open(t+1)``, and exit at
``open(t+1+horizon)`` on the adjusted-price basis. Raw-close close-to-close
returns may be shown only as an explicitly labelled diagnostic proxy. Graph,
screen, benchmark, audit and canonical backtest results must expose the label
version and adjustment basis rather than leaving the convention implicit.

When benchmark and strategy returns are compared, both MUST use the same
return window. A market benchmark computed from open prices uses
next-open-to-next-open returns; a close-only benchmark must be labelled as a
close-to-close proxy and cannot by itself qualify a publication-grade proof.
Adjusted and unadjusted prices MUST NOT be spliced row-by-row. If adjustment
coverage is incomplete, the runtime chooses one complete basis or fails the
publication-grade claim.

Supported connector families include local CSV, Parquet, DuckDB, Binance and
Yahoo public market data, Tushare, Polygon, Databento, CCXT, Snowflake,
BigQuery, and WRDS. A connector is conforming when it preserves the provider's access
boundary and normalizes the result to documented panel fields.

Warehouse connectors accept one read-only `SELECT` or `WITH` statement.
Multi-statement SQL, DDL, DML, transaction control, row locks and
server-side write operations are rejected before execution. Warehouse pulls
MUST expose bounded `maxRows` and timeout controls; BigQuery callers SHOULD
also set a `maxBytesBilled` preflight limit. Provider-side read-only
credentials remain required as the final control.

## Alpha Evidence

A `.qalpha` artifact SHOULD bind:

- factor graph or source.
- parameters and canonical output names.
- exact data-slice identity.
- evaluation window and benchmark.
- runtime, dependency and referenced capability-pack identity.
- canonical label and Alpha Graph compiler identity.
- optional sealed local candidate-family evidence with PBO and multiple-testing
  results.
- metrics, costs, diagnostics, and optional result tables.
- an optional Ed25519 signature.

The manifest MUST distinguish a private research pack from a
publication-grade pack. `publication.eligible=true` requires factor identity,
data identity, a complete audit, PIT semantics, cost model, benchmark,
selection protocol, out-of-sample evidence, capacity evidence and leakage
checks. Pack creation never publishes anything; publication remains an
explicit external action.

Before packaging, a runtime SHOULD expose `qalpha-audit/1` through
`evidence.audit`. The audit:

- fixes the factor direction using selection-period information only;
- rebuilds stored factor prefixes after removing all future dates;
- records a chronological selection window and a purged/embargoed holdout;
- runs the canonical local backtest after costs in both windows;
- compares the holdout against a declared real benchmark series;
- derives impact-cost scales from realised order notionals and observed market
  amount rather than an assumed AUM;
- reports `LOCAL_PROOF_OF_ALPHA` only when causality, holdout direction,
  after-cost excess, benchmark and capacity checks all pass.

When an audit is available, `evidence.pack` SHOULD accept it as `auditFile`
and embed the complete report at `evidence/audit.json`. The manifest MUST
mirror the audit format, status, validation flag and report digest. Packaging
MUST reject an audit whose data window or factor identity differs from the
pack. The independent verifier MUST expose the embedded audit report without
executing factor source, so out-of-sample and cost-sensitivity tables remain
available without the Quant runtime.

An audit may reject a factor with durable out-of-sample predictive power when
its net-of-cost portfolio remains inferior to the benchmark. That is a correct
outcome. `RESEARCH_ONLY` and `REJECTED_LEAKAGE` MUST NOT be presented as an
investment result or silently promoted to a proof.

Cryptographic validity and publisher trust are separate claims. A verifier MUST
report signature validity independently from whether a supplied public key is
trusted.

Every new `.qalpha` artifact SHOULD be independently verifiable without the
Quant runtime. A separately distributed `qalpha-verify.py` MUST verify the
manifest, every embedded member digest, and the optional Ed25519 signature
using only the Python standard library. It MUST NOT execute the archive,
execute factor source, or require the Quant runtime. Full value reproduction
remains a separate local operation that requires the bound market-data slice.

A `.qalpha` archive is data, not an executable program. A verifier MUST NOT
execute verification code taken from the artifact it is checking.

The portable verifier SHOULD stream member hashes instead of loading the whole
archive, and SHOULD impose explicit manifest, file-count, and uncompressed-size
limits before processing untrusted artifacts.

## Alpha Discovery

A local runtime SHOULD index `.qalpha` assets with `qalpha-index/1`.
Indexing MUST verify each archive and MUST NOT upload, execute, submit, or
publish it.

An index separates:

- archive integrity;
- cryptographic signature validity;
- publisher trust from supplied trust anchors;
- factor kind and identity;
- data-slice identity and window;
- research context and optional result metadata.

Assets over the same data-slice identity may be grouped for local comparison.
An index is discovery metadata, not an endorsement or ranking.

## Private Evaluation

Private evaluation binds a request, benchmark, data commitment, artifact
digest, evaluation window, and expiry. The evaluator signs a receipt only after
completing the run.

Software isolation using a process or OCI container is not a hardware enclave.
If remote attestation is unavailable, the receipt MUST state:

- `hardwareConfidential=false`
- `attestation=NOT_HARDWARE_ATTESTED`

An implementation MUST NOT describe self-reported environment fields as
verified remote attestation.

`sourceDisclosure` governs onward release of source in returned evidence. It
does not make source invisible to the evaluator operator. When the `.qalpha`
artifact contains source and no confidential-compute backend is present, the
request and receipt MUST declare `operatorCanReadArtifactSource=true` and
`confidentialCompute=false`. A "source never visible to the data custodian"
claim requires a separately attested confidential-compute implementation.

## Conformance

A runtime is protocol-conforming when it:

1. exposes the discovery commands and MCP methods above.
2. returns the same tool semantics on both transports.
3. preserves the local execution boundary.
4. separates research from submission and publication.
5. can run `qalpha-audit/1` before an Evidence Pack is promoted as a proof.
6. never exposes stored credentials in manifests, handshakes, or logs.
7. emits stable JSON artifacts and exact local output paths.

Conformance is machine-checkable with the separately distributed
`qprotocol-verify.py`, which uses only the Python standard library. It supports
both an explicit stdio command and a loopback Streamable HTTP endpoint:

```powershell
python qprotocol-verify.py `
  --command node `
  --arg electron/cli.cjs --arg mcp.serve `
  --json --output conformance.json

python qprotocol-verify.py `
  --url http://127.0.0.1:PORT/mcp `
  --token-env QUANT_AGENT_TOKEN `
  --json --output conformance-http.json
```

The verifier performs read-only discovery only. It checks the initialize
handshake, required tools and annotations, resources, prompts, local execution
claims, separation of research from submission, and accidental exposure of
environment credential values. The receipt format is `qprotocol-verify/1` and
binds a SHA-256 digest of the observed MCP transcript.

A conformance report proves that an observable implementation matches this
contract. It is not a security audit, factor-performance proof, publisher
identity, or investment endorsement.

Every published report MUST carry `generatedAt`, verifier version and digest,
the profile-manifest digest, the observed client/runtime versions and a
path-redacted target. A report generated by a source checkout or an unprepared
runtime MUST NOT be published as the current packaged-client report.

### Semantic Conformance

Structural conformance alone cannot prove that two runtimes calculate the same
Alpha. A runtime MAY additionally be checked with `--semantic`. This executes
one deterministic `alpha-graph-operator-vector/1` over synthetic panel data in
an operating-system temporary directory. The vector covers field selection,
per-stock lag, guarded division, a trailing window, and cross-sectional rank.
Observed values are compared against a standard-library implementation with an
absolute tolerance of `1e-9`.

Semantic conformance does not read user data, use a named factor, submit or
publish anything, or mutate persistent research state. Its receipt is embedded
at `semanticConformance` and reports `executedSyntheticVectors=true`,
`executedFactors=false`, `uploadedLocalContent=false`, and
`mutatedResearchState=false`.

The semantic suite executes through the runtime's own `quant_alpha_graph`
tool, so a desktop implementation must have its local Python research runtime
prepared before semantic mode can run. The verifier never installs, downloads
or repairs that runtime; an unprepared runtime is reported as a conformance
failure rather than silently switching to another implementation.

```powershell
python qprotocol-verify.py `
  --command node `
  --arg electron/cli.cjs --arg mcp.serve `
  --semantic --json --output conformance.json
```

### Independent Python Host

`quant_sdk.protocol_host` is a second, dependency-thin implementation of the
same MCP contract. It does not import Electron or Node and exposes only the
protocol-critical tools. The host can be launched directly:

```powershell
python -m quant_sdk.protocol_host --transport stdio
python -m quant_sdk.protocol_host `
  --transport http --host 127.0.0.1 --port 8765 `
  --token-env QUANT_AGENT_TOKEN
```

The HTTP transport requires a bearer token and refuses non-loopback bind
addresses. Both transports pass the structural and semantic suites:

```powershell
python qprotocol-verify.py `
  --command python `
  --arg -m --arg quant_sdk.protocol_host `
  --semantic --json
```

This runtime is a portability reference, not a claim that another independent
organization has adopted the protocol. It provides a headless path for
macOS/Linux development while the Windows desktop client remains a separate
delivery artifact and must not be described as code-signed unless the published
binary carries a verifiable publisher signature.

Protocol versioning is independent from client versioning. Breaking changes to
the handshake, resources, prompts, or tool envelope require a new protocol
major version.
