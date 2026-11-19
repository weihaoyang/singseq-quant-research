# Quant Research Protocol Changelog

## 2.0

- Separates protocol, client and competition product identities.
- Canonical three-profile tool manifest at `profiles.json`.
- Adds a 25-tool default `agent` surface for local data discovery, factor
  research, jobs and Evidence Packs without platform writes or a generic bridge.
- Consolidates legacy `host`, `local` and `research` profiles into `full`;
  removes the generic command bridge and non-research workspace, daemon,
  storage, report and package-import tools from the Agent protocol.
- Local execution is mandatory for ordinary factor calculation, sweeps,
  training, diagnosis and backtests.
- Research output is inert; submission, publication and ranking require
  separate explicit authorization.
- Defines portable Evidence Packs (`qalpha/1`), capability packs (`qcap/1`),
  connector/panel contracts (`qconnector/1`, `qpanel/2`) and benchmark packs
  (`qbench/1`).
- Adds structural and synthetic semantic conformance verification.
- Makes capability-pack version, content digest and factor/operator entry part
  of reproduction matching.

Protocol 2.0 uses MCP protocol version `2025-06-18`. A breaking change to
discovery, resources, prompts or the tool envelope requires a new protocol
major version.
