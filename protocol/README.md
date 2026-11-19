# Quant Research Protocol 2.0

This directory is the machine-readable protocol authority for tool surfaces,
conformance, local execution and portable Alpha evidence.

Normative artifacts:

- `AGENT_PROTOCOL.md`: discovery, MCP, local execution, data and Evidence Pack
  requirements.
- `profiles.json`: the single source for the `agent`, `core` and `full` tool
  sets. `agent` is the default 25-tool local research surface; `full` is the
  explicit advanced surface.
- `schemas.json`: every advertised tool's input schema, structured output
  schema and safety annotations, plus resources and prompts.
- `qprotocol-verify.py`: standalone standard-library verifier for structural
  and optional semantic conformance.
- `qalpha-verify.py`: standalone standard-library verifier for `.qalpha`
  archive integrity and signatures.

The desktop client, Python reference host and generated conformance reports
must agree with these artifacts. Tool counts are outputs of `profiles.json`;
they are not maintained as independent documentation.

Conformance proves the observable contract. It is not a security audit,
investment endorsement, publisher identity or proof that a factor is
profitable.
