# Security Policy

ToolScope TS sits between tool catalogs and model calls. Selection policy, tool identity, persisted descriptors, traces, and framework adapters should therefore be treated as security-sensitive boundaries.

## Supported versions

Security fixes are provided for:

- the latest version published on npm
- the current `main` branch until the next release

Older releases may not receive backported fixes. Upgrade to the latest version before reporting an issue that may already be resolved.

## Reporting a vulnerability

**Do not open a public GitHub issue for a suspected vulnerability.**

Use [GitHub's private vulnerability reporting](https://github.com/0xJord4n/ToolScope-TS/security/advisories/new) to send a confidential report to the maintainer.

Include as much of the following as possible:

- affected version or commit
- impact and realistic attack scenario
- minimal reproduction or proof of concept
- expected and actual behavior
- affected runtime and adapter
- suggested mitigation, if known

Please avoid including real credentials, private prompts, production tool catalogs, or customer data. Use synthetic examples wherever possible.

## Security considerations for users

- Treat tool descriptions, schemas, annotations, tags, and MCP catalog data as untrusted input.
- Treat `allowTags`, `denyTags`, namespaces, and policy callbacks as retrieval constraints—not authorization controls—when their inputs come from tool metadata.
- Enforce authorization independently at execution time using trusted application identity and permissions.
- Re-authorize the exact selected tool immediately before execution when permissions can change during a request.
- Do not persist executable closures in SQLite; persist JSON-safe descriptors and reconnect implementations in application code.
- Validate and protect embedding endpoints and custom rerankers as external dependencies.
- Redact prompt text and tool names before storing traces when they may contain sensitive data.
- Keep framework and runtime dependencies updated.

## Scope

Examples of security issues that should be reported privately include:

- policy bypasses or stale authorization metadata
- returning or executing a different tool than the selected original
- malformed vectors or persisted records causing unsafe selection
- MCP catalog invalidation or synchronization errors that retain revoked tools
- trace sinks exposing secrets despite configured redaction
- dependency or package-integrity issues affecting consumers

General bugs, feature requests, and usage questions can be reported through the public [issue tracker](https://github.com/0xJord4n/ToolScope-TS/issues).
