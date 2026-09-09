# Contributing to ToolScope TS

Contributions are welcome. Bug fixes, framework compatibility improvements, new retrieval backends, documentation corrections, and focused performance work are all useful.

## Before you start

- Search the [existing issues](https://github.com/0xJord4n/ToolScope-TS/issues) before opening a duplicate.
- Open an issue first for substantial API changes or new dependencies.
- Report security vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Development setup

ToolScope TS uses Bun for dependency management, tests, formatting, linting, and builds.

```bash
git clone https://github.com/0xJord4n/ToolScope-TS.git
cd ToolScope-TS
bun install
bun run check
bun run package:check
```

The repository currently validates against Bun 1.3.x. Use a current Bun release when developing locally.

## Making a change

1. Fork the repository.
2. Create a focused branch from `main`:

   ```bash
   git checkout -b fix/short-description
   ```

3. Make the smallest coherent change that solves the problem.
4. Add or update tests for behavior changes and bug fixes.
5. Update examples or public documentation when an API changes.
6. Run the complete verification suite:

   ```bash
   bun run check
   bun run package:check
   ```

7. Submit a pull request with a clear summary and test plan.

## Code standards

- Keep the package strict and type-safe.
- Preserve original executable tool objects through normalization and selection.
- Treat tags, namespaces, annotations, metadata, and executable identity as security-sensitive during synchronization.
- Validate malformed vectors, scores, provider responses, and persistent data fail-closed.
- Keep framework packages optional unless the core runtime directly depends on them.
- Do not add network calls, telemetry, or model downloads without explicit configuration.
- Never commit credentials, local databases, generated traces, or environment files.

## Commands

| Command                 | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `bun test`              | Run the test suite                                   |
| `bun run typecheck`     | Run strict TypeScript checks                         |
| `bun run lint`          | Run Oxlint with warnings denied                      |
| `bun run lint:fix`      | Apply safe lint fixes                                |
| `bun run format`        | Format the repository with Oxfmt                     |
| `bun run format:check`  | Check formatting without writing                     |
| `bun run build`         | Build ESM JavaScript and declarations into `dist/`   |
| `bun run check`         | Run formatting, linting, typecheck, tests, and build |
| `bun run package:check` | Validate and dry-run the npm package                 |

## Tests

Tests live in `tests/` and run with Bun's test runner. Prefer deterministic tests that do not require paid APIs or provider credentials.

- New features should include focused behavior tests.
- Bug fixes should include a regression test.
- Adapter changes should compile and run against the real framework package.
- Persistence changes should test reopening and malformed stored data.
- Selection changes should verify returned tool identity, policy behavior, and trace accuracy.

## Pull requests

A strong pull request includes:

- a concise explanation of the problem and solution
- links to relevant issues
- tests covering the changed behavior
- documentation for public API changes
- no unrelated formatting or refactoring
- a passing `bun run check` and `bun run package:check`

Use clear commit messages such as:

```text
feat: add a vector backend
fix: refresh changed policy metadata
docs: clarify MCP invalidation
```

By contributing, you agree that your contribution will be licensed under the repository's [Apache-2.0 license](LICENSE).
