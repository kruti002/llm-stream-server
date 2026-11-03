# llm-stream-server

A learning/interview project: a streaming LLM server built in small milestones,
with pluggable inference backends and a strong focus on explaining *why* each
piece is shaped the way it is.

> Status: **Milestone 1** — workspace scaffold plus the `core` crate (shared
> types, the `Backend` trait, and a deterministic `MockBackend`). The server,
> Python bindings, benchmark tool, and FFI crate land in later milestones.

## Workspace layout

```
llm-stream-server/
├── Cargo.toml          # workspace manifest (workspace.dependencies)
└── crates/
    └── core/           # types, error, Backend trait, MockBackend  (this milestone)
```

Future member crates named in the project plan — `server` (Axum), `py` (PyO3),
`bench`, and `ffi` — will be added under `crates/` and wired into the workspace
`members` list as they are built.

## Core crate

See [`crates/core/README.md`](crates/core/README.md) for the key design
decisions: why a `Backend` trait, why the token stream is boxed, and how
cancellation works.

## Checks

From this directory:

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
