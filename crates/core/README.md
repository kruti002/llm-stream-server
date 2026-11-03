# llm-stream-core

Core building blocks for the `llm-stream-server` project: the shared wire
types, a single error type, the `Backend` trait, and a deterministic
`MockBackend`.

This crate is intentionally dependency-light. It does **not** pull in an HTTP
client or any model runtime — Ollama and the optional `llama-cpp` backend arrive
in a later milestone. Everything here builds and tests without a GPU, a network,
or a downloaded model.

## What's in the box

| Item | Purpose |
|---|---|
| `GenerateRequest`, `TokenChunk` | serde-serializable wire types shared by the server, the Python bindings, and the bench tool |
| `Error` / `Result` | one coarse error enum (`thiserror`) mapping to HTTP status classes |
| `Backend` trait | the pluggable inference abstraction (`async_trait`) |
| `MockBackend` | deterministic, model-free backend used by tests and CI |

## Key design decisions

### Why a `Backend` trait?

Tokens can come from very different places — a local Ollama daemon over HTTP, an
in-process llama.cpp runtime, or a fake. Writing the server once against
`dyn Backend` keeps the backend a *startup* choice rather than something baked
into every handler. It also makes the whole stack testable: `MockBackend` gives
higher layers predictable output so we can assert on ordering, indices, and
cancellation without any model.

### Why is the stream boxed?

Each backend produces a *different* concrete `Stream` type (an HTTP body parser,
a channel receiver, and so on). A single trait method cannot name all of them,
so `generate_stream` returns a `BoxStream<'static, Result<TokenChunk, Error>>`.
Boxing erases the concrete type into one uniform return type — that is what
makes `dyn Backend` object-safe. The `'static` lifetime lets the stream be
handed to an SSE response that outlives the call that created it.

### How does cancellation work?

Dropping the token stream must stop generation (in the server, this happens when
a client disconnects). `MockBackend` models this with the same primitive a real
backend would use:

1. A background task produces tokens and pushes them through a capacity-1
   `tokio::sync::mpsc` channel (which also provides backpressure).
2. The returned stream owns a `tokio_util::sync::CancellationToken` **drop
   guard**. When the consumer drops the stream, the guard cancels the token.
3. The producer `select!`s between sleeping for the next token and the token
   being cancelled, so it stops at its next await point instead of finishing the
   sentence.

The `CancellationToken` is deliberately the reusable piece the server crate will
later connect to client-disconnect handling.

## Running the checks

From the workspace root (`llm-stream-server/`):

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
