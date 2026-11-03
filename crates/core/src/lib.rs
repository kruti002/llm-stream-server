//! # llm-stream-core
//!
//! The heart of the `llm-stream-server` project: the shared wire types, the
//! error type, the [`Backend`] abstraction, and a deterministic
//! [`MockBackend`] used by the test suite and CI.
//!
//! ## Key design decisions
//!
//! These are the choices worth being able to explain; the "why" lives next to
//! each item in its own module, summarized here.
//!
//! * **A `Backend` trait ([`backend`]).** Inference may come from Ollama,
//!   llama.cpp, or a mock. Programming the server against `dyn Backend` means
//!   the backend is a startup choice, and the deterministic [`MockBackend`]
//!   makes every higher layer testable without a model.
//!
//! * **The token stream is boxed ([`BoxStream`]).** Each backend yields a
//!   different concrete `Stream` type; boxing erases that difference into one
//!   uniform return type, which is what keeps `dyn Backend` object-safe. The
//!   `'static` bound lets the stream be handed off to an SSE response that
//!   outlives the originating call.
//!
//! * **Cancellation is explicit and reusable ([`mock`]).** Dropping the stream
//!   must stop generation. We model this with a [`tokio_util`] `CancellationToken`
//!   plus a channel: the stream owns a drop-guard that cancels the token, and
//!   the producer stops at its next await. That is the same primitive the
//!   server will wire to client-disconnect handling in a later milestone.
//!
//! * **A real [`OllamaBackend`] ([`ollama`]).** Streams tokens from a local
//!   Ollama daemon by parsing its newline-delimited JSON over `reqwest`'s byte
//!   stream. It reuses the same `CancellationToken` drop-guard pattern as the
//!   mock so dropping the stream stops reading the HTTP body, keeping
//!   cancellation behavior identical across backends.
//!
//! * **One error type ([`error`]).** A single coarse-grained [`Error`] enum
//!   keeps the trait object-safe and gives the server a stable set of variants
//!   to map onto HTTP status codes.
//!
//! [`BoxStream`]: futures::stream::BoxStream
//! [`tokio_util`]: https://docs.rs/tokio-util

mod backend;
mod error;
mod mock;
mod ollama;
mod types;

pub use backend::Backend;
pub use error::{Error, Result};
pub use mock::{MockBackend, MOCK_SENTENCE};
pub use ollama::{OllamaBackend, DEFAULT_OLLAMA_URL};
pub use types::{GenerateRequest, TokenChunk};
