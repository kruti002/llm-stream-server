//! The [`Backend`] abstraction.
//!
//! # Why a trait?
//!
//! Inference can come from very different sources: a local Ollama daemon over
//! HTTP, an in-process llama.cpp runtime, or — for tests and CI — a fake that
//! needs no model at all. Hiding all of them behind one trait means the server
//! is written once against `dyn Backend` and the concrete backend is chosen at
//! startup. It also makes the whole system testable: swapping in
//! [`MockBackend`](crate::MockBackend) lets the test suite exercise streaming,
//! ordering, and cancellation deterministically without a GPU or a network.
//!
//! # Why is the stream boxed?
//!
//! Each backend produces a *different* concrete `Stream` type (an HTTP body
//! parser, a channel receiver, etc.). A trait method cannot name all of those
//! at once, and `impl Trait` in return position is not available for the trait
//! objects we need. Boxing the stream into [`BoxStream`] erases the concrete
//! type to a single uniform return type, which is what makes `dyn Backend`
//! possible. `BoxStream<'static, _>` also means the stream owns everything it
//! needs and can outlive the `generate_stream` call — essential for handing it
//! off to an SSE response.

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{Error, Result};
use crate::types::{GenerateRequest, TokenChunk};

/// A pluggable source of token generation.
///
/// Implementors must be `Send + Sync` so a single backend instance can be
/// shared across the server's worker tasks behind an `Arc`.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Start generating for `req` and return a stream of token chunks.
    ///
    /// The outer `Result` reports failures that happen *before* streaming
    /// begins (unknown model, backend down). Once streaming starts, per-chunk
    /// failures are reported as `Err` items inside the stream so a partial
    /// result is still delivered up to the point of failure.
    ///
    /// Dropping the returned stream must stop generation promptly (see
    /// [`MockBackend`](crate::MockBackend) for the reference implementation of
    /// that contract).
    async fn generate_stream(
        &self,
        req: GenerateRequest,
    ) -> std::result::Result<BoxStream<'static, std::result::Result<TokenChunk, Error>>, Error>;

    /// List the model names this backend can serve.
    async fn list_models(&self) -> Result<Vec<String>>;
}
