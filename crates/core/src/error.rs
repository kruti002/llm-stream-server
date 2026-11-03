//! Error type shared across every backend.
//!
//! We use a single crate-wide [`Error`] enum (built with `thiserror`) rather
//! than per-backend error types. A unified error keeps the [`Backend`] trait
//! object-safe and lets the server layer map one set of variants onto HTTP
//! status codes without caring which backend produced the failure.
//!
//! [`Backend`]: crate::Backend

use thiserror::Error;

/// Errors that any backend may surface while listing models or streaming
/// tokens.
///
/// The variants are intentionally coarse: they describe *classes* of failure
/// the server needs to react to (which HTTP status to return, whether to
/// retry), not every low-level cause. Backend-specific detail is carried in the
/// `String` payloads so nothing is lost for logging.
#[derive(Debug, Error)]
pub enum Error {
    /// The backend could not be reached or is not ready to serve requests
    /// (e.g. the Ollama daemon is down, or a model runtime failed to start).
    /// Maps naturally onto a 503 at the HTTP layer.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    /// The requested model name is not known to the backend. Distinct from
    /// `BackendUnavailable` because the backend is healthy — the caller simply
    /// asked for something that does not exist (a 404 at the HTTP layer).
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Generation did not make progress within the allotted time. Kept separate
    /// from cancellation because a timeout is an upstream failure, whereas
    /// cancellation is a deliberate, expected outcome.
    #[error("operation timed out")]
    Timeout,

    /// Generation was cancelled, typically because the consumer dropped the
    /// token stream (in the server, because the client disconnected). This is a
    /// normal control-flow signal, not a bug.
    #[error("operation cancelled")]
    Cancelled,
}

/// Crate-wide result alias so call sites can write `Result<T>` instead of
/// repeating the error type everywhere.
pub type Result<T> = std::result::Result<T, Error>;
