//! HTTP-facing error handling.
//!
//! The core crate speaks in [`llm_stream_core::Error`] variants that are
//! transport-agnostic. This module is the single place that translates those
//! variants into HTTP status codes, so the mapping is defined once and every
//! handler stays thin.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use llm_stream_core::Error as CoreError;
use serde::Serialize;

/// A fatal startup error (bad config, backend failed to construct). Kept
/// separate from request-time errors because it happens before the server is
/// serving and just aborts boot.
#[derive(Debug)]
pub struct BootError(pub String);

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BootError {}

/// The JSON body returned for any error response.
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

/// A request-time error with a pre-computed HTTP status and message.
///
/// A small wrapper (rather than implementing [`IntoResponse`] for
/// [`CoreError`] directly) keeps us clear of the orphan rule and lets us
/// represent server-level conditions that are *not* core errors — notably
/// "at capacity" (`429`), which is a server policy decision, not a backend
/// failure.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl From<CoreError> for ApiError {
    fn from(e: CoreError) -> Self {
        // Map each core variant onto the HTTP status the spec calls for.
        let status = match &e {
            // Daemon down / backend not ready => the server is up but a
            // dependency isn't: 503 Service Unavailable.
            CoreError::BackendUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            // Caller asked for a model that doesn't exist => 404.
            CoreError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            // Upstream didn't respond in time => 504 Gateway Timeout.
            CoreError::Timeout => StatusCode::GATEWAY_TIMEOUT,
            // Generation was cancelled. 499 ("Client Closed Request") is a
            // non-standard nginx code that fits a client-disconnect exactly;
            // `from_u16` is used because axum has no named constant for it.
            CoreError::Cancelled => {
                StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
            }
        };
        ApiError {
            status,
            message: e.to_string(),
        }
    }
}

impl ApiError {
    /// The server is at its configured concurrency cap. Modeled as a dedicated
    /// `429` rather than overloading a backend error variant, because it is a
    /// backpressure decision made by the server, not a failure of the backend.
    pub fn too_many_requests() -> Self {
        ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "server at maximum concurrency; retry later".to_owned(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: self.message,
        };
        (self.status, Json(body)).into_response()
    }
}
