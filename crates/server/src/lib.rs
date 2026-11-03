//! # llm-stream-server
//!
//! An Axum HTTP server that exposes any [`llm_stream_core::Backend`] over a
//! small REST + SSE API. The crate is split into a library (router, handlers,
//! config, state) and a thin binary so that integration tests can build the
//! same [`Router`](axum::Router) and drive it in-process.
//!
//! ## What this milestone implements
//!
//! * `GET /health`, `GET /v1/models`, `POST /v1/generate` (non-streaming).
//! * A concurrency cap via a [`tokio::sync::Semaphore`] returning `429` when
//!   full (backpressure), configurable at startup.
//! * `tracing` + `tower-http`'s `TraceLayer` with a request-id layer.
//! * Graceful shutdown on Ctrl+C.
//! * Config via `clap` with environment-variable fallback.
//!
//! ## What is deliberately left as stubs for the user
//!
//! The interview-critical pieces are left unimplemented on purpose (see
//! [`routes::generate_stream`]):
//!
//! * the `POST /v1/generate/stream` SSE handler body, and
//! * tying the core `CancellationToken` to the client connection lifetime so a
//!   disconnect cancels the upstream generation.
//!
//! A failing integration test (`tests/server_api.rs`) specifies exactly what
//! those must satisfy.

pub mod config;
pub mod error;
pub mod routes;
pub mod state;

pub use config::{BackendKind, Config};
pub use state::AppState;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;

/// Build the application router with all middleware wired in.
///
/// Kept separate from the server bootstrap so tests can mount the exact same
/// routes/middleware onto an ephemeral port.
pub fn app(state: AppState) -> Router {
    // Middleware order matters: we stamp a request id FIRST so the TraceLayer
    // (and everything downstream) can log it, and propagate it back onto the
    // response for client-side correlation.
    //
    // `x-request-id` is the conventional header name.
    let middleware = ServiceBuilder::new()
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id());

    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/generate", post(routes::generate))
        // The SSE route is wired in, but its handler is a user-reserved stub.
        .route("/v1/generate/stream", post(routes::generate_stream))
        .layer(middleware)
        .with_state(Arc::new(state))
}
