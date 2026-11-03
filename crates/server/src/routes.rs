//! HTTP handlers.
//!
//! Each handler is intentionally thin: it acquires resources (a concurrency
//! permit), calls the backend through the trait, and maps the result. The
//! core-error -> HTTP-status mapping lives in [`crate::error`].

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream;
use futures::StreamExt;
use llm_stream_core::{GenerateRequest, TokenChunk};
use serde::Serialize;
use std::convert::Infallible;
use tokio::sync::OwnedSemaphorePermit;

use crate::error::ApiError;
use crate::state::AppState;

/// `GET /health` — a trivial liveness probe.
///
/// Returns 200 with a small JSON body. It does not touch the backend, so it
/// stays green even when the model/daemon is down (that is what `/v1/models`
/// is for).
pub async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

/// `GET /v1/models` — list the model names the configured backend can serve.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.backend.list_models().await?;
    Ok(Json(ModelsResponse { models }))
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<String>,
}

/// Response body for the non-streaming `POST /v1/generate`.
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    /// The full completion text, concatenated from every token chunk.
    pub text: String,
    /// Number of chunks that made up the completion (useful for debugging).
    pub chunks: usize,
}

/// `POST /v1/generate` — NON-streaming generation.
///
/// Drains the backend's token stream, concatenates the text, and returns the
/// whole completion as one JSON response. Streaming (SSE) is the separate
/// `/v1/generate/stream` route.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, ApiError> {
    // Backpressure: take a concurrency permit or reject with 429. We hold the
    // permit for the whole generation by keeping `_permit` alive until the end
    // of the function.
    let _permit = acquire_slot(&state)?;

    apply_default_model(&mut req, &state);

    let mut stream = state.backend.generate_stream(req).await?;

    let mut text = String::new();
    let mut chunks = 0usize;
    while let Some(item) = stream.next().await {
        let chunk: TokenChunk = item?;
        text.push_str(&chunk.text);
        chunks += 1;
    }

    Ok(Json(GenerateResponse { text, chunks }))
}

/// `POST /v1/generate/stream` — stream tokens to the client as Server-Sent
/// Events (SSE).
///
/// # What it does
///
/// * Acquires a concurrency permit exactly like [`generate`] (returns `429`
///   when the semaphore is full) and keeps it alive for the whole stream.
/// * Calls the backend's `generate_stream` to get the token stream, then maps
///   each [`TokenChunk`] to an SSE event whose `data` is the JSON-serialized
///   chunk (so the client sees `index`, `text`, `done`), preserving index
///   order. A final event named `done` marks a clean end (as opposed to the
///   connection dropping).
///
/// # Client-disconnect cancellation
///
/// Cancellation is achieved by *ownership*, not by explicit disconnect
/// detection. axum polls the `Sse` body only while the client is connected; on
/// disconnect it drops the body. The backend token stream is owned by that body
/// (moved into `tokens`, then `event_stream`, then `Sse::new`), and nothing
/// else holds it alive — so dropping the body drops the token stream, which
/// fires the core `CancellationToken` drop-guard and stops upstream generation.
/// The concurrency permit is parked in the trailing `done` future for the same
/// reason: it is released when the stream ends *or* when it is dropped early.
/// This is why the handler deliberately does **not** `tokio::spawn` a pump task
/// — a spawned task would outlive the dropped body and keep generating.
pub async fn generate_stream(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<GenerateRequest>,
) -> Response {
    let permit = match acquire_slot(&state) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };

    apply_default_model(&mut req, &state);

    let token_stream = match state.backend.generate_stream(req).await {
        Ok(s) => s,
        Err(e) => return ApiError::from(e).into_response(),
    };

    let tokens = token_stream.map(|item| {
        let event = match item {
            Ok(chunk) => Event::default()
                .json_data(&chunk)
                .unwrap_or_else(|e| Event::default().event("error").data(e.to_string())),
            Err(e) => Event::default().event("error").data(e.to_string()),
        };
        Ok::<_, Infallible>(event)
    });
    let done = stream::once(async move {
        // Holding the permit until the final event keeps the concurrency slot
        // reserved for the whole stream; dropping the stream early (client
        // disconnect) drops this future and releases it.
        let _permit = permit;
        Ok::<_, Infallible>(Event::default().event("done").data("[DONE]"))
    });
    let event_stream = tokens.chain(done);
    Sse::new(event_stream).into_response()
}

/// Acquire a generation permit or map "none available" to a 429 error.
///
/// Uses `try_acquire_owned` (non-blocking): if the cap is hit we reject
/// immediately rather than queueing, which is the backpressure behavior the
/// spec asks for. The returned owned permit can be held across `.await` points
/// and moved into the SSE body, which `try_acquire` (borrowed) could not.
fn acquire_slot(state: &AppState) -> Result<OwnedSemaphorePermit, ApiError> {
    Arc::clone(&state.generation_slots)
        .try_acquire_owned()
        .map_err(|_| {
            // Not a core::Error variant — construct the 429 directly. We model
            // "at capacity" as a dedicated response rather than overloading an
            // existing variant.
            ApiError::too_many_requests()
        })
}

/// Fill in the server's default model when the request didn't specify one.
fn apply_default_model(req: &mut GenerateRequest, state: &AppState) {
    if req.model.trim().is_empty() {
        req.model = state.default_model.clone();
    }
}
