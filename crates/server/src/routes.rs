//! HTTP handlers.
//!
//! Each handler is intentionally thin: it acquires resources (a concurrency
//! permit), calls the backend through the trait, and maps the result. The
//! core-error -> HTTP-status mapping lives in [`crate::error`].

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use llm_stream_core::{GenerateRequest, TokenChunk};
use serde::Serialize;
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

/// `POST /v1/generate/stream` — **USER-RESERVED STUB. DO NOT IMPLEMENT HERE.**
///
/// # The contract you must implement
///
/// This handler must stream the backend's tokens to the client as
/// Server-Sent Events (SSE):
///
/// * Acquire a concurrency permit exactly like [`generate`] (return `429` when
///   the semaphore is full), and keep it alive for the whole stream.
/// * Call `state.backend.generate_stream(req)` to get the
///   `BoxStream<Result<TokenChunk, Error>>`.
/// * Emit **one SSE event per token**, where each event's `data` is the
///   JSON-serialized [`TokenChunk`] (so the client sees `index`, `text`,
///   `done`). Preserve index order — the backend already yields in order, so do
///   not reorder or buffer out of sequence.
/// * After the terminal token (`done == true`) emit a final event named
///   `done` so the client knows the stream ended cleanly (as opposed to the
///   connection dropping). Use correct SSE framing (`event:`/`data:` fields,
///   blank-line terminators) — `axum::response::sse` gives you `Event` and
///   `Sse` helpers that handle the wire format.
///
/// # Client-disconnect cancellation (also yours to wire)
///
/// The core backends already cancel generation when the returned stream is
/// dropped (a `CancellationToken` drop-guard — see `llm_stream_core`'s mock and
/// Ollama backends). The integration point you must wire here:
///
/// * The SSE `Sse` response body is polled by axum only while the client is
///   connected. When the client disconnects, axum drops the response body,
///   which must in turn drop the backend token stream so its `CancellationToken`
///   fires and the upstream generation stops. In the simplest design that
///   happens automatically if the backend stream is *owned by* the SSE body and
///   nothing else holds it alive. If you introduce an intermediate task/channel
///   (e.g. to transform chunks into `Event`s), you MUST tie that task's
///   lifetime to the connection — hold a `CancellationToken` (or the drop-guard)
///   inside the SSE body so dropping the body cancels the task. Document which
///   approach you chose and why.
///
/// # Why this returns an error instead of panicking
///
/// So the rest of the server stays bootable and the other routes' tests can
/// run, this stub returns `501 Not Implemented` rather than calling `todo!()`
/// at the process level. A `todo!()` panic inside a handler would only abort
/// the one request on most runtimes, but returning an explicit, documented
/// error keeps the failure legible to the integration test (which asserts the
/// real SSE contract and therefore fails here until you implement it). See the
/// failing test `sse_stream_delivers_tokens_then_done_event` in
/// `tests/server_api.rs`.
// TODO(you): implement the SSE route — see the failing test
//            `sse_stream_delivers_tokens_then_done_event` and the
//            disconnect test `client_disconnect_cancels_generation`.
pub async fn generate_stream(
    State(_state): State<Arc<AppState>>,
    Json(_req): Json<GenerateRequest>,
) -> Response {
    // NOTE: the return type is `axum::response::Response` (not `Sse<...>`) on
    // purpose, so this stub type-checks while you have not yet produced an SSE
    // stream. Once you implement it, build and return an
    // `Sse::new(event_stream)` — it also implements `IntoResponse`, so you can
    // keep this signature or narrow it to `Sse<...>`.
    //
    // Returning 501 keeps the server runnable so the health/models/generate
    // and 429 tests pass; only the SSE/disconnect tests fail (at runtime, with
    // this guiding message) until you write the handler.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "POST /v1/generate/stream is not implemented yet — this is \
                      the user-reserved SSE handler. See routes::generate_stream \
                      docs and the failing integration tests."
        })),
    )
        .into_response()
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
