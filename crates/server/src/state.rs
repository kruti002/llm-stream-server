//! Shared application state handed to every axum handler.

use std::sync::Arc;

use llm_stream_core::Backend;
use tokio::sync::Semaphore;

/// State shared across all requests, stored in an `Arc` by the router.
///
/// The backend is a trait object so the server is agnostic to which concrete
/// implementation is running, and the semaphore is the single source of truth
/// for the concurrency cap.
#[derive(Clone)]
pub struct AppState {
    /// The configured backend. `Arc` so a single instance is shared across all
    /// worker tasks (the trait requires `Send + Sync`).
    pub backend: Arc<dyn Backend>,

    /// Default model name, used when a request omits one.
    pub default_model: String,

    /// Caps concurrent generations. A request acquires a permit before it
    /// starts streaming; when none are available it is rejected with `429`
    /// instead of piling up. Shared via `Arc` so permits are global to the
    /// process, not per-request.
    pub generation_slots: Arc<Semaphore>,
}

impl AppState {
    /// Build state with a concurrency cap of `max_concurrency` simultaneous
    /// generations.
    pub fn new(backend: Arc<dyn Backend>, default_model: String, max_concurrency: usize) -> Self {
        Self {
            backend,
            default_model,
            // Guard against a zero cap, which would make the server reject
            // everything; treat 0 as 1.
            generation_slots: Arc::new(Semaphore::new(max_concurrency.max(1))),
        }
    }
}
