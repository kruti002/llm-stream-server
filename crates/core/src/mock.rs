//! A deterministic, model-free backend used by tests and CI.
//!
//! [`MockBackend`] emits a fixed sentence one token at a time, with a
//! configurable per-token delay. It exists so the rest of the system can be
//! tested without a GPU, a network, or a downloaded model — the output is
//! fully predictable, so assertions on ordering, indices, the terminal `done`
//! flag, and timing are stable.
//!
//! # How cancellation works
//!
//! This mock deliberately uses the *same* pattern a real streaming backend
//! would: a background task produces tokens and pushes them through a
//! [`tokio::sync::mpsc`] channel, while the consumer reads the receiver as a
//! stream. Cancellation is driven by a [`CancellationToken`]:
//!
//! * The returned stream owns a [`DropGuard`]. When the consumer drops the
//!   stream (in the server, because the client disconnected), the guard cancels
//!   the token.
//! * The producer task `select!`s between sleeping for the next token and the
//!   token being cancelled, so it stops at the next await point and exits
//!   instead of producing the rest of the sentence.
//!
//! The `CancellationToken` is the reusable primitive the server crate will hook
//! into client-disconnect handling later; keeping it here means that behavior
//! is already exercised by the core tests.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::types::{GenerateRequest, TokenChunk};

/// The fixed sentence the mock streams back, split on whitespace into tokens.
/// Kept as a constant so tests can import the expected token count/order.
pub const MOCK_SENTENCE: &str = "The quick brown fox jumps over the lazy dog";

/// A deterministic backend that streams [`MOCK_SENTENCE`] token by token.
#[derive(Debug, Clone)]
pub struct MockBackend {
    /// Delay inserted before each emitted token. Configurable so tests can run
    /// fast (or assert that a larger delay is actually respected).
    per_token_delay: Duration,
}

impl MockBackend {
    /// Create a mock backend that waits `per_token_delay` between tokens.
    pub fn new(per_token_delay: Duration) -> Self {
        Self { per_token_delay }
    }

    /// Split the fixed sentence into its individual tokens.
    fn tokens() -> Vec<String> {
        MOCK_SENTENCE
            .split_whitespace()
            .map(str::to_owned)
            .collect()
    }
}

impl Default for MockBackend {
    /// A zero-delay mock — handy for tests that only care about ordering.
    fn default() -> Self {
        Self::new(Duration::ZERO)
    }
}

#[async_trait]
impl Backend for MockBackend {
    async fn generate_stream(
        &self,
        _req: GenerateRequest,
    ) -> std::result::Result<BoxStream<'static, std::result::Result<TokenChunk, Error>>, Error>
    {
        let tokens = Self::tokens();
        let delay = self.per_token_delay;

        // Bounded channel: a capacity of 1 gives natural backpressure, matching
        // what a real backend wants (don't race ahead of a slow consumer).
        let (tx, rx) = mpsc::channel::<std::result::Result<TokenChunk, Error>>(1);

        // One token owned by the producer, one kept by the stream as a drop
        // guard. Dropping the stream => guard dropped => token cancelled =>
        // producer stops at its next await.
        let cancel = CancellationToken::new();
        let producer_cancel = cancel.clone();

        tokio::spawn(async move {
            let last = tokens.len().saturating_sub(1);
            for (index, text) in tokens.into_iter().enumerate() {
                // Respect the configured delay, but bail out immediately if the
                // consumer has gone away while we were waiting.
                tokio::select! {
                    biased;
                    _ = producer_cancel.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }

                let chunk = TokenChunk {
                    text,
                    index: index as u64,
                    done: index == last,
                };

                // If the receiver was dropped the send fails; stop producing.
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        });

        // `drop_guard()` ties token cancellation to the lifetime of the stream.
        // We thread the guard through the stream's state so it lives exactly as
        // long as the stream does.
        let guard = cancel.drop_guard();
        let stream = futures::stream::unfold((rx, guard), |(mut rx, guard)| async move {
            // `recv` yields `None` once the producer finishes and the channel
            // closes; returning `None` ends the stream and drops the guard.
            rx.recv().await.map(|item| (item, (rx, guard)))
        });

        Ok(stream.boxed())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock-small".to_owned(), "mock-large".to_owned()])
    }
}
