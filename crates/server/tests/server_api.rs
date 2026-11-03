//! Integration tests: start the real server on an ephemeral port with a
//! [`MockBackend`] and drive it over HTTP.
//!
//! # What passes this milestone vs. what is the user's target
//!
//! These PASS now (implemented this milestone):
//!   * `health_endpoint_reports_ok`
//!   * `models_endpoint_lists_backend_models`
//!   * `generate_endpoint_returns_full_completion`
//!   * `semaphore_returns_429_when_concurrency_exceeded`
//!
//! These FAIL until the user writes the SSE handler + disconnect cancellation
//! (they assert the real SSE contract against the `501` stub, so they fail at
//! RUNTIME with a guiding assertion, not at compile time):
//!   * `sse_stream_delivers_tokens_then_done_event`  ← write the handler
//!   * `client_disconnect_cancels_generation`        ← wire cancellation
//!
//! The disconnect test is deliberately *tight*: it does not merely time the
//! response, it asserts via the mock's observers that the producer stopped
//! early (fewer than all tokens emitted) AND that the cancellation path fired
//! (the producer task exited promptly). See its body for the reasoning.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use llm_stream_core::{MockBackend, MOCK_SENTENCE};
use llm_stream_server::{app, AppState};
use tokio::net::TcpListener;

/// Per-token delay for the mock in tests. Nonzero so streaming/disconnect
/// timing is observable, but small enough to keep the passing tests fast.
const TOKEN_DELAY: Duration = Duration::from_millis(60);

/// A running test server plus a handle to the mock's observers.
struct TestServer {
    addr: SocketAddr,
    /// How many tokens the mock's producer has emitted so far.
    emitted: Arc<AtomicUsize>,
    /// Set once when the mock's producer task exits (for any reason).
    exited: Arc<AtomicBool>,
    /// Dropping this aborts the server task.
    _handle: tokio::task::JoinHandle<()>,
}

/// Spawn the server on an ephemeral port using a MockBackend wired with
/// observers, so tests can both talk HTTP and inspect producer liveness.
async fn spawn_server(max_concurrency: usize) -> TestServer {
    let emitted = Arc::new(AtomicUsize::new(0));
    let exited = Arc::new(AtomicBool::new(false));
    let backend =
        MockBackend::with_observers(TOKEN_DELAY, Arc::clone(&emitted), Arc::clone(&exited));

    let state = AppState::new(Arc::new(backend), "mock-small".to_owned(), max_concurrency);
    let router = app(state);

    // Port 0 => OS picks a free ephemeral port.
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let addr = listener.local_addr().expect("local addr");

    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });

    TestServer {
        addr,
        emitted,
        exited,
        _handle: handle,
    }
}

fn base(server: &TestServer) -> String {
    format!("http://{}", server.addr)
}

fn generate_body() -> serde_json::Value {
    serde_json::json!({
        "model": "mock-small",
        "prompt": "hello",
    })
}

// --------------------------------------------------------------------------
// PASSING this milestone
// --------------------------------------------------------------------------

#[tokio::test]
async fn health_endpoint_reports_ok() {
    let server = spawn_server(4).await;
    let resp = reqwest::get(format!("{}/health", base(&server)))
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_lists_backend_models() {
    let server = spawn_server(4).await;
    let resp = reqwest::get(format!("{}/v1/models", base(&server)))
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    let models = body["models"].as_array().expect("models array");
    assert!(
        models.iter().any(|m| m == "mock-small"),
        "mock backend should advertise mock-small, got {models:?}"
    );
}

#[tokio::test]
async fn generate_endpoint_returns_full_completion() {
    let server = spawn_server(4).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/generate", base(&server)))
        .json(&generate_body())
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");

    // The mock concatenates its fixed sentence token by token.
    let expected: String = MOCK_SENTENCE
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(body["text"].as_str().expect("text"), expected);
    assert_eq!(
        body["chunks"].as_u64().expect("chunks"),
        MOCK_SENTENCE.split_whitespace().count() as u64
    );
}

#[tokio::test]
async fn semaphore_returns_429_when_concurrency_exceeded() {
    // Cap of 1: the first generation holds the only permit for the duration of
    // the (delayed) stream, so a concurrent second request must be rejected
    // with 429 rather than queued.
    let server = spawn_server(1).await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/generate", base(&server));

    // Fire the first request and let it start holding the permit.
    let first = tokio::spawn({
        let client = client.clone();
        let url = url.clone();
        async move {
            client
                .post(url)
                .json(&generate_body())
                .send()
                .await
                .expect("first request")
        }
    });

    // Give the first request time to acquire the permit and begin streaming.
    tokio::time::sleep(TOKEN_DELAY).await;

    let second = client
        .post(&url)
        .json(&generate_body())
        .send()
        .await
        .expect("second request");
    assert_eq!(
        second.status(),
        429,
        "second concurrent generation must be rejected with 429"
    );

    let first = first.await.expect("join first");
    assert_eq!(first.status(), 200, "first generation should complete fine");
}

// --------------------------------------------------------------------------
// FAILING until the user implements the SSE handler + disconnect cancellation
// --------------------------------------------------------------------------

/// TARGET FOR THE USER: implement `POST /v1/generate/stream`.
///
/// Asserts the SSE contract: one event per token carrying the serialized
/// `TokenChunk` in index order, terminated by a `done` event. Fails today
/// because the handler is a `501` stub.
#[tokio::test]
async fn sse_stream_delivers_tokens_then_done_event() {
    let server = spawn_server(4).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/v1/generate/stream", base(&server)))
        .json(&generate_body())
        .send()
        .await
        .expect("request");

    // The stub returns 501; a correct implementation returns a 200 SSE stream.
    assert_eq!(
        resp.status(),
        200,
        "POST /v1/generate/stream must return a 200 SSE stream — implement \
         routes::generate_stream (currently a stub returning {})",
        resp.status()
    );

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(
        content_type.starts_with("text/event-stream"),
        "SSE responses must use the text/event-stream content type, got {content_type:?}"
    );

    // Read the raw SSE body and sanity-check framing: one data line per token
    // carrying a TokenChunk, and a terminal `done` event.
    let body = resp.text().await.expect("sse body");
    let expected_tokens: Vec<&str> = MOCK_SENTENCE.split_whitespace().collect();

    let data_lines: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(str::trim))
        .collect();
    assert!(
        data_lines.len() >= expected_tokens.len(),
        "expected at least one data event per token ({}), got {} — body was:\n{body}",
        expected_tokens.len(),
        data_lines.len()
    );

    // Each token event must deserialize back into a TokenChunk in index order.
    for (i, token) in expected_tokens.iter().enumerate() {
        let chunk: serde_json::Value =
            serde_json::from_str(data_lines[i]).expect("each data event is a JSON TokenChunk");
        assert_eq!(
            chunk["index"].as_u64(),
            Some(i as u64),
            "events in index order"
        );
        assert_eq!(chunk["text"].as_str(), Some(*token), "token text matches");
    }

    assert!(
        body.contains("event:done") || body.contains("event: done"),
        "stream must end with a `done` event; body was:\n{body}"
    );
}

/// TARGET FOR THE USER: wire client-disconnect cancellation.
///
/// This is the TIGHTENED cancellation test. "Generation stopped" is defined
/// precisely, in two independent ways, so the user gets an unambiguous target:
///
///   1. **Fewer than all tokens emitted.** With a per-token delay, dropping the
///      client after the first token must leave the mock having emitted
///      strictly fewer than the full sentence. A handler that keeps generating
///      after disconnect (no cancellation) would run to completion and emit all
///      tokens.
///   2. **The cancellation path fired.** The mock sets `exited` the moment its
///      producer task returns. With cancellation wired, dropping the client
///      cancels the core `CancellationToken`, so the producer exits *promptly*
///      — well within one token delay. We check `exited` shortly after the
///      drop; a producer still parked in `sleep(delay)` (because only
///      channel-close, not the token, fired) would NOT have exited yet.
///
/// Both must hold. Fails today because the stub never streams, so no tokens are
/// emitted and nothing to cancel — the first assertion (status 200 / stream
/// started) trips first and points the user at the handler.
#[tokio::test]
async fn client_disconnect_cancels_generation() {
    let server = spawn_server(4).await;
    let client = reqwest::Client::new();

    // Start the SSE request but read only the first chunk, then drop it to
    // simulate a client disconnecting mid-stream.
    let resp = client
        .post(format!("{}/v1/generate/stream", base(&server)))
        .json(&generate_body())
        .send()
        .await
        .expect("request");

    assert_eq!(
        resp.status(),
        200,
        "SSE stream must start (200) before we can test disconnect — implement \
         routes::generate_stream first (currently returns {})",
        resp.status()
    );

    let total_tokens = MOCK_SENTENCE.split_whitespace().count();

    // Wait for a couple of tokens to flow, then drop the response to disconnect.
    tokio::time::sleep(TOKEN_DELAY * 2).await;
    let emitted_at_drop = server.emitted.load(Ordering::SeqCst);
    assert!(
        emitted_at_drop >= 1,
        "the stream should have produced at least one token before we disconnect"
    );
    drop(resp);

    // Give the cancellation path a short window — much less than one token
    // delay's worth beyond what's already elapsed — to run the producer's
    // early return.
    tokio::time::sleep(TOKEN_DELAY / 2).await;

    assert!(
        server.exited.load(Ordering::SeqCst),
        "producer did not exit promptly after client disconnect; the \
         CancellationToken is not being tied to the connection lifetime"
    );

    let emitted_total = server.emitted.load(Ordering::SeqCst);
    assert!(
        emitted_total < total_tokens,
        "producer emitted {emitted_total}/{total_tokens} tokens after the client \
         disconnected; generation was NOT cancelled"
    );
}
