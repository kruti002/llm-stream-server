//! Behavioral tests for [`MockBackend`].
//!
//! These lock in the three properties the rest of the system relies on:
//! ordered tokens with a terminal `done`, a respected delay, and cancellation
//! on drop.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use llm_stream_core::{Backend, GenerateRequest, MockBackend, MOCK_SENTENCE};

/// Build a throwaway request; the mock ignores its contents.
fn req() -> GenerateRequest {
    GenerateRequest {
        model: "mock-small".to_owned(),
        prompt: "hello".to_owned(),
        max_tokens: None,
        temperature: None,
        stop: None,
    }
}

#[tokio::test]
async fn emits_tokens_in_order_with_terminal_done() {
    let backend = MockBackend::default();
    let stream = backend.generate_stream(req()).await.expect("stream starts");

    let chunks: Vec<_> = stream
        .map(|r| r.expect("no error chunks from the mock"))
        .collect()
        .await;

    let expected: Vec<&str> = MOCK_SENTENCE.split_whitespace().collect();
    assert_eq!(
        chunks.len(),
        expected.len(),
        "one chunk per word in the fixed sentence"
    );

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i as u64, "indices increment from zero");
        assert_eq!(chunk.text, expected[i], "text matches the fixed sentence");
        let is_last = i == expected.len() - 1;
        assert_eq!(chunk.done, is_last, "only the final chunk is done");
    }
}

#[tokio::test]
async fn respects_configured_delay() {
    // Four words at 20ms each => at least ~80ms. A loose lower bound avoids
    // flakiness while still proving the delay is applied per token.
    let per_token = Duration::from_millis(20);
    let backend = MockBackend::new(per_token);

    let start = Instant::now();
    let stream = backend.generate_stream(req()).await.expect("stream starts");
    let count = stream.count().await;
    let elapsed = start.elapsed();

    let expected_min = per_token * (count as u32);
    assert!(
        elapsed >= expected_min,
        "elapsed {elapsed:?} should be at least {expected_min:?} for {count} tokens"
    );
}

#[tokio::test]
async fn dropping_the_stream_stops_emission() {
    // This test observes the producer DIRECTLY, and specifically isolates the
    // `CancellationToken` path. A naive timing assertion ("we finished
    // quickly") would pass whether or not cancellation works, because it only
    // measures its own sleep. Worse, the mock has TWO ways to stop: the
    // cancellation token (the reusable one the server will wire to client
    // disconnects) and the channel closing when the receiver drops. A weak test
    // could pass on channel-close alone while the token is broken.
    //
    // To force the token to be the thing under test we use a LONG per-token
    // delay. The producer spends almost all its time parked in `sleep(delay)`.
    //   * If the token works: dropping the stream cancels it, the `select!`
    //     returns immediately, and the producer task exits in ~0ms.
    //   * If only channel-close worked: the producer is still asleep and will
    //     not even notice the dropped receiver until the delay elapses and it
    //     tries to send. So it would stay alive for most of `delay`.
    // We wait only a small fraction of `delay` after the drop and require the
    // producer to have already exited. That window is too short for the
    // channel-close path, so the assertion depends on the token.
    let per_token = Duration::from_millis(400);
    let emitted = Arc::new(AtomicUsize::new(0));
    let exited = Arc::new(AtomicBool::new(false));
    let backend = MockBackend::with_observers(per_token, Arc::clone(&emitted), Arc::clone(&exited));

    let total_tokens = MOCK_SENTENCE.split_whitespace().count();
    assert!(
        total_tokens > 2,
        "sentence must have several tokens so a surviving producer would keep emitting"
    );

    let mut stream = backend.generate_stream(req()).await.expect("stream starts");

    // Pull exactly one token, then drop the stream mid-sentence. At this point
    // the producer has looped back and is parked in `sleep(per_token)` waiting
    // to emit the second token.
    let first = stream.next().await.expect("first token").expect("ok chunk");
    assert_eq!(first.index, 0);
    assert!(!first.done, "we stopped before the end of the sentence");

    drop(stream);

    // Wait only a small fraction of one token delay. Long enough for a
    // cancelled producer to run its early-return and set the exit flag, far too
    // short for a producer stuck in `sleep(400ms)` to wake up.
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(
        exited.load(Ordering::SeqCst),
        "producer did not exit promptly after the stream was dropped; the \
         CancellationToken path is not cancelling the in-flight sleep"
    );
    // And it certainly must not have emitted the whole sentence.
    let count = emitted.load(Ordering::SeqCst);
    assert!(
        count < total_tokens,
        "producer emitted {count}/{total_tokens} tokens despite the drop; cancellation failed"
    );
}
