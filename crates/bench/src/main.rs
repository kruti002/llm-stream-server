//! # llm-stream-bench
//!
//! A command-line load + latency benchmark for the `llm-stream-server`
//! streaming API. It drives the server over HTTP (`reqwest` + rustls) and
//! reports latency/throughput statistics as a Markdown table suitable for a
//! README. It deliberately talks to the *server*, not the backends in-process,
//! so the numbers include the real HTTP/SSE path the server actually serves.
//!
//! ## What it measures (per concurrency level, over post-warmup requests)
//!
//! * **TTFT** — Time To First Token: wall time from sending the request to the
//!   first SSE `data:` token event. `stream` endpoint only; for `generate`
//!   (non-streaming) there is no "first token", so TTFT is reported as `N/A`
//!   and we measure total latency instead.
//! * **Per-stream tok/s** — for each request, `token_events / (t_last - t_first)`,
//!   reported as mean and p50 across requests.
//! * **Aggregate tok/s** — total tokens across the level / wall-clock duration
//!   of the whole sweep (captures the effect of concurrency).
//! * **Request latency** — end-to-end time per request, p50/p95.
//! * **Error rate** — fraction of requests that failed (non-200, connection
//!   error, or malformed SSE), per level.
//!
//! ## Honesty caveat (important for resume numbers)
//!
//! Against the **mock** backend the model is a fixed sentence with a fixed
//! per-token delay, so the numbers measure *server + harness overhead*, not
//! model speed. Against **Ollama** they measure real model throughput on the
//! user's hardware. The README spells this out; do not present mock numbers as
//! model performance.

mod stats;

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use futures::StreamExt;

/// Which server path to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lowercase")]
enum Endpoint {
    /// `POST /v1/generate/stream` (SSE). TTFT is meaningful here.
    Stream,
    /// `POST /v1/generate` (non-streaming JSON). TTFT is reported as N/A.
    Generate,
}

impl Endpoint {
    /// Path segment appended to `--url` for this endpoint.
    fn path(self) -> &'static str {
        match self {
            Endpoint::Stream => "/v1/generate/stream",
            Endpoint::Generate => "/v1/generate",
        }
    }
}

/// CLI surface. `clap` derive keeps the flags declared once as a struct.
#[derive(Debug, Parser)]
#[command(
    name = "llm-stream-bench",
    about = "Load + latency benchmark for llm-stream-server"
)]
struct Args {
    /// Base URL of the running server, e.g. `http://127.0.0.1:8080`.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,

    /// Which endpoint to hit: `stream` (SSE) or `generate` (non-streaming).
    #[arg(long, value_enum, default_value_t = Endpoint::Stream)]
    endpoint: Endpoint,

    /// Model name sent in each request body.
    #[arg(long, default_value = "mock-small")]
    model: String,

    /// Prompt text sent in each request body.
    #[arg(long, default_value = "Explain Rust ownership in one paragraph")]
    prompt: String,

    /// Comma-separated concurrency levels; one sweep is run per level.
    #[arg(long, value_delimiter = ',', default_value = "1,8,32")]
    concurrency: Vec<usize>,

    /// Total requests to complete per concurrency level (post-warmup).
    #[arg(long, default_value_t = 100)]
    requests: usize,

    /// Optional cap on tokens per request (forwarded as `max_tokens`).
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Warmup requests to issue and discard before measuring, per level.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Optional path to write the Markdown table. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

/// Per-request measurement collected by a worker.
#[derive(Debug, Clone)]
struct RequestSample {
    /// End-to-end wall time for the whole request, in milliseconds.
    latency_ms: f64,
    /// Time from send to first token event, in milliseconds. `None` for the
    /// non-streaming endpoint (or if the request failed before any token).
    ttft_ms: Option<f64>,
    /// Number of token events observed in this request.
    tokens: u64,
    /// Per-stream throughput (tokens / (last-first) seconds). `None` when it is
    /// undefined (fewer than 2 tokens, or non-streaming).
    tok_per_s: Option<f64>,
    /// `true` if the request failed (non-200, transport error, malformed SSE).
    failed: bool,
}

/// Summary statistics for one concurrency level, ready to render.
#[derive(Debug)]
struct LevelReport {
    concurrency: usize,
    endpoint: Endpoint,
    requests: usize,
    errors: usize,
    ttft_p50: Option<f64>,
    ttft_p95: Option<f64>,
    ttft_p99: Option<f64>,
    per_stream_mean: Option<f64>,
    per_stream_p50: Option<f64>,
    aggregate_tok_s: f64,
    latency_p50: Option<f64>,
    latency_p95: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // One shared client: reqwest pools connections, so reusing it across
    // requests is both faster and closer to how a real client behaves. rustls
    // is pinned at the workspace level (no OpenSSL).
    let client = reqwest::Client::builder()
        // No overall timeout: long streams are legitimate. A connect timeout
        // still guards against a server that never accepts.
        .connect_timeout(Duration::from_secs(5))
        .build()?;

    let target = format!("{}{}", args.url.trim_end_matches('/'), args.endpoint.path());

    wait_for_health(&client, &args.url).await?;

    let mut reports = Vec::new();
    for &level in &args.concurrency {
        if level == 0 {
            return Err("concurrency levels must be >= 1".into());
        }
        eprintln!(
            "running level: concurrency={level} requests={} warmup={} endpoint={:?}",
            args.requests, args.warmup, args.endpoint
        );
        let report = run_level(&client, &target, &args, level).await;
        reports.push(report);
    }

    let table = render_markdown(&args, &reports);

    // Always print a short human-readable summary to stdout; write the full
    // table either to stdout or the requested file.
    print_summary(&reports);
    match &args.output {
        Some(path) => {
            let mut f = std::fs::File::create(path)?;
            f.write_all(table.as_bytes())?;
            eprintln!("wrote markdown table to {}", path.display());
        }
        None => {
            println!("\n{table}");
        }
    }

    Ok(())
}

/// Poll `GET /health` until it returns 200 or we give up.
///
/// The server is started separately (as a background process); this makes the
/// bench robust to a small startup race without a fixed `sleep`.
async fn wait_for_health(client: &reqwest::Client, base_url: &str) -> Result<(), String> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {
                if Instant::now() >= deadline {
                    return Err(format!("server at {url} did not become healthy within 30s"));
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Run a full sweep for one concurrency level and summarize it.
///
/// ## Backpressure / worker-pool design
///
/// We keep exactly `level` requests in flight at once using a
/// `futures::stream::iter(..).buffer_unordered(level)` pipeline. `buffer_unordered`
/// polls up to `level` futures concurrently and only pulls the next request
/// from the iterator when one of the in-flight futures completes — that is the
/// backpressure: the generator never races ahead of the configured concurrency.
/// It is a dozen lines and needs no explicit semaphore or task spawning, which
/// keeps the data-flow easy to explain (and easy to get right) in an interview.
///
/// Warmup requests are run first, as their own buffered pass, and discarded so
/// JIT-like effects (connection setup, first-request allocation) do not skew
/// the measured numbers.
async fn run_level(
    client: &reqwest::Client,
    target: &str,
    args: &Args,
    level: usize,
) -> LevelReport {
    let body = build_body(args);

    // --- warmup (results discarded) ---
    if args.warmup > 0 {
        let warm = futures::stream::iter(0..args.warmup)
            .map(|_| one_request(client, target, &body, args.endpoint))
            .buffer_unordered(level);
        warm.for_each(|_| async {}).await;
    }

    // --- measured pass ---
    // A shared counter of tokens lets us compute aggregate throughput without
    // re-summing the samples; it is incremented inside each request future.
    let total_tokens = Arc::new(AtomicU64::new(0));

    let body_ref = &body;
    let sweep_start = Instant::now();
    let samples: Vec<RequestSample> = futures::stream::iter(0..args.requests)
        .map(|_| {
            let counter = Arc::clone(&total_tokens);
            async move {
                let sample = one_request(client, target, body_ref, args.endpoint).await;
                counter.fetch_add(sample.tokens, Ordering::Relaxed);
                sample
            }
        })
        .buffer_unordered(level)
        .collect()
        .await;
    let sweep_elapsed = sweep_start.elapsed();

    summarize(args.endpoint, level, args.requests, samples, sweep_elapsed)
}

/// Build the JSON request body once per level (identical for every request).
fn build_body(args: &Args) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": args.model,
        "prompt": args.prompt,
    });
    if let Some(mt) = args.max_tokens {
        body["max_tokens"] = serde_json::json!(mt);
    }
    body
}

/// Issue a single request and measure it.
///
/// For the `stream` endpoint we parse the SSE body incrementally so we can
/// timestamp the *first* `data:` token event precisely; for `generate` we just
/// read the JSON body and record total latency (TTFT is N/A).
async fn one_request(
    client: &reqwest::Client,
    target: &str,
    body: &serde_json::Value,
    endpoint: Endpoint,
) -> RequestSample {
    match endpoint {
        Endpoint::Stream => stream_request(client, target, body).await,
        Endpoint::Generate => generate_request(client, target, body).await,
    }
}

/// Drive the SSE endpoint and measure TTFT + per-stream throughput.
async fn stream_request(
    client: &reqwest::Client,
    target: &str,
    body: &serde_json::Value,
) -> RequestSample {
    let start = Instant::now();
    let resp = match client.post(target).json(body).send().await {
        Ok(r) => r,
        Err(_) => return failed_sample(start.elapsed()),
    };
    if !resp.status().is_success() {
        return failed_sample(start.elapsed());
    }

    // Read the body as a byte stream and parse SSE events incrementally. SSE
    // frames are separated by a blank line; each event is a set of lines where
    // `event:` names the type (default `message`) and `data:` carries payload.
    // Our server emits one token event per `data:` (JSON TokenChunk) and a
    // terminal `event: done` frame with data `[DONE]`.
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    let mut first_token_at: Option<Instant> = None;
    let mut last_token_at: Option<Instant> = None;
    let mut tokens: u64 = 0;
    let mut saw_done = false;
    let mut malformed = false;

    'outer: while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(_) => {
                malformed = true;
                break;
            }
        };
        // SSE is UTF-8 text; accumulate and split on blank lines (frame
        // boundaries). We keep any trailing partial frame in `buf`.
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(idx) = buf.find("\n\n") {
            let frame: String = buf.drain(..idx + 2).collect();
            let now = Instant::now();
            match classify_frame(&frame) {
                FrameKind::Token => {
                    if first_token_at.is_none() {
                        first_token_at = Some(now);
                    }
                    last_token_at = Some(now);
                    tokens += 1;
                }
                FrameKind::Done => {
                    saw_done = true;
                    break 'outer;
                }
                FrameKind::Error => {
                    malformed = true;
                    break 'outer;
                }
                FrameKind::Other => {}
            }
        }
    }

    let latency = start.elapsed();

    // A clean stream must terminate with the `done` event and carry tokens.
    // Anything else (no done, parse error, zero tokens) counts as a failure so
    // the error rate reflects malformed SSE honestly.
    let failed = malformed || !saw_done || tokens == 0;

    let ttft_ms = first_token_at.map(|t| duration_ms(t.saturating_duration_since(start)));

    // Per-stream throughput: tokens over the first-to-last-token window. Needs
    // at least two timestamps for a meaningful interval; otherwise undefined.
    let tok_per_s = match (first_token_at, last_token_at, tokens) {
        (Some(f), Some(l), n) if n >= 2 && l > f => {
            Some(n as f64 / duration_secs(l.saturating_duration_since(f)))
        }
        _ => None,
    };

    RequestSample {
        latency_ms: duration_ms(latency),
        ttft_ms: if failed { None } else { ttft_ms },
        tokens: if failed { 0 } else { tokens },
        tok_per_s: if failed { None } else { tok_per_s },
        failed,
    }
}

/// Drive the non-streaming endpoint. TTFT is N/A; we record total latency and
/// the chunk count the server reports so aggregate throughput still has tokens.
async fn generate_request(
    client: &reqwest::Client,
    target: &str,
    body: &serde_json::Value,
) -> RequestSample {
    let start = Instant::now();
    let resp = match client.post(target).json(body).send().await {
        Ok(r) => r,
        Err(_) => return failed_sample(start.elapsed()),
    };
    if !resp.status().is_success() {
        return failed_sample(start.elapsed());
    }
    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return failed_sample(start.elapsed()),
    };
    let latency = start.elapsed();

    // The server returns `{ text, chunks }`; use `chunks` as the token count.
    let tokens = json
        .get("chunks")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    RequestSample {
        latency_ms: duration_ms(latency),
        ttft_ms: None,
        tokens,
        tok_per_s: None,
        failed: false,
    }
}

/// A uniform "this request failed" sample.
fn failed_sample(elapsed: Duration) -> RequestSample {
    RequestSample {
        latency_ms: duration_ms(elapsed),
        ttft_ms: None,
        tokens: 0,
        tok_per_s: None,
        failed: true,
    }
}

/// The kinds of SSE frame we care about.
enum FrameKind {
    /// A default `data:` event carrying a token chunk.
    Token,
    /// The terminal `event: done` frame.
    Done,
    /// An `event: error` frame emitted by the server on backend failure.
    Error,
    /// Comments/empty frames we ignore.
    Other,
}

/// Classify one raw SSE frame (its lines) by inspecting its `event:` field.
///
/// Per the SSE spec an event with no `event:` field defaults to `message`; our
/// server emits token chunks as such default events with a `data:` payload.
fn classify_frame(frame: &str) -> FrameKind {
    let mut event_name: Option<&str> = None;
    let mut has_data = false;
    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim());
        } else if line.starts_with("data:") {
            has_data = true;
        }
    }
    match event_name {
        Some("done") => FrameKind::Done,
        Some("error") => FrameKind::Error,
        // Default event ("message") with a data payload == a token chunk.
        _ if has_data => FrameKind::Token,
        _ => FrameKind::Other,
    }
}

/// Turn a slice of samples into a `LevelReport`.
fn summarize(
    endpoint: Endpoint,
    concurrency: usize,
    requests: usize,
    samples: Vec<RequestSample>,
    sweep_elapsed: Duration,
) -> LevelReport {
    let errors = samples.iter().filter(|s| s.failed).count();

    // Only successful requests contribute to the latency/TTFT/throughput
    // distributions; failures are captured by the error rate instead.
    let ok: Vec<&RequestSample> = samples.iter().filter(|s| !s.failed).collect();

    let latencies: Vec<f64> = ok.iter().map(|s| s.latency_ms).collect();
    let ttfts: Vec<f64> = ok.iter().filter_map(|s| s.ttft_ms).collect();
    let per_stream: Vec<f64> = ok.iter().filter_map(|s| s.tok_per_s).collect();

    let total_tokens: u64 = ok.iter().map(|s| s.tokens).sum();
    let aggregate_tok_s = if sweep_elapsed.as_secs_f64() > 0.0 {
        total_tokens as f64 / sweep_elapsed.as_secs_f64()
    } else {
        0.0
    };

    LevelReport {
        concurrency,
        endpoint,
        requests,
        errors,
        ttft_p50: stats::percentile(&ttfts, 50.0),
        ttft_p95: stats::percentile(&ttfts, 95.0),
        ttft_p99: stats::percentile(&ttfts, 99.0),
        per_stream_mean: stats::mean(&per_stream),
        per_stream_p50: stats::percentile(&per_stream, 50.0),
        aggregate_tok_s,
        latency_p50: stats::percentile(&latencies, 50.0),
        latency_p95: stats::percentile(&latencies, 95.0),
    }
}

/// Render the full set of level reports as a Markdown table.
fn render_markdown(args: &Args, reports: &[LevelReport]) -> String {
    let mut out = String::new();
    let streaming = matches!(args.endpoint, Endpoint::Stream);

    out.push_str(&format!(
        "Benchmark: `{}` endpoint, model `{}`, {} requests/level ({} warmup discarded).\n\n",
        args.endpoint_label(),
        args.model,
        args.requests,
        args.warmup,
    ));

    out.push_str("| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Per-stream tok/s (mean) | Per-stream tok/s (p50) | Aggregate tok/s | Latency p50 (ms) | Latency p95 (ms) | Error rate |\n");
    out.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    for r in reports {
        let ttft = |v: Option<f64>| {
            if streaming {
                fmt_opt_ms(v)
            } else {
                "N/A".to_string()
            }
        };
        let err_rate = if r.requests > 0 {
            r.errors as f64 / r.requests as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {} | {} | {:.2}% |\n",
            r.concurrency,
            ttft(r.ttft_p50),
            ttft(r.ttft_p95),
            ttft(r.ttft_p99),
            fmt_opt_1(r.per_stream_mean),
            fmt_opt_1(r.per_stream_p50),
            r.aggregate_tok_s,
            fmt_opt_ms(r.latency_p50),
            fmt_opt_ms(r.latency_p95),
            err_rate * 100.0,
        ));
    }

    if !streaming {
        out.push_str(
            "\n_TTFT is N/A for the non-streaming `generate` endpoint; the latency columns are the end-to-end total instead._\n",
        );
    }
    out
}

/// Short human-readable summary printed to stdout regardless of `--output`.
fn print_summary(reports: &[LevelReport]) {
    eprintln!();
    for r in reports {
        let err_rate = if r.requests > 0 {
            r.errors as f64 / r.requests as f64 * 100.0
        } else {
            0.0
        };
        let ttft = match (r.endpoint, r.ttft_p50) {
            (Endpoint::Stream, Some(v)) => format!("TTFT p50 {v:.1}ms"),
            (Endpoint::Stream, None) => "TTFT p50 n/a".to_string(),
            (Endpoint::Generate, _) => "TTFT n/a".to_string(),
        };
        eprintln!(
            "  c={:>3}: {ttft}, aggregate {:.1} tok/s, latency p50 {}, errors {:.2}%",
            r.concurrency,
            r.aggregate_tok_s,
            fmt_opt_ms(r.latency_p50),
            err_rate,
        );
    }
}

/// Format an optional millisecond value with one decimal, or `N/A`.
fn fmt_opt_ms(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.1}"),
        None => "N/A".to_string(),
    }
}

/// Format an optional value with one decimal, or `N/A`.
fn fmt_opt_1(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.1}"),
        None => "N/A".to_string(),
    }
}

/// Convert a `Duration` to fractional milliseconds.
fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Convert a `Duration` to fractional seconds.
fn duration_secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

impl Args {
    /// A human label for the endpoint, matching the CLI value.
    fn endpoint_label(&self) -> &'static str {
        match self.endpoint {
            Endpoint::Stream => "stream",
            Endpoint::Generate => "generate",
        }
    }
}
