//! A real backend that talks to a local [Ollama](https://ollama.com) daemon.
//!
//! Ollama exposes an HTTP API. For generation we POST to `/api/generate` with
//! `{"model", "prompt", "stream": true, ...}` and the daemon replies with a
//! *newline-delimited JSON* (NDJSON) stream: one JSON object per line, each
//! carrying a `response` text fragment and a `done` flag that is `true` on the
//! final line. For the model list we GET `/api/tags`.
//!
//! # Why this shape
//!
//! * **Streaming over `reqwest`'s byte stream.** We ask reqwest for the body as
//!   a `Stream` of byte chunks rather than buffering the whole response. HTTP
//!   chunk boundaries do *not* line up with NDJSON line boundaries, so we keep
//!   a small reassembly buffer ([`NdjsonParser`]) that splits on `\n` and holds
//!   any trailing partial line until the next chunk completes it. That parser
//!   is pure and unit-tested with fixed bytes — no network required.
//!
//! * **Cancellation matches [`MockBackend`](crate::MockBackend).** The returned
//!   stream owns a [`CancellationToken`] drop-guard. Dropping the stream cancels
//!   the token; the producer task `select!`s the token against the next body
//!   chunk, so it stops reading from the HTTP body promptly when the consumer
//!   (the server, when a client disconnects) goes away. Inference here runs over
//!   HTTP, so there is no blocking work and no `spawn_blocking` — unlike the
//!   future llama.cpp backend.
//!
//! * **Retries only on initial connect.** Transient failures establishing the
//!   connection (daemon still starting) are retried with exponential backoff.
//!   Once the stream has started we do *not* retry: replaying a partially
//!   consumed generation would duplicate tokens, so a mid-stream failure is
//!   surfaced as an error item instead.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::types::{GenerateRequest, TokenChunk};

/// Default Ollama base URL. The daemon listens on `11434` by convention.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Backend that streams tokens from an Ollama daemon over HTTP.
#[derive(Debug, Clone)]
pub struct OllamaBackend {
    client: reqwest::Client,
    /// Base URL without a trailing slash, e.g. `http://localhost:11434`.
    base_url: String,
    /// How many times to retry *establishing the initial connection* before
    /// giving up (not counting the first attempt).
    connect_retries: u32,
}

impl OllamaBackend {
    /// Build a backend pointing at `base_url` with a per-request `timeout`.
    ///
    /// The timeout is applied as reqwest's overall request timeout; for a
    /// streaming response that bounds the time to *first* bytes and the gap
    /// between chunks via the connect timeout, which is what we want for
    /// detecting a dead daemon without killing a long but healthy stream.
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        // `connect_timeout` is deliberately separate from the body timeout:
        // we want to fail fast if the daemon is down, but not abort a slow
        // (yet alive) generation.
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .build()
            .map_err(|e| Error::BackendUnavailable(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url,
            connect_retries: 3,
        })
    }

    /// Convenience constructor using [`DEFAULT_OLLAMA_URL`] and a 30s timeout.
    pub fn with_defaults() -> Result<Self> {
        Self::new(DEFAULT_OLLAMA_URL, Duration::from_secs(30))
    }

    /// Override the number of initial-connect retries (default 3).
    pub fn with_connect_retries(mut self, retries: u32) -> Self {
        self.connect_retries = retries;
        self
    }

    /// POST the generate request, retrying the *initial connect* with
    /// exponential backoff. Returns the streaming response once headers arrive.
    async fn send_generate(&self, req: &GenerateRequest) -> Result<reqwest::Response> {
        let url = format!("{}/api/generate", self.base_url);
        let body = OllamaGenerateRequest::from(req);

        let mut attempt = 0;
        loop {
            match self.client.post(&url).json(&body).send().await {
                Ok(resp) => return classify_response(resp).await,
                Err(e) => {
                    // Only the *connect* phase is retried. A timeout or a
                    // connection error before any bytes is a candidate for a
                    // transient daemon-startup hiccup.
                    if attempt >= self.connect_retries || !is_retryable_connect(&e) {
                        return Err(map_reqwest_error(e));
                    }
                    // Exponential backoff: 50ms, 100ms, 200ms, ...
                    let backoff = Duration::from_millis(50 * (1u64 << attempt));
                    tokio::time::sleep(backoff).await;
                    attempt += 1;
                }
            }
        }
    }
}

#[async_trait]
impl Backend for OllamaBackend {
    async fn generate_stream(
        &self,
        req: GenerateRequest,
    ) -> std::result::Result<BoxStream<'static, std::result::Result<TokenChunk, Error>>, Error>
    {
        let resp = self.send_generate(&req).await?;

        // reqwest's byte stream. Each item is a chunk of body bytes whose
        // boundaries are unrelated to NDJSON line boundaries.
        let byte_stream = resp.bytes_stream();

        // Cancellation: the returned stream owns a drop-guard. On drop the
        // token is cancelled, which makes the combinator below stop awaiting
        // the next body chunk and end — closing the HTTP connection and halting
        // the upstream generation. Same primitive as MockBackend.
        let cancel = CancellationToken::new();
        // The producer keeps a clone to observe cancellation; the stream owns
        // the drop-guard so dropping the stream cancels the token.
        let guard = cancel.clone().drop_guard();

        // State threaded through `unfold`: the byte stream, the parser, a small
        // queue of chunks already parsed out of one HTTP chunk, the running
        // index, the drop-guard, and the cancellation handle.
        struct State {
            byte_stream: BoxStream<'static, reqwest::Result<bytes::Bytes>>,
            parser: NdjsonParser,
            pending: std::collections::VecDeque<std::result::Result<TokenChunk, Error>>,
            index: u64,
            finished: bool,
            cancel: CancellationToken,
            _guard: tokio_util::sync::DropGuard,
        }

        let state = State {
            byte_stream: byte_stream.boxed(),
            parser: NdjsonParser::default(),
            pending: std::collections::VecDeque::new(),
            index: 0,
            finished: false,
            cancel,
            _guard: guard,
        };

        let stream = futures::stream::unfold(state, |mut st| async move {
            loop {
                // Drain anything we already parsed from a previous HTTP chunk.
                if let Some(item) = st.pending.pop_front() {
                    return Some((item, st));
                }
                if st.finished {
                    return None;
                }

                // Wait for the next body chunk, but bail immediately if the
                // consumer dropped the stream (token cancelled).
                let next = tokio::select! {
                    biased;
                    _ = st.cancel.cancelled() => return None,
                    next = st.byte_stream.next() => next,
                };

                match next {
                    Some(Ok(bytes)) => {
                        // Feed the chunk to the parser and translate each
                        // completed NDJSON line into a TokenChunk.
                        for line in st.parser.push(&bytes) {
                            match parse_ollama_line(&line) {
                                Ok(Some(resp_line)) => {
                                    let done = resp_line.done;
                                    let chunk = TokenChunk {
                                        text: resp_line.response,
                                        index: st.index,
                                        done,
                                    };
                                    st.index += 1;
                                    st.pending.push_back(Ok(chunk));
                                    if done {
                                        st.finished = true;
                                        break;
                                    }
                                }
                                // Blank/whitespace-only line: skip it.
                                Ok(None) => {}
                                Err(e) => {
                                    st.pending.push_back(Err(e));
                                    st.finished = true;
                                    break;
                                }
                            }
                        }
                        // Loop back to emit from `pending`.
                    }
                    Some(Err(e)) => {
                        // A mid-stream transport error. We do NOT retry (that
                        // would duplicate already-delivered tokens); surface it
                        // as the stream's final item.
                        st.finished = true;
                        return Some((Err(map_reqwest_error(e)), st));
                    }
                    None => {
                        // Body ended. If the daemon never sent a done=true line
                        // we still just end the stream; any trailing partial
                        // line is intentionally dropped as incomplete.
                        return None;
                    }
                }
            }
        });

        Ok(stream.boxed())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let resp = classify_response(resp).await?;
        let tags: OllamaTags = resp
            .json()
            .await
            .map_err(|e| Error::BackendUnavailable(format!("invalid /api/tags response: {e}")))?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }
}

// ---------------------------------------------------------------------------
// Wire types for the Ollama API (private to this module).
// ---------------------------------------------------------------------------

/// Request body for `POST /api/generate`. We always set `stream: true`.
#[derive(Debug, serde::Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
}

/// Generation knobs nested under `options` in the Ollama request.
#[derive(Debug, serde::Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

impl<'a> From<&'a GenerateRequest> for OllamaGenerateRequest<'a> {
    fn from(req: &'a GenerateRequest) -> Self {
        let options = if req.max_tokens.is_some() || req.temperature.is_some() {
            Some(OllamaOptions {
                num_predict: req.max_tokens,
                temperature: req.temperature,
            })
        } else {
            None
        };
        Self {
            model: &req.model,
            prompt: &req.prompt,
            stream: true,
            options,
            stop: req.stop.as_deref(),
        }
    }
}

/// One NDJSON line of the `/api/generate` stream. We only need two fields; the
/// daemon sends more (timings, context) which serde ignores.
#[derive(Debug, Deserialize)]
struct OllamaResponseLine {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
}

/// Response of `GET /api/tags`.
#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTagModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    name: String,
}

// ---------------------------------------------------------------------------
// NDJSON reassembly — pure and unit-tested below.
// ---------------------------------------------------------------------------

/// Reassembles newline-delimited JSON lines from arbitrarily-chunked bytes.
///
/// HTTP delivers the body in chunks whose boundaries have nothing to do with
/// NDJSON line boundaries: a single `\n`-terminated JSON object can be split
/// across two chunks, and one chunk can contain several complete lines plus a
/// partial tail. [`push`](NdjsonParser::push) appends new bytes and returns the
/// lines that are now complete, keeping any trailing partial line buffered for
/// the next call. This is the piece most worth testing, so it is deliberately
/// free of any I/O.
#[derive(Debug, Default)]
pub struct NdjsonParser {
    buf: Vec<u8>,
}

impl NdjsonParser {
    /// Append `chunk` and return every complete line (without the trailing
    /// `\n`) that is now available. A partial final line is retained.
    ///
    /// Lines are returned as `String`s decoded lossily from UTF-8; Ollama only
    /// ever emits valid UTF-8 JSON, and lossy decoding keeps the parser
    /// infallible so malformed-JSON handling lives in one place
    /// ([`parse_ollama_line`]).
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut lines = Vec::new();
        // Repeatedly peel off everything up to the next newline.
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            // Exclude the trailing '\n' (and a '\r' if present for CRLF safety).
            let end = line.len() - 1;
            let end = if end > 0 && line[end - 1] == b'\r' {
                end - 1
            } else {
                end
            };
            lines.push(String::from_utf8_lossy(&line[..end]).into_owned());
        }
        lines
    }
}

/// Parse one NDJSON line into an [`OllamaResponseLine`].
///
/// Returns `Ok(None)` for a blank/whitespace-only line (which the daemon can
/// emit) and `Err(Error::BackendUnavailable)` for a line that is not valid
/// JSON — surfacing malformed output as an error rather than silently dropping
/// it.
fn parse_ollama_line(line: &str) -> Result<Option<OllamaResponseLine>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    // Ollama signals an unknown model inside the JSON as `{"error": "..."}`
    // (often with HTTP 200), so check for that before the normal shape.
    if let Ok(err) = serde_json::from_str::<OllamaError>(trimmed) {
        if let Some(message) = err.error {
            return Err(classify_ollama_error(&message));
        }
    }
    serde_json::from_str::<OllamaResponseLine>(trimmed)
        .map(Some)
        .map_err(|e| Error::BackendUnavailable(format!("malformed NDJSON line from Ollama: {e}")))
}

/// Shape of an in-band Ollama error object.
#[derive(Debug, Deserialize)]
struct OllamaError {
    #[serde(default)]
    error: Option<String>,
}

/// Map an Ollama error *message* onto a core error variant. An error mentioning
/// a missing model becomes [`Error::ModelNotFound`]; anything else is treated
/// as the backend being unable to serve the request.
fn classify_ollama_error(message: &str) -> Error {
    let lower = message.to_lowercase();
    if lower.contains("not found")
        || lower.contains("no such model")
        || lower.contains("try pulling")
    {
        Error::ModelNotFound(message.to_owned())
    } else {
        Error::BackendUnavailable(message.to_owned())
    }
}

/// Inspect an HTTP response's status and translate the error statuses onto core
/// error variants before we try to read the body as a stream.
async fn classify_response(resp: reqwest::Response) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    // Read the body for context; Ollama puts a JSON `error` message here.
    let body = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<OllamaError>(&body)
        .ok()
        .and_then(|e| e.error)
        .unwrap_or(body);
    if status == reqwest::StatusCode::NOT_FOUND {
        Err(Error::ModelNotFound(message))
    } else {
        Err(classify_ollama_error(&message))
    }
}

/// Is this a connection-establishment error worth retrying?
fn is_retryable_connect(e: &reqwest::Error) -> bool {
    e.is_connect() || e.is_timeout()
}

/// Map a transport-level reqwest error onto a core error variant.
fn map_reqwest_error(e: reqwest::Error) -> Error {
    if e.is_timeout() {
        Error::Timeout
    } else if e.is_connect() {
        Error::BackendUnavailable(format!("could not connect to Ollama: {e}"))
    } else {
        Error::BackendUnavailable(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- NDJSON reassembly across chunk boundaries ------------------------

    #[test]
    fn splits_multiple_lines_in_one_chunk() {
        let mut p = NdjsonParser::default();
        let lines = p.push(b"{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(lines, vec!["{\"a\":1}".to_owned(), "{\"b\":2}".to_owned()]);
    }

    #[test]
    fn holds_partial_line_until_completed() {
        let mut p = NdjsonParser::default();
        // First chunk ends mid-line: no complete line yet.
        let first = p.push(b"{\"response\":\"He");
        assert!(
            first.is_empty(),
            "partial line must be buffered, not emitted"
        );

        // Second chunk completes the line and starts another partial.
        let second = p.push(b"llo\",\"done\":false}\n{\"res");
        assert_eq!(
            second,
            vec!["{\"response\":\"Hello\",\"done\":false}".to_owned()]
        );

        // Final chunk completes the trailing line.
        let third = p.push(b"t\":1}\n");
        assert_eq!(third, vec!["{\"rest\":1}".to_owned()]);
    }

    #[test]
    fn strips_crlf_line_endings() {
        let mut p = NdjsonParser::default();
        let lines = p.push(b"{\"x\":1}\r\n");
        assert_eq!(lines, vec!["{\"x\":1}".to_owned()]);
    }

    // --- Line -> TokenChunk mapping ---------------------------------------

    #[test]
    fn parses_a_normal_response_line() {
        let line = r#"{"response":"Hello","done":false}"#;
        let parsed = parse_ollama_line(line)
            .expect("valid line")
            .expect("not blank");
        assert_eq!(parsed.response, "Hello");
        assert!(!parsed.done);
    }

    #[test]
    fn parses_a_done_true_terminal_line() {
        // The terminal line carries done=true and typically an empty response.
        let line = r#"{"response":"","done":true}"#;
        let parsed = parse_ollama_line(line)
            .expect("valid line")
            .expect("not blank");
        assert!(parsed.done, "terminal line must set done=true");
        assert_eq!(parsed.response, "");
    }

    #[test]
    fn blank_line_is_skipped() {
        assert!(parse_ollama_line("   ").expect("ok").is_none());
    }

    #[test]
    fn malformed_line_surfaces_an_error() {
        // Not valid JSON at all.
        let err = parse_ollama_line("this is not json").expect_err("must error");
        assert!(
            matches!(err, Error::BackendUnavailable(_)),
            "malformed NDJSON should surface as BackendUnavailable, got {err:?}"
        );
    }

    #[test]
    fn in_band_model_error_becomes_model_not_found() {
        let line = r#"{"error":"model 'nope' not found, try pulling it first"}"#;
        let err = parse_ollama_line(line).expect_err("must error");
        assert!(
            matches!(err, Error::ModelNotFound(_)),
            "an in-band 'not found' error should map to ModelNotFound, got {err:?}"
        );
    }

    #[test]
    fn full_ndjson_sequence_maps_to_ordered_chunks() {
        // Simulate the parser + line mapping over a realistic split stream.
        let mut p = NdjsonParser::default();
        let mut chunks: Vec<TokenChunk> = Vec::new();
        let mut index = 0u64;

        let feed: &[&[u8]] = &[
            b"{\"response\":\"Hel",
            b"lo\",\"done\":false}\n{\"response\":\" world\",\"done\":false}\n",
            b"{\"response\":\"\",\"done\":true}\n",
        ];
        for bytes in feed {
            for line in p.push(bytes) {
                if let Some(rl) = parse_ollama_line(&line).expect("valid") {
                    chunks.push(TokenChunk {
                        text: rl.response,
                        index,
                        done: rl.done,
                    });
                    index += 1;
                }
            }
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Hello");
        assert_eq!(chunks[0].index, 0);
        assert!(!chunks[0].done);
        assert_eq!(chunks[1].text, " world");
        assert_eq!(chunks[1].index, 1);
        assert!(chunks[2].done, "last chunk is terminal");
    }
}
