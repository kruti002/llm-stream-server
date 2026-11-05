//! # llm_stream — PyO3 bindings for the in-process core streaming client
//!
//! This crate exposes the Rust [`Backend`](llm_stream_core::Backend)
//! implementations (the deterministic `MockBackend` and the real
//! `OllamaBackend`) to Python as the `llm_stream` package. It binds the
//! *in-process* core client: there is no HTTP server and no subprocess, tokens
//! are produced by the Rust async stream and handed straight across the FFI
//! boundary.
//!
//! ```python
//! from llm_stream import Client
//! c = Client(backend="mock", model="mock-small")
//! print(c.generate("Hello"))          # full completion as a str (blocking)
//! for tok in c.stream("Explain Rust"):# a Python iterator yielding str tokens
//!     print(tok, end="", flush=True)
//! ```
//!
//! ## Bridging async Rust to blocking Python
//!
//! Python's `Client.generate`/`Client.stream` are *synchronous* from Python's
//! point of view, but the core `Backend` is async. We square that circle with a
//! tokio multi-thread [`Runtime`](tokio::runtime::Runtime) owned by each
//! `Client`:
//!
//! * `generate` uses `Runtime::block_on` to drive the whole stream to
//!   completion and concatenate the token text.
//! * `stream` spawns a task on the runtime that reads the async `BoxStream` and
//!   forwards each token through a [`std::sync::mpsc`] channel. The returned
//!   Python iterator's `__next__` does a *blocking* `recv` on that channel.
//!
//! ## The GIL (the headline interview point)
//!
//! Any time we block a Rust thread waiting on Python-independent work — here,
//! `Runtime::block_on` in `generate` and the blocking channel `recv` in the
//! iterator's `__next__` — we must RELEASE the GIL so other Python threads can
//! run meanwhile. If we held the GIL across those blocking waits, a single
//! `Client` doing a slow generation would freeze every other Python thread in
//! the process. We release it with [`Python::allow_threads`]: the closure
//! passed to it runs with the GIL dropped, and the GIL is re-acquired when the
//! closure returns. The exact spans are marked `GIL RELEASED` below.
//!
//! ## Cancellation
//!
//! The core backends cancel generation when their `BoxStream` is dropped (a
//! `CancellationToken` drop-guard; see the core crate). The Python iterator
//! owns the channel receiver and the runtime task owns the stream, so when the
//! Python iterator is garbage-collected we drop the receiver, the forwarding
//! task sees the channel close on its next `send`, returns, and drops the
//! `BoxStream` — cancelling generation. See [`TokenStream`] for the details.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use llm_stream_core::{
    Backend, Error, GenerateRequest, MockBackend, OllamaBackend, DEFAULT_OLLAMA_URL,
};

// Base exception for every error raised by this module. Subclassing
// `RuntimeError` means existing `except RuntimeError:` handlers keep working
// while callers that want to be specific can catch `LlmStreamError`.
create_exception!(llm_stream, LlmStreamError, PyRuntimeError);

// A backend that is reachable-but-refusing or simply down (maps from
// Error::BackendUnavailable / Error::Timeout).
create_exception!(llm_stream, BackendError, LlmStreamError);

// The requested model name is unknown to the backend (maps from
// Error::ModelNotFound).
create_exception!(llm_stream, ModelNotFoundError, LlmStreamError);

/// Translate a core [`Error`] into the most appropriate Python exception.
///
/// * A bad/empty model name is a *caller* mistake, so it becomes `ValueError`.
/// * A missing model is its own catchable type.
/// * Everything else is a backend/runtime condition.
fn map_core_error(err: Error) -> PyErr {
    match err {
        Error::ModelNotFound(msg) => ModelNotFoundError::new_err(msg),
        Error::BackendUnavailable(msg) => BackendError::new_err(msg),
        Error::Timeout => BackendError::new_err("operation timed out"),
        Error::Cancelled => LlmStreamError::new_err("operation cancelled"),
    }
}

/// Which concrete core backend a [`Client`] drives.
enum BackendKind {
    Mock(MockBackend),
    Ollama(OllamaBackend),
}

impl BackendKind {
    async fn generate_stream(
        &self,
        req: GenerateRequest,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<llm_stream_core::TokenChunk, Error>>,
        Error,
    > {
        match self {
            BackendKind::Mock(b) => b.generate_stream(req).await,
            BackendKind::Ollama(b) => b.generate_stream(req).await,
        }
    }
}

/// A Python-facing client over one in-process core backend.
///
/// Holds its own tokio multi-thread runtime so the async `Backend` can be
/// driven from synchronous Python calls. The backend is behind an `Arc` so the
/// streaming task spawned by [`Client::stream`] can own a clone that outlives
/// the method call.
#[pyclass]
struct Client {
    runtime: Arc<tokio::runtime::Runtime>,
    backend: Arc<BackendKind>,
    model: String,
}

#[pymethods]
impl Client {
    /// Construct a client over the chosen backend.
    ///
    /// `backend` must be `"mock"` or `"ollama"`. `ollama_url` is only consulted
    /// for the ollama backend and defaults to the daemon's conventional URL.
    #[new]
    #[pyo3(signature = (backend, model, ollama_url=None))]
    fn new(backend: &str, model: String, ollama_url: Option<String>) -> PyResult<Self> {
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be a non-empty string"));
        }

        let kind = match backend {
            "mock" => {
                // A small nonzero per-token delay keeps streaming observable
                // from Python (you can see tokens arrive one at a time) without
                // slowing the test suite down meaningfully.
                BackendKind::Mock(MockBackend::new(Duration::from_millis(5)))
            }
            "ollama" => {
                let url = ollama_url.unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_owned());
                let b = OllamaBackend::new(url, Duration::from_secs(30)).map_err(map_core_error)?;
                BackendKind::Ollama(b)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown backend {other:?}; expected \"mock\" or \"ollama\""
                )));
            }
        };

        // A multi-thread runtime (rather than current-thread) so the forwarding
        // task in `stream` keeps making progress on a runtime worker thread
        // while the Python thread is blocked in `__next__` waiting on the
        // channel. Built once per Client and shared via Arc.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| LlmStreamError::new_err(format!("failed to build tokio runtime: {e}")))?;

        Ok(Self {
            runtime: Arc::new(runtime),
            backend: Arc::new(kind),
            model,
        })
    }

    /// Run the backend stream to completion and return the full completion text.
    ///
    /// Blocking from Python's perspective. The GIL is released for the entire
    /// `block_on` so other Python threads keep running while generation is in
    /// flight.
    fn generate(&self, py: Python<'_>, prompt: String) -> PyResult<String> {
        let req = self.make_request(prompt);
        let runtime = Arc::clone(&self.runtime);
        let backend = Arc::clone(&self.backend);

        // GIL RELEASED for the whole blocking drive of the stream. Nothing
        // inside the closure touches a Python object, so it is safe (and
        // necessary) to drop the GIL here; re-acquired when the closure returns.
        let result: Result<String, Error> = py.allow_threads(move || {
            runtime.block_on(async move {
                let mut stream = backend.generate_stream(req).await?;
                let mut out = String::new();
                while let Some(item) = stream.next().await {
                    out.push_str(&item?.text);
                }
                Ok(out)
            })
        });

        result.map_err(map_core_error)
    }

    /// Return a Python iterator yielding token text one `str` at a time.
    ///
    /// Spawns a forwarding task on the runtime that feeds a channel; the
    /// returned [`TokenStream`] blocks on that channel in `__next__`.
    fn stream(&self, prompt: String) -> PyResult<TokenStream> {
        let req = self.make_request(prompt);
        let runtime = Arc::clone(&self.runtime);
        let backend = Arc::clone(&self.backend);

        // A rendezvous-ish channel: the forwarding task sends either token text
        // or a terminal error; the iterator receives on the Python side. A
        // sync (std) channel is used precisely because the *consumer* side is a
        // blocking Python call, not an async task.
        let (tx, rx) = mpsc::channel::<Result<String, Error>>();

        runtime.spawn(async move {
            // Build the stream; a pre-stream failure (unknown model, daemon
            // down) is forwarded as the single terminal item.
            let mut stream = match backend.generate_stream(req).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            while let Some(item) = stream.next().await {
                match item {
                    Ok(chunk) => {
                        // If the receiver (the Python iterator) was dropped,
                        // `send` errors. We break and fall out of the loop,
                        // which drops `stream` -> drops the core BoxStream ->
                        // fires its CancellationToken drop-guard -> cancels
                        // generation. This is the iterator-drop => cancel path.
                        if tx.send(Ok(chunk.text)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
            // `stream` is dropped here on any exit, cancelling the backend.
        });

        Ok(TokenStream {
            rx: std::sync::Mutex::new(rx),
        })
    }

    fn __repr__(&self) -> String {
        let kind = match &*self.backend {
            BackendKind::Mock(_) => "mock",
            BackendKind::Ollama(_) => "ollama",
        };
        format!("Client(backend={kind:?}, model={:?})", self.model)
    }
}

impl Client {
    /// Build the core request from a prompt, filling in this client's model and
    /// leaving the optional knobs at the backend defaults.
    fn make_request(&self, prompt: String) -> GenerateRequest {
        GenerateRequest {
            model: self.model.clone(),
            prompt,
            max_tokens: None,
            temperature: None,
            stop: None,
        }
    }
}

/// A blocking Python iterator over streamed tokens.
///
/// Holds the receiving end of the channel fed by the runtime task spawned in
/// [`Client::stream`]. Dropping this object (when Python garbage-collects the
/// iterator) drops `rx`; the next `send` in the forwarding task then fails,
/// the task breaks out of its loop and drops the core `BoxStream`, which
/// cancels generation. That is how iterator-drop maps to stream-cancellation.
#[pyclass]
struct TokenStream {
    // Wrapped in a `Mutex` for two reasons:
    //   * `#[pyclass]` requires the type to be `Sync`, and a bare
    //     `mpsc::Receiver` is `Send` but not `Sync`. `Mutex<T>` is `Sync`
    //     whenever `T: Send`, which lifts the receiver to `Sync`.
    //   * `Python::allow_threads` requires its closure to be `Ungil`; a shared
    //     reference to the receiver only qualifies once the receiver sits
    //     behind a `Sync` wrapper. `__next__` takes `&self`, so the lock is
    //     uncontended in practice (one iterator, one consuming thread).
    rx: std::sync::Mutex<mpsc::Receiver<Result<String, Error>>>,
}

#[pymethods]
impl TokenStream {
    /// An iterator is its own iterable.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Block until the next token is available.
    ///
    /// Returns `Ok(Some(text))` for a token, `Ok(None)` to signal
    /// `StopIteration` (channel closed = stream finished cleanly), or an
    /// exception if the backend surfaced an error.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        // GIL RELEASED across the blocking channel `recv`. This is the crucial
        // span: while this Python thread parks waiting for the next token, the
        // GIL is dropped so *other* Python threads can run. The closure never
        // touches a Python object, so dropping the GIL is sound; it is
        // re-acquired before we build the return value.
        let next = py.allow_threads(|| {
            // The lock is only ever held by this single consuming thread; the
            // producer uses the `Sender` half, not this receiver.
            let rx = self.rx.lock().expect("TokenStream mutex poisoned");
            rx.recv()
        });

        match next {
            // Channel closed: the forwarding task finished and dropped `tx`.
            // `None` makes PyO3 raise StopIteration for us.
            Err(mpsc::RecvError) => Ok(None),
            Ok(Ok(text)) => Ok(Some(text)),
            Ok(Err(e)) => Err(map_core_error(e)),
        }
    }
}

/// The `llm_stream` extension module.
#[pymodule]
fn llm_stream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<TokenStream>()?;
    m.add("LlmStreamError", m.py().get_type::<LlmStreamError>())?;
    m.add("BackendError", m.py().get_type::<BackendError>())?;
    m.add(
        "ModelNotFoundError",
        m.py().get_type::<ModelNotFoundError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Rust-side tests for the async -> blocking-channel bridge that underpins
    //! `Client::stream`. These exercise the exact forwarding/cancellation logic
    //! without needing a Python interpreter, so they run under a plain
    //! `cargo test` at the workspace root.

    use super::*;
    use std::time::Duration;

    /// Mirror of the forwarding task in `Client::stream`, parameterised over the
    /// backend so a test can drive it directly. Keeping this helper tiny and
    /// faithful to the real loop lets us assert the channel bridge end to end.
    fn spawn_forwarder(
        runtime: &tokio::runtime::Runtime,
        backend: Arc<BackendKind>,
        req: GenerateRequest,
    ) -> mpsc::Receiver<Result<String, Error>> {
        let (tx, rx) = mpsc::channel::<Result<String, Error>>();
        runtime.spawn(async move {
            let mut stream = match backend.generate_stream(req).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            while let Some(item) = stream.next().await {
                match item {
                    Ok(chunk) => {
                        if tx.send(Ok(chunk.text)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        });
        rx
    }

    fn req() -> GenerateRequest {
        GenerateRequest {
            model: "mock-small".to_owned(),
            prompt: "hi".to_owned(),
            max_tokens: None,
            temperature: None,
            stop: None,
        }
    }

    #[test]
    fn bridge_forwards_all_tokens_in_order_then_closes() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let backend = Arc::new(BackendKind::Mock(MockBackend::new(Duration::from_millis(
            1,
        ))));
        let rx = spawn_forwarder(&rt, backend, req());

        let mut got = Vec::new();
        while let Ok(item) = rx.recv() {
            got.push(item.expect("mock never errors"));
        }

        let expected: Vec<String> = llm_stream_core::MOCK_SENTENCE
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn dropping_receiver_cancels_generation() {
        // Prove iterator-drop => stream-drop => generation cancelled, using the
        // mock's observability hooks (the same pattern the core cancellation
        // test uses). A long per-token delay means the producer is parked in
        // `sleep` after the first token; dropping the receiver must make the
        // forwarder's next `send` fail, break its loop, and drop the core
        // BoxStream, whose CancellationToken drop-guard stops the producer at
        // its next await — so `exited` flips true quickly and `emitted` stays
        // small, rather than the producer running out the whole sentence.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let emitted = Arc::new(AtomicUsize::new(0));
        let exited = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(BackendKind::Mock(MockBackend::with_observers(
            Duration::from_millis(50),
            Arc::clone(&emitted),
            Arc::clone(&exited),
        )));
        let rx = spawn_forwarder(&rt, backend, req());

        // Receive exactly one token, then drop the receiver.
        let first = rx.recv().expect("first token");
        assert_eq!(first.unwrap(), "The");
        drop(rx);

        // Give the cancellation a moment to propagate, well under the time it
        // would take the producer to emit the remaining 8 tokens (8 * 50ms).
        std::thread::sleep(Duration::from_millis(150));

        assert!(
            exited.load(Ordering::SeqCst),
            "producer task must exit promptly after the receiver is dropped"
        );
        assert!(
            emitted.load(Ordering::SeqCst) < 9,
            "producer must have been cancelled before emitting the whole sentence, got {}",
            emitted.load(Ordering::SeqCst)
        );

        rt.shutdown_timeout(Duration::from_secs(5));
    }
}
