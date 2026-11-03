//! Wire types shared between the server, the backends, and the clients.
//!
//! Every type here derives `serde::{Serialize, Deserialize}` because the same
//! structs travel over HTTP/SSE (server crate), across the PyO3 boundary (py
//! crate), and through the benchmark tool. Defining them once in `core` keeps a
//! single source of truth for the public contract.

use serde::{Deserialize, Serialize};

/// A single generation request submitted to a backend.
///
/// Field types were chosen to mirror what inference backends (Ollama,
/// llama.cpp) actually accept, and the optional knobs are `Option<_>` so a
/// caller can omit them and let the backend apply its own defaults rather than
/// forcing a magic sentinel value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Name of the model to run, e.g. `"llama3.2:1b"`. Required — a backend
    /// cannot pick one for the caller.
    pub model: String,

    /// The prompt text to generate a continuation for.
    pub prompt: String,

    /// Upper bound on the number of tokens to generate. `None` means "use the
    /// backend's default limit". `u32` is plenty of headroom for context sizes
    /// and keeps the value non-negative by construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature. `None` defers to the backend default. `f32` matches
    /// the precision inference engines use for this parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Optional stop sequences; generation halts when any of these strings is
    /// produced. `None` (or an empty list) means "no custom stop sequences".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// One unit of streamed output.
///
/// A stream of `TokenChunk`s is how a backend reports progress incrementally.
/// The shape is deliberately small so it is cheap to serialize once per token
/// over SSE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenChunk {
    /// The text fragment for this chunk. Usually a single token/word, but a
    /// backend may emit multi-character pieces.
    pub text: String,

    /// Zero-based position of this chunk within the stream. Monotonically
    /// increasing, so a consumer can detect gaps or reordering.
    pub index: u64,

    /// `true` on the final chunk of a stream. Consumers use this to know the
    /// generation finished cleanly (as opposed to the stream being dropped).
    pub done: bool,
}
