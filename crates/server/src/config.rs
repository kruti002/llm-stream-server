//! Startup configuration, parsed from CLI flags with env-var fallback.
//!
//! We use `clap`'s derive API so the flags are declared once as a struct and
//! `clap` generates the parser, `--help`, and env-var wiring. Every flag also
//! reads an environment variable (via `env = ...`) so the same server can be
//! configured by flags in development and by env vars in a container, which is
//! the usual twelve-factor setup. Precedence is clap's default: an explicit
//! flag overrides the env var, which overrides the built-in default.

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use llm_stream_core::{Backend, MockBackend, OllamaBackend};

use crate::error::BootError;

/// Which backend the server should serve requests from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lowercase")]
pub enum BackendKind {
    /// The deterministic, model-free backend — used by CI and local dev.
    Mock,
    /// A real local Ollama daemon.
    Ollama,
}

/// Server configuration.
#[derive(Debug, Clone, Parser)]
#[command(name = "llm-stream-server", about = "Stream LLM tokens over HTTP/SSE")]
pub struct Config {
    /// Backend to serve from: `mock` (no model needed) or `ollama`.
    #[arg(long, value_enum, default_value_t = BackendKind::Mock, env = "LLM_BACKEND")]
    pub backend: BackendKind,

    /// Default model name advertised/used by the backend.
    #[arg(long, default_value = "llama3.2:1b", env = "LLM_MODEL")]
    pub model: String,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080, env = "LLM_PORT")]
    pub port: u16,

    /// Maximum number of generations running at once. Requests beyond this get
    /// an immediate `429` rather than queueing unbounded.
    #[arg(long, default_value_t = 8, env = "LLM_MAX_CONCURRENCY")]
    pub max_concurrency: usize,

    /// Base URL of the Ollama daemon (only used when `--backend ollama`).
    #[arg(long, default_value = llm_stream_core::DEFAULT_OLLAMA_URL, env = "LLM_OLLAMA_URL")]
    pub ollama_url: String,
}

impl Config {
    /// Construct the configured backend as a shared trait object.
    ///
    /// Returning `Arc<dyn Backend>` is what lets the rest of the server — and
    /// every axum handler — be written against the trait with no knowledge of
    /// which concrete backend is running.
    pub fn build_backend(&self) -> Result<Arc<dyn Backend>, BootError> {
        match self.backend {
            // A modest per-token delay keeps the mock's streaming observable in
            // demos/tests without being slow.
            BackendKind::Mock => Ok(Arc::new(MockBackend::new(Duration::from_millis(10)))),
            BackendKind::Ollama => {
                let backend = OllamaBackend::new(&self.ollama_url, Duration::from_secs(30))
                    .map_err(|e| BootError(format!("failed to build Ollama backend: {e}")))?;
                Ok(Arc::new(backend))
            }
        }
    }
}
