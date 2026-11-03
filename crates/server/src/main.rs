//! Binary entry point: parse config, build the backend + router, and serve
//! with graceful shutdown. All the interesting logic lives in the library
//! (`llm_stream_server`) so the same router can be exercised by integration
//! tests.

use std::net::SocketAddr;

use clap::Parser;
use llm_stream_server::{app, AppState, Config};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Structured logging. `RUST_LOG` controls the filter (e.g.
    // `RUST_LOG=info,llm_stream_server=debug`); default to `info` so a plain
    // run is useful without extra env setup.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::parse();
    tracing::info!(?config.backend, model = %config.model, port = config.port, max_concurrency = config.max_concurrency, "starting server");

    let backend = config.build_backend()?;
    let state = AppState::new(backend, config.model.clone(), config.max_concurrency);
    let router = app(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    // Graceful shutdown: `serve` stops accepting new connections and lets
    // in-flight requests drain when the shutdown future resolves (Ctrl+C).
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("server stopped");
    Ok(())
}

/// Resolves when the process receives Ctrl+C, triggering graceful shutdown.
async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        // If we can't install the handler, log and fall through so we don't
        // hang forever; the server will exit on the next shutdown path.
        tracing::error!(error = %e, "failed to listen for Ctrl+C");
    }
    tracing::info!("shutdown signal received");
}
