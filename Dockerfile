# Multi-stage build for the llm-stream-server binary.
#
# TLS is rustls (pure Rust, ring-based) everywhere in this workspace, so the
# runtime image needs NO OpenSSL / libssl package — there is nothing to apt-get
# install for TLS. The runtime stage therefore stays a plain slim Debian with
# only the compiled binary copied in.

# ---- Stage 1: builder -------------------------------------------------------
FROM rust:1-bookworm AS builder
WORKDIR /app

# Copy the whole workspace (Cargo.toml, Cargo.lock, all member crates) so the
# workspace resolves exactly as it does locally, then build the server binary
# in release mode.
COPY . .
RUN cargo build --release -p llm-stream-server

# ---- Stage 2: runtime -------------------------------------------------------
# debian:bookworm-slim is a small glibc base that matches the builder's glibc.
# No TLS packages needed (rustls), so nothing is installed here.
FROM debian:bookworm-slim AS runtime

# Copy only the compiled binary out of the builder's target/release directory.
COPY --from=builder /app/target/release/llm-stream-server /usr/local/bin/llm-stream-server

# Default to the mock backend (no model/network needed) on the default port.
# Everything is overridable via env (clap reads LLM_* vars) or by appending
# flags after the image name, e.g.
#   docker run -p 8080:8080 -e LLM_BACKEND=ollama -e LLM_OLLAMA_URL=... IMAGE
ENV LLM_BACKEND=mock \
    LLM_PORT=8080

# The server binds 0.0.0.0:8080 by default (see crates/server/src/config.rs).
EXPOSE 8080

ENTRYPOINT ["llm-stream-server"]
