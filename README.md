# llm-stream-server

A learning/interview project: a streaming LLM server built in small milestones,
with pluggable inference backends and a strong focus on explaining *why* each
piece is shaped the way it is.

> Status: **Milestone 4** — the `core` crate (types, `Backend` trait,
> `MockBackend`, `OllamaBackend`), the `server` crate (Axum REST + SSE), the
> `py` crate (PyO3 bindings), and now a `bench` crate: an HTTP load + latency
> benchmark that drives the server's streaming endpoint and reports real
> latency/throughput numbers. The FFI crate lands later.

## Workspace layout

```
llm-stream-server/
├── Cargo.toml          # workspace manifest (workspace.dependencies)
└── crates/
    ├── core/           # types, error, Backend trait, Mock + Ollama backends
    ├── server/         # Axum HTTP server: /health, /v1/models, /v1/generate(/stream)
    ├── py/             # PyO3 bindings (llm_stream)
    └── bench/          # HTTP load + latency benchmark tool  (this milestone)
```

## Core crate

See [`crates/core/README.md`](crates/core/README.md) for the key design
decisions: why a `Backend` trait, why the token stream is boxed, and how
cancellation works.

## Benchmarks

The `bench` crate (`crates/bench`) is an HTTP load generator that drives the
server's streaming endpoint and reports latency/throughput. See
[`crates/bench/README.md`](crates/bench/README.md) for what every metric means,
the worker-pool/backpressure design, how TTFT is measured from the SSE stream,
and the percentile method.

### Reproduce

Start the server with the mock backend, then run the benchmark against it:

```powershell
# shell 1 — start the server (mock backend, no model/network needed)
cargo run -p server --release -- --backend mock --model mock-small --port 8080 --max-concurrency 64

# shell 2 — drive it
cargo run -p bench --release -- `
    --url http://127.0.0.1:8080 `
    --endpoint stream `
    --model mock-small `
    --prompt "Explain Rust ownership in one paragraph" `
    --concurrency 1,8,32 `
    --requests 100
```

### Results (mock backend — server + harness overhead)

> **These are mock-backend numbers measured on this machine over localhost.**
> They measure the **server + HTTP/SSE + harness overhead**, *not* model speed:
> the mock emits a fixed 9-token sentence with a fixed ~10 ms per-token delay, so
> per-stream tok/s is dominated by that artificial delay. What they legitimately
> show is the request-path cost and how **aggregate** throughput scales with
> concurrency. **Ollama numbers measure real model throughput and depend on your
> own hardware/model** — run the command above with `--backend ollama` on the
> server to produce them.

Benchmark: `stream` endpoint, model `mock-small`, 100 requests/level (5 warmup discarded).

| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Per-stream tok/s (mean) | Per-stream tok/s (p50) | Aggregate tok/s | Latency p50 (ms) | Latency p95 (ms) | Error rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.5 | 16.1 | 19.3 | 71.3 | 72.1 | 63.4 | 140.3 | 151.3 | 0.00% |
| 8 | 15.1 | 25.0 | 25.1 | 72.0 | 72.0 | 493.2 | 140.0 | 148.8 | 0.00% |
| 32 | 15.2 | 29.2 | 30.1 | 72.6 | 72.7 | 1611.9 | 139.7 | 152.7 | 0.00% |

Per-stream throughput stays flat (~72 tok/s, the mock's fixed per-token rate)
while **aggregate** throughput scales roughly linearly with concurrency
(63 → 493 → 1612 tok/s), which is the server multiplexing independent streams.
Single-stream latency (~140 ms for a 9-token sentence) and TTFT (~15 ms) stay
stable as load rises, with a 0% error rate.

## Checks

From this directory (with cargo on PATH):

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
