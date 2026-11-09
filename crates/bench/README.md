# llm-stream-bench

A command-line load + latency benchmark for the `llm-stream-server` streaming
API. It drives the **running HTTP server** (over `reqwest` + rustls) and reports
latency/throughput statistics as a Markdown table you can paste into a README.

It deliberately talks to the *server over HTTP*, not the backends in-process, so
every number includes the real request path the server actually serves (routing,
concurrency permit, SSE framing).

## What it measures (per concurrency level, over post-warmup requests)

| Metric | Meaning |
| --- | --- |
| **TTFT p50/p95/p99** | **Time To First Token** — wall time from sending the request to the first SSE `data:` token event. Only meaningful for the `stream` endpoint; reported as `N/A` for `generate`. |
| **Per-stream tok/s** | For each request, `token_events / (t_last_token − t_first_token)`. Reported as mean and p50 across requests. This is the throughput a single client sees. |
| **Aggregate tok/s** | Total tokens emitted across all requests in the level ÷ the wall-clock duration of the whole sweep. This captures the benefit of concurrency. |
| **Latency p50/p95** | End-to-end time per request (send → final event). For `generate` this is the only latency number (there is no first token). |
| **Error rate** | Fraction of requests that failed: non-200 status, transport error, or malformed/incomplete SSE (missing terminal `done`, zero tokens, parse error). |

## How it works

### Worker pool / backpressure

For each concurrency level `N`, requests are driven through a
`futures::stream::iter(..).buffer_unordered(N)` pipeline. `buffer_unordered`
polls up to `N` request futures concurrently and only pulls the next request
from the iterator when one in-flight future completes. That *is* the
backpressure: the generator never races ahead of the configured concurrency, so
there are always exactly `N` requests in flight until the level is drained. No
explicit semaphore or task spawning is needed, which keeps the data flow easy to
reason about.

Warmup requests run first as their own buffered pass and are discarded, so
one-time costs (connection setup, first-request allocation) do not skew the
measured numbers.

### TTFT measurement

For the `stream` endpoint the SSE response body is read as a **byte stream** and
parsed incrementally. SSE frames are separated by a blank line (`\n\n`); each
frame's `event:` field names its type (absent ⇒ the default `message` event).
The server emits:

* one default `message` event per token, whose `data:` is the JSON `TokenChunk`,
* a terminal `event: done` frame with data `[DONE]`,
* an `event: error` frame if the backend fails.

The harness timestamps the moment it finishes reading the **first token frame**
with a monotonic `std::time::Instant`, and TTFT is `t_first_token − t_send`. It
is *not* the time to the whole response. The last-token timestamp and the token
count give per-stream throughput; seeing the `done` frame confirms a clean end.

All timing uses `std::time::Instant` (monotonic), never wall-clock time.

### Percentiles

Percentiles use the **nearest-rank** method on the sorted samples: for `n`
samples and percentile `p`, `rank = ceil(p/100 · n)` clamped to `[1, n]`, and we
return the value at 1-based position `rank`. This always returns a real observed
measurement (no interpolation), which is the honest thing to report for latency.
The helper is a dozen lines in `src/stats.rs` with unit tests, rather than a
heavy stats crate.

## Usage

The server must already be running. Start it with the mock backend for
deterministic, network-free numbers:

```powershell
# from the repo root, with cargo on PATH
cargo run -p server --release -- --backend mock --model mock-small --port 8080 --max-concurrency 64
```

Then, in another shell, run the benchmark:

```powershell
cargo run -p bench --release -- `
    --url http://127.0.0.1:8080 `
    --endpoint stream `
    --model mock-small `
    --prompt "Explain Rust ownership in one paragraph" `
    --concurrency 1,8,32 `
    --requests 100
```

The bench polls `/health` itself before starting, so a small startup race is
fine.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8080` | Base URL of the running server. |
| `--endpoint` | `stream` | `stream` ⇒ `/v1/generate/stream` (SSE), `generate` ⇒ `/v1/generate`. |
| `--model` | `mock-small` | Model name sent in each request. |
| `--prompt` | `Explain Rust ownership in one paragraph` | Prompt sent in each request. |
| `--concurrency` | `1,8,32` | Comma-separated levels; one sweep per level. |
| `--requests` | `100` | Measured requests per level. |
| `--max-tokens` | *(none)* | Optional `max_tokens` forwarded to the server. |
| `--warmup` | `5` | Requests discarded before measuring, per level. |
| `--output` | *(stdout)* | Path to write the Markdown table. |

## Mock vs. Ollama — read this before putting numbers on a resume

**Mock backend numbers measure the SERVER + HARNESS overhead, not model speed.**
The mock emits a fixed sentence with a fixed per-token delay (10 ms/token in the
default server config), so per-stream tok/s is dominated by that artificial
delay. What the mock numbers legitimately show is the HTTP/SSE path cost and how
**aggregate** throughput scales as concurrency rises — i.e. the server's ability
to multiplex streams, independent of any model.

**Ollama backend numbers measure real model throughput** on *your* hardware and
model. They are not comparable across machines and will differ from the mock.

Be explicit about which backend produced any number you report. Presenting mock
tok/s as model performance would be dishonest.
