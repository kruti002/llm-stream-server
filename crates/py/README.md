# llm-stream-py (`llm_stream`)

PyO3 bindings that expose the `llm-stream-core` backends to Python as the
`llm_stream` package. This binds the **in-process** Rust streaming client — it
does not talk to the HTTP server and never shells out.

```python
from llm_stream import Client

c = Client(backend="ollama", model="llama3.2:1b")   # backend in {"mock","ollama"}
                                                     # optional: ollama_url="http://host:11434"
print(c.generate("Hello"))                           # full completion as a str (blocking)

for tok in c.stream("Explain Rust ownership"):       # a Python iterator of str tokens
    print(tok, end="", flush=True)
```

Exceptions: `LlmStreamError` (base, subclasses `RuntimeError`), `BackendError`
(backend down / timed out), and `ModelNotFoundError` (unknown model). Bad
arguments (unknown backend name, empty model) raise `ValueError`.

## How it works

* Each `Client` owns a tokio **multi-thread runtime** that drives the async core
  `Backend`. Python-facing methods are synchronous/blocking.
* `generate` runs the stream to completion with `Runtime::block_on` and
  concatenates token text.
* `stream` spawns a forwarding task on the runtime that pushes tokens through a
  `std::sync::mpsc` channel; the returned iterator's `__next__` blocks on a
  channel `recv`.

### GIL handling (the design point)

While blocking on Python-independent work — `block_on` in `generate` and the
channel `recv` in the iterator's `__next__` — the binding releases the GIL via
`Python::allow_threads` so other Python threads keep running. See the
`GIL RELEASED` comments in `src/lib.rs` (in `Client::generate` and
`TokenStream::__next__`). Holding the GIL across those waits would freeze every
other Python thread in the process during a slow generation.

### Cancellation

The core backends cancel generation when their stream is dropped (a
`CancellationToken` drop-guard). The Python iterator owns the channel receiver;
when Python garbage-collects it, `rx` is dropped, the forwarding task's next
`send` fails, it breaks out and drops the core `BoxStream`, cancelling
generation.

## Building and testing

The machine uses the **GNU** Rust toolchain on Windows
(`stable-x86_64-pc-windows-gnu`, MinGW, no MSVC) with CPython 3.12. PyO3 builds
cleanly there; TLS in the ollama path is rustls (set in the core crate), so
nothing links OpenSSL.

```powershell
# from crates/py
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # (bash: source .venv/bin/activate)
pip install maturin pytest
maturin develop                        # builds with pyo3/extension-module and installs into the venv
pytest                                 # runs tests/ against the mock backend (no model, no network)
```

### Why `cargo test` at the workspace root still works

The `pyo3/extension-module` feature tells PyO3 not to link libpython (symbols
are resolved at runtime by the interpreter that loads the extension). A plain
`cargo test` builds a standalone test binary that *would* have to link libpython
itself, so `extension-module` must be **off** for it. We make `extension-module`
a default-off Cargo feature; maturin enables it via `pyproject.toml`
(`features = ["pyo3/extension-module"]`), while `cargo build`/`cargo test` at the
root build the crate without it and stay green.

## TODO (later milestone)

* asyncio / `astream` stretch: an async iterator (`async for tok in
  c.astream(...)`) bridging the stream to an asyncio event loop. Not implemented
  in this milestone.
