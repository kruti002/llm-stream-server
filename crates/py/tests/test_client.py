"""Python tests for the `llm_stream` extension, run against the MOCK backend.

These need no model and no network: the mock backend streams a fixed sentence
token by token. Run them with:

    # from crates/py, inside a venv with maturin + pytest installed
    maturin develop
    pytest
"""

import gc

import pytest

import llm_stream
from llm_stream import Client


# The mock backend streams this exact sentence, one whitespace-split token per
# chunk. Keep it in sync with MOCK_SENTENCE in crates/core/src/mock.rs.
MOCK_SENTENCE = "The quick brown fox jumps over the lazy dog"
MOCK_TOKENS = MOCK_SENTENCE.split()


def make_client():
    return Client(backend="mock", model="mock-small")


def test_generate_returns_full_sentence():
    c = make_client()
    # generate() concatenates the token text. The mock tokens are the words of
    # the sentence with no separators, so the concatenation is the words joined
    # with nothing between them.
    result = c.generate("Hello")
    assert result == "".join(MOCK_TOKENS)


def test_stream_yields_tokens_in_order_and_terminates():
    c = make_client()
    tokens = list(c.stream("Explain Rust ownership"))
    assert tokens == MOCK_TOKENS
    # A second independent stream also terminates (iterator is not a singleton).
    assert list(c.stream("again")) == MOCK_TOKENS


def test_stream_is_an_iterator():
    c = make_client()
    it = c.stream("Hello")
    assert iter(it) is it
    first = next(it)
    assert first == MOCK_TOKENS[0]


def test_stream_raises_stop_iteration_when_exhausted():
    c = make_client()
    it = c.stream("Hi")
    for _ in MOCK_TOKENS:
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_dropping_iterator_early_stops_generation():
    # Pull a single token, then drop the iterator before it is exhausted. The
    # underlying core BoxStream must be dropped, cancelling generation. We can't
    # observe the Rust cancellation flag from Python, but we CAN assert the
    # early drop does not raise and that a fresh stream still works afterwards —
    # i.e. the runtime/channel machinery survives an abandoned iterator.
    c = make_client()
    it = c.stream("Explain Rust ownership")
    first = next(it)
    assert first == MOCK_TOKENS[0]
    del it
    gc.collect()
    # The client is still usable: a new full stream completes normally.
    assert list(c.stream("again")) == MOCK_TOKENS


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError):
        Client(backend="nope", model="x")


def test_empty_model_raises_value_error():
    with pytest.raises(ValueError):
        Client(backend="mock", model="   ")


def test_custom_exceptions_are_exported():
    assert issubclass(llm_stream.ModelNotFoundError, llm_stream.LlmStreamError)
    assert issubclass(llm_stream.BackendError, llm_stream.LlmStreamError)
    assert issubclass(llm_stream.LlmStreamError, RuntimeError)
