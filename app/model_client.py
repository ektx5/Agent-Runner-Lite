"""The seam between our code and a foundation model.

We never import a vendor SDK. We define the smallest interface we need (`ModelClient`) and code
against that. A real adapter for Anthropic, OpenAI or Bedrock would implement the same one method.
For this exercise the only implementation is a scripted mock, so everything runs offline with no
API keys, no network, and no cost — and every test gives the same answer every time.

PROVIDED: ModelClient, ModelError/ThrottleError/FatalError, MockModelClient.
TASK 4:   complete_with_retry.
"""

from __future__ import annotations

import time
from typing import Protocol, Union, runtime_checkable

from app.config import SETTINGS, Settings


class ModelError(Exception):
    """Base class for anything that goes wrong talking to the model."""


class ThrottleError(ModelError):
    """TRANSIENT. Rate limited, or the provider had a wobble (HTTP 429 / 5xx).

    Worth retrying: the same request may well succeed a moment later.
    """


class FatalError(ModelError):
    """PERMANENT. A malformed request, a bad API key, a model that doesn't exist (HTTP 400 / 401).

    Not worth retrying: it will fail identically every time, and retrying burns time and quota
    while the caller waits for an answer that isn't coming.
    """


@runtime_checkable
class ModelClient(Protocol):
    """Anything with this method can be used as our model."""

    def complete(self, messages: list[dict]) -> str:  # returns the model's raw text
        ...


class MockModelClient:
    """A model that reads from a script instead of thinking.

    `responses` is consumed one entry per `complete()` call:
      - a string is returned as the model's reply;
      - an Exception instance is RAISED.

    That second part is the useful bit for testing: a script of
        [ThrottleError("429"), '{"intent": "final", "answer": "ok"}']
    lets you test your retry logic without a flaky network. `calls` counts invocations, so a test
    can assert exactly how many times the model was hit.
    """

    def __init__(self, responses: list[Union[str, Exception]]):
        self._responses = list(responses)
        self._cursor = 0
        self.calls = 0

    def complete(self, messages: list[dict]) -> str:
        self.calls += 1
        if self._cursor >= len(self._responses):
            # Script ran out. Return a safe "we're done" so a buggy loop can't hang forever.
            return '{"intent": "final", "answer": "done"}'
        item = self._responses[self._cursor]
        self._cursor += 1
        if isinstance(item, Exception):
            raise item
        return item


def complete_with_retry(
    model: ModelClient, messages: list[dict], settings: Settings = SETTINGS
) -> str:
    """TASK 4a — TODO(candidate): call the model, and survive a transient failure.

    Networks and model providers are unreliable. Some failures are worth retrying and some are
    not, and telling them apart is the whole skill here.

    What to do:
      - call `model.complete(messages)` and return its text on success.
      - on ThrottleError: wait, then try again — up to `settings.model_max_retries` retries.
        Sleep `settings.model_backoff_base_seconds * (2 ** attempt)` before each retry, where
        `attempt` starts at 0. That doubling is called exponential backoff: if the provider is
        overloaded, hammering it every 10ms makes things worse for everyone, so each wait is
        longer than the last.
      - if the retries run out, let the last ThrottleError propagate. Don't swallow it and don't
        return None — the caller needs to know this failed.
      - on FatalError (and any other exception): raise straight away, no retry.

    Use `time.sleep` — this whole service is synchronous, there's no async here.

    Test it by scripting the mock:
        MockModelClient([ThrottleError("429"), ThrottleError("429"), '{"intent":"final","answer":"hi"}'])
      → returns the JSON, and `model.calls == 3`
        MockModelClient([FatalError("bad key"), '{"intent":"final","answer":"hi"}'])
      → raises FatalError, and `model.calls == 1` (this is the assertion that proves you did NOT
        retry the fatal error — a test that only checks "it raised" would pass even if you
        retried three times first)

    Write those two tests before you write the function.
    """
    for attempt in range(settings.model_max_retries + 1):
        try:
            return model.complete(messages)
        except ThrottleError:
            if attempt >= settings.model_max_retries:
                raise
            time.sleep(settings.model_backoff_base_seconds * (2 ** attempt))
        except FatalError:
            raise
