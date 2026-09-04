# Agent Runner Lite — Intern Take-Home

Thanks for taking the time on this. You'll build a small **governed agent runner**: a service that
drives an AI agent through a tool-use loop, decides what the agent is allowed to do on its own, and
then checks that it actually did what was asked — and nothing more.

The full brief — the six tasks, what we look for, and the ground rules — is in **`BRIEF.md`**.
**Read that first.** This file is just how to run things, plus a map of the code, and it's where you
write up your work when you're done.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                                             # the example tests pass on a fresh checkout
uvicorn app.main:app --reload                         # http://127.0.0.1:8000/docs
```

Requires **Python 3.11+**. No API keys, no network, no external services — the language model is a
script (`app/seed.py`), so everything is deterministic and offline.

On a fresh checkout the app imports and `pytest` is green, but the six functions you're implementing
raise `NotImplementedError` (and `POST /runs` returns a 501). That's expected. Search the project for
`TODO(candidate)` to find them — there are six, numbered by task.

Nothing is persisted. Restart the server and your tasks and runs are gone. That's fine, don't work
around it.

## Where things are

```
app/
  models.py          the whole data contract — READ THIS FIRST, it's the map
  config.py          settings, all overridable by env var
  seed.py            toy contacts + 11 scripted model conversations
  store.py           two dicts standing in for a database
  model_client.py    TASK 4a — complete_with_retry     (mock model provided)
  tools.py           TASK 4b — send_message            (read tools + update_contact provided)
  autonomy.py        TASK 1  — evaluate_gate
  verifier.py        TASK 2  — verify
  agent.py           TASK 3  — run_agent               (five helpers provided)
  api.py             TASK 5  — start_run               (three other routes provided)
  main.py            the FastAPI app
tests/
  helpers.py         make_run() — builds an isolated run in one line
  conftest.py        store reset + a `client` fixture for HTTP tests
  test_example.py    four example tests showing the shapes you'll want
```

## A suggested first hour

If you're not sure where to start:

1. `pip install -r requirements.txt && pytest -q`. Green? Good.
2. Read **`app/models.py`** top to bottom. It's commented and it's the whole data model — most of
   the "what shape do I return?" questions are answered there.
3. Read **`app/seed.py`** to see what the mock model does. This is the trick that makes the whole
   thing testable, and it's worth understanding before anything else.
4. Open **`app/autonomy.py`** (Task 1). Write the tests for it first — `test_example.py` has a
   `parametrize` example to copy. Watch them fail, then make them pass.
5. Then `app/verifier.py` (Task 2), same way.

By then you'll have the shape of the codebase and two of the six tasks done.

## Useful to know

- **The scenarios in `app/seed.py` are your test fixtures.** There's one for each path you need to
  handle: `send_followup` (the happy path), `unknown_tool` (a hallucinated tool name),
  `tool_error` (a tool that fails), `three_writes` (the autonomy budget), `never_finishes` (the
  `max_steps` cap), `flaky_provider` (throttled then fine), `bad_credentials` (fatal, don't retry),
  `bad_json_then_good` and `always_bad_json` (malformed model output). Read the comments there.
- **`tests/helpers.py::make_run`** gives you a Run and its dependencies in one line, isolated. Use it
  for every loop test.
- **Everything is synchronous.** Plain `def`, `time.sleep`, no `await` anywhere. If you find
  yourself reaching for `asyncio`, you've gone off the path.
- **Settings are injected, not global.** Your functions take `settings`, so a test can say "budget
  of 1, retry twice" without touching the environment:
  `make_run(script, settings=replace(SETTINGS, max_auto_writes=1))`.
- Once Task 5 is done, `http://127.0.0.1:8000/docs` gives you a UI to create a task and start a run
  without writing any curl. Good for a sanity check that pytest can't give you.

---

# Your write-up

### What's working

All six tasks are fully implemented and all 40 tests pass (`pytest -q` is green).

| Task | Status |
|------|--------|
| **1** — `evaluate_gate` (governance policy) | ✅ Complete |
| **2** — `verify` (behaviour-equivalence check) | ✅ Complete |
| **3** — `run_agent` (agent loop) | ✅ Complete |
| **4a** — `complete_with_retry` (exponential backoff) | ✅ Complete |
| **4b** — `send_message` (idempotent write tool) | ✅ Complete |
| **5** — `POST /runs` (start-run endpoint) | ✅ Complete |
| **6** — Tests | ✅ 40 tests across 4 test files |

Nothing is half-finished or knowingly broken.

---

### Design decisions

**The agent loop (`run_agent`)**

The loop runs for at most `max_steps` turns. On each turn it:
1. Asks the model for an intent via `_decide` (which already handles re-prompting on bad JSON).
2. If the intent is `final`, the run completes.
3. If the tool name isn't in the registry, it emits a failed `tool_result` step and appends an observation — the model gets another turn. It does **not** crash, because hallucinated tool names are normal.
4. Every tool call (read or write) goes through `evaluate_gate` first, and a `gate` step is always emitted so the audit trail is complete.
5. If approval is required and denied, that is also fed back as an observation — the model can decide to do something else or finish.
6. A successful write appends to `run.effects` and increments `writes_done`. Both matter: `effects` is what the verifier reads; `writes_done` is what the gate's budget counts.

**What ends a run vs. what becomes an observation**

Only two things truly end a run: the model saying `final`, or something unrecoverable (a `FatalError` from the model, or hitting `max_steps`). Everything else — unknown tool, tool error, reviewer rejection — becomes an observation that is appended to the message history. The model gets a chance to respond to the error. This is the key resilience property the brief asks for.

**Idempotency key handling**

`_execute` in `agent.py` (provided) generates a stable key of the form `"{run_id}:{step_count}"` and injects it into `send_message` args. `send_message` in `tools.py` stores results in `self._idem` keyed by that string. On a repeat call with the same key, it returns the stored result with `"deduped": True` and does **not** append to `self.messages`. This means the key lives on the `Workspace` (scoped to one run), which is exactly right — a retry within the same run deduplicates correctly, while a fresh run starts with a clean `_idem`.

**Ambiguities resolved**

- The brief says "if retries run out, let the last ThrottleError propagate" — I treat exhausted retries and a final throttle error as the same: let it raise, the outer `try/except` in `run_agent` catches it, marks the run `failed`, and stores the message in `run.error`. No special-casing needed.
- Shadow mode: the gate always returns `simulate=True` for writes. `_execute` short-circuits and returns a sentinel dict without calling the real tool. `run.effects` still gets the `Effect` (with `simulated=True`) so the verifier can do its job identically in both modes.

---

### Testing approach

**Task 1 and Task 2 tests were written before the implementations** — the commit history (`Task 2: tests first for verifier` precedes `Task 2: implement verifier logic`) confirms this. The tests were run first to watch them fail with `NotImplementedError`, then the implementations were written to make them pass.

**What was tested:**

- `test_autonomy.py` — 10 parametrized cases covering every branch of the decision table, including the exact budget boundary (`writes_so_far == max_auto_writes` requires approval, `writes_so_far < max_auto_writes` is allowed).
- `test_verifier.py` — 5 cases: all match, missing effect, unexpected effect, simulated shadow effects, and the "one effect can only satisfy one expectation" rule (Rule 2).
- `test_resilience.py` — retry (throttled-twice-then-success asserts `model.calls == 3`; fatal-not-retried asserts `model.calls == 1`), idempotency (asserts on `ws.messages` length, not the return value), 6 agent-loop scenarios, and 2 end-to-end HTTP tests via the `client` fixture.
- `test_agent.py` — 5 focused loop tests (no tools, one write, shadow mode, unknown tool recovery, never-finishes).

**What was deliberately not tested:**

- Every permutation of the `three_writes` scenario under all autonomy levels — the gate is already unit-tested exhaustively in `test_autonomy.py`, so testing it again through the full loop would not catch new bugs.
- `bad_json_then_good` / `always_bad_json` — `_decide`'s re-prompting logic is provided code; testing it would just be testing the scaffold. The `never_finishes` test already exercises the `max_steps` path which is the relevant failure mode.
- The `flaky_provider` scenario end-to-end — covered adequately by the direct `complete_with_retry` unit tests.

---

### What was hardest

**Rule 2 in the verifier** — "each actual effect can only be used to satisfy one expectation" — was the subtlest part. The naive implementation loops over expectations and searches `run.effects` each time, which means a single real effect satisfies every expectation that matches it. The fix is to work off a mutable copy of `available_effects` and `pop` the matching entry when it is consumed. Writing the test for it first (two identical expectations, one real effect → 1 matched, 1 missing) made the bug immediately obvious and the fix clear.

**The gate step ordering in the loop** — I initially emitted the `gate` step after running the tool, which would have put things out of order in the audit trail. Re-reading the pseudocode in the docstring clarified that the gate step should be emitted immediately after the decision, before execution, so the trail reads: `gate → tool_call → tool_result`.

**Understanding `_execute`'s simulate path** — in shadow mode, `_execute` returns early with a synthetic result and never calls the real tool. This means `idempotency_key` is never injected (because the key injection only happens for the real `send_message` call). The `Effect` stored in shadow mode therefore doesn't include the key in its `args`. I made sure the verifier tests use `match` dicts that don't include the key, so shadow and real runs are verified identically — which is the whole point.

---

### What I'd do next

With another day I would tackle the optional extensions from the brief:

1. **Per-tool gate override** — add an optional `requires_approval: bool` field to `ToolDef` so that `send_message` can be forced to always require approval regardless of autonomy level. The gate would check this before the level-based logic.
2. **Structured logging** — replace the `Step` list with a `structlog` call on each `_emit`, using `run.id` as a correlation id. This makes the audit trail queryable without changing the data model.
3. **Summary line in `Verdict.detail`** — currently `detail` is `"N matched, N missing, N unexpected"`. A more useful version would name the specific missing expectations, e.g. `"missing: send_message to c_1"`.
4. **Async runs** — the synchronous model is fine for this exercise but in production a run takes minutes. I'd move `run_agent` to a background task (e.g. with FastAPI's `BackgroundTasks` or a queue), return a `202 Accepted` immediately, and poll `GET /runs/{id}` for status.

---

### Time spent

Approximately 4–5 hours total: ~30 min reading the scaffold and planning, ~3 hours implementing tasks 1–6 in order, ~1 hour writing and fixing tests.
