"""The HTTP surface. FastAPI routes.

Four endpoints. Three are provided as worked examples; the fourth is TASK 5, and it's the one that
makes the whole thing runnable from a browser.

Everything here is synchronous — plain `def`, not `async def`. FastAPI handles both, and running a
whole agent loop inline in the request is perfectly reasonable at this scale. (In production a run
takes minutes and goes on a queue, which is why the real service is async. Out of scope here.)
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent import AgentDeps, run_agent
from app.config import SETTINGS
from app.model_client import MockModelClient
from app.models import Run, Task, TaskSpec
from app.seed import CONTACTS, script_for
from app.store import store
from app.tools import Workspace, build_registry

router = APIRouter()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@router.post("/tasks", response_model=Task)
def create_task(spec: TaskSpec) -> Task:
    """PROVIDED. Note how little there is to it: FastAPI validated the body into a TaskSpec
    before this function was even called, using the model in app/models.py."""
    task = Task(id=_new_id("t"), **spec.model_dump())
    store.add_task(task)
    return task


@router.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: str) -> Task:
    """PROVIDED — your worked example of a path parameter and a 404."""
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    return task


class StartRunBody(BaseModel):
    task_id: str


@router.post("/runs", response_model=Run, status_code=201)
def start_run(body: StartRunBody) -> Run:
    """TASK 5 — TODO(candidate): start a run and return the finished result.

    Small, but it's the piece that ties everything together — with this done you can drive your
    agent from the interactive docs at /docs instead of only from pytest.

    What to do:

      1. Look up the task with `store.get_task(body.task_id)`. If it doesn't exist, raise
         `HTTPException(404, "task not found")`. Follow `get_task` above for the pattern.

      2. Build the Run:
             run = Run(id=_new_id("r"), task_id=task.id, autonomy=task.autonomy)
         and register it with `store.add_run(run)`.

      3. Build the dependencies. Give every run its OWN workspace, so one run's messages and
         contact edits can't leak into another's:

             ws = Workspace(CONTACTS)
             deps = AgentDeps(
                 model=MockModelClient(script_for(task.scenario)),
                 workspace=ws,
                 registry=build_registry(ws),
                 store=store,
                 settings=SETTINGS,
             )

         Note `build_registry(ws)` is built from that same workspace — the registry holds bound
         methods, so a registry made from a different Workspace would write to the wrong world.

      4. Call `run_agent(run, deps)` and return the run. Because this is synchronous, the loop has
         finished by the time you return, so the response contains the whole trace: every step, the
         effects, and the verdict.

    Check it by hand once it works:
        curl -s localhost:8000/api/v1/tasks -H 'content-type: application/json' \\
          -d '{"goal":"follow up with Acme","scenario":"send_followup","autonomy":"autonomous",
               "expected_effects":[{"tool":"send_message","match":{"contact_id":"c_1"}}]}'
        curl -s localhost:8000/api/v1/runs -H 'content-type: application/json' \\
          -d '{"task_id":"<the id you just got>"}'
    """
    task = store.get_task(body.task_id)
    if task is None:
        raise HTTPException(404, "task not found")

    run = Run(id=_new_id("r"), task_id=task.id, autonomy=task.autonomy)
    store.add_run(run)

    ws = Workspace(CONTACTS)
    deps = AgentDeps(
        model=MockModelClient(script_for(task.scenario)),
        workspace=ws,
        registry=build_registry(ws),
        store=store,
        settings=SETTINGS,
    )

    return run_agent(run, deps)


@router.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> Run:
    """PROVIDED. Returns a run and everything that happened during it."""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run
