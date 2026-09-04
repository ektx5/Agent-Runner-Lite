"""The tools the agent can call, and the toy world they operate on.

The domain is deliberately boring — a handful of CRM-ish contacts. Nobody is being assessed on
contact management. What matters is the machinery around the tools: which ones are allowed to run,
whether they can be safely retried, and whether we can prove afterwards what they did.

PROVIDED: Workspace read tools, update_contact, ToolDef, build_registry.
TASK 4b:  send_message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.models import ToolKind


class ToolError(Exception):
    """A tool failed in a way the agent might be able to work around.

    Important: the agent must NOT crash when it sees one of these. It feeds the error back to the
    model as an observation ("that contact doesn't exist") so the model gets a chance to try
    something else. A real agent hits these constantly — a stale id, a missing field — and a loop
    that dies on the first one is useless.
    """


class Workspace:
    """The in-memory world the tools act on. One per run, so runs can't interfere with each other."""

    def __init__(self, contacts: list[dict[str, Any]] | None = None):
        self.contacts: dict[str, dict[str, Any]] = {c["id"]: dict(c) for c in (contacts or [])}
        self.messages: list[dict[str, Any]] = []
        # Remembers the result of each idempotency key we've already handled. See send_message.
        self._idem: dict[str, dict[str, Any]] = {}

    # --- read tools: safe, repeatable, no governance needed ---

    def search_contacts(self, query: str = "", **_: Any) -> list[dict[str, Any]]:
        q = (query or "").lower()
        return [
            {k: c[k] for k in ("id", "name", "email", "stage")}
            for c in self.contacts.values()
            if q in c["name"].lower() or q in c.get("email", "").lower()
        ]

    def get_contact(self, contact_id: str = "", **_: Any) -> dict[str, Any]:
        contact = self.contacts.get(contact_id)
        if contact is None:
            raise ToolError(f"contact {contact_id!r} not found")
        return dict(contact)

    # --- write tools: these change the world, so the governance rules apply to them ---

    def update_contact(
        self, contact_id: str = "", fields: dict | None = None, **_: Any
    ) -> dict[str, Any]:
        """PROVIDED — your worked example of a write tool.

        Note the shape: validate, raise ToolError if the world isn't what we expected, mutate,
        return a plain dict. `send_message` below follows the same shape, plus one extra idea.
        """
        contact = self.contacts.get(contact_id)
        if contact is None:
            raise ToolError(f"contact {contact_id!r} not found")
        contact.update(fields or {})
        return dict(contact)

    def send_message(
        self, contact_id: str = "", body: str = "", idempotency_key: str = "", **_: Any
    ) -> dict[str, Any]:
        """TASK 4b — TODO(candidate): make sending a message safe to call twice.

        Here's the problem this solves. Sending a message is not like updating a field: if an
        update runs twice you end up in the same place, but if a *send* runs twice the customer
        gets two emails. And calls do get repeated — a retry after a timeout, a duplicated
        request, a bug in a loop. So the tool itself has to refuse to do the work twice.

        The standard fix is an idempotency key: the caller passes a string that identifies THIS
        PARTICULAR send. The tool remembers which keys it has already handled. A repeat of a key
        it has seen returns the original result instead of doing anything.

        The agent already generates a stable key per (run, step) for you — see `_execute` in
        app/agent.py. Your job is the remembering.

        What to do:
          - if `contact_id` or `idempotency_key` is missing, raise ToolError. A key is not
            optional; without one the caller can't be protected.
          - if the contact doesn't exist, raise ToolError (same as update_contact).
          - if `idempotency_key` is already in `self._idem`: return the stored result, but with
            "deduped": True, and do NOT append to self.messages.
          - otherwise: append the message to `self.messages`, build the result, store it in
            `self._idem` under the key, and return it with "deduped": False.
          - the returned dict should include at least "message_id" and "contact_id".

        A message id like f"m_{len(self.messages)}" is fine.

        The test that matters: call it twice with the same key, then assert
        `len(ws.messages) == 1`. Checking the return value alone isn't enough — the bug you're
        guarding against is the second message existing, so assert on the world, not the
        response. Write that test first.
        """
        if not contact_id:
            raise ToolError("contact_id is required")
        if not idempotency_key:
            raise ToolError("idempotency_key is required")
        if contact_id not in self.contacts:
            raise ToolError(f"contact {contact_id!r} not found")

        # Deduplicate: if we've seen this key before, return the cached result
        if idempotency_key in self._idem:
            return {**self._idem[idempotency_key], "deduped": True}

        # First time seeing this key: actually send
        result = {
            "message_id": f"m_{len(self.messages)}",
            "contact_id": contact_id,
            "deduped": False,
        }
        self.messages.append({"contact_id": contact_id, "body": body, "idempotency_key": idempotency_key})
        self._idem[idempotency_key] = result
        return result


@dataclass(frozen=True)
class ToolDef:
    """One entry in the registry: what the tool is called, whether it writes, and how to run it.

    `kind` is the field the governance policy reads. It's the only thing separating "look
    something up" from "change the world" as far as the gate is concerned.
    """

    name: str
    kind: ToolKind
    func: Callable[..., Any]
    description: str = ""


def build_registry(ws: Workspace) -> dict[str, ToolDef]:
    """PROVIDED. Maps tool names to their definitions.

    The agent looks tools up here by the name the model asked for. A name that isn't in this dict
    is an "unknown tool" — which happens for real, because a model can hallucinate a tool that
    sounds plausible. Your loop has to handle that without falling over.
    """
    return {
        "search_contacts": ToolDef(
            "search_contacts", "read", ws.search_contacts, "Find contacts by name or email."
        ),
        "get_contact": ToolDef("get_contact", "read", ws.get_contact, "Fetch one contact by id."),
        "update_contact": ToolDef(
            "update_contact", "write", ws.update_contact, "Update fields on a contact."
        ),
        "send_message": ToolDef(
            "send_message", "write", ws.send_message, "Send a message to a contact (idempotent)."
        ),
    }
