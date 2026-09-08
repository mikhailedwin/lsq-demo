"""Engine → API status/transcript callbacks (best effort, never block the pipeline)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .persona import Persona

logger = logging.getLogger("qav.callbacks")


class ApiCallbacks:
    def __init__(self, persona: Persona) -> None:
        self._url = f"{persona.callback_url.rstrip('/')}/v1/internal/events" if persona.callback_url else ""
        self._headers = {"authorization": f"Bearer {persona.callback_token}"}
        self._client = httpx.AsyncClient(timeout=5.0)
        self._tasks: set[asyncio.Task[None]] = set()

    def status(self, status: str, error: str | None = None) -> None:
        body: dict[str, Any] = {"type": "status", "status": status}
        if error:
            body["error"] = error[:2000]
        self._post(body)

    def transcript(self, role: str, content: str) -> None:
        if content.strip():
            self._post({"type": "transcript", "role": role, "content": content})

    def _post(self, body: dict[str, Any]) -> None:
        if not self._url:
            return

        async def _send() -> None:
            try:
                r = await self._client.post(self._url, json=body, headers=self._headers)
                if r.status_code >= 400:
                    logger.warning("callback %s rejected: %s %s", body["type"], r.status_code, r.text[:200])
            except Exception as e:  # noqa: BLE001
                logger.warning("callback %s failed: %s", body["type"], e)

        t = asyncio.create_task(_send())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=5.0)
        await self._client.aclose()
