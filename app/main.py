"""
FastAPI co-pilot service.

Endpoints:
  GET  /health                                health check + Neo4j ping
  POST /turn   (JSON: {conversation_id, text})  text-only turn
  POST /audio  (multipart: conversation_id, file)  full STT + pipeline
  WS   /ws/{conversation_id}                  streaming text turns over websocket

The pipeline per turn:
  text -> interpret + sentiment (parallel) -> state -> plan -> llm -> reply
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents import interpreter, llm, planner, sentiment, state, stt
from app.core.config import settings
from app.core.schemas import CopilotResponse
from app.graph.driver import close_driver, session as graph_session

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("copilot")

app = FastAPI(title="Customer-Care AI Co-pilot", version="0.1.0")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(_STATIC_DIR / "index.html")


class TurnIn(BaseModel):
    conversation_id: str | None = None
    text: str


@app.on_event("startup")
async def _startup():
    try:
        with graph_session() as s:
            s.run("RETURN 1").consume()
        log.info("Neo4j ready at %s", settings.neo4j_uri)
    except Exception as e:
        log.warning("Neo4j not reachable at %s: %s", settings.neo4j_uri, e)


@app.on_event("shutdown")
async def _shutdown():
    close_driver()


@app.get("/health")
async def health():
    out = {"status": "ok"}
    try:
        with graph_session() as s:
            s.run("RETURN 1").consume()
        out["neo4j"] = "up"
    except Exception as e:
        out["neo4j"] = f"down: {e}"
    return out


async def _process_turn(conversation_id: str, text: str) -> CopilotResponse:
    interp_task = asyncio.to_thread(interpreter.interpret, text)
    sent_task = asyncio.to_thread(sentiment.analyze, text)
    interp, sent = await asyncio.gather(interp_task, sent_task)

    st = state.build_state(conversation_id, text, interp, sent)
    plan = await asyncio.to_thread(planner.plan, st)

    try:
        reply = await llm.generate(st, plan)
        used_fallback = plan.used_fallback
    except Exception as e:
        log.exception("LLM call failed; returning planner-only fallback: %s", e)
        if plan.next_agent_tags:
            tags = ", ".join(plan.next_agent_tags)
            reply = f"[fallback] Suggested next action(s): {tags}."
        else:
            reply = "[fallback] Acknowledge the customer and ask one clarifying question."
        used_fallback = True

    return CopilotResponse(state=st, plan=plan, suggested_reply=reply, used_fallback=used_fallback)


@app.post("/turn", response_model=CopilotResponse)
async def turn(body: TurnIn):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    cid = body.conversation_id or str(uuid.uuid4())
    return await _process_turn(cid, body.text)


@app.post("/audio", response_model=CopilotResponse)
async def audio(
    conversation_id: str | None = Form(default=None),
    file: UploadFile = File(...),
):
    log.info("/audio endpoint hit — filename=%s, content_type=%s", file.filename, file.content_type)
    raw = await file.read()
    log.info("Audio file read, %d bytes", len(raw))
    text = await asyncio.to_thread(stt.transcribe, raw)
    if not text.strip():
        raise HTTPException(status_code=422, detail="empty transcription")
    cid = conversation_id or str(uuid.uuid4())
    return await _process_turn(cid, text)


@app.websocket("/ws/{conversation_id}")
async def ws(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            text = (msg or {}).get("text", "")
            if not text:
                await websocket.send_json({"error": "missing text"})
                continue
            res = await _process_turn(conversation_id, text)
            await websocket.send_json(res.model_dump())
    except WebSocketDisconnect:
        return


@app.post("/reset/{conversation_id}")
async def reset(conversation_id: str):
    state.reset(conversation_id)
    return {"ok": True, "conversation_id": conversation_id}
