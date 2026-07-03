"""
api.py
------
FastAPI wrapper exposing the agent as a POST /ask endpoint.

Start with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import ask

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Well Log & Seismic Q&A Agent",
    description=(
        "A LangGraph-powered agent that answers natural-language questions about "
        "oil & gas well log (LAS) and seismic (SEG-Y) data files."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="Natural-language question about well log or seismic data.",
        examples=["What is the average porosity in Well-Alpha between 2000 and 2500 m?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Well Log & Seismic Q&A Agent"}


@app.post("/ask", response_model=AskResponse, summary="Ask a question about well or seismic data")
def ask_endpoint(body: AskRequest):
    """
    Submit a natural-language question.  The agent will:
    1. Discover which data file your question refers to.
    2. Call the relevant tools to compute the answer from the real data.
    3. Return a precise, cited answer.

    **Example questions**:
    - *"What is the average GR in Well-Alpha between 2000 and 2500 m?"*
    - *"What are the amplitude stats at inline 105, crossline 60 in Survey-Apex between 200 and 400 ms?"*
    - *"Flag intervals where resistivity in Well-Beta exceeds 500 Ohm·m"*
    """
    t0 = time.perf_counter()
    try:
        answer = ask(body.question)
    except Exception as exc:
        logger.exception("Agent error for question: %s", body.question)
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        question=body.question,
        answer=answer,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
    )


@app.get("/tools", summary="List available agent tools")
def list_tools():
    """Return the names and descriptions of all registered tools."""
    from agent import _AGENT
    tools_info = []
    for t in _AGENT.tools:                         # type: ignore[attr-defined]
        tools_info.append({"name": t.name, "description": t.description})
    return {"tools": tools_info}
