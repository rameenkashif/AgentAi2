"""
agent.py
--------
LangGraph ReAct agent that answers natural-language questions about
well log (LAS) and seismic (SEG-Y) data using the tools defined in tools.py.

Usage (CLI):
    python agent.py "What is the average GR in Well-Alpha between 2000 and 2500 m?"
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# Logging — visible during demos (tool calls printed to console)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Copy .env.example → .env and fill in your key."
    )

# Resolve data directory once
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_HERE, "data"))

# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------
_LLM = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=ANTHROPIC_API_KEY,
    temperature=0,
    max_tokens=2048,
)

_SYSTEM_PROMPT = f"""You are an expert oil & gas data analyst assistant.
You have access to tools that read real well log (LAS) and seismic (SEG-Y) data files.

IMPORTANT RULES:
1. ALWAYS start by calling list_wells and/or list_seismic_surveys (data_dir="{DATA_DIR}") \
to discover which files exist before you try to read any specific file.
2. Match the well name or survey name from the user's question to the real filename \
returned by the discovery tools. Never guess a filename.
3. If the user mentions a well or survey name that cannot be found in the discovered \
files, ask for clarification rather than proceeding.
4. Every numeric answer MUST come from a real tool call. Never fabricate numbers.
5. If a question involves both well log and seismic data, call both relevant tools \
and combine the results in your final answer.
6. Provide concise, precise answers suited to a geoscience/engineering audience. \
Include units in every numeric result.
"""

_AGENT = create_react_agent(_LLM, ALL_TOOLS, prompt=_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Public ask() function
# ---------------------------------------------------------------------------

def ask(question: str) -> str:
    """
    Submit a natural-language question to the agent and return the answer as a string.
    Tool calls are logged to stdout so the reasoning is visible during demos.
    """
    logger.info("=" * 60)
    logger.info("QUESTION: %s", question)
    logger.info("=" * 60)

    result = _AGENT.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])

    # Log every tool call
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info("  ► TOOL CALLED: %s | args: %s",
                            tc["name"],
                            json.dumps({k: v for k, v in tc["args"].items()
                                        if k != "data_dir"}, default=str))
        elif isinstance(msg, ToolMessage):
            content_preview = str(msg.content)[:300]
            logger.info("  ◄ TOOL RESULT (%s): %s%s",
                        msg.name,
                        content_preview,
                        "…" if len(str(msg.content)) > 300 else "")

    # The last AI message is the final answer
    final = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final = msg.content
            break

    answer = final if isinstance(final, str) else str(final)
    logger.info("ANSWER: %s", answer)
    return answer


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<your question>\"")
        sys.exit(1)
    question_text = " ".join(sys.argv[1:])
    print("\n" + ask(question_text))
