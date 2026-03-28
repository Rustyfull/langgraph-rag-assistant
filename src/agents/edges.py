"""
Conditional edge functions for the LangGraph workflow.

Each function receives the current AgentState and returns
a string key that LangGraph uses to route to the next node.
"""

from __future__ import annotations

from src.agents import AgentState
from src.agents.nodes import generate
from src.utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

def route_after_grading(state:AgentState) -> str:
    """After document grading: route to web_search or generate.

    Returns
    --------
    "web_search"    - if grader flagged insufficient docs
    "generate"      - if enough relevant docs exist
    """
    if state.get("web_search_needed",False):
        logger.info("[Edge] route_after_grading → web_search")
        return "web_search"
    logger.info("[Edge] route_after_grading -> generate")
    return "generate"


def route_after_hallucination_check(
        state:AgentState
) -> str:
    """
    After hallucination check: decide whether to accept or retry.
    
    Logic:
    - If grounded AND addresses question -> "accept"
    - If not grounded -> "regenerate" (up to MAX_RETRIES)
    - If not addressing question -> "web_search" for more context
    - If max retries exceeded -> force "accept" to avoid infinite loop


    Returns:
    ---------
    "accept"        - final answer is good
    "regenerate"    - re-run generate node
    "web_search"    - fetch more context, then regenerate
    """
    retry_count = state.get("retry_count",0)

    if retry_count >= settings.max_retries:
        logger.warning(
            "[Edge] Max retries (%d) reached - forcing accept", settings.max_retries
        )
        return "accept"

    hallucination_score = state.get("hallucination_score","no")
    answer_score = state.get("answer_addresses_question", "no")

    if hallucination_score == "yes" and answer_score == "yes":
        logger.info("[Edge] route_after_hallucination_check → accept ✓")
        return "accept"

    if hallucination_score == "no":
        logger.info(
            "[Edge] route_after_hallucination_check → regenerate (not grounded)"
        )
        return "regenerate"
    
    # grounded but doesn't address the question -> need more context
    logger.info(
        "[Edge] route_after_hallucination_check + web_search (answer off-topic)"
    )
    return "web_search"


