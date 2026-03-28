"""
LangGraph workflow definition — Corrective RAG (CRAG) pattern.

Graph topology:
                         ┌──────────────────────┐
                         │      START            │
                         └──────────┬───────────┘
                                    │
                              [retrieve]
                                    │
                           [grade_documents]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              [web_search]                    [generate]
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                            [generate] (if from web)
                                    │
                       [check_hallucination]
                                    │
               ┌────────────────────┼──────────────────────┐
               │                    │                       │
          [accept]            [regenerate]            [web_search]
               │                    │                       │
             END              [generate]             [generate]
                                    │                       │
                         [check_hallucination]   [check_hallucination]
                                    └───────────────────────┘
"""


from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.edges import route_after_grading, route_after_hallucination_check
from src.agents.nodes import ( check_hallucination, generate, grade_documents,retrieve, web_search)
from src.agents.state import AgentState

def build_graph() -> StateGraph:
    """Build and compile the CRAG LangGraph workflow."""

    builder = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    builder.add_node("retrieve",retrieve)
    builder.add_node("grade_documents",grade_documents)
    builder.add_node("web_search", web_search)
    builder.add_node("generate",generate)
    builder.add_node("check_hallucination", check_hallucination)

    # ── Define edges ───────────────────────────────────────────────────────
    # Entry point
    builder.add_edge(START,"retrieve")
    builder.add_edge("retrieve","grade_documents")
    
    # After grading: either web search or generate
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "web_search":"web_search",
            "generate":"generate"
        },

    )

    # Web search always leads to generation
    builder.add_edge("web_search", "generate")

    # After generation quality check
    builder.add_edge("generate", "check_hallucination")

    # After quality check: accept, retry, or search for more context
    builder.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "accept":END,
            "regenerate":"generate",
            "web_search":"web_search"
        }
    )

    return builder.compile()


# Module-level singleton
graph = build_graph()


def run_query(question:str) -> dict:
    """
    Run a question through the RAG graph

    Args:
        question: The user's question string.

    Returns:
        Final AgentState with 'generation' containing the answer.
    """
    initial_state: AgentState = {
        "question":question,
        "documents":[],
        "generation":"",
        "web_search_needed":False,
        "hallucination_score":"no",
        "answer_addresses_question":"no",
        "retry_count":0
    }
    final_stage = graph.invoke(initial_state)
    return final_stage
    

if __name__ == "__main__":
    print(run_query("What is Langchain ?"))

    
    