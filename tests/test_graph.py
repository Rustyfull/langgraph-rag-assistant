"""
Tests for the LangGraph RAG workflow.
Uses mocking to avoid requiring real API keys in CI.
"""

from __future__ import annotations

from mailbox import Mailbox
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


from src.agents.state import  AgentState

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="The attention mechanism allows models to focus on relevant parts og the input sequence.",
            metadata={
                "source":"arxiv:1706.03762",
                "source_type":"arxiv"
            }

        ),

        Document(
                page_content="Self-attention computes a weighted sum of all positions in the sequence.",
                metadata={"source": "arxiv:1706.03762", "source_type": "arxiv"},
        )
    ]



@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        question="What is the attention mechanism?",
        documents=[],
        generation="",
        web_search_needed=False,
        hallucination_score="no",
        answer_addresses_question="no",
        retry_count=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tests: nodes
# ─────────────────────────────────────────────────────────────────────────────
class TestRetrieveNode:
    @patch("src.agents.nodes._get_retriever")
    def test_retrieve_returns_documents(self, mock_get_retriever, base_state, sample_docs):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = sample_docs
        mock_get_retriever.return_value = mock_retriever

        from src.agents.nodes import retrieve
        result = retrieve(base_state)

        assert  "documents" in result
        assert len(result["documents"]) == 2
        mock_retriever.retrieve.assert_called_once_with(base_state["question"])
        
        
    @patch("src.agents.nodes._get_retriever")
    def test_retrieve_handles_empty_results(self, mock_get_retriever, base_state):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from src.agents.nodes import retrieve
        result = retrieve(base_state)

        assert result["documents"] == []


class TestGradeDocumentsNode:
    @patch("src.agents.nodes._get_grader")
    def test_grade_sufficient_docs_no_web_search(self, mock_get_grader, base_state, sample_docs):
        mock_grader = MagicMock()
        mock_grader.filter_relevant.return_value = sample_docs
        mock_get_grader.return_value = mock_grader

        state = {**base_state, "documents":sample_docs}
        from src.agents.nodes import  grade_documents
        result = grade_documents(state)

        assert result["web_search_needed"] is False
        assert len(result["documents"]) == 2



    @patch("src.agents.nodes._get_grader")
    def test_grade_insufficient_docs_triggers_web_search(self,mock_get_grader, base_state, sample_docs):
        mock_grader = MagicMock()
        # Only 0 out of 2 docs pass grading -> trigger web search#
        mock_grader.filter_relevant.return_value = []
        mock_get_grader.return_value = mock_grader

        state = {**base_state, "documents":sample_docs}

        from src.agents.nodes import grade_documents
        result = grade_documents(state)

        assert result["web_search_needed"] is True


class TestGenerateNode:

    @patch("src.agents.nodes._get_llm")
    def test_generate_produces_text(self, mock_get_llm, base_state, sample_docs):

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_chain_result = "The attention mechanism allows models to focus on relevant input parts."

        with patch("src.agents.nodes.ChatPromptTemplate") as mock_prompt:

            mock_chain_v1 = MagicMock()
            mock_prompt.from_messages.return_value.__or__ = lambda s, o: mock_chain_v1

            from src.agents.nodes import generate
            state = {**base_state, "documents": sample_docs}

            with patch("src.agents.nodes.StrOutputParser"):

                mock_chain_v2 = MagicMock()
                mock_chain_v1.__or__ = lambda s, o: mock_chain_v2
                mock_chain_v2.invoke.return_value = mock_chain_result

                result = generate(state)

        assert "generation" in result
        assert result["generation"] == mock_chain_result



# ─────────────────────────────────────────────────────────────────────────────
# Tests: edges
# ─────────────────────────────────────────────────────────────────────────────
class TestEdges:
    def test_route_after_grading_web_search(self,base_state):
        from src.agents.edges import route_after_grading
        state = {**base_state, "web_search_needed":True}
        assert route_after_grading(state) == "web_search"

    def test_route_after_grading_generate(self, base_state):
        from src.agents.edges import route_after_grading
        state = {**base_state, "web_search_needed":False}
        assert route_after_grading(state) == "generate"


    def test_route_hallucination_accept(self, base_state):
        from src.agents.edges import route_after_hallucination_check
        state = {
            **base_state,
            "hallucination_score":"no",
            "answer_addresses_question":"yes",
            "retry_count":0
        }
        assert  route_after_hallucination_check(state) == "regenerate"



    def test_route_hallucination_max_retries_forces_accept(self, base_state):
        from src.agents.edges import  route_after_hallucination_check
        from src.utils import get_settings
        max_r = get_settings().max_retries
        state = {
            **base_state,
            "hallucination_score":"no",
            "answer_addresses_question":"no",
            "retry_count":max_r
        }
        assert route_after_hallucination_check(state) == "accept"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: graph structure
# ─────────────────────────────────────────────────────────────────────────────
class TestGraphStructure:
    def test_graph_compiles(self):
        """Ensure the graph can be built without errors."""
        from src.agents.graph import  build_graph
        g = build_graph()
        assert g is not None


    def test_graph_has_expected_nodes(self):
        from src.agents.graph import build_graph
        g = build_graph()
        nodenames = set(g.nodes.keys())
        expected = {"retrieve", "grade_documents", "web_search", "generate", "check_hallucination"}
        assert expected.issubset(nodenames)




