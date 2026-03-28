"""
Tests for RAG components: retriever, grader, ingest.
"""

from __future__ import annotations

from unittest.mock import  MagicMock, patch

from langchain_core.documents import Document
from src.rag.grader import GradeScore


class TestDocumentGrader:
    @patch("src.rag.grader.ChatOpenAI")
    def test_grade_relevant_document(self, mock_llm_class):
        """Grader should return relevant=True for a matching document."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = GradeScore(
            relevant=True,
            reasoning="Document directly discusses attention mechanism"
        )
        with patch("src.rag.grader.ChatPromptTemplate") as mock_prompt:

        