"""
LLM-based document relevance grader.
Uses structured output (Pydantic) for reliable binary classification.

"""

from __future__ import annotations
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils import  get_settings, get_logger

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()

class GradeScore(BaseModel):
    """Binary relevance score for a retrieval document."""

    relevant: bool = Field(
        description="True if the document is relevant to the user's question, False otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why the document is or is not relevant"
    )

GRADER_SYSTEM_PROMPT = """
You are an expert relevance grader for a RAG system.

Your task: assess whether a retrieved document contains information that is useful for answering the user's question.

Rules:
- Grade RELEVANT if the document contains ANY useful context, keywords, or related concepts.
- Grade NOT RELEVANT if the document is completely off-topic or contains only noise.
- Be lenient: partial relevance counts as relevant.

Respond with a JSON object matching the schema provided.
"""


GRADER_HUMAN_PROMPT = """
Question: {question}

Retrieved Docuemnt:
---
{document}
---

Is this document relevant to the question?
"""

class DocumentGrader:
    """
    Grades retrieved documents for relevance using an LLM with structured output
    """
    def __init__(self) -> None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature = 0,
        )
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", GRADER_SYSTEM_PROMPT),
                    ("human", GRADER_HUMAN_PROMPT)
                ]
            )   | llm.with_structured_output(GradeScore)
        )


    def grade(
            self,
    question: str,
    document: Document) -> GradeScore:
        """Grade a single document against a question"""
        return self._chain.invoke(
            {
                "question":question,
                "document":document.page_content[:2000]
            }
        )

    def filter_relevant(self,
                        question:str,
                        documents: list[Document]) -> list[Document]:
        """
        Filter a list of documents, keeping only those graded as relevant.

        Returns:
            List of relevant documents (may be empty).
        """
        relevant: list[Document] = []
        for doc in documents:
            score = self.grade(question, doc)
            status = "✓ relevant" if score.relevant else "✗ irrelevant"
            logger.debug(f"  [{status}] {score.reasoning[:80]}")
            if score.relevant:
                relevant.append(doc)
        logger.info(f"Grading: {len(relevant)}/{len(documents)} documents kept")
        return relevant