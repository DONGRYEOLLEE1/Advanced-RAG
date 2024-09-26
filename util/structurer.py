from typing import Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevant check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description = "Documents are relevant to the question, 'yes' or 'no'."
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: str = Field(
        description = "Answer is grounded in the facts, 'yes' or 'no'"
    )