from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="用户输入的问题",
    )


class SourceItem(BaseModel):
    source_number: int
    chunk_id: str
    source: str
    equipment: str
    category: str
    rerank_score: float
    text: str


class ChatResponse(BaseModel):
    status: str
    answer: str
    top1_score: float
    threshold: float
    sources: List[SourceItem]


class ErrorResponse(BaseModel):
    error: str
    message: str