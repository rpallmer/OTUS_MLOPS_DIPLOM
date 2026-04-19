from typing import TypedDict, List, Dict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

class EntitiesSchema(BaseModel):
    contract_id: Optional[str] = None
    account_number: Optional[str] = None
    address: Optional[str] = None  # Исправлена опечатка
    dates: Optional[List[str]] = None
    amounts: Optional[List[str]] = None
    other: Optional[List[str]] = None

class SubQuery(BaseModel):
    id: int = Field(ge=1, description="Уникальный ID под-вопроса")
    sub_query: str
    category: Literal["legal", "general_kb", "external_api", "operator_handoff"]
    entities: EntitiesSchema
    search_keywords: str
    routing_target: Literal[
        "legal_templates_collection",
        "reference_info_collection",
        "external_api_tools",
        "operator_queue"
    ]

class ItemResult(BaseModel):
    id: int
    answer: str
    status: Literal["processed", "operator_pending", "failed", "unofficial"]
    source: str
    is_official: bool = False
    error: Optional[str] = None

def merge_partial_results(old: Dict[int, ItemResult], new: Dict[int, ItemResult]) -> Dict[int, ItemResult]:
    return {**old, **new}

class WorkflowState(TypedDict):
    original_query: str
    decomposed_items: List[SubQuery]
    partial_results: Annotated[Dict[int, ItemResult], merge_partial_results]
    final_response: str
    status: Literal["init", "routing", "processing", "aggregating", "completed", "failed"]
    metadata: Dict[str, str]
    errors: List[str]
    current_index: int
    total_items: int

class RetrievedDoc(BaseModel):
    point_id: int | str
    score: float
    question: str
    answer: str
    source: str
    category: str
    dense_rank: int | None
    sparse_rank: int | None
    final_rank: int
