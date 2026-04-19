from models.schemas import WorkflowState

def route_after_decomposer(state: WorkflowState) -> str:
    if state.get("status") == "failed" or state.get("total_items", 0) == 0:
        return "aggregator"
    return "process_item"

def route_after_process(state: WorkflowState) -> str:
    if state["current_index"] < state["total_items"]:
        return "process_item"
    return "aggregator"