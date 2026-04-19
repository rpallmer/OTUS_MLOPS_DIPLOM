from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from models.schemas import WorkflowState
from graph.nodes import create_nodes
from graph.routing import route_after_decomposer, route_after_process
from llm.base import BaseLLMBackend

def build_workflow(llm: BaseLLMBackend):
    decomposer_node, process_item_node, aggregator_node = create_nodes(llm)

    workflow = StateGraph(WorkflowState)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("process_item", process_item_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.set_entry_point("decomposer")

    workflow.add_conditional_edges(
        "decomposer",
        route_after_decomposer,
        {"process_item": "process_item", "aggregator": "aggregator"}
    )
    workflow.add_conditional_edges(
        "process_item",
        route_after_process,
        {"process_item": "process_item", "aggregator": "aggregator"}
    )
    workflow.add_edge("aggregator", END)

    return workflow.compile(checkpointer=MemorySaver())