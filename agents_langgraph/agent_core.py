import logging
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from .utils import tools
from .agent_state import AgentState
from .nodes import assistant

builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()