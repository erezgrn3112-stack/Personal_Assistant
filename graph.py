import asyncio
import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agent import get_agent_llm
from nodes import create_agent_node

ROOT = Path(__file__).resolve().parent


def should_continue(state: AgentState) -> str:
    """Route to tools when the LLM requested tool calls, otherwise END."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def create_graph(mcp_tools: list):
    """Construct and compile the LangGraph workflow using the provided MCP tools."""
    llm_with_tools = get_agent_llm(mcp_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", create_agent_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(mcp_tools))

    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def main():
    """Generate graph visualization cleanly without booting servers or validating API keys."""
    print("📊 Generating graph visualization layout...")
    try:
        # We build a structural twin of the graph topology purely for visualization.
        # This completely bypasses the local MCP server boot and Groq's environment checks.
        preview_workflow = StateGraph(AgentState)

        # Visualization only cares about node names and their connection arrows.
        # We use simple placeholder functions to avoid initializing live dependencies.
        preview_workflow.add_node("agent", lambda state: state)
        preview_workflow.add_node("tools", lambda state: state)

        preview_workflow.add_edge(START, "agent")
        preview_workflow.add_edge("tools", "agent")
        preview_workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )

        preview_app = preview_workflow.compile()

        # Fetch the rendered layout from the Mermaid API engine
        image_data = preview_app.get_graph().draw_mermaid_png()
        output_filename = ROOT / "graph_flowchart.png"
        output_filename.write_bytes(image_data)

        print(f"🎉 Success! Visual flowchart saved locally as '{output_filename}'")

    except Exception as e:
        print(f"❌ Error during visualization: {e}")


if __name__ == "__main__":
    asyncio.run(main())