import json
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from agent import SYSTEM_PROMPT


def _format_state_context(state: AgentState) -> str:
    """Expose slot values to the LLM (they are not in message history alone)."""
    slots = {
        k: state.get(k)
        for k in (
            "person_name",
            "price_preference",
            "location_preference",
            "cuisine_preference",
            "invite_someone",
            "presented_options",
            "selected_restaurant",
            "booking_day",
            "booking_hour",
            "booking_success",
            "contact_phone",
            "whatsapp_sent",
        )
    }
    return json.dumps(slots, ensure_ascii=False, default=str)


def _sync_state_from_tool_messages(state: AgentState) -> dict:
    """Parse latest tool outputs back into state slots."""
    updates: dict = {}
    for msg in reversed(state.get("messages", [])):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            payload = json.loads(msg.content) if isinstance(msg.content, str) else None
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        status = payload.get("status")
        if status == "success" and payload.get("booked") is True:
            updates.setdefault("booking_success", True)
            updates.setdefault("booking_day", payload.get("day"))
            updates.setdefault("booking_hour", payload.get("hour"))
        elif status == "success" and "data" in payload:
            updates.setdefault("presented_options", payload["data"])
        elif status == "success" and "phone" in payload:
            updates.setdefault("contact_phone", payload["phone"])
            updates.setdefault(
                "person_name", payload.get("name") or state.get("person_name")
            )
        elif status == "success" and "sid" in payload:
            updates.setdefault("whatsapp_sent", True)
        break

    return updates


def create_agent_node(llm_with_tools):
    async def agent_node(state: AgentState):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\n[CURRENT STATE SLOTS]\n{state_context}"),
            ("placeholder", "{messages}"),
        ])

        formatted_prompt = prompt_template.invoke({
            "messages": state["messages"],
            "state_context": _format_state_context(state),
        })

        response = await llm_with_tools.ainvoke(formatted_prompt)

        updates = {"messages": [response]}
        updates.update(_sync_state_from_tool_messages(state))
        return updates

    return agent_node
