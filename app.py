import asyncio
import datetime
import queue
import sys
import threading
import uuid
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from graph import create_graph

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

INVOKE_TIMEOUT_SECONDS = 120


class MCPAgentRuntime:
    """Persistent MCP session and LangGraph app on a dedicated background event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._init_error: Exception | None = None

        self._stdio_cm = None
        self._session_cm = None
        self._session = None
        self._app = None

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=60):
            raise TimeoutError("MCP runtime failed to initialize within 60 seconds.")
        if self._init_error is not None:
            raise self._init_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._bootstrap())
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _bootstrap(self) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            cwd=str(ROOT),
        )

        # Manual context entry keeps stdio streams alive for the app lifetime.
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        tools = await load_mcp_tools(self._session)
        self._app = create_graph(tools)

    def invoke(self, inputs: dict, config: dict) -> dict:
        """Run the graph asynchronously on the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(
            self._app.ainvoke(inputs, config=config),
            self._loop,
        )
        return future.result(timeout=INVOKE_TIMEOUT_SECONDS)

    def stream(self, inputs: dict, config: dict):
        """Yield graph update chunks from the background loop (stream_mode='updates')."""
        q: queue.Queue = queue.Queue()

        async def _producer() -> None:
            try:
                async for chunk in self._app.astream(
                    inputs, config=config, stream_mode="updates"
                ):
                    q.put(chunk)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)

        asyncio.run_coroutine_threadsafe(_producer(), self._loop)

        while True:
            item = q.get(timeout=INVOKE_TIMEOUT_SECONDS)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


@st.cache_resource
def get_runtime() -> MCPAgentRuntime:
    """Initialize the MCP-backed LangGraph runtime once per Streamlit server process."""
    with st.spinner("Starting assistant..."):
        return MCPAgentRuntime()


def extract_assistant_content(result: dict) -> str:
    """Extract the final assistant text, falling back to the last non-empty AIMessage."""
    messages = result.get("messages", [])
    if not messages:
        return "No response generated."

    last_content = getattr(messages[-1], "content", None)
    if isinstance(last_content, str) and last_content.strip():
        return last_content

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if content and not isinstance(content, str):
                return str(content)

    return "No response generated."


def get_tool_status_text(tool_name: str, tool_args: dict | None) -> str:
    """Map an active tool call to a short Hebrew status line for the typing indicator."""
    args = tool_args if isinstance(tool_args, dict) else {}

    if tool_name == "search_restaurants":
        return "מחפש מסעדות מתאימות במאגר..."
    if tool_name == "check_table_availability":
        return 'בודק זמינות שולחן בלו"ז המסעדה...'
    if tool_name == "book_table_slot":
        return "משריין עבורך את השולחן המבוקש..."
    if tool_name == "search_contact":
        name = args.get("name") or args.get("query") or args.get("person_name")
        if name:
            return f"מחפש את המספר של {name} באנשי הקשר..."
        return "מחפש באנשי הקשר..."
    if tool_name == "send_whatsapp_invitation":
        recipient_name = args.get("recipient_name", "איש הקשר")
        return f"שולח הזמנת וואטסאפ ל{recipient_name} דרך Twilio..."

    return "מריץ פעולה ברקע..."


def _extract_tool_call_fields(tool_call) -> tuple[str, dict]:
    """Normalize tool call name and args from dict or LangChain tool-call objects."""
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
    else:
        tool_name = getattr(tool_call, "name", "") or ""
        tool_args = getattr(tool_call, "args", {}) or {}

    if not isinstance(tool_args, dict):
        tool_args = {}
    return tool_name, tool_args


st.set_page_config(page_title="Personal Assistant", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e5ddd5;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
        background-repeat: repeat;
        background-attachment: fixed;
    }
    .main { direction: rtl; }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 5px;
        max-width: 85%;
        border: none;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff !important;
        align-self: flex-start;
        border-top-right-radius: 0px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    [data-testid="stChatMessageUser"] {
        background-color: #dcf8c6 !important;
        align-self: flex-end;
        border-top-left-radius: 0px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        margin-right: auto;
    }
    [data-testid="stChatMessageContent"] { text-align: right; direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

with st.sidebar:
    st.header("🛠️ Test Management")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st.rerun()

    st.divider()
    st.caption(f"**Session ID:** `{st.session_state.config['configurable']['thread_id']}`")

st.title("💬 Personal Assistant")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["content"]:
                st.write(msg["content"])
                if "time" in msg:
                    display_time = msg["time"]
                else:
                    # Fallback for messages created before timestamps were persisted
                    display_time = datetime.datetime.now().strftime("%H:%M")
                if msg["role"] == "user":
                    st.caption(f"✓ {display_time}")
                else:
                    st.caption(display_time)


def process_and_display(user_input: str) -> None:
    """Append the user message, stream the graph, and render the assistant reply."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "time": datetime.datetime.now().strftime("%H:%M"),
        }
    )
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            typing_placeholder = st.empty()

            def render_status(text: str) -> None:
                safe_text = escape(text)
                typing_placeholder.markdown(
                    f"""
                    <div style="
                        display: inline-flex;
                        align-items: center;
                        gap: 10px;
                        direction: rtl;
                        padding: 8px 14px;
                        border-radius: 12px;
                        background-color: #f0f0f0;
                        color: #667781;
                        font-size: 0.95rem;
                    ">
                        <span>{safe_text}</span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;">
                            <span class="typing-dot" style="
                                display: inline-block;
                                width: 8px;
                                height: 8px;
                                border-radius: 50%;
                                background-color: #90949c;
                                animation: typing-bounce 1.4s infinite ease-in-out both;
                                animation-delay: -0.32s;
                            "></span>
                            <span class="typing-dot" style="
                                display: inline-block;
                                width: 8px;
                                height: 8px;
                                border-radius: 50%;
                                background-color: #90949c;
                                animation: typing-bounce 1.4s infinite ease-in-out both;
                                animation-delay: -0.16s;
                            "></span>
                            <span class="typing-dot" style="
                                display: inline-block;
                                width: 8px;
                                height: 8px;
                                border-radius: 50%;
                                background-color: #90949c;
                                animation: typing-bounce 1.4s infinite ease-in-out both;
                            "></span>
                        </span>
                    </div>
                    <style>
                    @keyframes typing-bounce {{
                        0%, 80%, 100% {{
                            transform: scale(0.6);
                            opacity: 0.5;
                        }}
                        40% {{
                            transform: scale(1);
                            opacity: 1;
                        }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            render_status("חושב על הצעד הבא...")

            inputs = {"messages": [HumanMessage(content=user_input)]}
            final_reply = ""

            try:
                runtime = get_runtime()
                for chunk in runtime.stream(inputs, st.session_state.config):
                    if "agent" not in chunk:
                        continue

                    last_msg = chunk["agent"]["messages"][-1]

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            tool_name, tool_args = _extract_tool_call_fields(tool_call)
                            render_status(get_tool_status_text(tool_name, tool_args))

                    if last_msg.content:
                        content = last_msg.content
                        if isinstance(content, str) and content.strip():
                            final_reply = content

                typing_placeholder.empty()

                if not final_reply:
                    final_reply = "No response generated."

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_reply,
                        "time": datetime.datetime.now().strftime("%H:%M"),
                    }
                )
                if final_reply:
                    st.write(final_reply)

            except TimeoutError:
                typing_placeholder.empty()
                st.error("The assistant took too long to respond. Please try again.")
            except Exception as exc:
                typing_placeholder.empty()
                st.error(f"Something went wrong: {exc}")

    st.rerun()


try:
    runtime = get_runtime()
except Exception as exc:
    st.error(f"Failed to start the assistant: {exc}")
    st.stop()

if prompt := st.chat_input("Message"):
    process_and_display(prompt)
