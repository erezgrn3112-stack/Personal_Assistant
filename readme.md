# Model Context Protocol (MCP) Powered AI Personal Assistant

A production-grade, event-driven AI Personal Assistant designed for seamless, contextual restaurant discovery, table availability confirmation, interactive reservation management, and automated guest coordination via personalized WhatsApp integration.

This system is engineered as a stateful, multi-turn agentic workflow. A responsive **Streamlit** frontend orchestrates an asynchronous **LangGraph** execution pipeline backed by **Groq (Llama-3.3-70B-Versatile)**, with all discrete side-effects and data layers decoupled via the standard **Model Context Protocol (MCP)**.

---

## Core Architecture & Tech Stack

| System Layer | Technology | Engineering Role |
| :--- | :--- | :--- |
| **User Interface** | Streamlit | RTL-optimized, interactive WhatsApp-style chat UI featuring dynamic execution logs. |
| **State Orchestration** | LangGraph | Stateful multi-turn directed acyclic graph (DAG) utilizing thread-isolated `MemorySaver` checkpointing. |
| **Reasoning Engine** | Groq (`llama-3.3-70b-versatile`) | Tool-bound, zero-temperature language model executing a highly grounded, native Hebrew system prompt. |
| **Protocol Adapter** | `langchain-mcp-adapters` | Structural mapping layer converting raw MCP tool schemas into LangChain-compatible functional assets. |
| **MCP Subprocess** | FastMCP (`mcp_server.py`) | Decoupled client-server stdio runtime managing local CSV transactions and Twilio configurations. |
| **Messaging Pipeline** | Twilio WhatsApp API | Downstream transactional delivery vehicle for dynamic, personalized customer invitations. |

### Architectural Data Flow

```text
[ User Interface ] (Streamlit Foreground Thread)
       │
       ▼ (Thread-Safe Sync-to-Async Queue Bridge)
[ MCPAgentRuntime ] (Dedicated Background Event Loop)
       │
       ├─► [ LangGraph Core ] (Stateful DAG Orchestrator)
       │          │
       │          ▼ (Dynamic Tool Routing Matrix)
       └─► [ MCP Stdio Client ] ◄──► [ Local FastMCP Server (mcp_server.py) ]
                                                │
                                                ├─► [ CSV Relational Mappings ]
                                                └─► [ Twilio Messaging Gateway ]