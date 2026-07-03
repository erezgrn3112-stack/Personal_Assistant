
# AI Personal Assistant (LangGraph & MCP)

A smart AI Personal Assistant that helps users discover restaurants in Israel, check table availability, book reservations, and automatically invite friends via WhatsApp. 

This project demonstrates a stateful, multi-turn LLM agent architecture built using LangGraph, utilizing Model Context Protocol (MCP) servers to isolate data access and external API integrations.

---

## Core Features

* **Contextual Category Mapping**: Semantically maps user inputs in natural Hebrew to fixed database schemas (e.g., mapping "Sushi" to `יפני`, "Moderate budget" to `$$`, or "Center area" to `תל אביב`).
* **Strict Operational Gatekeepers**: Prevents premature database queries by forcing the agent to resolve required parameters (like budget or exact day/hour) before running tools.
* **Human-in-the-Loop (HITL) Booking**: Gates the reservation process. The system checks table availability first, displays it to the user, and triggers the actual booking node only after explicit verbal confirmation (e.g., "Yes, lock it in").
* **Dynamic Status Logs & Bouncing Dots**: Replaces static spinners with a real-time, async-to-sync thread bridge (`queue.Queue`). It displays custom horizontal bouncing dots alongside specific Hebrew text corresponding to the exact tool being executed (e.g., "Searching contacts...", "Sending WhatsApp...").
* **WhatsApp-Style UI with Frozen Timestamps**: A fully customized Streamlit frontend featuring an RTL chat layout, message checkmarks, and static timestamps stored in the session state to prevent clock shifting on UI reruns.

---

## Tech Stack

* **Frontend**: Streamlit (Custom CSS for WhatsApp-style RTL chat interface)
* **Orchestration**: LangGraph (Stateful workflow topology with memory checkpointing)
* **LLM Engine**: Groq (`llama-3.3-70b-versatile` running at temperature 0)
* **Tool Adapter**: `langchain-mcp-adapters` (Bridges MCP tools into standard LangChain tools)
* **MCP Server**: FastMCP (`mcp_server.py` running as a local stdio subprocess)
* **Integrations**: Local CSV Databases (Restaurants, Schedules, Contacts) & Twilio WhatsApp API

---

## Project Structure

```text
Personal_Assistant/
├── app.py                 # Streamlit UI, main loop, and async-to-sync streaming queue bridge
├── agent.py               # Groq LLM configuration, system prompt, and mapping logic
├── graph.py               # LangGraph state machine workflow definition and routing
├── nodes.py               # Graph nodes, prompt generation, and state sync helpers
├── state.py               # AgentState TypedDict schema definition
├── mcp_server.py          # FastMCP server exposing local CSV tools and Twilio pipeline
├── restaurants_db.csv     # Local restaurant dataset
├── schedule_db.csv        # Stateful reservation schedule calendar
├── contacts_db.csv        # Contact directory linking names to phone numbers
├── .env                   # Local API credentials (git-ignored)
├── .gitignore             # Standard repository exclusion rules
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Personal_Assistant

```

### 2. Set up a virtual environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure environment variables

Create a `.env` file in the project root directory:

```env
# Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Twilio Credentials (for WhatsApp functionality)
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

```

### 5. Run the application

```bash
streamlit run app.py

```

---

## Graph Visualization

To generate a visual flowchart of the LangGraph state machine layout without connecting to any external APIs, run:

```bash
python graph.py

```

This saves a compiled architecture diagram on disk as `graph_flowchart.png`.

```

```