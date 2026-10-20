import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier



from mcp_client import (
    client,
    get_mcp_tools,
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=OPENAI_API_KEY,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


guardrail_reason = "Guardrail validation fallback allowed the request."



from langchain_core.output_parsers import string
# =========================
# Input Guardrail (TypeSafe Classifier)
# =========================
GUARDRAIL_CONFIDENCE_THRESHOLD = 0.75  # 75% confidence on a 0.0 - 1.0 probability scale

Jev_classifier = TypeSafeClassifier(
    model="jev-latest",
    api_key=os.environ["TYPESAFE_API_KEY"],
)


def jev_response(user_query: str, guardrail_prompt: str) -> tuple[bool, float]:
    """
    Classifies whether the user request is a valid travel-planning query.

    Returns:
        tuple[bool, float]: (is_allowed, confidence_score)
        - is_allowed: True if probability >= GUARDRAIL_CONFIDENCE_THRESHOLD (0.75)
        - confidence_score: Raw float probability between 0.0 and 1.0
    """
    try:
        response = Jev_classifier.invoke(
            {
                "state": (user_query,),
                "questions": {
                    "guardrail_Approval": Noul(instructions=guardrail_prompt)
                },
            }
        )
        score = float(response.nouls["guardrail_Approval"].noul)
        is_allowed = score >= GUARDRAIL_CONFIDENCE_THRESHOLD

        print(
            f"[GUARDRAIL] Query: '{user_query[:50]}...' | "
            f"Confidence: {score:.2f} (Threshold: {GUARDRAIL_CONFIDENCE_THRESHOLD}) | "
            f"Allowed: {is_allowed}"
        )
        return is_allowed, score

    except Exception as exc:
        print(f"[GUARDRAIL WARNING] Classification failed ({exc}). Falling open as safety fallback.")
        return True, 1.0


def guardrail_blocked_agent(state: TravelState):
    """
    Input Guardrail Node:
    Acts as the initial gatekeeper before any expensive specialist agents or MCP tools run.
    - If valid travel request: Sets guardrail_allowed=True and passes to supervisor.
    - If unrelated/harmful request: Sets guardrail_allowed=False and prepares early exit response.
    """
    print("\n--- [NODE: GUARDRAIL] Evaluating Input ---")
    query = state.get("user_query", "")
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = """Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing. You need to classify if it is a valid request or not."""

    allowed, confidence = jev_response(query, guardrail_prompt)

    if allowed:
        reason = "approved"
        return {
            "guardrail_allowed": True,
            "guardrail_reason": reason,
            "llm_calls": llm_calls + 1,
            "messages": [AIMessage(content="Guardrail approved travel request.")],
        }
    else:
        reason = (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "final_response": reason,
            "llm_calls": llm_calls + 1,
            "messages": [AIMessage(content=reason)],
        }



def supervisor_agent(state: TravelState):
    print("\n--- [NODE: SUPERVISOR AGENT] ---")
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)
    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"[SUPERVISOR WARNING] Fallback used ({exc})")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    print(f"[SUPERVISOR] Delegated Agent Plan: {selected_agents}")
    print(f"[SUPERVISOR] Reasoning: {reasoning}")
    if any(constraints.values()):
        print(f"[SUPERVISOR] Extracted Constraints: {constraints}")

    return {
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls + 1,
    }



# =========================
# Agentic MCP Helper
# =========================
def _run_agent_with_mcp(
    system_prompt: str,
    user_prompt: str,
    server_name: str,
    max_turns: int = 2,
) -> tuple[str, int]:
    """
    Agentic MCP execution helper:
    1. Loads cached tools for the requested MCP server.
    2. Binds tools directly to the model (llm.bind_tools).
    3. The LLM autonomously inspects tool schemas, determines which tools to call,
       and formulates search parameters dynamically from context.
    4. Executes requested tool calls and generates a polished specialist response.
    """
    try:
        tools = asyncio.run(get_mcp_tools(server_name))
    except Exception as exc:
        print(f"[MCP] Could not connect to '{server_name}' server: {exc}")
        tools = []

    if not tools:
        # Graceful fallback: LLM generates advisory guidance directly if server is offline
        resp = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(resp.content), 1

    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    llm_calls = 0
    for _ in range(max_turns):
        response = llm_with_tools.invoke(messages)
        llm_calls += 1
        messages.append(response)

        if not response.tool_calls:
            return str(response.content), llm_calls

        # Execute tool calls chosen autonomously by the LLM
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]
            tool = tool_map.get(t_name)

            print(f"[{server_name.upper()} TOOL CALL] Invoking {t_name} with {t_args}")
            if tool:
                try:
                    tool_output = asyncio.run(tool.ainvoke(t_args))
                    content = str(tool_output)
                except Exception as err:
                    content = f"Tool '{t_name}' execution error: {err}"
            else:
                content = f"Tool '{t_name}' not found."

            messages.append(ToolMessage(content=content, tool_call_id=t_id))

    final_resp = llm.invoke(messages)
    return str(final_resp.content), llm_calls + 1


# =========================
# Flight Agent - Agentic MCP
# =========================
def flight_agent(state: TravelState):
    """
    Agentic Flight Specialist:
    Uses AviationStack MCP tools if available, or synthesizes flight advice,
    likely departure/arrival airports, airlines, duration, and fare estimates.
    """
    print("\n--- [NODE: FLIGHT AGENT (AGENTIC MCP)] ---")
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    system_prompt = (
        "You are an expert travel flight planner. Analyze the user request and constraints. "
        "Use aviation tools if available. Provide:\n"
        "1. Likely departure airport\n"
        "2. Likely arrival airport\n"
        "3. Airlines serving this route\n"
        "4. Typical flight duration\n"
        "5. Estimated airfare range\n"
        "6. Peak season pricing warning\n"
        "7. Booking advice\n"
        "Return concise, realistic travel guidance."
    )
    user_prompt = f"User Request: {query}\nTrip Constraints: {constraints}"

    flight_results, calls = _run_agent_with_mcp(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        server_name="aviationstack",
    )

    return {
        "flight_results": flight_results,
        "messages": [AIMessage(content="Flight recommendations generated.")],
        "llm_calls": state.get("llm_calls", 0) + calls,
    }


# =========================
# Hotel Agent - Agentic MCP
# =========================
def hotel_agent(state: TravelState):
    """
    Agentic Hotel Specialist:
    Instead of dumping raw search results, the LLM autonomously formulates targeted
    Tavily search queries, analyzes live results, and synthesizes structured hotel advice.
    """
    print("\n--- [NODE: HOTEL AGENT (AGENTIC MCP)] ---")
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    system_prompt = (
        "You are an expert hotel and accommodation specialist. Use Tavily web search tools "
        "to find top-rated hotels, boutique stays, and neighborhood recommendations matching "
        "the traveler's budget and travel style. Synthesize a clean, professional recommendation "
        "with price ranges, areas, and amenities."
    )
    user_prompt = f"User Request: {query}\nTrip Constraints: {constraints}"

    hotel_results, calls = _run_agent_with_mcp(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        server_name="tavily",
    )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel recommendations processed.")],
        "llm_calls": state.get("llm_calls", 0) + calls,
    }


# =========================
# Weather Agent - Agentic MCP
# =========================
def weather_agent(state: TravelState):
    """
    Agentic Weather Specialist:
    The model autonomously decides whether to call get_current_weather, get_forecast,
    or both, extracts the destination on its own, and provides tailored weather & packing advice.
    """
    print("\n--- [NODE: WEATHER AGENT (AGENTIC MCP)] ---")
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    system_prompt = (
        "You are an expert travel meteorologist. Inspect the traveler's request and constraints. "
        "Use your weather tools (get_current_weather, get_forecast) to look up real-time weather "
        "and forecasts for the destination. Provide concise weather guidance, temperature ranges, "
        "and packing recommendations."
    )
    user_prompt = f"User Request: {query}\nTrip Constraints: {constraints}"

    weather_results, calls = _run_agent_with_mcp(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        server_name="weather",
    )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
        "llm_calls": state.get("llm_calls", 0) + calls,
    }


# =========================
# Budget Agent - Specialist
# =========================
def budget_agent(state: TravelState):
    print("\n--- [NODE: BUDGET AGENT] ---")
    print("[BUDGET] Analyzing trip affordability, cost categories, and savings...")
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    print("[BUDGET] Cost feasibility analysis completed.")
    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - Synthesis
# =========================
def itinerary_agent(state: TravelState):
    print("\n--- [NODE: ITINERARY AGENT] ---")
    print("[ITINERARY] Synthesizing specialist results into an integrated day-by-day plan...")
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    print("[ITINERARY] Draft itinerary created. Ready for human review.")
    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    print("\n--- [NODE: HUMAN APPROVAL (HITL INTERRUPT)] ---")
    print("[HITL] Graph paused! Waiting for traveler review/approval in web UI...")

    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    print(f"[HITL] Review received! Approved: {approved} | Feedback: '{human_feedback}'")
    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - Polishing
# =========================
def final_agent(state: TravelState):
    print("\n--- [NODE: FINAL POLISHING AGENT] ---")
    is_approved = state.get("approved", False)
    print(f"[FINAL] Polishing plan based on human review (Approved={is_approved})...")

    if is_approved:
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    print("[FINAL] Complete trip plan finalized successfully.")
    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    selected = _selected_agents(state)
    target = selected[0] if selected else "itinerary_agent"
    print(f"[ROUTER] Supervisor -> Dispatching to first specialist: '{target}'")
    return target


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                print(f"[ROUTER] Finished '{current_agent}' -> Next specialist: '{next_agent}'")
                return next_agent

        print(f"[ROUTER] Finished '{current_agent}' -> Specialists complete. Moving to 'itinerary_agent'")
        return "itinerary_agent"

    return route


def guardrail_node_conditional_edges(state: TravelState) -> str:
    """
    Conditional routing function evaluated immediately after the 'guardrail' node:

    Decision Logic:
    - If guardrail_allowed is True (Confidence >= 0.75):
      Route to 'supervisor' so the request can be parsed and dispatched to specialists.
    - If guardrail_allowed is False (Confidence < 0.75 or off-topic):
      Route directly to END. Execution halts early, saving LLM calls and API costs.
    """
    is_allowed = state.get("guardrail_allowed", True)

    if is_allowed:
        print("[ROUTER] Guardrail passed -> Routing to 'supervisor'")
        return "supervisor"
    else:
        reason = state.get("guardrail_reason", "Blocked by guardrail.")
        print(f"[ROUTER] Guardrail blocked request -> Routing directly to END. Reason: {reason}")
        return END

# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

# 1. Register all agent and process nodes
graph.add_node("guardrail", guardrail_blocked_agent)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

# 2. Workflow Entry: START always enters the Guardrail first
graph.add_edge(START, "guardrail")

# 3. Guardrail Branching:
# Pass the explicit mapping dictionary {"supervisor": "supervisor", END: END}
# so LangGraph's visualizer and compiler establish static edges to both destinations.
graph.add_conditional_edges(
    "guardrail",
    guardrail_node_conditional_edges,
    {
        "supervisor": "supervisor",
        END: END,
    },
)

# 4. Supervisor Branching: routes to the first specialist agent requested
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)

# =========================
# In-Memory Checkpointer (no database required)
# =========================
checkpointer = MemorySaver()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
