"""LangGraph agent foundation for AgentOps AI (Milestone 2: Tool Calling).

This module defines a cyclical LangGraph agent capable of deciding whether to
call tools (arithmetic calculation or local knowledge retrieval) before
producing a grounded final response.
"""

import os
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.calculator import calculate
from app.tools.knowledge import lookup_knowledge

# Load environment variables from .env file
load_dotenv()

# Registry of tools made available to the agent
TOOLS = [calculate, lookup_knowledge]


class AgentState(TypedDict):
    """Represents the state of the agent throughout execution.

    Attributes:
        user_input: The original request or objective provided by the user.
        response: The final generated response produced by the agent.
        messages: The sequence of conversation messages (Human, AI, Tool)
            tracking reasoning steps, tool invocations, and tool outputs.
    """

    user_input: str
    response: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_llm() -> ChatOpenAI:
    """Instantiate and return the ChatOpenAI client using environment variables.

    Supports OpenAI as well as OpenAI-compatible providers such as Groq, Ollama,
    or OpenRouter via OPENAI_BASE_URL.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please configure your OpenAI or Groq API key in your environment or .env file."
        )

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL") or os.getenv("GROQ_MODEL")

    # Auto-detect Groq keys (prefixed with gsk_) and configure defaults
    if api_key.startswith("gsk_"):
        if not base_url:
            base_url = "https://api.groq.com/openai/v1"
        if not model_name or model_name in ("llama-3.3-70b-versatile", "gpt-4o-mini"):
            model_name = "openai/gpt-oss-120b"

    if not model_name:
        model_name = "gpt-4o-mini"

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def agent_node(state: AgentState) -> Dict[str, Any]:
    """Agent node that invokes the tool-bound LLM.

    The LLM inspects the message history and autonomously decides whether to
    call one or more tools or generate a final direct response.

    Args:
        state: Current agent state containing conversation messages.

    Returns:
        A dictionary updating messages and response in the state.
    """
    llm = get_llm().bind_tools(TOOLS)
    messages = list(state.get("messages", []))
    if not messages and state.get("user_input"):
        messages = [HumanMessage(content=state["user_input"])]

    response_message: AIMessage = llm.invoke(messages)

    # If the model does not request any tool calls, this is the final response
    response_text = ""
    if not response_message.tool_calls:
        response_text = str(response_message.content)

    return {
        "messages": [response_message],
        "response": response_text,
    }


def create_agent_graph(checkpointer: Optional[MemorySaver] = None) -> CompiledStateGraph:
    """Build and compile the cyclical LangGraph agent workflow with tool calling.

    Workflow structure:
        START -> agent -> (conditional: tools_condition)
                   ↓ [tool calls requested]
                 tools -> agent -> END [when no tool calls remain]

    Args:
        checkpointer: Optional MemorySaver for persistent multi-turn conversation
            memory. When provided, state is checkpointed per thread_id, enabling
            stateful multi-turn sessions. When None, the agent is stateless.

    Returns:
        CompiledStateGraph instance ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Register agent and tool execution nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))

    # Define execution edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


# Compile the default graph instance
graph = create_agent_graph()


def run_agent(
    user_input: str,
    session_id: Optional[str] = None,
    checkpointer: Optional[MemorySaver] = None,
) -> str:
    """Run the AgentOps AI agent with the given user input and return the response.

    When a session_id and checkpointer are provided, the agent maintains
    conversation history across multiple calls within the same session, enabling
    stateful multi-turn interactions.

    Args:
        user_input: User question or objective.
        session_id: Optional thread ID for session-scoped memory continuity.
            Generate one with app.memory.checkpointer.generate_session_id().
        checkpointer: Optional MemorySaver instance for persistent state.
            When None, each call is independent (stateless).

    Returns:
        The agent's response text.
    """
    _graph = create_agent_graph(checkpointer=checkpointer) if checkpointer else graph

    # Run input through the security guardrail pipeline
    from app.security.guardrails import run_guardrails
    guardrail_result = run_guardrails(user_input)
    if not guardrail_result.is_safe:
        violation_summary = "; ".join(guardrail_result.violations)
        raise ValueError(f"Input blocked by security guardrails: {violation_summary}")
    # Use sanitized input (PII redacted) for all downstream processing
    user_input = guardrail_result.sanitized_input

    initial_state: AgentState = {
        "user_input": user_input,
        "response": "",
        "messages": [HumanMessage(content=user_input)],
    }
    config = {"configurable": {"thread_id": session_id}} if session_id else {}
    result = _graph.invoke(initial_state, config=config)

    # Return the response field if populated, or fallback to final AIMessage content
    if result.get("response"):
        return result["response"]

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)

    return ""
