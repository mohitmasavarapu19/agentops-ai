"""LangGraph agent foundation for AgentOps AI (Milestone 2: Tool Calling).

This module defines a cyclical LangGraph agent capable of deciding whether to
call tools (arithmetic calculation or local knowledge retrieval) before
producing a grounded final response.
"""

import os
from typing import Annotated, Any, Dict, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
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

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please configure your OpenAI API key in your environment or .env file."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, api_key=api_key)


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


def create_agent_graph() -> CompiledStateGraph:
    """Build and compile the cyclical LangGraph agent workflow with tool calling.

    Workflow structure:
        START -> agent -> (conditional: tools_condition)
                   ↓ [tool calls requested]
                 tools -> agent -> END [when no tool calls remain]

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

    return workflow.compile()


# Compile the default graph instance
graph = create_agent_graph()


def run_agent(user_input: str) -> str:
    """Run the AgentOps AI agent with the given user input and return the response.

    Args:
        user_input: User question or objective.

    Returns:
        The agent's response text.
    """
    initial_state: AgentState = {
        "user_input": user_input,
        "response": "",
        "messages": [HumanMessage(content=user_input)],
    }
    result = graph.invoke(initial_state)

    # Return the response field if populated, or fallback to final AIMessage content
    if result.get("response"):
        return result["response"]

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)

    return ""
