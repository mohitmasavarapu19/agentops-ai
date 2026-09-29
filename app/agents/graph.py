"""LangGraph agent foundation for AgentOps AI (Milestone 1).

This module defines a minimal LangGraph StateGraph agent that receives a user
query, processes it through an LLM node powered by langchain-openai, and returns
the agent's response.
"""

import os
from typing import Dict, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Load environment variables from .env file
load_dotenv()


class AgentState(TypedDict):
    """Represents the state of the agent throughout execution.

    Attributes:
        user_input: The request or objective provided by the user.
        response: The generated response produced by the agent.
    """

    user_input: str
    response: str


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


def agent_node(state: AgentState) -> Dict[str, str]:
    """Agent node that queries the LLM with the user input and updates the state.

    Args:
        state: Current agent state containing user_input.

    Returns:
        A dictionary with the updated response field.
    """
    llm = get_llm()
    result = llm.invoke(state["user_input"])
    return {"response": str(result.content)}


def create_agent_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph agent workflow.

    Workflow structure:
        START -> agent -> END

    Returns:
        CompiledStateGraph instance ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Register the primary agent node
    workflow.add_node("agent", agent_node)

    # Define the graph execution flow
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

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
    }
    result = graph.invoke(initial_state)
    return result["response"]
