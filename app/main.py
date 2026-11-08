"""Command-line entry point for the AgentOps AI agent."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to support running directly: python app/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.graph import run_agent
from app.memory.checkpointer import generate_session_id, get_default_checkpointer


def main() -> None:
    """Run the AgentOps AI CLI with optional stateful session memory."""
    parser = argparse.ArgumentParser(
        description="AgentOps AI - Autonomous Business Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Stateless single query:\n"
            "  python -m app.main \"What is the Starter plan price?\"\n\n"
            "  # Stateful session (auto-generates a session ID):\n"
            "  python -m app.main --memory\n\n"
            "  # Resume a specific session:\n"
            "  python -m app.main --memory --session-id <uuid>\n"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Query to run (omit for interactive mode)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable persistent multi-turn memory for this session",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Resume an existing session by ID (implies --memory)",
    )
    args = parser.parse_args()

    # Resolve memory / session configuration
    use_memory = args.memory or bool(args.session_id)
    checkpointer = get_default_checkpointer() if use_memory else None
    session_id = args.session_id or (generate_session_id() if use_memory else None)

    print("=" * 50)
    print("AgentOps AI - Agent Runner (Milestone 4: Memory)")
    print("=" * 50)
    if use_memory:
        print(f"Memory Mode: ON  |  Session ID: {session_id}")
        print("(Use this Session ID with --session-id to resume this conversation)")
    else:
        print("Memory Mode: OFF  (use --memory to enable)")

    if args.prompt:
        # Single-shot mode
        _run_single(args.prompt, session_id, checkpointer)
    else:
        # Interactive loop
        print("\nInteractive mode — type 'exit' or press Ctrl+C to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Goodbye!")
                break
            _run_single(user_input, session_id, checkpointer)


def _run_single(user_input: str, session_id, checkpointer) -> None:
    """Execute a single agent turn and print the response."""
    print(f"\nUser: {user_input}")
    print("Agent: ", end="", flush=True)
    try:
        response = run_agent(user_input, session_id=session_id, checkpointer=checkpointer)
        print(response)
        print()
    except ValueError as err:
        print(f"\n[Configuration Error] {err}")
        print("Hint: Copy .env.example to .env and set your OPENAI_API_KEY.")
        sys.exit(1)
    except Exception as err:
        print(f"\n[Execution Error] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
