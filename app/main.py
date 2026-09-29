"""Command-line entry point for testing the AgentOps AI agent."""

import sys
from pathlib import Path

# Add project root to sys.path to support running directly: python app/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.graph import run_agent


def main() -> None:
    """Run the AgentOps AI CLI tester."""
    print("=" * 50)
    print("AgentOps AI - Agent Runner (Milestone 3: Production RAG)")
    print("=" * 50)

    # Check for prompt passed via command-line arguments
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip()
    else:
        try:
            user_input = input("\nEnter your prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled. Exiting.")
            return

    if not user_input:
        print("No input provided. Exiting.")
        return

    print(f"\nUser Input: {user_input}")
    print("\nRunning agent...")

    try:
        response = run_agent(user_input)
        print("\nAgent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    except ValueError as err:
        print(f"\n[Configuration Error] {err}")
        print(
            "Hint: Copy .env.example to .env and set your OPENAI_API_KEY:\n"
            "  cp .env.example .env"
        )
        sys.exit(1)
    except Exception as err:
        print(f"\n[Execution Error] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
