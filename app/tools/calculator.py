"""Calculator tool for basic arithmetic operations in AgentOps AI."""

from langchain_core.tools import tool


@tool
def calculate(a: float, b: float, operation: str) -> str:
    """Perform basic arithmetic calculations between two numbers.

    Supported operations:
        - Addition: 'add', '+', 'plus', 'addition'
        - Subtraction: 'subtract', '-', 'minus', 'subtraction'
        - Multiplication: 'multiply', '*', 'times', 'multiplication'
        - Division: 'divide', '/', 'division'

    Args:
        a: The first numeric operand.
        b: The second numeric operand.
        operation: The arithmetic operation to execute.

    Returns:
        The computed result as a string, or an error message if the operation
        is unsupported or mathematically invalid (e.g. division by zero).
    """
    op = operation.strip().lower()

    if op in ("add", "+", "plus", "addition"):
        result = a + b
    elif op in ("subtract", "-", "minus", "subtraction"):
        result = a - b
    elif op in ("multiply", "*", "times", "multiplication"):
        result = a * b
    elif op in ("divide", "/", "division"):
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
    else:
        return (
            f"Error: Unsupported operation '{operation}'. "
            "Supported operations are: add (+), subtract (-), multiply (*), divide (/)."
        )

    # Format whole numbers without trailing decimal zeros
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
