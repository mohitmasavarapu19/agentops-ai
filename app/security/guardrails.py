"""Input security guardrails for AgentOps AI.

Provides layered defences before any user input is passed to the LLM:
  - Prompt injection detection (instruction-override patterns)
  - PII redaction (email, phone, credit card)
  - Input length enforcement
  - Null / whitespace input rejection
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH: int = 4096  # Hard limit in characters

# Patterns that commonly indicate prompt-injection attempts
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new|evil|unrestricted)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions)", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(your\s+)?(instructions?|guidelines?|rules?)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unrestricted|jailbreak)", re.IGNORECASE),
    re.compile(r"(dan|jailbreak|act\s+as\s+if\s+you\s+have\s+no\s+restrictions)", re.IGNORECASE),
    re.compile(r"system\s*:\s*(you\s+are|ignore|override)", re.IGNORECASE),
    re.compile(r"<\s*(system|instruction|prompt)\s*>", re.IGNORECASE),
]

# PII redaction patterns
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ \-]?){13,16}\b")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of running a user input through the guardrail pipeline.

    Attributes:
        is_safe: True if the input passed all checks.
        sanitized_input: The input after PII redaction (may differ from original).
        violations: List of human-readable violation messages (empty when safe).
    """
    is_safe: bool
    sanitized_input: str
    violations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_empty(text: str) -> Optional[str]:
    """Return a violation message if the text is empty or whitespace-only."""
    if not text or not text.strip():
        return "Input is empty or contains only whitespace."
    return None


def _check_length(text: str, max_length: int = MAX_INPUT_LENGTH) -> Optional[str]:
    """Return a violation message if the text exceeds the maximum allowed length."""
    if len(text) > max_length:
        return (
            f"Input length ({len(text)} chars) exceeds the maximum allowed "
            f"length of {max_length} characters."
        )
    return None


def _check_injection(text: str) -> Optional[str]:
    """Return a violation message if a prompt-injection pattern is detected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return (
                f"Potential prompt injection detected. "
                f"Matched pattern: '{pattern.pattern}'. "
                "This input has been blocked for safety."
            )
    return None


def _redact_pii(text: str) -> str:
    """Replace PII patterns with safe placeholder tokens.

    Redacts email addresses, phone numbers, and credit card numbers.

    Args:
        text: Raw input string.

    Returns:
        String with PII replaced by [REDACTED-EMAIL], [REDACTED-PHONE],
        and [REDACTED-CARD] respectively.
    """
    text = _EMAIL_PATTERN.sub("[REDACTED-EMAIL]", text)
    text = _PHONE_PATTERN.sub("[REDACTED-PHONE]", text)
    text = _CREDIT_CARD_PATTERN.sub("[REDACTED-CARD]", text)
    return text


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_guardrails(user_input: str) -> GuardrailResult:
    """Run the full security guardrail pipeline on a user input string.

    Checks (in order):
      1. Empty / whitespace rejection
      2. Length enforcement (max 4096 chars)
      3. Prompt-injection pattern detection
      4. PII redaction (email, phone, credit card)

    Args:
        user_input: The raw input string received from the user.

    Returns:
        GuardrailResult with is_safe flag, sanitized_input, and any violations.
    """
    violations: List[str] = []

    # --- Check 1: Empty input ---
    empty_violation = _check_empty(user_input)
    if empty_violation:
        return GuardrailResult(is_safe=False, sanitized_input="", violations=[empty_violation])

    # --- Check 2: Length ---
    length_violation = _check_length(user_input)
    if length_violation:
        violations.append(length_violation)
        return GuardrailResult(is_safe=False, sanitized_input=user_input, violations=violations)

    # --- Check 3: Injection ---
    injection_violation = _check_injection(user_input)
    if injection_violation:
        violations.append(injection_violation)
        return GuardrailResult(is_safe=False, sanitized_input=user_input, violations=violations)

    # --- Check 4: PII redaction (non-blocking, applied to safe inputs) ---
    sanitized = _redact_pii(user_input)

    return GuardrailResult(is_safe=True, sanitized_input=sanitized, violations=[])
