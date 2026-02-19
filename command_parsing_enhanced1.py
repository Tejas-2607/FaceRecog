"""
command_parsing_enhanced.py
===========================
Flexible command parser for face recognition surveillance system.

Accepts natural language like:
  "detect person right to User1"
  "find the person to the left of John"
  "identify who is next to Alice on the right"
  "scan left side of Bob"
  "show me who's beside Sarah"
  "who is standing right of Mike"
  "look for someone left to Emma"

Extracts:
  - reference_person: the anchor person name
  - direction: "left" | "right"
  - valid: True/False
  - error: description if invalid
"""

import re


# ── Keyword banks ────────────────────────────────────────────────────────────

ACTION_WORDS = {
    "detect", "find", "identify", "scan", "show", "look",
    "search", "locate", "check", "spot", "see", "get",
    "who", "what", "is", "are", "there"
}

# Words that signal "right direction"
RIGHT_WORDS = {
    "right", "right-side", "rightside", "rightof",
    "rhs",
}

# Words that signal "left direction"
LEFT_WORDS = {
    "left", "left-side", "leftside", "leftof",
    "lhs",
}

# Connector / filler words to strip out
CONNECTOR_WORDS = {
    "to", "of", "the", "a", "an", "on", "at", "in", "is", "are",
    "who", "beside", "next", "near", "adjacent", "side", "person",
    "someone", "anybody", "anyone", "standing", "sitting", "by",
    "from", "for", "me", "us", "them", "their", "and",
}

# All noise words combined (action + connector) for name extraction
ALL_NOISE_WORDS = ACTION_WORDS | CONNECTOR_WORDS | RIGHT_WORDS | LEFT_WORDS | {
    "second", "first", "third", "last", "another",
}


class CommandParser:
    """
    Parses free-form surveillance commands into structured data.

    Usage:
        parser = CommandParser()
        result = parser.parse("find person to the right of Alice")
        # -> {'valid': True, 'reference_person': 'Alice', 'direction': 'right', ...}
    """

    def parse(self, command: str) -> dict:
        if not command or not command.strip():
            return self._error("Empty command")

        raw      = command.strip()
        text     = raw.lower()
        words    = text.split()

        # ── 1. Detect direction ──────────────────────────────────────────────
        direction = None
        dir_index = -1

        for i, w in enumerate(words):
            clean_w = w.strip(".,!?;:")
            if clean_w in RIGHT_WORDS:
                direction = "right"
                dir_index = i
                break
            if clean_w in LEFT_WORDS:
                direction = "left"
                dir_index = i
                break

        if direction is None:
            return self._error(
                "No direction found. Use 'left' or 'right' in your command. "
                "Example: 'detect person right to Alice'"
            )

        # ── 2. Extract reference person name ────────────────────────────────
        # Strategy: take the original-case tokens, remove all noise words,
        # and what remains (with length > 1 char) is the name candidate.
        # We prefer tokens AFTER the direction word (common pattern: "right of Alice")
        # but also look before it ("Alice on the right").

        orig_words = raw.split()  # preserve original casing

        # Try tokens after direction index first (most common pattern)
        name = self._extract_name(orig_words, dir_index + 1, len(orig_words))

        # If nothing found after, try before direction index
        if not name:
            name = self._extract_name(orig_words, 0, dir_index)

        if not name:
            return self._error(
                "Could not identify a person name. "
                "Example: 'find person to the right of Alice'"
            )

        return {
            "valid": True,
            "reference_person": name,
            "direction": direction,
            "raw_command": raw,
        }

    def _extract_name(self, words: list, start: int, end: int) -> str:
        """
        From a slice of the original-case word list, collect tokens that
        are NOT noise/direction words and have length > 1.
        Returns the joined result or empty string.
        """
        name_tokens = []
        for w in words[start:end]:
            clean = w.strip(".,!?;:").lower()
            if clean and clean not in ALL_NOISE_WORDS and len(clean) > 1:
                # Keep original casing for names
                name_tokens.append(w.strip(".,!?;:"))

        return " ".join(name_tokens) if name_tokens else ""

    def format_feedback(self, result: dict) -> str:
        if not result.get("valid"):
            return f"Invalid: {result.get('error', 'Unknown error')}"

        name      = result["reference_person"]
        direction = result["direction"]
        arrow     = "→" if direction == "right" else "←"
        return (
            f"Looking for person to the {direction.upper()} of "
            f"'{name}' {arrow}"
        )

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "valid": False,
            "error": msg,
            "reference_person": None,
            "direction": None,
            "raw_command": "",
        }
