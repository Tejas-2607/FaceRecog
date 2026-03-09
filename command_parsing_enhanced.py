"""
command_parsing_enhanced.py
===========================
Flexible command parser for face recognition surveillance system.

Supports positional commands:
  "detect second person right to User1"
  "find third person on left of Alice"
  "identify first person right of Bob"
  "could you take a photo of the 3rd person to my left"
  "locate our chief guest — he is 2 to the left of Raj"
  ... as well as single-person: "detect User1"
"""

import re

# ── Keyword banks ────────────────────────────────────────────────────────────

ACTION_WORDS = {
    "detect", "find", "identify", "scan", "show", "look",
    "search", "locate", "check", "spot", "see", "get",
    "who", "what", "is", "are", "there",
    # natural spoken-command words (change 11)
    "take", "grab", "capture", "snap", "photo", "picture",
    "point", "highlight", "can", "could", "would", "please",
}

RIGHT_WORDS = {
    "right", "right-side", "rightside", "rightof", "rhs",
}

LEFT_WORDS = {
    "left", "left-side", "leftside", "leftof", "lhs",
}

# ── Position keyword → ordinal number ────────────────────────────────────────
POSITION_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    # bare digit strings — so "3 to the left" is treated as position 3 (change 13)
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    # natural aliases
    "next": 1, "nearest": 1,
}

CONNECTOR_WORDS = {
    "to", "of", "the", "a", "an", "on", "at", "in", "is", "are",
    "who", "beside", "next", "near", "adjacent", "side", "person",
    "someone", "anybody", "anyone", "standing", "sitting", "by",
    "from", "for", "me", "us", "them", "their", "and",
    # spoken filler words — filter these out during name extraction (change 12)
    # e.g. "could you find him", "locate our chief guest Mr Mohan"
    "him", "her", "his", "hers", "it", "this", "that",
    "just", "with", "our", "my", "your", "we",
    "today", "guest", "chief", "sir", "madam",
    "mr", "ms", "mrs", "dr", "here", "now", "okay", "ok",
}

# Noise for name extraction — exclude POSITION_WORDS so they don't eat names
ALL_NOISE_WORDS = (ACTION_WORDS | CONNECTOR_WORDS | RIGHT_WORDS | LEFT_WORDS
                   | set(POSITION_WORDS.keys()))


class CommandParser:
    """
    Parses free-form surveillance commands into structured data.

    Returns dict with keys:
      valid, reference_person, direction, position, mode, raw_command
    """

    def parse(self, command: str) -> dict:
        if not command or not command.strip():
            return self._error("Empty command")

        raw   = command.strip()
        text  = raw.lower()
        words = text.split()

        # ── 1. Extract position word (first / second / third …) ─────────────
        position  = 1  # default
        pos_index = -1
        for i, w in enumerate(words):
            clean = w.strip(".,!?;:")
            if clean in POSITION_WORDS:
                position  = POSITION_WORDS[clean]
                pos_index = i
                break

        # ── 2. Detect direction ──────────────────────────────────────────────
        direction = None
        dir_index = -1
        for i, w in enumerate(words):
            clean = w.strip(".,!?;:")
            if clean in RIGHT_WORDS:
                direction = "right"
                dir_index = i
                break
            if clean in LEFT_WORDS:
                direction = "left"
                dir_index = i
                break

        # ── 3. Single-person mode (no direction) ─────────────────────────────
        if direction is None:
            orig_words = raw.split()
            name = self._extract_name(orig_words, 0, len(orig_words))
            if not name:
                return self._error(
                    "Could not identify a person name. "
                    "Examples: 'detect Alice' or 'find second person right to Bob'"
                )
            return {
                "valid":            True,
                "reference_person": name,
                "direction":        None,
                "position":         1,
                "raw_command":      raw,
                "mode":             "single",
            }

        # ── 4. Directional mode — extract reference person name ───────────────
        orig_words = raw.split()
        # Prefer tokens after direction word
        name = self._extract_name(orig_words, dir_index + 1, len(orig_words))
        if not name:
            name = self._extract_name(orig_words, 0, dir_index)

        if not name:
            return self._error(
                "Could not identify a person name. "
                "Example: 'find second person to the right of Alice'"
            )

        return {
            "valid":            True,
            "reference_person": name,
            "direction":        direction,
            "position":         position,
            "raw_command":      raw,
            "mode":             "directional",
        }

    def _extract_name(self, words: list, start: int, end: int) -> str:
        name_tokens = []
        for w in words[start:end]:
            clean = w.strip(".,!?;:").lower()
            if clean and clean not in ALL_NOISE_WORDS and len(clean) > 1:
                name_tokens.append(w.strip(".,!?;:"))
        return " ".join(name_tokens) if name_tokens else ""

    def format_feedback(self, result: dict) -> str:
        if not result.get("valid"):
            return f"Invalid: {result.get('error', 'Unknown error')}"

        name      = result["reference_person"]
        direction = result.get("direction")
        position  = result.get("position", 1)
        ordinal   = {1:"1st", 2:"2nd", 3:"3rd"}.get(position, f"{position}th")

        if direction is None:
            return f"Detecting '{name}' 🎯"

        arrow = "→" if direction == "right" else "←"
        return (f"Looking for {ordinal} person to the {direction.upper()} of "
                f"'{name}' {arrow}")

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "valid": False,
            "error": msg,
            "reference_person": None,
            "direction": None,
            "position": 1,
            "raw_command": "",
        }