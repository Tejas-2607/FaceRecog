"""
Enhanced Command Parsing Module
Handles complex user commands for face recognition system with improved accuracy
"""

import re
from typing import Dict, Optional, Tuple


class CommandParser:
    """
    Robust command parser for spatial face recognition queries.
    
    Supports patterns like:
    - "detect the person left to User1"
    - "find second person on right of User1"
    - "who is first to the left of User1"
    - "show me person right of User1"
    """
    
    def __init__(self):
        # Direction patterns
        self.direction_patterns = {
            'left': r'\b(left|lft)\b',
            'right': r'\b(right|rgt|rght)\b'
        }
        
        # Position patterns (first, second, etc.)
        self.position_patterns = {
            1: r'\b(first|1st|one)\b',
            2: r'\b(second|2nd|two)\b',
            3: r'\b(third|3rd|three)\b',
            4: r'\b(fourth|4th|four)\b'
        }
        
        # Reference person patterns
        self.reference_pattern = r'\b(?:of|to)\s+([A-Za-z0-9_]+)\b'
        
        # Action patterns
        self.action_patterns = {
            'detect': r'\b(detect|find|show|locate|identify)\b',
            'capture': r'\b(capture|take|snap|photo|picture|save)\b'
        }
    
    def parse(self, command: str) -> Dict[str, any]:
        """
        Parse user command and extract all components.
        
        Args:
            command: User input string
            
        Returns:
            Dictionary containing:
                - action: 'detect' or 'capture'
                - direction: 'left' or 'right'
                - position: integer (1, 2, 3, etc.)
                - reference_person: string (username)
                - valid: boolean
                - error: string (if not valid)
        """
        command_lower = command.lower().strip()
        
        result = {
            'action': None,
            'direction': None,
            'position': 1,  # Default to first person
            'reference_person': None,
            'valid': False,
            'error': None,
            'original_command': command
        }
        
        # 1. Extract action
        result['action'] = self._extract_action(command_lower)
        
        # 2. Extract direction
        result['direction'] = self._extract_direction(command_lower)
        if not result['direction']:
            result['error'] = "Direction (left/right) not specified"
            return result
        
        # 3. Extract position (first, second, etc.)
        result['position'] = self._extract_position(command_lower)
        
        # 4. Extract reference person
        result['reference_person'] = self._extract_reference_person(command)
        if not result['reference_person']:
            result['error'] = "Reference person not specified (e.g., 'of User1')"
            return result
        
        # Mark as valid if we got direction and reference person
        result['valid'] = True
        result['error'] = None
        
        return result
    
    def _extract_action(self, command: str) -> str:
        """Extract action type (detect or capture)"""
        for action, pattern in self.action_patterns.items():
            if re.search(pattern, command):
                return action
        return 'detect'  # Default action
    
    def _extract_direction(self, command: str) -> Optional[str]:
        """Extract spatial direction"""
        for direction, pattern in self.direction_patterns.items():
            if re.search(pattern, command):
                return direction
        return None
    
    def _extract_position(self, command: str) -> int:
        """Extract position number (first=1, second=2, etc.)"""
        for position, pattern in self.position_patterns.items():
            if re.search(pattern, command):
                return position
        return 1  # Default to first person
    
    def _extract_reference_person(self, command: str) -> Optional[str]:
        """
        Extract reference person name (case-sensitive).
        Handles patterns like "of User1", "to User1", etc.
        """
        match = re.search(self.reference_pattern, command)
        if match:
            return match.group(1)
        
        # Fallback: look for capitalized words that might be names
        words = command.split()
        for word in reversed(words):  # Check from end
            if word[0].isupper() and len(word) > 1:
                return word.strip('.,!?')
        
        return None
    
    def format_feedback(self, parse_result: Dict) -> str:
        """
        Generate human-readable feedback message.
        
        Args:
            parse_result: Output from parse() method
            
        Returns:
            Formatted string describing the parsed command
        """
        if not parse_result['valid']:
            return f"❌ Invalid command: {parse_result['error']}"
        
        position_text = {
            1: "first",
            2: "second", 
            3: "third",
            4: "fourth"
        }.get(parse_result['position'], f"{parse_result['position']}th")
        
        action_text = "Detecting" if parse_result['action'] == 'detect' else "Capturing"
        
        return (
            f"✓ {action_text} {position_text} person on "
            f"{parse_result['direction']} of {parse_result['reference_person']}"
        )


# Legacy function for backward compatibility
def parse_command(command: str) -> Optional[str]:
    """
    Legacy function that returns just the direction.
    Kept for backward compatibility.
    """
    parser = CommandParser()
    result = parser.parse(command)
    return result['direction'] if result['valid'] else None


# Example usage and testing
if __name__ == "__main__":
    parser = CommandParser()
    
    test_commands = [
        "detect the person right to User1",
        "find person on left of User1",
        "who is on my right of User1",
        "show me second person left of User1",
        "capture first person on right of User1",
        "take photo of person left to User1",
        "detect third person right of User1",
        "invalid command without direction"
    ]
    
    print("=" * 60)
    print("COMMAND PARSER TESTING")
    print("=" * 60)
    
    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"\nCommand: {cmd}")
        print(f"Result: {parser.format_feedback(result)}")
        if result['valid']:
            print(f"  - Action: {result['action']}")
            print(f"  - Direction: {result['direction']}")
            print(f"  - Position: {result['position']}")
            print(f"  - Reference: {result['reference_person']}")
