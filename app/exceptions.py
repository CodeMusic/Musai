class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class MusaiError(Exception):
    """Base exception for all Musai errors"""


class TokenLimitExceeded(MusaiError):
    """Exception raised when the token limit is exceeded"""
