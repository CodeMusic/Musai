"""
Vision Router Module

Intelligently routes requests to appropriate models based on content type.
Text-only requests go to [llm], image-containing requests go to [llm.vision].
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional


class VisionRouter:
    """
    Routes requests to appropriate models based on content analysis.

    Uses psychological principles for decision-making:
    - Perceptual awareness: Detects visual content
    - Cognitive load balancing: Distributes processing appropriately
    - Response optimization: Ensures optimal model selection
    """

    def __init__(self):
        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }
        self.image_patterns = [
            r"<img[^>]+>",
            r"!\[.*?\]\(.*?\)",
            r"data:image/[^;]+;base64,",
            r"\.(jpg|jpeg|png|gif|bmp|tiff|webp)",
        ]

    def contains_visual_content(self, message: str) -> bool:
        """
        Analyzes message for visual content using perceptual awareness.

        Args:
            message: The message content to analyze

        Returns:
            True if visual content is detected, False otherwise
        """
        # Check for image file paths
        if any(ext in message.lower() for ext in self.image_extensions):
            return True

        # Check for image patterns (HTML img tags, markdown images, base64)
        for pattern in self.image_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        # Check for image-related keywords that suggest visual content
        visual_keywords = [
            "image",
            "picture",
            "photo",
            "screenshot",
            "capture",
            "visual",
            "chart",
            "graph",
            "diagram",
            "illustration",
        ]

        message_lower = message.lower()
        if any(keyword in message_lower for keyword in visual_keywords):
            return True

        return False

    def route_request(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Routes request to appropriate model based on content analysis.

        Args:
            message: The message content
            **kwargs: Additional request parameters

        Returns:
            Dictionary with routing configuration
        """
        has_visual_content = self.contains_visual_content(message)

        if has_visual_content:
            return {
                "config": "llm.vision",
                "reason": "perceptual_awareness_detected_visual_content",
                "model_type": "vision",
            }
        else:
            return {
                "config": "llm",
                "reason": "cognitive_processing_text_only",
                "model_type": "text",
            }

    def get_optimal_config(self, message: str) -> str:
        """
        Returns the optimal configuration name for the given message.

        Args:
            message: The message content

        Returns:
            Configuration name ('llm' or 'llm.vision')
        """
        routing = self.route_request(message)
        return routing["config"]


def create_vision_router() -> VisionRouter:
    """
    Factory function to create a vision router instance.

    Returns:
        Configured VisionRouter instance
    """
    return VisionRouter()


# Global router instance for easy access
vision_router = create_vision_router()
