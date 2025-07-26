"""
Enhanced LLM Module with Intelligent Vision Routing

This module extends the base LLM functionality with automatic routing
between text and vision models based on content analysis.
"""

import asyncio
import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.flow.vision_router import vision_router
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    """
    Token counting utility with perceptual awareness for visual content.
    """

    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        logger.debug(
            f"🔧 TokenCounter initialized with tokenizer: {type(tokenizer).__name__}"
        )

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        token_count = 0 if not text else len(self.tokenizer.encode(text))
        logger.debug(
            f"📊 Text token count: {token_count} tokens for text length: {len(text) if text else 0}"
        )
        return token_count

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions
        """
        detail = image_item.get("detail", "medium")
        logger.debug(f"🖼️ Processing image with detail level: {detail}")

        # For low detail, always return fixed token count
        if detail == "low":
            logger.debug(f"🖼️ Low detail image: {self.LOW_DETAIL_IMAGE_TOKENS} tokens")
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                token_count = self._calculate_high_detail_tokens(width, height)
                logger.debug(
                    f"🖼️ High detail image with dimensions {width}x{height}: {token_count} tokens"
                )
                return token_count
            else:
                # Default to high detail calculation without dimensions
                token_count = self._calculate_high_detail_tokens(
                    self.MAX_SIZE, self.MAX_SIZE
                )
                logger.debug(
                    f"🖼️ High detail image without dimensions: {token_count} tokens"
                )
                return token_count

        # Fallback for unknown detail levels
        logger.warning(f"🖼️ Unknown detail level '{detail}', using low detail tokens")
        return self.LOW_DETAIL_IMAGE_TOKENS

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """
        Calculate tokens for high detail images based on OpenAI's formula
        """
        # Scale to fit in 2048x2048 square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = min(self.MAX_SIZE / width, self.MAX_SIZE / height)
            width = int(width * scale)
            height = int(height * scale)

        # Scale shortest side to 768px
        if width < height:
            scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / width
        else:
            scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / height

        width = int(width * scale)
        height = int(height * scale)

        # Count 512px tiles
        tiles_wide = math.ceil(width / self.TILE_SIZE)
        tiles_high = math.ceil(height / self.TILE_SIZE)
        total_tiles = tiles_wide * tiles_high

        # Calculate tokens: 170 per tile + 85 base
        token_count = (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

        logger.debug(
            f"🖼️ High detail calculation: {width}x{height} -> {tiles_wide}x{tiles_high} tiles = {token_count} tokens"
        )
        return token_count

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """
        Count tokens for content that can be text or a list of text/image items
        """
        if isinstance(content, str):
            return self.count_text(content)

        total_tokens = 0
        for item in content:
            if isinstance(item, str):
                total_tokens += self.count_text(item)
            elif isinstance(item, dict):
                if item.get("type") == "image_url":
                    total_tokens += self.count_image(item)
                elif item.get("type") == "text":
                    total_tokens += self.count_text(item.get("text", ""))
                else:
                    logger.warning(f"🖼️ Unknown content item type: {item.get('type')}")

        return total_tokens

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Count tokens for tool calls"""
        total_tokens = 0
        for tool_call in tool_calls:
            # Count function name
            total_tokens += self.count_text(
                tool_call.get("function", {}).get("name", "")
            )
            # Count function arguments
            total_tokens += self.count_text(
                str(tool_call.get("function", {}).get("arguments", ""))
            )
            # Add format tokens
            total_tokens += self.FORMAT_TOKENS

        return total_tokens

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages"""
        total_tokens = 0
        for message in messages:
            # Add base message tokens
            total_tokens += self.BASE_MESSAGE_TOKENS

            # Count content tokens
            if "content" in message:
                total_tokens += self.count_content(message["content"])

            # Count tool calls if present
            if "tool_calls" in message:
                total_tokens += self.count_tool_calls(message["tool_calls"])

        return total_tokens


class EnhancedLLM:
    """
    Enhanced LLM with intelligent vision routing capabilities.

    Automatically routes requests to appropriate models:
    - Text-only requests → [llm] (e.g., qwen3)
    - Image-containing requests → [llm.vision] (e.g., llava:13b)
    """

    _instances: Dict[str, "EnhancedLLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            logger.debug(
                f"🔄 Creating new EnhancedLLM instance for config: {config_name}"
            )
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        else:
            logger.debug(
                f"🔄 Reusing existing EnhancedLLM instance for config: {config_name}"
            )
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            logger.info(f"🚀 Initializing EnhancedLLM with config: {config_name}")

            # Store the base config name for routing decisions
            self.base_config_name = config_name

            # Initialize with base config
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])

            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.base_url = llm_config.base_url
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.max_input_tokens = llm_config.max_input_tokens

            # Initialize token counter
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.token_counter = TokenCounter(self.tokenizer)
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize tokenizer: {e}")
                self.tokenizer = None
                self.token_counter = None

            # Initialize client based on API type
            self._initialize_client()

            # Track token usage
            self.total_input_tokens = 0
            self.total_completion_tokens = 0

    def _initialize_client(self):
        """Initialize the appropriate client based on API type"""
        if self.api_type.lower() == "azure":
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
            )
        elif self.api_type.lower() == "openai":
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        elif self.api_type.lower() == "ollama":
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def _get_optimal_config(self, message: str) -> str:
        """
        Determine the optimal configuration based on message content.

        Args:
            message: The message content to analyze

        Returns:
            Configuration name ('llm' or 'llm.vision')
        """
        routing = vision_router.route_request(message)
        optimal_config = routing["config"]

        logger.info(
            f"🧠 Vision router decision: {routing['reason']} -> {optimal_config}"
        )
        return optimal_config

    def _get_llm_for_config(self, config_name: str) -> "EnhancedLLM":
        """
        Get or create an LLM instance for the specified configuration.

        Args:
            config_name: The configuration name ('llm' or 'llm.vision')

        Returns:
            EnhancedLLM instance for the specified config
        """
        if config_name == self.base_config_name:
            return self

        # Create a new instance for the different config
        return EnhancedLLM(config_name=config_name)

    async def ask(self, message: str, **kwargs) -> str:
        """
        Enhanced ask method with automatic vision routing.

        Args:
            message: The message content
            **kwargs: Additional parameters

        Returns:
            Response from the appropriate model
        """
        # Determine optimal configuration
        optimal_config = self._get_optimal_config(message)

        # Get the appropriate LLM instance
        llm_instance = self._get_llm_for_config(optimal_config)

        # Use the base ask method from the appropriate instance
        return await llm_instance._ask_base(message, **kwargs)

    async def _ask_base(self, message: str, **kwargs) -> str:
        """
        Base ask method without routing logic.
        This is the original ask implementation.
        """
        # Implementation would be copied from the original LLM class
        # For now, this is a placeholder
        pass

    async def ask_with_images(self, message: str, images: List[str], **kwargs) -> str:
        """
        Enhanced ask_with_images method with automatic vision routing.

        Args:
            message: The message content
            images: List of image paths or URLs
            **kwargs: Additional parameters

        Returns:
            Response from the vision model
        """
        # For image requests, always use vision config
        vision_llm = self._get_llm_for_config("llm.vision")

        # Use the base ask_with_images method from the vision instance
        return await vision_llm._ask_with_images_base(message, images, **kwargs)

    async def _ask_with_images_base(
        self, message: str, images: List[str], **kwargs
    ) -> str:
        """
        Base ask_with_images method without routing logic.
        This is the original ask_with_images implementation.
        """
        # Implementation would be copied from the original LLM class
        # For now, this is a placeholder
        pass

    async def ask_tool(self, message: str, tools: List[dict], **kwargs) -> str:
        """
        Enhanced ask_tool method with automatic vision routing.

        Args:
            message: The message content
            tools: List of tool definitions
            **kwargs: Additional parameters

        Returns:
            Response from the appropriate model
        """
        # Determine optimal configuration
        optimal_config = self._get_optimal_config(message)

        # Get the appropriate LLM instance
        llm_instance = self._get_llm_for_config(optimal_config)

        # Use the base ask_tool method from the appropriate instance
        return await llm_instance._ask_tool_base(message, tools, **kwargs)

    async def _ask_tool_base(self, message: str, tools: List[dict], **kwargs) -> str:
        """
        Base ask_tool method without routing logic.
        This is the original ask_tool implementation.
        """
        # Implementation would be copied from the original LLM class
        # For now, this is a placeholder
        pass


# Create a factory function for easy usage
def create_enhanced_llm(config_name: str = "default") -> EnhancedLLM:
    """
    Factory function to create an enhanced LLM instance.

    Args:
        config_name: The base configuration name

    Returns:
        EnhancedLLM instance with vision routing capabilities
    """
    return EnhancedLLM(config_name=config_name)
