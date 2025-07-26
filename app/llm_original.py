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
from app.logger import logger  # Assuming a logger is set up in your app
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

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")
        logger.debug(f"🖼️ Processing image with detail level: {detail}")

        # For low detail, always return fixed token count
        if detail == "low":
            logger.debug(f"🖼️ Low detail image: {self.LOW_DETAIL_IMAGE_TOKENS} tokens")
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                token_count = self._calculate_high_detail_tokens(width, height)
                logger.debug(
                    f"🖼️ High detail image with dimensions {width}x{height}: {token_count} tokens"
                )
                return token_count

        default_tokens = (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )
        logger.debug(f"🖼️ Default image tokens for detail '{detail}': {default_tokens}")
        return default_tokens

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        logger.debug(f"🔢 Calculating high detail tokens for image {width}x{height}")

        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            logger.debug(
                f"📏 Scaled to fit in {self.MAX_SIZE}x{self.MAX_SIZE}: {width}x{height}"
            )

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        logger.debug(
            f"📏 Scaled shortest side to {self.HIGH_DETAIL_TARGET_SHORT_SIDE}: {scaled_width}x{scaled_height}"
        )

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y
        logger.debug(f"🧩 Tile calculation: {tiles_x}x{tiles_y} = {total_tiles} tiles")

        # Step 4: Calculate final token count
        token_count = (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS
        logger.debug(
            f"📊 Final token count: {total_tiles} tiles × {self.HIGH_DETAIL_TILE_TOKENS} + {self.LOW_DETAIL_IMAGE_TOKENS} = {token_count}"
        )
        return token_count

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            logger.debug("📝 Empty content, returning 0 tokens")
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        logger.debug(f"📝 Processing multimodal content with {len(content)} items")
        for i, item in enumerate(content):
            if isinstance(item, str):
                item_tokens = self.count_text(item)
                token_count += item_tokens
                logger.debug(f"📝 Item {i} (text): {item_tokens} tokens")
            elif isinstance(item, dict):
                if "text" in item:
                    item_tokens = self.count_text(item["text"])
                    token_count += item_tokens
                    logger.debug(f"📝 Item {i} (text in dict): {item_tokens} tokens")
                elif "image_url" in item:
                    item_tokens = self.count_image(item)
                    token_count += item_tokens
                    logger.debug(f"📝 Item {i} (image): {item_tokens} tokens")

        logger.debug(f"📊 Total content tokens: {token_count}")
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        logger.debug(f"🔧 Processing {len(tool_calls)} tool calls")
        for i, tool_call in enumerate(tool_calls):
            if "function" in tool_call:
                function = tool_call["function"]
                name_tokens = self.count_text(function.get("name", ""))
                args_tokens = self.count_text(function.get("arguments", ""))
                tool_tokens = name_tokens + args_tokens
                token_count += tool_tokens
                logger.debug(
                    f"🔧 Tool call {i}: {name_tokens} + {args_tokens} = {tool_tokens} tokens"
                )

        logger.debug(f"📊 Total tool call tokens: {token_count}")
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens
        logger.debug(f"📊 Starting message token count with {len(messages)} messages")

        for i, message in enumerate(messages):
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            role = message.get("role", "")
            role_tokens = self.count_text(role)
            tokens += role_tokens

            # Add content tokens
            if "content" in message:
                content_tokens = self.count_content(message["content"])
                tokens += content_tokens

            # Add tool calls tokens
            if "tool_calls" in message:
                tool_tokens = self.count_tool_calls(message["tool_calls"])
                tokens += tool_tokens

            # Add name and tool_call_id tokens
            name_tokens = self.count_text(message.get("name", ""))
            tool_call_id_tokens = self.count_text(message.get("tool_call_id", ""))
            tokens += name_tokens + tool_call_id_tokens

            total_tokens += tokens
            logger.debug(
                f"📝 Message {i} ({role}): {tokens} tokens (role: {role_tokens}, content: {content_tokens if 'content' in message else 0}, tools: {tool_tokens if 'tool_calls' in message else 0}, name: {name_tokens}, tool_call_id: {tool_call_id_tokens})"
            )

        logger.info(
            f"📊 Total message tokens: {total_tokens} for {len(messages)} messages"
        )
        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            logger.debug(f"🔄 Creating new LLM instance for config: {config_name}")
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        else:
            logger.debug(f"🔄 Reusing existing LLM instance for config: {config_name}")
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            logger.info(f"🚀 Initializing LLM with config: {config_name}")

            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])

            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            logger.info(
                f"🤖 LLM Configuration: model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature}, api_type={self.api_type}"
            )

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            logger.debug(f"📊 Token limits: max_input_tokens={self.max_input_tokens}")

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
                logger.debug(f"🔧 Initialized tokenizer for model: {self.model}")
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.warning(
                    f"⚠️ Model {self.model} not found in tiktoken presets, using cl100k_base as default"
                )

            # Initialize client based on API type
            if self.api_type == "azure":
                logger.debug("🔧 Initializing Azure OpenAI client")
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                logger.debug("🔧 Initializing AWS Bedrock client")
                self.client = BedrockClient()
            else:
                logger.debug("🔧 Initializing OpenAI client")
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)
            logger.info(f"✅ LLM initialization complete for config: {config_name}")

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            logger.debug("📝 Empty text, returning 0 tokens")
            return 0
        token_count = len(self.tokenizer.encode(text))
        logger.debug(
            f"📊 Text token count: {token_count} tokens for text length: {len(text)}"
        )
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        logger.debug(f"📊 Counting tokens for {len(messages)} messages")
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens

        total_tokens = input_tokens + completion_tokens
        cumulative_total = self.total_input_tokens + self.total_completion_tokens

        logger.info(
            f"📊 Token usage update: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={total_tokens}, Cumulative Total={cumulative_total}"
        )

        # Log warning if approaching limits
        if self.max_input_tokens and self.total_input_tokens > (
            self.max_input_tokens * 0.8
        ):
            logger.warning(
                f"⚠️ Approaching input token limit: {self.total_input_tokens}/{self.max_input_tokens} ({self.total_input_tokens/self.max_input_tokens*100:.1f}%)"
            )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            total_required = self.total_input_tokens + input_tokens
            is_within_limit = total_required <= self.max_input_tokens
            logger.debug(
                f"📊 Token limit check: current={self.total_input_tokens}, required={input_tokens}, total={total_required}, limit={self.max_input_tokens}, within_limit={is_within_limit}"
            )
            return is_within_limit
        # If max_input_tokens is not set, always return True
        logger.debug("📊 No token limit set, always allowing")
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            error_msg = f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"
            logger.error(f"🚫 Token limit exceeded: {error_msg}")
            return error_msg

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        logger.debug(
            f"📝 Formatting {len(messages)} messages, supports_images={supports_images}"
        )
        formatted_messages = []

        for i, message in enumerate(messages):
            logger.debug(f"📝 Processing message {i}: {type(message).__name__}")

            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()
                logger.debug(
                    f"📝 Converted Message object to dict: {message.get('role', 'unknown')}"
                )

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    error_msg = f"Message dict must contain 'role' field: {message}"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                role = message.get("role", "")
                logger.debug(f"📝 Processing message with role: {role}")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    logger.debug(f"🖼️ Processing base64 image in message {i}")
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                        logger.debug("📝 Initialized empty content list for image")
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                        logger.debug("📝 Converted string content to text object")
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]
                        logger.debug(
                            f"📝 Converted list content to proper format with {len(message['content'])} items"
                        )

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )
                    logger.debug("📝 Added base64 image to content")

                    # Remove the base64_image field
                    del message["base64_image"]
                    logger.debug("📝 Removed base64_image field from message")
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    logger.warning(
                        f"⚠️ Model doesn't support images, removing base64_image from message {i}"
                    )
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                    logger.debug(f"📝 Added message {i} to formatted list")
                else:
                    logger.debug(f"📝 Skipping message {i} - no content or tool_calls")
            else:
                error_msg = f"Unsupported message type: {type(message)}"
                logger.error(f"❌ {error_msg}")
                raise TypeError(error_msg)

        # Validate all messages have required fields
        for i, msg in enumerate(formatted_messages):
            if msg["role"] not in ROLE_VALUES:
                error_msg = f"Invalid role: {msg['role']}"
                logger.error(f"❌ {error_msg} in message {i}")
                raise ValueError(error_msg)

        logger.info(f"✅ Successfully formatted {len(formatted_messages)} messages")
        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Ask the LLM a question with retry logic."""
        logger.info(
            f"🤖 Starting LLM ask request with {len(messages)} messages, stream={stream}"
        )

        try:
            # Prepare messages
            all_messages = []
            if system_msgs:
                logger.debug(f"📝 Adding {len(system_msgs)} system messages")
                all_messages.extend(system_msgs)
            all_messages.extend(messages)
            logger.debug(f"📝 Total messages to process: {len(all_messages)}")

            # Estimate input tokens
            input_tokens = self.count_tokens(
                " ".join([msg.content for msg in all_messages if msg.content])
            )
            logger.debug(f"📊 Estimated input tokens: {input_tokens}")

            # Check token limits
            if self.max_input_tokens and input_tokens > self.max_input_tokens:
                error_msg = f"Input tokens ({input_tokens}) exceed limit ({self.max_input_tokens})"
                logger.error(f"🚫 {error_msg}")
                raise TokenLimitExceeded(error_msg)

            # Prepare parameters
            params = {
                "model": self.model,
                "messages": [msg.dict() for msg in all_messages],
                "stream": stream,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
                logger.debug(
                    f"🔧 Using reasoning model parameters: max_completion_tokens={self.max_tokens}"
                )
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
                logger.debug(
                    f"🔧 Using standard model parameters: max_tokens={self.max_tokens}, temperature={params['temperature']}"
                )

            # Log the request
            logger.info(
                f"🤖 Sending request to {self.model}... (this may take 10-60 seconds)"
            )
            logger.debug(f"📝 Request parameters: {params}")

            # Non-streaming request
            if not stream:
                logger.debug("📡 Making non-streaming API call")
                response = await self.client.chat.completions.create(**params)
                completion_text = response.choices[0].message.content or ""
                completion_tokens = self.count_tokens(completion_text)

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                logger.info(
                    f"✅ Received response from {self.model} ({completion_tokens} tokens)"
                )
                logger.debug(
                    f"📝 Response content length: {len(completion_text)} characters"
                )
                return completion_text

            # Streaming request, For streaming, update estimated token count before making the request
            logger.debug("📡 Making streaming API call")
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            chunk_count = 0
            logger.debug("📡 Starting to process streaming response")

            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                chunk_count += 1
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                logger.error("❌ Empty response from streaming LLM")
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"✅ Received streaming response from {self.model} ({completion_tokens} tokens, {chunk_count} chunks)"
            )
            logger.debug(
                f"📝 Streaming response content length: {len(full_response)} characters"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            logger.error("🚫 Token limit exceeded, not retrying")
            raise
        except ValueError:
            logger.exception(f"❌ Validation error in ask method")
            raise
        except OpenAIError as oe:
            logger.exception(f"❌ OpenAI API error in ask method")
            if isinstance(oe, AuthenticationError):
                logger.error("🔑 Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error(
                    "⏰ Rate limit exceeded. Consider increasing retry attempts."
                )
            elif isinstance(oe, APIError):
                logger.error(f"🌐 API error: {oe}")
            raise
        except Exception:
            logger.exception(f"❌ Unexpected error in ask method")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        logger.info(
            f"🖼️ Starting LLM ask_with_images request with {len(messages)} messages and {len(images)} images"
        )

        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                error_msg = f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            logger.debug(f"✅ Model {self.model} supports images")

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)
            logger.debug(
                f"📝 Formatted {len(formatted_messages)} messages with image support"
            )

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                error_msg = "The last message must be from the user to attach images"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Process the last user message to include images
            last_message = formatted_messages[-1]
            logger.debug(
                f"📝 Processing last user message to attach {len(images)} images"
            )

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content if isinstance(content, list) else []
            )
            logger.debug(
                f"📝 Converted content to multimodal format with {len(multimodal_content)} items"
            )

            # Add images to content
            for i, image in enumerate(images):
                logger.debug(f"🖼️ Processing image {i}: {type(image).__name__}")
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                    logger.debug(f"🖼️ Added string image URL: {image[:50]}...")
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                    logger.debug(f"🖼️ Added dict image with URL: {image['url'][:50]}...")
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                    logger.debug(f"🖼️ Added pre-formatted image object")
                else:
                    error_msg = f"Unsupported image format: {image}"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

            # Update the message with multimodal content
            last_message["content"] = multimodal_content
            logger.debug(
                f"📝 Updated last message with {len(multimodal_content)} content items"
            )

            # Add system messages if provided
            if system_msgs:
                logger.debug(f"📝 Adding {len(system_msgs)} system messages")
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            logger.debug(f"📊 Calculated input tokens: {input_tokens}")

            if not self.check_token_limit(input_tokens):
                error_msg = self.get_limit_error_message(input_tokens)
                logger.error(f"🚫 {error_msg}")
                raise TokenLimitExceeded(error_msg)

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
                logger.debug(
                    f"🔧 Using reasoning model parameters: max_completion_tokens={self.max_tokens}"
                )
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
                logger.debug(
                    f"🔧 Using standard model parameters: max_tokens={self.max_tokens}, temperature={params['temperature']}"
                )

            # Handle non-streaming request
            if not stream:
                logger.debug("📡 Making non-streaming API call with images")
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    logger.error("❌ Empty or invalid response from LLM")
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                logger.info(f"✅ Received response from {self.model} with images")
                return response.choices[0].message.content

            # Handle streaming request
            logger.debug("📡 Making streaming API call with images")
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            chunk_count = 0
            logger.debug("📡 Starting to process streaming response with images")

            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                chunk_count += 1
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                logger.error("❌ Empty response from streaming LLM")
                raise ValueError("Empty response from streaming LLM")

            logger.info(
                f"✅ Received streaming response from {self.model} with images ({chunk_count} chunks)"
            )
            return full_response

        except TokenLimitExceeded:
            logger.error("🚫 Token limit exceeded in ask_with_images, not retrying")
            raise
        except ValueError as ve:
            logger.error(f"❌ Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"❌ OpenAI API error in ask_with_images: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("🔑 Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error(
                    "⏰ Rate limit exceeded. Consider increasing retry attempts."
                )
            elif isinstance(oe, APIError):
                logger.error(f"🌐 API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        logger.info(
            f"🔧 Starting LLM ask_tool request with {len(messages)} messages, {len(tools) if tools else 0} tools"
        )

        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                error_msg = f"Invalid tool_choice: {tool_choice}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            logger.debug(f"✅ Tool choice validated: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS
            logger.debug(f"🖼️ Model {self.model} supports images: {supports_images}")

            # Format messages
            if system_msgs:
                logger.debug(f"📝 Formatting {len(system_msgs)} system messages")
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                logger.debug(f"📝 Formatting {len(messages)} messages")
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)
            logger.debug(f"📊 Base input tokens: {input_tokens}")

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                logger.debug(f"🔧 Processing {len(tools)} tools for token counting")
                for i, tool in enumerate(tools):
                    tool_tokens = self.count_tokens(str(tool))
                    tools_tokens += tool_tokens
                    logger.debug(f"🔧 Tool {i}: {tool_tokens} tokens")

            input_tokens += tools_tokens
            logger.debug(f"📊 Total input tokens (messages + tools): {input_tokens}")

            # Check if token limits are exceeded
            logger.debug(f"🔍 Checking token limits...")
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                logger.error(f"🚫 {error_message}")
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)
            logger.debug(f"✅ Token limits check passed")

            # Validate tools if provided
            if tools:
                logger.debug(f"🔧 Validating {len(tools)} tools")
                for i, tool in enumerate(tools):
                    if not isinstance(tool, dict) or "type" not in tool:
                        error_msg = f"Tool {i} must be a dict with 'type' field: {tool}"
                        logger.error(f"❌ {error_msg}")
                        raise ValueError(error_msg)
                logger.debug("✅ All tools validated successfully")
            else:
                logger.debug("✅ No tools to validate")

            # Set up the completion request
            logger.debug(f"🔧 Setting up API parameters...")
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
                logger.debug(
                    f"🔧 Using reasoning model parameters: max_completion_tokens={self.max_tokens}"
                )
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
                logger.debug(
                    f"🔧 Using standard model parameters: max_tokens={self.max_tokens}, temperature={params['temperature']}"
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            logger.debug(f"✅ API parameters configured")
            logger.debug(f"📡 Making tool API call with timeout: {timeout}s")
            logger.debug(f"📝 Request parameters: {params}")

            logger.info(f"🌐 Sending API request to {self.model}...")
            try:
                # Add timeout handling
                response: ChatCompletion = await asyncio.wait_for(
                    self.client.chat.completions.create(**params), timeout=timeout
                )
                logger.info(f"✅ API request completed successfully")
            except asyncio.TimeoutError:
                logger.error(f"⏰ API request timed out after {timeout} seconds")
                raise
            except Exception as api_error:
                logger.error(
                    f"❌ API request failed: {type(api_error).__name__}: {api_error}"
                )
                raise

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                logger.warning("⚠️ Invalid or empty response from LLM")
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            logger.info(f"✅ Received tool response from {self.model}")
            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            logger.error("🚫 Token limit exceeded in ask_tool, not retrying")
            raise
        except ValueError as ve:
            logger.error(f"❌ Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"❌ OpenAI API error in ask_tool: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("🔑 Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error(
                    "⏰ Rate limit exceeded. Consider increasing retry attempts."
                )
            elif isinstance(oe, APIError):
                logger.error(f"🌐 API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in ask_tool: {e}")
            raise
