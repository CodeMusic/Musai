import asyncio
import json
import re
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import (TOOL_CHOICE_TYPE, AgentState, Function, Message,
                        ToolCall, ToolChoice)
from app.tool import CreateChatCompletion, Terminate, ToolCollection

TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    current_base64_image: Optional[str] = None

    # Loop detection for preventing infinite retries
    recent_tool_results: List[dict] = Field(default_factory=list)
    max_result_history: int = Field(default=10)
    loop_detection_threshold: int = Field(default=2)

    # Enhanced monitoring and backoff
    tool_failure_counts: dict = Field(default_factory=dict)  # Track failures per tool
    max_tool_failures: int = Field(
        default=5
    )  # Max failures before suggesting alternatives

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    def _is_similar_result(self, result1: str, result2: str) -> bool:
        """Check if two results are substantially similar (indicating a loop)."""
        if not result1 or not result2:
            return False

        # Remove timestamps and variable content for comparison
        import re

        clean1 = re.sub(r"\d{4}-\d{2}-\d{2}.*?\|", "", result1)
        clean1 = re.sub(r"step \d+", "step X", clean1.lower())
        clean2 = re.sub(r"\d{4}-\d{2}-\d{2}.*?\|", "", result2)
        clean2 = re.sub(r"step \d+", "step X", clean2.lower())

        # For very short results (like single words), require exact match
        if len(clean1) < 50 or len(clean2) < 50:
            return clean1.strip() == clean2.strip()

        # For longer results, check similarity ratio
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity > 0.85  # 85% similarity threshold

    def _detect_tool_loop(self, tool_name: str, tool_args: dict, result: str) -> bool:
        """Detect if we're in a tool execution loop."""
        current_call = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": str(result),
            "step": self.current_step,
        }

        # Check for similar recent calls
        similar_count = 0
        for past_call in self.recent_tool_results[-5:]:  # Check last 5 calls
            if (
                past_call["tool_name"] == tool_name
                and past_call["tool_args"] == tool_args
                and self._is_similar_result(past_call["result"], current_call["result"])
            ):
                similar_count += 1

        # Add current call to history
        self.recent_tool_results.append(current_call)

        # Keep history size manageable
        if len(self.recent_tool_results) > self.max_result_history:
            self.recent_tool_results.pop(0)

        is_loop = similar_count >= self.loop_detection_threshold
        if is_loop:
            logger.warning(
                f"🔄 LOOP DETECTED: Tool '{tool_name}' has returned similar results {similar_count + 1} times"
            )
        return is_loop

    def _handle_detected_loop(self, tool_name: str, result: str) -> str:
        """Handle when a tool loop is detected."""
        logger.error(
            f"🚨 CRITICAL: Loop detected for tool '{tool_name}' - BLOCKING RETRY to prevent infinite loop"
        )

        # Specific handling for browser_use loops
        if tool_name == "browser_use":
            if "extract_content" in str(result):
                return (
                    f"🚨 BROWSER EXTRACTION LOOP BLOCKED: Page content extraction failed {self.loop_detection_threshold} times. "
                    f"**STOP IMMEDIATELY** - Do not retry the same extraction. Instead:\n"
                    f"1. 🔍 Use web_search to find the information on different sites\n"
                    f"2. 🌐 Try completely different URLs (not just variations)\n"
                    f"3. 📝 Try a broader extraction goal like 'get all visible text'\n"
                    f"4. 🔄 Use a different approach entirely (different tool)\n\n"
                    f"BLOCKED RESULT: {result}"
                )
            elif "go_to_url" in str(
                result
            ) or "Page loaded but contains error content" in str(result):
                return (
                    f"🚨 BROWSER NAVIGATION LOOP BLOCKED: URL navigation failed {self.loop_detection_threshold} times. "
                    f"**STOP TRYING THIS URL** - The website is blocking access or doesn't have the content. Instead:\n"
                    f"1. 🔍 **USE WEB_SEARCH IMMEDIATELY** - Search for the information on Google\n"
                    f"2. 🌐 Try completely different websites (not BRP/SeaDoo official sites)\n"
                    f"3. 📰 Look for news articles, reviews, or dealer sites with the information\n"
                    f"4. 🛍️ Try marketplace sites like AutoTrader, Kijiji, or boat dealers\n"
                    f"5. 💡 Search for '{url if 'url' in locals() else 'SeaDoo Spark'}' specifications on different sites\n\n"
                    f"BLOCKED RESULT: {result}"
                )
            else:
                return (
                    f"🚨 BROWSER TOOL LOOP BLOCKED: Browser action failed {self.loop_detection_threshold} times. "
                    f"**STOP RETRYING** - Switch to web_search or different approach immediately.\n\n"
                    f"BLOCKED RESULT: {result}"
                )

        # Generic loop handling
        return (
            f"⚠️ Tool loop detected: '{tool_name}' returned similar results {self.loop_detection_threshold} times. "
            f"**CRITICAL**: Do not retry the same action. Try alternative approaches instead.\n\n"
            f"Last result was: {result}"
        )

    def _parse_tool_calls_from_content(self, content: str) -> List[ToolCall]:
        """
        Parse tool calls from content when LLM returns tool call info as text.
        This is a fallback for when the model doesn't properly format tool calls.
        """
        tool_calls = []

        try:
            # First try to parse the entire content as JSON if it looks like a single tool call
            content_stripped = content.strip()
            if content_stripped.startswith("{") and content_stripped.endswith("}"):
                try:
                    tool_data = json.loads(content_stripped)
                    if "name" in tool_data and "arguments" in tool_data:
                        # Create a proper ToolCall object
                        function = Function(
                            name=tool_data["name"],
                            arguments=(
                                json.dumps(tool_data["arguments"])
                                if isinstance(tool_data["arguments"], dict)
                                else tool_data["arguments"]
                            ),
                        )
                        tool_call = ToolCall(
                            id="call_0",
                            function=function,
                            type="function",
                        )

                        # Verify the tool exists
                        if tool_data["name"] in self.available_tools.tool_map:
                            tool_calls.append(tool_call)
                            logger.info(
                                f"📝 Parsed single tool call from content: {tool_data['name']}"
                            )
                            return tool_calls
                except json.JSONDecodeError:
                    pass

            # Try to find JSON-like structures in the content using improved regex
            # This handles the format from the logs: {"name": "...", "arguments": {...}}
            json_pattern = (
                r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
            )
            matches = re.findall(json_pattern, content, re.DOTALL)

            for i, match in enumerate(matches):
                try:
                    # Parse the JSON-like structure
                    tool_data = json.loads(match)

                    # Check if it has the expected structure
                    if "name" in tool_data and "arguments" in tool_data:
                        # Create a proper ToolCall object
                        function = Function(
                            name=tool_data["name"],
                            arguments=(
                                json.dumps(tool_data["arguments"])
                                if isinstance(tool_data["arguments"], dict)
                                else tool_data["arguments"]
                            ),
                        )
                        tool_call = ToolCall(
                            id=f"call_{i}",
                            function=function,
                            type="function",
                        )

                        # Verify the tool exists
                        if tool_data["name"] in self.available_tools.tool_map:
                            tool_calls.append(tool_call)
                            logger.info(
                                f"📝 Parsed tool call from content: {tool_data['name']}"
                            )

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.debug(f"Failed to parse tool calls from content: {e}")

        return tool_calls

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Fallback: If no tool_calls but content contains tool call information
        if not tool_calls and content and self.tool_choices != ToolChoice.NONE:
            parsed_tool_calls = self._parse_tool_calls_from_content(content)
            if parsed_tool_calls:
                self.tool_calls = tool_calls = parsed_tool_calls
                logger.info(
                    f"🔧 Applied fallback tool call parsing, found {len(tool_calls)} tools"
                )

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self.current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"✅ Tool '{command.function.name}' completed successfully! Result length: {len(str(result))} chars"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self.current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool with timeout protection
            logger.info(f"🔧 Activating tool: '{name}' with args: {args}")

            # Track tool usage for monitoring
            if name not in self.tool_failure_counts:
                self.tool_failure_counts[name] = 0

            # Add timeout protection for tool execution
            try:
                logger.info(f"⏱️ Executing tool '{name}'... (timeout: 300 seconds)")
                result = await asyncio.wait_for(
                    self.available_tools.execute(name=name, tool_input=args),
                    timeout=300,  # 5 minute timeout for tool execution
                )
                logger.info(f"✅ Tool '{name}' completed successfully")
            except asyncio.TimeoutError:
                error_msg = f"⏰ Tool '{name}' timed out after 300 seconds"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            # Format result for display (standard case)
            if name == "planning":
                # Enhanced logging for planning tool
                logger.info(f"📋 PLANNING TOOL RESULT:\n{str(result)}")
                observation = (
                    f"📋 PLANNING RESULT:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )
            else:
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )

            # **FAILURE TRACKING**: Monitor tool failures for exponential backoff
            if hasattr(result, "error") and result.error:
                self.tool_failure_counts[name] += 1
                logger.warning(
                    f"⚠️ Tool '{name}' failed (failure count: {self.tool_failure_counts[name]}): {result.error}"
                )
            else:
                # Reset failure count on success
                self.tool_failure_counts[name] = 0

            return observation

        except Exception as e:
            # Track failures
            if name not in self.tool_failure_counts:
                self.tool_failure_counts[name] = 0
            self.tool_failure_counts[name] += 1

            error_msg = f"Tool '{name}' execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
