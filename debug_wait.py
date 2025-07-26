#!/usr/bin/env python3
"""
Debug script to identify where the system is getting stuck.
This script will help pinpoint exactly where long waits are occurring.
"""

import asyncio
import logging
import time
import traceback

from app.agent.musai import Musai
from app.logger import logger

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


async def debug_initialization():
    """Debug the initialization process step by step."""
    print("🔍 DEBUG: Starting initialization process...")

    try:
        # Step 1: Create Musai instance
        print("\n📝 Step 1: Creating Musai instance...")
        start_time = time.time()

        agent = Musai()
        elapsed = time.time() - start_time
        print(f"✅ Musai instance created in {elapsed:.2f} seconds")

        # Step 2: Initialize MCP servers
        print("\n📝 Step 2: Initializing MCP servers...")
        start_time = time.time()

        await agent.initialize_mcp_servers()
        elapsed = time.time() - start_time
        print(f"✅ MCP servers initialized in {elapsed:.2f} seconds")

        # Step 3: Test a simple task
        print("\n📝 Step 3: Testing simple task...")
        start_time = time.time()

        task = "Say hello world"
        result = await agent.run(task)
        elapsed = time.time() - start_time
        print(f"✅ Task completed in {elapsed:.2f} seconds")
        print(f"📄 Result: {result[:200]}{'...' if len(result) > 200 else ''}")

    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        raise


async def debug_browser_initialization():
    """Debug browser initialization specifically."""
    print("\n🔍 DEBUG: Testing browser initialization...")

    try:
        from app.tool.browser_use_tool import BrowserUseTool

        print("📝 Creating browser tool...")
        start_time = time.time()

        browser_tool = BrowserUseTool()
        elapsed = time.time() - start_time
        print(f"✅ Browser tool created in {elapsed:.2f} seconds")

        print("📝 Testing browser initialization...")
        start_time = time.time()

        # This will trigger browser initialization
        context = await browser_tool._ensure_browser_initialized()
        elapsed = time.time() - start_time
        print(f"✅ Browser initialized in {elapsed:.2f} seconds")

        # Clean up
        await browser_tool.cleanup()
        print("🧹 Browser cleanup completed")

    except Exception as e:
        print(f"❌ Error during browser initialization: {str(e)}")
        print(f"🔍 Traceback: {traceback.format_exc()}")


async def debug_llm_calls():
    """Debug LLM API calls."""
    print("\n🔍 DEBUG: Testing LLM calls...")

    try:
        from app.llm import LLM

        print("📝 Creating LLM instance...")
        start_time = time.time()

        llm = LLM()
        elapsed = time.time() - start_time
        print(f"✅ LLM instance created in {elapsed:.2f} seconds")

        print("📝 Testing simple LLM call...")
        start_time = time.time()

        from app.schema import Message

        messages = [Message.user_message("Say hello")]

        response = await llm.ask(messages)
        elapsed = time.time() - start_time
        print(f"✅ LLM call completed in {elapsed:.2f} seconds")
        print(f"📄 Response: {response[:100]}{'...' if len(response) > 100 else ''}")

    except Exception as e:
        print(f"❌ Error during LLM call: {str(e)}")
        print(f"🔍 Traceback: {traceback.format_exc()}")


async def main():
    """Run all debug tests."""
    print("🚀 Starting debug tests...")
    print("This will help identify where the system is getting stuck.")
    print("=" * 60)

    try:
        # Test 1: LLM calls
        await debug_llm_calls()

        # Test 2: Browser initialization
        await debug_browser_initialization()

        # Test 3: Full initialization
        await debug_initialization()

        print("\n" + "=" * 60)
        print("✅ All debug tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Debug test failed: {str(e)}")
        print("🔍 Check the logs above to identify where the issue occurs.")


if __name__ == "__main__":
    asyncio.run(main())
