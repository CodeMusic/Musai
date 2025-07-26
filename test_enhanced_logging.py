#!/usr/bin/env python3
"""
Test script to demonstrate enhanced logging functionality.
This script shows the improved logging that informs users about long waits during initialization.
"""

import asyncio
import logging
import time

from app.agent.musai import Musai
from app.logger import logger

# Configure logging to see the improved messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


async def test_enhanced_logging():
    """Test the enhanced logging functionality."""
    print("🧪 Testing enhanced logging functionality...")
    print("=" * 60)

    # Create a Musai agent (this will show initialization logs)
    print("\n📝 Creating Musai agent...")
    start_time = time.time()

    agent = await Musai.create()

    elapsed = time.time() - start_time
    print(f"✅ Agent created in {elapsed:.2f} seconds")
    print("=" * 60)

    # Test a simple task
    task = "Create a simple Python script that prints 'Hello, World!'"

    print(f"\n📝 Task: {task}")
    print("=" * 60)

    start_time = time.time()
    result = await agent.run(task)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print(f"📄 Total execution time: {elapsed:.2f} seconds")
    print(f"📄 Result: {result[:200]}{'...' if len(result) > 200 else ''}")


if __name__ == "__main__":
    print("🚀 Enhanced Logging Test")
    print(
        "This test demonstrates the improved logging that informs users about long waits."
    )
    print(
        "You should see informative messages about browser initialization, MCP connections, etc."
    )
    print()

    asyncio.run(test_enhanced_logging())
