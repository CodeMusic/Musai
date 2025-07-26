#!/usr/bin/env python3
"""
Test script to demonstrate improved logging functionality.
This script shows how the enhanced logging provides better context
about what agents are doing during execution.
"""

import asyncio
import logging

from app.agent.musai import Musai
from app.flow.planning import PlanningFlow

# Configure logging to see the improved messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


async def test_agent_logging():
    """Test the improved agent logging functionality."""
    print("🧪 Testing improved agent logging...")

    # Create a Musai agent
    agent = Musai()

    # Test a simple task
    task = "Create a simple Python script that prints 'Hello, World!'"

    print(f"\n📝 Task: {task}")
    print("=" * 60)

    result = await agent.run(task)

    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print(f"📄 Result: {result[:200]}{'...' if len(result) > 200 else ''}")


async def test_planning_flow_logging():
    """Test the improved planning flow logging functionality."""
    print("\n🧪 Testing improved planning flow logging...")

    # Create a planning flow with a Musai agent
    agent = Musai()
    flow = PlanningFlow(agents={"musai": agent})

    # Test a multi-step task
    task = "Create a data analysis script that reads a CSV file and generates a summary report"

    print(f"\n📝 Task: {task}")
    print("=" * 60)

    result = await flow.execute(task)

    print("\n" + "=" * 60)
    print("✅ Planning flow test completed!")
    print(f"📄 Result: {result[:200]}{'...' if len(result) > 200 else ''}")


async def main():
    """Run all tests."""
    print("🚀 Starting improved logging tests...")

    try:
        await test_agent_logging()
        await test_planning_flow_logging()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
