import argparse
import asyncio

from app.agent.musai import Musai
from app.logger import logger


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Musai agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    args = parser.parse_args()

    logger.info("🚀 Starting Musai agent...")
    logger.info(
        "📋 Initialization may take 30-60 seconds for first run (browser + MCP setup)"
    )

    # Create and initialize Musai agent
    agent = await Musai.create()
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("🔄 Processing your request...")
        await agent.run(prompt)
        logger.info("✅ Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        logger.info("🧹 Cleaning up agent resources...")
        await agent.cleanup()
        logger.info("👋 Musai agent shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
