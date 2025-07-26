import asyncio
import time

from app.agent.data_analysis import DataAnalysis
from app.agent.musai import Musai
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger


async def run_flow():
    agents = {
        "musai": Musai(),
    }
    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()
    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        logger.info("🚀 Starting Planning Flow...")
        logger.info(
            "📋 Initialization may take 30-60 seconds for first run (browser + MCP setup)"
        )

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.info("🔄 Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
