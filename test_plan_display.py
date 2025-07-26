#!/usr/bin/env python3
"""
Test script to demonstrate the beautiful plan display functionality.
This script shows how the plan guide looks with the original request and numbered steps.
"""

import asyncio
from app.flow.plan_display import PlanDisplay

async def test_plan_display():
    """Test the beautiful plan display functionality."""
    print("🧪 Testing beautiful plan display...")

    # Sample plan data
    original_request = "Create a data analysis script that reads a CSV file, performs statistical analysis, and generates a comprehensive report with visualizations"

    plan_data = {
        "title": "Data Analysis Pipeline",
        "plan_id": "test_plan_001",
        "steps": [
            "[MUSAI] Analyze the CSV file structure and data types",
            "[SEARCH] Research best practices for data visualization",
            "[CODE] Create Python script for data loading and preprocessing",
            "[PYTHON] Implement statistical analysis functions",
            "[FILE] Generate comprehensive analysis report",
            "[BROWSER] Create interactive visualizations",
            "[MUSAI] Validate results and finalize documentation"
        ],
        "step_statuses": [
            "completed",
            "in_progress",
            "not_started",
            "not_started",
            "not_started",
            "not_started",
            "not_started"
        ],
        "step_notes": [
            "Successfully analyzed file structure",
            "Currently researching visualization libraries",
            "",
            "",
            "",
            "",
            ""
        ]
    }

    # Create and display the plan guide
    plan_display = PlanDisplay()
    plan_display.display_plan_guide(original_request, plan_data)

    print("\n✅ Plan display test completed!")

if __name__ == "__main__":
    asyncio.run(test_plan_display())