import json
import re
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich import box
from app.logger import logger

class PlanDisplay:
    """
    Beautiful console display for Musai plans with program guide styling.
    Shows original request and numbered steps with tool choices.
    """

    def __init__(self):
        self.console = Console()

    def display_plan_guide(self, original_request: str, plan_data: Dict) -> None:
        """
        Display a beautiful program guide showing the original request and plan steps.

        Args:
            original_request: The original user request
            plan_data: The plan data containing title, steps, and statuses
        """
        try:
            # Create the main program guide
            self._display_header(original_request)
            self._display_plan_overview(plan_data)
            self._display_steps_guide(plan_data)
            self._display_footer()

        except Exception as e:
            logger.error(f"Error displaying plan guide: {e}")
            # Fallback to simple display
            self._display_fallback(original_request, plan_data)

    def _display_header(self, original_request: str) -> None:
        """Display the header with the original request."""
        # Create a beautiful header panel
        header_text = Text()
        header_text.append("🎭 ", style="bold magenta")
        header_text.append("MUSAI", style="bold cyan")
        header_text.append(" PROGRAM GUIDE", style="bold white")

        # Format the original request
        request_lines = self._wrap_text(original_request, width=80)
        request_text = Text()
        request_text.append("\n📋 ", style="bold yellow")
        request_text.append("ORIGINAL REQUEST:", style="bold white")
        request_text.append("\n")
        for line in request_lines:
            request_text.append(f"   {line}\n", style="dim white")

        # Create the header panel
        header_panel = Panel(
            Align.center(header_text + request_text),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2)
        )

        self.console.print(header_panel)
        self.console.print()

    def _display_plan_overview(self, plan_data: Dict) -> None:
        """Display plan overview with statistics."""
        title = plan_data.get("title", "Untitled Plan")
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])

        # Calculate statistics
        total_steps = len(steps)
        completed = sum(1 for status in step_statuses if status == "completed")
        in_progress = sum(1 for status in step_statuses if status == "in_progress")
        not_started = sum(1 for status in step_statuses if status == "not_started")
        blocked = sum(1 for status in step_statuses if status == "blocked")

        progress_percentage = (completed / total_steps * 100) if total_steps > 0 else 0

        # Create overview table
        overview_table = Table(
            title="📊 PLAN OVERVIEW",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        overview_table.add_column("Metric", style="bold white")
        overview_table.add_column("Value", style="cyan")
        overview_table.add_column("Status", style="green")

        overview_table.add_row("Plan Title", title, "📋")
        overview_table.add_row("Total Steps", str(total_steps), "📝")
        overview_table.add_row("Completed", str(completed), "✅")
        overview_table.add_row("In Progress", str(in_progress), "🔄")
        overview_table.add_row("Not Started", str(not_started), "⏳")
        overview_table.add_row("Blocked", str(blocked), "⚠️")
        overview_table.add_row("Progress", f"{progress_percentage:.1f}%", "📈")

        self.console.print(overview_table)
        self.console.print()

    def _display_steps_guide(self, plan_data: Dict) -> None:
        """Display the numbered steps with tool choices like a program guide."""
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])
        step_notes = plan_data.get("step_notes", [])

        # Create the steps guide table
        steps_table = Table(
            title="🎬 EXECUTION STEPS",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=100
        )

        steps_table.add_column("#", style="bold white", width=3)
        steps_table.add_column("Status", style="bold", width=8)
        steps_table.add_column("Step Description", style="white", width=50)
        steps_table.add_column("Tool Choice", style="cyan", width=25)
        steps_table.add_column("Notes", style="dim", width=15)

        for i, (step, status, notes) in enumerate(zip(steps, step_statuses, step_notes)):
            # Determine status icon and style
            status_icon, status_style = self._get_status_display(status)

            # Extract tool choice from step description
            tool_choice = self._extract_tool_choice(step)

            # Format step description (remove tool choice from display)
            step_display = self._format_step_description(step)

            # Format notes
            notes_display = notes if notes else ""

            steps_table.add_row(
                str(i + 1),
                f"{status_icon} {status.replace('_', ' ').title()}",
                step_display,
                tool_choice,
                notes_display,
                style=status_style
            )

        self.console.print(steps_table)
        self.console.print()

    def _get_status_display(self, status: str) -> tuple[str, str]:
        """Get the display icon and style for a step status."""
        status_map = {
            "completed": ("✅", "green"),
            "in_progress": ("🔄", "yellow"),
            "not_started": ("⏳", "dim"),
            "blocked": ("⚠️", "red")
        }
        return status_map.get(status, ("❓", "dim"))

    def _extract_tool_choice(self, step: str) -> str:
        """Extract tool choice from step description."""
        # Look for tool patterns like [TOOL_NAME] or [AGENT_NAME]
        tool_match = re.search(r'\[([A-Z_]+)\]', step)
        if tool_match:
            tool_name = tool_match.group(1)
            # Map tool names to friendly names
            tool_mapping = {
                "SEARCH": "🔍 Web Search",
                "BROWSER": "🌐 Browser",
                "CODE": "💻 Code Execution",
                "FILE": "📁 File Operations",
                "PLANNING": "📋 Planning",
                "MUSAI": "🎭 Musai Agent",
                "DATA_ANALYSIS": "📊 Data Analysis",
                "PYTHON": "🐍 Python Execution",
                "BASH": "💻 Terminal",
                "MCP": "🔗 MCP Tools"
            }
            return tool_mapping.get(tool_name, f"🔧 {tool_name}")
        return "🎯 General"

    def _format_step_description(self, step: str) -> str:
        """Format step description by removing tool choice markers."""
        # Remove tool choice markers like [TOOL_NAME]
        formatted = re.sub(r'\[[A-Z_]+\]\s*', '', step)
        return formatted.strip()

    def _display_footer(self) -> None:
        """Display the footer with execution information."""
        footer_text = Text()
        footer_text.append("🎭 ", style="bold magenta")
        footer_text.append("MUSAI", style="bold cyan")
        footer_text.append(" - Ready for execution! 🚀", style="bold white")

        footer_panel = Panel(
            Align.center(footer_text),
            border_style="green",
            box=box.SIMPLE,
            padding=(0, 2)
        )

        self.console.print(footer_panel)
        self.console.print()

    def _display_fallback(self, original_request: str, plan_data: Dict) -> None:
        """Fallback display if rich formatting fails."""
        print("\n" + "="*80)
        print("🎭 MUSAI PROGRAM GUIDE")
        print("="*80)
        print(f"📋 ORIGINAL REQUEST: {original_request}")
        print("-"*80)

        title = plan_data.get("title", "Untitled Plan")
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])

        print(f"📊 PLAN: {title}")
        print(f"📝 TOTAL STEPS: {len(steps)}")
        print("-"*80)
        print("🎬 EXECUTION STEPS:")
        print("-"*80)

        for i, (step, status) in enumerate(zip(steps, step_statuses)):
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄",
                "not_started": "⏳",
                "blocked": "⚠️"
            }.get(status, "❓")

            tool_choice = self._extract_tool_choice(step)
            step_display = self._format_step_description(step)

            print(f"{i+1:2d}. {status_icon} {step_display}")
            print(f"    🔧 Tool: {tool_choice}")
            print()

        print("="*80)
        print("🎭 MUSAI - Ready for execution! 🚀")
        print("="*80)

    def _wrap_text(self, text: str, width: int = 80) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + word) <= width:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        return lines