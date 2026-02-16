"""
Dashboard TUI for EvolveBot.

Stacked layout:
- Top: StatusBar (provider, model, goal, stats)
- Middle: ChatPanel (conversation)
- Bottom: 3-column (Skills | Sandbox | Logs)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.pretty import pprint
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.markdown import Markdown
from rich.console import Group


PROJECT_ROOT = Path(__file__).absolute().parent.parent
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
SKILLS_DIR = PROJECT_ROOT / "skills"


class StatusBar:
    """Top status bar showing provider, model, goal, and stats."""

    def __init__(self, provider: str = "", model: str = "", goal: str = ""):
        self.provider = provider
        self.model = model
        self.goal = goal
        self.turns = 0
        self.tokens = 0
        self.start_time = datetime.now()

    def update(self, provider: str = None, model: str = None, goal: str = None):
        if provider is not None:
            self.provider = provider
        if model is not None:
            self.model = model
        if goal is not None:
            self.goal = goal

    def increment_turn(self, tokens: int = 0):
        self.turns += 1
        self.tokens += tokens

    def render(self) -> Panel:
        duration = datetime.now() - self.start_time
        stats = f"Turns: {self.turns} | Tokens: ~{self.tokens} | Time: {duration.seconds // 60}m"

        content = Text()
        content.append(" Provider: ", style="bold cyan")
        content.append(self.provider or "N/A", style="cyan")
        content.append("  |  Model: ", style="bold cyan")
        content.append(self.model or "N/A", style="cyan")
        content.append("\n")
        content.append(" Goal: ", style="bold green")
        content.append(self.goal or "N/A", style="green")
        content.append("  |  ", style="dim")
        content.append(stats, style="dim")

        return Panel(content, title="Status", border_style="blue", height=5)


class ChatPanel:
    """Middle chat area with conversation history."""

    def __init__(self):
        self.messages = []
        self.max_messages = 100

    def add_message(self, role: str, content: str, tool_result: str = None):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "tool_result": tool_result,
                "timestamp": datetime.now(),
            }
        )
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def render(self, console: Console) -> Panel:
        if not self.messages:
            content = Text("No messages yet. Start a conversation!", style="dim italic")
            return Panel(content, title="Chat", border_style="green", padding=(1, 2))

        lines = []
        for msg in self.messages[-20:]:  # Last 20 messages
            role_color = {
                "system": "yellow",
                "user": "cyan",
                "assistant": "green",
                "tool": "magenta",
            }.get(msg["role"], "white")
            timestamp = msg["timestamp"].strftime("%H:%M")

            lines.append(
                Text(f"[{timestamp}] ", style="dim")
                + Text(f"{msg['role'].upper()}:", style=f"bold {role_color}")
            )
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(Text(content, style="white"))

            if msg.get("tool_result"):
                lines.append(
                    Text(f"  → Tool Result: {msg['tool_result'][:100]}...", style="dim magenta")
                )

        return Panel(
            Group(*lines),
            title="Chat",
            border_style="green",
            padding=(1, 2),
        )


class SkillsPanel:
    """Left bottom panel showing loaded skills."""

    def __init__(self):
        self.skills = []

    def update_skills(self, skills: list):
        self.skills = skills

    def render(self) -> Panel:
        if not self.skills:
            content = Text("No skills loaded", style="dim italic")
            return Panel(content, title="Skills", border_style="yellow", padding=(1, 1))

        table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 1))
        table.add_column("Skill", style="cyan")
        table.add_column("Triggers", style="dim")

        for skill in self.skills:
            triggers = ", ".join(skill.get("triggers", [])[:3])
            if len(skill.get("triggers", [])) > 3:
                triggers += "..."
            table.add_row(skill.get("name", "?"), triggers)

        return Panel(table, title="Skills", border_style="yellow", padding=(1, 1))


class SandboxPanel:
    """Middle bottom panel showing sandbox file tree."""

    def __init__(self, sandbox_path: Path = None):
        self.sandbox_path = sandbox_path or SANDBOX_DIR
        self.selected_file = None
        self.file_content = ""

    def refresh(self):
        if not self.sandbox_path.exists():
            return

    def render(self) -> Panel:
        if not self.sandbox_path.exists():
            return Panel(
                Text("Sandbox not found", style="red"), title="Sandbox", border_style="blue"
            )

        tree = Tree(f"📁 {self.sandbox_path.name}", guide_style="dim blue")
        self._build_tree(self.sandbox_path, tree)
        return Panel(tree, title="Sandbox", border_style="blue", padding=(1, 1))

    def _build_tree(self, path: Path, tree: Tree):
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items[:20]:  # Limit to 20 items
                if item.is_dir():
                    branch = tree.add(f"📁 {item.name}", style="bold cyan")
                    self._build_tree(item, branch)
                else:
                    icon = (
                        "📄"
                        if item.suffix in [".py", ".txt", ".md", ".json", ".yaml", ".yml"]
                        else "📎"
                    )
                    tree.add(f"{icon} {item.name}")
        except PermissionError:
            tree.add("[red]Permission denied[/red]")


class LogsPanel:
    """Right bottom panel showing execution logs."""

    def __init__(self, max_logs: int = 50):
        self.max_logs = max_logs
        self.logs = []

    def add_log(self, skill_name: str, args: dict, result: str, success: bool = True):
        self.logs.append(
            {
                "timestamp": datetime.now(),
                "skill": skill_name,
                "args": args,
                "result": result[:200] if result else "",
                "success": success,
            }
        )
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

    def render(self) -> Panel:
        if not self.logs:
            content = Text("No tool executions yet", style="dim italic")
            return Panel(content, title="Logs", border_style="magenta", padding=(1, 1))

        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        table.add_column("Time", style="dim", width=6)
        table.add_column("Skill", style="cyan", width=12)
        table.add_column("Result", style="white")

        for log in self.logs[-15:]:  # Last 15 logs
            time_str = log["timestamp"].strftime("%H:%M")
            status = "✓" if log["success"] else "✗"
            result_preview = (
                log["result"][:40] + "..." if len(log["result"]) > 40 else log["result"]
            )
            table.add_row(time_str, f"{status} {log['skill']}", result_preview)

        return Panel(table, title="Logs", border_style="magenta", padding=(1, 1))


class Dashboard:
    """Main dashboard with stacked layout."""

    def __init__(self, provider: str = "", model: str = "", goal: str = ""):
        self.console = Console()
        self.status = StatusBar(provider, model, goal)
        self.chat = ChatPanel()
        self.skills_panel = SkillsPanel()
        self.sandbox_panel = SandboxPanel()
        self.logs_panel = LogsPanel()

        self.layout = Layout()
        self._update_layout()

    def _update_layout(self):
        self.layout.split_column(
            Layout(name="status", size=5),
            Layout(name="chat", ratio=3),
            Layout(name="bottom", ratio=2),
        )
        self.layout["bottom"].split_row(
            Layout(name="skills", ratio=1),
            Layout(name="sandbox", ratio=1),
            Layout(name="logs", ratio=1),
        )

    def update_config(self, provider: str = None, model: str = None, goal: str = None):
        self.status.update(provider=provider, model=model, goal=goal)

    def update_skills(self, skills: list):
        self.skills_panel.update_skills(skills)

    def refresh_sandbox(self):
        self.sandbox_panel.refresh()

    def add_message(self, role: str, content: str, tool_result: str = None):
        self.chat.add_message(role, content, tool_result)

    def add_tool_log(self, skill_name: str, args: dict, result: str, success: bool = True):
        self.logs_panel.add_log(skill_name, args, result, success)

    def increment_turn(self, tokens: int = 0):
        self.status.increment_turn(tokens)

    def render(self):
        self.layout["status"].update(self.status.render())
        self.layout["chat"].update(self.chat.render(self.console))
        self.layout["skills"].update(self.skills_panel.render())
        self.layout["sandbox"].update(self.sandbox_panel.render())
        self.layout["logs"].update(self.logs_panel.render())
        return self.layout

    def print(self, message: str, style: str = ""):
        self.console.print(message, style=style)

    def input(self, prompt: str = "") -> str:
        return self.console.input(prompt)

    def clear(self):
        self.console.clear()


def create_dashboard(provider: str = "", model: str = "", goal: str = "") -> Dashboard:
    """Factory function to create a dashboard."""
    return Dashboard(provider=provider, model=model, goal=goal)
