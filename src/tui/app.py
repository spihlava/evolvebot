"""
Textual TUI for EvolveBot.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    Header,
    Footer,
    Input,
    RichLog,
    TabbedContent,
    TabPane,
    DataTable,
    DirectoryTree,
    Static,
    Label,
    Markdown,
)
from textual.binding import Binding

from src.agent import (
    Brain,
    PromptManager,
    EXECUTOR_PROMPT,
    load_skills,
    get_skill_summary,
    parse_tool_call,
    run_tool,
    evolve,
    wipe_sandbox,
    wipe_skills,
    retrain_prompt,
    DEFAULT_EXECUTOR,
    DEFAULT_ARCHITECT,
    SANDBOX_DIR,
    SKILLS_DIR,
    LengthGrader,
    LLMGrader,
)


class EvolveBotApp(App):
    """The EvolveBot Textual Application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-input {
        dock: bottom;
        margin: 1 1;
    }

    #chat-container {
        height: 100%;
        overflow-y: scroll;
        scrollbar-size: 1 1;
    }

    .message {
        padding: 1;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
    }

    .user-message {
        border: solid $accent;
        text-align: right;
    }

    .assistant-message {
        border: solid $success;
    }

    .tool-message {
        border: dashed $warning;
        background: $boost;
    }

    #status-bar {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear Chat"),
        ("f5", "reload_skills", "Reload Skills"),
    ]

    def __init__(self, goal: str = ""):
        super().__init__()
        self.goal = goal
        self.brain = None
        self.pm = None
        self.skills = []
        self.messages = []
        self.provider = os.getenv("EVOLVE_PROVIDER", "auto")
        self.current_response = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Goal: {self.goal}", id="status-bar")
        
        with TabbedContent():
            with TabPane("Chat", id="tab-chat"):
                yield Vertical(id="chat-container")
                yield Input(placeholder="Type a message or command...", id="chat-input")
            
            with TabPane("Skills", id="tab-skills"):
                yield DataTable(id="skills-table")
            
            with TabPane("Sandbox", id="tab-sandbox"):
                yield DirectoryTree(str(SANDBOX_DIR), id="sandbox-tree")
            
            with TabPane("Logs", id="tab-logs"):
                yield RichLog(id="tool-log", markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize backend on startup."""
        self.brain = Brain(provider=self.provider if self.provider != "auto" else None)
        self.pm = PromptManager(EXECUTOR_PROMPT)
        self.reload_skills()
        
        # Initial system message
        skills_summary = get_skill_summary(self.skills)
        self.messages = [
            {"role": "system", "content": self.pm.current.format(skills_summary=skills_summary)},
            {"role": "user", "content": f"GOAL: {self.goal}"},
        ]
        
        container = self.query_one("#chat-container")
        container.mount(Static(f"Goal: {self.goal}", classes="message user-message", markup=False))
        
        # Start the conversation if there's a goal
        if self.goal:
            self.process_turn()

    def reload_skills(self):
        """Reload skills from disk."""
        self.skills = load_skills()
        try:
            table = self.query_one("#skills-table")
            table.clear()
            table.add_columns("Name", "Description", "Triggers")
            for skill in self.skills:
                triggers = ", ".join(skill.get("triggers", [])[:3])
                table.add_row(skill.get("name"), skill.get("description"), triggers)
        except Exception:
            pass

    def action_reload_skills(self):
        """Action for key binding."""
        self.reload_skills()
        self.notify(f"Loaded {len(self.skills)} skills.")

    def action_clear_chat(self):
        self.query_one("#chat-container").remove_children()

    @on(Input.Submitted, "#chat-input")
    def handle_input(self, event: Input.Submitted):
        user_input = event.value.strip()
        if not user_input:
            return
        
        event.input.value = ""
        container = self.query_one("#chat-container")
        container.mount(Static(user_input, classes="message user-message", markup=False))
        container.scroll_end(animate=False)
        
        # Handle commands
        low_input = user_input.lower()
        if low_input in ["exit", "quit"]:
            self.exit()
            return
        
        if low_input == "wipe-sandbox":
            msg = wipe_sandbox()
            self.notify(msg)
            self.query_one("#sandbox-tree").reload()
            return
            
        if low_input == "wipe-skills":
            msg = wipe_skills()
            self.notify(msg)
            self.reload_skills()
            return

        if low_input == "force-evolve":
            self.evolve_worker()
            return
            
        # Normal chat
        self.messages.append({"role": "user", "content": user_input})
        self.process_turn()

    @work(exclusive=True, thread=True)
    def process_turn(self):
        """Run the agent loop in a worker."""
        tool_log = self.query_one("#tool-log")
        container = self.query_one("#chat-container")
        
        # Update system prompt with latest skills
        self.reload_skills() # Refresh skills list
        skills_summary = get_skill_summary(self.skills)
        self.messages[0]["content"] = self.pm.current.format(skills_summary=skills_summary)
        
        full_response = ""
        
        # Mount a new empty message for the assistant
        response_widget = Markdown("", classes="message assistant-message")
        self.call_from_thread(container.mount, response_widget)
        self.call_from_thread(container.scroll_end, animate=False)
        
        # Stream response
        try:
            for chunk in self.brain.chat(self.messages, stream=True):
                full_response += chunk
                self.call_from_thread(response_widget.update, full_response)
                self.call_from_thread(container.scroll_end, animate=False)
        except Exception as e:
            self.call_from_thread(response_widget.update, f"Error: {e}")
            return

        self.messages.append({"role": "assistant", "content": full_response})
        
        if "[[SUCCESS]]" in full_response:
            self.call_from_thread(self.notify, "Goal Reached! Evolution starting...")
            self.evolve_worker()
            return

        # Check for tools
        skill_name, args = parse_tool_call(full_response)
        if skill_name:
            self.call_from_thread(tool_log.write, f"[bold cyan]Running {skill_name}[/bold cyan]: {args}")
            self.call_from_thread(self.notify, f"Running {skill_name}...")
            
            try:
                result = run_tool(skill_name, args)
                res_str = json.dumps(result, indent=2)
                
                self.call_from_thread(tool_log.write, f"[green]Result:[/green]\n{res_str}\n")
                
                # Show tool result in chat too, but smaller
                tool_widget = Static(f"Tool Result ({skill_name}):\n{res_str}", classes="message tool-message", markup=False)
                self.call_from_thread(container.mount, tool_widget)
                self.call_from_thread(container.scroll_end, animate=False)
                
                # Update sandbox tree if file ops happened
                if skill_name == "shell" or "file" in str(args):
                    self.call_from_thread(self.query_one("#sandbox-tree").reload)

                self.messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
                
                # Recursive call to continue conversation
                self.process_turn()
                
            except Exception as e:
                self.call_from_thread(tool_log.write, f"[bold red]Tool Error:[/bold red] {e}")
                self.messages.append({"role": "user", "content": f"Tool error: {e}"})
                self.process_turn()

    @work(thread=True)
    def evolve_worker(self):
        """Run evolution in background."""
        tool_log = self.query_one("#tool-log")
        self.call_from_thread(tool_log.write, "[dim]Analyzing for skills...[/dim]\n")
        
        new_skill = evolve(self.messages, self.skills)
        
        if new_skill:
            self.call_from_thread(tool_log.write, f"[bold green]New Skill Created:[/bold green] {new_skill}\n")
            self.call_from_thread(self.notify, f"New Skill: {new_skill}")
            self.call_from_thread(self.reload_skills)
        else:
            self.call_from_thread(tool_log.write, "[dim]No new skill created.[/dim]\n")
