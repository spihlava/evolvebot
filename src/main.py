#!/usr/bin/env python3
"""
EvolveBot - Self-Evolving AI Agent with Skill Synthesis

Execution modes:
- direct: Dynamic import (fast, no isolation)
- chroot: Sandbox with chroot (requires root)
- docker: Docker container (most secure, needs docker.sock)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    LengthGrader,
    LLMGrader,
)
from src.tui.app import EvolveBotApp

load_dotenv()
console = Console()

# ============================================================================
# MAIN LOOP (CLI)
# ============================================================================


def main_cli():
    console.rule("[bold blue]EvolveBot CLI[/bold blue]")

    provider = os.getenv("EVOLVE_PROVIDER")
    brain = Brain(provider=provider, model=DEFAULT_EXECUTOR)
    pm = PromptManager(EXECUTOR_PROMPT)

    skills = load_skills()
    console.print(f"[dim]Loaded {len(skills)} skill(s)[/dim]")

    goal = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else console.input("\n[bold green]Goal?[/bold green] ").strip()
    )

    messages = [
        {"role": "system", "content": pm.current.format(skills_summary=get_skill_summary(skills))},
        {"role": "user", "content": f"GOAL: {goal}"},
    ]

    console.rule(f"[bold]Target: {goal}")
    console.print(
        "[dim]Commands: exit | retrain | force-evolve | wipe-sandbox | wipe-skills[/dim]\n"
    )

    while True:
        skills = load_skills()
        messages[0]["content"] = pm.current.format(skills_summary=get_skill_summary(skills))

        console.print("\n[bold blue]Bot Thinking...[/bold blue]")

        full_response = ""
        # 1. Live Streaming
        try:
            for chunk in brain.chat(messages, stream=True):
                full_response += chunk
                console.print(chunk, end="")
            console.print()  # Newline
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            messages.append({"role": "user", "content": f"Error: {e}"})
            continue

        messages.append({"role": "assistant", "content": full_response})

        # Check for success marker immediately
        if "[[SUCCESS]]" in full_response:
            console.print("\n[bold green]✨ Goal Reached![/bold green]")
            evolve(messages, skills, console=console)
            break

        # Handle skill invocation
        skill_name, args = parse_tool_call(full_response)
        if skill_name:
            try:
                # Cleaner tool execution UI
                with console.status(f"[bold cyan]⚡ Executing {skill_name}..."):
                    result = run_tool(skill_name, args)

                # Show results in a compact panel
                # We reuse the rich panel import inside the loop to avoid clutter if not needed
                from rich.panel import Panel
                res_str = str(result) # json.dumps(result, indent=2)
                if len(res_str) > 1000:
                    res_str = res_str[:1000] + "... (truncated)"
                
                console.print(
                    Panel(
                        res_str,
                        title=f"Result: {skill_name}",
                        border_style="cyan",
                    )
                )

                messages.append({"role": "user", "content": f"Tool result: {res_str}"})
                # AUTOMATIC CONTINUITY: Go straight back to the LLM
                continue
            except Exception as e:
                console.print(f"[bold red]Skill Error:[/bold red] {e}")
                messages.append({"role": "user", "content": f"Tool error: {e}"})
                continue

        # If no tool call and no success, finally ask the user
        user_input = console.input("\n[bold green]➜[/bold green] ").strip()
        low_input = user_input.lower()

        if low_input in ["exit", "quit"]:
            break
        if low_input == "retrain":
            rubric = console.input("[bold magenta]Rubric:[/bold magenta] ").strip()
            target_words = 100
            if "words" in rubric:
                import re
                match = re.search(r"(\d+)\s*words", rubric)
                if match:
                    target_words = int(match.group(1))

            graders = [
                LengthGrader(target_len=target_words, tolerance=target_words // 4),
                LLMGrader(brain, rubric),
            ]
            retrain_prompt(pm, brain, goal, graders, console=console)
            continue

        if low_input == "force-evolve":
            evolve(messages, skills, console=console)
            continue
        if low_input == "wipe-sandbox":
            wipe_sandbox()
            console.print("Sandbox wiped.")
            continue
        if low_input == "wipe-skills":
            wipe_skills()
            console.print("Skills wiped.")
            continue

        messages.append({"role": "user", "content": user_input})


def main_tui():
    """Textual TUI version of main."""
    goal = " ".join(sys.argv[1:])
    # If no goal provided in args, App could ask for it, 
    # but for now let's just start with empty goal or ask via input
    # Actually, the App structure I designed takes goal in __init__.
    # If empty, it just shows up empty.
    
    app = EvolveBotApp(goal=goal)
    app.run()


if __name__ == "__main__":
    # Check if textual is installed (it should be since we added it)
    # But also check env var to force CLI if needed
    FORCE_CLI = os.getenv("FORCE_CLI", "false").lower() == "true"
    
    if FORCE_CLI:
        main_cli()
    else:
        # Default to TUI
        main_tui()
