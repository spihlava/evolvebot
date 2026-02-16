#!/usr/bin/env python3
"""
EvolveBot - Self-Evolving AI Agent with Skill Synthesis

Execution modes:
- direct: Dynamic import (fast, no isolation)
- chroot: Sandbox with chroot (requires root)
- docker: Docker container (most secure, needs docker.sock)
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from tools import EntityMatchGrader, LengthGrader

load_dotenv()

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve(strict=False).parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
SKILLS_DIR.mkdir(exist_ok=True)
SANDBOX_DIR.mkdir(exist_ok=True)

DEFAULT_EXECUTOR = os.getenv("EXECUTOR_MODEL", "qwen3:8b")
DEFAULT_ARCHITECT = os.getenv("MINIMAX_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

EXEC_MODE = os.getenv("EXEC_MODE", "direct")  # direct | chroot | docker

# ============================================================================
# BRAIN - LLM Client
# ============================================================================


class Brain:
    def __init__(self, provider=None, model=None):
        self.model = model or DEFAULT_EXECUTOR
        self.provider = provider or self._detect_provider(self.model)
        self.client = None
        self._init_client()

    def _detect_provider(self, model):
        if "MiniMax" in model:
            return "minimax"
        if "gemini" in model:
            return "gemini"
        return os.getenv("EVOLVE_PROVIDER", "ollama")

    def _init_client(self):
        if self.provider in ["ollama", "openai-compatible"]:
            from openai import OpenAI

            base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
            self.client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "ollama"))
        elif self.provider == "gemini":
            from google import genai

            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self.provider == "minimax":
            from anthropic import Anthropic

            base_url = os.getenv("MINIMAX_API_BASE", "https://api.minimax.io/anthropic")
            api_key = os.getenv("MINIMAX_API_KEY")
            self.client = Anthropic(base_url=base_url, api_key=api_key)

    def chat(self, messages, model=None, temperature=0.7, stream=False):
        model = model or self.model
        try:
            if stream:
                return self._stream_handler(model, messages, temperature)
            
            if self.provider == "gemini":
                return self._gemini_chat(model, messages)
            elif self.provider == "minimax":
                return self._minimax_chat(model, messages, temperature)
            else:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
                return resp.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _minimax_chat(self, model, messages, temperature):
        """Minimax Anthropic-compatible chat."""
        system_msg, contents = self._prep_anthropic(messages)
        resp = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=contents,
            temperature=temperature
        )
        # Handle multiple content blocks (e.g. ThinkingBlock + TextBlock)
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "thinking"):
                # Optionally log or skip thinking
                pass
        return text

    def _prep_anthropic(self, messages):
        """Format messages for Anthropic-compatible API."""
        system_msg = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                contents.append({"role": m["role"], "content": m["content"]})
        return system_msg, contents

    def _gemini_chat(self, model, messages):
        """Gemini call with full conversation history."""
        system_msg, contents = self._prep_gemini(messages)
        config = {"system_instruction": system_msg} if system_msg else {}
        resp = self.client.models.generate_content(model=model, contents=contents, config=config)
        return resp.text

    def _prep_gemini(self, messages):
        system_msg = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})
        return system_msg, contents

    def _stream_handler(self, model, messages, temperature):
        """Unified streaming handler for Gemini, Anthropic, and OpenAI."""
        full_response = ""
        if self.provider == "gemini":
            system_msg, contents = self._prep_gemini(messages)
            config = {"system_instruction": system_msg} if system_msg else {}
            for chunk in self.client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                if chunk.text:
                    console.print(chunk.text, end="", highlight=False)
                    full_response += chunk.text
        elif self.provider == "minimax":
            system_msg, contents = self._prep_anthropic(messages)
            with self.client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system_msg,
                messages=contents,
                temperature=temperature
            ) as stream:
                for text in stream.text_stream:
                    console.print(text, end="", highlight=False)
                    full_response += text
        else:
            response = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    console.print(content, end="", highlight=False)
                    full_response += content
        console.print()
        return full_response


class LLMGrader:
    """Uses an LLM to grade output against a rubric (LLM-as-a-judge)."""

    def __init__(self, brain, rubric):
        self.brain = brain
        self.rubric = rubric

    def grade(self, source, summary):
        prompt = f"""Evaluate this summary based on the rubric.
Source: {source}
Summary: {summary}
Rubric: {self.rubric}

Return JSON: {{"score": 0.0-1.0, "passed": bool, "feedback": "reasoning"}}"""
        messages = [{"role": "user", "content": prompt}]
        response = self.brain.chat(messages, temperature=0.1)
        try:
            # Strip fences
            clean = re.sub(r"```json\s*|\s*```", "", response.strip())
            return json.loads(clean)
        except:
            return {"score": 0.0, "passed": False, "feedback": f"Failed to parse grader response: {response}"}


# ============================================================================
# SKILL ENGINE
# ============================================================================


def load_skills():
    """Walk skills/ dir and load every skill.yaml found."""
    skills = []
    for root, _dirs, files in os.walk(SKILLS_DIR):
        if "skill.yaml" in files:
            with open(Path(root) / "skill.yaml") as f:
                skill = yaml.safe_load(f)
                skill["_path"] = Path(root).name
                skills.append(skill)
    return skills


def _get_triggers(skill):
    """Accept both 'triggers' and 'trigger_phrases' keys."""
    return skill.get("triggers") or skill.get("trigger_phrases") or []


def get_skill_summary(skills):
    if not skills:
        return "No skills available yet."
    return "\n".join(
        [
            f"- {s.get('name')}: {s.get('description')} "
            f"(triggers: {', '.join(_get_triggers(s)[:3])})"
            for s in skills
        ]
    )


def run_tool(skill_name, args):
    """Execute a skill's tool.py with optional isolation."""
    tool_path = SKILLS_DIR / skill_name / "tool.py"
    if not tool_path.exists():
        return {"error": f"Tool not found: {skill_name}"}

    args_json = json.dumps(args)

    if EXEC_MODE == "docker":
        return _run_docker(tool_path, args_json)
    elif EXEC_MODE == "chroot":
        return _run_chroot(tool_path, args_json)
    else:
        return _run_direct(tool_path, args)


def _run_direct(tool_path, args):
    """Direct import — fast, no isolation. Tools get sandbox/ as working dir."""
    sandbox_str = str(SANDBOX_DIR.absolute())

    # Set env var so the skill's own os.environ.get("SANDBOX_DIR") works
    old_env = os.environ.get("SANDBOX_DIR")
    os.environ["SANDBOX_DIR"] = sandbox_str

    tools_path = Path(__file__).parent / "tools.py"
    spec = importlib.util.spec_from_file_location("tool", str(tool_path))
    module = importlib.util.module_from_spec(spec)

    # Inject tools.py helpers — rebuild each time so SANDBOX_DIR is current
    if tools_path.exists():
        tools_spec = importlib.util.spec_from_file_location("tools", str(tools_path))
        tools_module = importlib.util.module_from_spec(tools_spec)
        tools_module.SANDBOX_DIR = sandbox_str
        tools_spec.loader.exec_module(tools_module)
        for name in dir(tools_module):
            if not name.startswith("_"):
                setattr(module, name, getattr(tools_module, name))

    module.SANDBOX_DIR = sandbox_str
    try:
        spec.loader.exec_module(module)
    finally:
        # Restore env
        if old_env is None:
            os.environ.pop("SANDBOX_DIR", None)
        else:
            os.environ["SANDBOX_DIR"] = old_env
    result = module.run(args)
    # Normalize: accept dict or JSON string from run()
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {"result": result}
    elif not isinstance(result, dict):
        result = {"result": str(result)}
    return result


def _parse_tool_stdout(stdout):
    """Parse tool stdout — could be JSON dict or plain text."""
    stdout = stdout.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except json.JSONDecodeError:
            pass
    return {"result": stdout}


def _run_chroot(tool_path, args_json):
    """Chroot isolation — tool sees only sandbox as root."""
    skill_name = tool_path.parent.name
    chroot_root = SANDBOX_DIR / "chroots" / skill_name / "root"
    chroot_root.mkdir(parents=True, exist_ok=True)

    shutil.copy(tool_path, chroot_root / "tool.py")
    (chroot_root / "args.json").write_text(args_json)

    try:
        result = subprocess.run(
            ["chroot", str(chroot_root), "/usr/bin/python3", "tool.py", "args.json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _parse_tool_stdout(result.stdout)
        return {"error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "Tool timed out (30s)"}
    except FileNotFoundError:
        return _run_direct(tool_path, json.loads(args_json))


def _run_docker(tool_path, args_json):
    """Docker isolation — most secure, no network."""
    skill_name = tool_path.parent.name
    skills_abs = str(Path(SKILLS_DIR).absolute() / skill_name)
    sandbox_abs = str(Path(SANDBOX_DIR).absolute())
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{skills_abs}:/app",
        "-v",
        f"{sandbox_abs}:/sandbox",
        "-w",
        "/app",
        "--network",
        "none",
        "python:3.13-slim",
        "python",
        "tool.py",
        args_json,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return _parse_tool_stdout(result.stdout)
        return {"error": result.stderr}
    except FileNotFoundError:
        return {"error": "Docker not available. Set EXEC_MODE=direct."}
    except subprocess.TimeoutExpired:
        return {"error": "Tool timed out (30s)"}


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXECUTOR_PROMPT = """You are EvolveBot, a self-evolving AI agent.

## Your Mission
1. ALWAYS attempt to solve the user's goal — never refuse or give up
2. Use existing skills when relevant, otherwise solve it yourself
3. Ask questions ONLY for genuinely vague requests — if the goal is clear, just do it
4. When the goal is FULLY satisfied, end your message with [[SUCCESS]]

## Problem Solving
The "shell" skill is always available. Use it to solve ANY task:
- Run commands: CALL_SKILL: shell WITH_ARGS: {{"action": "run_command", "command": "curl -s wttr.in/London"}}
- Write files:  CALL_SKILL: shell WITH_ARGS: {{"action": "write_file", "path": "output.txt", "content": "hello"}}
- Read files:   CALL_SKILL: shell WITH_ARGS: {{"action": "read_file", "path": "output.txt"}}
- List files:   CALL_SKILL: shell WITH_ARGS: {{"action": "list_files", "path": "."}}
- Run scripts:  CALL_SKILL: shell WITH_ARGS: {{"action": "run_script", "script": "print(2+2)"}}

Be resourceful. Try multiple approaches if the first one fails.

## Available Skills
{skills_summary}

## Tool Usage
To call a skill:  CALL_SKILL: skill_name WITH_ARGS: {{"key": "value"}}

## File Operations
All file reads/writes go through ./sandbox/ directory.
Skills can: read_file, write_file, list_files, run_command (all sandboxed).
"""

METAPROMPT_OPTIMIZER_PROMPT = """You are a Metaprompt Optimization Agent.
Your goal is to improve an LLM's system prompt based on grader feedback.

Current System Prompt:
{current_prompt}

Task Input:
{task_input}

Agent Output:
{agent_output}

Grader Feedback:
{feedback}

Instruction:
Generate a NEW system prompt that addresses the failures highlighted by the graders while maintaining the agent's core capabilities.
Return ONLY the new system prompt text. Do not include any explanation or JSON formatting.
"""

ARCHITECT_PROMPT = """You are a Software Architect. Create a reusable skill from this successful conversation.

Return ONLY valid JSON (no markdown fences). 

If the conversation does NOT contain any new reusable logic or functionality (e.g. it's just a greeting or simple Q&A), return exactly:
{{"no_skill_needed": "reason why no skill is needed"}}

Otherwise, return:
{{"name": "skill_name", "description": "what it does", "triggers": ["phrase1", "phrase2"], "python_code": "..."}}

Requirements for python_code:
- Must define a run(args) function that returns a dict
- Must have if __name__ == '__main__': block that reads sys.argv[1] as JSON
- Use only standard libraries (os, json, sys, pathlib, datetime, re, math, etc.)
- All file operations MUST use SANDBOX_DIR env var or ./sandbox/ as base path
- Return {{"result": "..."}} on success, {{"error": "..."}} on failure

Example python_code structure:
import os, sys, json
SANDBOX = os.environ.get("SANDBOX_DIR", "./sandbox")

def run(args):
    # ... do work ...
    return {{"result": "done"}}

if __name__ == "__main__":
    args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {{}}
    print(json.dumps(run(args)))

IMPORTANT: Do NOT create a skill that duplicates an existing one. If an existing skill
already covers the same functionality, return exactly: {{"duplicate_of": "existing_skill_name"}}

Existing skills: {existing_skills}

Conversation:
{history}"""

# Required keys — we also accept trigger_phrases as an alias for triggers
SKILL_SCHEMA_REQUIRED = {"name", "python_code"}


# ============================================================================
# PROMPT MANAGEMENT (Cookbook-inspired)
# ============================================================================


class PromptManager:
    def __init__(self, initial_prompt):
        self.history = [initial_prompt]

    @property
    def current(self):
        return self.history[-1]

    def update(self, new_prompt):
        self.history.append(new_prompt)

    def rollback(self):
        if len(self.history) > 1:
            self.history.pop()


# ============================================================================
# RETRAINING ENGINE (Cookbook-inspired)
# ============================================================================


def retrain_prompt(prompt_manager, brain, task_input, graders, max_retries=3):
    """Iteratively optimize the prompt based on grader feedback."""
    console.rule("[bold blue]Retraining Optimization Loop")

    for i in range(max_retries):
        with console.status(f"[bold green]Iteration {i+1}: Generating response..."):
            messages = [
                {"role": "system", "content": prompt_manager.current},
                {"role": "user", "content": task_input},
            ]
            output = brain.chat(messages)
        
        console.print(Panel(output, title=f"Iteration {i+1} Output", border_style="dim"))

        with console.status("[bold yellow]Grading output..."):
            results = [g.grade(task_input, output) for g in graders]
            all_passed = all(r["passed"] for r in results)
            scores = [r["score"] for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0

        console.print(f"[bold]Grades:[/bold] Passed: {all_passed}, Avg Score: [cyan]{avg_score:.2f}[/cyan]")
        for r in results:
            status_char = "[green]✔[/green]" if r["passed"] else "[red]✘[/red]"
            console.print(f"  {status_char} {r.get('feedback', 'No feedback')}")

        if all_passed:
            console.print("[bold green]Success![/bold green] All graders passed.")
            return True

        # Failed — collect feedback for metaprompt
        feedback_text = "\n".join([f"- {r.get('feedback')}" for r in results if not r["passed"]])
        
        with console.status("[bold magenta]Consulting Metaprompt Architect..."):
            meta_messages = [
                {
                    "role": "system",
                    "content": METAPROMPT_OPTIMIZER_PROMPT.format(
                        current_prompt=prompt_manager.current,
                        task_input=task_input,
                        agent_output=output,
                        feedback=feedback_text,
                    ),
                },
                {"role": "user", "content": "Improve the system prompt based on this failure."},
            ]
            new_prompt = brain.chat(meta_messages)
            prompt_manager.update(new_prompt)
        
        console.print(f"[bold cyan]Prompt updated to version {len(prompt_manager.history)}[/bold cyan]")

    console.print("[bold red]Fail:[/bold red] Max retries hit without passing all graders.")
    return False


# ============================================================================
# EVOLUTION — skill synthesis from successful conversations
# ============================================================================


def evolve(history, existing_skills):
    """Call architect to synthesize a skill from conversation."""
    with console.status("[bold magenta]Analyzing conversation for skill synthesis..."):
        brain = Brain(model=DEFAULT_ARCHITECT)

        skills_json = json.dumps(
            [
                {
                    "name": s.get("name"),
                    "description": s.get("description"),
                    "triggers": _get_triggers(s),
                }
                for s in existing_skills
            ]
        )

        history_text = "\n".join([f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-10:]])

        messages = [
            {
                "role": "system",
                "content": ARCHITECT_PROMPT.format(existing_skills=skills_json, history=history_text),
            },
            {
                "role": "user",
                "content": "Create a skill from the conversation above. Return ONLY JSON.",
            },
        ]

        response = brain.chat(messages)

    clean = response.strip()
    # Strip thinking tags if they leaked into the text
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        skill_data = json.loads(clean)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error:[/bold red] Failed to parse architect response")
        console.print(f"[dim]Raw response: {clean[:200]}[/dim]")
        return None

    if "no_skill_needed" in skill_data:
        console.print(f"[dim]Evolution skipped: {skill_data['no_skill_needed']}[/dim]")
        return None

    if "duplicate_of" in skill_data:
        console.print(f"[yellow]Skipped[/yellow] — duplicate of '{skill_data['duplicate_of']}'")
        return None

    missing = SKILL_SCHEMA_REQUIRED - set(skill_data.keys())
    if missing:
        console.print(f"[bold red]Error:[/bold red] Skill JSON missing keys: {missing}")
        console.print(f"[dim]Response keys: {list(skill_data.keys())}[/dim]")
        if "no_skill_needed" not in skill_data:
             console.print(Panel(json.dumps(skill_data, indent=2), title="Architect Response Debug", border_style="red"))
        return None

    triggers = skill_data.get("triggers") or skill_data.get("trigger_phrases") or []
    name = re.sub(r"[^a-z0-9_]", "_", skill_data["name"].lower().strip())
    
    skill_path = SKILLS_DIR / name
    skill_path.mkdir(exist_ok=True)

    code = skill_data.get("python_code", "")
    try:
        compile(code, f"{name}/tool.py", "exec")
    except SyntaxError as e:
        console.print(f"[bold red]Error:[/bold red] Syntax error in generated code: {e}")
        shutil.rmtree(skill_path, ignore_errors=True)
        return None

    with open(skill_path / "skill.yaml", "w") as f:
        yaml.dump({
            "name": name,
            "description": skill_data.get("description", ""),
            "triggers": triggers,
            "version": "1.0",
            "created": datetime.now().isoformat(),
        }, f)

    with open(skill_path / "tool.py", "w") as f:
        f.write(code)

    console.print(f"[bold green]Success![/bold green] Skill [cyan]{name}[/cyan] created.")
    return name


# ============================================================================
# SANDBOX MANAGEMENT
# ============================================================================


def wipe_sandbox():
    """Clear all files in sandbox/ — the wipable work environment."""
    for item in SANDBOX_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    console.print(f"[bold yellow]Sandbox wiped.[/bold yellow]")


def wipe_skills():
    """Remove all generated skills."""
    for item in SKILLS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    console.print(f"[bold yellow]Skills wiped.[/bold yellow]")


def parse_tool_call(response):
    """
    Ultra-resilient tool call parser. 
    Matches:
    - CALL_SKILL: name WITH_ARGS: {json}
    - <CALL_SKILL: name WITH_ARGS> {json} </CALL_SKILL>
    - <minimax:tool_call><invoke name="name"><parameter name="x">y</parameter>...
    """
    # Try Regex for any format containing a skill name and a JSON block
    # Matches "CALL_SKILL: name" or "<CALL_SKILL: name"
    name_match = re.search(r"(?:CALL_SKILL:\s*|<CALL_SKILL:\s*)([a-zA-Z0-9_-]+)", response)
    json_match = re.search(r"(\{.*\})", response, re.DOTALL)
    
    if name_match and json_match:
        try:
            return name_match.group(1).strip(), json.loads(json_match.group(1))
        except:
            pass

    # Fallback to Minimax XML specific format
    if "<minimax:tool_call>" in response or "<invoke" in response:
        try:
            name_match = re.search(r'name=["\'](.*?)["\']', response)
            skill_name = name_match.group(1) if name_match else None
            params = {}
            param_matches = re.finditer(r'<parameter name=["\'](.*?)["\']>(.*?)</parameter>', response, re.DOTALL)
            for m in param_matches:
                params[m.group(1)] = m.group(2).strip()
            if skill_name and params:
                return skill_name, params
        except:
            pass
            
    return None, None


# ============================================================================
# MAIN LOOP
# ============================================================================


def main():
    console.print(Panel.fit(
        f"[bold blue]EvolveBot[/bold blue] - Self-Evolving Agent\n"
        f"Executor:  [cyan]{DEFAULT_EXECUTOR}[/cyan]\n"
        f"Architect: [cyan]{DEFAULT_ARCHITECT}[/cyan]\n"
        f"Exec mode: [green]{EXEC_MODE}[/green]",
        title="Welcome", border_style="blue"
    ))

    provider = os.getenv("EVOLVE_PROVIDER")
    brain = Brain(provider=provider, model=DEFAULT_EXECUTOR)
    pm = PromptManager(EXECUTOR_PROMPT)

    skills = load_skills()
    console.print(f"[dim]Loaded {len(skills)} skill(s)[/dim]")

    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else console.input("\n[bold green]Goal?[/bold green] ").strip()

    messages = [
        {"role": "system", "content": pm.current.format(skills_summary=get_skill_summary(skills))},
        {"role": "user", "content": f"GOAL: {goal}"},
    ]

    console.rule(f"[bold]Target: {goal}")
    console.print("[dim]Commands: exit | retrain | force-evolve | wipe-sandbox | wipe-skills[/dim]\n")

    while True:
        skills = load_skills()
        messages[0]["content"] = pm.current.format(skills_summary=get_skill_summary(skills))

        console.print("[bold blue]Bot:[/bold blue] ", end="")
        response = brain.chat(messages, stream=True)
        messages.append({"role": "assistant", "content": response})

        # Handle skill invocation
        skill_name, args = parse_tool_call(response)
        if skill_name:
            try:
                with console.status(f"[bold cyan]Running {skill_name}..."):
                    result = run_tool(skill_name, args)
                
                console.print(Panel(json.dumps(result, indent=2), title=f"Result: {skill_name}", border_style="cyan"))
                messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
                continue
            except Exception as e:
                console.print(f"[bold red]Skill Error:[/bold red] {e}")
                messages.append({"role": "user", "content": f"Tool error: {e}"})
                continue

        if "[[SUCCESS]]" in response:
            console.print("\n[bold green]Goal Reached![/bold green]")
            evolve(messages, skills)
            break

        user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        low_input = user_input.lower()
        
        if low_input in ["exit", "quit"]: break
        if low_input == "retrain":
            rubric = console.input("[bold magenta]Rubric:[/bold magenta] ").strip()
            target_words = 100
            if "words" in rubric:
                match = re.search(r"(\d+)\s*words", rubric)
                if match: target_words = int(match.group(1))
            
            graders = [
                LengthGrader(target_len=target_words, tolerance=target_words // 4),
                LLMGrader(brain, rubric),
            ]
            retrain_prompt(pm, brain, goal, graders)
            continue
        
        if low_input == "force-evolve": evolve(messages, skills); continue
        if low_input == "wipe-sandbox": wipe_sandbox(); continue
        if low_input == "wipe-skills": wipe_skills(); continue

        messages.append({"role": "user", "content": user_input})


if __name__ == "__main__":
    main()
