"""
Core logic for EvolveBot agent.
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

# Ensure project root is in path for imports if needed, though relative imports are better
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import EntityMatchGrader, LengthGrader

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).absolute().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
SKILLS_DIR.mkdir(exist_ok=True)
SANDBOX_DIR.mkdir(exist_ok=True)

DEFAULT_EXECUTOR = os.getenv("EXECUTOR_MODEL", "qwen3:8b")
DEFAULT_ARCHITECT = os.getenv("MINIMAX_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

EXEC_MODE = os.getenv("EXEC_MODE", "direct")  # direct | chroot | docker

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

SKILL_SCHEMA_REQUIRED = {"name", "python_code"}


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
                return self._stream_generator(model, messages, temperature)

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
            if stream:
                # Generator that yields the error
                def error_gen():
                    yield json.dumps({"error": str(e)})
                return error_gen()
            return json.dumps({"error": str(e)})

    def _minimax_chat(self, model, messages, temperature):
        """Minimax Anthropic-compatible chat."""
        system_msg, contents = self._prep_anthropic(messages)
        resp = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg,
            messages=contents,
            temperature=temperature,
        )
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "thinking"):
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

    def _stream_generator(self, model, messages, temperature):
        """Unified streaming generator for Gemini, Anthropic, and OpenAI."""
        if self.provider == "gemini":
            system_msg, contents = self._prep_gemini(messages)
            config = {"system_instruction": system_msg} if system_msg else {}
            for chunk in self.client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text
        elif self.provider == "minimax":
            system_msg, contents = self._prep_anthropic(messages)
            with self.client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system_msg,
                messages=contents,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            response = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


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
            clean = re.sub(r"```json\s*|\s*```", "", response.strip())
            return json.loads(clean)
        except:
            return {
                "score": 0.0,
                "passed": False,
                "feedback": f"Failed to parse grader response: {response}",
            }


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
    return "\\n".join(
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

    old_env = os.environ.get("SANDBOX_DIR")
    os.environ["SANDBOX_DIR"] = sandbox_str

    tools_path = Path(__file__).parent / "tools.py"
    spec = importlib.util.spec_from_file_location("tool", str(tool_path))
    module = importlib.util.module_from_spec(spec)

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
        if old_env is None:
            os.environ.pop("SANDBOX_DIR", None)
        else:
            os.environ["SANDBOX_DIR"] = old_env
    result = module.run(args)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {"result": result}
    elif not isinstance(result, dict):
        result = {"result": str(result)}
    return result


def _parse_tool_stdout(stdout):
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
# PROMPT MANAGEMENT
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
# RETRAINING ENGINE
# ============================================================================


def retrain_prompt(prompt_manager, brain, task_input, graders, console=None, max_retries=3):
    """Iteratively optimize the prompt based on grader feedback."""
    # Modified to accept optional console for output, otherwise silent/log
    if console:
        console.rule("[bold blue]Retraining Optimization Loop")

    for i in range(max_retries):
        if console:
            console.print(f"[bold green]Iteration {i + 1}: Generating response...[/bold green]")
        
        messages = [
            {"role": "system", "content": prompt_manager.current},
            {"role": "user", "content": task_input},
        ]
        output = brain.chat(messages)

        if console:
            # Avoid circular import if possible, but we use rich.panel here if console is passed
            from rich.panel import Panel
            console.print(Panel(output, title=f"Iteration {i + 1} Output", border_style="dim"))

        results = [g.grade(task_input, output) for g in graders]
        all_passed = all(r["passed"] for r in results)
        
        if console:
            for r in results:
                status_char = "[green]✔[/green]" if r["passed"] else "[red]✘[/red]"
                console.print(f"  {status_char} {r.get('feedback', 'No feedback')}")

        if all_passed:
            if console:
                console.print("[bold green]Success![/bold green] All graders passed.")
            return True

        feedback_text = "\\n".join([f"- {r.get('feedback')}" for r in results if not r["passed"]])

        if console:
            console.print("[bold magenta]Consulting Metaprompt Architect...[/bold magenta]")
            
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
        
        if console:
            console.print(
                f"[bold cyan]Prompt updated to version {len(prompt_manager.history)}[/bold cyan]"
            )

    if console:
        console.print("[bold red]Fail:[/bold red] Max retries hit without passing all graders.")
    return False


# ============================================================================
# EVOLUTION
# ============================================================================


def evolve(history, existing_skills, console=None):
    """Call architect to synthesize a skill from conversation."""
    if console:
        console.print("[bold magenta]Analyzing conversation for skill synthesis...[/bold magenta]")
        
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

    history_text = "\\n".join(
        [f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-10:]]
    )

    messages = [
        {
            "role": "system",
            "content": ARCHITECT_PROMPT.format(
                existing_skills=skills_json, history=history_text
            ),
        },
        {
            "role": "user",
            "content": "Create a skill from the conversation above. Return ONLY JSON.",
        },
    ]

    response = brain.chat(messages)

    clean = response.strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        skill_data = json.loads(clean)
    except json.JSONDecodeError:
        if console:
            console.print(f"[bold red]Error:[/bold red] Failed to parse architect response")
        return None

    if "no_skill_needed" in skill_data:
        if console:
            console.print(f"[dim]Evolution skipped: {skill_data['no_skill_needed']}[/dim]")
        return None

    if "duplicate_of" in skill_data:
        if console:
            console.print(f"[yellow]Skipped[/yellow] — duplicate of '{skill_data['duplicate_of']}'")
        return None

    missing = SKILL_SCHEMA_REQUIRED - set(skill_data.keys())
    if missing:
        if console:
            console.print(f"[bold red]Error:[/bold red] Skill JSON missing keys: {missing}")
        return None

    triggers = skill_data.get("triggers") or skill_data.get("trigger_phrases") or []
    name = re.sub(r"[^a-z0-9_]", "_", skill_data["name"].lower().strip())

    skill_path = SKILLS_DIR / name
    skill_path.mkdir(exist_ok=True)

    code = skill_data.get("python_code", "")
    try:
        compile(code, f"{name}/tool.py", "exec")
    except SyntaxError as e:
        if console:
            console.print(f"[bold red]Error:[/bold red] Syntax error in generated code: {e}")
        shutil.rmtree(skill_path, ignore_errors=True)
        return None

    with open(skill_path / "skill.yaml", "w") as f:
        yaml.dump(
            {
                "name": name,
                "description": skill_data.get("description", ""),
                "triggers": triggers,
                "version": "1.0",
                "created": datetime.now().isoformat(),
            },
            f,
        )

    with open(skill_path / "tool.py", "w") as f:
        f.write(code)

    if console:
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
    return "Sandbox wiped."


def wipe_skills():
    """Remove all generated skills."""
    for item in SKILLS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    return "Skills wiped."


def parse_tool_call(response):
    """
    Omni-resilient tool call parser.
    """
    name_patterns = [
        r"(?:CALL_SKILL:\s*|<CALL_SKILL:\s*|\[TOOL_CALL\]\s*|name=[\"'])([a-zA-Z0-9_-]+)",
        r"<invoke\s+name=[\"']([a-zA-Z0-9_-]+)[\"']",
        r"invoke\s+([a-zA-Z0-9_-]+)",
    ]

    skill_name = None
    for p in name_patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            skill_name = m.group(1).strip()
            break

    if not skill_name:
        if ("action" in response or "command" in response) and (
            "run_command" in response or "write_file" in response
        ):
            skill_name = "shell"
        else:
            return None, None

    json_match = re.search(r"(\{.*\})", response, re.DOTALL)
    if json_match:
        try:
            return skill_name, json.loads(json_match.group(1))
        except:
            pass

    params = {}
    param_matches = re.finditer(
        r'(?:<parameter name=["\']|<)([a-zA-Z0-9_-]+)["\']?>(.*?)</\1>', response, re.DOTALL
    )
    for m in param_matches:
        params[m.group(1)] = m.group(2).strip()

    if not params:
        kv_matches = re.finditer(r'([a-zA-Z0-9_-]+)\s*[:=][=>]?\s*["\'](.*?)["\']', response)
        for m in kv_matches:
            params[m.group(1)] = m.group(2).strip()

    if params:
        return skill_name, params

    return None, None
