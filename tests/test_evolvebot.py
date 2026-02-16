"""
Tests for EvolveBot — skill creation, tool execution, sandbox safety, and memory skill.

Run: pytest tests/ -v
"""

import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import yaml

# workspace fixture is in conftest.py (autouse)


# ===========================================================================
# 1. TOOLS.PY — sandbox safety
# ===========================================================================


class TestToolsSandbox:
    def test_write_and_read_file(self, workspace):
        import tools

        result = tools.write_file("hello.txt", "world")
        assert "Written" in result
        assert tools.read_file("hello.txt") == "world"

    def test_read_missing_file(self, workspace):
        import tools

        assert "not found" in tools.read_file("nonexistent.txt")

    def test_list_files(self, workspace):
        import tools

        tools.write_file("a.txt", "1")
        tools.write_file("b.txt", "2")
        files = tools.list_files(".")
        assert "a.txt" in files
        assert "b.txt" in files

    def test_path_traversal_blocked(self, workspace):
        import tools

        with pytest.raises(ValueError, match="escapes sandbox"):
            tools._safe_path("../../etc/passwd")

    def test_nested_directory_creation(self, workspace):
        import tools

        tools.write_file("memory/MEMORY.md", "# Memory")
        assert tools.read_file("memory/MEMORY.md") == "# Memory"

    def test_run_command(self, workspace):
        import tools

        result = tools.run_command("echo hello")
        assert "hello" in result


# ===========================================================================
# 2. SKILL LOADING — triggers vs trigger_phrases
# ===========================================================================


class TestSkillLoading:
    def _create_skill(self, workspace, name, yaml_data, code="def run(args): return {}"):
        skill_dir = workspace["skills"] / name
        skill_dir.mkdir()
        with open(skill_dir / "skill.yaml", "w") as f:
            yaml.dump(yaml_data, f)
        (skill_dir / "tool.py").write_text(code)

    def test_load_skills_triggers(self, workspace):
        import main

        self._create_skill(
            workspace,
            "greeting",
            {
                "name": "greeting",
                "description": "Says hello",
                "triggers": ["hello", "hi"],
            },
        )
        skills = main.load_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "greeting"

    def test_load_skills_trigger_phrases(self, workspace):
        import main

        self._create_skill(
            workspace,
            "sushi",
            {
                "name": "sushi",
                "description": "Find sushi",
                "trigger_phrases": ["find sushi", "no reservation sushi"],
            },
        )
        skills = main.load_skills()
        triggers = main._get_triggers(skills[0])
        assert "find sushi" in triggers

    def test_get_skill_summary_both_formats(self, workspace):
        import main

        skills = [
            {"name": "a", "description": "A skill", "triggers": ["do a"]},
            {"name": "b", "description": "B skill", "trigger_phrases": ["do b"]},
        ]
        summary = main.get_skill_summary(skills)
        assert "do a" in summary
        assert "do b" in summary

    def test_get_skill_summary_empty(self, workspace):
        import main

        assert "No skills" in main.get_skill_summary([])


# ===========================================================================
# 3. TOOL EXECUTION — _run_direct
# ===========================================================================


class TestToolExecution:
    def test_run_direct_returns_dict(self, workspace):
        """Tool that returns a dict directly."""
        import main

        skill_dir = workspace["skills"] / "greeter"
        skill_dir.mkdir()
        (skill_dir / "tool.py").write_text(textwrap.dedent("""\
            def run(args):
                name = args.get("name", "World")
                return {"result": f"Hello, {name}!"}
        """))
        (skill_dir / "skill.yaml").write_text("name: greeter\ndescription: test\ntriggers: []\n")

        result = main.run_tool("greeter", {"name": "Alice"})
        assert result == {"result": "Hello, Alice!"}

    def test_run_direct_returns_json_string(self, workspace):
        """Tool that returns json.dumps() — like the Tokyo sushi example."""
        import main

        skill_dir = workspace["skills"] / "sushi_finder"
        skill_dir.mkdir()
        (skill_dir / "tool.py").write_text(textwrap.dedent("""\
            import json
            def run(args):
                return json.dumps({"status": "Success", "spots": ["Midori", "Kurasushi"]})
        """))
        (skill_dir / "skill.yaml").write_text(
            "name: sushi_finder\ndescription: test\ntriggers: []\n"
        )

        result = main.run_tool("sushi_finder", {})
        assert result["status"] == "Success"
        assert "Midori" in result["spots"]

    def test_run_direct_returns_plain_string(self, workspace):
        """Tool that returns a plain string."""
        import main

        skill_dir = workspace["skills"] / "echo"
        skill_dir.mkdir()
        (skill_dir / "tool.py").write_text('def run(args): return "plain text"')
        (skill_dir / "skill.yaml").write_text("name: echo\ndescription: test\ntriggers: []\n")

        result = main.run_tool("echo", {})
        assert result == {"result": "plain text"}

    def test_run_tool_not_found(self, workspace):
        import main

        result = main.run_tool("nonexistent", {})
        assert "error" in result

    def test_tools_injected_into_skill(self, workspace):
        """Skill can use tools.py helpers (write_file, read_file) via injection."""
        import main

        skill_dir = workspace["skills"] / "writer"
        skill_dir.mkdir()
        (skill_dir / "tool.py").write_text(textwrap.dedent("""\
            def run(args):
                write_file("test_output.txt", args.get("content", ""))
                return {"result": read_file("test_output.txt")}
        """))
        (skill_dir / "skill.yaml").write_text("name: writer\ndescription: test\ntriggers: []\n")

        result = main.run_tool("writer", {"content": "injected!"})
        assert result["result"] == "injected!"
        assert (workspace["sandbox"] / "test_output.txt").read_text() == "injected!"


# ===========================================================================
# 4. PARSE TOOL STDOUT
# ===========================================================================


class TestParseToolStdout:
    def test_json_dict(self, workspace):
        import main

        assert main._parse_tool_stdout('{"status": "ok"}') == {"status": "ok"}

    def test_json_non_dict(self, workspace):
        import main

        assert main._parse_tool_stdout("[1, 2, 3]") == {"result": [1, 2, 3]}

    def test_plain_text(self, workspace):
        import main

        assert main._parse_tool_stdout("hello world") == {"result": "hello world"}

    def test_empty(self, workspace):
        import main

        assert main._parse_tool_stdout("") == {"result": ""}


# ===========================================================================
# 5. EVOLUTION — skill synthesis (mocked Gemini)
# ===========================================================================


class TestEvolution:
    def _mock_evolve(self, workspace, gemini_response):
        """Run evolve() with a mocked Brain that returns the given response."""
        import main

        mock_brain = MagicMock()
        mock_brain.chat.return_value = gemini_response

        with patch.object(main, "Brain", return_value=mock_brain):
            history = [
                {"role": "system", "content": "You are EvolveBot..."},
                {"role": "user", "content": "GOAL: help me remember things"},
                {"role": "assistant", "content": "I'll create a memory system. [[SUCCESS]]"},
            ]
            return main.evolve(history, [])

    def test_evolve_creates_skill(self, workspace):
        response = json.dumps(
            {
                "name": "calculator",
                "description": "Basic math",
                "triggers": ["calculate", "math"],
                "python_code": "def run(args):\n    return {'result': eval(args['expr'])}\n",
            }
        )
        name = self._mock_evolve(workspace, response)
        assert name == "calculator"
        assert (workspace["skills"] / "calculator" / "skill.yaml").exists()
        assert (workspace["skills"] / "calculator" / "tool.py").exists()

    def test_evolve_strips_markdown_fences(self, workspace):
        response = (
            "```json\n"
            + json.dumps(
                {
                    "name": "fenced",
                    "description": "test",
                    "triggers": [],
                    "python_code": "def run(args): return {'result': 'ok'}",
                }
            )
            + "\n```"
        )
        name = self._mock_evolve(workspace, response)
        assert name == "fenced"

    def test_evolve_accepts_trigger_phrases(self, workspace):
        response = json.dumps(
            {
                "name": "tp_skill",
                "description": "test trigger_phrases",
                "trigger_phrases": ["remember", "save memory"],
                "python_code": "def run(args): return {'result': 'ok'}",
            }
        )
        name = self._mock_evolve(workspace, response)
        assert name == "tp_skill"
        with open(workspace["skills"] / "tp_skill" / "skill.yaml") as f:
            data = yaml.safe_load(f)
        assert "remember" in data["triggers"]

    def test_evolve_rejects_syntax_error(self, workspace):
        response = json.dumps(
            {
                "name": "bad_skill",
                "description": "broken",
                "triggers": [],
                "python_code": "def run(args)\n    return {}",  # missing colon
            }
        )
        name = self._mock_evolve(workspace, response)
        assert name is None
        assert not (workspace["skills"] / "bad_skill").exists()

    def test_evolve_rejects_missing_keys(self, workspace):
        response = json.dumps(
            {
                "description": "no name or code",
                "triggers": [],
            }
        )
        name = self._mock_evolve(workspace, response)
        assert name is None

    def test_evolve_rejects_invalid_json(self, workspace):
        name = self._mock_evolve(workspace, "not json at all")
        assert name is None

    def test_evolve_sanitizes_skill_name(self, workspace):
        response = json.dumps(
            {
                "name": "My Cool Skill!!!",
                "description": "test",
                "triggers": [],
                "python_code": "def run(args): return {}",
            }
        )
        name = self._mock_evolve(workspace, response)
        assert name == "my_cool_skill___"


# ===========================================================================
# 6. MEMORY SKILL — end-to-end creation and execution
# ===========================================================================

MEMORY_SKILL_CODE = textwrap.dedent("""\
    import os
    import sys
    import json
    from pathlib import Path
    from datetime import datetime

    SANDBOX = os.environ.get("SANDBOX_DIR", "./sandbox")

    def _memory_dir():
        d = Path(SANDBOX) / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(args):
        action = args.get("action", "read")
        memory_dir = _memory_dir()
        memory_file = memory_dir / "MEMORY.md"
        history_file = memory_dir / "HISTORY.md"

        if action == "remember":
            # Save a fact to MEMORY.md
            fact = args.get("fact", "")
            if not fact:
                return {"error": "No fact provided"}
            existing = memory_file.read_text() if memory_file.exists() else "# Memory\\n\\n"
            memory_file.write_text(existing + f"- {fact}\\n")
            # Also log to HISTORY.md
            timestamp = datetime.now().isoformat()
            history_entry = f"[{timestamp}] REMEMBERED: {fact}\\n"
            with open(history_file, "a") as f:
                f.write(history_entry)
            return {"result": f"Saved: {fact}"}

        elif action == "recall":
            # Read MEMORY.md
            if not memory_file.exists():
                return {"result": "No memories yet."}
            return {"result": memory_file.read_text()}

        elif action == "search":
            # Grep HISTORY.md for a keyword
            keyword = args.get("keyword", "")
            if not history_file.exists():
                return {"result": "No history yet."}
            lines = history_file.read_text().splitlines()
            matches = [l for l in lines if keyword.lower() in l.lower()]
            return {"result": "\\n".join(matches) if matches else f"No matches for '{keyword}'"}

        elif action == "list_events":
            # Return last N entries from HISTORY.md
            n = args.get("count", 10)
            if not history_file.exists():
                return {"result": "No history yet."}
            lines = history_file.read_text().splitlines()
            return {"result": "\\n".join(lines[-n:])}

        else:
            return {"error": f"Unknown action: {action}"}

    if __name__ == "__main__":
        input_args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        print(json.dumps(run(input_args)))
""")


class TestMemorySkill:
    """
    End-to-end: simulate Gemini creating the memory skill via evolve(),
    then run it through the real tool execution pipeline.
    """

    def _create_memory_skill_via_evolve(self, workspace):
        """Simulate the architect generating the memory skill."""
        import main

        gemini_response = json.dumps(
            {
                "name": "memory",
                "description": "Long-term memory with MEMORY.md and HISTORY.md. "
                "Stores facts, searches history, recalls context.",
                "triggers": ["remember", "recall", "search memory", "save fact"],
                "python_code": MEMORY_SKILL_CODE,
            }
        )
        mock_brain = MagicMock()
        mock_brain.chat.return_value = gemini_response

        history = [
            {"role": "system", "content": "You are EvolveBot..."},
            {"role": "user", "content": "GOAL: I need you to remember things across sessions"},
            {
                "role": "assistant",
                "content": "I'll create a memory system with MEMORY.md for facts "
                "and HISTORY.md as an event log. [[SUCCESS]]",
            },
        ]
        with patch.object(main, "Brain", return_value=mock_brain):
            return main.evolve(history, [])

    def test_memory_skill_created(self, workspace):
        name = self._create_memory_skill_via_evolve(workspace)
        assert name == "memory"
        assert (workspace["skills"] / "memory" / "skill.yaml").exists()
        assert (workspace["skills"] / "memory" / "tool.py").exists()

        with open(workspace["skills"] / "memory" / "skill.yaml") as f:
            meta = yaml.safe_load(f)
        assert "remember" in meta["triggers"]

    def test_memory_remember_and_recall(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        # Remember a fact
        result = main.run_tool("memory", {"action": "remember", "fact": "User prefers dark mode"})
        assert result["result"] == "Saved: User prefers dark mode"

        # Recall all memories
        result = main.run_tool("memory", {"action": "recall"})
        assert "dark mode" in result["result"]
        assert "# Memory" in result["result"]

    def test_memory_multiple_facts(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        main.run_tool("memory", {"action": "remember", "fact": "The API uses OAuth2"})
        main.run_tool("memory", {"action": "remember", "fact": "Alice is the project lead"})
        main.run_tool("memory", {"action": "remember", "fact": "Deploy to us-east-1"})

        result = main.run_tool("memory", {"action": "recall"})
        content = result["result"]
        assert "OAuth2" in content
        assert "Alice" in content
        assert "us-east-1" in content

    def test_memory_search_history(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        main.run_tool("memory", {"action": "remember", "fact": "Meeting with Alice at 3pm"})
        main.run_tool("memory", {"action": "remember", "fact": "Deadline is Friday"})
        main.run_tool("memory", {"action": "remember", "fact": "Meeting notes: discussed auth"})

        # Search for meetings
        result = main.run_tool("memory", {"action": "search", "keyword": "meeting"})
        assert "Alice" in result["result"]
        assert "auth" in result["result"]

        # Search for deadline
        result = main.run_tool("memory", {"action": "search", "keyword": "deadline"})
        assert "Friday" in result["result"]

        # Search for nonexistent
        result = main.run_tool("memory", {"action": "search", "keyword": "unicorn"})
        assert "No matches" in result["result"]

    def test_memory_list_events(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        main.run_tool("memory", {"action": "remember", "fact": "Event 1"})
        main.run_tool("memory", {"action": "remember", "fact": "Event 2"})
        main.run_tool("memory", {"action": "remember", "fact": "Event 3"})

        result = main.run_tool("memory", {"action": "list_events", "count": 2})
        lines = result["result"].strip().splitlines()
        assert len(lines) == 2
        assert "Event 2" in lines[0]
        assert "Event 3" in lines[1]

    def test_memory_empty_recall(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        result = main.run_tool("memory", {"action": "recall"})
        assert "No memories" in result["result"]

    def test_memory_empty_fact_rejected(self, workspace):
        import main

        self._create_memory_skill_via_evolve(workspace)

        result = main.run_tool("memory", {"action": "remember", "fact": ""})
        assert "error" in result

    def test_memory_files_in_sandbox(self, workspace):
        """Verify memory files are created inside sandbox/memory/."""
        import main

        self._create_memory_skill_via_evolve(workspace)

        main.run_tool("memory", {"action": "remember", "fact": "Test fact"})

        memory_dir = workspace["sandbox"] / "memory"
        assert memory_dir.is_dir()
        assert (memory_dir / "MEMORY.md").exists()
        assert (memory_dir / "HISTORY.md").exists()
        assert "Test fact" in (memory_dir / "MEMORY.md").read_text()

    def test_memory_survives_reload(self, workspace):
        """Memory persists when skill is re-loaded (simulates restart)."""
        import main

        self._create_memory_skill_via_evolve(workspace)

        main.run_tool("memory", {"action": "remember", "fact": "Persistent fact"})

        # Simulate restart: re-load skills
        skills = main.load_skills()
        assert any(s["name"] == "memory" for s in skills)

        # Data should still be there
        result = main.run_tool("memory", {"action": "recall"})
        assert "Persistent fact" in result["result"]


# ===========================================================================
# 7. SANDBOX WIPE
# ===========================================================================


class TestSandboxWipe:
    def test_wipe_sandbox(self, workspace):
        import main
        import tools

        tools.write_file("keep_me_not.txt", "bye")
        assert (workspace["sandbox"] / "keep_me_not.txt").exists()

        main.wipe_sandbox()
        assert not (workspace["sandbox"] / "keep_me_not.txt").exists()
        # Sandbox dir itself still exists
        assert workspace["sandbox"].is_dir()

    def test_wipe_skills(self, workspace):
        import main

        skill_dir = workspace["skills"] / "to_delete"
        skill_dir.mkdir()
        (skill_dir / "tool.py").write_text("def run(args): pass")

        main.wipe_skills()
        assert not skill_dir.exists()
        assert workspace["skills"].is_dir()


# ===========================================================================
# 8. CALL_SKILL PARSING (from main loop response text)
# ===========================================================================


class TestCallSkillParsing:
    def test_parse_call_skill(self, workspace):
        """Verify the CALL_SKILL: ... WITH_ARGS: ... parsing works."""
        response = 'I\'ll search for that. CALL_SKILL: memory WITH_ARGS: {"action": "recall"}'
        parts = response.split("CALL_SKILL:")[1]
        skill_name = parts.split("WITH_ARGS:")[0].strip()
        args_str = parts.split("WITH_ARGS:")[1].strip()
        args = json.loads(args_str.split("\n")[0].strip())

        assert skill_name == "memory"
        assert args == {"action": "recall"}

    def test_parse_call_skill_with_trailing_text(self, workspace):
        response = (
            "CALL_SKILL: sushi WITH_ARGS: "
            '{"neighborhood": "Shibuya"}\nLet me know if you need more.'
        )
        parts = response.split("CALL_SKILL:")[1]
        skill_name = parts.split("WITH_ARGS:")[0].strip()
        args_str = parts.split("WITH_ARGS:")[1].strip()
        args = json.loads(args_str.split("\n")[0].strip())

        assert skill_name == "sushi"
        assert args == {"neighborhood": "Shibuya"}


# ===========================================================================
# 9. CRON SKILL — schedule reminders and recurring tasks
# ===========================================================================

CRON_SKILL_CODE = textwrap.dedent("""\
    import os
    import sys
    import json
    import uuid
    from pathlib import Path
    from datetime import datetime

    SANDBOX = os.environ.get("SANDBOX_DIR", "./sandbox")

    def _jobs_file():
        d = Path(SANDBOX) / "cron"
        d.mkdir(parents=True, exist_ok=True)
        return d / "jobs.json"

    def _load_jobs():
        f = _jobs_file()
        if not f.exists():
            return []
        return json.loads(f.read_text())

    def _save_jobs(jobs):
        _jobs_file().write_text(json.dumps(jobs, indent=2))

    def run(args):
        action = args.get("action", "list")

        if action == "add":
            message = args.get("message", "")
            if not message:
                return {"error": "No message provided"}
            job = {
                "job_id": str(uuid.uuid4())[:8],
                "message": message,
                "created": datetime.now().isoformat(),
                "type": "one-time" if args.get("at") else "recurring",
            }
            if args.get("every_seconds"):
                job["every_seconds"] = args["every_seconds"]
            if args.get("cron_expr"):
                job["cron_expr"] = args["cron_expr"]
            if args.get("at"):
                job["at"] = args["at"]
            jobs = _load_jobs()
            jobs.append(job)
            _save_jobs(jobs)
            return {"result": f"Job {job['job_id']} added", "job_id": job["job_id"]}

        elif action == "list":
            jobs = _load_jobs()
            if not jobs:
                return {"result": "No scheduled jobs."}
            return {"result": jobs}

        elif action == "remove":
            job_id = args.get("job_id", "")
            jobs = _load_jobs()
            before = len(jobs)
            jobs = [j for j in jobs if j["job_id"] != job_id]
            if len(jobs) == before:
                return {"error": f"Job {job_id} not found"}
            _save_jobs(jobs)
            return {"result": f"Job {job_id} removed"}

        else:
            return {"error": f"Unknown action: {action}"}

    if __name__ == "__main__":
        input_args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        print(json.dumps(run(input_args)))
""")


class TestCronSkill:
    """End-to-end: create a cron/scheduler skill via evolve(), run it."""

    def _create_cron_skill(self, workspace):
        import main

        gemini_response = json.dumps(
            {
                "name": "cron",
                "description": "Schedule reminders and recurring tasks.",
                "triggers": ["remind me", "schedule", "every hour", "set timer"],
                "python_code": CRON_SKILL_CODE,
            }
        )
        mock_brain = MagicMock()
        mock_brain.chat.return_value = gemini_response
        history = [
            {"role": "system", "content": "You are EvolveBot..."},
            {"role": "user", "content": "GOAL: schedule recurring reminders"},
            {"role": "assistant", "content": "I created a cron scheduler. [[SUCCESS]]"},
        ]
        with patch.object(main, "Brain", return_value=mock_brain):
            return main.evolve(history, [])

    def test_cron_skill_created(self, workspace):
        name = self._create_cron_skill(workspace)
        assert name == "cron"
        assert (workspace["skills"] / "cron" / "tool.py").exists()

    def test_cron_add_recurring(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool(
            "cron", {"action": "add", "message": "Take a break!", "every_seconds": 1200}
        )
        assert "job_id" in result
        assert "added" in result["result"]

    def test_cron_add_one_time(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool(
            "cron", {"action": "add", "message": "Meeting reminder", "at": "2026-02-16T15:00:00"}
        )
        assert "job_id" in result

        # Verify it's stored as one-time
        jobs_file = workspace["sandbox"] / "cron" / "jobs.json"
        jobs = json.loads(jobs_file.read_text())
        assert jobs[0]["type"] == "one-time"
        assert jobs[0]["at"] == "2026-02-16T15:00:00"

    def test_cron_add_cron_expression(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool(
            "cron", {"action": "add", "message": "Daily standup", "cron_expr": "0 9 * * 1-5"}
        )
        assert "job_id" in result

    def test_cron_list_empty(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool("cron", {"action": "list"})
        assert "No scheduled" in result["result"]

    def test_cron_list_with_jobs(self, workspace):
        import main

        self._create_cron_skill(workspace)

        main.run_tool("cron", {"action": "add", "message": "Job A", "every_seconds": 60})
        main.run_tool("cron", {"action": "add", "message": "Job B", "every_seconds": 120})

        result = main.run_tool("cron", {"action": "list"})
        jobs = result["result"]
        assert len(jobs) == 2
        assert jobs[0]["message"] == "Job A"
        assert jobs[1]["message"] == "Job B"

    def test_cron_remove(self, workspace):
        import main

        self._create_cron_skill(workspace)

        add_result = main.run_tool(
            "cron", {"action": "add", "message": "Remove me", "every_seconds": 300}
        )
        job_id = add_result["job_id"]

        # Remove it
        result = main.run_tool("cron", {"action": "remove", "job_id": job_id})
        assert "removed" in result["result"]

        # Should be empty now
        result = main.run_tool("cron", {"action": "list"})
        assert "No scheduled" in result["result"]

    def test_cron_remove_nonexistent(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool("cron", {"action": "remove", "job_id": "fake123"})
        assert "error" in result

    def test_cron_add_no_message(self, workspace):
        import main

        self._create_cron_skill(workspace)

        result = main.run_tool("cron", {"action": "add"})
        assert "error" in result

    def test_cron_jobs_persist_in_sandbox(self, workspace):
        import main

        self._create_cron_skill(workspace)

        main.run_tool("cron", {"action": "add", "message": "Persist me", "every_seconds": 60})

        jobs_file = workspace["sandbox"] / "cron" / "jobs.json"
        assert jobs_file.exists()
        data = json.loads(jobs_file.read_text())
        assert len(data) == 1
        assert data[0]["message"] == "Persist me"


# ===========================================================================
# 10. WEATHER SKILL — fetch weather via curl (mocked)
# ===========================================================================

WEATHER_SKILL_CODE = textwrap.dedent("""\
    import os
    import sys
    import json
    import subprocess
    from pathlib import Path

    SANDBOX = os.environ.get("SANDBOX_DIR", "./sandbox")

    def run(args):
        location = args.get("location", "London")
        fmt = args.get("format", "short")

        # URL-encode spaces
        loc_encoded = location.replace(" ", "+")

        if fmt == "short":
            format_str = "%l:+%c+%t+%h+%w"
        elif fmt == "full":
            format_str = None  # full forecast
        else:
            format_str = fmt

        if format_str:
            url = f"https://wttr.in/{loc_encoded}?format={format_str}"
        else:
            url = f"https://wttr.in/{loc_encoded}?T"

        try:
            result = subprocess.run(
                ["curl", "-s", url],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                # Save to sandbox for history
                history_dir = Path(SANDBOX) / "weather"
                history_dir.mkdir(parents=True, exist_ok=True)
                (history_dir / f"{loc_encoded}.txt").write_text(result.stdout)
                return {"result": result.stdout.strip(), "location": location}
            return {"error": f"Failed to fetch weather: {result.stderr}"}
        except FileNotFoundError:
            return {"error": "curl not available"}
        except subprocess.TimeoutExpired:
            return {"error": "Weather request timed out"}

    if __name__ == "__main__":
        input_args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        print(json.dumps(run(input_args)))
""")


class TestWeatherSkill:
    """Weather skill — creation and execution with mocked curl."""

    def _create_weather_skill(self, workspace):
        import main

        gemini_response = json.dumps(
            {
                "name": "weather",
                "description": "Get current weather and forecasts via wttr.in (no API key).",
                "trigger_phrases": ["weather", "forecast", "temperature", "rain today"],
                "python_code": WEATHER_SKILL_CODE,
            }
        )
        mock_brain = MagicMock()
        mock_brain.chat.return_value = gemini_response
        history = [
            {"role": "system", "content": "You are EvolveBot..."},
            {"role": "user", "content": "GOAL: check the weather"},
            {"role": "assistant", "content": "I'll use wttr.in for weather. [[SUCCESS]]"},
        ]
        with patch.object(main, "Brain", return_value=mock_brain):
            return main.evolve(history, [])

    def test_weather_skill_created_with_trigger_phrases(self, workspace):
        name = self._create_weather_skill(workspace)
        assert name == "weather"

        import yaml as _yaml

        with open(workspace["skills"] / "weather" / "skill.yaml") as f:
            meta = _yaml.safe_load(f)
        assert "weather" in meta["triggers"]
        assert "forecast" in meta["triggers"]

    @patch("subprocess.run")
    def test_weather_fetch_short(self, mock_run, workspace):
        import main

        self._create_weather_skill(workspace)

        # Mock curl response
        mock_run.return_value = MagicMock(
            returncode=0, stdout="London: ⛅️ +8°C 71% ↙5km/h", stderr=""
        )

        result = main.run_tool("weather", {"location": "London", "format": "short"})
        assert result["location"] == "London"
        assert "+8°C" in result["result"]

    @patch("subprocess.run")
    def test_weather_saves_to_sandbox(self, mock_run, workspace):
        import main

        self._create_weather_skill(workspace)

        mock_run.return_value = MagicMock(
            returncode=0, stdout="Tokyo: ☀️ +15°C 55% →3km/h", stderr=""
        )

        main.run_tool("weather", {"location": "Tokyo"})

        weather_file = workspace["sandbox"] / "weather" / "Tokyo.txt"
        assert weather_file.exists()
        assert "15°C" in weather_file.read_text()

    @patch("subprocess.run")
    def test_weather_url_encodes_spaces(self, mock_run, workspace):
        import main

        self._create_weather_skill(workspace)

        mock_run.return_value = MagicMock(returncode=0, stdout="New York: 🌧 +5°C", stderr="")

        result = main.run_tool("weather", {"location": "New York"})
        assert result["location"] == "New York"

        # Verify the file was saved with + encoding
        assert (workspace["sandbox"] / "weather" / "New+York.txt").exists()

    @patch("subprocess.run")
    def test_weather_curl_failure(self, mock_run, workspace):
        import main

        self._create_weather_skill(workspace)

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection refused")

        result = main.run_tool("weather", {"location": "Nowhere"})
        assert "error" in result

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_weather_no_curl(self, mock_run, workspace):
        import main

        self._create_weather_skill(workspace)

        result = main.run_tool("weather", {"location": "London"})
        assert result["error"] == "curl not available"
