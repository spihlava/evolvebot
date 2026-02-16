"""
Shell skill — bootstrap tool for running commands, reading/writing files,
and executing scripts in the sandbox.

Actions:
  run_command  — execute a shell command (cwd=sandbox)
  write_file   — write content to a file in sandbox
  read_file    — read a file from sandbox
  list_files   — list files in a sandbox subdirectory
  run_script   — write a Python script to sandbox and execute it
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SANDBOX = os.environ.get("SANDBOX_DIR", "./sandbox")


def _safe_path(path):
    sandbox = Path(os.path.realpath(SANDBOX))
    target = Path(os.path.realpath(sandbox / path))
    if not str(target).startswith(str(sandbox)):
        raise ValueError(f"Path escapes sandbox: {path}")
    return target


def run(args):
    action = args.get("action", "run_command")

    if action == "run_command":
        command = args.get("command", "")
        if not command:
            return {"error": "No command provided"}
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30, cwd=SANDBOX
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                output = result.stderr.strip() or output
                return {"error": output or f"Exit code {result.returncode}"}
            return {"result": output}
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (30s)"}

    elif action == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return {"error": "No path provided"}
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"result": f"Written: {path}"}

    elif action == "read_file":
        path = args.get("path", "")
        if not path:
            return {"error": "No path provided"}
        target = _safe_path(path)
        if not target.exists():
            return {"error": f"File not found: {path}"}
        return {"result": target.read_text()}

    elif action == "list_files":
        path = args.get("path", ".")
        target = _safe_path(path)
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}
        return {"result": [f.name for f in target.iterdir()]}

    elif action == "run_script":
        script = args.get("script", "")
        if not script:
            return {"error": "No script provided"}
        script_path = _safe_path("_tmp_script.py")
        script_path.write_text(script)
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=SANDBOX,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                output = result.stderr.strip() or output
                return {"error": output or f"Exit code {result.returncode}"}
            return {"result": output}
        except subprocess.TimeoutExpired:
            return {"error": "Script timed out (30s)"}
        finally:
            script_path.unlink(missing_ok=True)

    else:
        return {
            "error": f"Unknown action: {action}. "
            "Use: run_command, write_file, read_file, list_files, run_script"
        }


if __name__ == "__main__":
    input_args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(input_args)))
