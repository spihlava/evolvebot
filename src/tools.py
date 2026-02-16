"""
tools.py — Sandbox-aware file and command helpers for generated skills.

This module is automatically injected into skills when run in direct mode.
Skills running in docker/chroot mode should use sys.argv[1] for args and
print JSON to stdout.

All file operations are restricted to SANDBOX_DIR (./sandbox/ by default).
"""

import os
import subprocess
from pathlib import Path

# Anchor paths to project root
PROJECT_ROOT = Path(__file__).absolute().parent.parent
DEFAULT_SANDBOX = PROJECT_ROOT / "sandbox"

# Default; overridden by _run_direct before injection
SANDBOX_DIR_STR = os.environ.get("SANDBOX_DIR", str(DEFAULT_SANDBOX.absolute()))
SANDBOX_DIR = Path(SANDBOX_DIR_STR)
SANDBOX_DIR.mkdir(exist_ok=True, parents=True)


def _get_sandbox():
    """Return current sandbox path — reads module-level SANDBOX_DIR."""
    # Use the global which _run_direct sets before injecting
    return SANDBOX_DIR


def _safe_path(path):
    """Resolve a path relative to sandbox, reject escapes."""
    sandbox = Path(os.path.realpath(_get_sandbox()))
    target = Path(os.path.realpath(sandbox / path))
    if not str(target).startswith(str(sandbox)):
        raise ValueError(f"Path escapes sandbox: {path}")
    return target


def read_file(path):
    """Read a file from sandbox/."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: {path} not found"
    return target.read_text()


def write_file(path, content):
    """Write a file into sandbox/."""
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: {target.relative_to(Path(os.path.realpath(_get_sandbox())))}"


def list_files(dir_path="."):
    """List files in a sandbox subdirectory."""
    target = _safe_path(dir_path)
    if not target.is_dir():
        return []
    return [f.name for f in target.iterdir()]


def run_command(command):
    """Run a shell command with sandbox/ as cwd. No network access."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15, cwd=_get_sandbox()
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (15s)"


# ============================================================================
# GRADERS (Cookbook-inspired)
# ============================================================================


class Grader:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def grade(self, source, summary):
        raise NotImplementedError


class LengthGrader(Grader):
    """Measures deviation from an expected length."""

    def __init__(self, target_len=100, tolerance=20):
        super().__init__("length")
        self.target_len = target_len
        self.tolerance = tolerance

    def grade(self, _source, summary):
        words = len(summary.split())
        diff = abs(words - self.target_len)
        score = max(0, 1 - (diff / self.tolerance))
        passed = score >= 0.7
        feedback = f"Length {words} words (target {self.target_len} ± {self.tolerance})"
        return {"score": score, "passed": passed, "feedback": feedback}


class EntityMatchGrader(Grader):
    """Checks if critical entities from source are in summary."""

    def __init__(self, entities=None):
        super().__init__("entity_match")
        self.entities = entities or []

    def grade(self, source, summary):
        found = [e for e in self.entities if e.lower() in summary.lower()]
        score = len(found) / len(self.entities) if self.entities else 1.0
        missing = [e for e in self.entities if e not in found]
        passed = score >= 0.8
        feedback = f"Matched {len(found)}/{len(self.entities)} entities. Missing: {missing}"
        return {"score": score, "passed": passed, "feedback": feedback}
