"""Shared pytest fixtures for EvolveBot tests."""

import pytest


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Create isolated skills/ and sandbox/ per test, patch main to use them."""
    skills = tmp_path / "skills"
    sandbox = tmp_path / "sandbox"
    skills.mkdir()
    sandbox.mkdir()

    import main

    monkeypatch.setattr(main, "SKILLS_DIR", skills)
    monkeypatch.setattr(main, "SANDBOX_DIR", sandbox)
    monkeypatch.setattr(main, "EXEC_MODE", "direct")

    import tools

    monkeypatch.setattr(tools, "SANDBOX_DIR", str(sandbox))

    yield {"skills": skills, "sandbox": sandbox}
