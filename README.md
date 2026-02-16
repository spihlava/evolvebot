# EvolveBot

Self-Evolving AI Agent with Skill Synthesis and Autonomous Retraining — under 1000 lines of Python.

## Quick Start

### Local (recommended)

```bash
# 1. Copy env and add your API keys (Minimax, Gemini, or Ollama)
cp .env.example .env
# edit .env → set MINIMAX_API_KEY, set EVOLVE_PROVIDER=minimax

# 2. Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the bot
./scripts/cli.sh                     # interactive mode
./scripts/cli.sh "summarize this text" # pass goal directly
```

### Docker (Minimax/Cloud)

```bash
cp .env.example .env
# edit .env → set MINIMAX_API_KEY, set EVOLVE_PROVIDER=minimax

docker compose up -d --build
docker attach evolvebot-app
# Ctrl+P, Ctrl+Q to detach without stopping
```

## How It Works

1. **Conversation** — Executor chats with user, solves tasks using available skills.
2. **Autonomous Retraining** — If a task requires specific constraints (e.g., "summary must be 50 words"), use the `retrain` command. The bot will iteratively optimize its own **System Prompt** using a grading loop.
3. **Grading Engine** — Outputs are evaluated by multiple graders:
    - **LengthGrader**: Validates word counts.
    - **EntityMatchGrader**: Ensures critical data is preserved.
    - **LLMGrader**: "LLM-as-a-judge" evaluates quality against a rubric.
4. **Metaprompt Optimization** — When grades are low, a Metaprompt Architect analyzes the failures and rewrites the agent's instructions.
5. **Skill Synthesis** — When a goal is reached (`[[SUCCESS]]`), the Architect analyzes the conversation and generates a new permanent Python skill in `skills/`.

## Provider Configuration

| Provider | Use case | Config |
|----------|----------|--------|
| **Minimax** | Primary: Fast, high-reasoning, Anthropic-compatible | `EVOLVE_PROVIDER=minimax` |
| **Gemini** | Cloud, fast, native Google support | `EVOLVE_PROVIDER=gemini` |
| **Ollama** | Local, private, requires GPU | `EVOLVE_PROVIDER=ollama` |

## Interactive Commands

| Command | Effect |
|---------|--------|
| `exit` / `quit` | End session |
| `retrain` | Start an iterative prompt optimization loop for the current goal |
| `force-evolve` | Manually trigger skill synthesis from history |
| `wipe-sandbox` | Clear all files in `sandbox/` |
| `wipe-skills` | Delete all generated skills |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVOLVE_PROVIDER` | minimax | `minimax`, `gemini`, or `ollama` |
| `EXECUTOR_MODEL` | MiniMax-M2.5 | Model for conversation & retraining |
| `MINIMAX_MODEL` | MiniMax-M2.5 | Model for skill synthesis & architecting |
| `MINIMAX_API_KEY` | (required) | Minimax API key |
| `MINIMAX_API_BASE` | https://api.minimax.io/anthropic | Anthropic-compatible endpoint |
| `EXEC_MODE` | direct | `direct` / `chroot` / `docker` |

## Directory Structure

```
evolvebot/
├── src/                    # Source code
│   ├── main.py             #   Core: Brain, skills engine, evolution, main loop
│   └── tools.py            #   Sandbox-aware helpers injected into skills
├── tests/                  # Test suite
│   ├── conftest.py         #   Shared fixtures (workspace)
│   └── test_evolvebot.py   #   All tests
├── docker/                 # Docker config
│   ├── Dockerfile
│   └── entrypoint.sh       #   Waits for Ollama, auto-pulls model
├── scripts/                # Utility scripts
│   ├── cli.sh              #   Run bot locally
│   └── wipe.sh             #   Clean up runtime dirs
├── skills/                 # Skills (builtin + generated)
│   ├── shell/              #   Builtin: commands, files, scripts
│   └── <generated>/        #   Auto-created by evolve()
├── sandbox/                # Wipable work directory for file I/O
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Security

- Generated code is syntax-checked before saving
- `tools.py` rejects path traversal attempts (`../`)
- Docker exec mode runs tools with `--network none`
- Sandbox is the only writable directory for skills
- API keys belong in `.env` (gitignored) — never commit them
