#!/usr/bin/env bash
# wipe.sh — Clean up runtime directories
#
# Usage:
#   ./wipe.sh              Wipe sandbox (preserves memory and skills)
#   ./wipe.sh --memory     Also wipe sandbox/memory/
#   ./wipe.sh --skills     Also wipe skills/
#   ./wipe.sh --all        Wipe everything (sandbox + memory + skills)
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SANDBOX_DIR="$PROJECT_DIR/sandbox"
SKILLS_DIR="$PROJECT_DIR/skills"
MEMORY_DIR="$SANDBOX_DIR/memory"

wipe_memory=false
wipe_skills=false

for arg in "$@"; do
    case "$arg" in
        --memory)  wipe_memory=true ;;
        --skills)  wipe_skills=true ;;
        --all)     wipe_memory=true; wipe_skills=true ;;
        -h|--help)
            sed -n '2,7p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg (try --help)"; exit 1 ;;
    esac
done

# Wipe sandbox (preserve memory/ unless --memory or --all)
if [ -d "$SANDBOX_DIR" ]; then
    for item in "$SANDBOX_DIR"/*; do
        [ ! -e "$item" ] && continue
        basename="$(basename "$item")"
        [ "$basename" = ".gitkeep" ] && continue
        if [ "$basename" = "memory" ] && [ "$wipe_memory" = false ]; then
            continue
        fi
        rm -rf "$item"
    done
    echo "[wipe] sandbox/ cleaned"
else
    echo "[wipe] sandbox/ not found, skipping"
fi

# Wipe memory
if [ "$wipe_memory" = true ] && [ -d "$MEMORY_DIR" ]; then
    rm -rf "$MEMORY_DIR"
    echo "[wipe] sandbox/memory/ removed"
fi

# Wipe skills (preserve builtins)
if [ "$wipe_skills" = true ] && [ -d "$SKILLS_DIR" ]; then
    for item in "$SKILLS_DIR"/*/; do
        [ ! -d "$item" ] && continue
        # Skip builtin skills (have "builtin: true" in skill.yaml)
        if [ -f "$item/skill.yaml" ] && grep -q "^builtin: true" "$item/skill.yaml" 2>/dev/null; then
            echo "[wipe] keeping builtin: $(basename "$item")"
            continue
        fi
        rm -rf "$item"
    done
    echo "[wipe] skills/ cleaned (builtins preserved)"
fi

echo "[wipe] done"
