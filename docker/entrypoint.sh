#!/bin/bash
set -e

OLLAMA_URL="${OPENAI_BASE_URL%/v1}"  # strip /v1 suffix
MODEL="${EXECUTOR_MODEL:-qwen3:8b}"

echo "[entrypoint] Waiting for Ollama at $OLLAMA_URL ..."
for i in $(seq 1 30); do
    if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "[entrypoint] Ollama is ready."
        break
    fi
    sleep 2
done

# Pull model if not already present
if ! curl -sf "$OLLAMA_URL/api/tags" | python3 -c "
import sys, json
tags = json.load(sys.stdin)
names = [m['name'] for m in tags.get('models', [])]
sys.exit(0 if any('$MODEL' in n for n in names) else 1)
" 2>/dev/null; then
    echo "[entrypoint] Pulling $MODEL ..."
    curl -sf "$OLLAMA_URL/api/pull" -d "{\"name\": \"$MODEL\"}" | while read -r line; do
        status=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
        [ -n "$status" ] && echo "[pull] $status"
    done
    echo "[entrypoint] Model $MODEL ready."
else
    echo "[entrypoint] Model $MODEL already available."
fi

exec python3 src/main.py "$@"
