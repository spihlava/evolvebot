FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY docker/entrypoint.sh /app/

RUN mkdir -p skills sandbox

ENV EXECUTOR_MODEL=MiniMax-M2.5
ENV MINIMAX_MODEL=MiniMax-M2.5
ENV GEMINI_MODEL=gemini-2.5-flash
ENV EVOLVE_PROVIDER=minimax
ENV EXEC_MODE=direct

ENTRYPOINT ["bash", "entrypoint.sh"]
