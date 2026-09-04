#!/usr/bin/env bash
#
# Shuts down what start.sh brought up.
#
#   ./stop.sh          stop the dashboard, unload models (Ollama keeps running)
#   ./stop.sh --all    also quit the Ollama app entirely
#
# Only touches `streamlit run app.py` on this project's port — any other
# Streamlit app you have running is left alone.
#
set -uo pipefail

PORT=8502
HOST_URL="http://127.0.0.1:11434"
QUIT_OLLAMA=false
[ "${1:-}" = "--all" ] && QUIT_OLLAMA=true

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Dashboard. Match on the command line, not just the port, so we never kill
#    somebody else's server that happens to have claimed 8502.
# ---------------------------------------------------------------------------
killed_any=false
for pid in $(pgrep -f "streamlit run app.py" 2>/dev/null); do
  echo "→ Stopping dashboard (PID ${pid})…"
  kill "$pid" 2>/dev/null
  killed_any=true
done

if [ "$killed_any" = true ]; then
  # give it a moment to release the port, then escalate if it's wedged
  for _ in $(seq 1 20); do
    pgrep -f "streamlit run app.py" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if pgrep -f "streamlit run app.py" >/dev/null 2>&1; then
    echo "  still alive after 10s — sending SIGKILL"
    pkill -9 -f "streamlit run app.py" 2>/dev/null
    sleep 1
  fi
else
  echo "→ No dashboard running."
fi

if lsof -nP -tiTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  note: port ${PORT} is still held by something else:"
  lsof -nP -iTCP:${PORT} -sTCP:LISTEN | tail -n +2 | sed 's/^/    /'
else
  echo "  port ${PORT} free ✓"
fi

# ---------------------------------------------------------------------------
# 2. Release model RAM now rather than waiting out the keep-alive timer.
# ---------------------------------------------------------------------------
if curl -sf "${HOST_URL}/api/tags" >/dev/null 2>&1; then
  resident="$(ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}')"
  if [ -n "$resident" ]; then
    for m in $resident; do
      echo "→ Unloading ${m}…"
      ollama stop "$m" >/dev/null 2>&1 || true
    done
  else
    echo "→ No models resident."
  fi
else
  echo "→ Ollama not reachable; nothing to unload."
fi

# ---------------------------------------------------------------------------
# 3. Optionally shut the server down too. Off by default: leaving it running
#    costs nothing once models are unloaded, and makes the next start fast.
# ---------------------------------------------------------------------------
if [ "$QUIT_OLLAMA" = true ]; then
  echo "→ Quitting Ollama…"
  osascript -e 'quit app "Ollama"' >/dev/null 2>&1 || true
  pkill -f "ollama serve" 2>/dev/null
  sleep 1
fi

echo
echo "Resident models:"
ollama ps 2>/dev/null || echo "  (Ollama not running)"
echo
echo "Done. Restart with ./start.sh"
