#!/usr/bin/env bash
#
# One-shot launcher for the Multi-Mouse Behavioral Dashboard.
#
#   ./start.sh
#
# Configures Ollama for strict one-model-at-a-time, makes sure the three
# chatbot models are pulled, warms the default one, and opens the dashboard.
#
set -euo pipefail

# Keep this order in sync with DEFAULT_MODELS in app.py — the app's dropdown
# defaults to its first entry, and we preload ours, so a mismatch means the
# warmed model isn't the selected one and the first message pays a cold load.
MODELS=("llama3.2:3b" "gemma3:1b" "qwen3.5:4b")
DEFAULT_MODEL="${MODELS[0]}"
KEEP_ALIVE="10m"
PORT=8502
HOST_URL="http://127.0.0.1:11434"

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Server config: only one model resident, idle models release RAM after 10m.
#    launchctl setenv is what the Ollama *app* reads; the exports cover the case
#    where we fall back to launching `ollama serve` ourselves below.
# ---------------------------------------------------------------------------
echo "→ Configuring Ollama (max 1 loaded model, ${KEEP_ALIVE} keep-alive)…"
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_KEEP_ALIVE "$KEEP_ALIVE"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="$KEEP_ALIVE"

# ---------------------------------------------------------------------------
# 2. Restart Ollama so it picks the new environment up. A server started before
#    `launchctl setenv` ran is still using the old values, so a plain "is it
#    up?" check isn't enough — it has to be bounced at least once.
# ---------------------------------------------------------------------------
old_serve_pid="$(pgrep -f 'ollama serve' 2>/dev/null | head -1 || true)"

if [ -n "$old_serve_pid" ] || pgrep -x Ollama >/dev/null 2>&1; then
  echo "→ Restarting Ollama to apply settings…"
  # 'quit app "Ollama"' does NOT work on the menubar app, and even when the GUI
  # does go away the 'ollama serve' child survives and keeps holding 11434 —
  # so the relaunched app reuses the stale server and the env never applies.
  # Kill the server process explicitly; that's the one that reads the config.
  osascript -e 'quit app "Ollama"' >/dev/null 2>&1 || true
  pkill -x Ollama >/dev/null 2>&1 || true
  pkill -f "ollama serve" >/dev/null 2>&1 || true

  for _ in $(seq 1 20); do
    pgrep -f "ollama serve" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "  server won't exit gracefully — forcing"
    pkill -9 -f "ollama serve" >/dev/null 2>&1 || true
    sleep 1
  fi
fi

launched=false
if [ -d /Applications/Ollama.app ]; then
  # Right after a kill, macOS may still be tearing the app down and `open`
  # fails with -600 (procNotFound). It clears in a second or two, so retry.
  for _ in $(seq 1 10); do
    if open -a Ollama 2>/dev/null; then
      launched=true
      break
    fi
    sleep 1
  done
fi

if [ "$launched" = false ]; then
  # No app, or the GUI refused to relaunch — run the server ourselves. This
  # inherits the exports above, so the limit applies either way. nohup so it
  # outlives this script.
  echo "  (starting 'ollama serve' directly)"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  sleep 2
fi

echo -n "→ Waiting for Ollama to accept connections"
for _ in $(seq 1 60); do
  if curl -sf "${HOST_URL}/api/tags" >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -sf "${HOST_URL}/api/tags" >/dev/null 2>&1; then
  echo
  echo "✗ Ollama did not come up on ${HOST_URL}. Try opening the Ollama app manually."
  exit 1
fi

# A reachable server proves nothing about *which* server answered — a stale
# process that never saw the config responds just as happily. Confirm the
# running process actually carries the limit.
new_serve_pid="$(pgrep -f 'ollama serve' 2>/dev/null | head -1 || true)"
if [ -n "$new_serve_pid" ]; then
  if ps eww -p "$new_serve_pid" 2>/dev/null | tr ' ' '\n' | grep -q "^OLLAMA_MAX_LOADED_MODELS=1$"; then
    echo "→ Verified: server PID ${new_serve_pid} has OLLAMA_MAX_LOADED_MODELS=1"
  else
    echo "✗ Server PID ${new_serve_pid} is running WITHOUT OLLAMA_MAX_LOADED_MODELS=1."
    echo "  Models could stack in RAM. Quit Ollama from the menubar and rerun this script."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 3. Make sure all three models exist locally. They are NOT all loaded into RAM
#    — that would defeat MAX_LOADED_MODELS=1 and swap a 16 GB machine. Only the
#    default is warmed; the sidebar dropdown swaps the others in on demand.
# ---------------------------------------------------------------------------
installed="$(ollama list | tail -n +2 | awk '{print $1}')"
for m in "${MODELS[@]}"; do
  if echo "$installed" | grep -qx "$m"; then
    echo "→ ${m} … already pulled"
  else
    echo "→ ${m} … pulling (first run only)"
    ollama pull "$m"
  fi
done

# ---------------------------------------------------------------------------
# 4. Warm the default model so the first question doesn't pay the load cost.
#    A request with a model but an empty prompt loads it without generating.
# ---------------------------------------------------------------------------
echo "→ Preloading ${DEFAULT_MODEL}…"
curl -sf "${HOST_URL}/api/generate" \
  -d "{\"model\":\"${DEFAULT_MODEL}\",\"prompt\":\"\",\"keep_alive\":\"${KEEP_ALIVE}\"}" \
  >/dev/null || echo "  (preload failed — the app will load it on first message)"

echo
echo "Resident models (should never be more than one):"
ollama ps
echo

# ---------------------------------------------------------------------------
# 5. Dashboard. A previous run of this same app may still be holding the port;
#    take it over so you don't end up with two copies, one on stale code.
#    Anything else on the port is left alone and we fail loudly instead.
# ---------------------------------------------------------------------------
holder="$(lsof -nP -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$holder" ]; then
  if ps -p "$holder" -o command= | grep -q "streamlit run app.py"; then
    echo "→ Port ${PORT} held by an earlier dashboard (PID ${holder}) — restarting it…"
    kill "$holder" 2>/dev/null || true
    for _ in $(seq 1 20); do
      lsof -nP -tiTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 0.5
    done
  else
    echo "✗ Port ${PORT} is in use by something that isn't this dashboard:"
    ps -p "$holder" -o pid=,command=
    echo "  Stop it, or change PORT at the top of this script."
    exit 1
  fi
fi

echo "→ Starting dashboard on http://localhost:${PORT}"
echo "  (Ctrl-C to stop. Ollama keeps running; 'ollama stop <model>' frees RAM.)"
echo
exec streamlit run app.py --server.port "$PORT"
