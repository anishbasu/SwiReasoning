#!/bin/bash

# --- CONFIGURATION ---
LOCAL_DIR="$(pwd)/"
REMOTE_TARGET="root@103.196.86.132:/workspace/"
PORT="16613"
# ---------------------

# Ensure tools are installed locally
if ! command -v fswatch &> /dev/null || ! command -v rsync &> /dev/null; then
    echo "Error: fswatch or rsync is not installed locally. Run: brew install fswatch rsync"
    exit 1
fi

# The exact flags that worked in your terminal
RSYNC_FLAGS=(
    -avz
    -e "ssh -p $PORT"
    --exclude=".git/"
    --exclude="sync.sh"
    --filter=":- .gitignore"
)

echo "Performing initial sync..."
rsync "${RSYNC_FLAGS[@]}" "$LOCAL_DIR" "$REMOTE_TARGET"

echo "Watching $LOCAL_DIR for changes..."
echo "Press [CTRL+C] to stop."

# Passing the absolute path ($LOCAL_DIR) instead of "." prevents fswatch bugs
fswatch -o -t 0.5 "$LOCAL_DIR" | while read -r event; do
    echo "Change detected. Syncing..."
    rsync "${RSYNC_FLAGS[@]}" "$LOCAL_DIR" "$REMOTE_TARGET"
    echo "Sync complete."
done