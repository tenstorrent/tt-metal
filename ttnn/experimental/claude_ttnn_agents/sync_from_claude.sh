#!/bin/bash
# Sync files FROM .claude back to claude_ttnn_agents source folder
# Use this when you've made changes in .claude and want to commit them
#
# Uses rsync to comprehensively mirror .claude/ → experimental/, excluding
# runtime-only files and preserving experimental-only files.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SRC="$REPO_ROOT/.claude/"
DST="$SCRIPT_DIR/"

echo "Syncing from .claude to claude_ttnn_agents source..."
echo "  Source: $SRC"
echo "  Dest:   $DST"
echo ""

# Check if .claude directory exists
if [ ! -d "$SRC" ]; then
    echo "Error: $SRC does not exist"
    echo "Run activate_agents.sh first to set up the .claude directory"
    exit 1
fi

# Common rsync flags:
#   -a            archive (preserves permissions, timestamps, etc.)
#   -v            verbose
#   --delete      remove files in DST that are not in SRC (after excludes)
#
# Excludes — runtime-only files that live in .claude/ but shouldn't be committed:
#   settings.local.json   local Claude Code settings
#   active_logging        runtime logging state
#   logs/                 runtime logs
#   __pycache__/          Python bytecode cache
#
# Excludes — experimental-only files that don't exist in .claude/:
#   README.md             experimental directory readme
#   activate_agents.sh    deployment script (experimental → .claude direction)
#   sync_from_claude.sh   this script itself

RSYNC_OPTS=(
    -a
    --delete
    --exclude='settings.local.json'
    --exclude='active_logging'
    --exclude='logs/'
    --exclude='__pycache__/'
    --exclude='README.md'
    --exclude='activate_agents.sh'
    --exclude='sync_from_claude.sh'
)

# Dry-run first
echo "=== Dry run (changes that would be made) ==="
echo ""
rsync "${RSYNC_OPTS[@]}" --dry-run -v "$SRC" "$DST" 2>&1 | grep -v '^\.' | grep -v '^$' || true
echo ""

# Confirm
read -p "Apply these changes? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Apply
rsync "${RSYNC_OPTS[@]}" -v "$SRC" "$DST"

echo ""
echo "=============================================="
echo "Sync complete!"
echo "=============================================="
echo ""
echo "Files synced to: $DST"
echo ""
echo "You can now review and commit changes with:"
echo "  git diff $SCRIPT_DIR"
echo "  git add $SCRIPT_DIR && git commit -m 'Update claude agents'"
echo ""
