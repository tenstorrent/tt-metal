#!/bin/bash
# Sync files FROM .claude back to claude_ttnn_agents source folder
# Use this when you've made changes in .claude and want to commit them

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Syncing from .claude to claude_ttnn_agents source..."
echo "Repository root: $REPO_ROOT"
echo ""

# Check if .claude directory exists
if [ ! -d "$REPO_ROOT/.claude" ]; then
    echo "Error: $REPO_ROOT/.claude does not exist"
    echo "Run activate_agents.sh first to set up the .claude directory"
    exit 1
fi

# Sync agent definitions
echo "Syncing agent definitions..."
if [ -d "$REPO_ROOT/.claude/agents" ]; then
    cp "$REPO_ROOT/.claude/agents/ttnn-operation-analyzer.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-operation-planner.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-operation-scaffolder.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-factory-builder.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-kernel-designer.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-kernel-writer.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-riscv-debugger.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    cp "$REPO_ROOT/.claude/agents/ttnn-pipeline-analyzer.md" "$SCRIPT_DIR/agents/" 2>/dev/null || true
    echo "  - Synced agent definitions"
else
    echo "  - Skipping agents (directory not found)"
fi

# Sync scaffolder scripts
echo "Syncing scaffolder scripts..."
if [ -d "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder" ]; then
    cp -r "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder/"* "$SCRIPT_DIR/scripts/ttnn-operation-scaffolder/"
    echo "  - Synced scaffolder scripts"
else
    echo "  - Skipping scaffolder scripts (directory not found)"
fi

# Sync logging scripts
echo "Syncing logging scripts..."
if [ -d "$REPO_ROOT/.claude/scripts/logging" ]; then
    cp -r "$REPO_ROOT/.claude/scripts/logging/"* "$SCRIPT_DIR/scripts/logging/"
    echo "  - Synced logging scripts"
else
    echo "  - Skipping logging scripts (directory not found)"
fi

# Sync reference documents
echo "Syncing reference documents..."
if [ -d "$REPO_ROOT/.claude/references" ]; then
    cp "$REPO_ROOT/.claude/references/"*.md "$SCRIPT_DIR/references/" 2>/dev/null || true
    # Sync logging subdirectory
    if [ -d "$REPO_ROOT/.claude/references/logging" ]; then
        mkdir -p "$SCRIPT_DIR/references/logging"
        cp "$REPO_ROOT/.claude/references/logging/"*.md "$SCRIPT_DIR/references/logging/" 2>/dev/null || true
        echo "  - Synced reference documents (including logging/)"
    else
        echo "  - Synced reference documents"
    fi
else
    echo "  - Skipping references (directory not found)"
fi

# Sync workflow documentation
echo "Syncing workflow documentation..."
if [ -f "$REPO_ROOT/.claude/subagent_breakdown.md" ]; then
    cp "$REPO_ROOT/.claude/subagent_breakdown.md" "$SCRIPT_DIR/"
    echo "  - Synced subagent_breakdown.md"
else
    echo "  - Skipping subagent_breakdown.md (file not found)"
fi

# Sync CLAUDE.md (with confirmation)
echo ""
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    read -p "Sync CLAUDE.md from repo root? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$REPO_ROOT/CLAUDE.md" "$SCRIPT_DIR/"
        echo "  - Synced CLAUDE.md"
    else
        echo "  - Skipping CLAUDE.md"
    fi
fi

echo ""
echo "=============================================="
echo "Sync complete!"
echo "=============================================="
echo ""
echo "Files synced to: $SCRIPT_DIR"
echo ""
echo "You can now review and commit changes with:"
echo "  git diff $SCRIPT_DIR"
echo "  git add $SCRIPT_DIR && git commit -m 'Update claude agents'"
echo ""
