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
    mkdir -p "$SCRIPT_DIR/agents"
    for agent in "$REPO_ROOT/.claude/agents/"*.md; do
        if [ -f "$agent" ]; then
            cp "$agent" "$SCRIPT_DIR/agents/"
            echo "  - $(basename "$agent")"
        fi
    done
else
    echo "  - Skipping agents (directory not found)"
fi

# Sync scaffolder scripts
echo "Syncing scaffolder scripts..."
if [ -d "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder" ]; then
    mkdir -p "$SCRIPT_DIR/scripts/ttnn-operation-scaffolder"
    cp -r "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder/"* "$SCRIPT_DIR/scripts/ttnn-operation-scaffolder/"
    echo "  - Synced scaffolder scripts"
else
    echo "  - Skipping scaffolder scripts (directory not found)"
fi

# Sync logging scripts
echo "Syncing logging scripts..."
if [ -d "$REPO_ROOT/.claude/scripts/logging" ]; then
    mkdir -p "$SCRIPT_DIR/scripts/logging"
    cp -r "$REPO_ROOT/.claude/scripts/logging/"* "$SCRIPT_DIR/scripts/logging/"
    echo "  - Synced logging scripts"
else
    echo "  - Skipping logging scripts (directory not found)"
fi

# Sync reference documents
echo "Syncing reference documents..."
if [ -d "$REPO_ROOT/.claude/references" ]; then
    mkdir -p "$SCRIPT_DIR/references"
    # Sync top-level references
    for ref in "$REPO_ROOT/.claude/references/"*.md; do
        if [ -f "$ref" ]; then
            cp "$ref" "$SCRIPT_DIR/references/"
            echo "  - $(basename "$ref")"
        fi
    done
    # Sync logging subdirectory
    if [ -d "$REPO_ROOT/.claude/references/logging" ]; then
        mkdir -p "$SCRIPT_DIR/references/logging"
        for ref in "$REPO_ROOT/.claude/references/logging/"*.md; do
            if [ -f "$ref" ]; then
                cp "$ref" "$SCRIPT_DIR/references/logging/"
                echo "  - logging/$(basename "$ref")"
            fi
        done
    fi
else
    echo "  - Skipping references (directory not found)"
fi

# Sync workflow documentation
echo "Syncing workflow documentation..."
if [ -f "$REPO_ROOT/.claude/subagent_breakdown.md" ]; then
    cp "$REPO_ROOT/.claude/subagent_breakdown.md" "$SCRIPT_DIR/"
    echo "  - subagent_breakdown.md"
else
    echo "  - Skipping subagent_breakdown.md (file not found)"
fi

# Sync skills
echo "Syncing skills..."
if [ -d "$REPO_ROOT/.claude/skills" ]; then
    mkdir -p "$SCRIPT_DIR/skills"
    for skill_dir in "$REPO_ROOT/.claude/skills/"*/; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            mkdir -p "$SCRIPT_DIR/skills/$skill_name"
            cp -r "$skill_dir"* "$SCRIPT_DIR/skills/$skill_name/"
            echo "  - $skill_name"
        fi
    done
else
    echo "  - Skipping skills (directory not found)"
fi

# Sync CLAUDE.md (with confirmation)
echo ""
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    read -p "Sync CLAUDE.md from repo root? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$REPO_ROOT/CLAUDE.md" "$SCRIPT_DIR/"
        echo "  - CLAUDE.md"
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
