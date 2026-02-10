#!/bin/bash
# Activate Claude TTNN Agents
# This script copies agents to their proper locations and configures Claude Code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Activating Claude TTNN Agents..."
echo "Repository root: $REPO_ROOT"

# Create .claude directories if they don't exist
mkdir -p "$REPO_ROOT/.claude/agents"
mkdir -p "$REPO_ROOT/.claude/scripts"
mkdir -p "$REPO_ROOT/.claude/references"
mkdir -p "$REPO_ROOT/.claude/logs"

# Copy agent definitions
echo "Installing agent definitions..."
for agent in "$SCRIPT_DIR/agents/"*.md; do
    if [ -f "$agent" ]; then
        cp "$agent" "$REPO_ROOT/.claude/agents/"
        echo "  - $(basename "$agent")"
    fi
done

# Copy scaffolder scripts (used by ttnn-operation-scaffolder agent)
echo "Installing scaffolder scripts..."
if [ -d "$SCRIPT_DIR/scripts/ttnn-operation-scaffolder" ]; then
    cp -r "$SCRIPT_DIR/scripts/ttnn-operation-scaffolder" "$REPO_ROOT/.claude/scripts/"
    chmod +x "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder/"*.py 2>/dev/null || true
    chmod +x "$REPO_ROOT/.claude/scripts/ttnn-operation-scaffolder/"*.sh 2>/dev/null || true
    echo "  - ttnn-operation-scaffolder/"
fi

# Copy logging scripts (used by agents for structured logging)
echo "Installing logging scripts..."
if [ -d "$SCRIPT_DIR/scripts/logging" ]; then
    cp -r "$SCRIPT_DIR/scripts/logging" "$REPO_ROOT/.claude/scripts/"
    chmod +x "$REPO_ROOT/.claude/scripts/logging/"*.sh 2>/dev/null || true
    echo "  - logging/"
fi

# Copy standalone utility scripts
echo "Installing utility scripts..."
for script in "$SCRIPT_DIR/scripts/"*.sh; do
    if [ -f "$script" ]; then
        cp "$script" "$REPO_ROOT/.claude/scripts/"
        chmod +x "$REPO_ROOT/.claude/scripts/$(basename "$script")"
        echo "  - $(basename "$script")"
    fi
done

# Copy reference documents
echo "Installing reference documents..."
mkdir -p "$REPO_ROOT/.claude/references"
for ref in "$SCRIPT_DIR/references/"*.md; do
    if [ -f "$ref" ]; then
        cp "$ref" "$REPO_ROOT/.claude/references/"
        echo "  - $(basename "$ref")"
    fi
done

# Copy reference subdirectories
if [ -d "$SCRIPT_DIR/references/logging" ]; then
    mkdir -p "$REPO_ROOT/.claude/references/logging"
    for ref in "$SCRIPT_DIR/references/logging/"*.md; do
        if [ -f "$ref" ]; then
            cp "$ref" "$REPO_ROOT/.claude/references/logging/"
            echo "  - logging/$(basename "$ref")"
        fi
    done
fi

if [ -d "$SCRIPT_DIR/references/generic_op_template" ]; then
    cp -r "$SCRIPT_DIR/references/generic_op_template" "$REPO_ROOT/.claude/references/"
    echo "  - generic_op_template/"
fi

# Copy workflow documentation
echo "Installing workflow documentation..."
if [ -f "$SCRIPT_DIR/subagent_breakdown.md" ]; then
    cp "$SCRIPT_DIR/subagent_breakdown.md" "$REPO_ROOT/.claude/"
    echo "  - subagent_breakdown.md"
fi

# Copy skills (user-invocable slash commands)
echo "Installing skills..."
if [ -d "$SCRIPT_DIR/skills" ]; then
    mkdir -p "$REPO_ROOT/.claude/skills"
    for skill_dir in "$SCRIPT_DIR/skills/"*/; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            cp -r "$skill_dir" "$REPO_ROOT/.claude/skills/"
            echo "  - $skill_name"
        fi
    done
fi

# Copy CLAUDE.md to repo root
echo "Installing CLAUDE.md..."
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    echo "CLAUDE.md already exists in $REPO_ROOT"
    # Always create backup before asking
    cp "$REPO_ROOT/CLAUDE.md" "$REPO_ROOT/CLAUDE.md.old"
    echo "Created backup: $REPO_ROOT/CLAUDE.md.old"

    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$SCRIPT_DIR/CLAUDE.md" "$REPO_ROOT/"
        echo "CLAUDE.md overwritten (backup saved as CLAUDE.md.old)"
    else
        echo "Skipping CLAUDE.md installation (backup remains at CLAUDE.md.old)"
    fi
else
    cp "$SCRIPT_DIR/CLAUDE.md" "$REPO_ROOT/"
    echo "CLAUDE.md installed"
fi

# Configure Claude Code settings
echo "Configuring Claude Code settings..."

# Use Python to safely configure all JSON files
export REPO_ROOT
python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

repo_root = os.environ.get('REPO_ROOT', os.getcwd())
home_dir = str(Path.home())

# 1. Configure ~/.claude.json (MCP server + project context)
claude_json_path = os.path.join(home_dir, '.claude.json')
claude_config = {}

if os.path.exists(claude_json_path):
    try:
        with open(claude_json_path, 'r') as f:
            claude_config = json.load(f)
    except json.JSONDecodeError:
        claude_config = {}

# Add MCP server
if 'mcpServers' not in claude_config:
    claude_config['mcpServers'] = {}

if 'deepwiki' not in claude_config['mcpServers']:
    claude_config['mcpServers']['deepwiki'] = {
        'type': 'http',
        'url': 'https://mcp.deepwiki.com/mcp'
    }
    print("Added DeepWiki MCP server to ~/.claude.json")
else:
    print("DeepWiki MCP server already configured")

# Add project context
if 'projects' not in claude_config:
    claude_config['projects'] = {}

if repo_root not in claude_config['projects']:
    claude_config['projects'][repo_root] = {}

if 'deepWikiContext' not in claude_config['projects'][repo_root]:
    claude_config['projects'][repo_root]['deepWikiContext'] = 'tenstorrent/tt-metal'
    print(f"Added DeepWiki context for {repo_root}")
else:
    print("DeepWiki project context already configured")

# Ensure allowedTools exists
if 'allowedTools' not in claude_config['projects'][repo_root]:
    claude_config['projects'][repo_root]['allowedTools'] = []

with open(claude_json_path, 'w') as f:
    json.dump(claude_config, f, indent=2)

# 2. Configure .claude/settings.local.json (permission)
settings_dir = os.path.join(repo_root, '.claude')
settings_path = os.path.join(settings_dir, 'settings.local.json')

os.makedirs(settings_dir, exist_ok=True)

settings = {}
if os.path.exists(settings_path):
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except json.JSONDecodeError:
        settings = {}

if 'permissions' not in settings:
    settings['permissions'] = {'allow': [], 'deny': [], 'ask': []}
if 'allow' not in settings['permissions']:
    settings['permissions']['allow'] = []

if 'mcp__deepwiki__ask_question' not in settings['permissions']['allow']:
    settings['permissions']['allow'].append('mcp__deepwiki__ask_question')
    print("Added DeepWiki permission to .claude/settings.local.json")
else:
    print("DeepWiki permission already configured")

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2)

print("\nConfiguration complete!")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Claude TTNN Agents activated successfully!"
echo "=============================================="
echo ""
echo "Installed to: $REPO_ROOT/.claude/"
echo ""
echo "DeepWiki MCP configured in:"
echo "  - ~/.claude.json (MCP server + project context)"
echo "  - $REPO_ROOT/.claude/settings.local.json (permission)"
echo ""
echo "To use the agents, restart Claude Code in the repository root."
echo "The agents will be available via the Task tool."
echo ""
echo "See .claude/subagent_breakdown.md for detailed workflow documentation."
echo ""
