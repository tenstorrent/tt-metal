#!/bin/bash
# Activate Claude TTNN Agents
# This script copies agents to their proper locations and configures Claude Code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Activating Claude TTNN Agents..."
echo "Repository root: $REPO_ROOT"

# Create .claude/agents directory if it doesn't exist
mkdir -p "$REPO_ROOT/.claude/agents"

# Copy agent definitions
echo "Installing agent definitions..."
cp "$SCRIPT_DIR/agents/ttnn-operation-analyzer.md" "$REPO_ROOT/.claude/agents/"
cp "$SCRIPT_DIR/agents/ttnn-operation-planner.md" "$REPO_ROOT/.claude/agents/"
cp "$SCRIPT_DIR/agents/ttnn-operation-scaffolder.md" "$REPO_ROOT/.claude/agents/"
cp "$SCRIPT_DIR/agents/ttnn-factory-builder.md" "$REPO_ROOT/.claude/agents/"

# Copy workflow documentation
echo "Installing workflow documentation..."
cp "$SCRIPT_DIR/subagent_breakdown.md" "$REPO_ROOT/.claude/"

# Copy CLAUDE.md to repo root
echo "Installing CLAUDE.md..."
cp "$SCRIPT_DIR/CLAUDE.md" "$REPO_ROOT/"

# Configure Claude Code settings
echo "Configuring Claude Code settings..."

# Use Python to safely configure all JSON files
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
echo "Installed files:"
echo "  - $REPO_ROOT/CLAUDE.md"
echo "  - $REPO_ROOT/.claude/subagent_breakdown.md"
echo "  - $REPO_ROOT/.claude/agents/ttnn-operation-analyzer.md"
echo "  - $REPO_ROOT/.claude/agents/ttnn-operation-planner.md"
echo "  - $REPO_ROOT/.claude/agents/ttnn-operation-scaffolder.md"
echo "  - $REPO_ROOT/.claude/agents/ttnn-factory-builder.md"
echo ""
echo "DeepWiki MCP configured in:"
echo "  - ~/.claude.json (MCP server + project context)"
echo "  - $REPO_ROOT/.claude/settings.local.json (permission)"
echo ""
echo "To use the agents, restart Claude Code in the repository root."
echo "The agents will be available via the Task tool."
echo ""
echo "Workflow:"
echo "  1. ttnn-operation-analyzer  -> Analyze reference operation"
echo "  2. ttnn-operation-planner   -> Design new operation (USER REVIEW)"
echo "  3. ttnn-operation-scaffolder -> Build Stages 1-3"
echo "  4. ttnn-factory-builder     -> Build Stages 4-6"
echo ""
echo "See .claude/subagent_breakdown.md for detailed workflow documentation."
