# Claude TTNN Agents

This package contains Claude Code agents for creating new TTNN operations.

## Quick Start

1. Run the activation script:
   ```bash
   ./ttnn/experimental/claude_ttnn_agents/activate_agents.sh
   ```

2. Start Claude Code in the repository root.

3. The agents will be available via the Task tool.

## Agents

| Agent | Purpose |
|-------|---------|
| ttnn-operation-analyzer | Analyze reference operation |
| ttnn-operation-planner | Design new operation spec |
| ttnn-operation-scaffolder | Build Stages 1-3 (API, validation, registration) |
| ttnn-factory-builder | Build Stages 4-6 (device op, factory, kernels) |

## Workflow

See `subagent_breakdown.md` for detailed workflow documentation.

## DeepWiki Integration

The activation script configures the DeepWiki MCP server for accessing
tt-metal documentation. It modifies:

- `~/.claude.json` - Adds MCP server and project context
- `.claude/settings.local.json` - Adds permission for the tool

DeepWiki provides context about:
- Hardware architecture (Tensix cores, NoC, etc.)
- Kernel development patterns
- API documentation
- Existing operation implementations

### Manual Setup (Alternative)
```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```
Then add `"deepWikiContext": "tenstorrent/tt-metal"` to your project in `~/.claude.json`.
