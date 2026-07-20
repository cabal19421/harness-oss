# mcp/ — drop-in MCP servers

Each `*.json` here defines one or more [Model Context Protocol](https://modelcontextprotocol.io)
servers. The loader merges them all into a single `{"mcpServers": {...}}` config
(`harness.extensions.merged_mcp_config()`) that an agent runtime (Claude Code,
etc.) can consume.

Two accepted shapes — the standard multi-server map:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    }
  }
}
```

…or a single bare server object:

```json
{ "name": "git", "command": "uvx", "args": ["mcp-server-git"] }
```

Fields: `command` (required), `args` (optional), `transport` (default `stdio`).
The harness **declares/merges** these configs; launching the servers is the
agent runtime's job. See [example-filesystem.json](example-filesystem.json).
