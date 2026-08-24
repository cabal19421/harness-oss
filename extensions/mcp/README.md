# mcp/ — drop-in MCP servers

Each `*.json` here defines one or more [Model Context Protocol](https://modelcontextprotocol.io)
servers. The loader merges them all into a single `{"mcpServers": {...}}` config
(`harness.extensions.merged_mcp_config()`, or `harness extensions --mcp-config`)
that an MCP-capable agent runtime can consume.

Two accepted shapes — the standard multi-server map:

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path/to/dir"]
    }
  }
}
```

…or a single bare server object (`name` names it; `command` **or** `url` is
required):

```json
{ "name": "git", "command": "uvx", "args": ["mcp-server-git"] }
```

## Fields

The merge is **lossless**: whatever you write in a server object comes back out
of `merged_mcp_config()` unchanged, so anything the runtime understands works
here even if harness has no field for it —

| field     | notes |
|-----------|-------|
| `type`    | `stdio` (default), `http`, `sse` (deprecated upstream), `ws`. `streamable-http` is normalized to `http`. Harness's older flat `"transport"` key is still read, but `type` is what gets written. |
| `command` | required for `stdio` servers (omit it entirely for remote ones — an empty string is the shape the runtime rejects) |
| `args`    | optional argv list |
| `env`     | how nearly every real server receives its API key — passed through untouched |
| `url`     | remote (`http`/`sse`/`ws`) servers; `headers` alongside it |
| anything else | `timeout`, `headers`, … — passed through untouched |

A remote server:

```json
{ "mcpServers": { "docs": {
  "type": "http",
  "url": "https://mcp.example.com/v1",
  "headers": { "Authorization": "Bearer ${DOCS_TOKEN}" } } } }
```

Use **absolute paths** in server arguments: a relative path like `"."` resolves
against the runtime's working directory, not against this folder.

A server that is malformed (neither `command` nor `url`), that collides with a
name another drop-in already defined, or that uses a name the runtime reserves
for a built-in (`workspace`, `computer-use`, …) is reported
by `harness extensions` instead of silently winning or losing the merge.

The harness **declares/merges** these configs; launching the servers is the
agent runtime's job. See [example-filesystem.json](example-filesystem.json).
