# image-mcp

An MCP (Model Context Protocol) server in Rust that generates images with
OpenAI's **gpt-image-2** model.

Built with:

- [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) — the official Rust MCP SDK (stdio transport)
- [`async-openai`](https://github.com/64bit/async-openai) — Rust bindings for the OpenAI API

## Tool

### `generate_image`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string (required) | — | Text description of the desired image (up to 32k chars) |
| `size` | string | `auto` | `WIDTHxHEIGHT`, e.g. `1024x1024`, `1536x1024`, `2048x2048`. gpt-image-2 accepts any resolution with edges that are multiples of 16, longest edge ≤ 3840px, aspect ratio ≤ 3:1 |
| `quality` | string | `auto` | `low`, `medium`, `high` or `auto` |
| `output_format` | string | `png` | `png`, `jpeg` or `webp` |
| `n` | integer | `1` | Number of images (1–10) |
| `save_dir` | string | — | If set, also writes the image(s) to this directory and returns the file paths |

Images are returned as base64-encoded MCP image content blocks.

## Build

```sh
cargo build --release
```

## Configuration

Requires the `OPENAI_API_KEY` environment variable. Note that gpt-image-2 may
require [API organization verification](https://help.openai.com/en/articles/10910291-api-organization-verification)
on your OpenAI account.

### Claude Code

```sh
claude mcp add image-gen -e OPENAI_API_KEY=sk-... -- /path/to/image-mcp/target/release/image-mcp
```

### Generic MCP client (`mcpServers` JSON)

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "/path/to/image-mcp/target/release/image-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Logging

Logs go to stderr (stdout is reserved for the MCP protocol). Control verbosity
with `RUST_LOG`, e.g. `RUST_LOG=debug`.
