# image-mcp

An MCP (Model Context Protocol) server in Rust that generates images with
OpenAI's **gpt-image-2** or Google's **Gemini** (Nano Banana) image models.

Built with:

- [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) — the official Rust MCP SDK (stdio transport)
- [`async-openai`](https://github.com/64bit/async-openai) — Rust bindings for the OpenAI API
- [`reqwest`](https://github.com/seanmonstar/reqwest) — for the Gemini `generateContent` REST call

## Tool

### `generate_image`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string (required) | — | Text description of the desired image (up to 32k chars) |
| `model` | string | `gpt-image-2` | `gpt-image-2` (OpenAI), `nano-banana-2` (Gemini 3.1 Flash Image), `nano-banana-pro` (Gemini 3 Pro Image), or any `gemini-*` model id |
| `size` | string | `auto` | gpt-image-2: `WIDTHxHEIGHT`, e.g. `1024x1024`, `2048x2048` (edges multiple of 16, longest edge ≤ 3840px, aspect ≤ 3:1). Gemini: an aspect ratio like `16:9`, `1:1`, `9:16` |
| `quality` | string | `auto` | `low`, `medium`, `high` or `auto`. gpt-image-2 only |
| `output_format` | string | `png` | `png`, `jpeg` or `webp`. gpt-image-2 only — Gemini returns PNG |
| `n` | integer | `1` | Number of images (1–10). For Gemini this issues one request per image |
| `save_dir` | string | — | If set, also writes the image(s) to this directory and returns the file paths |

Images are returned as base64-encoded MCP image content blocks.

The model is selected per call via the `model` parameter — both backends are
served by the same tool.

## Build

```sh
cargo build --release
```

## Configuration

Set the API key for whichever backend(s) you use:

- `OPENAI_API_KEY` — for `gpt-image-2` (the default). Note gpt-image-2 may
  require [API organization verification](https://help.openai.com/en/articles/10910291-api-organization-verification)
  on your OpenAI account.
- `GEMINI_API_KEY` — for the Gemini (`nano-banana-*` / `gemini-*`) models.

You only need the key for the backend you actually call.

### Claude Code

```sh
claude mcp add image-gen \
  -e OPENAI_API_KEY=sk-... \
  -e GEMINI_API_KEY=... \
  -- /path/to/image-mcp/target/release/image-mcp
```

### Generic MCP client (`mcpServers` JSON)

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "/path/to/image-mcp/target/release/image-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "..."
      }
    }
  }
}
```

## Logging

Logs go to stderr (stdout is reserved for the MCP protocol). Control verbosity
with `RUST_LOG`, e.g. `RUST_LOG=debug`.
