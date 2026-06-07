use std::path::PathBuf;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::images::{
        CreateImageRequestArgs, Image, ImageModel, ImageOutputFormat, ImageQuality, ImageSize,
    },
};
use base64::Engine;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GenerateImageArgs {
    /// Text description of the desired image. Up to 32,000 characters.
    pub prompt: String,
    /// Image size as `WIDTHxHEIGHT` (e.g. "1024x1024", "1536x1024", "2048x2048")
    /// or "auto" (default). gpt-image-2 accepts any resolution where both edges
    /// are multiples of 16, the longest edge is <= 3840px and aspect ratio <= 3:1.
    pub size: Option<String>,
    /// Rendering quality: "low", "medium", "high" or "auto" (default).
    pub quality: Option<String>,
    /// Output format: "png" (default), "jpeg" or "webp".
    pub output_format: Option<String>,
    /// Number of images to generate, 1-10. Defaults to 1.
    pub n: Option<u8>,
    /// Optional directory to also save the generated image(s) to disk.
    /// Files are named image-<timestamp>-<index>.<ext>.
    pub save_dir: Option<String>,
}

#[derive(Clone)]
pub struct ImageServer {
    client: Client<OpenAIConfig>,
    tool_router: ToolRouter<Self>,
}

impl Default for ImageServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl ImageServer {
    pub fn new() -> Self {
        Self {
            // Reads OPENAI_API_KEY from the environment.
            client: Client::new(),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "generate_image",
        description = "Generate one or more images from a text prompt using OpenAI's gpt-image-2 model. Returns the image(s) as base64-encoded content, and optionally saves them to disk."
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
    ) -> Result<CallToolResult, McpError> {
        if std::env::var("OPENAI_API_KEY").map_or(true, |k| k.is_empty()) {
            return Err(McpError::internal_error(
                "OPENAI_API_KEY environment variable is not set",
                None,
            ));
        }

        let size = match args.size.as_deref() {
            None | Some("auto") => ImageSize::Auto,
            Some(s) => ImageSize::Other(s.to_string()),
        };

        let quality = match args.quality.as_deref() {
            None | Some("auto") => ImageQuality::Auto,
            Some("low") => ImageQuality::Low,
            Some("medium") => ImageQuality::Medium,
            Some("high") => ImageQuality::High,
            Some(other) => {
                return Err(McpError::invalid_params(
                    format!("invalid quality '{other}': expected low, medium, high or auto"),
                    None,
                ));
            }
        };

        let (output_format, mime_type, ext) = match args.output_format.as_deref() {
            None | Some("png") => (ImageOutputFormat::Png, "image/png", "png"),
            Some("jpeg") | Some("jpg") => (ImageOutputFormat::Jpeg, "image/jpeg", "jpg"),
            Some("webp") => (ImageOutputFormat::Webp, "image/webp", "webp"),
            Some(other) => {
                return Err(McpError::invalid_params(
                    format!("invalid output_format '{other}': expected png, jpeg or webp"),
                    None,
                ));
            }
        };

        let n = args.n.unwrap_or(1).clamp(1, 10);

        let request = CreateImageRequestArgs::default()
            .prompt(&args.prompt)
            .model(ImageModel::GptImage2)
            .n(n)
            .size(size)
            .quality(quality)
            .output_format(output_format)
            .build()
            .map_err(|e| McpError::internal_error(format!("failed to build request: {e}"), None))?;

        tracing::info!(n, "requesting image generation from gpt-image-2");

        let response = self
            .client
            .images()
            .generate(request)
            .await
            .map_err(|e| McpError::internal_error(format!("OpenAI API error: {e}"), None))?;

        let mut contents = Vec::new();
        let mut saved_paths = Vec::new();

        for (index, image) in response.data.iter().enumerate() {
            let b64 = match image.as_ref() {
                Image::B64Json { b64_json, .. } => b64_json.clone(),
                Image::Url { url, .. } => {
                    return Err(McpError::internal_error(
                        format!("unexpected URL response from gpt-image-2: {url}"),
                        None,
                    ));
                }
            };

            if let Some(dir) = &args.save_dir {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64.as_bytes())
                    .map_err(|e| {
                        McpError::internal_error(format!("failed to decode image data: {e}"), None)
                    })?;
                let dir = PathBuf::from(dir);
                tokio::fs::create_dir_all(&dir).await.map_err(|e| {
                    McpError::internal_error(format!("failed to create directory: {e}"), None)
                })?;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let path = dir.join(format!("image-{timestamp}-{index}.{ext}"));
                tokio::fs::write(&path, &bytes).await.map_err(|e| {
                    McpError::internal_error(
                        format!("failed to write {}: {e}", path.display()),
                        None,
                    )
                })?;
                saved_paths.push(path.display().to_string());
            }

            contents.push(Content::image(b64.as_str(), mime_type));
        }

        if !saved_paths.is_empty() {
            contents.insert(
                0,
                Content::text(format!("Saved image(s) to: {}", saved_paths.join(", "))),
            );
        }

        Ok(CallToolResult::success(contents))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ImageServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Image generation server backed by OpenAI's gpt-image-2 model. \
                 Use the generate_image tool with a text prompt; optionally control \
                 size, quality, output format, image count, and save results to disk. \
                 Requires the OPENAI_API_KEY environment variable.",
            )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Log to stderr only: stdout is reserved for the MCP stdio transport.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tracing::info!("starting image-mcp server (gpt-image-2)");

    let service = ImageServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
