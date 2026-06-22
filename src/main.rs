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
    /// Model to use. "gpt-image-2" (default, OpenAI) or a Gemini image model:
    /// "nano-banana-2" (Gemini 3.1 Flash Image), "nano-banana-pro" (Gemini 3 Pro
    /// Image), or any "gemini-*" model id. OpenAI models need OPENAI_API_KEY;
    /// Gemini models need GEMINI_API_KEY.
    pub model: Option<String>,
    /// Image size. For gpt-image-2: `WIDTHxHEIGHT` (e.g. "1024x1024") or "auto"
    /// (default) — edges multiple of 16, longest edge <= 3840px, aspect <= 3:1.
    /// For Gemini models: an aspect ratio like "16:9", "1:1" or "9:16" (anything
    /// without a colon is ignored).
    pub size: Option<String>,
    /// Rendering quality: "low", "medium", "high" or "auto" (default).
    /// gpt-image-2 only — ignored by Gemini models.
    pub quality: Option<String>,
    /// Output format: "png" (default), "jpeg" or "webp". gpt-image-2 only —
    /// Gemini models return PNG.
    pub output_format: Option<String>,
    /// Number of images to generate, 1-10. Defaults to 1. For Gemini this issues
    /// one request per image (the API returns a single image per call).
    pub n: Option<u8>,
    /// Optional directory to also save the generated image(s) to disk.
    /// Files are named image-<timestamp>-<index>.<ext>.
    pub save_dir: Option<String>,
}

// Gemini generateContent response. Raw REST uses snake_case (inline_data/mime_type);
// the JS SDK examples use camelCase — accept both via serde aliases.
#[derive(Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize)]
struct GeminiPart {
    // Absent on text-only parts.
    #[serde(default, alias = "inlineData")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Deserialize)]
struct GeminiInlineData {
    #[serde(alias = "mimeType")]
    mime_type: String,
    data: String,
}

fn ext_for_mime(mime: &str) -> &'static str {
    match mime {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        _ => "png",
    }
}

#[derive(Clone)]
pub struct ImageServer {
    client: Client<OpenAIConfig>,
    http: reqwest::Client,
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
            http: reqwest::Client::new(),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "generate_image",
        description = "Generate one or more images from a text prompt. Defaults to OpenAI's gpt-image-2; set `model` to a Gemini image model (e.g. \"nano-banana-2\") to use Google Gemini instead. Returns the image(s) as base64-encoded content, and optionally saves them to disk."
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
    ) -> Result<CallToolResult, McpError> {
        let model = args.model.as_deref().unwrap_or("gpt-image-2");
        let gemini_model = match model {
            "nano-banana-2" => Some("gemini-3.1-flash-image".to_string()),
            "nano-banana-pro" => Some("gemini-3-pro-image".to_string()),
            m if m.starts_with("gemini") => Some(m.to_string()),
            _ => None,
        };

        // Each backend returns (base64, mime_type, file_ext) tuples; saving is shared.
        let images = match &gemini_model {
            Some(gm) => self.generate_gemini(&args, gm).await?,
            None => self.generate_openai(&args).await?,
        };

        let mut contents = Vec::new();
        let mut saved_paths = Vec::new();

        for (index, (b64, mime_type, ext)) in images.iter().enumerate() {
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

            contents.push(Content::image(b64.as_str(), mime_type.as_str()));
        }

        if !saved_paths.is_empty() {
            contents.insert(
                0,
                Content::text(format!("Saved image(s) to: {}", saved_paths.join(", "))),
            );
        }

        Ok(CallToolResult::success(contents))
    }

    async fn generate_openai(
        &self,
        args: &GenerateImageArgs,
    ) -> Result<Vec<(String, String, String)>, McpError> {
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

        let mut out = Vec::new();
        for image in response.data.iter() {
            match image.as_ref() {
                Image::B64Json { b64_json, .. } => {
                    out.push((b64_json.to_string(), mime_type.to_string(), ext.to_string()))
                }
                Image::Url { url, .. } => {
                    return Err(McpError::internal_error(
                        format!("unexpected URL response from gpt-image-2: {url}"),
                        None,
                    ));
                }
            }
        }
        Ok(out)
    }

    async fn generate_gemini(
        &self,
        args: &GenerateImageArgs,
        model: &str,
    ) -> Result<Vec<(String, String, String)>, McpError> {
        let key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
        if key.is_empty() {
            return Err(McpError::internal_error(
                "GEMINI_API_KEY environment variable is not set",
                None,
            ));
        }

        // `size` is reused as the aspect ratio for Gemini (e.g. "16:9"); quality
        // and output_format don't map to the Gemini API.
        let mut generation_config = serde_json::json!({ "responseModalities": ["IMAGE"] });
        if let Some(aspect) = args.size.as_deref().filter(|s| s.contains(':')) {
            generation_config["responseFormat"] =
                serde_json::json!({ "image": { "aspectRatio": aspect } });
        }
        let body = serde_json::json!({
            "contents": [{ "parts": [{ "text": args.prompt.as_str() }] }],
            "generationConfig": generation_config,
        });

        // The Gemini API returns a single image per call (candidateCount > 1 is
        // rejected), so issue one request per requested image.
        let url =
            format!("https://generativelanguage.googleapis.com/v1/models/{model}:generateContent");
        let n = args.n.unwrap_or(1).clamp(1, 10);

        tracing::info!(n, model, "requesting image generation from Gemini");

        let mut out = Vec::new();
        for _ in 0..n {
            let response = self
                .http
                .post(&url)
                .header("x-goog-api-key", &key)
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    McpError::internal_error(format!("Gemini API request failed: {e}"), None)
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let detail = response.text().await.unwrap_or_default();
                return Err(McpError::internal_error(
                    format!("Gemini API error {status}: {detail}"),
                    None,
                ));
            }

            let parsed: GeminiResponse = response.json().await.map_err(|e| {
                McpError::internal_error(format!("failed to parse Gemini response: {e}"), None)
            })?;

            for candidate in parsed.candidates {
                for part in candidate.content.parts {
                    if let Some(inline) = part.inline_data {
                        let ext = ext_for_mime(&inline.mime_type).to_string();
                        out.push((inline.data, inline.mime_type, ext));
                    }
                }
            }
        }

        if out.is_empty() {
            return Err(McpError::internal_error(
                "Gemini returned no image data",
                None,
            ));
        }
        Ok(out)
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
                "Image generation server. Use the generate_image tool with a text prompt; \
                 optionally control size, quality, output format, image count, and save \
                 results to disk. Defaults to OpenAI's gpt-image-2 (needs OPENAI_API_KEY); \
                 set `model` to a Gemini image model such as \"nano-banana-2\" or \
                 \"nano-banana-pro\" to use Google Gemini instead (needs GEMINI_API_KEY).",
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

    tracing::info!("starting image-mcp server (gpt-image-2 + gemini)");

    let service = ImageServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemini_response_camel_and_snake() {
        // camelCase (JS SDK form), with a leading text part to skip.
        let camel = r#"{"candidates":[{"content":{"parts":[
            {"text":"here you go"},
            {"inlineData":{"mimeType":"image/png","data":"AAAA"}}
        ]}}]}"#;
        let r: GeminiResponse = serde_json::from_str(camel).unwrap();
        let img = r.candidates[0]
            .content
            .parts
            .iter()
            .find_map(|p| p.inline_data.as_ref())
            .unwrap();
        assert_eq!(img.mime_type, "image/png");
        assert_eq!(img.data, "AAAA");
        assert_eq!(ext_for_mime(&img.mime_type), "png");

        // snake_case (raw REST form).
        let snake = r#"{"candidates":[{"content":{"parts":[
            {"inline_data":{"mime_type":"image/jpeg","data":"BBBB"}}
        ]}}]}"#;
        let r: GeminiResponse = serde_json::from_str(snake).unwrap();
        let img = r.candidates[0]
            .content
            .parts
            .iter()
            .find_map(|p| p.inline_data.as_ref())
            .unwrap();
        assert_eq!(img.data, "BBBB");
        assert_eq!(ext_for_mime(&img.mime_type), "jpg");
    }
}
