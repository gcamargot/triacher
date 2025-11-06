use std::env;

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

/// Calls a local Ollama server to summarize the transcript into Markdown.
pub async fn summarize_markdown(
    client: &Client,
    model: &str,
    transcript: &str,
    host_override: Option<&str>,
) -> Result<String> {
    let host_raw = host_override
        .map(|s| s.to_string())
        .or_else(|| env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
    let host = normalize_ollama_host(&host_raw);
    let url = format!("{}/api/generate", host);

    let system_instructions = "You are an expert note-taker for university lectures. Produce a concise, well-structured Markdown summary with:
- Title
- TL;DR (3-7 bullets)
- Key Takeaways
- Outline of Topics (use timestamps if present in the text)
- Action Items / Follow-ups
- Important Terms & Definitions
- A final Spanish section for next class preparation:
  * If the professor mentions a specific date or day for the next class, add a heading 'Para DDMM' (replace DDMM with the numeric day and month, e.g., 1503 for 15/03) and list what to preparar/estudiar.
  * Otherwise, add 'Para la próxima clase' with concrete items to preparar/estudiar.
Keep it accurate, faithful, and free of fabrication. Return only Markdown.";

    let prompt = format!(
        "{}\n\nTranscript:\n---\n{}\n---",
        system_instructions, transcript
    );

    let body = GenerateRequest {
        model,
        prompt,
        stream: false,
        options: Some(serde_json::json!({
            "temperature": 0.2,
        })),
    };

    let res = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to reach Ollama: {}", e))?;

    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "<no body>".to_string());
        return Err(anyhow!(
            "Ollama returned HTTP {} for model '{}': {}",
            status, model, text
        ));
    }

    let parsed: GenerateResponse = res.json().await?;
    Ok(parsed.response)
}

fn normalize_ollama_host(raw: &str) -> String {
    let mut h = raw.trim().to_string();
    if h.is_empty() {
        return "http://127.0.0.1:11434".to_string();
    }
    if !(h.starts_with("http://") || h.starts_with("https://")) {
        h = format!("http://{}", h);
    }
    h.trim_end_matches('/').to_string()
}
