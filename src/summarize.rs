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
    prompt_override: Option<&str>,
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

    let instructions = prompt_override.unwrap_or(system_instructions);
    let prompt = format!("{}\n\nTranscript:\n---\n{}\n---", instructions, transcript);

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
            status,
            model,
            text
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // -------------------------------------------------------------------------
    // normalize_ollama_host tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_ollama_host(""), "http://127.0.0.1:11434");
    }

    #[test]
    fn test_normalize_whitespace_only() {
        assert_eq!(normalize_ollama_host("   "), "http://127.0.0.1:11434");
    }

    #[test]
    fn test_normalize_no_scheme() {
        assert_eq!(
            normalize_ollama_host("localhost:11434"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn test_normalize_with_http() {
        assert_eq!(
            normalize_ollama_host("http://localhost:11434"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn test_normalize_with_https() {
        assert_eq!(
            normalize_ollama_host("https://ollama.example.com"),
            "https://ollama.example.com"
        );
    }

    #[test]
    fn test_normalize_trailing_slash() {
        assert_eq!(
            normalize_ollama_host("http://localhost:11434/"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn test_normalize_multiple_trailing_slashes() {
        assert_eq!(
            normalize_ollama_host("http://localhost:11434///"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn test_normalize_ip_address() {
        assert_eq!(
            normalize_ollama_host("192.168.1.100:11434"),
            "http://192.168.1.100:11434"
        );
    }

    // -------------------------------------------------------------------------
    // summarize_markdown integration tests (with mock server)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_summarize_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "response": "# Summary\n\n- Point 1\n- Point 2"
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new();
        let result = summarize_markdown(
            &client,
            "llama3.1:8b",
            "Test transcript content",
            Some(&mock_server.uri()),
            None,
        )
        .await;

        assert!(result.is_ok());
        let summary = result.unwrap();
        assert!(summary.contains("# Summary"));
        assert!(summary.contains("Point 1"));
    }

    #[tokio::test]
    async fn test_summarize_with_custom_prompt() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "response": "Custom response"
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new();
        let result = summarize_markdown(
            &client,
            "llama3.1:8b",
            "Test transcript",
            Some(&mock_server.uri()),
            Some("Custom prompt for summary"),
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Custom response");
    }

    #[tokio::test]
    async fn test_summarize_http_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let client = Client::new();
        let result = summarize_markdown(
            &client,
            "llama3.1:8b",
            "Test transcript",
            Some(&mock_server.uri()),
            None,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("500"));
    }

    #[tokio::test]
    async fn test_summarize_model_not_found() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(404).set_body_string("model not found"))
            .mount(&mock_server)
            .await;

        let client = Client::new();
        let result = summarize_markdown(
            &client,
            "nonexistent:model",
            "Test transcript",
            Some(&mock_server.uri()),
            None,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("404"));
        assert!(err.to_string().contains("nonexistent:model"));
    }

    #[tokio::test]
    async fn test_summarize_connection_refused() {
        let client = Client::new();
        // Use a port that's definitely not listening
        let result = summarize_markdown(
            &client,
            "llama3.1:8b",
            "Test transcript",
            Some("http://127.0.0.1:59999"),
            None,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to reach Ollama"));
    }
}
