use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};

/// Run whisper.cpp binary on a WAV chunk and return the transcribed text.
pub fn transcribe_chunk_with_cli(
    whisper_cli: &Path,
    model_path: &Path,
    wav_path: &Path,
    language: Option<&str>,
    out_prefix: &Path,
) -> Result<String> {
    let cli = whisper_cli
        .to_str()
        .ok_or_else(|| anyhow!("invalid whisper_cli path"))?;
    let model = model_path
        .to_str()
        .ok_or_else(|| anyhow!("invalid model path"))?;
    let input = wav_path
        .to_str()
        .ok_or_else(|| anyhow!("invalid wav path"))?;
    let prefix = out_prefix
        .to_str()
        .ok_or_else(|| anyhow!("invalid out prefix path"))?;

    let mut args = vec!["-m", model, "-f", input, "-otxt", "-of", prefix];
    if let Some(lang) = language {
        args.push("-l");
        args.push(lang);
    }

    let status = Command::new(cli)
        .args(&args)
        .status()
        .with_context(|| format!("failed to spawn whisper CLI at {}", cli))?;
    if !status.success() {
        return Err(anyhow!("whisper CLI failed with status {}", status));
    }

    let txt_path = out_prefix.with_extension("txt");
    if !txt_path.exists() {
        return Err(anyhow!(
            "Whisper output not found at {}. Check whisper.cpp version compatibility.",
            txt_path.display()
        ));
    }
    let text = fs::read_to_string(&txt_path)
        .with_context(|| format!("failed to read {}", txt_path.display()))?;
    Ok(text)
}
