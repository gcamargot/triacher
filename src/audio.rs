use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Result};

/// Extracts mono 16kHz WAV audio suitable for Whisper
pub fn extract_audio_ffmpeg(input_video: &Path, output_wav: &Path) -> Result<()> {
    let input = input_video
        .to_str()
        .ok_or_else(|| anyhow!("Invalid input video path"))?;
    let output = output_wav
        .to_str()
        .ok_or_else(|| anyhow!("Invalid output audio path"))?;

    let status = Command::new("ffmpeg")
        .args([
            "-y", // overwrite
            "-i",
            input,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output,
        ])
        .status()?;

    if !status.success() {
        return Err(anyhow!("ffmpeg failed to extract audio"));
    }

    Ok(())
}

