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

/// Optionally trim silence from a WAV using ffmpeg's silenceremove filter.
pub fn trim_silence_ffmpeg(input_wav: &Path, output_wav: &Path) -> Result<()> {
    let input = input_wav
        .to_str()
        .ok_or_else(|| anyhow!("Invalid input wav path"))?;
    let output = output_wav
        .to_str()
        .ok_or_else(|| anyhow!("Invalid output wav path"))?;

    // Parameters: start_periods=1, start_duration=0, start_threshold=-50dB
    // and same for end; also remove internal long silences.
    // A conservative filter to avoid clipping speech.
    let filter = "silenceremove=1:0:-50dB:1:0:-50dB";

    let status = Command::new("ffmpeg")
        .args(["-y", "-i", input, "-af", filter, output])
        .status()?;
    if !status.success() {
        return Err(anyhow!("ffmpeg failed to trim silence"));
    }
    Ok(())
}
