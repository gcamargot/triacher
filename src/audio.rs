use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};

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

/// Duration of a WAV file in seconds.
pub fn wav_duration_secs(path: &Path) -> Result<f64> {
    let reader = hound::WavReader::open(path).context("failed to open wav for duration")?;
    let spec = reader.spec();
    let secs = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(secs)
}

/// Duration of a video file in seconds using ffprobe (part of ffmpeg).
pub fn video_duration_secs_ffprobe(path: &Path) -> Result<f64> {
    let input = path
        .to_str()
        .ok_or_else(|| anyhow!("invalid video path for ffprobe"))?;
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input,
        ])
        .output()
        .context("failed to run ffprobe")?;
    if !out.status.success() {
        return Err(anyhow!("ffprobe failed with status {}", out.status));
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let val: f64 = s.parse().context("failed to parse ffprobe duration")?;
    Ok(val)
}
