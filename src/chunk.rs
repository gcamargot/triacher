use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

/// Segments a WAV file into fixed-length chunks using ffmpeg's segment muxer.
/// Returns the list of chunk file paths in chronological order.
pub fn segment_wav_ffmpeg(input_wav: &Path, out_dir: &Path, segment_secs: u32) -> Result<Vec<PathBuf>> {
    if segment_secs == 0 {
        return Err(anyhow!("segment_secs must be > 0"));
    }

    fs::create_dir_all(out_dir).context("failed to create chunks dir")?;
    // Remove existing chunks
    for entry in fs::read_dir(out_dir)? {
        let p = entry?.path();
        if p.is_file() {
            let _ = fs::remove_file(p);
        }
    }

    let input = input_wav
        .to_str()
        .ok_or_else(|| anyhow!("invalid input wav path"))?;
    let pattern = out_dir.join("chunk_%06d.wav");
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| anyhow!("invalid output pattern path"))?;

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input,
            "-f",
            "segment",
            "-segment_time",
            &segment_secs.to_string(),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            pattern_str,
        ])
        .status()?;

    if !status.success() {
        return Err(anyhow!("ffmpeg failed to segment audio"));
    }

    let mut files: Vec<PathBuf> = fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wav"))
        .collect();
    files.sort();
    Ok(files)
}

