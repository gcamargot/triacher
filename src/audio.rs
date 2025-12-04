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

// =============================================================================
// Integration Tests (require ffmpeg)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
    }

    // -------------------------------------------------------------------------
    // wav_duration_secs tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_wav_duration_jfk_sample() {
        let wav_path = fixtures_dir().join("jfk_11s.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test: fixture not found at {:?}", wav_path);
            return;
        }

        let duration = wav_duration_secs(&wav_path).unwrap();
        // JFK sample is approximately 11 seconds
        assert!(
            duration > 10.0 && duration < 12.0,
            "Expected ~11s, got {}",
            duration
        );
    }

    #[test]
    fn test_wav_duration_chunk() {
        let wav_path = fixtures_dir().join("test_chunk_001.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let duration = wav_duration_secs(&wav_path).unwrap();
        // Chunks are 2 seconds
        assert!(
            duration > 1.9 && duration < 2.1,
            "Expected ~2s, got {}",
            duration
        );
    }

    #[test]
    fn test_wav_duration_nonexistent() {
        let result = wav_duration_secs(Path::new("/nonexistent/file.wav"));
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // video_duration_secs_ffprobe tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_video_duration_short_video() {
        let video_path = fixtures_dir().join("short_video.mp4");
        if !video_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let duration = video_duration_secs_ffprobe(&video_path).unwrap();
        // Short video is 10 seconds
        assert!(
            duration > 9.0 && duration < 11.0,
            "Expected ~10s, got {}",
            duration
        );
    }

    #[test]
    fn test_video_duration_nonexistent() {
        let result = video_duration_secs_ffprobe(Path::new("/nonexistent/video.mp4"));
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // extract_audio_ffmpeg tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_audio_creates_wav() {
        let video_path = fixtures_dir().join("short_video.mp4");
        if !video_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("extracted.wav");

        let result = extract_audio_ffmpeg(&video_path, &output_wav);
        assert!(result.is_ok(), "Failed to extract audio: {:?}", result);
        assert!(output_wav.exists(), "Output WAV was not created");
    }

    #[test]
    fn test_extract_audio_correct_format() {
        let video_path = fixtures_dir().join("short_video.mp4");
        if !video_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("extracted.wav");

        extract_audio_ffmpeg(&video_path, &output_wav).unwrap();

        // Verify format using hound
        let reader = hound::WavReader::open(&output_wav).unwrap();
        let spec = reader.spec();

        assert_eq!(spec.sample_rate, 16000, "Expected 16kHz sample rate");
        assert_eq!(spec.channels, 1, "Expected mono audio");
        assert_eq!(spec.bits_per_sample, 16, "Expected 16-bit audio");
    }

    #[test]
    fn test_extract_audio_duration_matches() {
        let video_path = fixtures_dir().join("short_video.mp4");
        if !video_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("extracted.wav");

        extract_audio_ffmpeg(&video_path, &output_wav).unwrap();

        let video_dur = video_duration_secs_ffprobe(&video_path).unwrap();
        let audio_dur = wav_duration_secs(&output_wav).unwrap();

        // Audio duration should be within 0.5s of video duration
        let diff = (video_dur - audio_dur).abs();
        assert!(
            diff < 0.5,
            "Duration mismatch: video={}, audio={}",
            video_dur,
            audio_dur
        );
    }

    #[test]
    fn test_extract_audio_nonexistent_input() {
        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("output.wav");

        let result = extract_audio_ffmpeg(Path::new("/nonexistent/video.mp4"), &output_wav);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // trim_silence_ffmpeg tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trim_silence_runs() {
        let wav_path = fixtures_dir().join("jfk_11s.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("trimmed.wav");

        let result = trim_silence_ffmpeg(&wav_path, &output_wav);
        assert!(result.is_ok(), "trim_silence_ffmpeg failed: {:?}", result);
        assert!(output_wav.exists(), "Output WAV was not created");
    }

    #[test]
    fn test_trim_silence_output_valid() {
        let wav_path = fixtures_dir().join("jfk_11s.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test: fixture not found");
            return;
        }

        let temp = tempdir().unwrap();
        let output_wav = temp.path().join("trimmed.wav");

        trim_silence_ffmpeg(&wav_path, &output_wav).unwrap();

        // Verify output is valid WAV
        let duration = wav_duration_secs(&output_wav).unwrap();
        assert!(
            duration > 0.0,
            "Trimmed audio should have positive duration"
        );
    }
}
