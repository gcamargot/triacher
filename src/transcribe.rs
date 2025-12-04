use std::path::Path;

use anyhow::Result;
use hound::WavReader;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperError};

/// Timing metadata for each decoded segment.
/// Kept for potential subtitle/timestamp features.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
}

/// Transcribe using an existing WhisperContext (allows reusing the loaded model).
pub fn transcribe_wav_with_ctx(
    ctx: &WhisperContext,
    wav_path: &Path,
    language: Option<&str>,
) -> Result<(String, Vec<Segment>)> {
    // Read PCM samples (i16) from WAV and detect channel count
    let mut reader = WavReader::open(wav_path)?;
    let spec = reader.spec();
    let channels = spec.channels;
    let samples_i16: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.unwrap_or_default())
        .collect();

    // Convert to f32 in [-1, 1]
    let mut samples_f32 = vec![0.0f32; samples_i16.len()];
    convert_integer_to_float_audio(&samples_i16, &mut samples_f32)?;

    // If stereo, convert to mono; if already mono, keep as-is
    let mono = if channels == 2 {
        whisper_rs::convert_stereo_to_mono_audio(&samples_f32)?
    } else {
        samples_f32
    };

    // Create decoding state from provided context
    let mut state = ctx.create_state()?;

    // Faster decoding: Greedy with best_of = 1
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    // Use all available CPU cores
    let n_threads = std::cmp::max(1, num_cpus::get() as i32);
    params.set_n_threads(n_threads);
    if let Some(lang) = language {
        params.set_language(Some(lang));
    }

    state.full(params, &mono[..])?;

    // Collect segments
    let mut transcript = String::new();
    let mut segments = Vec::new();
    let n = state.full_n_segments()?;
    for i in 0..n {
        let raw_text = state.full_get_segment_text(i)?;
        let t0_10ms = state.full_get_segment_t0(i)? as i32; // 10ms units
        let t1_10ms = state.full_get_segment_t1(i)? as i32;
        let seg = Segment {
            start_ms: t0_10ms * 10,
            end_ms: t1_10ms * 10,
            text: raw_text.clone(),
        };
        if !transcript.is_empty() {
            transcript.push(' ');
        }
        transcript.push_str(raw_text.trim());
        segments.push(seg);
    }

    Ok((transcript, segments))
}

fn convert_integer_to_float_audio(samples: &[i16], output: &mut [f32]) -> Result<(), WhisperError> {
    if samples.len() != output.len() {
        return Err(WhisperError::InputOutputLengthMismatch {
            input_len: samples.len(),
            output_len: output.len(),
        });
    }
    for (i, o) in samples.iter().zip(output.iter_mut()) {
        *o = *i as f32 / 32768.0;
    }
    Ok(())
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // convert_integer_to_float_audio tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convert_integer_to_float_basic() {
        let samples: Vec<i16> = vec![0, 16384, -16384, 32767, -32768];
        let mut output = vec![0.0f32; samples.len()];

        convert_integer_to_float_audio(&samples, &mut output).unwrap();

        assert!((output[0] - 0.0).abs() < 0.001, "0 should map to 0.0");
        assert!((output[1] - 0.5).abs() < 0.001, "16384 should map to ~0.5");
        assert!(
            (output[2] - (-0.5)).abs() < 0.001,
            "-16384 should map to ~-0.5"
        );
        assert!((output[3] - 1.0).abs() < 0.001, "32767 should map to ~1.0");
        assert!(
            (output[4] - (-1.0)).abs() < 0.001,
            "-32768 should map to -1.0"
        );
    }

    #[test]
    fn test_convert_integer_to_float_empty() {
        let samples: Vec<i16> = vec![];
        let mut output: Vec<f32> = vec![];

        let result = convert_integer_to_float_audio(&samples, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_integer_to_float_length_mismatch() {
        let samples: Vec<i16> = vec![0, 1, 2];
        let mut output = vec![0.0f32; 2]; // Wrong size

        let result = convert_integer_to_float_audio(&samples, &mut output);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Segment struct tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_segment_creation() {
        let seg = Segment {
            start_ms: 1000,
            end_ms: 2500,
            text: "Hello world".to_string(),
        };

        assert_eq!(seg.start_ms, 1000);
        assert_eq!(seg.end_ms, 2500);
        assert_eq!(seg.text, "Hello world");
    }

    #[test]
    fn test_segment_clone() {
        let seg = Segment {
            start_ms: 0,
            end_ms: 500,
            text: "Test".to_string(),
        };

        let cloned = seg.clone();
        assert_eq!(cloned.start_ms, seg.start_ms);
        assert_eq!(cloned.end_ms, seg.end_ms);
        assert_eq!(cloned.text, seg.text);
    }

    // -------------------------------------------------------------------------
    // Integration tests for transcribe_wav_with_ctx
    // -------------------------------------------------------------------------
    //
    // NOTE: These tests require a Whisper model file (e.g., res/ggml-base.bin).
    // They are marked with #[ignore] by default to avoid CI failures.
    //
    // To run these tests locally:
    //   1. Download a Whisper model: scripts/get_whisper_latest.sh
    //   2. Place it at res/ggml-base.bin (or update MODEL_PATH below)
    //   3. Run: cargo test -- --ignored
    //
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "requires whisper model at res/ggml-base.bin"]
    fn test_transcribe_jfk_sample() {
        use std::path::PathBuf;
        use whisper_rs::WhisperContextParameters;

        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("res")
            .join("ggml-base.bin");

        if !model_path.exists() {
            eprintln!("Skipping: model not found at {:?}", model_path);
            return;
        }

        let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("jfk_11s.wav");

        if !wav_path.exists() {
            eprintln!("Skipping: fixture not found at {:?}", wav_path);
            return;
        }

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(&model_path.to_string_lossy(), params).unwrap();
        let (transcript, segments) = transcribe_wav_with_ctx(&ctx, &wav_path, Some("en")).unwrap();

        // JFK speech should contain recognizable words
        assert!(!transcript.is_empty(), "Transcript should not be empty");
        assert!(!segments.is_empty(), "Should have at least one segment");

        // Verify segment timing makes sense
        for seg in &segments {
            assert!(seg.end_ms >= seg.start_ms, "End should be >= start");
            assert!(!seg.text.is_empty(), "Segment text should not be empty");
        }

        println!("Transcript: {}", transcript);
    }

    #[test]
    #[ignore = "requires whisper model at res/ggml-base.bin"]
    fn test_transcribe_chunk() {
        use std::path::PathBuf;
        use whisper_rs::WhisperContextParameters;

        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("res")
            .join("ggml-base.bin");

        if !model_path.exists() {
            return;
        }

        let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("test_chunk_001.wav");

        if !wav_path.exists() {
            return;
        }

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(&model_path.to_string_lossy(), params).unwrap();
        let result = transcribe_wav_with_ctx(&ctx, &wav_path, Some("en"));

        // Even for silence/short audio, should not error
        assert!(result.is_ok(), "Transcription should succeed: {:?}", result);
    }
}
