use std::path::Path;

use hound::WavReader;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError};
use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
}

pub fn transcribe_wav(
    model_path: &Path,
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

    // Load Whisper model
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path"))?,
        WhisperContextParameters::default(),
    )?;
    let mut state = ctx.create_state()?;

    let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 1.0 });
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

fn convert_integer_to_float_audio(
    samples: &[i16],
    output: &mut [f32],
) -> Result<(), WhisperError> {
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
