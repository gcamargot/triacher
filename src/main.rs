mod audio;
mod cli;
mod summarize;
mod transcribe;
mod chunk;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use cli::Cli;
use reqwest::Client;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    // Validate inputs
    if !args.input.exists() {
        anyhow::bail!("Input video not found: {}", args.input.display());
    }
    if !args.whisper_model.exists() {
        anyhow::bail!(
            "Whisper model not found: {} (expected ggml .bin)",
            args.whisper_model.display()
        );
    }

    // Prepare output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;
    let audio_path = args.output.join("audio.wav");
    let transcript_path = args.output.join("transcript.txt");
    let summary_path = args.output.join("summary.md");

    // 1) Extract audio with ffmpeg
    audio::extract_audio_ffmpeg(&args.input, &audio_path)
        .context("ffmpeg audio extraction failed")?;

    // Optional: trim silence
    let audio_for_transcript = if args.trim_silence {
        let trimmed = args.output.join("audio_trimmed.wav");
        audio::trim_silence_ffmpeg(&audio_path, &trimmed)
            .context("ffmpeg silence trimming failed")?;
        trimmed
    } else {
        audio_path.clone()
    };

    // 2) Transcribe with Whisper
    // Resolve language preference (case-insensitive) with convenience flags
    let mut lang = args.language.as_ref().map(|s| s.to_lowercase());
    if args.en { lang = Some("en".to_string()); }
    if args.es { lang = Some("es".to_string()); }

    let transcript = if args.chunk_secs > 0 {
        let chunks_dir = args.output.join("chunks");
        println!("Segmenting audio into ~{}s chunks…", args.chunk_secs);
        let files = chunk::segment_wav_ffmpeg(&audio_for_transcript, &chunks_dir, args.chunk_secs)
            .context("Audio segmentation failed")?;
        println!("Created {} chunk(s)", files.len());

        let default_conc = std::cmp::max(1, num_cpus::get() / 2);
        let max_conc = if args.concurrency == 0 { default_conc } else { args.concurrency };
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_conc));
        let lang_clone = lang.clone();

        // Reuse a single WhisperContext across chunks
        let ctx = std::sync::Arc::new(whisper_rs::WhisperContext::new_with_params(
            args.whisper_model
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?,
            whisper_rs::WhisperContextParameters::default(),
        )?);

        let mut handles = Vec::with_capacity(files.len());
        for (idx, path) in files.iter().enumerate() {
            let p = path.clone();
            let ctx2 = ctx.clone();
            let l = lang_clone.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let handle = tokio::task::spawn_blocking(move || {
                let _permit = permit; // hold until end
                let start = std::time::Instant::now();
                println!("[chunk {}/?] Transcribing {}…", idx + 1, p.file_name().and_then(|n| n.to_str()).unwrap_or("?"));
                let r = transcribe::transcribe_wav_with_ctx(&ctx2, Path::new(&p), l.as_deref())
                    .map(|(t, _)| (idx, t));
                let elapsed = start.elapsed();
                println!("[chunk {}/?] Done in {:.1}s", idx + 1, elapsed.as_secs_f32());
                r
            });
            handles.push(handle);
        }

        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            let r = h.await.expect("transcription task panicked")?;
            results.push(r);
        }
        results.sort_by_key(|(idx, _)| *idx);
        let mut combined = String::new();
        for (_, t) in results {
            if !combined.is_empty() { combined.push('\n'); }
            combined.push_str(&t);
        }
        combined
    } else {
        // Non-chunked: reuse single context as well
        let ctx = whisper_rs::WhisperContext::new_with_params(
            args.whisper_model
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?,
            whisper_rs::WhisperContextParameters::default(),
        )?;
        let (transcript_text, _segments) =
            transcribe::transcribe_wav_with_ctx(&ctx, Path::new(&audio_for_transcript), lang.as_deref())
                .context("Whisper transcription failed")?;
        transcript_text
    };

    // Save transcript
    fs::write(&transcript_path, &transcript).context("Failed to write transcript.txt")?;

    // 3) Summarize via Ollama
    let client = Client::new();
    let summary = summarize::summarize_markdown(&client, &args.ollama_model, &transcript, args.ollama_host.as_deref())
        .await
        .context("Ollama summarization failed")?;

    fs::write(&summary_path, &summary).context("Failed to write summary.md")?;

    println!(
        "Done.\n- Audio: {}\n- Transcript: {}\n- Summary: {}",
        audio_path.display(),
        transcript_path.display(),
        summary_path.display()
    );

    Ok(())
}
