mod audio;
mod cli;
mod summarize;
mod transcribe;
mod chunk;
mod gpu;
mod live;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use cli::Cli;
use reqwest::Client;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    // Dispatch subcommands
    if let Some(cmd) = &args.command {
        match cmd {
            cli::Commands::Live(largs) => {
                // Require Metal for live
                if !args.use_metal {
                    anyhow::bail!("El modo 'live' requiere GPU Metal. Use --use-metal y apunte --whisper-cli al binario de whisper.cpp.");
                }
                let lang = resolve_lang(&args);
                let whisper_cli_path = std::path::Path::new(&args.whisper_cli).to_path_buf();
                return live::run_live(
                    &args.output,
                    std::path::Path::new(&args.whisper_model),
                    &whisper_cli_path,
                    lang.as_deref(),
                    &args.ollama_model,
                    args.ollama_host.as_deref(),
                    largs,
                )
                .await;
            }
        }
    }

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

    // Prepare output directories and file names based on input video name
    let video_stem = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let out_audio_dir = args.output.join("audio");
    let out_transcript_dir = args.output.join("transcript");
    let out_summarys_dir = args.output.join("summarys");

    fs::create_dir_all(&out_audio_dir).context("Failed to create audio output directory")?;
    fs::create_dir_all(&out_transcript_dir).context("Failed to create transcript output directory")?;
    fs::create_dir_all(&out_summarys_dir).context("Failed to create summarys output directory")?;

    let audio_path = out_audio_dir.join(format!("{}.wav", video_stem));
    let transcript_path = out_transcript_dir.join(format!("{}.txt", video_stem));
    let summary_path = out_summarys_dir.join(format!("{}.md", video_stem));

    // 1) Extract audio with ffmpeg (skip if cached matches duration)
    let mut need_extract = true;
    if audio_path.exists() {
        if let (Ok(v_secs), Ok(a_secs)) = (
            audio::video_duration_secs_ffprobe(&args.input),
            audio::wav_duration_secs(&audio_path),
        ) {
            let diff = (v_secs - a_secs).abs();
            let tol = 1.0_f64.max(v_secs * 0.005); // 1s or 0.5%
            if diff <= tol {
                println!(
                    "Cached audio found (Δ={:.2}s <= {:.2}s). Skipping extraction.",
                    diff, tol
                );
                need_extract = false;
            } else {
                println!(
                    "Cached audio duration mismatch (video {:.2}s vs audio {:.2}s). Re-extracting.",
                    v_secs, a_secs
                );
            }
        }
    }
    if need_extract {
        audio::extract_audio_ffmpeg(&args.input, &audio_path)
            .context("ffmpeg audio extraction failed")?;
    }

    // Optional: trim silence
    let audio_for_transcript = if args.trim_silence {
        let trimmed = out_audio_dir.join(format!("{}_trimmed.wav", video_stem));
        audio::trim_silence_ffmpeg(&audio_path, &trimmed)
            .context("ffmpeg silence trimming failed")?;
        trimmed
    } else {
        audio_path.clone()
    };

    // 2) Transcribe with Whisper
    // Resolve language preference (case-insensitive) with convenience flags
    let lang = resolve_lang(&args);

    let transcript = if args.use_metal {
        // GPU path via whisper.cpp CLI
        let mut cli_path = std::path::PathBuf::from(&args.whisper_cli);
        if !cli_path.exists() {
            // Try common alternative locations
            let candidates = [
                "whisper.cpp/main",
                "whisper.cpp/build/bin/whisper",
                "whisper.cpp/build/bin/whisper-cli",
            ];
            let mut found = None;
            for c in candidates.iter() {
                let p = std::path::Path::new(c);
                if p.exists() { found = Some(p.to_path_buf()); break; }
            }
            if let Some(p) = found { cli_path = p; }
        }
        if !cli_path.exists() {
            anyhow::bail!(
                "whisper CLI not found at '{}' or common locations. Build whisper.cpp with Metal (e.g., 'cd whisper.cpp && make -j'), or pass --whisper-cli with the built binary path (e.g., whisper.cpp/build/bin/whisper).",
                args.whisper_cli
            );
        }

        // Determine concurrency (default 1 for GPU)
        let max_conc = if args.concurrency == 0 { 1 } else { args.concurrency };
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_conc));

        if args.chunk_secs > 0 {
            let chunks_dir = out_audio_dir.join(format!("chunks_{}", video_stem));
            println!("Segmenting audio into ~{}s chunks…", args.chunk_secs);
            let files = chunk::segment_wav_ffmpeg(&audio_for_transcript, &chunks_dir, args.chunk_secs)
                .context("Audio segmentation failed")?;
            println!("Created {} chunk(s)", files.len());

            let lang_clone = lang.clone();
            let cli = cli_path.to_string_lossy().to_string();
            let model = args.whisper_model.clone();

            let mut handles = Vec::with_capacity(files.len());
            for (idx, p) in files.iter().enumerate() {
                let permit = semaphore.clone().acquire_owned().await.unwrap();
                let p2 = p.clone();
                let cli2 = cli.clone();
                let model2 = model.clone();
                let l = lang_clone.clone();
                let out_prefix = chunks_dir.join(format!("out_{:06}", idx));
                let handle = tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let start = std::time::Instant::now();
                    println!(
                        "[chunk {}/?] Transcribing {} via GPU…",
                        idx + 1,
                        p2.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                    );
                    let txt = gpu::transcribe_chunk_with_cli(
                        std::path::Path::new(&cli2),
                        std::path::Path::new(&model2),
                        std::path::Path::new(&p2),
                        l.as_deref(),
                        std::path::Path::new(&out_prefix),
                    )
                    .map(|t| (idx, t));
                    let elapsed = start.elapsed();
                    println!("[chunk {}/?] Done in {:.1}s", idx + 1, elapsed.as_secs_f32());
                    txt
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
            // Single-pass GPU transcription (no chunking)
            println!("Transcribing full audio via GPU…");
            let out_prefix = out_transcript_dir.join(format!("{}_gpu_full", video_stem));
            let text = gpu::transcribe_chunk_with_cli(
                std::path::Path::new(&cli_path),
                std::path::Path::new(&args.whisper_model),
                std::path::Path::new(&audio_for_transcript),
                lang.as_deref(),
                std::path::Path::new(&out_prefix),
            )?;
            text
        }
    } else if args.chunk_secs > 0 {
        let chunks_dir = out_audio_dir.join(format!("chunks_{}", video_stem));
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

    // 3) Summarize via Ollama (optional)
    if !args.skip_summary {
        let client = Client::new();
        let summary = summarize::summarize_markdown(
            &client,
            &args.ollama_model,
            &transcript,
            args.ollama_host.as_deref(),
            args.summary_prompt.as_deref(),
        )
        .await
        .context("Ollama summarization failed")?;

        fs::write(&summary_path, &summary).context("Failed to write summary.md")?;
    }

    println!(
        "Done.\n- Audio: {}\n- Transcript: {}{}",
        audio_path.display(),
        transcript_path.display(),
        if args.skip_summary { "\n- Summary: (skipped)".to_string() } else { format!("\n- Summary: {}", summary_path.display()) }
    );

    Ok(())
}

fn resolve_lang(args: &Cli) -> Option<String> {
    let mut lang = args.language.as_ref().map(|s| s.to_lowercase());
    if args.en { lang = Some("en".to_string()); }
    if args.es { lang = Some("es".to_string()); }
    lang
}
