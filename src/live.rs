use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Result};

use crate::cli::LiveArgs;
use crate::gpu;
use crate::summarize;
use reqwest::Client;

/// Minimum expected size for a 2s chunk of 16kHz mono 16-bit audio (~64KB)
const MIN_CHUNK_SIZE: u64 = 60_000;

pub async fn run_live(
    base_output: &Path,
    whisper_model: &Path,
    whisper_cli: &Path,
    language: Option<&str>,
    ollama_model: &str,
    ollama_host: Option<&str>,
    live: &LiveArgs,
) -> Result<()> {
    if live.list_devices {
        let devs = list_devices()?;
        println!("{}", devs);
        return Ok(());
    }

    if live.meeting_device.is_none() {
        return Err(anyhow!("Debe especificar --meeting-device (ej.: 'BlackHole 2ch'). Use --list-devices para ver opciones."));
    }
    if !whisper_cli.exists() {
        return Err(anyhow!(
            "--whisper-cli no encontrado en {}",
            whisper_cli.display()
        ));
    }

    let session = live.session.clone().unwrap_or_else(now_session_name);
    let paths = prepare_paths(base_output, &session)?;
    fs::create_dir_all(paths.transcript_path.parent().unwrap())?;
    fs::create_dir_all(paths.summary_path.parent().unwrap())?;

    append_log(
        &paths,
        &format!("session={} start={:?}", session, SystemTime::now()),
    )?;

    let procs = spawn_capture(live, &paths)?;
    println!("Grabando sesión '{}'… Ctrl+C para finalizar.", session);

    // Live transcription from meeting chunks
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let mut transcript_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.transcript_path)?;
    let session_start = Instant::now();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("\nFinalizando sesión…");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                // Read chunks directory, skip iteration if not available yet
                let files: Vec<PathBuf> = match fs::read_dir(&paths.chunks_dir) {
                    Ok(entries) => entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wav"))
                        .collect(),
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            eprintln!("Warning: Failed to read chunks dir: {}", e);
                        }
                        continue;
                    }
                };
                let mut files = files;
                files.sort();

                for f in files {
                    if seen.contains(&f) { continue; }

                    // Wait for file to stabilize (check 3 times over 600ms)
                    let mut stable = false;
                    let mut last_size = 0u64;
                    for _ in 0..3 {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        let size = f.metadata().map(|m| m.len()).unwrap_or(0);
                        if size > 0 && size == last_size {
                            stable = true;
                            break;
                        }
                        last_size = size;
                    }

                    // Skip if not stable or too small
                    if !stable || last_size < MIN_CHUNK_SIZE {
                        continue;
                    }

                    // Prepare values for spawn_blocking
                    let whisper_cli_owned = whisper_cli.to_path_buf();
                    let whisper_model_owned = whisper_model.to_path_buf();
                    let f_clone = f.clone();
                    let language_owned = language.map(|s| s.to_string());
                    let out_prefix = f.with_extension("");

                    // Transcribe in blocking task to avoid blocking async runtime
                    let transcribe_result = tokio::task::spawn_blocking(move || {
                        gpu::transcribe_chunk_with_cli(
                            &whisper_cli_owned,
                            &whisper_model_owned,
                            &f_clone,
                            language_owned.as_deref(),
                            &out_prefix,
                        )
                    }).await;

                    let text = match transcribe_result {
                        Ok(Ok(t)) => t,
                        Ok(Err(e)) => {
                            eprintln!("Warning: Failed to transcribe chunk {}: {}", f.display(), e);
                            seen.insert(f.clone()); // Mark as seen to avoid retrying
                            continue;
                        }
                        Err(e) => {
                            eprintln!("Warning: Transcription task panicked for {}: {}", f.display(), e);
                            seen.insert(f.clone());
                            continue;
                        }
                    };

                    // Use elapsed wall-clock time for timestamps
                    let elapsed_secs = session_start.elapsed().as_secs();
                    let ts_str = format_ts(elapsed_secs);
                    let line = format!("[{}] {}", ts_str, text.trim());
                    println!("{}", line);
                    writeln!(transcript_file, "{}", line)?;
                    transcript_file.flush()?; // Ensure data is written immediately
                    append_log(&paths, &format!("chunk={} ts={} bytes={} text_len={}", chunk_index(&f), ts_str, last_size, text.len()))?;
                    seen.insert(f.clone());
                }
            }
        }
    }

    // Cleanup
    kill_procs(procs);

    // Concat meeting chunks
    if live.meeting_device.is_some() {
        let _ = concat_meeting(&paths);
    }

    // Optional mix
    if live.mix_audio && paths.meeting_wav.exists() && paths.mic_wav.exists() {
        let mix_path = paths.audio_dir.join("mix.wav");
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                paths.meeting_wav.to_str().unwrap(),
                "-i",
                paths.mic_wav.to_str().unwrap(),
                "-filter_complex",
                "amix=inputs=2:duration=longest",
                mix_path.to_str().unwrap(),
            ])
            .status()?;
        if !status.success() {
            eprintln!("Warning: amix failed");
        }
    }

    append_log(&paths, "session ended").ok();

    // Prompt for summary
    println!("\nOpciones de resumen:\n  1) Clase de la Universidad\n  2) Presentación de la Universidad\n  3) Reunión de trabajo\n  4) Otro (especificar)\nSeleccione [1-4]: ");
    let mut input = String::new();
    let prompt = match std::io::stdin().read_line(&mut input) {
        Ok(_) => {
            match input.trim() {
                "1" => templates::clase(),
                "2" => templates::presentacion(),
                "3" => templates::reunion(),
                _ => {
                    println!("Ingrese prompt personalizado (una línea): ");
                    let mut p = String::new();
                    match std::io::stdin().read_line(&mut p) {
                        Ok(_) if !p.trim().is_empty() => p.trim().to_string(),
                        _ => {
                            eprintln!("Warning: Could not read custom prompt, using default meeting template");
                            templates::reunion()
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not read input ({}), using default meeting template",
                e
            );
            templates::reunion()
        }
    };

    // Read transcript
    let transcript_text = fs::read_to_string(&paths.transcript_path).unwrap_or_default();
    let client = Client::new();
    let summary = summarize::summarize_markdown(
        &client,
        ollama_model,
        &transcript_text,
        ollama_host,
        Some(&prompt),
    )
    .await?;
    fs::write(&paths.summary_path, summary)?;

    println!("\nResumen generado: {}", paths.summary_path.display());
    Ok(())
}

mod templates {
    pub fn clase() -> String {
        "Eres un tomador de apuntes experto para clases universitarias. Resume en Markdown con:\n- Título\n- TL;DR (3–7 bullets)\n- Conceptos clave (con definiciones)\n- Esquema por temas con timestamps\n- Ejemplos y demostraciones\n- Preguntas abiertas\n- Para DDMM / Para la próxima clase (tareas/lecturas)\n- Referencias\nDevuelve solo Markdown.".to_string()
    }
    pub fn presentacion() -> String {
        "Actúas como analista de presentaciones académicas. Resume en Markdown con:\n- Título de la presentación\n- Oradores y afiliaciones\n- Agenda\n- Mensajes clave\n- Evidencias/datos citados\n- Preguntas de la audiencia\n- Decisiones o conclusiones\n- Próximos eventos/acciones\nDevuelve solo Markdown.".to_string()
    }
    pub fn reunion() -> String {
        "Eres un PM asistente. Produce un acta en Markdown con:\n- Título de la reunión\n- Participantes\n- Objetivo\n- Decisiones\n- Acciones (responsable y due date sugerido)\n- Riesgos y dependencias\n- Próximos pasos y fecha tentativo\n- Para la próxima reunión (preparación)\nDevuelve solo Markdown.".to_string()
    }
}

pub struct LivePaths {
    #[allow(dead_code)]
    pub session_dir: PathBuf,
    #[allow(dead_code)]
    pub video_dir: PathBuf,
    pub audio_dir: PathBuf,
    pub chunks_dir: PathBuf,
    pub video_file: PathBuf,
    pub mic_wav: PathBuf,
    pub meeting_concat_list: PathBuf,
    pub meeting_wav: PathBuf,
    pub transcript_path: PathBuf,
    pub summary_path: PathBuf,
    pub log_path: PathBuf,
}

pub fn now_session_name() -> String {
    let now = chrono::Local::now();
    now.format("%Y%m%d_%H%M%S").to_string()
}

pub fn prepare_paths(base_output: &Path, session: &str) -> Result<LivePaths> {
    let session_dir = base_output.join("live").join(session);
    let video_dir = session_dir.join("video");
    let audio_dir = session_dir.join("audio");
    let chunks_dir = audio_dir.join("meeting_chunks");
    fs::create_dir_all(&video_dir)?;
    fs::create_dir_all(&chunks_dir)?;

    Ok(LivePaths {
        session_dir: session_dir.clone(),
        video_dir,
        audio_dir: audio_dir.clone(),
        chunks_dir,
        video_file: session_dir.join("video").join(format!("{}.mp4", session)),
        mic_wav: audio_dir.join("mic.wav"),
        meeting_concat_list: audio_dir.join("meeting.txt"),
        meeting_wav: audio_dir.join("meeting.wav"),
        transcript_path: base_output
            .join("transcript")
            .join(format!("{}.txt", session)),
        summary_path: base_output
            .join("summaries")
            .join(format!("{}.md", session)),
        log_path: session_dir.join("live.log"),
    })
}

pub fn list_devices() -> Result<String> {
    let out = Command::new("ffmpeg")
        .args(["-f", "avfoundation", "-list_devices", "true", "-i", ""]) // prints to stderr
        .output()?;
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    Ok(stderr)
}

pub struct LiveProcs {
    pub video: Option<Child>,
    pub mic: Option<Child>,
    pub meeting_chunker: Option<Child>,
}

pub fn spawn_capture(live: &LiveArgs, paths: &LivePaths) -> Result<LiveProcs> {
    // Video: screen capture
    let video = match Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-i",
            &format!("{}:none", live.screen),
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "1",
            paths.video_file.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => Some(child),
        Err(e) => {
            eprintln!("Warning: Failed to start screen capture: {}", e);
            None
        }
    };

    // Mic capture (continuous)
    let mic = if let Some(mic_dev) = &live.mic_device {
        match Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "avfoundation",
                "-i",
                &format!(":{}", mic_dev),
                "-ar",
                "16000",
                "-ac",
                "1",
                paths.mic_wav.to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => Some(child),
            Err(e) => {
                eprintln!("Warning: Failed to start mic capture: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Meeting chunker (2s segments)
    let meeting_chunker = if let Some(meet_dev) = &live.meeting_device {
        let pattern = paths.chunks_dir.join("chunk_%06d.wav");
        match Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "avfoundation",
                "-i",
                &format!(":{}", meet_dev),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "segment",
                "-segment_time",
                "2",
                "-reset_timestamps",
                "1",
                pattern.to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => Some(child),
            Err(e) => {
                eprintln!("Warning: Failed to start meeting audio capture: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Critical check: meeting_chunker is required for live transcription
    if live.meeting_device.is_some() && meeting_chunker.is_none() {
        return Err(anyhow!(
            "Failed to start meeting audio capture - ensure ffmpeg is installed and the device name is correct"
        ));
    }

    Ok(LiveProcs {
        video,
        mic,
        meeting_chunker,
    })
}

pub fn kill_procs(mut procs: LiveProcs) {
    let _ = procs.video.as_mut().map(|c| c.kill());
    let _ = procs.mic.as_mut().map(|c| c.kill());
    let _ = procs.meeting_chunker.as_mut().map(|c| c.kill());
}

pub fn append_log(paths: &LivePaths, line: &str) -> Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}

/// Format seconds into MM:SS timestamp string
pub fn format_ts(sec: u64) -> String {
    let m = sec / 60;
    let s = sec % 60;
    format!("{:02}:{:02}", m, s)
}

/// Extract chunk index from filename like "chunk_000123.wav" -> 123
pub fn chunk_index(path: &Path) -> usize {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit_once('_').map(|(_, idx)| idx.to_string()))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn concat_meeting(paths: &LivePaths) -> Result<()> {
    let mut list = String::new();
    let mut files: Vec<PathBuf> = fs::read_dir(&paths.chunks_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wav"))
        .collect();
    files.sort();
    for p in files {
        list.push_str(&format!("file '{}'\n", p.display()));
    }
    fs::write(&paths.meeting_concat_list, list)?;
    // Copy concat via re-encode-safe=0, but wav copy works with concat demuxer
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            paths.meeting_concat_list.to_str().unwrap(),
            "-c",
            "copy",
            paths.meeting_wav.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err(anyhow!("ffmpeg concat failed"));
    }
    Ok(())
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // -------------------------------------------------------------------------
    // format_ts tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_ts_zero() {
        assert_eq!(format_ts(0), "00:00");
    }

    #[test]
    fn test_format_ts_seconds_only() {
        assert_eq!(format_ts(5), "00:05");
        assert_eq!(format_ts(59), "00:59");
    }

    #[test]
    fn test_format_ts_minutes_and_seconds() {
        assert_eq!(format_ts(60), "01:00");
        assert_eq!(format_ts(65), "01:05");
        assert_eq!(format_ts(125), "02:05");
    }

    #[test]
    fn test_format_ts_large_values() {
        assert_eq!(format_ts(3600), "60:00"); // 1 hour
        assert_eq!(format_ts(3661), "61:01"); // 1 hour, 1 minute, 1 second
        assert_eq!(format_ts(7200), "120:00"); // 2 hours
    }

    // -------------------------------------------------------------------------
    // chunk_index tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_chunk_index_valid() {
        let path = Path::new("/path/to/chunk_000000.wav");
        assert_eq!(chunk_index(path), 0);

        let path = Path::new("/path/to/chunk_000001.wav");
        assert_eq!(chunk_index(path), 1);

        let path = Path::new("/path/to/chunk_000123.wav");
        assert_eq!(chunk_index(path), 123);
    }

    #[test]
    fn test_chunk_index_large_number() {
        let path = Path::new("chunk_999999.wav");
        assert_eq!(chunk_index(path), 999999);
    }

    #[test]
    fn test_chunk_index_no_underscore() {
        let path = Path::new("audio.wav");
        assert_eq!(chunk_index(path), 0);
    }

    #[test]
    fn test_chunk_index_non_numeric_suffix() {
        let path = Path::new("chunk_abc.wav");
        assert_eq!(chunk_index(path), 0);
    }

    #[test]
    fn test_chunk_index_empty_suffix() {
        let path = Path::new("chunk_.wav");
        assert_eq!(chunk_index(path), 0);
    }

    #[test]
    fn test_chunk_index_different_prefix() {
        let path = Path::new("audio_segment_042.wav");
        assert_eq!(chunk_index(path), 42);
    }

    // -------------------------------------------------------------------------
    // now_session_name tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_now_session_name_format() {
        let name = now_session_name();
        // Format: YYYYMMDD_HHMMSS (15 chars)
        assert_eq!(name.len(), 15);
        assert!(name.chars().nth(8) == Some('_'));
        // All other chars should be digits
        for (i, c) in name.chars().enumerate() {
            if i == 8 {
                assert_eq!(c, '_');
            } else {
                assert!(
                    c.is_ascii_digit(),
                    "Expected digit at position {}, got '{}'",
                    i,
                    c
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // prepare_paths tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prepare_paths_creates_directories() {
        let temp = tempdir().unwrap();
        let base = temp.path();
        let session = "test_session";

        let paths = prepare_paths(base, session).unwrap();

        // Verify directories were created
        assert!(paths.chunks_dir.exists());
        assert!(base.join("live").join(session).join("video").exists());
        assert!(base.join("live").join(session).join("audio").exists());
    }

    #[test]
    fn test_prepare_paths_correct_structure() {
        let temp = tempdir().unwrap();
        let base = temp.path();
        let session = "my_meeting";

        let paths = prepare_paths(base, session).unwrap();

        // Check path structure
        assert!(paths
            .video_file
            .to_string_lossy()
            .contains("my_meeting.mp4"));
        assert!(paths.mic_wav.to_string_lossy().contains("mic.wav"));
        assert!(paths.meeting_wav.to_string_lossy().contains("meeting.wav"));
        assert!(paths
            .transcript_path
            .to_string_lossy()
            .contains("my_meeting.txt"));
        assert!(paths
            .summary_path
            .to_string_lossy()
            .contains("my_meeting.md"));
        assert!(paths.log_path.to_string_lossy().contains("live.log"));
    }

    #[test]
    fn test_prepare_paths_transcript_in_base() {
        let temp = tempdir().unwrap();
        let base = temp.path();
        let session = "sess";

        let paths = prepare_paths(base, session).unwrap();

        // Transcript should be in base/transcript/, not in session dir
        assert_eq!(
            paths.transcript_path,
            base.join("transcript").join("sess.txt")
        );
    }

    #[test]
    fn test_prepare_paths_summary_in_summaries() {
        let temp = tempdir().unwrap();
        let base = temp.path();
        let session = "sess";

        let paths = prepare_paths(base, session).unwrap();

        // Summary should be in base/summaries/
        assert_eq!(paths.summary_path, base.join("summaries").join("sess.md"));
    }

    // -------------------------------------------------------------------------
    // append_log tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_append_log_creates_file() {
        let temp = tempdir().unwrap();
        let paths = prepare_paths(temp.path(), "log_test").unwrap();

        append_log(&paths, "test line 1").unwrap();

        assert!(paths.log_path.exists());
        let content = fs::read_to_string(&paths.log_path).unwrap();
        assert!(content.contains("test line 1"));
    }

    #[test]
    fn test_append_log_appends() {
        let temp = tempdir().unwrap();
        let paths = prepare_paths(temp.path(), "log_test2").unwrap();

        append_log(&paths, "line 1").unwrap();
        append_log(&paths, "line 2").unwrap();
        append_log(&paths, "line 3").unwrap();

        let content = fs::read_to_string(&paths.log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line 1");
        assert_eq!(lines[1], "line 2");
        assert_eq!(lines[2], "line 3");
    }
}
