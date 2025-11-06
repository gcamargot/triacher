use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Context, Result};

use crate::cli::LiveArgs;
use crate::gpu;
use crate::summarize;
use reqwest::Client;

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
        return Err(anyhow!("--whisper-cli no encontrado en {}", whisper_cli.display()));
    }

    let session = live.session.clone().unwrap_or_else(|| now_session_name());
    let paths = prepare_paths(base_output, &session)?;
    fs::create_dir_all(paths.transcript_path.parent().unwrap())?;
    fs::create_dir_all(paths.summary_path.parent().unwrap())?;

    append_log(&paths, &format!("session={} start={:?}", session, SystemTime::now()))?;

    let procs = spawn_capture(live, &paths)?;
    println!("Grabando sesión '{}'… Ctrl+C para finalizar.", session);

    // Live transcription from meeting chunks
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let mut transcript_file = OpenOptions::new().create(true).append(true).open(&paths.transcript_path)?;
    let base_ts = SystemTime::now();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("\nFinalizando sesión…");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                let mut files: Vec<PathBuf> = fs::read_dir(&paths.chunks_dir)
                    .unwrap_or_else(|_| fs::read_dir(".").unwrap())
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wav"))
                    .collect();
                files.sort();
                for f in files {
                    if seen.contains(&f) { continue; }
                    // ffmpeg may still be writing; check size stabilizes
                    let s1 = f.metadata().map(|m| m.len()).unwrap_or(0);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    let s2 = f.metadata().map(|m| m.len()).unwrap_or(0);
                    if s1 == 0 || s1 != s2 { continue; }

                    // transcribe
                    let idx = chunk_index(&f);
                    let out_prefix = f.with_extension("");
                    let text = gpu::transcribe_chunk_with_cli(
                        whisper_cli,
                        whisper_model,
                        &f,
                        language,
                        &out_prefix,
                    )?;

                    let ts_str = format_ts(idx as u64 * 2);
                    let line = format!("[{}] {}", ts_str, text.trim());
                    println!("{}", line);
                    writeln!(transcript_file, "{}", line)?;
                    append_log(&paths, &format!("chunk={} ts={} bytes={} text_len={}", idx, ts_str, s2, text.len()))?;
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
                "-y","-i", paths.meeting_wav.to_str().unwrap(),
                "-i", paths.mic_wav.to_str().unwrap(),
                "-filter_complex","amix=inputs=2:duration=longest",
                mix_path.to_str().unwrap()
            ]).status()?;
        if !status.success() { eprintln!("Warning: amix failed"); }
    }

    append_log(&paths, "session ended").ok();

    // Prompt for summary
    println!("\nOpciones de resumen:\n  1) Clase de la Universidad\n  2) Presentación de la Universidad\n  3) Reunión de trabajo\n  4) Otro (especificar)\nSeleccione [1-4]: ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    let choice = input.trim();
    let prompt = match choice {
        "1" => templates::clase(),
        "2" => templates::presentacion(),
        "3" => templates::reunion(),
        _ => {
            println!("Ingrese prompt personalizado (una línea): ");
            let mut p = String::new();
            std::io::stdin().read_line(&mut p).ok();
            p.trim().to_string()
        }
    };

    // Read transcript
    let transcript_text = fs::read_to_string(&paths.transcript_path).unwrap_or_default();
    let client = Client::new();
    let summary = summarize::summarize_markdown(&client, ollama_model, &transcript_text, ollama_host, Some(&prompt)).await?;
    fs::write(&paths.summary_path, summary)?;

    println!("\nResumen generado: {}", paths.summary_path.display());
    Ok(())
}

fn chunk_index(path: &Path) -> usize {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit_once('_').map(|(_, idx)| idx.to_string()))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
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
    pub session_dir: PathBuf,
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
        transcript_path: base_output.join("transcript").join(format!("{}.txt", session)),
        summary_path: base_output.join("summarys").join(format!("{}.md", session)),
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

pub fn spawn_capture(
    live: &LiveArgs,
    paths: &LivePaths,
) -> Result<LiveProcs> {
    // Video: screen capture
    let video = Command::new("ffmpeg")
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
        .ok();

    // Mic capture (continuous)
    let mic = if let Some(mic_dev) = &live.mic_device {
        Command::new("ffmpeg")
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
            .ok()
    } else {
        None
    };

    // Meeting chunker (2s segments)
    let meeting_chunker = if let Some(meet_dev) = &live.meeting_device {
        let pattern = paths.chunks_dir.join("chunk_%06d.wav");
        Command::new("ffmpeg")
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
            .ok()
    } else {
        None
    };

    Ok(LiveProcs { video, mic, meeting_chunker })
}

pub fn kill_procs(mut procs: LiveProcs) {
    let _ = procs.video.as_mut().map(|c| c.kill());
    let _ = procs.mic.as_mut().map(|c| c.kill());
    let _ = procs.meeting_chunker.as_mut().map(|c| c.kill());
}

pub fn append_log(paths: &LivePaths, line: &str) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(&paths.log_path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}

fn format_ts(sec: u64) -> String {
    let m = sec / 60;
    let s = sec % 60;
    format!("{:02}:{:02}", m, s)
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
        list.push_str(&format!("file '{}')\n", p.display()).replace(")\\n", "\n"));
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
