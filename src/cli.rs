use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "ia_content_creator", version, about = "Transcribe and summarize lecture videos locally")] 
pub struct Cli {
    /// Input video file (lecture/class recording)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for artifacts (created if missing)
    #[arg(short, long, default_value = "build")] 
    pub output: PathBuf,

    /// Whisper model file path (ggml .bin)
    #[arg(long, default_value = "res/ggml-small.bin")]
    pub whisper_model: PathBuf,

    /// Ollama model name (e.g., llama3.1:8b, mistral:7b)
    #[arg(long, default_value = "llama3.1:8b")]
    pub ollama_model: String,

    /// Ollama host URL (overrides OLLAMA_HOST), e.g., http://127.0.0.1:11434
    #[arg(long)]
    pub ollama_host: Option<String>,

    /// Use whisper.cpp binary with Metal (GPU) instead of whisper-rs (CPU)
    #[arg(long, default_value_t = false)]
    pub use_metal: bool,

    /// Path to whisper.cpp binary (built with Metal). Default: whisper.cpp/main
    #[arg(long, default_value = "whisper.cpp/main")]
    pub whisper_cli: String,

    /// Force language code for Whisper (e.g., en, es). Case-insensitive.
    #[arg(long)]
    pub language: Option<String>,

    /// Convenience flag: set language to English (same as --language en)
    #[arg(long, alias = "EN")]
    pub en: bool,

    /// Convenience flag: set language to Spanish (same as --language es)
    #[arg(long, alias = "ES", alias = "Es")]
    pub es: bool,

    /// Chunk duration in seconds for parallel transcription (0 = disable)
    #[arg(long, default_value_t = 300)]
    pub chunk_secs: u32,

    /// Max concurrent transcription tasks (0 = auto n_cores/2)
    #[arg(long, default_value_t = 0)]
    pub concurrency: usize,

    /// Remove long silences from audio before transcribing (ffmpeg silenceremove)
    #[arg(long, default_value_t = false)]
    pub trim_silence: bool,
}
