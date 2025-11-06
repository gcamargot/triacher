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

    /// Force language code for Whisper (e.g., en, es). Case-insensitive.
    #[arg(long)]
    pub language: Option<String>,

    /// Convenience flag: set language to English (same as --language en)
    #[arg(long, alias = "EN")]
    pub en: bool,

    /// Convenience flag: set language to Spanish (same as --language es)
    #[arg(long, alias = "ES", alias = "Es")]
    pub es: bool,
}
