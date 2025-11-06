use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "ia_content_creator", version, about = "Transcribe and summarize lecture videos locally")] 
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
    /// Input video file (lecture/class recording)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for artifacts (created if missing)
    #[arg(short, long, default_value = "outputs")] 
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

    /// Prompt personalizado para el resumen en Ollama (sobrescribe el prompt por defecto)
    #[arg(long)]
    pub summary_prompt: Option<String>,

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

    /// Saltar la etapa de resumen (útil para benchmarks de transcripción)
    #[arg(long, default_value_t = false)]
    pub skip_summary: bool,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Captura de pantalla + audio (meeting/mic) con transcripción en vivo y resumen al finalizar
    Live(LiveArgs),
}

#[derive(clap::Args, Debug)]
pub struct LiveArgs {
    /// Lista dispositivos de avfoundation (pantalla/audio) y termina
    #[arg(long, default_value_t = false)]
    pub list_devices: bool,

    /// Índice de pantalla a capturar (por defecto 0)
    #[arg(long, default_value_t = 0)]
    pub screen: u32,

    /// Nombre exacto del dispositivo de audio para la meeting (ej.: "BlackHole 2ch")
    #[arg(long)]
    pub meeting_device: Option<String>,

    /// Nombre exacto del dispositivo de audio para el micrófono
    #[arg(long)]
    pub mic_device: Option<String>,

    /// Generar pista mezclada (meeting+mic) al finalizar
    #[arg(long, default_value_t = false)]
    pub mix_audio: bool,

    /// Nombre de la sesión; si se omite se usa fecha-hora
    #[arg(long)]
    pub session: Option<String>,
}
