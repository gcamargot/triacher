# README

## ia_content_creator
Transcribe a lecture/class video locally and produce a Markdown summary using a local LLM via Ollama.

### Prerequisites
- ffmpeg installed and available on PATH
- Whisper model file at `res/ggml-small.bin` (or pass `--whisper-model`)
- Ollama running locally with a model pulled (e.g., `ollama pull llama3.1:8b`)

### Usage
```
OLLAMA_HOST=http://127.0.0.1:11434 cargo run -- \
  --input /path/to/lecture.mp4 \
  --output build \
  --whisper-model res/ggml-small.bin \
  --ollama-model llama3.1:8b \
  --chunk-secs 300 \
  --concurrency 0 \
  --es
```

Outputs:
- `outputs/audio/<video>.wav` — extracted mono 16kHz WAV
- `outputs/transcript/<video>.txt` — raw transcript
- `outputs/summarys/<video>.md` — Markdown summary. Incluye una sección final en español:
  - Si el profesor menciona fecha/día de próxima clase: encabezado "Para DDMM" (DDMM numérico, ej. 1503 para 15/03) con tareas a preparar/estudiar.
  - En caso contrario: "Para la próxima clase" con elementos concretos.

Language options:
- `--language en` (case-insensitive), or convenience flags `--en` / `--es`.

Environment:
- `OLLAMA_HOST` overrides default `http://127.0.0.1:11434`.

### Performance options
- `--chunk-secs 300` segmenta el audio y transcribe en paralelo.
- `--concurrency N` limita tareas en paralelo (por defecto, n_cores/2 si N=0).
- `--trim-silence` recorta silencios largos antes de transcribir.
- Transcripción más rápida por defecto: estrategia Greedy y uso de todos los núcleos.

### GPU (Metal) con whisper.cpp
- Compilar submódulo: `cd whisper.cpp && make -j` (requiere Xcode CLT y cmake)
- Ejecutar con GPU:
  - `target/release/ia_content_creator --input ./video.mp4 --whisper-model res/ggml-small.bin --use-metal --whisper-cli whisper.cpp/main --chunk-secs 600 --concurrency 1 --es`
- Nota: `--concurrency` por defecto 1 en GPU. Aumenta con cautela.
