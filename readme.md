# README

## ia_content_creator
Transcribe un video de clase localmente y genera un resumen en Markdown usando un LLM local vía Ollama.

## Requisitos
- ffmpeg instalado en PATH (incluye ffprobe)
- Modelo Whisper en `res/ggml-small.bin` (o pasar `--whisper-model`)
- Ollama corriendo con un modelo disponible (ej.: `ollama pull llama3.1:8b`)

## Uso rápido
```
OLLAMA_HOST=http://127.0.0.1:11434 cargo run -- \
  --input /ruta/a/clase.mp4 \
  --output outputs \
  --whisper-model res/ggml-small.bin \
  --ollama-model llama3.1:8b \
  --chunk-secs 300 \
  --concurrency 0 \
  --es
```

## Salidas
- `outputs/audio/<video>.wav` — audio mono 16 kHz
- `outputs/transcript/<video>.txt` — transcripción
- `outputs/summarys/<video>.md` — resumen Markdown. Incluye una sección final en español:
  - Si el profesor menciona fecha/día de próxima clase: encabezado "Para DDMM" (DDMM numérico, ej. 1503 para 15/03) con tareas a preparar/estudiar.
  - En caso contrario: "Para la próxima clase" con elementos concretos.

Opciones de idioma:
- `--language en` (insensible a mayúsculas) o atajos `--en` / `--es`.

Prompt personalizado para el resumen:
- `--summary-prompt "..."` permite sobrescribir las instrucciones por defecto que se envían a Ollama. El transcript se adjunta automáticamente debajo del prompt que indiques.

## Decisiones de diseño
- Por qué ffmpeg: herramienta estándar, multiplataforma y muy rápida para extraer/segmentar audio, con filtros útiles (silenceremove) y ffprobe para medir duración.
- Por qué Whisper: modelo de ASR robusto (encoder–decoder Transformer) entrenado con audio–texto a gran escala; hay implementación eficiente en C++ (whisper.cpp) y bindings en Rust (whisper-rs).
- Por qué Ollama: facilita correr LLMs locales sin conexión, con API HTTP simple (`/api/generate`) y soporte para múltiples modelos.

## Tecnologías
- ffmpeg / ffprobe
  - Qué es: suite de procesamiento multimedia; usamos extracción a WAV mono 16 kHz y `silenceremove` opcional.
  - Cómo se levanta: binarios del sistema (brew/apt). Ya se llama desde el CLI.
  - Interacción: procesos externos con argumentos `-i`, `-ar 16000`, `-ac 1`, `-af silenceremove`, etc.
- Whisper (CPU y GPU Metal)
  - Qué es: ASR tipo encoder–decoder Transformer, entrenado con millones de horas de audio con texto (etiquetado débil). Formato ggml.
  - CPU: `whisper-rs` carga `res/ggml-*.bin` y decodifica (Greedy, multi‑thread).
  - GPU (Metal): `whisper.cpp` compilado con Metal (binario `whisper`/`whisper-cli`) procesa por chunks; el CLI concatena los resultados.
- Ollama (resumen)
  - Qué es: orquestador de LLMs locales (p.ej., Llama 3, Mistral). Modelos entrenados con grandes corpus de texto.
  - Cómo se levanta: `ollama serve` y `ollama pull <modelo>`.
  - Interacción: POST `OLLAMA_HOST/api/generate` con `model`, `prompt`, `stream=false`.

## Comparativa de performance (mediciones locales)

Medido en MacBook Pro M3 Pro, clip de 5 minutos, Greedy decoding, `--skip-summary` (solo transcripción):

| Configuración           | Duración clip | Tiempo real |
|------------------------|---------------|-------------|
| CPU (small, c=1)       | 5:00          | 23.9 s      |
| GPU Metal (small, c=1) | 5:00          | 13.9 s      |

Mejora con GPU ≈ 42% (13.9 s vs 23.9 s). El resultado puede variar según carga, modelo y chunking.

### Medir tiempo
- macOS: `/usr/bin/time -l target/release/ia_content_creator --input clip_5min.mp4 --output outputs --whisper-model res/ggml-small.bin --use-metal --whisper-cli whisper.cpp/build/bin/whisper-cli --chunk-secs 600 --concurrency 1 --skip-summary --es`
  - `real` corresponde al tiempo total (ej.: `13.90 real`).

## Opciones de performance
- `--chunk-secs 300` segmenta el audio y transcribe en paralelo.
- `--concurrency N` controla tareas en paralelo (CPU: auto n_cores/2 si N=0; GPU: por defecto 1).
- `--trim-silence` recorta silencios antes de transcribir.
- CPU: decodificación Greedy + todos los núcleos por tarea.

## GPU (Metal) con whisper.cpp
- Compilar: `cd whisper.cpp && make GGML_METAL=1 -j` (o CMake con `-DGGML_METAL=ON`).
- Ejecutar: `--use-metal --whisper-cli whisper.cpp/build/bin/whisper-cli` (o `whisper.cpp/main`).
- Recomendación: `--chunk-secs 600 --concurrency 1` en GPU.
