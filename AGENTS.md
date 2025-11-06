# Repository Guidelines

## Project Structure & Module Organization
- `src/` — Rust source (entry: `src/main.rs`).
- `res/` — local assets: Whisper model (`ggml-base.bin`) and base video (`Highres.mp4`).
- `build/` — generated artifacts (`output.wav`, `subtitles.srt`, `final_video.mp4`, `video_with_subtitles.mp4`).
- `target/` — Cargo build output (ignored).
- `tests/` — integration tests (add as needed).

## Build, Test, and Development Commands
- `cargo build` — compile the project.
- `cargo run` — run the CLI end‑to‑end (requires API key, model, and ffmpeg).
- `cargo fmt --all` — format with rustfmt.
- `cargo clippy --all-targets -- -D warnings` — lint; treat warnings as errors.
- `cargo test` — run unit/integration tests when added.

Prerequisites: install `ffmpeg`; place model at `res/ggml-base.bin`; ensure a base video at `res/Highres.mp4`.

## Coding Style & Naming Conventions
- Use rustfmt defaults; run `cargo fmt` before commits.
- Prefer `snake_case` for functions/files, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for consts.
- Keep modules small and focused; extract helpers from `main.rs` as the code grows (e.g., `src/audio.rs`, `src/video.rs`).

## Testing Guidelines
- Co-locate unit tests with modules using `#[cfg(test)]` and add integration tests under `tests/`.
- Add tests for new logic (parsing, time-scaling, SRT formatting); mock external calls.
- Aim to keep `cargo clippy` clean and avoid `allow` unless justified.

## Security & Configuration
- API key: store in `src/apikey.txt` (git-ignored). Example: a single line with the key.
- Do not commit keys, media, or model files. `.gitignore` already blocks common artifacts.
- Large binaries (models/videos) should be managed outside Git or via release assets.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`). Example from history: `fix: Erase some common warnings`.
- PRs should include: clear description, linked issue, how to run/validate, and sample output (paths under `build/`).
- Keep changes focused; update docs when changing commands, structure, or config.
