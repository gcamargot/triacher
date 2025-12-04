//! CLI Integration Tests
//!
//! Tests for command-line argument parsing and validation.
//! These are the highest priority tests as they ensure users can
//! interact with the application correctly.

use assert_cmd::Command;
use predicates::prelude::*;

#[allow(deprecated)] // cargo_bin is stable and widely used; alternative is more complex
fn cmd() -> Command {
    Command::cargo_bin("ia_content_creator").unwrap()
}

// =============================================================================
// Subcommand Requirement Tests
// =============================================================================

#[test]
fn test_no_subcommand_shows_help() {
    cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage:"))
        .stderr(predicate::str::contains("process"))
        .stderr(predicate::str::contains("live"));
}

#[test]
fn test_help_flag_shows_usage() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Transcribe and summarize"))
        .stdout(predicate::str::contains("process"))
        .stdout(predicate::str::contains("live"));
}

#[test]
fn test_version_flag() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("ia_content_creator"));
}

// =============================================================================
// Process Subcommand Tests
// =============================================================================

#[test]
fn test_process_requires_input() {
    cmd()
        .arg("process")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--input"));
}

#[test]
fn test_process_help() {
    cmd()
        .args(["process", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--input"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--chunk-secs"))
        .stdout(predicate::str::contains("--skip-summary"));
}

#[test]
fn test_process_rejects_nonexistent_input() {
    cmd()
        .args(["process", "--input", "/nonexistent/video.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_process_with_valid_input_but_missing_model() {
    // Using a fixture file that exists
    cmd()
        .args([
            "process",
            "--input",
            "tests/fixtures/short_video.mp4",
            "--whisper-model",
            "/nonexistent/model.bin",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Whisper model not found"));
}

// =============================================================================
// Live Subcommand Tests
// =============================================================================

#[test]
fn test_live_help() {
    cmd()
        .args(["live", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--meeting-device"))
        .stdout(predicate::str::contains("--mic-device"))
        .stdout(predicate::str::contains("--list-devices"))
        .stdout(predicate::str::contains("--screen"));
}

#[test]
fn test_live_requires_metal_flag() {
    cmd()
        .args(["live", "--meeting-device", "BlackHole 2ch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Metal"));
}

#[test]
fn test_live_list_devices_no_meeting_required() {
    // --list-devices should work without --meeting-device
    // It will still fail because --use-metal is required, but not for missing device
    cmd()
        .args(["live", "--list-devices"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Metal"));
}

#[test]
fn test_live_requires_meeting_device_when_not_listing() {
    cmd()
        .args(["live", "--use-metal"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--meeting-device"));
}

// =============================================================================
// Global Options Tests
// =============================================================================

#[test]
fn test_language_en_flag() {
    // Just verify the flag is accepted (will fail later for missing input)
    cmd()
        .args(["--en", "process", "--input", "/nonexistent.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found")); // Fails on file, not flag
}

#[test]
fn test_language_es_flag() {
    cmd()
        .args(["--es", "process", "--input", "/nonexistent.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_language_explicit() {
    cmd()
        .args(["--language", "fr", "process", "--input", "/nonexistent.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_whisper_model_option() {
    cmd()
        .args([
            "--whisper-model",
            "custom/model.bin",
            "process",
            "--input",
            "/nonexistent.mp4",
        ])
        .assert()
        .failure();
}

#[test]
fn test_ollama_model_option() {
    cmd()
        .args([
            "--ollama-model",
            "custom:model",
            "process",
            "--input",
            "/nonexistent.mp4",
        ])
        .assert()
        .failure();
}

#[test]
fn test_ollama_host_option() {
    cmd()
        .args([
            "--ollama-host",
            "http://custom:11434",
            "process",
            "--input",
            "/nonexistent.mp4",
        ])
        .assert()
        .failure();
}

// =============================================================================
// Process-specific Options Tests
// =============================================================================

#[test]
fn test_process_chunk_secs_option() {
    cmd()
        .args([
            "process",
            "--input",
            "/nonexistent.mp4",
            "--chunk-secs",
            "120",
        ])
        .assert()
        .failure();
}

#[test]
fn test_process_concurrency_option() {
    cmd()
        .args([
            "process",
            "--input",
            "/nonexistent.mp4",
            "--concurrency",
            "4",
        ])
        .assert()
        .failure();
}

#[test]
fn test_process_trim_silence_flag() {
    cmd()
        .args(["process", "--input", "/nonexistent.mp4", "--trim-silence"])
        .assert()
        .failure();
}

#[test]
fn test_process_skip_summary_flag() {
    cmd()
        .args(["process", "--input", "/nonexistent.mp4", "--skip-summary"])
        .assert()
        .failure();
}

// =============================================================================
// Live-specific Options Tests
// =============================================================================

#[test]
fn test_live_screen_option() {
    cmd()
        .args([
            "live",
            "--use-metal",
            "--meeting-device",
            "Test Device",
            "--screen",
            "1",
        ])
        .assert()
        .failure(); // Will fail on whisper-cli, but screen option accepted
}

#[test]
fn test_live_session_option() {
    cmd()
        .args([
            "live",
            "--use-metal",
            "--meeting-device",
            "Test Device",
            "--session",
            "my_test_session",
        ])
        .assert()
        .failure();
}

#[test]
fn test_live_mix_audio_flag() {
    cmd()
        .args([
            "live",
            "--use-metal",
            "--meeting-device",
            "Test Device",
            "--mic-device",
            "Mic Device",
            "--mix-audio",
        ])
        .assert()
        .failure();
}

#[test]
fn test_live_output_option() {
    cmd()
        .args([
            "live",
            "--use-metal",
            "--meeting-device",
            "Test Device",
            "--output",
            "custom_output",
        ])
        .assert()
        .failure();
}
