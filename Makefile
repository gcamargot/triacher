# Makefile for ia_content_creator
# Provides convenient commands for building, testing, and development

.PHONY: all build test test-unit test-all test-live test-ignored fmt lint check clean coverage help

# Default target
all: check test

# Build the project
build:
	cargo build

# Build release version
release:
	cargo build --release

# Run all tests (excluding ignored tests)
test:
	cargo test

# Run unit tests only (fast, no external dependencies)
test-unit:
	cargo test --lib

# Run all tests including integration tests
test-all:
	cargo test --all-targets

# Run live capture tests (requires audio device)
test-live:
	cargo test live:: -- --nocapture

# Run ignored tests (requires whisper model at res/ggml-base.bin)
test-ignored:
	cargo test -- --ignored

# Run tests with verbose output
test-verbose:
	cargo test -- --nocapture

# Format code
fmt:
	cargo fmt --all

# Check formatting without applying changes
fmt-check:
	cargo fmt --all -- --check

# Run clippy linter
lint:
	cargo clippy --all-targets -- -D warnings

# Run all checks (fmt + clippy)
check: fmt-check lint

# Clean build artifacts
clean:
	cargo clean

# Generate test coverage report (requires cargo-tarpaulin)
# Install: cargo install cargo-tarpaulin
coverage:
	@command -v cargo-tarpaulin >/dev/null 2>&1 || { \
		echo "cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"; \
		exit 1; \
	}
	cargo tarpaulin --out Html --output-dir coverage

# Generate coverage with Lcov format (for CI/CD integration)
coverage-lcov:
	@command -v cargo-tarpaulin >/dev/null 2>&1 || { \
		echo "cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"; \
		exit 1; \
	}
	cargo tarpaulin --out Lcov --output-dir coverage

# Run the CLI in process mode (requires input file)
run-process:
	@echo "Usage: cargo run -- process <input-video>"
	@echo "Example: cargo run -- process res/Highres.mp4"

# Run the CLI in live mode (requires audio device)
run-live:
	@echo "Usage: cargo run -- live [options]"
	@echo "Example: cargo run -- live --duration 60"

# Download whisper model (base)
setup-model:
	./scripts/get_whisper_latest.sh

# Help
help:
	@echo "Available targets:"
	@echo "  build        - Compile the project"
	@echo "  release      - Compile release version"
	@echo "  test         - Run all tests (excluding ignored)"
	@echo "  test-unit    - Run unit tests only (fast)"
	@echo "  test-all     - Run all tests including integration"
	@echo "  test-live    - Run live capture tests"
	@echo "  test-ignored - Run ignored tests (requires model)"
	@echo "  test-verbose - Run tests with output"
	@echo "  fmt          - Format code with rustfmt"
	@echo "  fmt-check    - Check formatting"
	@echo "  lint         - Run clippy linter"
	@echo "  check        - Run fmt-check + lint"
	@echo "  clean        - Remove build artifacts"
	@echo "  coverage     - Generate HTML coverage report"
	@echo "  coverage-lcov- Generate Lcov coverage report"
	@echo "  setup-model  - Download whisper model"
	@echo "  help         - Show this help"
