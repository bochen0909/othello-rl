# Building and Installing the Othello Rust Extension

This document describes how to build and install the Othello Rust extension module for Python.

## Prerequisites

- Rust toolchain (rustc, cargo)
- Python 3.12+
- maturin (Python package for building Rust extensions)

## Installation

### Development Mode (Recommended for Development)

Install the package in editable mode, which allows you to make changes to the Rust code and rebuild without reinstalling:

```bash
# Install maturin if not already installed
pip install maturin

# Build and install in development mode
maturin develop --manifest-path rust/othello/Cargo.toml
```

### Production Build

For a production build with optimizations:

```bash
# Build release version
maturin develop --release --manifest-path rust/othello/Cargo.toml
```

## Verification

After installation, verify that the module can be imported:

```python
import othello_rust

# Create a game instance
game = othello_rust.OthelloGame()

# Test basic functionality
board = game.get_board()
print(f"Board shape: {board.shape}")  # Should be (8, 8)

valid_moves = game.get_valid_moves()
print(f"Valid moves: {valid_moves.sum()}")  # Should be 4 initially
```

## Running Tests

Run the Python test suite to verify the bindings work correctly:

```bash
# Run unit tests
python -m pytest rust/othello/tests/test_bindings.py -v

# Run property-based tests
python -m pytest rust/othello/tests/test_bindings_properties.py -v

# Run all tests
python -m pytest rust/othello/tests/ -v
```

## Rebuilding

After making changes to the Rust code, rebuild with:

```bash
maturin develop --manifest-path rust/othello/Cargo.toml
```

The `--release` flag can be added for optimized builds, but development builds are faster to compile.

## Troubleshooting

### Import Error

If you get an import error, ensure:
1. The virtual environment is activated
2. maturin develop completed successfully
3. The module name matches `othello_rust` (as defined in bindings.rs)

### Build Errors

If you encounter build errors:
1. Ensure Rust toolchain is up to date: `rustup update`
2. Check that PyO3 dependencies match your Python version
3. Try cleaning the build: `cargo clean` in the rust/othello directory

### Performance

For maximum performance, always use release builds in production:
```bash
maturin develop --release --manifest-path rust/othello/Cargo.toml
```

Release builds are significantly faster (3-10x) than debug builds.
