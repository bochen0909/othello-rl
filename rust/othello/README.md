# Othello Rust Engine

High-performance Othello (Reversi) game engine implemented in Rust with Python bindings via PyO3.

## Project Structure

- `src/lib.rs` - Core game logic and types
- `Cargo.toml` - Rust project configuration with PyO3 support

## Core Types

- `Player` - Enum representing Black or White player
- `Cell` - Enum representing Empty, Black, or White cell state
- `Board` - Main game state structure with 8x8 board

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## Initial Setup

The board starts with 4 pieces in the center:
```
  0 1 2 3 4 5 6 7
0 . . . . . . . .
1 . . . . . . . .
2 . . . . . . . .
3 . . . ○ ● . . .
4 . . . ● ○ . . .
5 . . . . . . . .
6 . . . . . . . .
7 . . . . . . . .
```

Where ● = Black, ○ = White, . = Empty
