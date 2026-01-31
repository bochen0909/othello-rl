# Engine Analysis and Conversion Plan

## Summary
Three external Othello engines have been analyzed for conversion to Rust:
1. **aelskels** - Alpha-beta pruning AI
2. **drohh** - Standard Othello implementation  
3. **nealetham** - Naive greedy AI

## Detailed Analysis

### 1. aelskels Engine

**Algorithm**: Alpha-beta pruning with 5-turn lookahead
**Language**: C++
**Files**: 
- `Player.h` / `Player.cpp` - Core player and AI logic
- `Map.h` / `Map.cpp` - Game board management
- `View1.h` / `View1.cpp` - Display/UI
- `Main.cpp` - Entry point
- Report: `INFO_H304___rapport.pdf` - Academic documentation

**Key Structures**:
- `Player` class (handles AI, human, and file-based players)
  - `cross_validate_movement()` - Validates and applies moves
  - `apply_color()` - Flips pieces
  - Move tracking and piece counting
- `Map` class - 8x8 board state using `map<string, int>` (coordinate -> color)
  - Position format: ASCII coordinates (e.g., "a1")
  
**Decision Algorithm**:
- Alpha-beta pruning with depth 5 (looks 5 moves ahead)
- Evaluates board states using heuristic scoring
- Can take up to 20 seconds per move (not suitable for high-frequency training)
- Moves are deterministic

**Board Representation**: 
- Uses string keys like "a1", "b2" etc. mapping to color (int)
- Need to convert to standard 8x8 bit representation (0-63 actions)

**Dependencies**: 
- Standard C++ library only (no external dependencies)
- Uses `<map>`, `<vector>`, `<algorithm>`, `<random>`, `<chrono>`, `<thread>`

**Complexity**: Medium (well-structured with classes)
**Difficulty**: Medium (need to adapt coordinate system and heuristics)

---

### 2. drohh Engine

**Algorithm**: Standard Othello rules engine with directional flood-fill
**Language**: C++
**Files**:
- `main.cpp` (700 lines) - Single monolithic file with all logic
- No separate header files

**Key Functions**:
- `place_disc(board, row, col, player)` - Places disc and flips opponent pieces
- Directional validation using delta arrays: `{{-1,-1}, {-1,0}, ..., {1,1}}`
- Brute force validation: checks 8 directions for each move
- Game loop with human vs AI modes

**Decision Algorithm**:
- Currently appears to use random or basic greedy selection
- No complex heuristics visible
- Moves are deterministic given valid move list

**Board Representation**:
- 8x8 2D array of chars
- 'X' = black, 'O' = white, '.' = empty
- Direct row/col indexing (0-7)
- Easy to convert to bit representation

**Dependencies**: 
- Standard C++ library only
- Uses `<vector>`, `<iostream>`, `<array>` etc.

**Complexity**: Low (monolithic, straightforward logic)
**Difficulty**: Easy (simple bit manipulation, directional scanning)

**Note**: Code is 700 lines in single file - good candidate for first conversion

---

### 3. nealetham Engine

**Algorithm**: Naive greedy heuristic + lookahead preparation
**Language**: C++  
**Files**:
- `othello.cpp` (290 lines) - Main game logic
- `othello-naive-ai.cpp` / `othello-naive-ai.h` - AI implementation
- Makefile for building

**Key Functions**:
- `is_possible_move(row, col, opponent)` - Move validation
- `make_move(row, col, current_player)` - Updates board with piece flips
- Naive AI: selects move with maximum immediate piece gain
- Caches valid moves per player

**Decision Algorithm**:
- Greedy: picks move that captures most opponent pieces
- Simple and fast
- Deterministic given valid move list
- Good baseline for comparison

**Board Representation**:
- `std::array<std::array<int, 8>, 8> board` - 8x8 int array
- 0 = empty, 1 = player, -1 = opponent
- Also tracks scores per player
- Clean and efficient

**Data Structures**:
- `std::vector<std::array<int, 2>>` for valid moves (precomputed)
- `std::array<int, 2>` for each move coordinate
- Efficient for iteration

**Dependencies**: 
- Standard C++ library only
- Uses `<array>`, `<vector>`, `<algorithm>`

**Complexity**: Low (simple heuristic, clean code)
**Difficulty**: Easy (straightforward greedy logic, clean board representation)

---

## Conversion Strategy

### Execution Order (Recommend Reverse Alphabetical):
1. **nealetham** (FIRST) - Easiest, clean structure, good reference
2. **drohh** (SECOND) - Medium complexity, single file structure
3. **aelskels** (THIRD) - Most complex, coordinate conversion needed

### Key Conversion Tasks:

#### Board Representation Conversion:
- **All engines**: Convert to Rust `Board` type already in use by main engine
- **aelskels special**: Convert ASCII coordinate strings ("a1") to action integers (0-63)
- Use formula: `action = row * 8 + col`

#### Move Validation (8-Direction Scanning):
- All engines scan 8 directions from placed piece
- All check for opponent pieces in line followed by player piece
- Rust conversion straightforward using direction deltas

#### AI Decision Logic:
- **aelskels**: Implement minimax/alpha-beta in Rust (most complex)
- **drohh**: Determine and implement actual algorithm (appears greedy/random)
- **nealetham**: Direct greedy scoring heuristic (easiest)

#### PyO3 Bindings:
- Export `compute_move(board: &Board, player: u8) -> u8` from each
- Convert Python board state to Rust Board type at boundary

---

## Identified Dependencies and Special Considerations

**External Dependencies**: NONE for all three engines
- All use only standard C++ library functions
- No special libraries, no OpenGL, no custom frameworks

**Performance Considerations**:
- **aelskels**: 5-turn lookahead is expensive (up to 20s per move)
  - May need to reduce depth for real-time training
  - OR cache results
- **drohh**: Unknown algorithm complexity (need to profile)
- **nealetham**: O(1) greedy heuristic - very fast

**Platform-Specific Code**: None detected
- All cross-platform C++
- No Windows/Mac specific code

---

## Rust Conversion Readiness

| Engine | Readiness | Notes |
|--------|-----------|-------|
| nealetham | HIGH | Clean code, simple heuristic, easy conversion |
| drohh | MEDIUM | Single file, straightforward but needs profiling |
| aelskels | MEDIUM-HIGH | Complex but well-structured, coordinate mapping needed |

**Estimated Effort**:
- nealetham: ~3-4 hours
- drohh: ~4-5 hours  
- aelskels: ~6-8 hours
- Total: ~13-17 hours for full conversion
