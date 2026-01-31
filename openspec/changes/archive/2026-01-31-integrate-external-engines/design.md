## Context

Currently, the Othello RL environment supports three opponent types: random, greedy, and self-play. These are implemented directly in Python/Rust bindings. Three external C/C++ Othello engines (aelskens, drohh, nealetham) exist in the `tmp/` folder but are not integrated into the training pipeline.

The goal is to convert these engines to Rust and make them available as training opponents, allowing agents to learn from diverse strategic approaches. This requires:
1. Understanding and converting C/C++ engine code to Rust
2. Creating Python FFI bindings to call engine move selection
3. Integrating engines into the environment's opponent policy system
4. Updating training scripts to expose engine selection

## Goals / Non-Goals

**Goals:**
- Convert all three external C/C++ engines to Rust implementations
- Integrate engines as selectable opponents (via `--opponent aelskens|drohh|nealetham`)
- Maintain backward compatibility with existing opponent types (random, greedy, self)
- Ensure engine move computation is performant (sub-millisecond target per move)
- Provide clean opponent configuration interface

**Non-Goals:**
- Optimizing or improving the engine algorithms themselves
- Benchmarking or comparing engine strength
- Building a GUI or visualization for engine moves
- Creating a generic "plugin" system for arbitrary external engines
- Changing the environment's core game logic or observation space

## Decisions

### Decision 1: Rust Conversion Approach
**Choice**: Convert C/C++ engines directly to Rust with minimal algorithmic changes.
**Rationale**: 
- Maintains engine behavior consistency while leveraging Rust's safety and performance
- Avoids FFI complexity of calling C/C++ from Python
- Enables easy integration with existing PyO3 Rust bindings
**Alternatives**:
- Call C/C++ directly via Python ctypes/cffi: More complex FFI, harder to maintain
- Rewrite engines from scratch: High effort, risk of behavior changes

### Decision 2: Opponent Configuration Structure
**Choice**: Use a configuration struct (OpponentConfig) that maps engine names to Rust handler functions, with lazy initialization.
**Rationale**:
- Clean separation between environment and engine logic
- Extensible for future opponent types without modifying core environment
- Easy to enable/disable engines at compile or runtime
**Alternatives**:
- Hardcoded if/else statements in environment: Less maintainable, harder to extend
- Fully pluggable module system: Over-engineered for current scope

### Decision 3: File Organization
**Choice**: Create `rust/engines/` subfolder with separate modules for each engine:
```
rust/engines/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Export public API
│   ├── aelskens.rs           # aelskens engine
│   ├── drohh.rs              # drohh engine
│   └── nealetham.rs          # nealetham engine
└── tests/
    └── integration_tests.rs
```
**Rationale**:
- Clear separation of engine implementations
- Easier to manage dependencies per engine if needed
- Shared Cargo.toml simplifies build configuration
**Alternatives**:
- Separate crate for each engine: Adds build complexity
- All in single file: Hard to navigate and maintain

### Decision 4: Rust-Python Interface
**Choice**: Extend existing PyO3 bindings in `rust/othello/src/bindings.rs` with engine move computation functions.
**Rationale**:
- Leverages existing PyO3 setup
- Single compilation/build step via maturin
- Consistent with current architecture
**Alternatives**:
- Separate PyO3 library for engines: Adds complexity, duplicate build setup
- Direct Python ctypes wrapper: Performance overhead, FFI maintenance burden

### Decision 5: Move Selection Interface
**Choice**: Each engine exposes `fn compute_move(board: &Board, player: u8) -> u8` returning action 0-63.
**Rationale**:
- Matches current greedy/random opponent interface
- Board type already defined in core engine
- Deterministic per engine (no randomness in engine selection)
**Alternatives**:
- Return move + confidence/evaluation: Adds complexity, not needed for opponent
- Async move computation: Unnecessary overhead

## Risks / Trade-offs

**[Risk] Engine Algorithm Conversion Errors**
- *Description*: C/C++ to Rust conversion might introduce bugs changing engine behavior
- *Mitigation*: 
  - Create comprehensive unit tests comparing engine moves against original implementations where possible
  - Include validation tests in training (monitor opponent win rates for consistency)
  - Keep original C/C++ code as reference documentation

**[Risk] Performance Regression**
- *Description*: Rust conversion or FFI overhead could make moves slower than needed
- *Mitigation*:
  - Benchmark move computation latency vs. greedy opponent
  - Use Rust optimizations (release build, inlining)
  - Profile FFI boundary if bottleneck identified

**[Trade-off] Code Duplication
- *Description*: Engine implementations require substantial Rust code beyond the original C/C++
- *Impact*: Larger codebase to maintain, but essential for integration
- *Mitigation*: Clear documentation of what each engine does, modular structure

**[Risk] Limited Engine Documentation**
- *Description*: C/C++ source code in `tmp/` may lack documentation making conversion difficult
- *Mitigation*:
  - Study code structure and add inline comments during conversion
  - Test incrementally against known board positions
  - Reach out to original authors if available

**[Trade-off] Python Environment Integration
- *Description*: Extending environment's opponent logic adds complexity to gym wrapper
- *Impact*: Slightly larger env.py file, more opponent types to handle
- *Mitigation*: Clean opponent factory function, move logic into separate helper

## Migration Plan

1. **Phase 1**: Convert first engine (recommend `aelskens` as smallest scope)
   - Convert C/C++ to Rust in `rust/engines/`
   - Create PyO3 bindings for move computation
   - Add unit tests for move validation
   - Integrate into environment's opponent system
   - Test via training script

2. **Phase 2**: Convert remaining engines (drohh, nealetham)
   - Follow same pattern as Phase 1
   - Reuse shared binding infrastructure

3. **Phase 3**: Integration testing
   - Run full training pipeline with each engine as opponent
   - Benchmark move latency
   - Verify backward compatibility with existing opponents

4. **Rollback Strategy**: Keep original C/C++ code in `tmp/` as reference; if critical issues arise, revert to greedy/random opponents

## Open Questions

1. Do the C/C++ engines have any external dependencies (libraries, specialized data structures) that complicate conversion?
2. Are there any known differences in move computation between engines that should be preserved?
3. Should engines be optional dependencies (compile-time feature flags) or always included?
4. What is acceptable latency for engine move computation (current greedy: <0.1ms)?
