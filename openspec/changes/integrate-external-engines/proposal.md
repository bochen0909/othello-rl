## Why

The current training environment supports only random and greedy opponents, limiting the diversity of strategies available for training agents. Integrating high-performance external Othello engines (aelskens, drohh, nealetham) as training opponents will enable agents to learn from diverse playing styles and improve game-playing ability. These engines implement sophisticated algorithms and represent different strategic approaches to Othello, providing superior training diversity compared to built-in opponents.

## What Changes

- Convert three external C/C++ Othello engines from `tmp/` to Rust implementations
- Integrate converted engines as selectable opponent options in the training environment
- Add new opponent selection to the training script (via `--opponent` flag and environment configuration)
- Create Python-Rust FFI bindings to call engine move selection from the environment
- Maintain backward compatibility with existing opponent types (random, greedy, self)

## Capabilities

### New Capabilities

- `external-engine-opponents`: Ability to select and play against external Othello engines as opponents during environment steps
- `engine-move-computation`: Rust FFI capability to compute engine moves from board state with proper integration into environment logic
- `opponent-configuration`: Configuration system to register and manage external engines by name (aelskens, drohh, nealetham)

### Modified Capabilities

- `opponent-policy`: Extend existing opponent policy system to support external engine opponents while maintaining compatibility with random, greedy, and self-play modes

## Impact

- **Code changes**: 
  - New Rust engine conversions in `rust/engines/` subfolder
  - PyO3 bindings for engine move selection
  - Environment wrapper updates to recognize engine opponent types
  - Training script updates to accept engine names
- **APIs**: Environment `opponent` parameter will accept engine names (strings)
- **Dependencies**: No new external dependencies; uses existing PyO3 stack
- **Backward compatibility**: All existing opponent modes (random, greedy, self) continue to work unchanged
