## ADDED Requirements

### Requirement: Opponent policy supports external engine opponents
The opponent policy system SHALL be extended to recognize and handle external engine opponent types.

#### Scenario: Engine opponent is initialized successfully
- **WHEN** opponent is created with type "aelskels"
- **THEN** opponent policy initializes engine and is ready for move generation

#### Scenario: Engine opponent generates moves during environment step
- **WHEN** environment step is taken and opponent is an external engine
- **THEN** engine opponent generates valid move via PyO3 binding call

#### Scenario: Multiple engine types can be used in same session
- **WHEN** environments are created with different engine opponents (aelskels, drohh, nealetham)
- **THEN** each environment correctly uses its configured engine

### Requirement: Opponent policy maintains backward compatibility
Existing opponent types (random, greedy, self) SHALL continue to function without modification.

#### Scenario: Random opponent still works
- **WHEN** environment is created with opponent="random"
- **THEN** environment uses random opponent without any changes to behavior

#### Scenario: Greedy opponent still works
- **WHEN** environment is created with opponent="greedy"
- **THEN** environment uses greedy opponent without any changes to behavior

#### Scenario: Self-play still works
- **WHEN** environment is created with opponent="self"
- **THEN** environment uses self-play without any changes to behavior

#### Scenario: Custom callable opponents still work
- **WHEN** environment is created with opponent=custom_function
- **THEN** environment uses custom function as opponent without any changes

### Requirement: Opponent policy factory handles all opponent types
The opponent factory function SHALL accept opponent parameter and return appropriate opponent implementation.

#### Scenario: Factory recognizes engine name
- **WHEN** factory function receives "aelskels" as opponent parameter
- **THEN** factory returns engine opponent implementation

#### Scenario: Factory recognizes built-in opponent name
- **WHEN** factory function receives "random" or "greedy" as opponent parameter
- **THEN** factory returns built-in opponent implementation

#### Scenario: Factory handles custom callable
- **WHEN** factory function receives callable as opponent parameter
- **THEN** factory returns callable wrapped as opponent

### Requirement: Opponent policy integrates with environment step logic
Engine opponents SHALL integrate seamlessly into the environment's step function.

#### Scenario: Engine opponent move is used in step
- **WHEN** environment is stepped and opponent is an engine
- **THEN** engine move is applied to board and game continues normally

#### Scenario: Engine opponent respects valid move constraints
- **WHEN** engine opponent makes a move
- **THEN** move is checked for validity before application to board

#### Scenario: Engine opponent handles no-valid-moves case
- **WHEN** engine has no valid moves available
- **THEN** turn passes to player without error or game state corruption

### Requirement: Opponent policy works with training script
Training script SHALL accept engine names via `--opponent` flag and pass to environment.

#### Scenario: Training script passes engine opponent to environment
- **WHEN** training script is run with `--opponent aelskels`
- **THEN** environment is created with opponent="aelskels"

#### Scenario: Training script works with all engine types
- **WHEN** training script is run with `--opponent drohh` and `--opponent nealetham`
- **THEN** training executes successfully with each engine as opponent
