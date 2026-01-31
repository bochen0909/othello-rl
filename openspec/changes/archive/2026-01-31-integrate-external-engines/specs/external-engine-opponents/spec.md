## ADDED Requirements

### Requirement: Environment accepts external engine opponents
The environment SHALL accept external engine names (`aelskens`, `drohh`, `nealetham`) as valid values for the `opponent` parameter when creating an Othello-v0 environment.

#### Scenario: Create environment with aelskens opponent
- **WHEN** user calls `gym.make("Othello-v0", opponent="aelskels")`
- **THEN** environment initializes successfully without raising an error

#### Scenario: Create environment with drohh opponent
- **WHEN** user calls `gym.make("Othello-v0", opponent="drohh")`
- **THEN** environment initializes successfully without raising an error

#### Scenario: Create environment with nealetham opponent
- **WHEN** user calls `gym.make("Othello-v0", opponent="nealetham")`
- **THEN** environment initializes successfully without raising an error

### Requirement: Engine opponent makes valid moves
When an external engine is selected as opponent, it SHALL generate valid moves during environment steps that conform to Othello rules.

#### Scenario: Engine opponent plays first move
- **WHEN** environment is reset with engine opponent and first move is taken by agent
- **THEN** engine returns a valid move action in range [0, 63] that is legal from current board state

#### Scenario: Engine opponent responds to board state changes
- **WHEN** agent makes a move that changes the board state
- **THEN** engine generates a new valid move based on the updated board state

#### Scenario: Engine handles positions with no valid moves
- **WHEN** engine has no valid moves available
- **THEN** engine returns None or special indicator allowing environment to pass turn to agent

### Requirement: Engine opponent selection via training script
The training script SHALL accept engine names via `--opponent` flag as alternative to built-in opponents.

#### Scenario: Train with aelskens opponent
- **WHEN** user runs `python scripts/train_othello.py --opponent aelskels --num-iterations 10`
- **THEN** training executes without error using aelskels as opponent

#### Scenario: Train with drohh opponent
- **WHEN** user runs `python scripts/train_othello.py --opponent drohh --num-iterations 10`
- **THEN** training executes without error using drohh as opponent

#### Scenario: Train with nealetham opponent
- **WHEN** user runs `python scripts/train_othello.py --opponent nealetham --num-iterations 10`
- **THEN** training executes without error using nealetham as opponent

### Requirement: Engine opponent provides deterministic moves
Each external engine SHALL produce deterministic move selection (same board state always produces same move).

#### Scenario: Same board state produces same move
- **WHEN** engine selects a move from a specific board state twice
- **THEN** both calls return identical move action

#### Scenario: Determinism across game instances
- **WHEN** two identical games are played against same engine with identical agent moves
- **THEN** engine makes identical moves in corresponding game states
