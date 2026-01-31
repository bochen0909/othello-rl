## ADDED Requirements

### Requirement: External engines can be registered as opponents
The system SHALL provide a mechanism to register external engines and map them to opponent implementations.

#### Scenario: Engine registry contains all three engines
- **WHEN** environment initialization occurs
- **THEN** internal engine registry contains entries for aelskels, drohh, and nealetham

#### Scenario: Opponent can be configured by engine name
- **WHEN** user specifies `opponent="aelskels"` in environment configuration
- **THEN** system retrieves engine from registry and configures it as opponent

### Requirement: Opponent configuration accepts engine names
The opponent parameter in environment initialization SHALL accept external engine names as valid string values.

#### Scenario: aelskels is valid opponent value
- **WHEN** `gym.make("Othello-v0", opponent="aelskels")` is called
- **THEN** environment recognizes "aelskels" as valid opponent configuration

#### Scenario: drohh is valid opponent value
- **WHEN** `gym.make("Othello-v0", opponent="drohh")` is called
- **THEN** environment recognizes "drohh" as valid opponent configuration

#### Scenario: nealetham is valid opponent value
- **WHEN** `gym.make("Othello-v0", opponent="nealetham")` is called
- **THEN** environment recognizes "nealetham" as valid opponent configuration

#### Scenario: Invalid engine name is rejected
- **WHEN** `gym.make("Othello-v0", opponent="invalid_engine")` is called
- **THEN** environment raises ValueError with helpful message listing available engines

### Requirement: Engine factory function creates opponent callables
The system SHALL provide a factory function that creates opponent callable functions for each engine.

#### Scenario: Factory creates aelskels opponent callable
- **WHEN** opponent factory is called with engine name "aelskels"
- **THEN** factory returns a callable that accepts board state and returns move action

#### Scenario: Engine opponent callable accepts observation
- **WHEN** engine opponent callable is invoked with observation array
- **THEN** callable returns integer action in range [0, 63]

### Requirement: Engine configuration is consistent across training runs
Engine configuration for a given engine name SHALL produce identical behavior across multiple training runs (deterministic).

#### Scenario: Same engine configuration is created consistently
- **WHEN** environment is created with same engine name multiple times
- **THEN** each environment instance uses identical engine configuration

#### Scenario: Engine moves are reproducible
- **WHEN** multiple training runs use same engine with identical agent moves
- **THEN** engine produces identical moves in equivalent game states

### Requirement: Engine list can be queried
The system SHALL provide a way to query available external engines.

#### Scenario: Available engines can be listed
- **WHEN** system is queried for available engine opponents
- **THEN** system returns list containing ["aelskels", "drohh", "nealetham"]

#### Scenario: Engine list is accessible to training script
- **WHEN** user runs training script with `--opponent help` or similar
- **THEN** system displays list of available opponent options including all three engines
