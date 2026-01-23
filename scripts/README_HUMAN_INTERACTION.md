# Human Interaction Utilities for Othello

This directory contains scripts for human interaction with the Othello RL environment.

## Scripts

### 1. play_human_vs_agent.py (Console Version)

Play Othello as a human against an AI agent or built-in opponent in the terminal.

**Features:**
- Play against random or greedy opponents
- Play against trained RL agents (requires checkpoint)
- Choose to play as Black or White
- Interactive console input with move validation
- Visual board display after each move
- Game statistics at the end

**Usage:**

```bash
# Play against random opponent (as Black)
python scripts/play_human_vs_agent.py --opponent random

# Play against greedy opponent (as White)
python scripts/play_human_vs_agent.py --opponent greedy --human-color white

# Play against trained agent
python scripts/play_human_vs_agent.py --checkpoint /path/to/checkpoint
```

**Controls:**
- Enter move as a number 0-63 (row * 8 + col)
- Valid moves are marked with `*` on the board
- Type `q` to quit at any time

**Board Positions:**
```
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55
 56 57 58 59 60 61 62 63
```

### 2. play_human_vs_agent_gui.py (GUI Version)

Play Othello with a graphical interface using Pygame.

**Features:**
- Beautiful graphical board with pieces
- Click to make moves (no typing numbers!)
- Visual indicators for valid moves (yellow dots)
- Real-time score display
- Smooth gameplay experience
- Play against random, greedy, or trained agents

**Installation:**

```bash
# Install pygame
pip install pygame

# Or with poetry
poetry install --extras gui
```

**Usage:**

```bash
# Play against random opponent (as Black)
python scripts/play_human_vs_agent_gui.py --opponent random

# Play against greedy opponent (as White)
python scripts/play_human_vs_agent_gui.py --opponent greedy --human-color white

# Play against trained agent
python scripts/play_human_vs_agent_gui.py --checkpoint /path/to/checkpoint
```

**Controls:**
- Click on yellow dots to make valid moves
- Press `ESC` or `Q` to quit
- Press `R` to restart after game ends

**Visual Guide:**
- Green board with grid
- Black and white pieces
- Yellow dots show valid moves
- Orange border highlights opponent's last move (for 1.5 seconds)
- 800ms delay after opponent moves so you can see what happened
- Score and status displayed at bottom

### 3. watch_agents_play.py

Watch two agents play Othello against each other (spectator mode).

**Features:**
- Watch games between any combination of agents
- Support for random, greedy, and trained agents
- Play multiple games with statistics
- Configurable delay between moves
- Color swapping for fair evaluation
- Detailed game statistics

**Usage:**

```bash
# Watch random vs greedy (1 game)
python scripts/watch_agents_play.py --agent1 random --agent2 greedy

# Watch 10 games with statistics
python scripts/watch_agents_play.py --agent1 random --agent2 greedy --num-games 10

# Watch with faster moves (0.5s delay)
python scripts/watch_agents_play.py --agent1 random --agent2 greedy --delay 0.5

# Watch trained agent vs greedy
python scripts/watch_agents_play.py --agent1-checkpoint /path/to/checkpoint --agent2 greedy

# Fair evaluation with color swapping
python scripts/watch_agents_play.py --agent1 random --agent2 greedy --num-games 10 --swap-colors

# Statistics only (no rendering)
python scripts/watch_agents_play.py --agent1 random --agent2 greedy --num-games 100 --no-render
```

**Statistics Displayed:**
- Win rates for each agent
- Draw rate
- Average final scores
- Average moves per game
- Individual game results (for ≤10 games)

## Requirements

### Console Scripts (play_human_vs_agent.py, watch_agents_play.py)
- Python 3.8+
- gymnasium
- numpy
- The Othello environment (`aip_rl.othello`)

### GUI Script (play_human_vs_agent_gui.py)
- All of the above, plus:
- pygame (install with `pip install pygame` or `poetry install --extras gui`)

### For Trained Agents
- ray[rllib]

## Examples

### Example 1: Test Your Skills (GUI)

```bash
# Install pygame first
poetry install --extras gui

# Play with beautiful graphics
python scripts/play_human_vs_agent_gui.py --opponent greedy
```

### Example 2: Test Your Skills (Console)

```bash
# Play as Black against greedy opponent
python scripts/play_human_vs_agent.py --opponent greedy
```

### Example 3: Evaluate Agent Performance

```bash
# Watch 50 games between random and greedy with statistics
python scripts/watch_agents_play.py \
    --agent1 random \
    --agent2 greedy \
    --num-games 50 \
    --swap-colors \
    --no-render
```

### Example 4: Test Trained Agent

```bash
# Watch trained agent play against greedy opponent
python scripts/watch_agents_play.py \
    --agent1-checkpoint /path/to/checkpoint \
    --agent2 greedy \
    --num-games 10 \
    --delay 0.5
```

## Tips

### For Human Players:
- Study the valid moves (marked with `*`) before making your move
- Try to control corners (0, 7, 56, 63) - they cannot be flipped
- Think ahead about how your opponent might respond
- The greedy opponent is challenging but beatable with strategy

### For Evaluation:
- Use `--swap-colors` to ensure fair comparison (eliminates first-move advantage)
- Use `--no-render` with many games for faster evaluation
- Compare win rates over at least 50-100 games for statistical significance
- Test trained agents against multiple opponent types

## Board Symbols

- `●` - Black piece
- `○` - White piece
- `.` - Empty square
- `*` - Valid move position

## Troubleshooting

**"Invalid move" error:**
- Make sure you're entering a number between 0-63
- Check that the position is marked with `*` (valid move)

**"No valid moves" message:**
- This is normal in Othello - your turn is automatically passed
- The game continues with the opponent's turn

**Checkpoint loading fails:**
- Ensure Ray RLlib is installed: `pip install ray[rllib]`
- Verify the checkpoint path is correct
- Check that the checkpoint is compatible with the current environment

## See Also

- `train_othello.py` - Train RL agents
- `test_ppo_training.py` - Test PPO training
- Main environment: `aip_rl/othello/env.py`
