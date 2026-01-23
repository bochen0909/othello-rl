# Othello GUI Features

## Overview

The GUI version (`play_human_vs_agent_gui.py`) provides a beautiful graphical interface for playing Othello using Pygame.

## Installation

```bash
# Install pygame
pip install pygame

# Or with poetry
poetry install --extras gui
```

## Visual Features

### Game Board
- **8x8 green board** with clear grid lines
- **Black and white pieces** rendered as circles with borders
- **Yellow dots** indicate valid moves
- **Smooth rendering** at 30 FPS

### Information Panel
- **Real-time score** showing piece counts for both players
- **Player color** display (Black or White)
- **Opponent type** (random, greedy, or trained agent)
- **Status messages** for game state and turn information
- **Win/Loss/Draw** announcements with color coding
- **Orange border** highlights opponent's last move for 1.5 seconds

### User Interface
- **Click-to-move**: Simply click on any yellow dot to make your move
- **Visual feedback**: Immediate board updates after each move
- **No typing required**: All interaction via mouse clicks
- **Keyboard shortcuts**:
  - `ESC` or `Q`: Quit game
  - `R`: Restart game (after game ends)

## Usage Examples

### Play Against Random Opponent
```bash
python scripts/play_human_vs_agent_gui.py --opponent random
```

### Play Against Greedy Opponent
```bash
python scripts/play_human_vs_agent_gui.py --opponent greedy
```

### Play as White
```bash
python scripts/play_human_vs_agent_gui.py --opponent greedy --human-color white
```

### Play Against Trained Agent
```bash
python scripts/play_human_vs_agent_gui.py --checkpoint /path/to/checkpoint
```

## Gameplay Flow

1. **Game starts** with initial board position
2. **Valid moves** appear as yellow dots
3. **Click a yellow dot** to place your piece
4. **Board updates** showing your move
5. **800ms pause** - opponent "thinks"
6. **Opponent moves** automatically
7. **Orange border** highlights opponent's move for 1.5 seconds
8. **Score updates** in real-time
9. **Game ends** when no more moves available
10. **Press R** to play again or **ESC** to quit

## Visual Layout

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│          8x8 Othello Board              │
│        (with pieces and dots)           │
│                                         │
│                                         │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ ● Black: 12  ○ White: 8                 │
│                                         │
│ You are: BLACK                          │
│ Opponent: greedy                        │
│                                         │
│ Your turn!                              │
└─────────────────────────────────────────┘
```

## Color Scheme

- **Board**: Dark green background with lighter green cells
- **Grid lines**: Black borders
- **Black pieces**: Black circles with gray borders
- **White pieces**: White circles with gray borders
- **Valid moves**: Yellow dots
- **Last move highlight**: Orange border (fades after 1.5 seconds)
- **Info panel**: Light gray background
- **Win message**: Green text
- **Lose message**: Red text

## Performance

- Runs at 30 FPS for smooth visuals
- Minimal CPU usage during idle
- Instant response to clicks
- No lag or stuttering

## Advantages Over Console Version

1. **Easier to use**: Click instead of typing numbers
2. **Better visualization**: See the board at a glance
3. **More intuitive**: Valid moves clearly marked
4. **Visual feedback**: Opponent's last move is highlighted
5. **Paced gameplay**: 800ms delay lets you see what happened
6. **More engaging**: Visual feedback and smooth animations
7. **Less error-prone**: Can't click invalid moves

## Tips for Playing

- **Corner strategy**: Corners (positions 0, 7, 56, 63) are valuable
- **Edge control**: Edges are harder to flip
- **Mobility**: Keep your options open
- **Tempo**: Sometimes passing moves to opponent is strategic
- **Endgame**: Count pieces carefully in final moves

## Troubleshooting

### Pygame not installed
```
Error: pygame is required for the GUI version.
```
**Solution**: Run `pip install pygame` or `poetry install --extras gui`

### Window doesn't appear
- Check if pygame is properly installed
- Try running from terminal/command line
- Ensure display is available (not SSH without X forwarding)

### Slow performance
- Close other applications
- Update graphics drivers
- Reduce window size (modify CELL_SIZE in code)

### Can't click moves
- Ensure you're clicking on yellow dots (valid moves)
- Check if it's your turn (status message shows "Your turn!")
- Try clicking center of the cell

## Future Enhancements

Possible improvements for future versions:
- Move history display
- Undo/redo functionality
- Save/load game states
- Multiple difficulty levels
- Sound effects
- Animations for piece flips
- Timer/clock display
- Move suggestions/hints
- Game statistics tracking
