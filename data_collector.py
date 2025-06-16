#!/usr/bin/env python3
"""
pgnmentor_lstm_dataset_converter.py

Convert all PGN files from a local PGN Mentor download into NumPy datasets
suitable for training a CNN+LSTM chess AI. Each game becomes one sample:
  - X_games.npy: an object array of shape (num_games,), where each entry
    is an array of shape (timesteps, 768) containing the board vectors
    for plies 0 through T−2.
  - y_labels.npy: a 2D int8 array of shape (num_games, 4096), where each
    entry is a one-hot vector encoding the last move of that game.

Usage:
    python pgnmentor_lstm_dataset_converter.py

Requires:
    pip install numpy chess
"""
import os
import chess
import chess.pgn
import numpy as np

def board_to_vector(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into a 768-dimensional one-hot vector:
    flattened 8×8×12 tensor (12 piece planes).
    """
    vec = np.zeros(64 * 12, dtype=np.int8)
    mapping = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            idx = mapping[piece.symbol()]
            vec[sq * 12 + idx] = 1
    return vec

def move_to_one_hot(move: chess.Move) -> np.ndarray:
    """
    One-hot encode a chess.Move into a 4096-dimensional vector:
    index = from_square * 64 + to_square.
    """
    one_hot = np.zeros(64 * 64, dtype=np.int8)
    idx = move.from_square * 64 + move.to_square
    one_hot[idx] = 1
    return one_hot

def main():
    PGN_DIR = 'pgnmentor_tournaments'  # change if your directory differs
    if not os.path.isdir(PGN_DIR):
        raise SystemExit(f"Directory '{PGN_DIR}' not found.")

    X_games = []    # each element: ndarray of shape (timesteps, 768)
    y_labels = []   # each element: ndarray of shape (4096,)
    game_count = 0

    print(f"Scanning directory: {PGN_DIR}")
    for fname in os.listdir(PGN_DIR):
        if not fname.lower().endswith('.pgn'):
            continue
        path = os.path.join(PGN_DIR, fname)
        print(f"Processing {fname}...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                frames = []
                moves = []
                for mv in game.mainline_moves():
                    # record current board
                    frames.append(board_to_vector(board))
                    moves.append(mv)
                    board.push(mv)
                if len(frames) >= 2:
                    # inputs: all but last ply
                    X_games.append(np.stack(frames[:-1], axis=0))
                    # label: one-hot of last move
                    y_labels.append(move_to_one_hot(moves[-1]))
                    game_count += 1

    if game_count == 0:
        raise SystemExit("No valid games found.")

    # Preserve per-game lengths
    X_games = np.array(X_games, dtype=object)
    # Stack labels into a 2D array (num_games, 4096)
    y_labels = np.stack(y_labels, axis=0)

    # Save datasets
    np.save('X_games.npy', X_games)
    np.save('y_labels.npy', y_labels)

    print(f"Converted {game_count} games.")
    print("Saved: X_games.npy and y_labels.npy")

if __name__ == '__main__':
    main()