# Chess AI

A simple neural network-based chess engine.

## Training
1. Put `X_games.npy` and `y_labels.npy` in a `training_data/` directory.
2. Run `python model_trainer.py` to train the model. If loading a saved
   model fails with a message about `model.weights.h5`, remove any existing

   `chess_ai_model.keras` and `chess_ai_model.weights.h5` files and rerun
   the script.
   `chess_ai_model.keras` file and rerun the script.

## Playing
Run `python chess_game.py` after training to play against the AI.
