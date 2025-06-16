import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os


def pad_game_sequences(games, max_timesteps):
    zero_frame = np.zeros((8, 8, 12), dtype=np.uint8)
    padded = []
    for g in games:
        # g is now (T,8,8,12)
        if g.shape[0] < max_timesteps:
            pad_count = max_timesteps - g.shape[0]
            # Pad at the beginning so the last board state is preserved
            g_padded = np.concatenate(
                [np.repeat(zero_frame[None], pad_count, axis=0), g], axis=0)
        else:
            # Keep the final `max_timesteps` positions before the last move
            g_padded = g[-max_timesteps:]
        padded.append(g_padded)
    return np.stack(padded, axis=0)


def create_chess_nn_model(max_timesteps, vocab_size):
    model = models.Sequential([
        layers.Input(shape=(max_timesteps, 8, 8, 12)),
        layers.TimeDistributed(layers.Conv2D(
            16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))),
        layers.TimeDistributed(layers.BatchNormalization()),

        layers.TimeDistributed(layers.Conv2D(
            16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))),
        layers.TimeDistributed(layers.BatchNormalization()),

        layers.TimeDistributed(layers.Conv2D(
            64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))),
        layers.TimeDistributed(layers.BatchNormalization()),

        layers.TimeDistributed(layers.GlobalAveragePooling2D()),

        layers.Masking(mask_value=0.0),

        layers.LSTM(64, name='temporal_lstm'),

        layers.Dropout(0.5),
        layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_save_model(X_train, y_train, model_path='chess_ai_model.keras'):
    """Load an existing model if possible; otherwise create a new one."""
    model = None
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Loaded existing model.")
        except Exception as e:
            print(f"[WARN] Could not load model '{model_path}': {e}")

    if model is None:
        print("Creating new model.")
        max_timesteps = X_train.shape[1]
        vocab_size = int(y_train.max()) + 1
        model = create_chess_nn_model(max_timesteps, vocab_size)

    model.summary()

    checkpoint = ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )
    lr_schedule = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
    )

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        shuffle=True,
        callbacks=[checkpoint, lr_schedule, early_stop]
    )
    print("Training done.")
    return model


if __name__ == "__main__":
    print("Loading training data...")
    try:
        # load per-game sequences of 768-dim vectors
        raw_X = np.load('training_data/X_games.npy', allow_pickle=True)
        raw_y = np.load('training_data/y_labels.npy', allow_pickle=True)

        raw_X = raw_X[:100000]
        raw_y = raw_y[:100000]

        # --- NEW: reshape each game from (T,768) → (T,8,8,12) ---
        games = [g.reshape(-1, 8, 8, 12) for g in raw_X]

        max_timesteps = 30
        X_data = pad_game_sequences(games, max_timesteps)
        # Convert one-hot labels to integer class indices
        if raw_y.ndim > 1:
            y_data = raw_y.argmax(axis=1).astype(np.int32)
        else:
            y_data = raw_y.astype(np.int32)

        print(f"Data loaded. X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")

        train_and_save_model(X_data, y_data)

    except FileNotFoundError:
        print("Error: Training data not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
