import os
import glob
import logging
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from music21 import converter, instrument, note, chord, stream, pitch, key

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configure TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available")
else:
    print("No GPU available, using CPU")

# Constants
SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001


def is_valid_midi(file_path):
    with open(file_path, "rb") as file:
        return file.read(4) == b"MThd"


def preprocess_midi(file_path):
    midi = converter.parse(file_path)
    parts = instrument.partitionByInstrument(midi)

    data = []
    for i, part in enumerate(parts):
        notes_to_parse = part.recurse()
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                data.append(
                    {
                        "track": i,
                        "pitch": element.pitch.midi,
                        "duration": element.duration.quarterLength,
                        "offset": element.offset,
                        "velocity": element.volume.velocity,
                    }
                )
            elif isinstance(element, chord.Chord):
                for chord_note in element:
                    data.append(
                        {
                            "track": i,
                            "pitch": chord_note.pitch.midi,
                            "duration": element.duration.quarterLength,
                            "offset": element.offset,
                            "velocity": element.volume.velocity,
                        }
                    )
    return data


def create_sequences(data, sequence_length):
    input_sequences = []
    output_sequences = []
    for i in range(0, len(data) - sequence_length, 1):
        input_sequences.append(data[i : i + sequence_length])
        output_sequences.append(data[i + sequence_length])
    return input_sequences, output_sequences


def encode_note(note, vocab_size):
    # Encode pitch, duration, velocity into a single integer
    pitch = note["pitch"]
    duration = min(int(note["duration"] * 4), 15)  # Quantize duration to 16 levels
    velocity = min(int(note["velocity"] / 8), 15)  # Quantize velocity to 16 levels
    return pitch + (duration * 128) + (velocity * 128 * 16)


def decode_note(encoded, vocab_size):
    velocity = (encoded // (128 * 16)) % 16
    duration = (encoded // 128) % 16
    pitch = encoded % 128
    return {"pitch": pitch, "duration": duration / 4, "velocity": velocity * 8}


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class MusicTransformer(keras.Model):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0.1,
        mlp_dropout=0.1,
    ):
        super(MusicTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout

        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ]
        self.mlp = keras.Sequential(
            [
                layers.Dense(mlp_units, activation="relu"),
                layers.Dropout(mlp_dropout),
                layers.Dense(vocab_size),
            ]
        )

    def call(self, inputs):
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        # Apply global average pooling to reduce the sequence dimension
        x = tf.reduce_mean(x, axis=1)
        x = self.mlp(x)
        return x


def create_model(vocab_size):
    model = MusicTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_transformer_blocks=6,
        mlp_units=1024,
        dropout=0.1,
        mlp_dropout=0.1,
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def generate_music(model, start_tokens, vocab_size, num_generate, temperature=1.0):
    start_tokens = [token for token in start_tokens]
    tokens_generated = []
    for _ in range(num_generate):
        pad_len = SEQUENCE_LENGTH - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len < 0:
            x = start_tokens[:SEQUENCE_LENGTH]
            sample_index = SEQUENCE_LENGTH - 1
        elif pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = np.array([x])
        y = model.predict(x, verbose=0)[0]
        y = y / temperature
        top_k = 10
        top_indices = np.argsort(y)[-top_k:]
        res = int(
            tf.random.categorical(tf.math.log([y[top_indices]]), num_samples=1).numpy()[
                0
            ][0]
        )
        pred = top_indices[res]
        tokens_generated.append(pred)
        start_tokens.append(pred)
    return tokens_generated


def apply_basic_music_theory(generated_notes, key_signature):
    # Define the scale based on the key signature
    scale = key_signature.getScale()
    scale_pitches = [p.midi for p in scale.getPitches()]

    corrected_notes = []
    for note in generated_notes:
        pitch = note["pitch"]
        # If the pitch is not in the scale, move it to the nearest scale pitch
        if pitch % 12 not in [p % 12 for p in scale_pitches]:
            nearest_pitch = min(scale_pitches, key=lambda x: abs(x - pitch))
            note["pitch"] = nearest_pitch
        corrected_notes.append(note)

    return corrected_notes


def save_model(model, filepath):
    model.save(filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath):
    model = tf.keras.models.load_model(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model


def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


def main(input_folder, output_folder, model_path):
    logging.info("Starting MIDI generation process")

    if not os.path.exists(input_folder):
        logging.error(f"Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    midi_files = glob.glob(os.path.join(input_folder, "*.mid"))
    if not midi_files:
        logging.error(
            f"No MIDI files found in '{input_folder}'. Please add some MIDI files and try again."
        )
        sys.exit(1)

    all_notes = []
    for file in midi_files:
        try:
            if not is_valid_midi(file):
                logging.warning(f"Skipping non-MIDI file: {file}")
                continue
            notes = preprocess_midi(file)
            all_notes.extend(notes)
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")

    logging.info(f"Processed {len(midi_files)} MIDI files")
    logging.debug(f"Total notes extracted: {len(all_notes)}")

    if len(all_notes) == 0:
        logging.error(
            "No valid notes extracted from MIDI files. Please check your input files."
        )
        sys.exit(1)

    # Create vocabulary
    unique_notes = set(tuple(note.items()) for note in all_notes)
    vocab_size = len(unique_notes)
    note_to_index = {note: i for i, note in enumerate(unique_notes)}
    index_to_note = {i: note for note, i in note_to_index.items()}

    logging.info(f"Vocabulary size: {vocab_size}")

    # Create sequences
    input_sequences, output_sequences = create_sequences(all_notes, SEQUENCE_LENGTH)

    if len(input_sequences) == 0:
        logging.error(
            "Not enough data to create sequences. Please provide more or longer MIDI files."
        )
        sys.exit(1)

    # Convert to numpy arrays and encode
    X = np.array(
        [
            [note_to_index[tuple(note.items())] for note in seq]
            for seq in input_sequences
        ]
    )
    y = np.array([note_to_index[tuple(note.items())] for note in output_sequences])

    # One-hot encode the labels
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    logging.debug(f"Input shape: {X.shape}, Output shape: {y_one_hot.shape}")

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logging.info("Loaded existing model.")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}. Will train a new model.")
            model = None
    else:
        model = None

    if model is None:
        try:
            model = create_model(vocab_size)
            logging.info("Model built. Starting training...")
            history = model.fit(
                X,
                y_one_hot,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=0.1,
                verbose=1,
            )
            logging.info("Model training completed")
            logging.debug(
                f"Final training accuracy: {history.history['accuracy'][-1]:.4f}"
            )

            # Save the trained model
            save_model(model, model_path)
        except Exception as e:
            logging.error(f"Error during model building or training: {str(e)}")
            sys.exit(1)

    try:
        # Generate new music
        seed_sequence = X[np.random.randint(0, len(X))]
        generated_indices = generate_music(model, seed_sequence, vocab_size, 500)
        generated_notes = [dict(index_to_note[idx]) for idx in generated_indices]

        # Apply basic music theory (assuming C major for simplicity)
        key_signature = key.Key("C")
        corrected_notes = apply_basic_music_theory(generated_notes, key_signature)

        # Convert generated notes to a music21 stream
        output_stream = stream.Stream()
        for note_data in corrected_notes:
            if isinstance(note_data["pitch"], list):
                # It's a chord
                chord_notes = [note.Note(pitch=p) for p in note_data["pitch"]]
                c = chord.Chord(chord_notes)
                c.duration.quarterLength = note_data["duration"]
                c.duration.quarterLength = note_data["duration"]
                c.volume.velocity = note_data["velocity"]
                output_stream.append(c)
            else:
                # It's a single note
                n = note.Note(pitch=note_data["pitch"])
                n.duration.quarterLength = note_data["duration"]
                n.volume.velocity = note_data["velocity"]
                output_stream.append(n)

        # Save the generated MIDI
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        base_output_file = os.path.join(
            output_folder, "generated_trance_transformer.mid"
        )
        output_file = get_unique_filename(base_output_file)
        output_stream.write("midi", fp=output_file)
        logging.info(f"Generated MIDI saved to {output_file}")
    except Exception as e:
        logging.error(f"Error during MIDI generation or saving: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    input_folder = "trance_midis"  # Folder containing input MIDI files
    output_folder = "output"  # Folder to save generated MIDI files
    model_path = "trance_transformer_model"  # Path to save/load the model
    main(input_folder, output_folder, model_path)
