import os
import glob
import logging
import sys
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow import keras

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_midi(file_path):
    logging.debug(f"Preprocessing MIDI file: {file_path}")
    # Load the MIDI file
    midi = converter.parse(file_path)

    # Extract the notes and chords
    notes = []
    for element in midi.recurse():
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    logging.debug(f"Extracted {len(notes)} notes/chords from {file_path}")
    return notes


def create_sequences(notes, sequence_length=100):
    logging.debug(f"Creating sequences with length {sequence_length}")
    # Create input sequences and corresponding outputs
    input_sequences = []
    output_sequences = []
    for i in range(0, len(notes) - sequence_length, 1):
        input_sequences.append(notes[i : i + sequence_length])
        output_sequences.append(notes[i + sequence_length])

    logging.debug(f"Created {len(input_sequences)} input sequences")
    return input_sequences, output_sequences


def get_unique_filename(base_filename):
    """
    Generate a unique filename by appending a number if the file already exists.
    """
    if not os.path.exists(base_filename):
        return base_filename

    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


def positional_encoding(position, d_model):
    angle_rads = np.arange(d_model)[np.newaxis, :] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(d_model)]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):  # Changed this line
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)  # Changed this line
        return x


def build_transformer(
    vocab_size,
    d_model=256,
    num_layers=4,
    num_heads=8,
    dff=1024,
    maximum_position_encoding=10000,
):
    inputs = tf.keras.Input(shape=(None,))
    enc_padding_mask = None
    encoder = Encoder(
        num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding
    )
    enc_output = encoder(inputs, training=True, mask=enc_padding_mask)

    # Add this line to get the last time step of the sequence
    last_time_step = enc_output[:, -1, :]

    # Change this line to use the last time step
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(last_time_step)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def generate_midi(
    model, start_sequence, vocab_to_int, int_to_vocab, num_notes=500, temperature=1.0
):
    current_sequence = start_sequence
    output_sequence = []

    for _ in range(num_notes):
        prediction_input = np.array([vocab_to_int[note] for note in current_sequence])
        prediction_input = prediction_input.reshape(1, -1)

        predictions = model.predict(prediction_input, verbose=0)[0]
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted_note = int_to_vocab[np.argmax(predictions)]

        output_sequence.append(predicted_note)
        current_sequence = current_sequence[1:] + [predicted_note]

    output_stream = stream.Stream()

    for item in output_sequence:
        if ("." in item) or item.isdigit():
            chord_notes = item.split(".")
            chord_obj = chord.Chord(chord_notes)
            output_stream.append(chord_obj)
        else:
            output_stream.append(note.Note(item))

    return output_stream


def save_model(model, filepath):
    """Save the trained model to a file."""
    model.save(filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a trained model from a file."""
    model = tf.keras.models.load_model(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model


def main(input_folder, output_folder, model_path):
    logging.info("Starting MIDI generation process")

    # Check if input folder exists and contains MIDI files
    if not os.path.exists(input_folder):
        logging.error(f"Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    midi_files = glob.glob(os.path.join(input_folder, "*.mid"))
    if not midi_files:
        logging.error(
            f"No MIDI files found in '{input_folder}'. Please add some MIDI files and try again."
        )
        sys.exit(1)

    # Load and preprocess MIDI files
    all_notes = []
    for file in midi_files:
        try:
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
    vocab = sorted(set(all_notes))
    vocab_size = len(vocab)
    vocab_to_int = {note: number for number, note in enumerate(vocab)}
    int_to_vocab = {number: note for number, note in enumerate(vocab)}
    logging.info(f"Vocabulary size: {vocab_size}")

    # Create sequences
    input_sequences, output_sequences = create_sequences(all_notes)

    if len(input_sequences) == 0:
        logging.error(
            "Not enough data to create sequences. Please provide more or longer MIDI files."
        )
        sys.exit(1)

    # Prepare data for training
    X = np.array([[vocab_to_int[note] for note in seq] for seq in input_sequences])
    y = np.array([vocab_to_int[seq] for seq in output_sequences])
    logging.debug(f"Input shape: {X.shape}, Output shape: {y.shape}")

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
            model = build_transformer(vocab_size)
            logging.info("Model built. Starting training...")
            history = model.fit(X, y, batch_size=256, epochs=50, verbose=1)
            logging.info("Model training completed")
            logging.debug(
                f"Final training accuracy: {history.history['accuracy'][-1]:.4f}"
            )

            # Save the trained model
            save_model(model, model_path)
        except Exception as e:
            logging.error(f"Error during model building or training: {str(e)}")
            sys.exit(1)

    # Generate new MIDI
    try:
        start_sequence = input_sequences[0]
        generated_stream = generate_midi(
            model, start_sequence, vocab_to_int, int_to_vocab
        )

        # Save the generated MIDI
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        base_output_file = os.path.join(
            output_folder, "generated_trance_transformer.mid"
        )
        output_file = get_unique_filename(base_output_file)
        generated_stream.write("midi", fp=output_file)
        logging.info(f"Generated MIDI saved to {output_file}")
    except Exception as e:
        logging.error(f"Error during MIDI generation or saving: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    input_folder = "trance_midis"  # Folder containing input MIDI files
    output_folder = "output"  # Folder to save generated MIDI files
    model_path = "trance_transformer_model"  # Path to save/load the model
    main(input_folder, output_folder, model_path)
