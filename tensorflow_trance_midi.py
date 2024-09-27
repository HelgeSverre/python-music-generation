import os
import glob
from music21 import converter, instrument, note, chord, stream
import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess_midi(file_path):
    # Load the MIDI file
    midi = converter.parse(file_path)

    # Extract the notes and chords
    notes = []
    for element in midi.recurse():
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def create_sequences(notes, sequence_length=100):
    # Create input sequences and corresponding outputs
    input_sequences = []
    output_sequences = []
    for i in range(0, len(notes) - sequence_length, 1):
        input_sequences.append(notes[i : i + sequence_length])
        output_sequences.append(notes[i + sequence_length])

    return input_sequences, output_sequences


def build_model(vocab_size, sequence_length):
    model = keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, 100, input_length=sequence_length),
            keras.layers.LSTM(256, return_sequences=True),
            keras.layers.LSTM(256),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def generate_midi(model, start_sequence, vocab_to_int, int_to_vocab, num_notes=500):
    # Generate a sequence of notes
    current_sequence = start_sequence
    output_sequence = []

    for _ in range(num_notes):
        prediction_input = np.array([vocab_to_int[note] for note in current_sequence])
        prediction_input = prediction_input.reshape(1, -1)

        prediction = model.predict(prediction_input, verbose=0)
        predicted_note = int_to_vocab[np.argmax(prediction)]

        output_sequence.append(predicted_note)
        current_sequence = current_sequence[1:] + [predicted_note]

    # Create a music21 stream
    output_stream = stream.Stream()

    for item in output_sequence:
        if ("." in item) or item.isdigit():
            chord_notes = item.split(".")
            chord_obj = chord.Chord(chord_notes)
            output_stream.append(chord_obj)
        else:
            output_stream.append(note.Note(item))

    return output_stream


def main(input_folder, output_folder):
    # Load and preprocess MIDI files
    all_notes = []
    for file in glob.glob(os.path.join(input_folder, "*.mid")):
        # if not file.startswith("Above & Beyond"):
        if not file.startswith("Tiesto"):
            continue

        notes = preprocess_midi(file)
        all_notes.extend(notes)

    # Create vocabulary
    vocab = sorted(set(all_notes))
    vocab_size = len(vocab)
    vocab_to_int = {note: number for number, note in enumerate(vocab)}
    int_to_vocab = {number: note for number, note in enumerate(vocab)}

    # Create sequences
    input_sequences, output_sequences = create_sequences(all_notes)

    # Prepare data for training
    X = np.array([[vocab_to_int[note] for note in seq] for seq in input_sequences])
    y = np.array([vocab_to_int[seq] for seq in output_sequences])
    y = keras.utils.to_categorical(y, num_classes=vocab_size)

    # Build and train the model
    model = build_model(vocab_size, len(X[0]))
    model.fit(X, y, batch_size=64, epochs=50)

    # Generate new MIDI
    start_sequence = input_sequences[0]
    generated_stream = generate_midi(model, start_sequence, vocab_to_int, int_to_vocab)

    # Save the generated MIDI
    output_file = os.path.join(output_folder, "generated_trance.mid")
    generated_stream.write("midi", fp=output_file)
    print(f"Generated MIDI saved to {output_file}")


if __name__ == "__main__":
    input_folder = "trance_midis"  # Folder containing input MIDI files
    output_folder = "generated"  # Folder to save generated MIDI files
    main(input_folder, output_folder)
