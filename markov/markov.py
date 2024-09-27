import os
import pickle
from collections import defaultdict
import random
import mido

from utils import get_unique_filename


class MarkovMIDIGenerator:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))
        self.note_lengths = defaultdict(list)

    def analyze_midi_folder(self, folder_path, filter=None):
        for filename in os.listdir(folder_path):
            if filename.endswith(".mid"):
                try:
                    if filter is not None and filter not in filename:
                        continue

                    self.analyze_midi_file(os.path.join(folder_path, filename))
                    # print(f"✅  Analyzed: {filename}")
                except Exception as e:
                    print(f"❌  Error analyzing {filename}: {str(e)}")

    def analyze_midi_file(self, file_path):
        try:
            mid = mido.MidiFile(file_path)
        except Exception as e:
            print(f"Error reading MIDI file {file_path}: {str(e)}")
            return

        previous_note = None
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    if previous_note is not None:
                        self.model[previous_note][msg.note] += 1
                        self.note_lengths[previous_note].append(current_time)
                    previous_note = msg.note
                    current_time = 0
                current_time += msg.time

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((dict(self.model), dict(self.note_lengths)), f)
            print(f"💾 Model saved to: {file_path}")

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            model, note_lengths = pickle.load(f)
        self.model = defaultdict(lambda: defaultdict(int), model)
        self.note_lengths = defaultdict(list, note_lengths)
        print(f"📁 Model loaded from: {file_path}")

    def generate_sequence(self, length, start_note=None):
        if start_note is None:
            start_note = random.choice(list(self.model.keys()))

        sequence = [start_note]
        for _ in range(length - 1):
            current_note = sequence[-1]
            if current_note in self.model:
                next_note = random.choices(
                    list(self.model[current_note].keys()),
                    weights=list(self.model[current_note].values()),
                )[0]
                sequence.append(next_note)
            else:
                # If we reach a dead end, choose a random note
                sequence.append(random.choice(list(self.model.keys())))
        return sequence

    def generate_midi(self, sequence, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.Message("program_change", program=0, time=0))

        current_time = 0
        for note in sequence:
            # Note on
            track.append(
                mido.Message("note_on", note=note, velocity=64, time=current_time)
            )

            # Determine note length
            if note in self.note_lengths and self.note_lengths[note]:
                note_length = int(
                    sum(self.note_lengths[note]) / len(self.note_lengths[note])
                )
            else:
                note_length = 480  # Default to a quarter note if no data

            # Note off
            track.append(
                mido.Message("note_off", note=note, velocity=64, time=note_length)
            )

            current_time = 0  # Reset time for next note_on event

        mid.save(output_file)


# Usage example
for filter in ["Above", "Tiesto", "Armin", "Gareth"]:
    print("\n------------------------------------------------------------")
    print(f"Building markov model for '{filter.upper()}' MIDI Files")
    print("------------------------------------------------------------")
    generator = MarkovMIDIGenerator()
    generator.analyze_midi_folder("trance_midis/", filter=filter)
    path = get_unique_filename(f"markov_generated_{filter}.mid")
    generator.save_model(f"markov_model_trance_{filter}.pkl")
    sequence = generator.generate_sequence(100)
    generator.generate_midi(sequence, path)
    print(f" ===> Generating MIDI file: {path}")

    # generator.load_model('markov_model_trance.pkl')
