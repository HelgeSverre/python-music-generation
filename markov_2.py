import os
import random
import mido
import logging

from utils import get_unique_filename

logging.basicConfig(level=logging.INFO)

class MarkovMIDIGenerator:
    def __init__(self):
        self.model = {}
        self.note_properties = {}

    def analyze_midi_folder(self, folder_path, filter=None):
        analyzed_files = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".mid") and (filter is None or filter in filename):
                try:
                    self.analyze_midi_file(os.path.join(folder_path, filename))
                    analyzed_files += 1
                    logging.info(f"Successfully analyzed: {filename}")
                except Exception as e:
                    logging.error(f"Error analyzing {filename}: {str(e)}")
        logging.info(f"Total files analyzed: {analyzed_files}")

    def analyze_midi_file(self, file_path):
        try:
            mid = mido.MidiFile(file_path)
        except Exception as e:
            logging.error(f"Error reading MIDI file {file_path}: {str(e)}")
            return

        previous_notes = []
        current_time = 0
        for msg in mido.merge_tracks(mid.tracks):
            if msg.type == "note_on" and msg.velocity > 0:
                if len(previous_notes) >= 2:
                    key = tuple(previous_notes[-2:])
                    self.model.setdefault(key, {})
                    self.model[key][msg.note] = self.model[key].get(msg.note, 0) + 1
                    self.note_properties.setdefault(key, []).append((current_time, msg.velocity))
                previous_notes.append(msg.note)
                current_time = 0
            current_time += msg.time

    def generate_sequence(self, length):
        if not self.model:
            raise ValueError("No data available. Cannot generate sequence.")

        start_notes = random.choice(list(self.model.keys()))
        sequence = list(start_notes)

        for _ in range(length - len(start_notes)):
            current_key = tuple(sequence[-2:])
            if current_key in self.model:
                next_note = random.choices(
                    list(self.model[current_key].keys()),
                    weights=list(self.model[current_key].values())
                )[0]
            else:
                next_note = random.choice(list(self.model.keys())[0])
            sequence.append(next_note)
        return sequence

    def generate_midi(self, sequence, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.Message("program_change", program=0, time=0))

        current_time = 0
        for i in range(len(sequence)):
            key = tuple(sequence[max(0, i-2):i])
            note = sequence[i]
            if key in self.note_properties:
                time, velocity = random.choice(self.note_properties[key])
            else:
                time, velocity = 480, 64

            track.append(mido.Message("note_on", note=note, velocity=velocity, time=current_time))
            track.append(mido.Message("note_off", note=note, velocity=velocity, time=time))
            current_time = 0

        mid.save(output_file)

if __name__ == "__main__":
    generator = MarkovMIDIGenerator()
    generator.analyze_midi_folder("trance_midis/", filter="Above")

    if not generator.model:
        logging.error("No valid data extracted from MIDI files. Cannot generate sequence.")
    else:
        try:
            sequence = generator.generate_sequence(50)
            output_file = get_unique_filename("generated_trance_above.mid")
            generator.generate_midi(sequence, output_file)
            logging.info(f"MIDI file generated: {output_file}")
        except ValueError as e:
            logging.error(f"Error generating sequence: {str(e)}")