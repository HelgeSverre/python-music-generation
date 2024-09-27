import os
import glob
from music21 import converter, note, chord, stream
import matplotlib.pyplot as plt


def is_valid_midi(file_path):
    try:
        with open(file_path, "rb") as file:
            return file.read(4) == b"MThd"
    except Exception as e:
        print(f"Error validating MIDI file {file_path}: {str(e)}")
        return False


def analyze_midi(file_path):
    try:
        midi = converter.parse(file_path)
        parts = midi.getElementsByClass(stream.Part)

        pitches = []
        durations = []
        num_notes = 0
        num_chords = 0

        for part in parts:
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    num_notes += 1
                    pitches.append(element.pitch.midi)
                    durations.append(element.duration.quarterLength)
                elif isinstance(element, chord.Chord):
                    num_chords += 1
                    for chord_note in element:
                        num_notes += 1
                        pitches.append(chord_note.pitch.midi)
                        durations.append(element.duration.quarterLength)

        return {
            "num_notes": num_notes,
            "num_chords": num_chords,
            "pitches": pitches,
            "durations": durations,
        }
    except Exception as e:
        print(f"Error processing MIDI file {file_path}: {str(e)}")
        return None


def calculate_suitability_score(midi_data):
    # Criteria to score the MIDI file
    if midi_data is None or midi_data["num_notes"] == 0:
        return 0  # Invalid or empty file, unsuitable for training

    num_notes = midi_data["num_notes"]
    num_chords = midi_data["num_chords"]
    pitches = midi_data["pitches"]
    durations = midi_data["durations"]

    # 1. Pitch Variety (More variety = better)
    pitch_variety_score = len(set(pitches)) / num_notes if num_notes > 0 else 0

    # 2. Duration Variety (More variety = better)
    duration_variety_score = len(set(durations)) / num_notes if num_notes > 0 else 0

    # 3. Chord Content (More polyphony = better)
    chord_ratio_score = num_chords / num_notes if num_notes > 0 else 0

    # 4. Note Density (Optimal density = better)
    # For note density, we set a reasonable range for num_notes. Files with too few or too many notes might be unsuitable.
    if num_notes < 50:
        note_density_score = 0.2  # Too few notes
    elif num_notes > 1000:
        note_density_score = 0.2  # Too many notes
    else:
        note_density_score = 1.0  # Suitable number of notes

    # Weighted sum to calculate the final suitability score
    suitability_score = (
        0.4 * pitch_variety_score
        + 0.3 * duration_variety_score
        + 0.2 * chord_ratio_score
        + 0.1 * note_density_score
    )

    return suitability_score


def evaluate_midi_files(input_folder):
    midi_files = glob.glob(os.path.join(input_folder, "*.mid"))

    scores = {}
    for file in midi_files:
        if not is_valid_midi(file):
            print(f"Skipping invalid MIDI file: {file}")
            continue

        print(f"Analyzing {file}...")
        midi_data = analyze_midi(file)
        suitability_score = calculate_suitability_score(midi_data)
        scores[file] = suitability_score

    return scores


def plot_scores(scores):
    file_names = list(scores.keys())
    file_scores = list(scores.values())

    plt.figure(figsize=(10, 6))
    plt.barh(file_names, file_scores)
    plt.xlabel("Suitability Score")
    plt.ylabel("MIDI File")
    plt.title("Suitability Scores for MIDI Files")
    plt.show()


def main(input_folder):
    print(f"Evaluating MIDI files in {input_folder}...")
    scores = evaluate_midi_files(input_folder)

    # Print results
    for file, score in scores.items():
        print(f"{file}: Suitability Score = {score:.2f}")

    # Optionally, plot the scores
    plot_scores(scores)


if __name__ == "__main__":
    input_folder = "../trance_midis"  # Update this path to your input folder
    main(input_folder)
