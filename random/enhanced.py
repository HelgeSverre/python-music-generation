import random
import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt
import numpy as np
from colorama import init, Fore, Style

from pygame_generate import play_midi
from utils import get_unique_filename

# Initialize colorama
init(autoreset=True)

# MIDI Settings
BPM = 138
TICKS_PER_BEAT = 480
NOTE_LENGTH_IN_TICKS = int(TICKS_PER_BEAT / 4)  # 16th note
TEMPO = mido.bpm2tempo(BPM)

# Define scales and chord intervals
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 11]
MELODIC_MINOR_SCALE = [0, 2, 3, 5, 7, 9, 11]

CHORD_INTERVALS_MINOR = [0, 3, 7]
CHORD_INTERVALS_MAJOR = [0, 4, 7]

# Trance-specific chord progressions
TRANCE_PROGRESSIONS = {
    "classic_uplifting": "i-VI-VII-i",
    "emotional_trance": "i-III-VI-VII",
    "epic_trance": "i-VI-III-VII",
    "melancholic_trance": "i-v-VI-iv",
    "driving_trance": "i-VII-VI-V",
    "euphoric_trance": "i-III-IV-VI",
    "dark_trance": "i-v-VI-VII",
    "progressive_trance": "i-VI-v-IV",
}

PROGRESSION_MAP = {
    "i": 0,
    "II": 2,
    "III": 3,
    "iv": 5,
    "v": 7,
    "VI": 8,
    "VII": 10,
    "I": 0,
    "ii": 2,
    "iii": 3,
    "IV": 5,
    "V": 7,
    "vi": 8,
    "vii": 10,
}


def generate_progressions_from_string(root_note, progression_str):
    progression = []
    bassline_root_notes = []

    print(Fore.GREEN + f"Generating progression: {progression_str}")

    for symbol in progression_str.split("-"):
        degree = PROGRESSION_MAP[symbol]
        if symbol.isupper():
            chord = [root_note + degree + i for i in CHORD_INTERVALS_MAJOR]
        else:
            chord = [root_note + degree + i for i in CHORD_INTERVALS_MINOR]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def generate_motif_from_chord(chord, scale, length=8):
    print(Fore.CYAN + f"Generating motif from chord: {chord}")
    return [random.choice(chord + scale) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    print(Fore.YELLOW + f"Applying {variation_type} variation to motif: {motif}")

    if variation_type == "transpose":
        interval = random.choice([-2, -1, 1, 2])
        return [(note + interval) for note in motif]
    elif variation_type == "invert":
        first_note = motif[0]
        return [(first_note - (note - first_note)) for note in motif]
    elif variation_type == "reverse":
        return motif[::-1]
    elif variation_type == "rhythmic":
        return [note if random.random() > 0.3 else 0 for note in motif]
    else:
        return motif


def create_melody_from_progression(progression, scale, total_beats):
    melody = []
    print(Fore.MAGENTA + f"Creating melody from progression: {progression}")
    while len(melody) < total_beats:
        chord = progression[len(melody) // 32 % len(progression)]
        motif = generate_motif_from_chord(chord, scale)
        if len(melody) > 0:
            variation_type = random.choice(
                ["transpose", "invert", "reverse", "rhythmic", "none"]
            )
            motif = apply_variation(motif, variation_type=variation_type)
        melody.extend(motif)

    return melody[:total_beats]  # Trim to exact length if necessary


def create_bassline(bassline_root_notes, total_beats):
    print(Fore.BLUE + "Creating bassline...")
    bassline = []
    patterns = [
        lambda note: [note - 12, 0, note - 12, 0] * 2,  # Standard offbeat
        lambda note: [note - 12, 0, 0, note - 12, 0, 0, note - 12, 0],  # Syncopated
        lambda note: [note - 12, note, note - 12, 0] * 2,  # Octave jump
    ]

    while len(bassline) < total_beats:
        root_note = bassline_root_notes[len(bassline) // 32 % len(bassline_root_notes)]
        pattern = random.choice(patterns)(root_note)
        bassline.extend(pattern)

    return bassline[:total_beats]  # Trim to exact length if necessary


def create_pad(progression, total_beats):
    print(Fore.LIGHTMAGENTA_EX + "Creating pad...")
    pad = []
    while len(pad) * 8 < total_beats:
        chord = progression[len(pad) % len(progression)]
        pad.append(chord)
    return pad


def create_midi(melody, bassline, pad, filename="classic_trance_melody.mid"):
    mid = MidiFile()
    print(Fore.LIGHTBLUE_EX + "Creating MIDI file...")

    # Melody track
    melody_track = MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage("set_tempo", tempo=TEMPO))

    for note in melody:
        if note > 0:
            melody_track.append(Message("note_on", note=note, velocity=64, time=0))
            melody_track.append(
                Message("note_off", note=note, velocity=64, time=NOTE_LENGTH_IN_TICKS)
            )
        else:
            melody_track.append(
                Message("note_off", note=0, velocity=0, time=NOTE_LENGTH_IN_TICKS)
            )

    # Bassline track
    bassline_track = MidiTrack()
    mid.tracks.append(bassline_track)

    for note in bassline:
        if note > 0:
            bassline_track.append(Message("note_on", note=note, velocity=80, time=0))
            bassline_track.append(
                Message("note_off", note=note, velocity=80, time=NOTE_LENGTH_IN_TICKS)
            )
        else:
            bassline_track.append(
                Message("note_off", note=0, velocity=0, time=NOTE_LENGTH_IN_TICKS)
            )

    # Pad track
    pad_track = MidiTrack()
    mid.tracks.append(pad_track)

    for chord in pad:
        for note in chord:
            pad_track.append(Message("note_on", note=note, velocity=40, time=0))
        for note in chord:
            pad_track.append(
                Message(
                    "note_off", note=note, velocity=40, time=NOTE_LENGTH_IN_TICKS * 8
                )
            )

    mid.save(filename)
    print(Fore.GREEN + Style.BRIGHT + f"MIDI file saved as {filename}")


def plot_melody_and_bass(melody, bassline, pad):
    print(Fore.LIGHTWHITE_EX + "Plotting melody, bassline, and pad...")
    time_stamps = np.arange(len(melody)) * 0.25  # 16th notes time stamps
    plt.figure(figsize=(15, 8))

    plt.step(
        time_stamps, melody, where="mid", label="Trance Melody", marker="o", alpha=0.7
    )
    plt.step(
        time_stamps, bassline, where="mid", label="Bassline", marker="x", alpha=0.7
    )

    pad_notes = [note for chord in pad for note in chord]
    pad_times = np.repeat(np.arange(0, len(pad) * 2, 2), 3)
    plt.scatter(pad_times, pad_notes, label="Pad", alpha=0.3, s=50)

    plt.xlabel("Beats")
    plt.ylabel("MIDI Note")
    plt.title("Classic Trance Melody with Bassline and Pad")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("classic_trance_visualization.png")
    plt.close()


def main():
    keys = {
        f"{note}{octave}": midi_num
        for octave in [3, 4]
        for note, midi_num in zip(
            ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
            range(57 + (octave - 3) * 12, 69 + (octave - 3) * 12),
        )
    }
    root_note = random.choice(list(keys.values()))
    progression_choice = random.choice(list(TRANCE_PROGRESSIONS.keys()))
    progression_str = TRANCE_PROGRESSIONS[progression_choice]

    print(Fore.GREEN + f"Using progression: {progression_str} ({progression_choice})")

    progression, bassline_root_notes = generate_progressions_from_string(
        root_note, progression_str
    )
    scale = [
        root_note + interval
        for interval in random.choice(
            [MINOR_SCALE, HARMONIC_MINOR_SCALE, MELODIC_MINOR_SCALE]
        )
    ]

    total_bars = 32  # A typical length for a trance section
    total_beats = total_bars * 8  # 8 sixteenth notes per bar

    melody = create_melody_from_progression(progression, scale, total_beats)
    bassline = create_bassline(bassline_root_notes, total_beats)
    pad = create_pad(progression, total_beats)

    midi_filename = get_unique_filename("generated/classic_trance.mid")
    create_midi(melody, bassline, pad, filename=midi_filename)
    plot_melody_and_bass(melody, bassline, pad)

    print(Fore.GREEN + Style.BRIGHT + "Classic trance melody generated successfully!")
    print(Fore.YELLOW + f"MIDI file: {midi_filename}")
    print(Fore.YELLOW + "Visualization: classic_trance_visualization.png")

    play_midi(midi_filename)


if __name__ == "__main__":
    main()
