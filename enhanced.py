import random
import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt
import pygame
import time
import numpy as np
from colorama import init, Fore, Style

from trance_3 import play_midi
from utils import get_unique_filename

# Initialize colorama
init(autoreset=True)

# MIDI Settings
BPM = 138
TICKS_PER_BEAT = 480
NOTE_LENGTH_IN_TICKS = int(TICKS_PER_BEAT / 4)  # 16th note
TEMPO = mido.bpm2tempo(BPM)

# Initialize pygame mixer for MIDI playback
pygame.mixer.init()

# Define scales and chord intervals
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
CHORD_INTERVALS_MAJOR = [0, 4, 7]
CHORD_INTERVALS_MINOR = [0, 3, 7]

# Predefined chord progressions
PROGRESSION_MAP = {
    "I": 0,
    "ii": 2,
    "iii": 4,
    "IV": 5,
    "V": 7,
    "vi": 9,
    "vii°": 11,
}

PREDEFINED_PROGRESSIONS = {
    "classic_pop": "I-V-vi-IV",
    "ballad": "I-vi-IV-V",
    "pachelbel": "I-V-vi-iii-IV-I-IV-V",
    "blues": "I-IV-V",
    "axis_of_awesome": "I-V-vi-IV",
    "popular_combination": "I-IV-vi-V",
    "epic_hits": "I-vi-IV-V",
    "dramatic_minor_major": "i-VII-III-VI",
    "twist_on_classic_minor": "i-V-vi-IV",
    "bold_uplifting": "i-III-VII-VI",
    "classic_twist_with_iii": "I-V-vi-iii-IV",
    "subtle_tension": "I-V-vi-V",
    "unique_tension": "IV-I-ii",
    "unusual_vi_start": "vi-V-IV-V",
    "classic_minor": "i-VI-III-VII",
}


def generate_progressions_from_string(root_note, progression_str, mode="major"):
    intervals = CHORD_INTERVALS_MAJOR if mode == "major" else CHORD_INTERVALS_MINOR
    progression = []
    bassline_root_notes = []

    print(Fore.GREEN + f"Generating progression: {progression_str}")

    for symbol in progression_str.split("-"):
        degree = PROGRESSION_MAP[symbol]
        chord = [root_note + degree + i for i in intervals]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def generate_motif_from_chord(chord, length=8):
    print(Fore.CYAN + f"Generating motif from chord: {chord}")
    return [random.choice(chord) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    print(Fore.YELLOW + f"Applying {variation_type} variation to motif: {motif}")

    if variation_type == "transpose":
        interval = random.choice([-2, -1, 0, 1, 2])
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


def create_melody_from_progression(progression, bars=8):
    melody = []
    print(Fore.MAGENTA + f"Creating melody from progression: {progression}")
    for bar in range(bars):
        chord = progression[bar % len(progression)]
        motif = generate_motif_from_chord(chord)
        if bar > 0:
            variation_type = random.choice(
                ["transpose", "invert", "reverse", "rhythmic", "none"]
            )
            motif = apply_variation(motif, variation_type=variation_type)
        melody.extend(motif)
    return melody


def create_bassline(bassline_root_notes, bars=8):
    print(Fore.BLUE + "Creating bassline...")
    bassline = []
    for bar in range(bars):
        root_note = bassline_root_notes[bar % len(bassline_root_notes)]
        pattern = random.choice(
            [
                [root_note - 12, 0, root_note - 12, 0] * 2,  # Standard offbeat
                [
                    root_note - 12,
                    0,
                    0,
                    root_note - 12,
                    0,
                    0,
                    root_note - 12,
                    0,
                ],  # Syncopated
                [root_note - 12, 0, 0, 0] * 2,  # Sparse
            ]
        )
        bassline.extend(pattern)
    return bassline


def create_pad(progression, bars=8):
    print(Fore.LIGHTMAGENTA_EX + "Creating pad...")
    pad = []
    for bar in range(bars):
        chord = progression[bar % len(progression)]
        pad.extend([chord] * 8)  # Hold each chord for a full bar
    return pad


def create_midi(melody, bassline, pad, filename="enhanced_trance_melody.mid"):
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
    plt.title("Enhanced Trance Melody with Bassline and Pad")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("trance_visualization.png")
    plt.close()


def main():
    keys = {
        f"{note}{octave}": midi_num
        for octave in [3, 4]
        for note, midi_num in zip(
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            range(48 + (octave - 3) * 12, 60 + (octave - 3) * 12),
        )
    }
    root_note = random.choice(list(keys.values()))
    progression_choice = random.choice(list(PREDEFINED_PROGRESSIONS.keys()))
    progression_str = PREDEFINED_PROGRESSIONS[progression_choice]

    print(Fore.GREEN + f"Using progression: {progression_str} ({progression_choice})")

    progression, bassline_root_notes = generate_progressions_from_string(
        root_note, progression_str
    )
    melody = create_melody_from_progression(progression, bars=8)
    bassline = create_bassline(bassline_root_notes, bars=8)
    pad = create_pad(progression, bars=8)

    midi_filename = get_unique_filename("generated/enhanced_trance.mid")
    create_midi(melody, bassline, pad, filename=midi_filename)
    plot_melody_and_bass(melody, bassline, pad)

    print(Fore.GREEN + Style.BRIGHT + "Enhanced trance melody generated successfully!")
    print(Fore.YELLOW + f"MIDI file: {midi_filename}")
    print(Fore.YELLOW + "Visualization: trance_visualization.png")

    play_midi(midi_filename)


if __name__ == "__main__":
    main()
