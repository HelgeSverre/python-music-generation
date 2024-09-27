import random
import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt
import pygame
import time

from colorama import init, Fore, Style
from utils import get_unique_filename

# Initialize colorama
init(autoreset=True)

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Initialize pygame mixer for MIDI playback
pygame.mixer.init()

# Define Major and Minor chord intervals
chord_intervals_major = [0, 4, 7]  # (Root, Major 3rd, Perfect 5th)
chord_intervals_minor = [0, 3, 7]  # (Root, Minor 3rd, Perfect 5th)

# Predefined chord progressions
progression_map = {
    "I": 0,
    "ii": 2,
    "iii": 4,
    "IV": 5,
    "V": 7,
    "vi": 9,
    "vii°": 11,
}

# Some common progressions
predefined_progressions = {
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
    """Generate a progression based on a string like 'I-V-vi-iii-IV'."""
    intervals = chord_intervals_major if mode == "major" else chord_intervals_minor
    progression = []
    bassline_root_notes = []

    print(Fore.GREEN + f"Generating progression: {progression_str}")

    # Split the progression string and generate chords
    for symbol in progression_str.split("-"):
        degree = progression_map[symbol]
        chord = [root_note + degree + i for i in intervals]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def generate_motif_from_chord(chord, length=8):
    """Generate a simple motif based on the notes of the given chord."""
    print(Fore.CYAN + f"Generating motif from chord: {chord}")
    return [random.choice(chord) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    """Apply a variation to the motif."""
    print(Fore.YELLOW + f"Applying {variation_type} variation to motif: {motif}")

    if variation_type == "transpose":
        interval = random.choice([-2, -1, 0, 1, 2])
        return [(note + interval) for note in motif]
    elif variation_type == "invert":
        first_note = motif[0]
        return [(first_note - (note - first_note)) for note in motif]
    else:
        return motif


def create_melody_from_progression(progression, bars=4):
    """Generate a melody by following the chord progression and applying variations."""
    melody = []
    print(Fore.MAGENTA + f"Creating melody from progression: {progression}")
    for bar in range(bars):
        chord = progression[bar % len(progression)]
        motif = generate_motif_from_chord(chord)
        if bar > 0:
            variation_type = random.choice(["transpose", "invert", "none"])
            motif = apply_variation(motif, variation_type=variation_type)
        melody.extend(motif)
    return melody


def create_bassline(bassline_root_notes, bars=4):
    """Create an offbeat bassline based on the root notes of the chord progression."""
    print(Fore.BLUE + "Creating bassline...")
    bassline = []
    for bar in range(bars):
        root_note = bassline_root_notes[bar % len(bassline_root_notes)]
        # Add offbeat bassline (on the 'and' of every beat)
        bassline.extend(
            [0, root_note - 12, 0, root_note - 12, 0, root_note - 12, 0, root_note - 12]
        )  # 16th notes
    return bassline


def create_midi(melody, bassline, filename="randomized_trance_melody.mid"):
    """Generate MIDI file with the given melody and bassline."""
    mid = MidiFile()

    print(Fore.LIGHTBLUE_EX + "Creating MIDI file...")

    # Track for melody
    melody_track = MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage("set_tempo", tempo=tempo))

    for note in melody:
        if note > 0:  # Ignore rests (0s in the melody)
            melody_track.append(Message("note_on", note=note, velocity=64, time=0))
            melody_track.append(
                Message("note_off", note=note, velocity=64, time=note_length_in_ticks)
            )

    # Track for bassline
    bassline_track = MidiTrack()
    mid.tracks.append(bassline_track)

    for note in bassline:
        if note > 0:  # Play bassline note on offbeat
            bassline_track.append(Message("note_on", note=note, velocity=80, time=0))
            bassline_track.append(
                Message("note_off", note=note, velocity=80, time=note_length_in_ticks)
            )
        else:
            bassline_track.append(
                Message("note_off", note=0, velocity=0, time=note_length_in_ticks)
            )

    mid.save(filename)
    print(Fore.GREEN + Style.BRIGHT + f"MIDI file saved as {filename}")


def play_midi(filename):
    """Play the saved MIDI file using pygame."""
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():  # Wait until playback is finished
            time.sleep(1)
    except pygame.error as e:
        print(f"Error playing MIDI file: {e}")


def plot_melody_and_bass(melody, bassline):
    """Plot the melody and bassline using matplotlib."""
    print(Fore.LIGHTWHITE_EX + "Plotting melody and bassline...")
    time_stamps = [i * 0.25 for i in range(len(melody))]  # 16th notes time stamps
    plt.figure(figsize=(12, 6))

    # Plot melody
    plt.step(time_stamps, melody, where="mid", label="Trance Melody", marker="o")

    # Plot bassline (shifted downward for visual distinction)
    bassline_shifted = [note if note > 0 else 0 for note in bassline]
    plt.step(
        time_stamps,
        bassline_shifted,
        where="mid",
        label="Bassline (Offbeat)",
        marker="x",
    )

    plt.xlabel("Beats")
    plt.ylabel("Notes")
    plt.title("Randomized Trance Melody with Bassline (MIDI Visualization)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Randomly pick a root note (for any of the 12 major/minor scales)
    keys = {
        "C3": 48,
        "C#3": 49,
        "D3": 50,
        "D#3": 51,
        "E3": 52,
        "F3": 53,
        "F#3": 54,
        "G3": 55,
        "G#3": 56,
        "A3": 57,
        "A#3": 58,
        "B3": 59,
    }

    # pick from the list of keys
    root_note = random.choice(list(keys.values()))

    # Choose a predefined progression or create a custom one
    progression_choice = random.choice(list(predefined_progressions.keys()))
    progression_str = predefined_progressions[progression_choice]

    print(Fore.GREEN + f"Using progression: {progression_str} ({progression_choice})")

    # Generate progression and bassline based on the chosen progression string
    progression, bassline_root_notes = generate_progressions_from_string(
        root_note, progression_str
    )

    # Generate a melody based on the chosen progression
    melody = create_melody_from_progression(progression, bars=8)

    # Generate a bassline following the root notes of the chosen progression
    bassline = create_bassline(bassline_root_notes, bars=8)

    # Save the MIDI file
    midi_filename = get_unique_filename("generated/trance_3.mid")
    create_midi(melody, bassline, filename=midi_filename)

    # Play the saved MIDI file
    # play_midi(midi_filename)
