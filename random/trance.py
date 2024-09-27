import random
from os import system

import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Harmonic context: C major and its relative minor, A minor
# I - V - vi - IV progression in C major, and vi - V - i - III in A minor
major_progression = [
    [60, 64, 67],  # C major (I)
    [67, 71, 74],  # G major (V)
    [57, 60, 64],  # A minor (vi)
    [65, 69, 72],  # F major (IV)
]

minor_progression = [
    [57, 60, 64],  # A minor (i)
    [67, 71, 74],  # G major (V)
    [60, 64, 67],  # C major (III)
    [62, 65, 69],  # D minor (iv)
]

# Bass root notes for major and minor progressions
major_bassline_root_notes = [60, 67, 57, 65]  # Bass for C, G, A, F
minor_bassline_root_notes = [57, 67, 60, 62]  # Bass for A, G, C, D


def generate_motif_from_chord(chord, length=8):
    """Generate a simple motif based on the notes of the given chord."""
    return [random.choice(chord) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    """Apply a variation to the motif."""
    if variation_type == "transpose":
        interval = random.choice([-2, -1, 0, 1, 2])  # Minor transposition
        return [(note + interval) for note in motif]
    elif variation_type == "invert":
        first_note = motif[0]
        return [(first_note - (note - first_note)) for note in motif]
    else:
        return motif


def create_melody_from_progression(progression, bars=4):
    """Generate a melody by following the chord progression and applying variations."""
    melody = []
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
    bassline = []
    for bar in range(bars):
        root_note = bassline_root_notes[bar % len(bassline_root_notes)]
        # Add offbeat bassline (on the 'and' of every beat)
        bassline.extend(
            [0, root_note - 12, 0, root_note - 12, 0, root_note - 12, 0, root_note - 12]
        )  # 16th notes
    return bassline


def create_midi(
    melody, bassline, filename="generated/trance_melody_with_dynamic_harmony.mid"
):
    """Generate MIDI file with the given melody and bassline."""
    mid = MidiFile()

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
        else:
            melody_track.append(
                Message("note_off", note=0, velocity=0, time=note_length_in_ticks)
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
    print(f"MIDI file saved as {filename}")


def plot_melody_and_bass(melody, bassline):
    """Plot the melody and bassline using matplotlib."""
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
    plt.title("Trance Melody with Dynamic Harmony (MIDI Visualization)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Switch between major and minor progressions
    progression = major_progression + minor_progression
    bassline_root_notes = major_bassline_root_notes + minor_bassline_root_notes

    # Generate a melody based on the mixed progression
    melody = create_melody_from_progression(progression, bars=8)

    # Generate a bassline following the root notes of the mixed progression
    bassline = create_bassline(bassline_root_notes, bars=8)

    # Create MIDI file
    create_midi(melody, bassline)

    # Plot the melody and bassline
    # plot_melody_and_bass(melody, bassline)
