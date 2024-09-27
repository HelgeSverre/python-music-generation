import random
import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt
import pygame
import time

from libs.utils import get_unique_filename

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Initialize pygame mixer for MIDI playback
pygame.mixer.init()

# Define Major and Minor progressions for all keys
chord_intervals_major = [0, 4, 7]  # (Root, Major 3rd, Perfect 5th)
chord_intervals_minor = [0, 3, 7]  # (Root, Minor 3rd, Perfect 5th)


# Generate progressions for any root note
def generate_progressions(root_note):
    """Generate major and minor progressions for a given root note."""
    major_progression = [
        [root_note + i for i in chord_intervals_major],  # I
        [root_note + 7 + i for i in chord_intervals_major],  # V
        [root_note + 9 + i for i in chord_intervals_minor],  # vi
        [root_note + 5 + i for i in chord_intervals_major],  # IV
    ]

    minor_progression = [
        [root_note + i - 3 for i in chord_intervals_minor],  # i
        [root_note + 7 + i for i in chord_intervals_major],  # V
        [root_note + i - 3 for i in chord_intervals_minor],  # i
        [root_note + 4 + i for i in chord_intervals_major],  # III
    ]

    major_bassline_root_notes = [root_note, root_note + 7, root_note + 9, root_note + 5]
    minor_bassline_root_notes = [
        root_note - 3,
        root_note + 7,
        root_note - 3,
        root_note + 4,
    ]

    return (
        major_progression,
        minor_progression,
        major_bassline_root_notes,
        minor_bassline_root_notes,
    )


def generate_motif_from_chord(chord, length=8):
    """Generate a simple motif based on the notes of the given chord."""
    return [random.choice(chord) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    """Apply a variation to the motif."""
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


def create_midi(melody, bassline, filename="randomized_trance_melody.mid"):
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
    root_note = random.choice(
        range(48, 60)
    )  # Choose a root note between C3 (48) and B3 (59)

    # Generate major and minor progressions for the selected key
    (
        major_progression,
        minor_progression,
        major_bassline_root_notes,
        minor_bassline_root_notes,
    ) = generate_progressions(root_note)

    # Randomly choose either major or minor progression
    if random.choice(["major", "minor"]) == "major":
        progression = major_progression
        bassline_root_notes = major_bassline_root_notes
    else:
        progression = minor_progression
        bassline_root_notes = minor_bassline_root_notes

    # Generate a melody based on the chosen progression
    melody = create_melody_from_progression(progression, bars=8)

    # Generate a bassline following the root notes of the chosen progression
    bassline = create_bassline(bassline_root_notes, bars=8)

    # Save the MIDI file
    midi_filename = get_unique_filename("generated/pygame.mid")
    create_midi(melody, bassline, filename=midi_filename)

    # Play the saved MIDI file
    # play_midi(midi_filename)
