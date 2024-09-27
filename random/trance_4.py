import random
import mido
from mido import MidiFile, MidiTrack, Message
import pygame
import time

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Initialize pygame mixer for MIDI playback
pygame.mixer.init()

# Define Major and Minor chord intervals with voicings (add 7ths for a richer sound)
chord_intervals_major = [0, 4, 7, 11, 14]  # (Root, Major 3rd, Perfect 5th, Major 7th)
chord_intervals_minor = [0, 3, 7, 10, 14]  # (Root, Minor 3rd, Perfect 5th, Major 7th)

# Predefined chord progressions with uppercase (Major) and lowercase (Minor) notation
progression_map = {
    "I": 0,  # Major 1st
    "ii": 2,  # Minor 2nd
    "iii": 4,  # Minor 3rd
    "IV": 5,  # Major 4th
    "V": 7,  # Major 5th
    "vi": 9,  # Minor 6th
    "vii°": 11,  # Diminished 7th
}

# Some common progressions (uppercase for major, lowercase for minor)
predefined_progressions = {
    "classic_pop": "I-V-vi-IV",
    # "ballad": "I-vi-IV-V",
    # "pachelbel": "I-V-vi-iii-IV-I-IV-V",
    "axis_of_awesome": "I-V-vi-IV",
    "bold_uplifting": "i-III-VII-VI",
    "subtle_tension": "I-V-vi-V",
    # "unique_tension": "IV-I-ii",
    "unusual_vi_start": "vi-V-IV-V",
    "classic_minor": "i-VI-III-VII",
}


def generate_progressions_from_string(root_note, progression_str):
    """Generate a progression based on a string like 'I-V-vi-iii-IV'."""
    progression = []
    bassline_root_notes = []

    # Split the progression string and generate chords
    for symbol in progression_str.split("-"):
        degree = progression_map[symbol]

        # Determine whether the chord is major or minor based on its case
        if symbol.isupper():
            intervals = chord_intervals_major  # Major chord
        else:
            intervals = chord_intervals_minor  # Minor chord

        # Generate the chord with appropriate intervals
        chord = [root_note + degree + i for i in intervals]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def generate_arpeggio_from_chord(chord, pattern="ascending", length=16):
    """Generate an arpeggio from the given chord using a fixed pattern."""
    if pattern == "ascending":
        return [chord[i % len(chord)] for i in range(length)]
    elif pattern == "descending":
        return [chord[len(chord) - (i % len(chord)) - 1] for i in range(length)]
    elif pattern == "alternating":
        return [
            chord[i % len(chord)] if i % 2 == 0 else chord[(i + 1) % len(chord)]
            for i in range(length)
        ]
    else:
        return chord * (length // len(chord))  # Fallback, repeat the chord


def generate_structured_motif(chord, length=8, pattern="repeat"):
    """Generate a structured motif based on a fixed pattern."""
    if pattern == "repeat":
        motif = [
            chord[0],
            chord[1],
            chord[2],
            chord[0],
        ]  # Simple repetition of chord notes
    elif pattern == "ascending":
        motif = [chord[i % len(chord)] for i in range(4)]  # Ascending motif
    elif pattern == "descending":
        motif = [
            chord[len(chord) - (i % len(chord)) - 1] for i in range(4)
        ]  # Descending motif

    return motif * (length // len(motif))  # Repeat the motif to fit the length


def create_rigid_melody_from_progression(progression, bars=4):
    """Generate a structured melody based on fixed motifs and arpeggios."""
    melody = []
    for bar in range(bars):
        chord = progression[bar % len(progression)]
        if bar % 2 == 0:
            motif = generate_structured_motif(chord, length=8, pattern="repeat")
        else:
            motif = generate_arpeggio_from_chord(chord, pattern="ascending", length=8)

        melody.extend(motif)
    return melody


def create_fixed_bassline(bassline_root_notes, bars=4):
    """Create a strict offbeat bassline based on root notes."""
    bassline = []
    for bar in range(bars):
        root_note = bassline_root_notes[bar % len(bassline_root_notes)]
        # Fixed offbeat bassline pattern
        bassline.extend(
            [0, root_note - 12, 0, root_note - 12, 0, root_note - 12, 0, root_note - 12]
        )
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

    # Pick from the list of keys
    root_note = random.choice(list(keys.values()))

    print(f"Using root note: {root_note}")

    # Choose a predefined progression or create a custom one
    progression_choice = random.choice(list(predefined_progressions.keys()))
    progression_str = predefined_progressions[progression_choice]

    print(f"Using progression: {progression_str} ({progression_choice})")

    # Generate progression and bassline based on the chosen progression string
    progression, bassline_root_notes = generate_progressions_from_string(
        root_note, progression_str
    )

    # Generate a more structured melody based on the progression
    melody = create_rigid_melody_from_progression(progression, bars=8)

    # Generate a fixed bassline based on root notes
    bassline = create_fixed_bassline(bassline_root_notes, bars=8)

    # Save the MIDI file
    midi_filename = get_unique_filename("generated/pygame.mid")
    create_midi(melody, bassline, filename=midi_filename)

    # Play the saved MIDI file
    play_midi(midi_filename)
