import random
import mido
from mido import MidiFile, MidiTrack, Message
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

# Define Minor chord intervals with voicings (add 7ths for a richer sound)
chord_intervals_minor = [0, 3, 7, 10]  # (Root, Minor 3rd, Perfect 5th, Minor 7th)

# Predefined minor-only chord progressions for uplifting trance
progression_map = {
    "i": 0,  # Minor 1st
    "ii": 2,  # Minor 2nd
    "iii": 4,  # Minor 3rd
    "iv": 5,  # Minor 4th
    "v": 7,  # Minor 5th
    "VI": 9,  # Major 6th (common in minor progressions)
    "VII": 10,  # Major 7th (common in minor progressions)
}

# Classic uplifting trance progressions in minor
predefined_progressions = {
    "uplifting_1": "i-VI-III-VII",
    "uplifting_2": "i-VII-VI-VII",
    "uplifting_3": "i-VI-VII-VI",
    "uplifting_4": "i-VI-iv-VII",
    "uplifting_5": "i-VI-iv-v",
}


def generate_progressions_from_string(root_note, progression_str):
    """Generate a progression based on a string like 'i-VI-III-VII'."""
    progression = []
    bassline_root_notes = []

    # Split the progression string and generate chords (all minor)
    for symbol in progression_str.split("-"):
        degree = progression_map[symbol]
        intervals = chord_intervals_minor  # Use minor intervals for all chords

        # Generate the chord with minor intervals
        chord = [root_note + degree + i for i in intervals]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def generate_ascending_arpeggio(chord, length=16, pitch_range=(0, 2)):
    """Generate an ascending arpeggio from the given chord with minimized jumps."""
    arpeggio = []
    current_note = chord[0]  # Start from the root note

    for _ in range(length):
        # Ascend to the next note within the pitch range
        next_note = chord[(arpeggio.count(current_note) % len(chord))] + random.randint(
            *pitch_range
        )
        arpeggio.append(next_note)

    return arpeggio


def generate_descending_bassline(bassline_root_notes, length=8):
    """Create a descending bassline pattern from the root notes."""
    bassline = []
    for root_note in bassline_root_notes:
        for i in range(length):
            bassline.append(root_note - i)  # Descend by one note each time
    return bassline


def create_rigid_melody_from_progression(progression, bars=4):
    """Generate a structured melody with ascending arpeggios."""
    melody = []
    for bar in range(bars):
        chord = progression[bar % len(progression)]
        # Generate ascending arpeggios for melody
        arpeggio = generate_ascending_arpeggio(chord, length=8, pitch_range=(-1, 1))
        melody.extend(arpeggio)
    return melody


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
        if note > 0:  # Play bassline note
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
    # Randomly pick a root note (for any of the 12 minor scales)
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

    # Choose a predefined minor progression
    progression_choice = random.choice(list(predefined_progressions.keys()))
    progression_str = predefined_progressions[progression_choice]

    print(f"Using progression: {progression_str} ({progression_choice})")

    # Generate progression and bassline based on the chosen progression string
    progression, bassline_root_notes = generate_progressions_from_string(
        root_note, progression_str
    )

    # Generate a structured melody based on the progression (ascending arpeggio)
    melody = create_rigid_melody_from_progression(progression, bars=8)

    # Generate a descending bassline based on root notes
    bassline = generate_descending_bassline(bassline_root_notes, length=8)

    # Save the MIDI file
    midi_filename = get_unique_filename("../output/midi/pygame.mid")
    create_midi(melody, bassline, filename=midi_filename)
