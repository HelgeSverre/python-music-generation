import random
import mido
from mido import MidiFile, MidiTrack, Message
from colorama import init, Fore, Style

from libs.utils import get_unique_filename

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


def create_uplifting_arpeggio(chord, octave_range=2):
    """Create an uplifting trance arpeggio pattern based on the 12342314 sequence."""
    base_note = min(chord)
    arpeggio_pattern = [0, 1, 2, 3, 2, 1, 3]  # Represents the 12342314 pattern
    full_arpeggio = []

    for octave in range(octave_range):
        for step in arpeggio_pattern:
            note = base_note + step + (octave * 12)
            full_arpeggio.append(note)

    return full_arpeggio


def create_melody_from_progression(progression, total_beats):
    melody = []
    print(
        Fore.MAGENTA
        + f"Creating uplifting trance melody from progression: {progression}"
    )

    arpeggio_length = 8  # 8 sixteenth notes for one complete 12342314 pattern

    while len(melody) < total_beats:
        chord = progression[len(melody) // 32 % len(progression)]
        arpeggio = create_uplifting_arpeggio(chord)

        # Repeat the arpeggio pattern for the duration of the chord (usually 1 bar)
        for _ in range(32 // arpeggio_length):
            melody.extend(arpeggio)

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
    while len(pad) * 32 < total_beats:
        chord = progression[len(pad) % len(progression)]
        pad.extend([chord] * 32)  # Hold each chord for a full bar
    return pad[:total_beats]


def create_midi(melody, bassline, pad, filename="uplifting_trance_melody.mid"):
    mid = MidiFile()
    print(Fore.LIGHTBLUE_EX + "Creating MIDI file...")

    # Melody track
    melody_track = MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage("set_tempo", tempo=TEMPO))

    for note in melody:
        melody_track.append(Message("note_on", note=note, velocity=64, time=0))
        melody_track.append(
            Message("note_off", note=note, velocity=64, time=NOTE_LENGTH_IN_TICKS)
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
                    "note_off", note=note, velocity=40, time=NOTE_LENGTH_IN_TICKS * 32
                )
            )

    mid.save(filename)
    print(Fore.GREEN + Style.BRIGHT + f"MIDI file saved as {filename}")


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

    total_bars = 32  # A typical length for a trance section
    total_beats = total_bars * 8  # 8 sixteenth notes per bar

    melody = create_melody_from_progression(progression, total_beats)
    bassline = create_bassline(bassline_root_notes, total_beats)
    pad = create_pad(progression, total_beats)

    path = get_unique_filename("../output/midi/uplifting_trance_melody.mid")

    create_midi(melody, bassline, pad, filename=path)

    print(Fore.GREEN + Style.BRIGHT + "Uplifting trance melody generated successfully!")
    print(Fore.YELLOW + "MIDI file: uplifting_trance_melody.mid")


if __name__ == "__main__":
    main()
