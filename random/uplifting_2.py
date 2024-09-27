import random
import mido
from mido import MidiFile, MidiTrack, Message
from colorama import init, Fore, Style

from libs.utils import get_unique_filename
from pygame_generate import play_midi

# Initialize colorama
init(autoreset=True)

# MIDI Settings
BPM = 138
TICKS_PER_BEAT = 480
NOTE_LENGTH_IN_TICKS = int(TICKS_PER_BEAT / 4)  # 16th note
TEMPO = mido.bpm2tempo(BPM)

# Scales and chord intervals
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
CHORD_INTERVALS_MINOR = [0, 3, 7]
CHORD_INTERVALS_MAJOR = [0, 4, 7]

# Trance-specific chord progressions
TRANCE_PROGRESSIONS = {
    "classic_uplifting": "i-VI-VII-i",
    "emotional_trance": "i-III-VI-VII",
    "epic_trance": "i-VI-III-VII",
    "melancholic_trance": "i-v-VI-iv",
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

    for symbol in progression_str.split("-"):
        degree = PROGRESSION_MAP[symbol]
        if symbol.isupper():
            chord = [root_note + degree + i for i in CHORD_INTERVALS_MAJOR]
        else:
            chord = [root_note + degree + i for i in CHORD_INTERVALS_MINOR]
        progression.append(chord)
        bassline_root_notes.append(root_note + degree)

    return progression, bassline_root_notes


def create_complex_melody(progression, total_beats):
    melody_layers = [[], [], []]  # Three layers of melody
    scale = [note for chord in progression for note in chord]

    for layer in range(3):
        beat = 0
        while beat < total_beats:
            chord = progression[beat // 32 % len(progression)]
            if random.random() < 0.7:  # 70% chance of a note
                note = random.choice(chord)
                duration = random.choice([1, 2, 4, 8])  # Varied note lengths
                melody_layers[layer].extend([note] + [0] * (duration - 1))
                beat += duration
            else:
                melody_layers[layer].append(0)  # Rest
                beat += 1

    return melody_layers


def create_bassline(bassline_root_notes, total_beats):
    bassline = []
    for bar in range(total_beats // 16):
        root_note = bassline_root_notes[bar % len(bassline_root_notes)]
        pattern = random.choice(
            [
                [root_note, 0, root_note, 0] * 2,
                [root_note, root_note, 0, 0, root_note, 0, root_note, 0],
                [root_note, 0, 0, 0, root_note, 0, root_note, root_note],
            ]
        )
        bassline.extend(pattern * 2)  # Extend pattern to fill a full bar
    return bassline[:total_beats]


def create_pad(progression, total_beats):
    pad = []
    for bar in range(total_beats // 32):
        chord = progression[bar % len(progression)]
        pad.extend([chord] * 32)  # Hold each chord for a full bar
    return pad[:total_beats]


def create_midi(melody_layers, bassline, pad, filename="complex_uplifting_trance.mid"):
    mid = MidiFile()
    print(Fore.LIGHTBLUE_EX + "Creating MIDI file...")

    # Melody tracks
    for i, melody in enumerate(melody_layers):
        melody_track = MidiTrack()
        mid.tracks.append(melody_track)
        if i == 0:
            melody_track.append(mido.MetaMessage("set_tempo", tempo=TEMPO))

        for note in melody:
            if note > 0:
                melody_track.append(Message("note_on", note=note, velocity=64, time=0))
                melody_track.append(
                    Message(
                        "note_off", note=note, velocity=64, time=NOTE_LENGTH_IN_TICKS
                    )
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
                    "note_off", note=note, velocity=40, time=NOTE_LENGTH_IN_TICKS * 32
                )
            )

    mid.save(filename)
    print(Fore.GREEN + Style.BRIGHT + f"MIDI file saved as {filename}")


def main():
    keys = {
        f"{note}{octave}": midi_num
        for octave in [3, 4, 5]
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
    total_beats = total_bars * 16  # 16 sixteenth notes per bar

    melody_layers = create_complex_melody(progression, total_beats)
    bassline = create_bassline(bassline_root_notes, total_beats)
    pad = create_pad(progression, total_beats)

    path = get_unique_filename("../output/midi/complex_uplifting_trance.mid")
    create_midi(melody_layers, bassline, pad, filename=path)

    print(
        Fore.GREEN
        + Style.BRIGHT
        + "Complex uplifting trance melody generated successfully!"
    )
    print(Fore.YELLOW + "MIDI file: complex_uplifting_trance.mid")

    play_midi(path)


if __name__ == "__main__":
    main()
