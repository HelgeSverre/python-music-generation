import random
import mido
from mido import MidiFile, MidiTrack, Message
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

# Trance-specific chord progressions
TRANCE_PROGRESSIONS = {
    "classic_uplifting": [0, 5, 7, 0],  # i-VI-VII-i in minor
    "emotional_trance": [0, 3, 5, 7],  # i-III-VI-VII in minor
    "epic_trance": [0, 5, 3, 7],  # i-VI-III-VII in minor
}


def create_chord(root, chord_type="minor"):
    if chord_type == "minor":
        return [root, root + 3, root + 7, root + 10]
    else:  # major
        return [root, root + 4, root + 7, root + 11]


def generate_progression(root_note, progression_type):
    progression = []
    for degree in TRANCE_PROGRESSIONS[progression_type]:
        chord_root = (root_note + degree) % 12
        chord = create_chord(
            chord_root, "minor" if degree in [0, 2, 3, 5, 7] else "major"
        )
        progression.append(chord)
    return progression


def create_arpeggio_pattern(chord, pattern=[0, 1, 2, 0, 2, 1]):
    return [chord[i] for i in pattern]


def create_melody_and_chords(progression, total_bars):
    melody = []
    chords = []
    bass = []
    arp_pattern = [0, 1, 2, 0, 2, 1]

    for bar in range(total_bars):
        chord = progression[bar % len(progression)]
        bass_note = chord[0] - 12  # Root note, one octave lower

        # Create bassline
        bass.extend([bass_note] + [0] * 15)  # One bass note per bar

        # Create chord arpeggio
        bar_arpeggio = create_arpeggio_pattern(chord, arp_pattern)
        melody.extend(bar_arpeggio * 8)  # Repeat the pattern 8 times to fill the bar

        # Add full chord for pad
        chords.extend([chord] * 16)  # Hold chord for entire bar

    return melody, chords, bass


def create_midi(melody, chords, bass, filename="uplifting_trance_arpeggio.mid"):
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

    # Bass track
    bass_track = MidiTrack()
    mid.tracks.append(bass_track)
    for note in bass:
        if note > 0:
            bass_track.append(Message("note_on", note=note, velocity=80, time=0))
            bass_track.append(
                Message(
                    "note_off", note=note, velocity=80, time=NOTE_LENGTH_IN_TICKS * 16
                )
            )
        else:
            bass_track.append(
                Message("note_off", note=0, velocity=0, time=NOTE_LENGTH_IN_TICKS)
            )

    # Chord (pad) track
    chord_track = MidiTrack()
    mid.tracks.append(chord_track)
    for chord in chords:
        for note in chord:
            chord_track.append(Message("note_on", note=note, velocity=40, time=0))
        for note in chord:
            chord_track.append(
                Message(
                    "note_off", note=note, velocity=40, time=NOTE_LENGTH_IN_TICKS * 16
                )
            )

    mid.save(filename)
    print(Fore.GREEN + Style.BRIGHT + f"MIDI file saved as {filename}")


def main():
    root_notes = {
        "A": 57,
        "A#": 58,
        "B": 59,
        "C": 60,
        "C#": 61,
        "D": 62,
        "D#": 63,
        "E": 64,
        "F": 65,
        "F#": 66,
        "G": 67,
        "G#": 68,
    }
    root_note = random.choice(list(root_notes.values()))
    progression_type = random.choice(list(TRANCE_PROGRESSIONS.keys()))

    print(Fore.GREEN + f"Using progression type: {progression_type}")
    print(
        Fore.GREEN
        + f"Root note: {list(root_notes.keys())[list(root_notes.values()).index(root_note)]}"
    )

    progression = generate_progression(root_note, progression_type)

    total_bars = 32  # A typical length for a trance section

    melody, chords, bass = create_melody_and_chords(progression, total_bars)

    path = get_unique_filename("generated/uplifting_trance_arpeggio.mid")

    create_midi(melody, chords, bass, filename=path)

    print(
        Fore.GREEN
        + Style.BRIGHT
        + "Uplifting trance arpeggio pattern generated successfully!"
    )
    print(Fore.YELLOW + "MIDI file: uplifting_trance_arpeggio.mid")

    play_midi(path)


if __name__ == "__main__":
    main()
