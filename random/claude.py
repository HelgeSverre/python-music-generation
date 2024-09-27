import random
import midiutil
import matplotlib.pyplot as plt
import numpy as np

# Constants
BPM = 138
BEATS_PER_BAR = 4
TICKS_PER_BEAT = 480
BARS = 16  # Extended to 16 bars for a full phrase

# Scales and chords
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 11]


def get_chord(scale, degree):
    return [scale[degree], scale[(degree + 2) % 7], scale[(degree + 4) % 7]]


def create_progression(key, is_minor=False):
    if is_minor:
        scale = [(note + key) % 12 for note in NATURAL_MINOR_SCALE]
        progression = [
            get_chord(scale, 0),  # i
            get_chord(scale, 5),  # VI
            get_chord(scale, 3),  # iv
            [
                (scale[4] + 12) % 12,
                scale[6],
                scale[1],
            ],  # V (borrowed from harmonic minor)
        ]
    else:
        scale = [(note + key) % 12 for note in MAJOR_SCALE]
        progression = [
            get_chord(scale, 0),  # I
            get_chord(scale, 5),  # vi
            get_chord(scale, 3),  # IV
            get_chord(scale, 4),  # V
        ]
    return progression


def create_melody(progression, key, is_minor=False):
    melody = []
    scale = NATURAL_MINOR_SCALE if is_minor else MAJOR_SCALE
    scale = [(note + key) % 12 for note in scale]

    for chord in progression:
        for _ in range(BEATS_PER_BAR):
            if random.random() < 0.7:  # 70% chance of a note
                note = random.choice(chord) + random.choice([0, 12, 24])
                note = max(0, min(note, 127))  # Ensure note is within MIDI range
                melody.append(note)
            else:
                melody.append(None)  # Rest
    return melody


def create_bassline(progression):
    bassline = []
    for chord in progression:
        root = chord[0] - 12
        root = max(0, min(root, 127))  # Ensure note is within MIDI range
        bassline.extend([root] + [None] * (BEATS_PER_BAR - 1))
    return bassline


def create_midi(melody, bassline, chord_prog, filename):
    midi = midiutil.MIDIFile(3)  # 3 tracks: melody, bassline, and chords
    midi.addTempo(0, 0, BPM)

    # Add melody
    for i, note in enumerate(melody):
        if note is not None:
            start_time = i * 0.25  # Each 16th note is 0.25 beats
            duration = 0.25
            midi.addNote(0, 0, note, start_time, duration, 100)

    # Add bassline
    for i, note in enumerate(bassline):
        if note is not None:
            start_time = i
            duration = 1  # Hold for a quarter note
            midi.addNote(1, 0, note, start_time, duration, 100)

    # Add chords
    for i, chord in enumerate(chord_prog):
        for note in chord:
            start_time = i * BEATS_PER_BAR
            duration = BEATS_PER_BAR
            midi.addNote(
                2, 0, note + 60, start_time, duration, 80
            )  # Add 60 to move chords to middle range

    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)


def visualize_melody(melody, bassline, chord_prog):
    plt.figure(figsize=(15, 8))

    # Melody visualization
    x_melody = [i * 0.25 for i in range(len(melody))]
    y_melody = [note if note is not None else np.nan for note in melody]
    plt.scatter(x_melody, y_melody, alpha=0.6, label="Melody", color="blue")

    # Bassline visualization
    x_bass = [i for i in range(len(bassline))]
    y_bass = [note if note is not None else np.nan for note in bassline]
    plt.scatter(x_bass, y_bass, alpha=0.6, label="Bassline", color="red")

    # Chord visualization
    for i, chord in enumerate(chord_prog):
        for note in chord:
            plt.plot(
                [i * BEATS_PER_BAR, (i + 1) * BEATS_PER_BAR],
                [note + 60, note + 60],
                color="green",
                alpha=0.3,
            )

    plt.title("Advanced Trance Melody, Bassline, and Chord Visualization")
    plt.xlabel("Time (beats)")
    plt.ylabel("MIDI Note")
    plt.legend()
    plt.grid(True)
    plt.savefig("generated/advanced_trance_visualization.png")
    plt.close()


def main():
    # Start in A minor (key = 9)
    key = 9
    is_minor = True

    intro_progression = create_progression(key, is_minor)
    intro_melody = create_melody(intro_progression, key, is_minor)
    intro_bassline = create_bassline(intro_progression)

    # Switch to C major (relative major, key = 0)
    key = 0
    is_minor = False

    main_progression = create_progression(key, is_minor)
    main_melody = create_melody(main_progression, key, is_minor)
    main_bassline = create_bassline(main_progression)

    # Combine intro and main sections
    full_melody = intro_melody + main_melody
    full_bassline = intro_bassline + main_bassline
    full_progression = intro_progression + main_progression

    create_midi(
        full_melody,
        full_bassline,
        full_progression,
        "generated/advanced_trance_melody_2.mid",
    )
    visualize_melody(full_melody, full_bassline, full_progression)
    print(
        "Advanced trance melody generated and saved as 'generated/advanced_trance_melody.mid'"
    )
    print(
        "Advanced melody visualization saved as 'generated/advanced_trance_visualization.png'"
    )


if __name__ == "__main__":
    main()
