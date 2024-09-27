import random
import mido
from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Scale setup (C minor)
scale = [60, 62, 63, 65, 67, 68, 70, 72]  # C minor scale

# Common Trance Chord Progression in C minor: I - VI - III - VII (Cm - Ab - Eb - Bb)
chord_progression = [
    [60, 63, 67],  # C minor (Cm)
    [68, 72, 65],  # Ab major (Ab)
    [63, 67, 70],  # Eb major (Eb)
    [65, 70, 74],  # Bb major (Bb)
]

# Root notes of the chord progression for the bassline (C, Ab, Eb, Bb) transposed down one octave
bassline_root_notes = [
    60 - 12,
    68 - 12,
    63 - 12,
    65 - 12,
]  # Bass will play root notes an octave lower


def generate_motif_from_chord(chord, length=8):
    """Generate a simple motif based on the notes of the given chord."""
    return [random.choice(chord) for _ in range(length)]


def apply_variation(motif, variation_type="transpose"):
    """Apply a variation to the motif."""
    if variation_type == "transpose":
        # Transpose the motif up or down a random interval (within a 5-note range)
        interval = random.choice([-2, -1, 0, 1, 2])
        return [(note + interval) for note in motif]
    elif variation_type == "invert":
        # Invert the motif around the first note
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
            # Apply variation to the motif for subsequent bars
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
            [0, root_note, 0, root_note, 0, root_note, 0, root_note]
        )  # 16th note representation
    return bassline


def create_midi(
    melody,
    bassline,
    filename="../output/midi/trance_melody_with_transposed_bassline.mid",
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
            # If the note is 0, it's a rest, so we just wait for the length of the note
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
    bassline_shifted = [
        note if note > 0 else 0 for note in bassline
    ]  # No need to shift since already transposed
    plt.step(
        time_stamps,
        bassline_shifted,
        where="mid",
        label="Bassline (Offbeat)",
        marker="x",
    )

    plt.yticks(
        scale + [n - 12 for n in scale],
        [
            "C",
            "D",
            "D#",
            "F",
            "G",
            "G#",
            "A#",
            "C",
            "C(bass)",
            "D(bass)",
            "D#(bass)",
            "F(bass)",
            "G(bass)",
            "G#(bass)",
            "A#(bass)",
            "C(bass)",
        ],
    )
    plt.xlabel("Beats")
    plt.ylabel("Notes")
    plt.title("Trance Melody with Offbeat Transposed Bassline (MIDI Visualization)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generate a melody based on the common progression with variations applied over 8 bars
    melody = create_melody_from_progression(chord_progression, bars=8)

    # Generate a bassline following the root notes of the chord progression, transposed by 1 octave
    bassline = create_bassline(bassline_root_notes, bars=8)

    # Create MIDI file
    create_midi(melody, bassline)

    # Plot the melody and bassline
    plot_melody_and_bass(melody, bassline)
