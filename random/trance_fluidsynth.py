import random
import mido
from mido import MidiFile, MidiTrack, Message
import fluidsynth

# MIDI Settings
bpm = 138
ticks_per_beat = 480  # Standard MIDI setting
note_length_in_ticks = int(ticks_per_beat / 4)  # 16th note
tempo = mido.bpm2tempo(bpm)

# Path to your SoundFont file (Download a SoundFont like GeneralUser GS)
soundfont_path = "your_soundfont.sf2"

# Initialize FluidSynth for MIDI playback
fs = fluidsynth.Synth()
fs.start(driver="coreaudio")  # Use coreaudio for macOS

# Load the SoundFont file
sfid = fs.sfload(soundfont_path)
fs.program_select(0, sfid, 0, 0)

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


def convert_midi_to_audio(midi_file, output_wav):
    """Convert the MIDI file to an audio WAV file using FluidSynth."""
    fs = fluidsynth.Synth()
    fs.start(driver="coreaudio")  # Use coreaudio driver for macOS
    sfid = fs.sfload(soundfont_path)  # Load the SoundFont
    fs.program_select(0, sfid, 0, 0)  # Select the SoundFont for channel 0

    # Convert the MIDI to audio and save to file
    fs.midi(midi_file)
    fs.delete()
    print(f"Audio file saved as {output_wav}")


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
    midi_filename = "randomized_trance_melody.mid"
    create_midi(melody, bassline, filename=midi_filename)

    # Convert the MIDI file to a WAV audio file
    output_wav_filename = "randomized_trance_melody.wav"
    convert_midi_to_audio(midi_filename, output_wav_filename)

    # You can now play the audio file using any media player, or check the saved output WAV file.
