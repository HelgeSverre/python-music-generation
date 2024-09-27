import random
from midiutil import MIDIFile

from libs.utils import get_unique_filename


class MusicalCryptographer:
    def __init__(self):
        self.notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.durations = [0.25, 0.5, 1]  # quarter, half, whole notes
        self.char_to_note = {}
        self.note_to_char = {}
        self.initialize_mappings()

    def initialize_mappings(self):
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for char in characters:
            note = random.choice(self.notes)
            octave = random.randint(4, 6)
            duration = random.choice(self.durations)
            self.char_to_note[char] = (note, octave, duration)
            self.note_to_char[(note, octave)] = char

    def encrypt(self, message):
        encrypted = []
        for char in message.upper():
            if char in self.char_to_note:
                encrypted.append(self.char_to_note[char])
            elif char == " ":
                encrypted.append(("REST", 0, 1))
        return encrypted

    def decrypt(self, encrypted):
        decrypted = ""
        for note, octave, _ in encrypted:
            if note == "REST":
                decrypted += " "
            elif (note, octave) in self.note_to_char:
                decrypted += self.note_to_char[(note, octave)]
        return decrypted

    def create_midi(self, encrypted, filename="../output/midi/encrypted_message.mid"):
        midi = MIDIFile(1)
        track = 0
        time = 0
        midi.addTrackName(track, time, "Encrypted Message")
        midi.addTempo(track, time, 120)

        for note, octave, duration in encrypted:
            if note != "REST":
                pitch = self.notes.index(note) + (octave * 12) + 60
                midi.addNote(track, 0, pitch, time, duration, 100)
            time += duration

        path = get_unique_filename(filename)
        with open(path, "wb") as output_file:
            midi.writeFile(output_file)
            return path


# Usage example
cryptographer = MusicalCryptographer()

message = "HELLO WORLD 2023"
encrypted = cryptographer.encrypt(message)
print("Encrypted:", encrypted)

decrypted = cryptographer.decrypt(encrypted)
print("Decrypted:", decrypted)

out = cryptographer.create_midi(encrypted)
print("MIDI file created: ", out)
