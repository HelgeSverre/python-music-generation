import mido
import numpy as np
from fractions import Fraction


class RhythmicComplexityAnalyzer:
    def __init__(self, midi_file):
        self.midi_file = midi_file
        self.midi = mido.MidiFile(midi_file)
        self.ticks_per_beat = self.midi.ticks_per_beat
        self.grid = None
        self.total_ticks = self.calculate_total_ticks()

    def calculate_total_ticks(self):
        total = 0
        for track in self.midi.tracks:
            track_ticks = sum(msg.time for msg in track)
            total = max(total, track_ticks)
        return total

    def create_grid(self):
        self.grid = np.zeros(self.total_ticks + 1)  # +1 to include the last tick

    def populate_grid(self):
        self.create_grid()
        for track in self.midi.tracks:
            current_tick = 0
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    if current_tick < len(self.grid):
                        self.grid[current_tick] += 1
                current_tick += msg.time
                if current_tick >= len(self.grid):
                    break

    def analyze_syncopation(self):
        if len(self.grid) < 4:
            return 0  # Not enough data for syncopation analysis
        beat_strength = [3, 1, 2, 1] * (len(self.grid) // 4 + 1)  # Assuming 4/4 time
        beat_strength = beat_strength[: len(self.grid)]
        syncopation_score = sum(
            self.grid[i] * (4 - strength) for i, strength in enumerate(beat_strength)
        )
        return syncopation_score / len(self.grid)

    def find_polyrhythms(self):
        polyrhythms = []
        for i in range(2, 9):  # Check for polyrhythms up to 8 against 4
            if self.ticks_per_beat % i == 0:  # Ensure we can divide evenly
                rhythm = self.grid[:: self.ticks_per_beat // i]
                if np.sum(rhythm) > 0 and i != 4:  # Exclude the main meter
                    polyrhythms.append(f"4:{i}")
        return polyrhythms

    def analyze_metric_structure(self):
        if len(self.grid) < self.ticks_per_beat * 8:
            return 0  # Not enough data for metric analysis
        measures = [
            sum(self.grid[i : i + self.ticks_per_beat * 4])
            for i in range(
                0, len(self.grid) - self.ticks_per_beat * 4, self.ticks_per_beat * 4
            )
        ]
        if len(measures) < 2:
            return 0
        changes = [abs(measures[i] - measures[i - 1]) for i in range(1, len(measures))]
        return sum(changes) / len(changes)

    def analyze(self):
        self.populate_grid()
        syncopation = self.analyze_syncopation()
        polyrhythms = self.find_polyrhythms()
        metric_complexity = self.analyze_metric_structure()

        return {
            "syncopation_score": syncopation,
            "polyrhythms": polyrhythms,
            "metric_complexity": metric_complexity,
        }


if __name__ == "__main__":
    try:
        analyzer = RhythmicComplexityAnalyzer(
            "trance_midis/Abovebeyond_-_WalterWhite__FrozenRay_20130606000940.mid"
        )
        results = analyzer.analyze()

        print(f"Syncopation Score: {results['syncopation_score']:.2f}")
        print(f"Detected Polyrhythms: {', '.join(results['polyrhythms'])}")
        print(f"Metric Complexity: {results['metric_complexity']:.2f}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check if the MIDI file path is correct and the file is valid.")
