import os
import random
import mido
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from libs.utils import get_unique_filename


class MarkovMIDIGenerator:
    def __init__(self):
        self.model = {}
        self.note_properties = {}

    def analyze_midi_folder(self, folder_path, filter=None):
        analyzed_files = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".mid") and (filter is None or filter in filename):
                try:
                    self.analyze_midi_file(os.path.join(folder_path, filename))
                    analyzed_files += 1
                    print(f"Analyzed: {filename}")
                except Exception as e:
                    print(f"Failed: {filename} - {str(e)}")
        print(f"Total files: {analyzed_files}")

    def analyze_midi_file(self, file_path):
        try:
            mid = mido.MidiFile(file_path)
        except Exception as e:
            print(f"Error reading MIDI file {file_path}: {str(e)}")
            return

        previous_notes = []
        current_time = 0
        for msg in mido.merge_tracks(mid.tracks):
            if msg.type == "note_on" and msg.velocity > 0:
                if len(previous_notes) >= 2:
                    key = tuple(previous_notes[-2:])
                    self.model.setdefault(key, {})
                    self.model[key][msg.note] = self.model[key].get(msg.note, 0) + 1
                    self.note_properties.setdefault(key, []).append(
                        (current_time, msg.velocity)
                    )
                previous_notes.append(msg.note)
                current_time = 0
            current_time += msg.time

    def generate_sequence(self, length):
        if not self.model:
            raise ValueError("No data available. Cannot generate sequence.")

        start_notes = random.choice(list(self.model.keys()))
        sequence = list(start_notes)

        for _ in range(length - len(start_notes)):
            current_key = tuple(sequence[-2:])
            if current_key in self.model:
                next_note = random.choices(
                    list(self.model[current_key].keys()),
                    weights=list(self.model[current_key].values()),
                )[0]
            else:
                next_note = random.choice(list(self.model.keys())[0])
            sequence.append(next_note)
        return sequence

    def generate_midi(self, sequence, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.Message("program_change", program=0, time=0))

        current_time = 0
        for i in range(len(sequence)):
            key = tuple(sequence[max(0, i - 2) : i])
            note = sequence[i]
            # if key in self.note_properties:
            #     time, velocity = random.choice(self.note_properties[key])
            # else:
            #     time, velocity = 480, 64

            time, velocity = 480, 64

            track.append(
                mido.Message("note_on", note=note, velocity=velocity, time=current_time)
            )
            track.append(
                mido.Message("note_off", note=note, velocity=velocity, time=time)
            )
            current_time = 0

        mid.save(output_file)

    def save_model(self, file_path):
        """Save the model and note properties to a file (JSON format)."""
        data = {
            "model": {str(k): v for k, v in self.model.items()},
            "note_properties": {str(k): v for k, v in self.note_properties.items()},
        }
        try:
            with open(file_path, "w") as f:
                json.dump(data, f)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")

    def load_model(self, file_path):
        """Load the model and note properties from a file (JSON format)."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                self.model = {
                    tuple(map(int, k[1:-1].split(", "))): v
                    for k, v in data["model"].items()
                }
                self.note_properties = {
                    tuple(map(int, k[1:-1].split(", "))): v
                    for k, v in data["note_properties"].items()
                }
            print(f"Model loaded from {file_path}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")

    def visualize_markov_chain(self, top_n=10):
        """Visualize the top N most frequent transitions in the Markov model."""
        G = nx.DiGraph()

        for key, next_notes in self.model.items():
            print(key, next_notes)
            top_transitions = sorted(
                next_notes.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
            for next_note, weight in top_transitions:
                G.add_edge(key, next_note, weight=weight)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=1500,
            edge_color="gray",
            arrows=True,
        )

        edge_labels = {(k, v): d["weight"] for k, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title(f"Top {top_n} Transitions in Markov Model of MIDI Note Transitions")
        plt.show()

    def create_transition_heatmap(self, top_n=100):
        """Create a heatmap of transition probabilities for the top N most common states."""
        # Get the top N most common states
        state_counts = {
            state: sum(transitions.values())
            for state, transitions in self.model.items()
        }
        top_states = sorted(state_counts, key=state_counts.get, reverse=True)[:top_n]

        # Create a matrix of transition probabilities
        matrix = np.zeros((top_n, top_n))
        for i, state in enumerate(top_states):
            total = sum(self.model[state].values())
            for j, next_state in enumerate(top_states):
                matrix[i, j] = (
                    self.model[state].get(next_state, 0) / total if total else 0
                )

        # Create the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_states,
            yticklabels=top_states,
        )
        plt.title(f"Transition Probabilities for Top {top_n} States")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.show()

    def summarize_model(self, top_n=5):
        """Provide a simple summary of the Markov model."""
        print(f"Model Summary (Top {top_n} transitions for each state):")
        print("---------------------------------------------------")

        for state, transitions in self.model.items():
            print(f"\nState: {state}")
            sorted_transitions = sorted(
                transitions.items(), key=lambda x: x[1], reverse=True
            )
            for next_note, count in sorted_transitions[:top_n]:
                print(f"  → {next_note}: {count}")

        print("\nOverall Statistics:")
        print(f"Total number of states: {len(self.model)}")
        total_transitions = sum(len(transitions) for transitions in self.model.values())
        print(f"Total number of transitions: {total_transitions}")
        avg_transitions = total_transitions / len(self.model) if self.model else 0
        print(f"Average transitions per state: {avg_transitions:.2f}")


if __name__ == "__main__":
    folder = "../trance_midis/"
    # folder = "D:\\Samples\\001 - MIDI\\Hardstyle midis 2009"

    generator = MarkovMIDIGenerator()

    # Example usage of loading and saving models
    model_path = get_unique_filename("../output/markov_models/markov_2.json")
    old = "../output/markov_models/markov_2.json"

    # generator.load_model(old)
    # generator.summarize_model()
    # generator.visualize_markov_chain()
    # generator.create_transition_heatmap()
    # exit()

    # Load model if it exists, otherwise analyze folder and save the model
    if os.path.exists(model_path):
        generator.load_model(model_path)
    else:
        generator.analyze_midi_folder(folder)
        if generator.model:
            generator.save_model(model_path)

    if not generator.model:
        print("No valid data extracted from MIDI files. Cannot generate sequence.")
    else:
        try:
            sequence = generator.generate_sequence(100)
            output_file = get_unique_filename("../output/midi/markov_hardstyle.mid")
            generator.generate_midi(sequence, output_file)
            print(f"MIDI file generated: {output_file}")
        except ValueError as e:
            print(f"Error generating sequence: {str(e)}")
