import argparse
import csv
import sys
from mido import MidiFile
from colorama import init, Fore, Style
from pygame.midi import midi_to_ansi_note


def midi_to_csv(midi_file, output_file):
    try:
        mid = MidiFile(midi_file)
    except IOError:
        print(
            f"{Fore.RED}Error: Could not open MIDI file '{midi_file}'{Style.RESET_ALL}"
        )
        sys.exit(1)

    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["midi_note", "note_name", "event", "velocity", "delta", "cumulative_time"]
        )

        current_time = 0
        for msg in mid:
            current_time += msg.time
            if msg.type in ["note_on", "note_off"]:
                csv_writer.writerow(
                    [
                        msg.note,
                        midi_to_ansi_note(msg.note),
                        msg.type,
                        msg.velocity,
                        msg.time,
                        round(current_time * 1000, 2),
                    ]
                )

    print(
        f"{Fore.GREEN}Successfully converted MIDI to CSV: {output_file}{Style.RESET_ALL}"
    )


def main():
    init()  # Initialize colorama

    parser = argparse.ArgumentParser(description="Convert MIDI file to CSV format")
    parser.add_argument("input", help="Input MIDI file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file (default: output.csv)",
        default="output.csv",
    )

    args = parser.parse_args()
    print(f"{Fore.CYAN}Converting {args.input} to {args.output}{Style.RESET_ALL}")
    midi_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
