# Python Scripts for Music Generation

[![AI-Generated](https://img.shields.io/badge/Contains-AI--Generated--Code-brightgreen)](https://github.com/HelgeSverre/python-music-generation)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎵 Project Description

This repository contains a diverse collection of Python scripts designed to generate MIDI files for music creation.

All code in this project is **AI-generated**, showcasing various approaches to algorithmic music composition, including
Markov chains, machine learning models, and rule-based systems.

**Note:** This project is intended for experimentation, learning, and exploration of AI-driven music generation
techniques, rather than professional music production.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/HelgeSverre/python-music-generation.git
   cd python-music-generation
   ```

2. Create and activate a virtual environment:
   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies (might be outdated):
   ```shell
   pip install -r requirements.txt
   ```

4. For Mac users, install TensorFlow with Metal support:
   ```shell
   pip install tensorflow-macos==2.13.0
   pip install tensorflow-metal==1.0.0
   ```

## 📁 Project Structure

```
.
├── README.md
├── docs/
├── libs/
├── markov/
├── models/
├── output/
│   ├── images/
│   ├── markov_models/
│   └── midi/
├── random/
├── tensorflow/
└── tools/
```

### ❗ Not included in this repository

- `trance_midis/`: MIDI files used by various scripts, not included in this repository as it contains copyrighted
  material. Replace it with your own MIDI file collection.

### Key Directories and Files

- `docs/`: Documentation files, including `transformer_trance.md`
    - Explains the Transformer model for trance music generation (AI-generated documentation, use with caution)
- `libs/`: Utility functions and helper scripts
    - `utils.py`: Utility functions for music generation (e.g., getting unique file names with number suffixes)
- `markov/`: Markov chain-based music generation scripts
- `models/`: Saved TensorFlow models for music generation
- `output/`: Generated MIDI files, images, and Markov models
- `random/`: Various music generation scripts using different techniques
- `tensorflow/`: TensorFlow-based music generation scripts (currently broken)
- `tools/`: Utility tools, including MIDI to CSV converter

## 🎹 Running the Scripts

1. Ensure your virtual environment is activated.

2. Navigate to the desired script directory (e.g., `random/`, `tensorflow/`, or `markov/`).

3. Run a script:
   ```shell
   python script_name.py
   ```

   For example:
   ```shell
   python random/trance.py
   ```

4. Check the `output/midi/` directory for generated MIDI files.

## 🧠 Scripts Overview

1. **Markov Chain Models** (`markov/`):
    - `markov.py`, `markov_2.py`, `markov_3.py`: Different implementations of Markov chain-based music generation

2. **Random Generation** (`random/`):
    - `claude.py`: Claude AI-generated music script
    - `trance.py`, `trance_2.py`, ...: Various trance music generation scripts using different rule-based generation
      techniques
    - `uplifting.py`, `uplifting_2.py`: Uplifting trance generation scripts

    - Other scripts for generating hardstyle, arpeggios, and more
3. **TensorFlow Models** (`tensorflow/`): (currently broken)
    - `tensorflow_trance_midi.py`: TensorFlow-based trance MIDI generation
    - `transformer_trance.py`: Transformer model for trance music generation

4. **Utility Scripts**:
    - `tools/midi-to-csv.py`: Convert MIDI files to CSV format (
      `midi_note ,note_name ,event ,velocity ,delta ,cumulative_time`)
    - `libs/utils.py`: Utility functions used across scripts

## 🎧 Output

Generated MIDI files can be found in the `output/midi/` directory. You can use various software to play these files:

- [VLC Media Player](https://www.videolan.org/vlc/) (free, cross-platform)
- [MuseScore](https://musescore.org/) (free, open-source music notation software)
- Your preferred Digital Audio Workstation (DAW)

## 🛠️ Development

To format the code with the Black code style, run:

```shell
pipx run black *.py
```

## 🤝 Contributing

While this project is primarily an AI experiment, contributions are welcome! If you have ideas for improvements or new
AI-generated scripts, feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤖 AI Models Used

- **OpenAI**: ChatGPT (`gpt-4o` and `o1-preview`)
- **Anthropic**: Claude Sonnet 3.5 (`claude-3-5-sonnet-20240620`)

---

**Disclaimer:** The code in this repository, including the README, is _mostly_ AI-generated and may contain errors or inconsistencies. Use at your own
discretion and always review the code before running it in your environment.