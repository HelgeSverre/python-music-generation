# Miscellaneous Python Scripts for Music Generation

## Install on mac

```shell
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.0
```

## Install all things

```shell
pip install -r requirements.txt
```

## Running the Scripts

```shell
python -m venv venv
source .venv/bin/activate

## AI Generated - ChatGPT
python main.py

## AI Generated - Claude
python claude.py
```

## Formatting the Code

```shell
pipx run black *.py
```