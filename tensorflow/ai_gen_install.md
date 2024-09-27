# ai_gen.py install guide

## Create a new virtual environment

```shell
python3 -m venv new_venv
```

## Activate the new environment

```shell
source new_venv/bin/activate
```

## Install the required packages

```shell
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.0
pip install music21 numpy
```

## Run your script

```shell
python ai_gen.py
```