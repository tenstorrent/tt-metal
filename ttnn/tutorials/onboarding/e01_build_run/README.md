# E01: Build and Run

Learn to build tt-metal and run a simple ttnn operation.

## Goal

Verify your tt-metal setup by running a simple script that:
1. Opens a Tenstorrent device
2. Adds two tensors on the device
3. Closes the device

## Setup

```bash
git submodule update --init --recursive
./build_metal.sh -e
./create_venv.sh
source python_env/bin/activate
export PYTHONPATH=$(pwd)
```

## Run

```bash
python e01_build_run/run.py
```

## Expected Output

```
a = [[1.0, 2.0], [3.0, 4.0]]
b = [[10.0, 20.0], [30.0, 40.0]]
a + b = [[11.0, 22.0], [33.0, 44.0]]
Success!
```
