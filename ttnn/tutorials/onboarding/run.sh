#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run onboarding tests
# Usage: ./run.sh "e01 and solution"

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/ttnn/tutorials/onboarding/.build:$PYTHONPATH"
pytest ttnn/tutorials/onboarding/*/test.py -k "$1"
