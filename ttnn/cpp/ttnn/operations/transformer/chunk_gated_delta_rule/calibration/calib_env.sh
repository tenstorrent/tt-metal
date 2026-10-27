# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Source this before the calibration scripts, with the tree's python_env active. Defaults: the repository this file
# lives in, captures under generated/gdn_calib. Export any of the variables before sourcing to override.
#   TT_METAL_HOME  the tree under calibration (a Release build with the device profiler, installed into the venv)
#   CALIB_OUT      capture directory, one sub-directory per label
#   CALIB_LOCK     a lock file that serialises device use between users of one board
#   TT_METAL_CACHE a private JIT cache, so programs of another tree are never served from it
export CALIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TT_METAL_HOME:=$(cd "$CALIB_DIR/../../../../../../.." && pwd)}"
export TT_METAL_HOME
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$TT_METAL_HOME"
: "${CALIB_OUT:=$TT_METAL_HOME/generated/gdn_calib}"
: "${CALIB_LOCK:=$CALIB_OUT/device.lock}"
: "${TT_METAL_CACHE:=$CALIB_OUT/jit-cache}"
export CALIB_OUT CALIB_LOCK TT_METAL_CACHE
mkdir -p "$CALIB_OUT"
touch "$CALIB_LOCK"
