# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op's HEIGHT-sharded and layouts unit tests (plus the interleaved file with SS_UNIT_ALL=1) re-bound to this dir's
op_copy/ (or another package with SS_UNIT_PKG), its knobs at their defaults; SS_UNIT_KNOBS="KNOB=int,..." overrides
descriptor knobs (e.g. SD_STREAM_MIN_SUBBLOCK_ROWS=1 streams every slab-direct block). Run with --import-mode=prepend."""

import importlib
import os

import pytest

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.stem1_slab_stream." + os.environ.get("SS_UNIT_PKG", "op_copy")
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
_UT = "tests.ttnn.unit_tests.operations.conv2d_3x3."
_KNOBS = {k: int(v) for k, v in (kv.split("=") for kv in os.environ.get("SS_UNIT_KNOBS", "").split(",") if kv)}


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    for k, v in _KNOBS.items():
        assert hasattr(_pd, k), k
        monkeypatch.setattr(_pd, k, v)
    _pd.select_blocking.cache_clear()
    yield
    _pd.select_blocking.cache_clear()


_MODS = ["test_conv2d_3x3_height_sharded", "test_conv2d_3x3_layouts"]
if os.environ.get("SS_UNIT_ALL"):
    _MODS += ["test_conv2d_3x3"]
for _mod_name in _MODS:
    _m = importlib.import_module(_UT + _mod_name)
    for _name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights", "default_compute_kernel_config"):
        if hasattr(_m, _name):
            setattr(_m, _name, getattr(_op, _name))
    if hasattr(_m, "_op_module"):
        _m._op_module = _op
    if hasattr(_m, "_pd"):
        _m._pd = _pd
    for _name in dir(_m):
        if _name.startswith("test_"):
            globals()[f"test_{_mod_name[len('test_conv2d_3x3'):].lstrip('_') or 'main'}__{_name[5:]}"] = getattr(
                _m, _name
            )
