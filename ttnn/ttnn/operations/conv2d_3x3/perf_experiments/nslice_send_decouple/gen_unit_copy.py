# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Writes a copy of the BLOCK unit tests that runs this dir's op copy with the ring reduce-scatter (no ttnn, no device).

    python3 gen_unit_copy.py [out_dir|-] [op_copy|graduated]   # then run_safe_pytest.sh --dev <out>/test_block_*.py

The torch-import hook rejects a module-level `import torch` anywhere under ttnn/, so the copy binds `torch` to a
proxy that imports it on first use (inside the tests). Every test of
tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py runs unchanged against
op_copy/ with NSD_DEFINES = ("NSD_RING",) (or NSD_UNIT_DEFINES, comma-separated).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
SRC = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py"
BASE = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple."


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else Path(__file__).resolve().parent
    # op_copy (ring via NSD_DEFINES) or graduated (the patched op: the ring is its only N-slice path).
    copy = sys.argv[2] if len(sys.argv) > 2 else "op_copy"
    PKG = BASE + copy
    out.mkdir(parents=True, exist_ok=True)
    s = SRC.read_text()
    assert "\nimport torch\n" in s
    s = s.replace(
        "\nimport torch\n",
        "\n\n\nclass _LazyTorch:\n"
        "    def __getattr__(self, name):\n"
        "        import torch as _torch\n\n"
        "        return getattr(_torch, name)\n\n\n"
        "torch = _LazyTorch()\n",
        1,
    )
    old_import = "from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights\n"
    assert old_import in s
    s = s.replace(old_import, "")
    old_mod = '_op_module = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")\n'
    old_pd = '_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")\n'
    assert old_mod in s and old_pd in s
    s = s.replace(old_mod, f'_op_module = importlib.import_module("{PKG}.conv2d_3x3")\n')
    s = s.replace(
        old_pd,
        f'_pd = importlib.import_module("{PKG}.conv2d_3x3_program_descriptor")\n'
        "conv2d_3x3 = _op_module.conv2d_3x3\n"
        "prepare_conv2d_bias = _op_module.prepare_conv2d_bias\n"
        "prepare_conv2d_weights = _op_module.prepare_conv2d_weights\n"
        "import os as _os\n\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def _nsd_ring(monkeypatch):\n"
        '    defines = tuple(x for x in _os.environ.get("NSD_UNIT_DEFINES", "NSD_RING").split(",") if x)\n'
        '    monkeypatch.setattr(_pd, "NSD_DEFINES", defines, raising=False)\n'
        "    _pd.select_blocking.cache_clear()\n"
        "    yield\n"
        "    _pd.select_blocking.cache_clear()\n",
    )
    name = "test_block_ring.py" if copy == "op_copy" else f"test_block_{copy}.py"
    (out / name).write_text(s)
    print(out / name)


if __name__ == "__main__":
    main()
