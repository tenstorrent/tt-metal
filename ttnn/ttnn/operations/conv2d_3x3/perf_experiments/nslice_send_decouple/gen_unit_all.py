# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Copies the conv2d_3x3 unit test files so they run graduated/ (the real op with graduation.patch applied) instead of
the real op (no ttnn, no device).

    python3 gen_unit_all.py      # writes unit_grad/test_grad_<file>.py into this dir

`import torch` becomes a proxy that imports torch on first use (the torch-import hook rejects a module-level
`import torch` anywhere under ttnn/), and every `ttnn.operations.conv2d_3x3` import points at graduated/.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
UNIT = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3"
PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple.graduated"
FILES = [
    "test_conv2d_3x3.py",
    "test_conv2d_3x3_height_sharded.py",
    "test_conv2d_3x3_layouts.py",
    "test_conv2d_3x3_block_sharded.py",
]

LAZY = (
    "\n\n\nclass _LazyTorch:\n"
    "    def __getattr__(self, name):\n"
    "        import torch as _torch\n\n"
    "        return getattr(_torch, name)\n\n\n"
    "torch = _LazyTorch()\n"
)


def main():
    out = HERE / "unit_grad"
    out.mkdir(exist_ok=True)
    for name in FILES:
        s = (UNIT / name).read_text()
        assert "\nimport torch\n" in s, name
        s = s.replace("\nimport torch\n", LAZY, 1)
        s = s.replace("from ttnn.operations.conv2d_3x3 import", f"from {PKG}.conv2d_3x3 import")
        s = s.replace('"ttnn.operations.conv2d_3x3.', f'"{PKG}.')
        assert "ttnn.operations.conv2d_3x3 " not in s and '"ttnn.operations.conv2d_3x3"' not in s, name
        (out / f"test_grad_{name[len('test_'):]}").write_text(s)
        print(out / f"test_grad_{name[len('test_'):]}")


if __name__ == "__main__":
    main()
