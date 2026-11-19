# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: op_base_abl / op_copy_abl = op_base / op_copy plus ablation_r13/ablation_hooks.patch (CONV2D_3X3_ABLATE
stubs; only the `fill` stub is used here: the weight-stub hunks of issue_weight_slice do not apply to op_copy and are
dropped), and their zone-stripped twins op_base_ablnz / op_copy_ablnz.
    python3 mk_abl.py"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "ablation_r13" / "ablation_hooks.patch"
for v in ("op_base", "op_copy"):
    dst = HERE / (v + "_abl")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(HERE / v, dst, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["patch", "-s", "-p5", "-d", str(dst), "-i", str(PATCH)], capture_output=True)
    d = dst / "conv2d_3x3_program_descriptor.py"
    t = d.read_text()
    if "+ _br0_defines\n            + _abl_defines" not in t and "_br0_defines" in t:
        t = t.replace("            + _br0_defines,\n", "            + _br0_defines\n            + _abl_defines,\n", 1)
    d.write_text(t)
    assert t.count("_abl_defines") >= 4, (v, t.count("_abl_defines"))
    for f in list(dst.rglob("*.rej")) + list(dst.rglob("*.orig")):
        f.unlink()
    nz = HERE / (v + "_ablnz")
    shutil.rmtree(nz, ignore_errors=True)
    shutil.copytree(dst, nz)
    for f in (nz / "kernels").iterdir():
        t = f.read_text()
        f.write_text("".join(l for l in t.splitlines(True) if "MaybeDeviceZoneScope(" not in l))
    print(v, "abl twins built")
