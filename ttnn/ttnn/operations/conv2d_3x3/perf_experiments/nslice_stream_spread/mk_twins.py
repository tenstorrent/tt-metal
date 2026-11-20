# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: build the zone-stripped twins op_base_nz / op_copy_nz (/ grad_nz) of op_base / op_copy (/ grad), for
unzoned verdict numbers (as ablation_r13/ab_unzoned.sh strips the real op).
    python3 mk_twins.py"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
for v in ("op_base", "op_copy", "grad"):
    src = HERE / v
    if not src.exists():
        continue
    dst = HERE / (v + "_nz")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for f in (dst / "kernels").iterdir():
        t = f.read_text()
        f.write_text("".join(l for l in t.splitlines(True) if "MaybeDeviceZoneScope(" not in l))
    print(v, "twin built")
