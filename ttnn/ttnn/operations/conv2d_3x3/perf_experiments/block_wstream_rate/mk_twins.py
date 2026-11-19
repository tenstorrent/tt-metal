# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: build the twins of op_base / op_copy (/ grad):
  <v>_nz   zones stripped (unzoned verdict numbers, as ablation_r12/ab_unzoned.sh strips the real op);
  <v>_ablnz both;
  <v>_abl  the ring hop's NoC writes stubbed under CONV2D_3X3_ABLATE=scatter (ablation_r12/ablation_hooks.patch's two
           scatter hunks, plus the descriptor's define plumbing where missing), zones kept.
    python3 mk_twins.py"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
for v in ("op_base", "op_copy", "grad2"):
    src = HERE / v
    if not src.exists():
        continue
    for suffix in ("_nz", "_abl", "_ablnz"):
        dst = HERE / (v + suffix)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        if suffix in ("_nz", "_ablnz"):
            for f in (dst / "kernels").iterdir():
                t = f.read_text()
                f.write_text("".join(l for l in t.splitlines(True) if "MaybeDeviceZoneScope(" not in l))
        if suffix == "_nz":
            continue
        r = dst / "kernels" / "conv2d_3x3_reader.cpp"
        t = r.read_text()
        pat = "noc_async_write_one_packet_with_trid(\n                                src + off, get_noc_addr(nx, ny, dst + off), bytes, trid);\n"
        assert t.count(pat) == 1, v
        t = t.replace(pat, "#ifndef CONV2D_3X3_ABLATE_SCATTER\n" + pat + "#endif\n")
        r.write_text(t)
        c = dst / "kernels" / "conv2d_3x3_combine_dataflow.hpp"
        t = c.read_text()
        pat = "    uint32_t row_peer_coords_arg) {\n"
        assert t.count(pat) == 1
        t = t.replace(pat, pat + "#ifdef CONV2D_3X3_ABLATE_SCATTER\n    return;\n#endif\n")
        c.write_text(t)
        d = dst / "conv2d_3x3_program_descriptor.py"
        t = d.read_text()
        if "_abl_defines" not in t:
            old = '    _async_bias_defines = [("NSLICE_ASYNC_BIAS", "1")] if g.k_split and b.nslice and has_bias and not kwf else []\n'
            assert t.count(old) == 1
            t = t.replace(
                old,
                old
                + '    import os as _os\n\n    _abl_defines = [\n        (f"CONV2D_3X3_ABLATE_{t.upper()}", "1") for t in '
                + '_os.environ.get("CONV2D_3X3_ABLATE", "").split(",") if t\n    ]\n',
            )
            t = t.replace(
                "defines=_kwf_dm_defines + _pad_dm_defines + _ring_owner_defines + _sd_dm_defines,",
                "defines=_kwf_dm_defines + _pad_dm_defines + _ring_owner_defines + _sd_dm_defines + _abl_defines,",
            )
            assert t.count("_abl_defines") == 2, v
        d.write_text(t)
    print(v, "twins built")
