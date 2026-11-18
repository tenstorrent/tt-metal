# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core, compute-only bench of conv2d_3x3's K-blocked fp32 packer-L1-accumulation matmul walk.

in0 / in1 are bf8_b tiles sharded in L1 (`kb_store` K-blocks, cycled); the PreKBlockFn pushes each K-block
(stand-in for the op's tilize / weight writer) and, with `dirty_hook`, leaves the packer state the op's tilize
hook leaves (l1-acc off, bf8 pack format, default 1-tile MOP). Partials are fp32, sharded, capacity exactly
bm*bn tiles (the op's contract). Precision config is the op's: LoFi, fp32_dest_acc_en, approx off, half-sync.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

TILE = 32
KERNEL = Path(__file__).parent / "kernels" / "kwalk_compute.cpp"
CB_IN0, CB_IN1, CB_PARTIALS, CB_OUT = 0, 1, 5, 16
TARGETS = {"Interm": 0, "Out": 1, "OutWithRelu": 2}


@dataclass(frozen=True)
class Shape:
    name: str
    bm: int
    bk: int
    bn: int
    nkb: int
    sbh: int = 1
    sbw: int = 4
    caller_owns: bool = False
    target: str = "Interm"
    kb_store: int = 2  # distinct K-blocks resident in L1 (cycled)
    m_iters: int = 4  # M-blocks per launch (in-kernel)
    out_dtype: str = "bfloat8_b"  # Out / OutWithRelu target format (the op's output dtype)


# The two focus walks (per core) handed over by the coordinator.
FOCUS = {
    "res5": Shape("res5", bm=7, bk=3, bn=16, nkb=6, caller_owns=True, target="Interm"),
    "sem_res2": Shape("sem_res2", bm=5, bk=12, bn=8, nkb=6, caller_owns=False, target="Interm"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    impl: int  # 0 helper, 1 hoisted per-tile, 2 hoisted row-MOP
    sbw: int = 0  # 0 = shape's
    skip_compute: bool = False
    force_no_l1_acc: bool = False
    keep_fast_drain: bool = False
    exact: bool = True  # must be bit-identical to baseline


VARIANTS = {
    v.name: v
    for v in (
        Variant("baseline", 0),
        Variant("hoist", 1),
        Variant("hoist_rowmop", 2),
        Variant("hoist_rowmop_fastdrain", 2, keep_fast_drain=True),
        # Diagnostics (mechanism, not candidates).
        Variant("diag_base_sb2", 0, sbw=2),
        Variant("diag_base_sb1", 0, sbw=1),
        Variant("diag_hoist_sb1", 1, sbw=1),
        Variant("diag_base_skipmm", 0, skip_compute=True, exact=False),
        Variant("diag_hoist_skipmm", 1, skip_compute=True, exact=False),
        Variant("diag_rowmop_skipmm", 2, skip_compute=True, exact=False),
        Variant("diag_hoist_noacc", 1, force_no_l1_acc=True, exact=False),
        Variant("diag_rowmop_noacc", 2, force_no_l1_acc=True, exact=False),
    )
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def shard_cfg(h_tiles, w_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def effective_shape(shape: Shape, variant: Variant) -> Shape:
    return replace(shape, sbw=variant.sbw) if variant.sbw else shape


def create_program_descriptor(in0, in1, partials, out, shape: Shape, variant: Variant):
    s = effective_shape(shape, variant)
    if s.bm % s.sbh or s.bn % s.sbw or s.sbh * s.sbw > 4:
        raise ValueError(f"bad subblock {s.sbh}x{s.sbw} for bm={s.bm} bn={s.bn} (fp32 half-sync DEST = 4 tiles)")
    if s.caller_owns and s.target != "Interm":
        raise ValueError("caller_owns requires the Interm target")
    defines = [("SKIP_COMPUTE", "1")] if variant.skip_compute else []
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[
            s.bm,
            s.bk,
            s.bn,
            s.nkb,
            s.sbh,
            s.sbw,
            int(s.caller_owns),
            TARGETS[s.target],
            variant.impl,
            s.m_iters,
            1,  # dirty_hook: the op always runs its tilize hook between K-blocks
            int(variant.force_no_l1_acc),
            int(variant.keep_fast_drain),
        ],
        defines=defines,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN0, in0),
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN1, in1),
        ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIALS, partials),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
