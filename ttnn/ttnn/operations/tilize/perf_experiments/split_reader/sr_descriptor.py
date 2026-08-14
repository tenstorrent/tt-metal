# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness for design lamp L4 (`split_reader`) — HOST side.

Reconstructs just the tilize DATA-MOVEMENT scheme (reader / compute / drain) for
a handful of fixed plans, so the ONLY thing that varies between arms is which
RISC issues which reads. Nothing here imports the real op: the block model, the
work assignment and the CB geometry are restated so a change in the op cannot
silently move the baseline underneath the measurement.

Block model (identical to the op, op_design.md §1): a block is 1 tile-row x
`wt_chunk` tile-columns. Two work assignments:

  W_REGION (destination-local plans) — each core owns the tile band of its own
      output shard, one W chunk per row (`wt_chunk == shard_wt`).
  W_BLOCKS (interleaved destination) — each core owns a contiguous range of the
      global W-chunk-major block index.

Arms (`variant`):
  baseline          NCRISC reads every block into cb_in0; compute tilizes;
                    BRISC drains (or writes). The op's current scheme.
  drain_compute     as baseline, but COMPUTE drains its own output CB. Isolates
                    the cost of moving the drain off BRISC.
  no_drain          as baseline, but NOBODY drains. Legal only when the output CB
                    is aliased on the resident shard (the ring is the shard).
  split_interleave  NCRISC reads even blocks -> cb_in0, BRISC odd blocks ->
                    cb_in1, compute alternates, nobody drains.
  split_half        NCRISC reads blocks [0,n0) -> cb_in0, BRISC [n0,n) -> cb_in1
                    (each CB deep enough to hold its whole half), nobody drains.
  split_half_drain  split_half with COMPUTE draining (for a plan whose output CB
                    is not aliased and therefore must be drained).
  split_raw         split_interleave with the reads issued raw instead of through
                    read_sticks_for_tilize (prices the helper's per-call prologue).
  split_rw          interleaved DESTINATION only: BRISC does half the reads AND
                    all the writes. The arm that asks whether the pattern applies
                    when BRISC is not free.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_H = 32
TILE_W = 32
ELEM_BYTES = 2  # bf16 everywhere in this bench (the precision contract is fixed)
IN_TILE_BYTES = TILE_H * TILE_W * ELEM_BYTES
OUT_TILE_BYTES = TILE_H * TILE_W * ELEM_BYTES

CB_IN0 = 0
CB_IN1 = 1
CB_OUT = 16

W_BLOCKS = 0
W_REGION = 1

R_ALIGNED = 0
R_SPAN = 1

# The op's own blocking constants, restated (tilize_program_descriptor.py).
CB_L1_BUDGET = 1_048_576
FAST_TILIZE_MAX_W = 255
PIPELINE_BLOCKS_PER_CORE = 4
MIN_PIPELINE_READ_BYTES = 1024


def _div_up(a, b):
    return -(-a // b)


def _derive_blocking(nt_h, wt, num_cores):
    """The op's derive_blocking(), restated for bf16 interleaved-in/interleaved-out."""
    cap = max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // (2 * (IN_TILE_BYTES + OUT_TILE_BYTES))))
    n_want = max(1, _div_up(num_cores, nt_h))
    n_pipeline = _div_up(num_cores * PIPELINE_BLOCKS_PER_CORE, nt_h)
    n_transfer_cap = max(1, (wt * IN_TILE_BYTES // TILE_H) // MIN_PIPELINE_READ_BYTES)
    n_want = max(n_want, min(n_pipeline, n_transfer_cap))
    n_want = max(n_want, _div_up(wt, cap))
    n_want = min(n_want, wt)
    n_chunks = next(c for c in range(n_want, wt + 1) if wt % c == 0)
    return wt // n_chunks, n_chunks


def core_range_set(n):
    """`n` cores in one row-major band starting at (0,0) — the shard grid."""
    if n <= 8:
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))])
    rows, rem = divmod(n, 8)
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, rows - 1))]
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rem - 1, rows)))
    return ttnn.CoreRangeSet(ranges)


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------
PLANS = {
    # Destination-local, aligned source: the crossover the coordinator flagged.
    "crossover": dict(shape=[1, 1, 2048, 256], src=("dram", 0), dst=("height", 8)),
    # Destination-local, PAGED source (a source shard narrower than a tensor row):
    # the reshard. Reader takes the raw span loop, exactly as the op's R_PAD.
    "reshard": dict(shape=[1, 1, 1024, 256], src=("width", 2), dst=("height", 8)),
    # Same shape family as `crossover` at 2x the tile-rows — confirms the DRAM
    # rule scales rather than being one shape's accident.
    "crossover_big": dict(shape=[1, 1, 4096, 256], src=("dram", 0), dst=("height", 8)),
    # Interleaved DRAM on BOTH sides: BRISC is NOT free (it issues the writes).
    "dram": dict(shape=[1, 1, 2048, 2048], src=("dram", 0), dst=("dram", 0)),
    # Small destination-local plans (per-core-overhead regime). `small` sits below
    # the op's own MIN_STREAM_READ_BYTES gate (a 128 B read), `small_wide` above it.
    "small": dict(shape=[1, 1, 512, 64], src=("dram", 0), dst=("height", 4)),
    "small_wide": dict(shape=[1, 1, 512, 256], src=("dram", 0), dst=("height", 4)),
}


def input_memory_config(plan):
    kind, n = plan["src"]
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    shape = plan["shape"]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    shard = [rows, shape[-1] // n]
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set(n), shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def output_memory_config(plan):
    kind, n = plan["dst"]
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    shape = plan["shape"]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    shard = [rows // n, shape[-1]]
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set(n), shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def allocate_output(device, plan):
    memory_config = output_memory_config(plan)
    shape = ttnn.Shape(plan["shape"])
    if memory_config.is_sharded():
        spec = ttnn.TensorSpec(
            shape,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            memory_config.memory_layout,
            memory_config.shard_spec,
            memory_config.buffer_type,
        )
    else:
        spec = ttnn.TensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=memory_config.buffer_type)
    return ttnn.allocate_tensor_on_device(spec, device)


# --------------------------------------------------------------------------
# Geometry of a plan (the same numbers the op would derive)
# --------------------------------------------------------------------------
def geometry(plan):
    shape = plan["shape"]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    nt_h = rows // TILE_H
    wt = shape[-1] // TILE_W

    src_kind, src_n = plan["src"]
    if src_kind == "dram":
        src_page_bytes, src_row_pages, regime = shape[-1] * ELEM_BYTES, 1, R_ALIGNED
    else:
        src_page_bytes = (shape[-1] // src_n) * ELEM_BYTES
        src_row_pages = src_n
        regime = R_SPAN

    dst_kind, dst_n = plan["dst"]
    if dst_kind == "height":
        work_mode = W_REGION
        num_cores = dst_n
        wt_chunk = wt  # the aliased shard's full width, one W chunk per row
        shard_ht = nt_h // dst_n
        blocks_per_core = shard_ht
        regions = [(i * shard_ht, blocks_per_core) for i in range(dst_n)]
        out_local = True
    else:
        work_mode = W_BLOCKS
        num_cores = 64
        wt_chunk, n_chunks = _derive_blocking(nt_h, wt, num_cores)
        total_blocks = nt_h * n_chunks
        per_core = total_blocks // num_cores
        assert per_core * num_cores == total_blocks, "bench plans use an exact split"
        regions = [(i * per_core, per_core) for i in range(num_cores)]
        blocks_per_core = per_core
        shard_ht = 0
        out_local = False

    return dict(
        nt_h=nt_h,
        wt=wt,
        wt_chunk=wt_chunk,
        regime=regime,
        src_page_bytes=src_page_bytes,
        src_row_pages=src_row_pages,
        work_mode=work_mode,
        num_cores=num_cores,
        regions=regions,
        blocks_per_core=blocks_per_core,
        shard_ht=shard_ht,
        out_local=out_local,
    )


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
# Keys: ncrisc_reads / brisc_reads (who issues), two_cbs, split, use_helper, and
# `drain` — WHICH role drains the output CB: "brisc" (the op's scheme), "ncrisc"
# (the mirror, used to price NOC0 vs NOC1 for the SAME single-reader work),
# "compute", or "none" (legal only on an aliased output CB).
#
# `weight` is NCRISC's SHARE of the blocks on a contiguous split (split == 2).
# The two data-movement RISCs are not interchangeable issuers — BRISC's reads
# ride NOC1 — so an even 50/50 split is only optimal when the two NoCs deliver
# this plan's reads at the same rate. `baseline_brisc` measures that ratio; the
# weighted arms spend it.
# `dyn` puts BOTH data-movement RISCs on NOC_0 under NOC_MODE.DM_DYNAMIC_NOC.
# In the default DEDICATED mode that is ILLEGAL: ncrisc_noc_reads_flushed()
# compares the NIU's hardware response counter against the issuing RISC's OWN
# local counter, so two RISCs sharing one NoC desynchronize. DYNAMIC mode sums
# both RISCs' counters (noc_nonblocking_api.h ncrisc_dynamic_noc_reads_flushed),
# which makes the shared NoC safe at the price of a barrier that also waits for
# the other RISC's outstanding reads.
def _v(
    ncrisc_reads=True,
    brisc_reads=False,
    two_cbs=0,
    split=0,
    use_helper=1,
    drain="brisc",
    weight=0.5,
    dyn=False,
    trid=False,
):
    return dict(
        ncrisc_reads=ncrisc_reads,
        brisc_reads=brisc_reads,
        two_cbs=two_cbs,
        split=split,
        use_helper=use_helper,
        drain=drain,
        weight=weight,
        dyn=dyn,
        trid=trid,
    )


VARIANTS = {
    # ---- single-reader arms (the op's current scheme + its mirror) ----------
    "baseline": _v(),
    "baseline_raw": _v(use_helper=0),
    "drain_compute": _v(drain="compute"),
    "no_drain": _v(drain="none"),
    # The SAME single-reader work moved to BRISC/NOC1. This is the control that
    # says whether a "half" issued on NOC1 costs the same as one issued on NOC0.
    "baseline_brisc": _v(ncrisc_reads=False, brisc_reads=True, drain="ncrisc"),
    "baseline_brisc_nodrain": _v(ncrisc_reads=False, brisc_reads=True, drain="none"),
    # ---- split arms --------------------------------------------------------
    "split_interleave": _v(brisc_reads=True, two_cbs=1, split=1, drain="none"),
    "split_half": _v(brisc_reads=True, two_cbs=1, split=2, drain="none"),
    "split_interleave_drain": _v(brisc_reads=True, two_cbs=1, split=1, drain="compute"),
    "split_raw": _v(brisc_reads=True, two_cbs=1, split=1, use_helper=0, drain="none"),
    # Weighted contiguous splits — NCRISC keeps the larger share because its
    # reads ride the faster NoC on this plan.
    "split_w62": _v(brisc_reads=True, two_cbs=1, split=2, drain="none", weight=0.625),
    "split_w75": _v(brisc_reads=True, two_cbs=1, split=2, drain="none", weight=0.75),
    "split_w87": _v(brisc_reads=True, two_cbs=1, split=2, drain="none", weight=0.875),
    # BOTH readers on NOC_0 (dynamic-NoC mode) — removes the NOC_1 handicap at
    # the price of sharing one NoC and a joined read barrier.
    "baseline_dyn": _v(dyn=True, drain="none"),
    "split_interleave_dyn": _v(brisc_reads=True, two_cbs=1, split=1, drain="none", dyn=True),
    "split_half_dyn": _v(brisc_reads=True, two_cbs=1, split=2, drain="none", dyn=True),
    # Shared NOC_0 WITHOUT the joined barrier: each reader tags its reads with
    # its own transaction id and barriers on that id alone. Needs the raw issue
    # path (the helper owns its barrier and exposes no trid).
    "split_trid_dyn": _v(brisc_reads=True, two_cbs=1, split=1, drain="none", dyn=True, trid=True, use_helper=0),
    # ---- interleaved destination: BRISC keeps the writes AND takes half the
    #      reads (the arm that asks whether the pattern applies when BRISC is
    #      not free) -----------------------------------------------------------
    "split_rw": _v(brisc_reads=True, two_cbs=1, split=1, drain="brisc"),
}


def create_program_descriptor(input_tensor, output_tensor, plan, variant):
    cfg = VARIANTS[variant]
    g = geometry(plan)
    wt_chunk = g["wt_chunk"]
    n_blocks = g["blocks_per_core"]
    two_cbs = cfg["two_cbs"]
    split = cfg["split"]
    drain = cfg["drain"]

    if drain == "none" and not g["out_local"]:
        raise ValueError("no-drain is only legal with an aliased output CB")
    compute_drain = {"brisc": 0, "ncrisc": 0, "compute": 1, "none": 2}[drain]

    # NCRISC's share on a contiguous split (at least one block each side).
    n0 = min(n_blocks - 1, max(1, int(round(n_blocks * cfg["weight"])))) if split == 2 else 0

    cores = (
        list(ttnn.get_optimal_worker_cores_for_sharded_tensor(output_tensor))
        if g["out_local"]
        else list(ttnn.corerange_to_cores(core_range_set(g["num_cores"]), g["num_cores"], True))
    )
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})
    tile_descriptor = ttnn.TileDescriptor(TILE_H, TILE_W)

    # ---- circular buffers -------------------------------------------------
    # Input CB depth: 2 blocks for a streaming split, the WHOLE half for the
    # contiguous split (a contiguous half cannot overlap at depth 2 — the second
    # reader would fill two blocks and stall until compute reaches its half).
    def _in_cb(index, blocks):
        return ttnn.CBDescriptor(
            total_size=blocks * wt_chunk * IN_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=ttnn.bfloat16,
                    page_size=IN_TILE_BYTES,
                    tile=tile_descriptor,
                )
            ],
        )

    if split == 2:
        depth0, depth1 = n0, n_blocks - n0
    else:
        depth0 = depth1 = 2
    cbs = [_in_cb(CB_IN0, depth0)]
    if two_cbs:
        cbs.append(_in_cb(CB_IN1, depth1))

    if g["out_local"]:
        out_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor, core_ranges=all_cores)
        out_cb.total_size = g["shard_ht"] * g["wt"] * OUT_TILE_BYTES
        out_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=ttnn.bfloat16,
                page_size=OUT_TILE_BYTES,
                tile=tile_descriptor,
            )
        ]
    else:
        out_cb = ttnn.CBDescriptor(
            total_size=2 * wt_chunk * OUT_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT,
                    data_format=ttnn.bfloat16,
                    page_size=OUT_TILE_BYTES,
                    tile=tile_descriptor,
                )
            ],
        )
    cbs.append(out_cb)

    # ---- kernels ----------------------------------------------------------
    src_accessor = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    dst_accessor = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    def _dm_ct(do_read, do_write, cb_in, phase, trid_of):
        return (
            [
                do_read,
                do_write,
                g["regime"],
                cb_in,
                CB_OUT,
                g["work_mode"],
                TILE_H,
                wt_chunk,
                g["nt_h"],
                g["wt"],
                ELEM_BYTES,
                g["src_page_bytes"],
                g["src_row_pages"],
                split,
                phase,
                cfg["use_helper"],
                OUT_TILE_BYTES,
                trid_of,
            ]
            + list(src_accessor)
            + list(dst_accessor)
        )

    # The drain/write duty: an interleaved destination needs a real accessor
    # write (1); an aliased one only needs the CB popped (2).
    write_role = 1 if not g["out_local"] else 2
    ncrisc_ct = _dm_ct(
        1 if cfg["ncrisc_reads"] else 0, write_role if drain == "ncrisc" else 0, CB_IN0, 0, 1 if cfg["trid"] else 0
    )
    brisc_ct = _dm_ct(
        1 if cfg["brisc_reads"] else 0,
        write_role if drain == "brisc" else 0,
        CB_IN1 if two_cbs else CB_IN0,
        1 if cfg["ncrisc_reads"] else 0,
        2 if cfg["trid"] else 0,
    )

    ncrisc_rt = ttnn.RuntimeArgs()
    brisc_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    for index, core in enumerate(cores):
        origin, blocks = g["regions"][index]
        if g["work_mode"] == W_REGION:
            args = [src_addr, dst_addr, 0, blocks, origin, 0, n0]
        else:
            args = [src_addr, dst_addr, origin, blocks, 0, 0, n0]
        ncrisc_rt[core.x][core.y] = list(args)
        brisc_rt[core.x][core.y] = list(args)
        compute_rt[core.x][core.y] = [blocks, n0]

    if cfg["dyn"]:
        # Both DM RISCs on NOC_0. BOTH must be DYNAMIC: the dynamic barrier sums
        # the two RISCs' issue counters, so a dedicated-mode partner would not
        # publish into the shared counter and the sum would be wrong.
        types = ttnn._ttnn.types
        ncrisc_config = ttnn.DataMovementConfigDescriptor(
            types.DataMovementProcessor.RISCV_1, types.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
        brisc_config = ttnn.DataMovementConfigDescriptor(
            types.DataMovementProcessor.RISCV_0, types.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
    else:
        ncrisc_config = ttnn.ReaderConfigDescriptor()  # NCRISC / NOC_0
        brisc_config = ttnn.WriterConfigDescriptor()  # BRISC / NOC_1

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr_dm.cpp"),
            core_ranges=all_cores,
            compile_time_args=ncrisc_ct,
            runtime_args=ncrisc_rt,
            config=ncrisc_config,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr_dm.cpp"),
            core_ranges=all_cores,
            compile_time_args=brisc_ct,
            runtime_args=brisc_rt,
            config=brisc_config,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=[wt_chunk, two_cbs, split, compute_drain],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(),  # precision contract untouched
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
