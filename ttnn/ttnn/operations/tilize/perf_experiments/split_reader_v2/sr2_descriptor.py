# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness for `split_reader_v2` — HOST side.

Reconstructs the tilize reader / COMPUTE / writer pipeline for a set of fixed
plans, so the ONLY thing that varies between arms is which RISC issues which
reads and how compute is fed. Nothing here imports the real op: the block model,
the work assignment and the CB geometry are restated so a change in the op cannot
silently move the baseline underneath the measurement. `test_anchor.py` is what
pins the reconstruction against the real op's own number.

Block model (identical to the op, op_design.md §1): a block is 1 tile-row x
`wt_chunk` tile-columns.

  W_REGION (destination-local plans) — each core owns the tile band of its own
      output shard, one W chunk per row (`wt_chunk == shard_wt`).
  W_BLOCKS (interleaved destination) — each core owns a contiguous range of the
      global W-chunk-major block index.

What is NEW versus Perf-1's `split_reader/sr_descriptor.py`:
  * every arm runs a REAL library tilize compute (that was already true) and the
    arm set now contains the CONTROL that isolates the compute-side alternation
    tax (`alt_tax`: one CB, one reader, but the per-block back-to-back helper
    form) — Perf-1 never separated it from the split itself;
  * `split == 3`, the PERIODIC weighted interleave, which delivers Perf-1's 3:1
    ratio at a BOUNDED CB depth (Perf-1's contiguous weighted split needs a CB as
    deep as half the core's blocks, i.e. L1 that scales with the tensor);
  * per-arm CB depths, so the L1 cost of every arm is explicit;
  * arms that keep a real writer duty (`*_bdrain`) for the destination-local
    plans that still need their writer (the output-format pad stamp).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "experiment_kernels"

TILE_H = 32
TILE_W = 32

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

_ELEM_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4}
_TORCH_OF = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def _div_up(a, b):
    return -(-a // b)


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
# `src` / `dst`: ("dram", 0) | ("height", n) | ("width", n)
PLANS = {
    # ---- the two focus plans the coordinator flagged -----------------------
    # Destination-local, aligned DRAM source: the crossover.
    "crossover": dict(shape=[1, 1, 2048, 256], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    # Destination-local, PAGED L1 source (a source shard narrower than a tensor
    # row): the reshard. Reader takes the raw span loop, exactly as the op's R_PAD.
    "reshard": dict(shape=[1, 1, 1024, 256], src=("width", 2), dst=("height", 8), dtype="bf16"),
    # ---- domain sweep around them -----------------------------------------
    # 2x the tile-rows: does the DRAM rule scale or is it one shape's accident?
    "crossover_big": dict(shape=[1, 1, 4096, 256], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    # The READ-TRANSFER-SIZE ladder on an otherwise identical destination-local
    # DRAM plan. `wt_chunk` is pinned to the destination shard's own width, so W
    # is the only knob that moves the per-stick transfer:
    #   256 cols -> 512 B (`crossover`), 512 -> 1 KB, 1024 -> 2 KB, 2048 -> 4 KB.
    # This is the ladder that decides whether the split's domain is stated over
    # the transfer size or over the plan.
    "crossover_512": dict(shape=[1, 1, 2048, 512], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    "crossover_wide": dict(shape=[1, 1, 2048, 1024], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    "crossover_2048": dict(shape=[1, 1, 2048, 2048], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    # The same ladder on the L1 gather: a 2 KB row span, split across two source
    # shard pages of 1 KB each.
    "reshard_wide": dict(shape=[1, 1, 512, 1024], src=("width", 2), dst=("height", 8), dtype="bf16"),
    # An L1 source that is NOT paged: a HEIGHT-sharded source on a DIFFERENT core
    # count than the destination, so a source shard spans a whole tensor row
    # (src_row_pages == 1, regime R_ALIGNED) but lives in another core's L1.
    # This is what separates "predicate on the READ REGIME" from "predicate on the
    # source BUFFER TYPE".
    "gather_h": dict(shape=[1, 1, 2048, 256], src=("height", 4), dst=("height", 8), dtype="bf16"),
    # The same crossover at fp32 — 2x the bytes per block, same block count.
    "crossover_fp32": dict(shape=[1, 1, 2048, 256], src=("dram", 0), dst=("height", 8), dtype="fp32"),
    # ONE block per core (NT_H == num_cores): the degenerate pipeline.
    "crossover_1blk": dict(shape=[1, 1, 256, 256], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    # Deep pipeline: 32 blocks per core.
    "crossover_tall": dict(shape=[1, 1, 8192, 256], src=("dram", 0), dst=("height", 8), dtype="bf16"),
    # The reshard with a 4-core (128 B page) source and at fp32.
    "reshard_w4": dict(shape=[1, 1, 1024, 256], src=("width", 4), dst=("height", 8), dtype="bf16"),
    "reshard_fp32": dict(shape=[1, 1, 1024, 256], src=("width", 2), dst=("height", 8), dtype="fp32"),
    # Small destination-local plans (per-core-overhead regime).
    "small": dict(shape=[1, 1, 512, 64], src=("dram", 0), dst=("height", 4), dtype="bf16"),
    "small_wide": dict(shape=[1, 1, 512, 256], src=("dram", 0), dst=("height", 4), dtype="bf16"),
    # ---- the EXCLUSION: interleaved DRAM on both sides. BRISC is NOT free
    #      (it issues the real output writes).
    "dram": dict(shape=[1, 1, 2048, 2048], src=("dram", 0), dst=("dram", 0), dtype="bf16"),
    "dram_small": dict(shape=[1, 1, 2048, 256], src=("dram", 0), dst=("dram", 0), dtype="bf16"),
}


def plan_dtype(plan):
    return _TORCH_OF[plan.get("dtype", "bf16")]


def elem_bytes(plan):
    return _ELEM_BYTES[plan_dtype(plan)]


def in_tile_bytes(plan):
    return TILE_H * TILE_W * elem_bytes(plan)


def out_tile_bytes(plan):
    return TILE_H * TILE_W * elem_bytes(plan)


def _derive_blocking(nt_h, wt, num_cores, itb, otb):
    """The op's derive_blocking(), restated (interleaved in AND out)."""
    cap = max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // (2 * (itb + otb))))
    n_want = max(1, _div_up(num_cores, nt_h))
    n_pipeline = _div_up(num_cores * PIPELINE_BLOCKS_PER_CORE, nt_h)
    n_transfer_cap = max(1, (wt * itb // TILE_H) // MIN_PIPELINE_READ_BYTES)
    n_want = max(n_want, min(n_pipeline, n_transfer_cap))
    n_want = max(n_want, _div_up(wt, cap))
    n_want = min(n_want, wt)
    n_chunks = next(c for c in range(n_want, wt + 1) if wt % c == 0)
    return wt // n_chunks, n_chunks


def input_memory_config(plan):
    kind, n = plan["src"]
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    shape = plan["shape"]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    if kind == "width":
        shard = [rows, shape[-1] // n]
        layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        shard = [rows // n, shape[-1]]
        layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return ttnn.MemoryConfig(
        layout,
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
    dtype = plan_dtype(plan)
    if memory_config.is_sharded():
        spec = ttnn.TensorSpec(
            shape,
            dtype,
            ttnn.TILE_LAYOUT,
            memory_config.memory_layout,
            memory_config.shard_spec,
            memory_config.buffer_type,
        )
    else:
        spec = ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, buffer_type=memory_config.buffer_type)
    return ttnn.allocate_tensor_on_device(spec, device)


# --------------------------------------------------------------------------
# Geometry of a plan (the same numbers the op would derive)
# --------------------------------------------------------------------------
def geometry(plan):
    shape = plan["shape"]
    eb = elem_bytes(plan)
    rows = 1
    for d in shape[:-1]:
        rows *= d
    nt_h = rows // TILE_H
    wt = shape[-1] // TILE_W

    src_kind, src_n = plan["src"]
    if src_kind == "dram":
        src_page_bytes, src_row_pages, regime = shape[-1] * eb, 1, R_ALIGNED
    elif src_kind == "width":
        src_page_bytes = (shape[-1] // src_n) * eb
        src_row_pages = src_n
        regime = R_SPAN
    else:
        src_page_bytes, src_row_pages, regime = shape[-1] * eb, 1, R_ALIGNED

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
        wt_chunk, n_chunks = _derive_blocking(nt_h, wt, num_cores, in_tile_bytes(plan), out_tile_bytes(plan))
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
        elem_bytes=eb,
    )


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
# `drain` — WHICH role drains the output CB: "brisc" (the op's scheme), "compute",
# or "none" (legal exactly when the output CB is aliased on the resident shard,
# so its ring holds the core's whole output and cb_reserve_back never blocks).
# `depth0` / `depth1` — CB depth IN BLOCKS for cb_in0 / cb_in1. The whole L1 cost
# of an arm is (depth0 + depth1) * wt_chunk * in_tile_bytes; the op ships 2.
# `split` 0 none | 1 stride-2 interleave | 2 contiguous halves | 3 periodic
# weighted interleave (period / share).
# `dyn` puts BOTH data-movement RISCs on NOC_0 under NOC_MODE.DM_DYNAMIC_NOC.
# In the default DEDICATED mode that is ILLEGAL: ncrisc_noc_reads_flushed()
# compares the NIU's hardware response counter against the issuing RISC's OWN
# local counter, so two RISCs sharing one NoC desynchronize. `trid` gives each
# reader its own transaction id so it barriers only on its own reads.
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
    force_bb=0,
    depth0=2,
    depth1=2,
    period=4,
    share=3,
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
        force_bb=force_bb,
        depth0=depth0,
        depth1=depth1,
        period=period,
        share=share,
    )


VARIANTS = {
    # ---- the HONEST baseline: today's op scheme ----------------------------
    # NCRISC reads every block through the library helper; compute is ONE helper
    # call for the whole core; BRISC drains (destination-local) or writes.
    "op_baseline": _v(),
    # ---- controls that decompose the split's win ---------------------------
    # (1) free BRISC of the drain, change NOTHING else.
    "nodrain": _v(drain="none"),
    # (2) the ALTERNATION TAX, alone: still ONE reader and ONE CB, but compute
    #     takes the per-block back-to-back helper form the split forces on it.
    "alt_tax": _v(drain="none", force_bb=1),
    # (3) compute drains instead of BRISC (the form a destination-local plan that
    #     still needs its writer would have to take if the writer were removed).
    "cdrain": _v(drain="compute"),
    # ---- split arms --------------------------------------------------------
    # dedicated dual-NoC 50/50 stride-2 interleave, nobody drains.
    "split_il": _v(brisc_reads=True, two_cbs=1, split=1, drain="none"),
    # the same but with BRISC keeping the drain duty (the out_fill / pad-stamp
    # shape of the integration, where the writer cannot be deleted).
    "split_il_bdrain": _v(brisc_reads=True, two_cbs=1, split=1, drain="brisc"),
    # the same but with COMPUTE draining.
    "split_il_cdrain": _v(brisc_reads=True, two_cbs=1, split=1, drain="compute"),
    # Perf-1's contiguous weighted splits. Kept ONLY to reproduce Perf-1's number
    # — their cb_in depth is HALF THE CORE'S BLOCKS, i.e. L1 that scales with the
    # tensor, which is why `split_p75` below exists.
    "split_w75": _v(brisc_reads=True, two_cbs=1, split=2, drain="none", weight=0.75, depth0=0, depth1=0),
    # NEW: the same 3:1 issue ratio as a PERIODIC interleave — bounded CB depth.
    "split_p75": _v(brisc_reads=True, two_cbs=1, split=3, drain="none", period=4, share=3, depth0=4, depth1=2),
    "split_p67": _v(brisc_reads=True, two_cbs=1, split=3, drain="none", period=3, share=2, depth0=3, depth1=2),
    # The dedicated-NoC ratio sweep: NCRISC's share of the blocks. 1:1 is
    # `split_il`. The two DM RISCs are NOT interchangeable issuers, so the
    # OPTIMAL ratio is a property of the source, and the ratio that maximises the
    # WORST plan is what a predicate-free integration would ship.
    "split_p60": _v(brisc_reads=True, two_cbs=1, split=3, drain="none", period=5, share=3, depth0=3, depth1=2),
    "split_p57": _v(brisc_reads=True, two_cbs=1, split=3, drain="none", period=7, share=4, depth0=4, depth1=3),
    # 2:1 with COMPUTE draining — the predicate-free flavor for a plan that must
    # keep its writer kernel (an output-format pad stamp).
    "split_p67_cdrain": _v(
        brisc_reads=True, two_cbs=1, split=3, drain="compute", period=3, share=2, depth0=3, depth1=2
    ),
    # BOTH readers on NOC_0 (dynamic-NoC mode) + per-RISC transaction id: removes
    # the NOC_1 handicap at the price of sharing one NoC. Needs the raw issue path
    # (the helper owns its barrier and exposes no trid).
    "split_trid": _v(brisc_reads=True, two_cbs=1, split=1, drain="none", dyn=True, trid=True, use_helper=0),
    # The INTEGRATION forms: the same two winners with COMPUTE draining the output
    # CB, so the aliased output CB keeps exactly one consumer (the op's design
    # contract) instead of none. This is the arm the recipe below actually ships.
    "split_trid_cdrain": _v(brisc_reads=True, two_cbs=1, split=1, drain="compute", dyn=True, trid=True, use_helper=0),
    "split_trid_p75": _v(
        brisc_reads=True,
        two_cbs=1,
        split=3,
        drain="none",
        dyn=True,
        trid=True,
        use_helper=0,
        period=4,
        share=3,
        depth0=4,
        depth1=2,
    ),
    # The raw-issue single-reader control for the trid arms (same issue path, one
    # reader) so the trid arm's delta is not confounded with helper-vs-raw.
    "raw_baseline": _v(drain="none", use_helper=0),
    # ---- interleaved destination: BRISC keeps the writes AND takes half the
    #      reads (the arm that asks whether the pattern applies when BRISC is
    #      not free) -----------------------------------------------------------
    "split_rw": _v(brisc_reads=True, two_cbs=1, split=1, drain="brisc"),
    # 7:1 — BRISC keeps the writes and takes only an eighth of the reads.
    "split_rw_p88": _v(brisc_reads=True, two_cbs=1, split=3, drain="brisc", period=8, share=7, depth0=4, depth1=2),
}


def cb_l1_bytes(plan, variant):
    """Input-CB L1 bytes per core for this arm (the op ships `op_baseline`)."""
    cfg = VARIANTS[variant]
    g = geometry(plan)
    d0, d1 = _depths(cfg, g)
    return (d0 + (d1 if cfg["two_cbs"] else 0)) * g["wt_chunk"] * in_tile_bytes(plan)


def _depths(cfg, g):
    n_blocks = g["blocks_per_core"]
    if cfg["split"] == 2:
        n0 = min(n_blocks - 1, max(1, int(round(n_blocks * cfg["weight"]))))
        return n0, n_blocks - n0
    d0 = min(cfg["depth0"], n_blocks) or 1
    d1 = min(cfg["depth1"], n_blocks) or 1
    return d0, d1


def create_program_descriptor(input_tensor, output_tensor, plan, variant):
    cfg = VARIANTS[variant]
    g = geometry(plan)
    wt_chunk = g["wt_chunk"]
    n_blocks = g["blocks_per_core"]
    two_cbs = cfg["two_cbs"]
    split = cfg["split"]
    drain = cfg["drain"]
    itb, otb = in_tile_bytes(plan), out_tile_bytes(plan)

    if drain == "none" and not g["out_local"]:
        raise ValueError("no-drain is only legal with an aliased output CB")
    compute_drain = {"brisc": 0, "compute": 1, "none": 2}[drain]

    n0 = min(n_blocks - 1, max(1, int(round(n_blocks * cfg["weight"])))) if split == 2 else 0
    depth0, depth1 = _depths(cfg, g)

    cores = (
        list(ttnn.get_optimal_worker_cores_for_sharded_tensor(output_tensor))
        if g["out_local"]
        else list(ttnn.corerange_to_cores(core_range_set(g["num_cores"]), g["num_cores"], True))
    )
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})
    tile_descriptor = ttnn.TileDescriptor(TILE_H, TILE_W)
    dtype = plan_dtype(plan)

    # ---- circular buffers -------------------------------------------------
    def _in_cb(index, blocks):
        return ttnn.CBDescriptor(
            total_size=blocks * wt_chunk * itb,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=dtype,
                    page_size=itb,
                    tile=tile_descriptor,
                )
            ],
        )

    cbs = [_in_cb(CB_IN0, depth0)]
    if two_cbs:
        cbs.append(_in_cb(CB_IN1, depth1))

    if g["out_local"]:
        out_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor, core_ranges=all_cores)
        out_cb.total_size = g["shard_ht"] * g["wt"] * otb
        out_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=dtype, page_size=otb, tile=tile_descriptor)
        ]
    else:
        out_cb = ttnn.CBDescriptor(
            total_size=2 * wt_chunk * otb,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=dtype, page_size=otb, tile=tile_descriptor)
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
                g["elem_bytes"],
                g["src_page_bytes"],
                g["src_row_pages"],
                split,
                phase,
                cfg["use_helper"],
                otb,
                trid_of,
                cfg["period"],
                cfg["share"],
            ]
            + list(src_accessor)
            + list(dst_accessor)
        )

    # An interleaved destination needs a real accessor write (1); an aliased one
    # only needs the CB popped (2).
    write_role = 1 if not g["out_local"] else 2
    ncrisc_ct = _dm_ct(1 if cfg["ncrisc_reads"] else 0, 0, CB_IN0, 0, 1 if cfg["trid"] else 0)
    brisc_ct = _dm_ct(
        1 if cfg["brisc_reads"] else 0,
        write_role if drain == "brisc" else 0,
        CB_IN1 if two_cbs else CB_IN0,
        1,
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

    # The precision contract is a FIXED input: the compute config below mirrors
    # the op's own rule for this dtype pair and is IDENTICAL on every arm.
    compute_config = ttnn.ComputeConfigDescriptor()
    if dtype == ttnn.float32:
        compute_config.fp32_dest_acc_en = True
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[CB_IN0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        unpack_modes[CB_IN1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr2_dm.cpp"),
            core_ranges=all_cores,
            compile_time_args=ncrisc_ct,
            runtime_args=ncrisc_rt,
            config=ncrisc_config,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr2_dm.cpp"),
            core_ranges=all_cores,
            compile_time_args=brisc_ct,
            runtime_args=brisc_rt,
            config=brisc_config,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sr2_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=[
                wt_chunk,
                two_cbs,
                split,
                compute_drain,
                cfg["force_bb"],
                0,  # needs_cast — every plan here is same-dtype in/out
                cfg["period"],
                cfg["share"],
            ],
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
