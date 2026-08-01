# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: replace `moe_fused_swiglu`'s binary column reduce tree + root-only SwiGLU
epilogue with a TWO-PHASE REDUCE-SCATTER that distributes the epilogue with it.

This is a MICRO-BENCHMARK of ONE stage of the op — the per-grid-column cross-K reduce of the gate
and up partials, plus the SiLU/multiply epilogue that follows it. It does NOT touch the real op.
Everything upstream (the gate/up matmuls that PRODUCE the partials) and downstream (the `h`
all-gather and the `down` matmul that CONSUME the result) is held constant and trivial: each core's
partials are seeded by one local L1->L1 NoC read of a resident height-sharded bfp8 tensor, and the
finished `h` block is written back to a resident height-sharded bfp8 tensor. The `seed_only` variant
measures that common floor so the reduce delta is attributable.

VARIANTS
--------
`tree` (the honest BASELINE — the op's shipped approach, reconstructed faithfully):
    KGROUPS cores per column; a Hillis-Steele doubling tree (`_reduce_tree` in
    moe_fused_swiglu_program_descriptor.py) funnels every core's full `T = m_eff * HN_PAD` gate AND
    up bfp8 block to the column root. Transport = raw unicast + two counting semaphores (SEM_GO
    parent-invite / SEM_DATA child-signal), REDUCE_SLOTS = 1, and the child's landing address is its
    OWN `get_write_ptr(cb_reduce_*_in)` as a proxy for the parent's (identical CB layout on every
    core + whole-CB push) — the op's exact trick. The root does up to `ceil(log2(K))` in-place
    `add<>` passes on gate and the same on up, with its FINAL gate add carrying SiLU on the PACKER
    thread via `add_bias_bcast_rows<Elementwise, ..., SiluActivation>` walked as `m_eff` SEPARATE
    calls (the helper's bias index does not advance with in0_subblock, bias_add_helpers.inl:141),
    then the SwiGLU `mul` into cb_h_local. Non-roots `copy` both blocks into send CBs first.
    Every eltwise pass uses the op's `blk_in`/`blk_out`/`blk_shape` spelling
    (PerChunk + OperandKind::Block + EltwiseShape::tiles(n, ELTWISE_BLK)) so the baseline gets the
    graduated Perf-1 DEST window and is NOT a straw man.

`rs_*` (the CANDIDATE family — reduce-scatter):
    Every core in the column owns a DISJOINT slice of the T-tile block. All contributors PUSH their
    slice of gate and up straight into each worker's landing CB (raw unicast + one counting
    semaphore, the same transport primitives and the same landing-address proxy as the tree edge).
    Each worker reduces only its own slice over all contributors with the op's in-place
    `add<blk_in(acc), blk_in(gather), blk_out(acc)>` pattern.

    ROUND-1 HANG, ROOT-CAUSED (and it was NOT the in-place add): round 1's `reduce_tree_shape`
    bench hung on `add<in(cb), in(cb2), out(cb)>` and worked around it with a chain of single-use CBs
    costing 89-714 KB. The in-place add is fine — `eltwise_chain` pops its inputs inside
    `elem_apply_compute` and only reserves the output later in `elem_apply_pack`
    (eltwise_chain.inl:2570-2585), so with the op's PerChunk policies a 1-deep accumulator CB
    recycles its own pages in ring order, which is exactly why the real op's 48-page `cb_gate_acc`
    works every dispatch. The actual bug is a BENCH-ONLY CB-OWNERSHIP violation:

        `cb_push_back` OVERWRITES the shared L1 `tiles_received` word with the PUSHING RISC-V's own
        LOCAL push count. So exactly ONE RISC-V may ever push a given CB.

    Round 1 (and the first cut of this bench) seeded the accumulator from the READER, then let the
    in-place add make PACK a SECOND pusher whose local count started at 0 — PACK's first push drove
    the shared received count BACKWARDS (12 -> 8) and the consumer's `cb_wait_front` deadlocked on
    the second DEST window. Reproduced here at K=4 (only the one core that had a child hung; the
    childless leaves, which do no in-place add, completed).
    FIX, which also keeps the pass count byte-identical to the op: the reader seeds
    `cb_gate_in` / `cb_up_in`, and the FIRST reduce add is OUT-OF-PLACE
    (`gate_in + rg_in -> gate_acc`), so PACK is the sole pusher of the accumulator and the root still
    does exactly `fan_in` gate passes and `fan_in` up passes, as the op does. In the CANDIDATE the
    accumulator is never written by compute at all (the writer sends straight out of it), so it needs
    no staging pair — which is one reason the candidate's L1 is SMALLER.

    Two slice axes, chosen HOST-SIDE only (identical kernels):
      - `flat`: balanced split of the T tile indices over min(K, T) workers (ragged 5/5/../4/4 at
        T=48, K=10).
      - `m`:    balanced split of the m_eff TOKEN tile-rows over min(K, m_eff) workers, so a worker
        owns whole rows of HN_PAD tiles. At m_eff <= K this is exactly the strided
        "core s owns rows s, s+K, ..." assignment, and it leaves K - m_eff cores idle.
      Both are CONTIGUOUS tile ranges because the gate/up block layout is `m * HN_PAD + n`
      (OUT_SUBBLOCK_H_GU == 1, SubblockMajor), so every gather leg is ONE coalesced transaction.
    Two epilogue placements:
      - `epi` (distributed): the worker's LAST gate add is the SiLU-fused `add_bias_bcast_rows`, in
        `ceil(A / 8)` calls instead of the root's `m_eff` (a slice of A <= 8 tiles is ONE DEST
        window, so the m_eff-call walk collapses for free), then the SwiGLU `mul` on its own slice,
        then the finished `h` slice is unicast straight into the root's cb_h_local at its tile
        offset — the gather IS the assembly, no copy and no add at the root.
      - `noepi` (epilogue stays at the root): contributors are the K-1 NON-root cores; workers
        reduce their slice of those and scatter the partial sums into the root's cb_reduce_*_full;
        the root then folds in its OWN partial with the same m_eff-call SiLU bias-add walk and does
        the full-block `mul`. Isolates how much of the win is the adds vs. the epilogue.

PRECISION CONTRACT — FROZEN, and identical for every variant (never a lever here):
math_fidelity=LoFi, math_approx_mode=True, fp32_dest_acc_en=False, dst_full_sync_en=False,
bfp8_pack_precise=True; gate/up partials and `h` are bfloat8_b.
"""

from dataclasses import dataclass

import ttnn

TILE = 32
BFP8_TILE_BYTES = 1088  # bfloat8_b 32x32 tile, matches the op's CB table
DEST_LIMIT_TILES = 8  # DEST_AUTO_LIMIT_TILES in the op's descriptor (fp32_dest_acc_en=False)
ELTWISE_BLK = 8  # the graduated Perf-1 DEST window; SAME in baseline and candidate

# ---- CB indices (one flat namespace; each variant allocates only what it uses) ----
CB_GATE_ACC = 0  # my gate partial (seeded); baseline: also the in-place reduce accumulator
CB_UP_ACC = 1
CB_RG_IN = 2  # baseline: incoming child gate partial (REDUCE_SLOTS = 1)
CB_RU_IN = 3
CB_GATE_SEND = 4  # baseline: non-root's block handed to the writer
CB_UP_SEND = 5
CB_GATE_SILU = 6  # SiLU(sum(gate)) — full block (baseline / noepi) or slice (epi)
CB_H_LOCAL = 7  # the column's finished h block (root)
CB_GG = 8  # candidate: gathered gate slices, one slot per contributor
CB_GU = 9  # candidate: gathered up slices
CB_SACC_G = 10  # candidate: this worker's gate slice accumulator (in-place)
CB_SACC_U = 11
CB_HSLICE = 12  # candidate `epi`: this worker's finished h slice
CB_SEND_G = 13  # candidate `noepi`: this worker's reduced gate slice, handed to the writer
CB_SEND_U = 14
CB_RG_FULL = 15  # candidate `noepi`: root's landing block for the scattered gate sums
CB_RU_FULL = 16
CB_GATE_IN = 17  # baseline ONLY: reader-seeded staging, so PACK owns cb_gate_acc's push (see header)
CB_UP_IN = 18
CB_USUM = 19  # candidate `noepi` root: sum(up); a FRESH CB because cb_ru_full's pusher is the reader
CB_H_LAND = 20  # candidate `unfused`: separate h landing that the root then COPIES into cb_h_local

SEM_GO = 0  # baseline: parent -> child "a landing slot is free". candidate: the peer invite.
SEM_DATA = 1  # both: contributor -> receiver "my data landed"
SEM_H = 2  # candidate: worker -> root "my h (or partial-sum) slice landed"


# ---------------------------------------------------------------------------
# Host plumbing
# ---------------------------------------------------------------------------


def _column_cores(k, x=0, y0=0):
    return [(x, y0 + row) for row in range(k)]


def _grid_cores(k, ncols):
    """All HGROUPS x KGROUPS worker cores, enumerated in the SHARD order a ROW_MAJOR height-sharded
    tensor uses over the same CoreRangeSet: shard index = row * ncols + col."""
    return [(col, row) for row in range(k) for col in range(ncols)]


def _virtual(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _core_range(cores):
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))])


def _cb(cb_index, core_ranges, num_pages, page_bytes=BFP8_TILE_BYTES, dtype=ttnn.bfloat8_b):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_bytes)],
    )


def _kernel(source, core_ranges, compile_time_args, runtime_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=config,
    )


def make_sharded_config(device, k, n_tiles, ncols=1):
    return ttnn.create_sharded_memory_config(
        shape=(TILE, n_tiles * TILE),
        core_grid=_core_range(_grid_cores(k, ncols)),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _compute_config():
    # PRECISION CONTRACT — byte-identical to moe_fused_swiglu.default_compute_kernel_config().
    # A fixed input to EVERY variant; never touched for speed.
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        bfp8_pack_precise=True,
    )


@dataclass(frozen=True)
class Layout:
    """`ncols` independent KGROUPS-deep reduce columns, laid out exactly as the op's HGROUPS x
    KGROUPS worker grid. Every collective stays INSIDE its column, so `ncols > 1` adds nothing but
    the real thing this bench otherwise cannot see: the other columns' concurrent NoC traffic."""

    k: int
    m_eff: int
    hn_pad: int
    ncols: int

    @property
    def t_tiles(self):
        return self.m_eff * self.hn_pad

    @property
    def core_range(self):
        return _core_range(_grid_cores(self.k, self.ncols))

    def col_core(self, col, row):
        return (col, row)

    def shard_index(self, col, row):
        return row * self.ncols + col


def build_layout(k, m_eff, hn_pad, ncols=1):
    return Layout(k, m_eff, hn_pad, ncols)


def hillis_steele_tree(k, root=0):
    """The op's SHIPPED tree (`_reduce_tree`), one grid column of `k` rows. Root fan-in is
    ceil(log2(k)) because the accumulator (relative index 0) stays the SAME physical node at every
    doubling level."""
    info = {}
    for y in range(k):
        r = (y - root) % k
        children = []
        s = 1
        while s < k:
            if r % (2 * s) == 0 and r + s < k:
                children.append((root + r + s) % k)
            s *= 2
        parent = None if r == 0 else (root + r - (r & (-r))) % k
        info[y] = {"parent": parent, "children": children}
    return info


def _largest_divisor_le(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _lcm(a, b):
    from math import gcd

    return a * b // gcd(a, b)


def slice_plan(kind, k, m_eff, hn_pad):
    """(assigned_tiles per row, offset_tiles per row, slice_cb_pages). `assigned == 0` = IDLE core.

    THE CB PAGE-COUNT RULE that shapes all three plans. A CB's write pointer advances by the pages
    pushed and wraps only at the CB END, so a block that starts mid-CB and runs past the end
    SILENTLY OVERRUNS into the next CB: for every CB the kernel cycles in blocks of `B` pages, the
    CB's page count `P` must satisfy `P % B == 0`. Every core shares ONE CBDescriptor (a per-core-
    range split is not an option — the allocator collapses a new sub-range's address to the CB base),
    so a plan where different workers own DIFFERENT slice sizes forces `P = lcm(sizes)`.
    Measured, before this was understood: the ragged 5/5/5/5/5/5/5/5/4/4 split with `P = 5` scored
    PCC 0.709-0.886 (the assigned-4 workers' packs wrapped 3 pages past their CB) while every
    uniform-slice plan scored >= 0.9955.

      - `flat`   : the largest divisor of T that is <= K workers, so every slice is T/W — UNIFORM,
                   and the slice CBs stay at their natural `T/W` pages.
      - `m`      : the largest divisor of m_eff that is <= K workers, each owning whole token
                   tile-rows (`m_eff/W * HN_PAD` tiles). At m_eff <= K this is exactly the strided
                   "core s owns rows s, s+K, ..." assignment and it idles K - m_eff cores.
      - `ragged` : min(K, T) workers with the balanced ragged split — MORE workers and a SHORTER
                   critical slice than `flat`, paid for with `lcm(a_min, a_max)`-page slice CBs.
    """
    t = m_eff * hn_pad
    if kind == "flat":
        w = _largest_divisor_le(t, k)
        assigned = [t // w] * w + [0] * (k - w)
    elif kind == "m":
        w = _largest_divisor_le(m_eff, k)
        assigned = [(m_eff // w) * hn_pad] * w + [0] * (k - w)
    elif kind == "ragged":
        w = min(k, t)
        base, rem = divmod(t, w)
        assigned = [base + (1 if i < rem else 0) for i in range(w)] + [0] * (k - w)
    else:
        raise ValueError(f"unknown slice kind {kind!r}")
    offsets, acc = [], 0
    for a in assigned:
        offsets.append(acc)
        acc += a
    assert acc == t, (assigned, t)
    live = [a for a in assigned if a]
    slice_pages = live[0]
    for a in live[1:]:
        slice_pages = _lcm(slice_pages, a)
    return assigned, offsets, slice_pages


# ===========================================================================
# BASELINE — the op's shipped tree
# ===========================================================================

_TREE_READER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// PARENT side of the reduce tree — mirrors moe_fused_swiglu_reader.cpp's `reader_reduce` zone at
// REDUCE_SLOTS = 1: reserve the whole landing CB, invite one child, wait for its data, push whole.
// The seed prologue stands in for the gate/up matmuls (held constant across every variant).
void kernel_main() {
    constexpr uint32_t cb_gate_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rg_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ru_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h_local = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t PB = get_compile_time_arg_val(6);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(7);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(8);
    constexpr uint32_t SEED_ONLY = get_compile_time_arg_val(9);

    const uint32_t gate_addr = get_arg_val<uint32_t>(0);
    const uint32_t up_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_root = get_arg_val<uint32_t>(3);
    const uint32_t num_children = get_arg_val<uint32_t>(4);
    constexpr uint32_t RT_CHILDREN = 5;

    constexpr uint32_t BYTES = T * PB;
    const uint32_t mx = my_x[noc_index];
    const uint32_t my = my_y[noc_index];

    // ---- seed: my gate/up partials are ALREADY produced (the matmul is out of scope). They land in
    // cb_*_in, NOT in the accumulator, because the accumulator's ONLY legal pusher is PACK (see the
    // CB-ownership note in this file's header). The reduce's FIRST add reads cb_*_in out of place, so
    // the pass count is byte-identical to the op's. ----
    cb_reserve_back(cb_gate_in, T);
    cb_reserve_back(cb_up_in, T);
    noc_async_read(get_noc_addr(mx, my, gate_addr), get_write_ptr(cb_gate_in), BYTES);
    noc_async_read(get_noc_addr(mx, my, up_addr), get_write_ptr(cb_up_in), BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_gate_in, T);
    cb_push_back(cb_up_in, T);

    if constexpr (SEED_ONLY) {
        return;  // floor variant: the two seed reads and nothing else
    }

    volatile tt_l1_ptr uint32_t* sem_data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_data_id)));
    const uint32_t sem_go = static_cast<uint32_t>(get_semaphore(sem_go_id));
    uint32_t arrivals = 0;
    for (uint32_t c = 0; c < num_children; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 0);
        const uint32_t cy = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 1);
        cb_reserve_back(cb_rg_in, T);
        cb_reserve_back(cb_ru_in, T);
        noc_semaphore_inc(get_noc_addr(cx, cy, sem_go), 1);
        arrivals += 1;
        noc_semaphore_wait_min(sem_data_ptr, arrivals);
        cb_push_back(cb_rg_in, T);
        cb_push_back(cb_ru_in, T);
    }

    if (is_root) {
        cb_wait_front(cb_h_local, T);
        noc_async_write(get_read_ptr(cb_h_local), get_noc_addr(mx, my, h_addr), BYTES);
        noc_async_write_barrier();
        cb_pop_front(cb_h_local, T);
    }
    // The invite increments above are non-posted atomics; on a childless / non-root core nothing
    // after them drains the queue, and the firmware's exit-time
    // ASSERT(ncrisc_noc_nonposted_atomics_flushed) would trip. One flush per kernel, not per invite
    // (a per-invite barrier would serialise the tree edge and make the baseline a straw man).
    noc_async_atomic_barrier();
}
"""

_TREE_WRITER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// CHILD side — mirrors moe_fused_swiglu_writer.cpp's `writer_reduce_child` zone. Landing address =
// this core's OWN get_write_ptr(cb_reduce_*_in): every core has the identical CB layout and the CB
// is pushed WHOLE, so the write pointer is always the CB base. The trailing
// noc_async_atomic_barrier() is required because the remote increment is this kernel's last NoC op
// and the firmware's exit-time nonposted-atomics assert would otherwise trip.
void kernel_main() {
    constexpr uint32_t cb_gate_send = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up_send = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rg_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ru_in = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t PB = get_compile_time_arg_val(5);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(6);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(7);
    constexpr uint32_t SEED_ONLY = get_compile_time_arg_val(8);

    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t parent_x = get_arg_val<uint32_t>(1);
    const uint32_t parent_y = get_arg_val<uint32_t>(2);

    if constexpr (SEED_ONLY) {
        return;
    }
    if (is_root) {
        return;
    }

    constexpr uint32_t BYTES = T * PB;
    volatile tt_l1_ptr uint32_t* sem_go_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_go_id)));

    cb_wait_front(cb_gate_send, T);
    cb_wait_front(cb_up_send, T);
    noc_semaphore_wait_min(sem_go_ptr, 1);
    noc_async_write(
        get_read_ptr(cb_gate_send), get_noc_addr(parent_x, parent_y, get_write_ptr(cb_rg_in)), BYTES);
    noc_async_write(get_read_ptr(cb_up_send), get_noc_addr(parent_x, parent_y, get_write_ptr(cb_ru_in)), BYTES);
    noc_async_write_barrier();
    noc_semaphore_inc(
        get_noc_addr(parent_x, parent_y, static_cast<uint32_t>(get_semaphore(sem_data_id))), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_gate_send, T);
    cb_pop_front(cb_up_send, T);
}
"""

_TREE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(0);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(1);
constexpr uint32_t cb_rg_in = get_compile_time_arg_val(2);
constexpr uint32_t cb_ru_in = get_compile_time_arg_val(3);
constexpr uint32_t cb_gate_send = get_compile_time_arg_val(4);
constexpr uint32_t cb_up_send = get_compile_time_arg_val(5);
constexpr uint32_t cb_gate_silu = get_compile_time_arg_val(6);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(7);
constexpr uint32_t T = get_compile_time_arg_val(8);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(9);
constexpr uint32_t M_EFF = get_compile_time_arg_val(10);
constexpr uint32_t BLK = get_compile_time_arg_val(11);
constexpr uint32_t SEED_ONLY = get_compile_time_arg_val(12);
constexpr uint32_t cb_gate_in = get_compile_time_arg_val(13);
constexpr uint32_t cb_up_in = get_compile_time_arg_val(14);

// PERF-1 blocked-eltwise spelling, copied verbatim from moe_fused_swiglu_compute.cpp. Anything else
// makes eltwise_chain SILENTLY clamp block_size to 1 and turns this baseline into a straw man.
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
ALWI auto blk_shape(uint32_t n) { return EltwiseShape::tiles(n, BLK); }

// The op's `compute_reduce` + `compute_swiglu` zones for one column, verbatim in structure:
// per child an in-place gate add and an in-place up add; the ROOT's LAST gate add instead runs
// add_bias_bcast_rows with SiLU on the packer thread, walked as M_EFF separate 1 x HN_PAD calls.
void kernel_main() {
    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t num_children = get_arg_val<uint32_t>(1);

    if constexpr (SEED_ONLY) {
        return;
    }

    compute_kernel_hw_startup(cb_gate_in, cb_rg_in, cb_gate_acc);
    ActivationInitHelper<KernelActivation::SILU>::init();

    CircularBuffer gate_in_buf(cb_gate_in), gate_buf(cb_gate_acc), rg_buf(cb_rg_in), silu_buf(cb_gate_silu);

    for (uint32_t c = 0; c < num_children; ++c) {
        const bool final_child = (c + 1 == num_children);
        const bool first_child = (c == 0);
        if (is_root && final_child) {
            // Root's LAST gate add: SiLU rides the PACKER thread. One call per token tile-row
            // because the helper's bias index does not advance with in0_subblock
            // (bias_add_helpers.inl:141) — this M_EFF-call walk is exactly what the op pays today.
            rg_buf.wait_front(T);
            for (uint32_t m = 0; m < M_EFF; ++m) {
                add_bias_bcast_rows<
                    BiasBroadcast::Elementwise,
                    OutputCBLayout::SubblockMajor,
                    bias_add_config::NoPostBias,
                    SiluActivation>(
                    first_child ? gate_in_buf : gate_buf,
                    rg_buf,
                    silu_buf,
                    BiasAddShape::of(1, 1, 1, HN_PAD),
                    {},
                    m * HN_PAD);
            }
            rg_buf.pop_front(T);
        } else if (first_child) {
            add<blk_in(cb_gate_in), blk_in(cb_rg_in), blk_out(cb_gate_acc)>(blk_shape(T));
        } else {
            add<blk_in(cb_gate_acc), blk_in(cb_rg_in), blk_out(cb_gate_acc)>(blk_shape(T));
        }
        if (first_child) {
            add<blk_in(cb_up_in), blk_in(cb_ru_in), blk_out(cb_up_acc)>(blk_shape(T));
        } else {
            add<blk_in(cb_up_acc), blk_in(cb_ru_in), blk_out(cb_up_acc)>(blk_shape(T));
        }
    }

    if (is_root) {
        mul<blk_in(cb_gate_silu), blk_in(cb_up_acc), blk_out(cb_h_local)>(blk_shape(T));
    } else if (num_children == 0) {
        copy<blk_in(cb_gate_in), blk_out(cb_gate_send)>(blk_shape(T));
        copy<blk_in(cb_up_in), blk_out(cb_up_send)>(blk_shape(T));
    } else {
        copy<blk_in(cb_gate_acc), blk_out(cb_gate_send)>(blk_shape(T));
        copy<blk_in(cb_up_acc), blk_out(cb_up_send)>(blk_shape(T));
    }
}
"""


def create_tree_descriptor(device, gate_t, up_t, h_t, layout, *, seed_only=False):
    k, t = layout.k, layout.t_tiles
    tree = hillis_steele_tree(k, root=0)
    cr = layout.core_range
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()

    for col in range(layout.ncols):
        for row in range(k):
            x, y = layout.col_core(col, row)
            info = tree[row]
            is_root = 1 if info["parent"] is None else 0
            coords = []
            for c in info["children"]:
                coords += list(_virtual(device, *layout.col_core(col, c)))
            reader_rt[x][y] = [
                gate_t.buffer_address(),
                up_t.buffer_address(),
                h_t.buffer_address(),
                is_root,
                len(info["children"]),
            ] + coords
            pvx, pvy = (0, 0) if info["parent"] is None else _virtual(device, *layout.col_core(col, info["parent"]))
            writer_rt[x][y] = [is_root, pvx, pvy]
            compute_rt[x][y] = [is_root, len(info["children"])]

    so = 1 if seed_only else 0
    cbs = [
        _cb(CB_GATE_ACC, cr, t),
        _cb(CB_UP_ACC, cr, t),
        _cb(CB_RG_IN, cr, t),
        _cb(CB_RU_IN, cr, t),
        _cb(CB_GATE_SEND, cr, t),
        _cb(CB_UP_SEND, cr, t),
        _cb(CB_GATE_SILU, cr, t),
        _cb(CB_H_LOCAL, cr, t),
        # BENCH-ONLY staging (see the CB-ownership note in the header): in the real op the gate/up
        # matmul packs straight into cb_gate_acc, so these two do not exist there. They are excluded
        # from the op-equivalent L1 accounting.
        _cb(CB_GATE_IN, cr, t),
        _cb(CB_UP_IN, cr, t),
    ]
    sems = [
        ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=cr, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=cr, initial_value=0),
    ]
    reader = _kernel(
        _TREE_READER,
        cr,
        [CB_GATE_IN, CB_UP_IN, CB_RG_IN, CB_RU_IN, CB_H_LOCAL, t, BFP8_TILE_BYTES, SEM_GO, SEM_DATA, so],
        reader_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    writer = _kernel(
        _TREE_WRITER,
        cr,
        [CB_GATE_SEND, CB_UP_SEND, CB_RG_IN, CB_RU_IN, t, BFP8_TILE_BYTES, SEM_GO, SEM_DATA, so],
        writer_rt,
        ttnn.WriterConfigDescriptor(),
    )
    compute = _kernel(
        _TREE_COMPUTE,
        cr,
        [
            CB_GATE_ACC,
            CB_UP_ACC,
            CB_RG_IN,
            CB_RU_IN,
            CB_GATE_SEND,
            CB_UP_SEND,
            CB_GATE_SILU,
            CB_H_LOCAL,
            t,
            layout.hn_pad,
            layout.m_eff,
            ELTWISE_BLK,
            so,
            CB_GATE_IN,
            CB_UP_IN,
        ],
        compute_rt,
        _compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)


def cb_pages_tree(layout, *, op_equivalent=True):
    """8 CBs of T pages. The two bench-only staging CBs are excluded when `op_equivalent` (the real
    op's matmul packs directly into cb_gate_acc, so they have no counterpart there)."""
    return (8 if op_equivalent else 10) * layout.t_tiles


# ===========================================================================
# CANDIDATE — two-phase reduce-scatter
# ===========================================================================

_RS_READER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// RECEIVER side of the reduce-scatter. Reserve the landing CBs WHOLE (so every peer's
// own-write-pointer proxy is the CB base on every core), invite every peer once (this is the
// generalisation of the tree's SEM_GO: with m_blocks > 1 the landing CB must be known-free before a
// contributor may write into it), then wait for all contributors and push WHOLE.
void kernel_main() {
    constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up_acc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gg = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gu = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h_local = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rg_full = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ru_full = get_compile_time_arg_val(6);
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t PB = get_compile_time_arg_val(8);
    constexpr uint32_t GATHER_CAP = get_compile_time_arg_val(9);
    constexpr uint32_t NC = get_compile_time_arg_val(10);       // contributors landing in my CBs
    constexpr uint32_t K = get_compile_time_arg_val(11);        // column height (invite fan-out)
    constexpr uint32_t DIST_EPI = get_compile_time_arg_val(12);
    constexpr uint32_t sem_invite_id = get_compile_time_arg_val(13);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_h_id = get_compile_time_arg_val(15);
    constexpr uint32_t FUSE_H = get_compile_time_arg_val(16);
    constexpr uint32_t cb_h_land = get_compile_time_arg_val(17);

    const uint32_t gate_addr = get_arg_val<uint32_t>(0);
    const uint32_t up_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_root = get_arg_val<uint32_t>(3);
    const uint32_t assigned = get_arg_val<uint32_t>(4);
    const uint32_t num_workers = get_arg_val<uint32_t>(5);
    constexpr uint32_t RT_PEERS = 6;  // K (vx, vy) pairs — the whole column

    constexpr uint32_t BYTES = T * PB;
    const uint32_t mx = my_x[noc_index];
    const uint32_t my = my_y[noc_index];

    // ---- seed: my gate/up partials are ALREADY produced (identical to the baseline's prologue) ----
    cb_reserve_back(cb_gate_acc, T);
    cb_reserve_back(cb_up_acc, T);
    noc_async_read(get_noc_addr(mx, my, gate_addr), get_write_ptr(cb_gate_acc), BYTES);
    noc_async_read(get_noc_addr(mx, my, up_addr), get_write_ptr(cb_up_acc), BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_gate_acc, T);
    cb_push_back(cb_up_acc, T);

    // ---- reserve every landing region BEFORE inviting ----
    if (assigned) {
        cb_reserve_back(cb_gg, GATHER_CAP);
        cb_reserve_back(cb_gu, GATHER_CAP);
    }
    if (is_root) {
        if constexpr (DIST_EPI) {
            cb_reserve_back(FUSE_H ? cb_h_local : cb_h_land, T);
        } else {
            cb_reserve_back(cb_rg_full, T);
            cb_reserve_back(cb_ru_full, T);
        }
    }
    const uint32_t sem_invite = static_cast<uint32_t>(get_semaphore(sem_invite_id));
    for (uint32_t p = 0; p < K; ++p) {
        const uint32_t px = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 0);
        const uint32_t py = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 1);
        noc_semaphore_inc(get_noc_addr(px, py, sem_invite), 1);
    }
    noc_async_atomic_barrier();

    // ---- collect my contributors ----
    if (assigned) {
        volatile tt_l1_ptr uint32_t* data_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_data_id)));
        noc_semaphore_wait_min(data_ptr, NC);
        cb_push_back(cb_gg, GATHER_CAP);
        cb_push_back(cb_gu, GATHER_CAP);
    }

    // ---- root: the scatter-gather lands here ----
    if (is_root) {
        volatile tt_l1_ptr uint32_t* h_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_h_id)));
        noc_semaphore_wait_min(h_ptr, num_workers);
        if constexpr (DIST_EPI && FUSE_H) {
            // The gather IS the assembly: the workers' finished h slices already tile cb_h_local, so
            // nothing on the compute thread produces it and there is no CB handoff to wait on.
            noc_async_write(get_write_ptr(cb_h_local), get_noc_addr(mx, my, h_addr), BYTES);
            noc_async_write_barrier();
        } else {
            if constexpr (DIST_EPI) {
                cb_push_back(cb_h_land, T);  // `unfused`: compute must still COPY it into cb_h_local
            } else {
                cb_push_back(cb_rg_full, T);
                cb_push_back(cb_ru_full, T);
            }
            cb_wait_front(cb_h_local, T);
            noc_async_write(get_read_ptr(cb_h_local), get_noc_addr(mx, my, h_addr), BYTES);
            noc_async_write_barrier();
            cb_pop_front(cb_h_local, T);
        }
    }
}
"""

_RS_WRITER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// SENDER side of the reduce-scatter, plus the finished-slice scatter to the root. Same primitives
// as the tree edge (raw unicast + counting semaphore + own-write-pointer landing proxy), just
// K disjoint destinations instead of one parent. Each leg is ONE coalesced transaction because a
// slice is a CONTIGUOUS tile range in the `m * HN_PAD + n` gate/up layout.
//
// The contributor reads cb_gate_acc / cb_up_acc DIRECTLY (a dataflow kernel is a first-class CB
// consumer — CB counters live in L1), which is what deletes the op's cb_gate_send / cb_up_send pair
// AND the two full-block copies every non-root core does today.
void kernel_main() {
    constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up_acc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gg = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gu = get_compile_time_arg_val(3);
    constexpr uint32_t cb_hslice = get_compile_time_arg_val(4);
    constexpr uint32_t cb_send_g = get_compile_time_arg_val(5);
    constexpr uint32_t cb_send_u = get_compile_time_arg_val(6);
    constexpr uint32_t cb_h_local = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rg_full = get_compile_time_arg_val(8);
    constexpr uint32_t cb_ru_full = get_compile_time_arg_val(9);
    constexpr uint32_t T = get_compile_time_arg_val(10);
    constexpr uint32_t PB = get_compile_time_arg_val(11);
    constexpr uint32_t K = get_compile_time_arg_val(12);
    constexpr uint32_t DIST_EPI = get_compile_time_arg_val(13);
    constexpr uint32_t sem_invite_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(15);
    constexpr uint32_t sem_h_id = get_compile_time_arg_val(16);
    constexpr uint32_t FUSE_H = get_compile_time_arg_val(17);
    constexpr uint32_t cb_h_land = get_compile_time_arg_val(18);

    const uint32_t is_contributor = get_arg_val<uint32_t>(0);
    const uint32_t my_slot = get_arg_val<uint32_t>(1);  // my slot index in every worker's landing CB
    const uint32_t assigned = get_arg_val<uint32_t>(2);
    const uint32_t offset = get_arg_val<uint32_t>(3);
    const uint32_t root_x = get_arg_val<uint32_t>(4);
    const uint32_t root_y = get_arg_val<uint32_t>(5);
    constexpr uint32_t RT_DESTS = 6;  // K quadruples: vx, vy, dst_offset, dst_assigned

    volatile tt_l1_ptr uint32_t* invite_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_invite_id)));

    if (is_contributor) {
        // Every peer has reserved its landing CBs (K invites, one per column core).
        noc_semaphore_wait_min(invite_ptr, K);
        cb_wait_front(cb_gate_acc, T);
        cb_wait_front(cb_up_acc, T);
        const uint32_t gsrc = get_read_ptr(cb_gate_acc);
        const uint32_t usrc = get_read_ptr(cb_up_acc);
        const uint32_t gg_base = get_write_ptr(cb_gg);
        const uint32_t gu_base = get_write_ptr(cb_gu);
        for (uint32_t d = 0; d < K; ++d) {
            const uint32_t dst_a = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 3);
            if (dst_a == 0) {
                continue;  // idle core: owns no slice, receives nothing
            }
            const uint32_t vx = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 0);
            const uint32_t vy = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 1);
            const uint32_t dst_off = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 2);
            const uint32_t slot_bytes = my_slot * dst_a * PB;
            const uint32_t nbytes = dst_a * PB;
            noc_async_write(gsrc + dst_off * PB, get_noc_addr(vx, vy, gg_base + slot_bytes), nbytes);
            noc_async_write(usrc + dst_off * PB, get_noc_addr(vx, vy, gu_base + slot_bytes), nbytes);
        }
        noc_async_write_barrier();
        const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(sem_data_id));
        for (uint32_t d = 0; d < K; ++d) {
            if (get_arg_val<uint32_t>(RT_DESTS + 4 * d + 3) == 0) {
                continue;
            }
            const uint32_t vx = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 0);
            const uint32_t vy = get_arg_val<uint32_t>(RT_DESTS + 4 * d + 1);
            noc_semaphore_inc(get_noc_addr(vx, vy, sem_data), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(cb_gate_acc, T);
        cb_pop_front(cb_up_acc, T);
    }

    if (assigned == 0) {
        return;
    }

    // ---- scatter my finished slice into the root ----
    const uint32_t sem_h = static_cast<uint32_t>(get_semaphore(sem_h_id));
    const uint32_t nbytes = assigned * PB;
    if constexpr (DIST_EPI) {
        constexpr uint32_t cb_h_dst = FUSE_H ? cb_h_local : cb_h_land;
        cb_wait_front(cb_hslice, assigned);
        noc_async_write(
            get_read_ptr(cb_hslice),
            get_noc_addr(root_x, root_y, get_write_ptr(cb_h_dst) + offset * PB),
            nbytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_h), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_hslice, assigned);
    } else {
        cb_wait_front(cb_send_g, assigned);
        cb_wait_front(cb_send_u, assigned);
        noc_async_write(
            get_read_ptr(cb_send_g),
            get_noc_addr(root_x, root_y, get_write_ptr(cb_rg_full) + offset * PB),
            nbytes);
        noc_async_write(
            get_read_ptr(cb_send_u),
            get_noc_addr(root_x, root_y, get_write_ptr(cb_ru_full) + offset * PB),
            nbytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_h), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_send_g, assigned);
        cb_pop_front(cb_send_u, assigned);
    }
}
"""

_RS_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(0);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(1);
constexpr uint32_t cb_gg = get_compile_time_arg_val(2);
constexpr uint32_t cb_gu = get_compile_time_arg_val(3);
constexpr uint32_t cb_sacc_g = get_compile_time_arg_val(4);
constexpr uint32_t cb_sacc_u = get_compile_time_arg_val(5);
constexpr uint32_t cb_silu = get_compile_time_arg_val(6);
constexpr uint32_t cb_hslice = get_compile_time_arg_val(7);
constexpr uint32_t cb_send_g = get_compile_time_arg_val(8);
constexpr uint32_t cb_send_u = get_compile_time_arg_val(9);
constexpr uint32_t cb_rg_full = get_compile_time_arg_val(10);
constexpr uint32_t cb_ru_full = get_compile_time_arg_val(11);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(12);
constexpr uint32_t T = get_compile_time_arg_val(13);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(14);
constexpr uint32_t M_EFF = get_compile_time_arg_val(15);
constexpr uint32_t BLK = get_compile_time_arg_val(16);
constexpr uint32_t DEST_LIMIT = get_compile_time_arg_val(17);
constexpr uint32_t NC = get_compile_time_arg_val(18);           // contributors per worker (>= 2)
constexpr uint32_t GATHER_CAP = get_compile_time_arg_val(19);   // whole landing CB, in tiles
constexpr uint32_t DIST_EPI = get_compile_time_arg_val(20);
constexpr uint32_t cb_usum = get_compile_time_arg_val(21);       // `noepi` root: sum(up)
constexpr uint32_t FUSE_H = get_compile_time_arg_val(22);
constexpr uint32_t cb_h_land = get_compile_time_arg_val(23);     // `unfused`: separate h landing

// SAME blocked-eltwise spelling as the baseline (and as the op). Not optional: `input(cb)` /
// `output(cb)` default to per-TILE lifecycles, which makes eltwise_chain silently clamp block_size
// to 1 (eltwise_chain.inl:3054).
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
ALWI auto blk_shape(uint32_t n) { return EltwiseShape::tiles(n, BLK); }

void kernel_main() {
    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t assigned = get_arg_val<uint32_t>(1);

    const bool root_epilogue = (is_root != 0) && (DIST_EPI == 0);
    const bool root_h_copy = (is_root != 0) && (DIST_EPI != 0) && (FUSE_H == 0);
    if (assigned == 0 && !root_epilogue && !root_h_copy) {
        return;  // idle core (fewer slices than cores): no slice work, no epilogue
    }

    compute_kernel_hw_startup(cb_gg, cb_gg, cb_sacc_g);
    ActivationInitHelper<KernelActivation::SILU>::init();

    CircularBuffer gg_buf(cb_gg), gu_buf(cb_gu), sacc_g_buf(cb_sacc_g), silu_buf(cb_silu);
    CircularBuffer rgf_buf(cb_rg_full), gate_buf(cb_gate_acc);

    if (assigned) {
        // ---- gate slice. Contributor 0 SEEDS the accumulator; 1..NC-2 accumulate IN PLACE (the
        // op's own add<blk_in(acc), blk_in(in), blk_out(acc)> on a 1-deep accumulator CB); the LAST
        // contributor folds in with SiLU riding the PACKER thread. A slice of `assigned` tiles is at
        // most ONE DEST window, so the root's M_EFF-call bias walk collapses to
        // ceil(assigned / DEST_LIMIT) calls for free. ----
        copy<blk_in(cb_gg), blk_out(cb_sacc_g)>(blk_shape(assigned));
        for (uint32_t i = 0; i + 2 < NC; ++i) {
            add<blk_in(cb_sacc_g), blk_in(cb_gg), blk_out(cb_sacc_g)>(blk_shape(assigned));
        }
        if constexpr (DIST_EPI) {
            gg_buf.wait_front(assigned);
            for (uint32_t t0 = 0; t0 < assigned; t0 += DEST_LIMIT) {
                uint32_t w = assigned - t0;
                if (w > DEST_LIMIT) {
                    w = DEST_LIMIT;
                }
                add_bias_bcast_rows<
                    BiasBroadcast::Elementwise,
                    OutputCBLayout::SubblockMajor,
                    bias_add_config::NoPostBias,
                    SiluActivation>(sacc_g_buf, gg_buf, silu_buf, BiasAddShape::of(1, 1, 1, w), {}, t0);
            }
            gg_buf.pop_front(assigned);
        } else {
            // The last add lands in a FRESH CB so the writer can never observe a mid-chain state of
            // the in-place accumulator.
            add<blk_in(cb_sacc_g), blk_in(cb_gg), blk_out(cb_send_g)>(blk_shape(assigned));
        }

        // ---- up slice ----
        copy<blk_in(cb_gu), blk_out(cb_sacc_u)>(blk_shape(assigned));
        for (uint32_t i = 0; i + 2 < NC; ++i) {
            add<blk_in(cb_sacc_u), blk_in(cb_gu), blk_out(cb_sacc_u)>(blk_shape(assigned));
        }
        if constexpr (DIST_EPI) {
            add<blk_in(cb_sacc_u), blk_in(cb_gu), blk_out(cb_sacc_u)>(blk_shape(assigned));
        } else {
            add<blk_in(cb_sacc_u), blk_in(cb_gu), blk_out(cb_send_u)>(blk_shape(assigned));
        }

        // Drain the padding tail of the landing CBs so the pop total equals the reader's WHOLE-CB
        // push and the write pointer stays block-aligned on every core (the landing-proxy contract).
        const uint32_t live = NC * assigned;
        if (GATHER_CAP > live) {
            gg_buf.pop_front(GATHER_CAP - live);
            gu_buf.pop_front(GATHER_CAP - live);
        }

        if constexpr (DIST_EPI) {
            mul<blk_in(cb_silu), blk_in(cb_sacc_u), blk_out(cb_hslice)>(blk_shape(assigned));
        }
    }

    if (root_h_copy) {
        // `unfused` arm: the workers' h slices landed in a SEPARATE CB, so the root pays one extra
        // full-block copy to assemble cb_h_local. This is exactly the pass the `epi` arm's fused
        // gather removes.
        copy<blk_in(cb_h_land), blk_out(cb_h_local)>(blk_shape(T));
    }

    if (root_epilogue) {
        // The op's ORIGINAL root epilogue, on the scatter-assembled full block: fold in this core's
        // OWN partial with SiLU on the packer (M_EFF calls of 1 x HN_PAD), then SwiGLU.
        gate_buf.wait_front(T);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            add_bias_bcast_rows<
                BiasBroadcast::Elementwise,
                OutputCBLayout::SubblockMajor,
                bias_add_config::NoPostBias,
                SiluActivation>(rgf_buf, gate_buf, silu_buf, BiasAddShape::of(1, 1, 1, HN_PAD), {}, m * HN_PAD);
        }
        gate_buf.pop_front(T);
        // NOT in place on cb_ru_full: its pusher is the READER (the scatter landed there), and a CB
        // may only ever be pushed by one RISC-V (header note).
        add<blk_in(cb_ru_full), blk_in(cb_up_acc), blk_out(cb_usum)>(blk_shape(T));
        mul<blk_in(cb_silu), blk_in(cb_usum), blk_out(cb_h_local)>(blk_shape(T));
    }
}
"""


def rs_plan(layout, slice_kind="flat", dist_epi=True, root=0):
    k = layout.k
    assigned, offsets, slice_pages = slice_plan(slice_kind, k, layout.m_eff, layout.hn_pad)
    a_max = max(assigned)
    workers = [r for r in range(k) if assigned[r] > 0]
    # DIST_EPI: every core contributes its partial to the scatter. NOEPI: the root holds its own
    # partial back and folds it in during its epilogue, so it is not a contributor.
    contributors = list(range(k)) if dist_epi else [r for r in range(k) if r != root]
    return assigned, offsets, a_max, workers, contributors, len(contributors) * a_max, slice_pages


def create_rs_descriptor(device, gate_t, up_t, h_t, layout, *, slice_kind="flat", dist_epi=True, fuse_h=True, root=0):
    """`slice_kind` in {"flat", "m", "ragged"}; `dist_epi=False` keeps the epilogue at the root;
    `fuse_h=False` lands the finished slices in a separate CB the root then COPIES into cb_h_local."""
    k, t = layout.k, layout.t_tiles
    assigned, offsets, a_max, workers, contributors, gather_cap, slice_pages = rs_plan(
        layout, slice_kind, dist_epi, root
    )
    nc = len(contributors)
    if nc < 2:
        raise ValueError(f"reduce-scatter needs >= 2 contributors per worker; got {nc} (k={k}, dist_epi={dist_epi})")
    # The CB page-count rule (see slice_plan): every eltwise pass on a slice CB pushes `assigned`
    # pages, so the page count must be a multiple of EVERY worker's `assigned`, and each individual
    # DEST window must not straddle the CB end either — which is only guaranteed when a pass is a
    # single window (assigned <= ELTWISE_BLK) or all workers agree.
    if a_max > ELTWISE_BLK and len(set(a for a in assigned if a)) > 1:
        raise ValueError(f"slice sizes {sorted(set(assigned))} with a_max {a_max} > ELTWISE_BLK are not expressible")

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for col in range(layout.ncols):
        all_virtual = [_virtual(device, *layout.col_core(col, r)) for r in range(k)]
        root_vx, root_vy = all_virtual[root]
        peer_args = []
        for vx, vy in all_virtual:
            peer_args += [vx, vy]
        dest_args = []
        for r in range(k):
            dest_args += [all_virtual[r][0], all_virtual[r][1], offsets[r], assigned[r]]

        for r in range(k):
            x, y = layout.col_core(col, r)
            is_root = 1 if r == root else 0
            reader_rt[x][y] = [
                gate_t.buffer_address(),
                up_t.buffer_address(),
                h_t.buffer_address(),
                is_root,
                assigned[r],
                len(workers),
            ] + peer_args
            is_contrib = 1 if r in contributors else 0
            my_slot = contributors.index(r) if is_contrib else 0
            writer_rt[x][y] = [is_contrib, my_slot, assigned[r], offsets[r], root_vx, root_vy] + dest_args
            compute_rt[x][y] = [is_root, assigned[r]]

    cr = layout.core_range
    cbs = [
        _cb(CB_GATE_ACC, cr, t),
        _cb(CB_UP_ACC, cr, t),
        _cb(CB_GG, cr, gather_cap),
        _cb(CB_GU, cr, gather_cap),
        _cb(CB_SACC_G, cr, slice_pages),
        _cb(CB_SACC_U, cr, slice_pages),
        _cb(CB_GATE_SILU, cr, slice_pages if dist_epi else t),
        _cb(CB_H_LOCAL, cr, t),
    ]
    if dist_epi:
        cbs.append(_cb(CB_HSLICE, cr, slice_pages))
        if not fuse_h:
            cbs.append(_cb(CB_H_LAND, cr, t))
    else:
        cbs += [
            _cb(CB_SEND_G, cr, slice_pages),
            _cb(CB_SEND_U, cr, slice_pages),
            _cb(CB_RG_FULL, cr, t),
            _cb(CB_RU_FULL, cr, t),
            _cb(CB_USUM, cr, t),
        ]
    sems = [
        ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=cr, initial_value=0),  # the peer invite
        ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=cr, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_H, core_ranges=cr, initial_value=0),
    ]
    de = 1 if dist_epi else 0
    fh = 1 if fuse_h else 0
    reader = _kernel(
        _RS_READER,
        cr,
        [
            CB_GATE_ACC,
            CB_UP_ACC,
            CB_GG,
            CB_GU,
            CB_H_LOCAL,
            CB_RG_FULL,
            CB_RU_FULL,
            t,
            BFP8_TILE_BYTES,
            gather_cap,
            nc,
            k,
            de,
            SEM_GO,
            SEM_DATA,
            SEM_H,
            fh,
            CB_H_LAND,
        ],
        reader_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    writer = _kernel(
        _RS_WRITER,
        cr,
        [
            CB_GATE_ACC,
            CB_UP_ACC,
            CB_GG,
            CB_GU,
            CB_HSLICE,
            CB_SEND_G,
            CB_SEND_U,
            CB_H_LOCAL,
            CB_RG_FULL,
            CB_RU_FULL,
            t,
            BFP8_TILE_BYTES,
            k,
            de,
            SEM_GO,
            SEM_DATA,
            SEM_H,
            fh,
            CB_H_LAND,
        ],
        writer_rt,
        ttnn.WriterConfigDescriptor(),
    )
    compute = _kernel(
        _RS_COMPUTE,
        cr,
        [
            CB_GATE_ACC,
            CB_UP_ACC,
            CB_GG,
            CB_GU,
            CB_SACC_G,
            CB_SACC_U,
            CB_GATE_SILU,
            CB_HSLICE,
            CB_SEND_G,
            CB_SEND_U,
            CB_RG_FULL,
            CB_RU_FULL,
            CB_H_LOCAL,
            t,
            layout.hn_pad,
            layout.m_eff,
            ELTWISE_BLK,
            DEST_LIMIT_TILES,
            nc,
            gather_cap,
            de,
            CB_USUM,
            fh,
            CB_H_LAND,
        ],
        compute_rt,
        _compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)


def cb_pages_rs(layout, slice_kind="flat", dist_epi=True, fuse_h=True, root=0):
    t = layout.t_tiles
    _, _, _, _, _, gather_cap, sp = rs_plan(layout, slice_kind, dist_epi, root)
    pages = 2 * t + 2 * gather_cap + 2 * sp + t  # accs + landing + slice accs + h_local
    pages += sp if dist_epi else t  # cb_gate_silu
    pages += sp if dist_epi else (2 * sp + 3 * t)  # hslice | send pair + landing pair + usum
    if dist_epi and not fuse_h:
        pages += t  # cb_h_land
    return pages


# ---------------------------------------------------------------------------
# One entry point per variant
# ---------------------------------------------------------------------------

#: variant -> (slice_kind, dist_epi, fuse_h). `epi` = the epilogue is distributed with the scatter and
#: the finished slices are gathered STRAIGHT into cb_h_local; `unfused` = same but through a separate
#: landing CB the root then copies; `noepi` = the epilogue stays whole at the root.
RS_VARIANTS = {
    "rs_flat_epi": ("flat", True, True),
    "rs_m_epi": ("m", True, True),
    "rs_ragged_epi": ("ragged", True, True),
    "rs_flat_unfused": ("flat", True, False),
    "rs_ragged_unfused": ("ragged", True, False),
    "rs_flat_noepi": ("flat", False, True),
    "rs_m_noepi": ("m", False, True),
    "rs_ragged_noepi": ("ragged", False, True),
}
VARIANTS = ("seed_only", "tree") + tuple(RS_VARIANTS)


def run_variant(device, gate_t, up_t, h_t, variant, layout):
    if variant == "seed_only":
        desc = create_tree_descriptor(device, gate_t, up_t, h_t, layout, seed_only=True)
    elif variant == "tree":
        desc = create_tree_descriptor(device, gate_t, up_t, h_t, layout)
    elif variant in RS_VARIANTS:
        kind, de, fh = RS_VARIANTS[variant]
        desc = create_rs_descriptor(device, gate_t, up_t, h_t, layout, slice_kind=kind, dist_epi=de, fuse_h=fh)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return ttnn.generic_op([gate_t, up_t, h_t], desc)


def cb_pages(variant, layout, *, op_equivalent=True):
    if variant in ("seed_only", "tree"):
        return cb_pages_tree(layout, op_equivalent=op_equivalent)
    kind, de, fh = RS_VARIANTS[variant]
    return cb_pages_rs(layout, slice_kind=kind, dist_epi=de, fuse_h=fh)


def cb_bytes(variant, layout, *, op_equivalent=True):
    """Per-core CB footprint in bytes. `op_equivalent` drops the baseline's bench-only seed staging,
    so the number is directly comparable to the op's shipped 8-CB reduce/epilogue set."""
    return cb_pages(variant, layout, op_equivalent=op_equivalent) * BFP8_TILE_BYTES
