# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the shape of the KGROUPS-deep gate/up cross-column reduce tree.

This is a MICRO-BENCHMARK of ONE part of `moe_fused_swiglu` — the column reduce that combines
KGROUPS cores' [m_eff, HN_PAD] bfp8 partials (both gate and up) into one sum. It does NOT touch the
real op. It reconstructs, in isolation:

  - the honest baseline: the op's shipped Hillis-Steele-style doubling tree (`_reduce_tree` in
    `moe_fused_swiglu_program_descriptor.py`), root fan-in = ceil(log2(KGROUPS)) (4 at KGROUPS=10),
    transport = raw unicast + 2 counting semaphores (SEM_GO parent-invite / SEM_DATA child-signal),
    REDUCE_SLOTS=1 (one child landing at a time — the shipped, non-regressed configuration; R2's
    lever 1 measured REDUCE_SLOTS=2 as a +2.0% regression, so it is not re-floated here).
  - candidate `fanin2`: a genuinely bounded max-fan-in-2 binary MERGE tree (every node, not just the
    root, does <= 2 sequential adds) landing at the SAME designated root, over the SAME transport.
  - candidate `twophase`: a tile-index reduce-SCATTER (the `two_phase_reduce_mcast` shape from
    `examples/tensix_all_reduce`) — each of the KGROUPS-1 non-root cores becomes a "worker" owning a
    disjoint, CONTIGUOUS tile-slice, pulls that slice from all KGROUPS contributors (itself
    included) via `noc_async_read`, reduces it locally (worker fan-in = KGROUPS-1, but over
    N/(KGROUPS-1) tiles instead of N), then unicasts its finished slice directly into the ROOT's
    result shard (a pure concatenation — no add at the root at all).

All three variants share: the SAME per-core local partial (a resident, height-sharded bfloat8_b
tensor — the payload each core is assumed to already hold, since PRODUCING it is the matmul stage,
out of scope for this idea), the SAME transport primitives (raw unicast + counting semaphores, no
mcast_pipe — a tree edge is point-to-point, matching the real op's own documented reason for not
using mcast_pipe here), and the SAME precision contract (LoFi / approx / no fp32 DEST — set by the
caller, never touched here).
"""

from dataclasses import dataclass

import ttnn

TILE = 32
BFP8_TILE_BYTES = 1088  # bfloat8_b tile size (32x32), matches op_design.md's CB table.

CB_LOCAL = 2  # reduce accumulator, STEP 0 (self-seeded from the resident local-partial tensor)
CB_INCOMING = 3  # landing buffer for a child's unicast (REDUCE_SLOTS=1: one slot, reused serially)
CB_FINAL = 4  # compute's commit buffer: root's reader drains it to `result`; non-root's writer ships it to its parent
CB_GATHER_SLICE = 5  # two-phase only: a worker's per-contributor gather staging buffer
CB_WORKER_OUT = 6  # two-phase only: unused placeholder kept for index stability (see CB_WORKER_STEP_BASE)
# CBs have a ONE-PRODUCER, ONE-CONSUMER contract (documented throughout this codebase). Both
# "true in-place" (`add<input(cb),input(cb2),output(cb)>`, output CB == an input CB in the SAME
# call) and "ping-pong" (one CB flip-flopping between an add's output at one step and an input at
# ANOTHER, separate eltwise_chain call later) VIOLATE it — both measured a genuine device HANG in
# this environment (isolated single-core repro, `probes/probe_030.py`..`035.py`), even though the
# real op's own compute.cpp comments document the in-place form as safe (root cause not isolated
# further — out of this bake-off's scope, it is not the idea being measured). FIX: a linear CHAIN
# of single-use CBs, one per reduce step — each CB is written by exactly one producer (one add's
# output, or the reader's prologue self-read for step 0) and read by exactly one consumer (the
# next add's input, or the final copy) for the entire kernel lifetime. Tree variants: fixed 4 steps
# (max fan-in over this bake-off's k in {4,8,10} is 4, hillis_steele's root at k=10).
CB_STEP1 = 7
CB_STEP2 = 8
CB_STEP3 = 9
CB_STEP4 = 10
# Two-phase: up to k-1 steps (k <= 10 in this bake-off), generated per-k at descriptor-build time.
CB_WORKER_STEP_BASE = 11

SEM_GO = 0  # parent -> child: "I have a landing slot free, ship now"
SEM_DATA = 1  # child -> parent: "my data has landed"
SEM_PROGRESS = 2  # two-phase only: worker -> root: "my slice has landed in your result shard"


# ---------------------------------------------------------------------------
# Tree construction (host-only, no device calls). Both return
# {row: {"parent": Optional[row], "children": [row, ...]}} for rows 0..k-1, root at `root`.
# ---------------------------------------------------------------------------


def hillis_steele_tree(k, root=0):
    """The op's SHIPPED tree (`_reduce_tree` in moe_fused_swiglu_program_descriptor.py), for one
    column of `k` rows. Root fan-in = ceil(log2(k)) because the accumulator (relative index r=0)
    is the SAME physical node at every doubling level."""
    info = {}
    for y in range(k):
        r = (y - root) % k
        children = []
        s = 1
        while s < k:
            if r % (2 * s) == 0 and r + s < k:
                children.append((root + r + s) % k)
            s *= 2
        parent = None
        if r != 0:
            low = r & (-r)
            parent = (root + r - low) % k
        info[y] = {"parent": parent, "children": children}
    return info


def fanin2_tree(k, root=0):
    """A genuine max-fan-in-2 binary MERGE tree over `k` rows, final sum lands at `root`.

    Unlike Hillis-Steele, the "accumulator" role MOVES to a fresh node at each merge instead of
    staying fixed at the root, so no node (root included) ever does more than 2 sequential adds,
    regardless of k. Built recursively: `build(subset, sub_root)` reduces `subset` into `sub_root`
    by splitting the OTHER members into two halves, recursing on each half (giving each half its
    OWN sub-root, capped at 2 children by the same rule), then merging the two half sub-roots into
    `sub_root` with exactly 2 adds (1 if a half is empty).
    """
    nodes = list(range(k))
    info = {n: {"parent": None, "children": []} for n in nodes}

    def build(subset, sub_root):
        others = [n for n in subset if n != sub_root]
        if not others:
            return
        if len(others) <= 2:
            for c in others:
                info[c]["parent"] = sub_root
                info[sub_root]["children"].append(c)
            return
        mid = len(others) // 2
        half_a, half_b = others[:mid], others[mid:]
        sub_a, sub_b = half_a[0], half_b[0]
        build(half_a, sub_a)
        build(half_b, sub_b)
        info[sub_a]["parent"] = sub_root
        info[sub_root]["children"].append(sub_a)
        info[sub_b]["parent"] = sub_root
        info[sub_root]["children"].append(sub_b)

    build(nodes, root)
    return info


def tree_depth(info, root=0):
    def depth(node):
        children = info[node]["children"]
        if not children:
            return 0
        return 1 + max(depth(c) for c in children)

    return depth(root)


def tree_root_adds(info, root=0):
    return len(info[root]["children"])


def tree_max_fanin(info):
    return max(len(n["children"]) for n in info.values())


# ---------------------------------------------------------------------------
# Host plumbing shared by every variant.
# ---------------------------------------------------------------------------


def _column_cores(k, x=0, y0=0):
    return [(x, y0 + row) for row in range(k)]


def _virtual(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _core_range(cores):
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))])


def _normal_cb(cb_index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_bytes)],
    )


def _inline_kernel(source, core_ranges, compile_time_args, runtime_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=config,
    )


def make_sharded_config(device, k, n_tiles, x=0, y0=0):
    core_range = _core_range(_column_cores(k, x, y0))
    return ttnn.create_sharded_memory_config(
        shape=(TILE, n_tiles * TILE),
        core_grid=core_range,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _compute_config():
    # PRECISION CONTRACT — identical to moe_fused_swiglu.default_compute_kernel_config(). Fixed
    # input to every variant; never a lever here.
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        bfp8_pack_precise=True,
    )


# ---------------------------------------------------------------------------
# Kernels shared by `hillis_steele` and `fanin2` — ONLY the host-side tree-construction function
# differs between those two variants; the transport + add mechanism below is IDENTICAL, which is
# what isolates "tree shape" as the one measured variable.
# ---------------------------------------------------------------------------

_TREE_READER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// PARENT side of the reduce tree (mirrors moe_fused_swiglu_reader.cpp's `reader_reduce` zone):
// self-seed cb_local from the resident local-partial tensor, then invite each child in turn
// (REDUCE_SLOTS=1 — one landing slot, reused serially) and wait for its data.
void kernel_main() {
    constexpr uint32_t cb_local = get_compile_time_arg_val(0);
    constexpr uint32_t cb_incoming = get_compile_time_arg_val(1);
    constexpr uint32_t cb_final = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(5);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(6);

    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t result_addr = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t num_children = get_arg_val<uint32_t>(3);
    constexpr uint32_t children_base = 4;

    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    // ---- prologue: self-seed cb_local from this core's own resident shard (NOT part of the idea
    // being measured — it stands in for "the matmul already produced my local partial", held fixed
    // and identical across every variant) ----
    {
        cb_reserve_back(cb_local, num_tiles);
        const uint32_t wp = get_write_ptr(cb_local);
        noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], local_addr), wp, payload_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_local, num_tiles);
    }

    // ---- parent side: invite each child, wait for its data (THE MEASURED PART) ----
    const uint32_t sem_data_addr = static_cast<uint32_t>(get_semaphore(sem_data_id));
    volatile tt_l1_ptr uint32_t* sem_data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data_addr);
    uint32_t data_arrivals = 0;
    for (uint32_t c = 0; c < num_children; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(children_base + 2 * c + 0);
        const uint32_t cy = get_arg_val<uint32_t>(children_base + 2 * c + 1);
        cb_reserve_back(cb_incoming, num_tiles);
        noc_semaphore_inc(get_noc_addr(cx, cy, static_cast<uint32_t>(get_semaphore(sem_go_id))), 1);
        data_arrivals += 1;
        noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
        cb_push_back(cb_incoming, num_tiles);
    }

    // ---- root only: commit the final sum into the resident result tensor ----
    if (is_root) {
        cb_wait_front(cb_final, num_tiles);
        const uint32_t rp = get_read_ptr(cb_final);
        noc_async_write(rp, get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr), payload_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_final, num_tiles);
    }
}
"""

_TREE_WRITER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// CHILD side of the reduce tree (mirrors moe_fused_swiglu_writer.cpp's `writer_reduce_child`
// zone): wait for the parent's invite, unicast cb_final into the parent's cb_incoming landing
// slot (the child's OWN get_write_ptr(cb_incoming) is a proxy for the parent's, since every core
// has an identical CB layout — the real op's exact trick), signal, done. Root has no parent: no-op.
void kernel_main() {
    constexpr uint32_t cb_final = get_compile_time_arg_val(0);
    constexpr uint32_t cb_incoming = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t sem_go_id = get_compile_time_arg_val(4);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(5);

    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t parent_x = get_arg_val<uint32_t>(1);
    const uint32_t parent_y = get_arg_val<uint32_t>(2);

    if (is_root) {
        return;
    }

    constexpr uint32_t payload_bytes = num_tiles * page_bytes;
    volatile tt_l1_ptr uint32_t* sem_go_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(sem_go_id)));

    cb_wait_front(cb_final, num_tiles);
    noc_semaphore_wait_min(sem_go_ptr, 1);
    const uint32_t dst_addr = get_write_ptr(cb_incoming);  // same relative offset on every core
    noc_async_write(get_read_ptr(cb_final), get_noc_addr(parent_x, parent_y, dst_addr), payload_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(parent_x, parent_y, static_cast<uint32_t>(get_semaphore(sem_data_id))), 1);
    // The remote atomic increment above is this kernel's LAST NoC op — with nothing after it to
    // naturally drain it (unlike the real op's writer, which always has an output write-back
    // after its own reduce-tree signal), the firmware's exit-time
    // ASSERT(ncrisc_noc_nonposted_atomics_flushed) trips without an explicit flush here
    // (found via dump_lightweight_asserts.py on a genuine device run — every non-root writer
    // asserted at brisc.cc:517 until this barrier was added).
    noc_async_atomic_barrier();
    cb_pop_front(cb_final, num_tiles);
}
"""

_TREE_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

using namespace compute_kernel_lib;

// The ADD (mirrors moe_fused_swiglu_compute.cpp's `compute_reduce` zone, minus the packer-thread
// SiLU fuse on the root's last add — SiLU is documented as free/overlapped there, so dropping it
// does not change which mechanism (transport vs. add) dominates). For each child: `add<>`'s own
// default per-tile WaitPolicy/PopPolicy on `input(cb_incoming)` waits for its unicast to land and
// pops it — NO manual cb_wait_front/cb_pop_front around the call (that double-manages the SAME CB
// against add<>'s own internal DataflowBuffer bookkeeping). LINEAR CHAIN of single-use CBs, not
// in-place/ping-pong reuse (see the CB_STEP* comment at the top of this file for why): step 0 is
// cb_local (seeded once by the reader's prologue); each add writes a BRAND NEW cb_stepN never
// touched again, so every CB has exactly one producer and one consumer for the kernel's lifetime.
// Fixed at 4 steps (this bake-off's max fan-in, hillis_steele's root at k=10). A leaf (0 children)
// takes none of the ifs below; its result is already step 0 (cb_local). Finally commit whichever
// step is live into cb_final for the reader (root) / writer (non-root) to drain.
void kernel_main() {
    constexpr uint32_t cb_local = get_compile_time_arg_val(0);   // step 0
    constexpr uint32_t cb_incoming = get_compile_time_arg_val(1);
    constexpr uint32_t cb_final = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_step1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_step2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_step3 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_step4 = get_compile_time_arg_val(7);

    const uint32_t num_children = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_local, cb_incoming, cb_local);

    if (num_children >= 1) {
        add<input(cb_local), input(cb_incoming), output(cb_step1)>(EltwiseShape::tiles(num_tiles));
    }
    if (num_children >= 2) {
        add<input(cb_step1), input(cb_incoming), output(cb_step2)>(EltwiseShape::tiles(num_tiles));
    }
    if (num_children >= 3) {
        add<input(cb_step2), input(cb_incoming), output(cb_step3)>(EltwiseShape::tiles(num_tiles));
    }
    if (num_children >= 4) {
        add<input(cb_step3), input(cb_incoming), output(cb_step4)>(EltwiseShape::tiles(num_tiles));
    }

    if (num_children == 0) {
        copy<input(cb_local), output(cb_final)>(EltwiseShape::tiles(num_tiles));
    } else if (num_children == 1) {
        copy<input(cb_step1), output(cb_final)>(EltwiseShape::tiles(num_tiles));
    } else if (num_children == 2) {
        copy<input(cb_step2), output(cb_final)>(EltwiseShape::tiles(num_tiles));
    } else if (num_children == 3) {
        copy<input(cb_step3), output(cb_final)>(EltwiseShape::tiles(num_tiles));
    } else {
        copy<input(cb_step4), output(cb_final)>(EltwiseShape::tiles(num_tiles));
    }
}
"""


@dataclass(frozen=True)
class TreeLayout:
    k: int
    x: int
    y0: int
    cores: tuple

    @property
    def core_range(self):
        return _core_range(self.cores)


def build_tree_layout(k, x=0, y0=0):
    return TreeLayout(k, x, y0, tuple(_column_cores(k, x, y0)))


def create_tree_descriptor(device, local_tensor, result_tensor, tree, layout, n_tiles, *, writer_noc=None):
    """`tree` variant-agnostic: pass `hillis_steele_tree(k)` or `fanin2_tree(k)`."""
    max_fanin = tree_max_fanin(tree)
    if max_fanin > 4:
        raise ValueError(f"reduce_tree_shape's compute kernel is fixed at a 4-step chain; got max fan-in {max_fanin}")
    core_range = layout.core_range
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    for row in range(layout.k):
        x, y = layout.cores[row]
        info = tree[row]
        is_root = 1 if info["parent"] is None else 0
        children = info["children"]
        coord_args = []
        for c in children:
            cx, cy = layout.cores[c]
            vx, vy = _virtual(device, cx, cy)
            coord_args += [vx, vy]
        reader_rt[x][y] = [
            local_tensor.buffer_address(),
            result_tensor.buffer_address(),
            is_root,
            len(children),
        ] + coord_args

        if info["parent"] is not None:
            px, py = layout.cores[info["parent"]]
            pvx, pvy = _virtual(device, px, py)
        else:
            pvx, pvy = 0, 0
        writer_rt[x][y] = [is_root, pvx, pvy]
        compute_rt[x][y] = [is_root, len(children)]

    cbs = [
        _normal_cb(CB_LOCAL, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_INCOMING, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_FINAL, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_STEP1, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_STEP2, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_STEP3, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
        _normal_cb(CB_STEP4, core_range, n_tiles, BFP8_TILE_BYTES, ttnn.bfloat8_b),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=core_range, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=core_range, initial_value=0),
    ]
    reader = _inline_kernel(
        _TREE_READER_KERNEL,
        core_range,
        [CB_LOCAL, CB_INCOMING, CB_FINAL, n_tiles, BFP8_TILE_BYTES, SEM_GO, SEM_DATA],
        reader_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    if writer_noc is None:
        writer_config = ttnn.WriterConfigDescriptor()
    else:
        writer_config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=writer_noc)
    writer = _inline_kernel(
        _TREE_WRITER_KERNEL,
        core_range,
        [CB_FINAL, CB_INCOMING, n_tiles, BFP8_TILE_BYTES, SEM_GO, SEM_DATA],
        writer_rt,
        writer_config,
    )
    compute = _inline_kernel(
        _TREE_COMPUTE_KERNEL,
        core_range,
        [CB_LOCAL, CB_INCOMING, CB_FINAL, n_tiles, CB_STEP1, CB_STEP2, CB_STEP3, CB_STEP4],
        compute_rt,
        _compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)


def run_tree_variant(device, local_tensor, result_tensor, variant, k, n_tiles, *, root=0, writer_noc=None):
    if variant == "hillis_steele":
        tree = hillis_steele_tree(k, root=root)
    elif variant == "fanin2":
        tree = fanin2_tree(k, root=root)
    else:
        raise ValueError(f"unknown tree variant {variant!r}")
    layout = build_tree_layout(k)
    descriptor = create_tree_descriptor(
        device, local_tensor, result_tensor, tree, layout, n_tiles, writer_noc=writer_noc
    )
    return ttnn.generic_op([local_tensor, result_tensor], descriptor)


# ---------------------------------------------------------------------------
# `twophase` — tile-index reduce-scatter (the tensix_all_reduce `two_phase_reduce_mcast` shape,
# adapted: workers pull CONTIGUOUS slices instead of strided tile-index slices, so each of the k
# reads per worker is one coalesced transaction instead of `assigned` single-tile ones; and there
# is no broadcast-back leg, since this op's consumer only needs the assembled sum at `root`, not
# an all-reduce). Root does ZERO adds — every tile lands via a disjoint unicast write.
# ---------------------------------------------------------------------------


def _twophase_kernel_sources(k):
    """Generate the worker + compute kernel C++ source for a `k`-core column.

    Same one-producer/one-consumer-per-CB fix as the tree variants (see the CB_STEP* comment at
    the top of this file): a linear chain of `k` single-use CBs (`cb_step0`.."cb_step{k-1}`),
    literal CB indices baked directly into the generated source (Python already knows them at
    descriptor-build time, so no variadic compile-time-arg scheme is needed). `cb_step0` is seeded
    by `copy<input(cb_gather),...>` (contributor 0's slice); `cb_step{i}` for i>=1 accumulates
    contributor i via `add<input(cb_step{i-1}), input(cb_gather), output(cb_step{i})>` — each
    step's output CB is touched exactly once as a producer and exactly once as a consumer for the
    kernel's whole lifetime. The final result is always `cb_step{k-1}`.
    """
    step_cbs = [CB_WORKER_STEP_BASE + i for i in range(k)]
    final_cb = step_cbs[k - 1]
    steps_decl = "\n    ".join(f"constexpr uint32_t cb_step{i} = {cb};" for i, cb in enumerate(step_cbs))

    worker = f"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// Worker: pull my disjoint, contiguous tile-slice from all `group_size` contributors (self
// included) with ONE coalesced read per contributor (not per-tile), let compute reduce them (into
// the single-use CB chain, cb_step0..cb_step{k - 1}), then unicast the finished slice straight
// into the root's result shard and signal progress. Root: wait for all workers — NO reduce work
// and NO reads at all.
void kernel_main() {{
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t group_size = get_compile_time_arg_val(2);
    constexpr uint32_t sem_progress_id = get_compile_time_arg_val(3);
    constexpr uint32_t final_cb = {final_cb};

    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t result_addr = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t num_workers = get_arg_val<uint32_t>(3);
    const uint32_t assigned = get_arg_val<uint32_t>(4);   // this worker's tile-slice width (0 if root)
    const uint32_t offset = get_arg_val<uint32_t>(5);     // this worker's tile offset into [0, N)
    const uint32_t root_x = get_arg_val<uint32_t>(6);
    const uint32_t root_y = get_arg_val<uint32_t>(7);
    constexpr uint32_t contributors_base = 8;  // group_size (vx, vy) pairs follow

    if (is_root) {{
        volatile tt_l1_ptr uint32_t* progress_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            static_cast<uint32_t>(get_semaphore(sem_progress_id)));
        noc_semaphore_wait_min(progress_ptr, num_workers);
        return;
    }}

    const uint32_t slice_bytes = assigned * page_bytes;

    // ---- gather: one contiguous, coalesced read per contributor (not per-tile) ----
    cb_reserve_back(cb_gather, group_size * assigned);
    const uint32_t gather_addr = get_write_ptr(cb_gather);
    for (uint32_t contributor = 0; contributor < group_size; ++contributor) {{
        const uint32_t cx = get_arg_val<uint32_t>(contributors_base + 2 * contributor + 0);
        const uint32_t cy = get_arg_val<uint32_t>(contributors_base + 2 * contributor + 1);
        noc_async_read(
            get_noc_addr(cx, cy, local_addr + offset * page_bytes),
            gather_addr + contributor * slice_bytes,
            slice_bytes);
    }}
    noc_async_read_barrier();
    cb_push_back(cb_gather, group_size * assigned);

    // ---- reduce happens on compute (drains cb_gather through the single-use CB chain); scatter
    // the result to root ----
    cb_wait_front(final_cb, assigned);
    const uint32_t acc_ptr = get_read_ptr(final_cb);
    noc_async_write(acc_ptr, get_noc_addr(root_x, root_y, result_addr + offset * page_bytes), slice_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, static_cast<uint32_t>(get_semaphore(sem_progress_id))), 1);
    // Same fix as the tree writer's SEM_DATA increment (see its comment): this atomic is the last
    // NoC op before the kernel returns, so it needs an explicit flush or the firmware's exit-time
    // nonposted-atomics assert can trip.
    noc_async_atomic_barrier();
    cb_pop_front(final_cb, assigned);
}}
"""

    add_lines = [f"    copy<input(cb_gather), output(cb_step0)>(EltwiseShape::tiles(assigned));"]
    for i in range(1, k):
        add_lines.append(
            f"    add<input(cb_step{i - 1}), input(cb_gather), output(cb_step{i})>(EltwiseShape::tiles(assigned));"
        )
    body = "\n".join(add_lines)

    compute = f"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

using namespace compute_kernel_lib;

// Worker: cb_gather holds `group_size` back-to-back `assigned`-tile slices (one per contributor,
// FIFO order = contributor order). Contributor 0 SEEDS cb_step0 via `copy`; contributors 1..
// group_size-1 accumulate via `add` into a BRAND NEW cb_stepN each time — each call consumes
// exactly the NEXT `assigned` tiles waiting in cb_gather's FIFO (default per-tile wait/pop), so no
// tile-offset addressing is needed. Linear CHAIN of single-use CBs, not in-place/ping-pong reuse —
// see the CB_STEP* comment at the top of this file: any CB that flip-flops between being an add's
// output and later an input (in-place OR ping-pong) measured a genuine device hang here. Root: no
// reduce work.
void kernel_main() {{
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t group_size = get_compile_time_arg_val(1);
    {steps_decl}

    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t assigned = get_arg_val<uint32_t>(1);

    if (is_root) {{
        return;
    }}

    compute_kernel_hw_startup(cb_gather, cb_gather, cb_step0);

{body}
}}
"""
    return worker, compute, step_cbs


def create_twophase_descriptor(device, local_tensor, result_tensor, layout, n_tiles, root=0):
    k = layout.k
    workers = [row for row in range(k) if row != root]
    num_workers = len(workers)  # every non-root core is a worker whenever n_tiles >= k - 1 (true here)
    base = n_tiles // num_workers
    rem = n_tiles % num_workers
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    root_x, root_y = layout.cores[root]
    root_vx, root_vy = _virtual(device, root_x, root_y)
    all_virtual = [_virtual(device, cx, cy) for cx, cy in layout.cores]

    offset = 0
    for row in range(k):
        x, y = layout.cores[row]
        is_root = 1 if row == root else 0
        if is_root:
            dataflow_rt[x][y] = [
                local_tensor.buffer_address(),
                result_tensor.buffer_address(),
                is_root,
                num_workers,
                0,
                0,
                root_vx,
                root_vy,
            ]
            compute_rt[x][y] = [is_root, 0]
        else:
            widx = workers.index(row)
            assigned = base + (1 if widx < rem else 0)
            my_offset = offset
            offset += assigned
            coord_args = []
            for cx, cy in all_virtual:
                coord_args += [cx, cy]
            dataflow_rt[x][y] = [
                local_tensor.buffer_address(),
                result_tensor.buffer_address(),
                is_root,
                num_workers,
                assigned,
                my_offset,
                root_vx,
                root_vy,
            ] + coord_args
            compute_rt[x][y] = [is_root, assigned]

    max_assigned = base + (1 if rem else 0)
    core_range = layout.core_range
    worker_source, compute_source, step_cbs = _twophase_kernel_sources(k)
    cbs = [
        _normal_cb(CB_GATHER_SLICE, core_range, k * max_assigned, BFP8_TILE_BYTES, ttnn.bfloat8_b),
    ] + [_normal_cb(cb, core_range, max_assigned, BFP8_TILE_BYTES, ttnn.bfloat8_b) for cb in step_cbs]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=core_range, initial_value=0),
    ]
    dataflow = _inline_kernel(
        worker_source,
        core_range,
        [CB_GATHER_SLICE, BFP8_TILE_BYTES, k, SEM_PROGRESS],
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _inline_kernel(
        compute_source,
        core_range,
        [CB_GATHER_SLICE, k],
        compute_rt,
        _compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


def run_twophase(device, local_tensor, result_tensor, k, n_tiles, root=0):
    layout = build_tree_layout(k)
    descriptor = create_twophase_descriptor(device, local_tensor, result_tensor, layout, n_tiles, root=root)
    return ttnn.generic_op([local_tensor, result_tensor], descriptor)
