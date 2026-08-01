# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the SHAPE of the moe_fused_swiglu reduce-tree TRANSPORT.

MICRO-BENCHMARK of ONE part of `moe_fused_swiglu` — the child -> parent unicast edge of the
per-column gate/up reduce tree, plus its two-semaphore handshake. It does NOT touch the real op.
The tree TOPOLOGY is held fixed at the op's shipped one (`_reduce_tree`); the only thing that varies
is HOW the bytes cross the edge:

  * `n_ch` / `ch_tiles`      — TWO 52 224 B unicasts (gate + up, the shipped shape) vs ONE merged
                               104 448 B unicast (fewer, larger transactions).
  * `owner`                  — WHICH data-movement RISC-V issues the unicast, i.e. WHICH NoC.
                               The op's child ships on the writer = BRISC = **NOC_1**
                               (`preferred_noc_for_dram_write`, kernel_types.hpp:141); the reader =
                               NCRISC = **NOC_0**. `dual` splits gate onto NOC_0 and up onto NOC_1
                               so BOTH networks inject concurrently into the parent.
                               A single kernel may NOT be moved onto the other RISC-V's NoC:
                               `DM_DEDICATED_NOC` assumes one NoC per DM RISC-V, and putting both
                               DM kernels on the same NoC is a measured device HANG (recorded by the
                               sibling `reduce_tree_shape` bench). So "pick a NoC" here always means
                               "pick which of the two DM kernels issues the write".
  * `slots`                  — concurrent landing slots in the parent (`REDUCE_SLOTS`), 1 = the
                               shipped invite-one-child-at-a-time protocol.
  * `use_invite`             — whether the `SEM_GO` parent invite exists at all. Dropping it is only
                               race-free when every child owns its OWN slot (`slots >= fan_in`), so
                               the no-invite arm is measured ONLY in that configuration and ONLY as
                               a diagnostic of what the invite costs.
  * `orient`                 — hop DIRECTION. `op` is the shipped mapping
                               `y = (root_y + r) % KGROUPS` (every edge goes to a LOWER relative
                               index, i.e. decreasing y modulo KGROUPS = NOC_1's routing
                               direction); `mirror` is `y = (root_y - r) % KGROUPS`, which flips
                               every edge.

Everything else is held constant per /perf-lab's concept-isolation table: the same 11 x KGROUPS
grid (so the NoC contention of 11 concurrent column trees is present, exactly as in the op), the
same bfp8 payload, the same tree, the same add sequence, the same precision contract
(LoFi / approx / no fp32 DEST / bfp8_pack_precise — a FIXED input, never a lever).

Two measurement modes:
  * `add` mode  — the parent's landing CBs are consumed by real bfp8 `add<>` calls (the op's
                  `compute_reduce`, blocked with ELTWISE_BLK exactly as Perf 1 graduated). This is
                  the REALISTIC setting: it is where the transport can hide under the adds, so it is
                  the arm that answers "does the win survive?". Correctness = PCC of the root's sum.
  * `xfer` mode — no compute kernel at all; the child ships straight out of its resident L1 shard
                  and the parent only handshakes. Tree serialisation is preserved (a node cannot
                  ship until its own children have landed). This is the PURE transport number, i.e.
                  the one to compare against the /perf-ceiling-dm bound. Correctness = the last
                  child's bytes must be byte-exact at the parent's landing address.

MEASURED (blackhole_p150, 11 x 10 grid, CHIP_FREQ 1350 MHz, 104 448 B per tree edge, medians of 3,
run-to-run spread <= 2%). `add` mode is the one that decides:

  | arm                                     | xfer ns |  add ns | add vs base | + L1 B  |
  |-----------------------------------------|---------|---------|-------------|---------|
  | baseline (shipped)                      |   8 185 |  21 201 |        —    |       0 |
  | one_write (merge the 2 unicasts)        |   8 036 |  21 147 |    -0.20 %  |       0 |
  | dir_noc (shortest-path per-edge NoC)    |   7 984 |  21 170 |    -0.09 %  |       0 |
  | dir_noc + one_write                     |   7 999 |  21 099 |    -0.46 %  |       0 |
  | slots2, WHOLE-WAVE push (R2's lever 1)  |   7 697 |  23 750 |   +12.08 %  | 104 448 |
  | slots2, PER-SLOT push                   |   7 687 |  20 846 |    -1.62 %  | 104 448 |
  | dir_noc + slots2 + PER-SLOT push  <-- W |   6 449 |  19 642 |    -7.35 %  | 104 448 |
  | dir_noc + slots4 + PER-SLOT push        |   6 346 |  19 647 |    -7.28 %  | 313 344 |
  | send the SAME bytes on NOC_0 instead    |  15 764 |  28 878 |   +36.28 %  |       0 |
  | mirror (hop direction flipped)          |  15 715 |  28 899 |   +36.38 %  |       0 |
  | mirror + dir_noc (NoC re-matched)       |   8 046 |  21 202 |    +0.06 %  |       0 |
  | twosided placement (children straddle)  |  13 985 |  26 182 |   +23.56 %  |       0 |
  | dual-NoC split of ONE edge              |   9 499 |  21 835 |    +3.04 %  |       0 |

THREE MECHANISMS, all measured here:

 1. A single contention-free edge costs the SAME on either NoC (3 140 ns on NOC_1 vs 3 197 ns on
    NOC_0 for 104 448 B). The 1.92x above is therefore NOT link speed — it is TORUS-WRAPPED HOP
    COUNT: NOC_1 routes decreasing, NOC_0 increasing, and a mismatched edge takes the long way round
    the 12-row NoC torus. 99 concurrent long paths overlap into congestion. Total payload hops
    predicts the ranking: 273 (baseline) / 197 (dir_noc) / 907 (mirror) / 915 (send_noc0).
    => The op's shipped tree is ALREADY direction-matched. Mis-orienting it costs up to 1.92x.
 2. Concurrency only pays if the concurrent transfers DO NOT SHARE A NoC. `slots2` alone is worth
    -6 % of pure transport; `dir_noc` alone -2.5 %; together -21.4 %, because the parent's two
    in-flight children then land on DIFFERENT networks instead of queueing on NOC_1.
 3. Refinement 2's `REDUCE_SLOTS` 1->2 regression (+2.0 % in the op, reproduced here as +12.08 %)
    was NOT "concurrency does not help". It was the WHOLE-CB WAVE PUSH: a single shared arrival
    counter tells the parent HOW MANY children arrived but not WHICH, so it cannot publish a slot
    until the whole wave is in, which destroys the shipped protocol's transfer/add interleave. ONE
    COUNTER PER SLOT fixes exactly that and turns +12.08 % into -1.62 % on its own.

CEILING (/perf-ceiling-dm, calibrated on this box rather than read off noc_latencies.yaml — see
`test_transport_ceiling_calibration`): marginal L1->L1 unicast bandwidth ~80 B/ns = 59 B/cycle at
1350 MHz against Blackhole's 64 B/cycle NoC = 93 % of the wire; fixed cost ~580 ns/program. The root
takes fan_in x 104 448 = 417 792 B, so the single-NoC destination-port floor is 5 222 ns and the
both-NoCs floor is 2 611 ns. Baseline pure transport, with the bench's scaffolding (program fixed
cost + the root's echo write-out) subtracted, is ~6 300 ns = 1.21x the single-NoC floor; the winner
is ~4 620 ns, i.e. 0.88x that floor (it genuinely uses both networks) and 1.77x the two-NoC floor.
"""

from dataclasses import dataclass, field

import ttnn

TILE = 32
BFP8_TILE_BYTES = 1088  # bfloat8_b 32x32 tile, matches op_design.md's CB table

# CB indices. Channel `ch` owns LOCAL[ch], IN[ch] and STEP[ch][1..MAX_FANIN].
CB_LOCAL = (0, 1)
CB_IN = (2, 3)
CB_STEP = ((4, 5, 6, 7), (8, 9, 10, 11))
MAX_FANIN = 4  # ceil(log2(10)) — the shipped tree's root fan-in at KGROUPS = 10

SEM_GO = 0  # parent -> child: "your landing slot is free, ship now"  (MONOTONE, never reset)
# child -> parent: "my bytes landed", ONE COUNTER PER LANDING SLOT (IDs SEM_DATA_BASE .. +slots-1).
# A single shared counter only tells the parent HOW MANY children arrived, not WHICH — so with more
# than one slot in flight it cannot publish a slot until the whole wave is in. Per-slot counters let
# the parent publish slot s the moment slot s lands, which is what keeps the shipped protocol's
# transfer/add INTERLEAVE while still having several transfers in flight. All MONOTONE, never reset.
SEM_DATA_BASE = 1
SEM_TREE = 5  # xfer mode only: DM0 -> DM1 on the SAME core, "my children have landed"


# ---------------------------------------------------------------------------
# Tree (host-only). Identical rule to moe_fused_swiglu_program_descriptor.py::_reduce_tree,
# with the row mapping factored out so the hop DIRECTION can be flipped.
# ---------------------------------------------------------------------------
def _rel_children(r, kgroups):
    """The doubling tree's children of relative index `r` — TOPOLOGY, independent of placement."""
    out, s = [], 1
    while s < kgroups:
        if r % (2 * s) == 0 and r + s < kgroups:
            out.append(r + s)
        s *= 2
    return out


def _rel_parent(r):
    return None if r == 0 else r - (r & (-r))


def twosided_offsets(kgroups):
    """Signed row offsets for each relative index so a parent's children STRADDLE it.

    WHY THIS EXISTS (measured mechanism, see the bench README/report): NoC_1 routes in the
    DECREASING physical direction and NoC_0 in the increasing one, so for a given edge exactly ONE
    NoC has a short path and the other must wrap the whole torus — measured 1.93x on this very edge
    (`send_noc0` / `mirror` arms). The shipped mapping `y = (root_y + r) % KGROUPS` makes EVERY edge
    decreasing, so all of a parent's concurrent children contend on NoC_1 alone. Placing a parent's
    children on ALTERNATING sides lets two of them land CONCURRENTLY on DIFFERENT NoCs, each still
    on its own short path — the only way to get 2x destination bandwidth without changing the tree's
    depth, fan-in or edge count (splitting ONE edge across both NoCs cannot work: half the bytes
    would have to take the long way round).

    The tree TOPOLOGY over relative indices is untouched; only the r -> row assignment changes.
    Offsets are drawn from `range(-(k//2), k - k//2)`, which is exactly `k` values, distinct mod `k`,
    so `y = (root_y + offset) % k` is still a bijection onto the column's rows.
    """
    pos = {0: 0}
    up = [o for o in range(1, kgroups - kgroups // 2)]  # +1, +2, ... (nearest first)
    down = [-o for o in range(1, kgroups // 2 + 1)]  # -1, -2, ... (nearest first)
    queue = [0]
    while queue:
        r = queue.pop(0)
        for i, c in enumerate(_rel_children(r, kgroups)):
            pool_a, pool_b = (up, down) if i % 2 == 0 else (down, up)
            pool = pool_a if pool_a else pool_b
            pos[c] = pool.pop(0)
            queue.append(c)
    return pos


def reduce_tree(kgroups, hgroups, orient="op"):
    """{(x, y): {is_root, parent, children}} — the op's per-column doubling tree.

    `orient="op"`       : relative index r sits at y = (root_y + r) % kgroups (the SHIPPED mapping;
                          every edge then runs in the decreasing-y direction).
    `orient="mirror"`   : y = (root_y - r) % kgroups — flips every edge's direction.
    `orient="twosided"` : y = (root_y + twosided_offsets[r]) % kgroups — a parent's children
                          straddle it, so concurrent arrivals can use BOTH NoCs.
    """
    if orient == "twosided":
        offs = twosided_offsets(kgroups)
    elif orient == "mirror":
        offs = {r: -r for r in range(kgroups)}
    else:
        offs = {r: r for r in range(kgroups)}
    inv = {o % kgroups: r for r, o in offs.items()}
    info = {}
    for x in range(hgroups):
        root_y = x % kgroups

        def at(r):
            return (root_y + offs[r]) % kgroups

        for y in range(kgroups):
            r = inv[(y - root_y) % kgroups]
            children = [(x, at(c)) for c in _rel_children(r, kgroups)]
            p = _rel_parent(r)
            parent = None if p is None else (x, at(p))
            info[(x, y)] = {"is_root": r == 0, "parent": parent, "children": children}
    return info


def tree_max_fanin(tree):
    return max(len(n["children"]) for n in tree.values())


# ---------------------------------------------------------------------------
# Variant description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    n_ch: int = 2  # 2 = separate gate/up CBs (shipped); 1 = one merged CB
    owner: tuple = (1, 1)  # per channel: 0 = DM0/NCRISC/NOC_0, 1 = DM1/BRISC/NOC_1
    slots: int = 1
    use_invite: bool = True
    orient: str = "op"
    dir_noc: bool = False  # per-EDGE owner chosen from the physical hop direction
    per_slot_push: bool = False  # publish each landing slot as it arrives, not the whole wave
    mode: str = "add"  # "add" | "xfer"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def resolved_owner(self):
        return tuple(self.owner[: self.n_ch])


def _pow2_slots(slots, fan_in):
    return max(1, min(slots, fan_in))


# ---------------------------------------------------------------------------
# Host plumbing
# ---------------------------------------------------------------------------
def _core_range(hgroups, kgroups):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))],
    )


def _virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _cb(index, core_ranges, num_pages, page_size=BFP8_TILE_BYTES):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat8_b, page_size=page_size)
        ],
    )


def _kernel(source, core_ranges, ct_args, rt_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=config,
    )


def compute_config():
    """PRECISION CONTRACT — byte-identical to moe_fused_swiglu.default_compute_kernel_config().
    A FIXED input to every variant; never a lever in this bake-off."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def sharded_config(device, hgroups, kgroups, total_tiles):
    """One [32, total_tiles*32] bfp8 shard per core, ROW_MAJOR over the hgroups x kgroups grid, so
    shard index i lives on core (i % hgroups, i // hgroups)."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, total_tiles * TILE),
        core_grid=_core_range(hgroups, kgroups),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# ---------------------------------------------------------------------------
# The DATA-MOVEMENT kernel. ONE source, compiled twice per program: once as DM0
# (NCRISC / NOC_0 — the op's reader, which owns the PARENT side) and once as DM1
# (BRISC / NOC_1 — the op's writer, which owns the CHILD side in the shipped op).
#
# RAW-DATAFLOW BYPASS, carried from the real op and re-justified here: raw `noc_async_write` + two
# counting semaphores instead of `mcast_pipe`'s SenderPipe. A tree edge is POINT-TO-POINT with a
# per-node destination; SenderPipe is a rectangle multicast, and `mcast_pipe`'s
# `DataReadySignal::Counter` path is a documented HANG (inc_multicast gets the include-source fan-out
# while the atomic is exclude-source, and send_data_'s NOC_CMD_VC_LINKED chain is never released when
# the signal rides a different command buffer). No in-tree helper expresses this edge, so it stays
# raw — exactly as moe_fused_swiglu_writer.cpp already does.
# ---------------------------------------------------------------------------
_DM_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

// ---- RAW-DATAFLOW BYPASS (justified): raw unicast + counting semaphores, NOT mcast_pipe.
// A reduce-tree edge is point-to-point with a per-node destination, so `SenderPipe` (a rectangle
// multicast) does not express it, and `mcast_pipe`'s `DataReadySignal::Counter` path is a
// documented hang. Same bypass, same reason, as moe_fused_swiglu_writer.cpp's own tree edge.
//
// ---- THE LANDING-ADDRESS PROXY (preserved by every variant here).
// The child unicasts to its OWN `get_write_ptr(cb_in[ch]) + slot * stride` as a stand-in for the
// parent's address. That is valid because (a) every core has an identical CB layout and (b) the
// landing CB is reserved and pushed WHOLE, so its write pointer is always the CB base. Merging the
// two channels into one CB keeps both properties (one CB, still pushed whole).

constexpr uint32_t IS_DM0 = get_compile_time_arg_val(0);
constexpr uint32_t ADD_MODE = get_compile_time_arg_val(1);
constexpr uint32_t N_CH = get_compile_time_arg_val(2);
constexpr uint32_t CH_TILES = get_compile_time_arg_val(3);
constexpr uint32_t SLOTS = get_compile_time_arg_val(4);
constexpr uint32_t USE_INVITE = get_compile_time_arg_val(5);
constexpr uint32_t ARRIVALS_PER_CHILD = get_compile_time_arg_val(6);
constexpr uint32_t CB_IN_0 = get_compile_time_arg_val(7);
constexpr uint32_t CB_IN_1 = get_compile_time_arg_val(8);
constexpr uint32_t CB_LOCAL_0 = get_compile_time_arg_val(9);
constexpr uint32_t CB_LOCAL_1 = get_compile_time_arg_val(10);
// Half of the payload = one gate/up channel's worth of tiles. The prologue seed and the root's
// result write-out are ALWAYS issued as two HALF_BYTES transactions, in BOTH the split and the
// merged variant, so those two scaffolding stages are transaction-identical and cannot leak a
// "fewer transactions" advantage into a number that is supposed to measure the TREE EDGE only.
constexpr uint32_t HALF_TILES = get_compile_time_arg_val(11);
// Publish the landing CB ONE SLOT AT A TIME (as each slot's own arrival counter fires) instead of
// the whole wave at once. This is the "have both" protocol: several children in flight AND the
// shipped one-slot protocol's interleave, where the parent's compute starts adding child c while
// child c+1 is still on the wire.
constexpr uint32_t PER_SLOT_PUSH = get_compile_time_arg_val(12);

constexpr uint32_t CH_BYTES = CH_TILES * 1088;
constexpr uint32_t HALF_BYTES = HALF_TILES * 1088;
constexpr uint32_t SLOT_STRIDE = CH_TILES * 1088;

// Runtime args (identical layout on both DM RISC-Vs so the two share one source):
//   0 local_addr, 1 result_addr, 2 is_root, 3 num_children, 4 my_slot, 5 parent_x, 6 parent_y,
//   7 send_mask (bit ch set = THIS RISC-V ships channel ch), 8 last_slot,
//   9 final_cb_0, 10 final_cb_1, 11.. children (vx, vy) pairs
constexpr uint32_t RT_CHILDREN = 11;

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t result_addr = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t num_children = get_arg_val<uint32_t>(3);
    const uint32_t my_slot = get_arg_val<uint32_t>(4);
    const uint32_t parent_x = get_arg_val<uint32_t>(5);
    const uint32_t parent_y = get_arg_val<uint32_t>(6);
    const uint32_t send_mask = get_arg_val<uint32_t>(7);
    const uint32_t last_slot = get_arg_val<uint32_t>(8);
    const uint32_t final_cb_0 = get_arg_val<uint32_t>(9);
    const uint32_t final_cb_1 = get_arg_val<uint32_t>(10);

    const uint32_t sem_go_addr = static_cast<uint32_t>(get_semaphore(SEM_GO_ID));
    volatile tt_l1_ptr uint32_t* sem_go_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go_addr);
    const uint32_t sem_tree_addr = static_cast<uint32_t>(get_semaphore(SEM_TREE_ID));

    if (IS_DM0) {
        // ---- prologue (add mode only): seed the accumulator from this core's resident shard.
        // NOT the idea under measurement — it stands in for "the gate/up matmul already produced my
        // local partial", held identical in every variant.
        if (ADD_MODE) {
            cb_reserve_back(CB_LOCAL_0, CH_TILES);
            const uint32_t wp0 = get_write_ptr(CB_LOCAL_0);
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], local_addr), wp0, HALF_BYTES);
            if (N_CH == 2) {
                cb_reserve_back(CB_LOCAL_1, CH_TILES);
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], local_addr + HALF_BYTES),
                               get_write_ptr(CB_LOCAL_1), HALF_BYTES);
            } else {
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], local_addr + HALF_BYTES),
                               wp0 + HALF_BYTES, HALF_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_LOCAL_0, CH_TILES);
            if (N_CH == 2) {
                cb_push_back(CB_LOCAL_1, CH_TILES);
            }
        }

        // ---- PARENT side: invite in waves of SLOTS, wait on the per-slot MONOTONE counters ----
        // The whole landing CB is reserved and (in total) pushed per wave WHATEVER the push
        // granularity is: that is what keeps every core's landing write pointer wrapping back to the
        // CB base, which is the invariant the child's "my write pointer IS my parent's address"
        // proxy stands on. PER_SLOT_PUSH only changes WHEN the slots inside a wave are published.
        uint32_t round = 0;
        for (uint32_t c0 = 0; c0 < num_children; c0 += SLOTS) {
            uint32_t wave = num_children - c0;
            if (wave > SLOTS) {
                wave = SLOTS;
            }
            ++round;
            if (ADD_MODE) {
                cb_reserve_back(CB_IN_0, SLOTS * CH_TILES);
                if (N_CH == 2) {
                    cb_reserve_back(CB_IN_1, SLOTS * CH_TILES);
                }
            }
            if (USE_INVITE) {
                for (uint32_t c = c0; c < c0 + wave; ++c) {
                    const uint32_t cx = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 0);
                    const uint32_t cy = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 1);
                    noc_semaphore_inc(get_noc_addr(cx, cy, sem_go_addr), 1);
                }
            }
            for (uint32_t s = 0; s < wave; ++s) {
                volatile tt_l1_ptr uint32_t* slot_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    static_cast<uint32_t>(get_semaphore(SEM_DATA_BASE_ID + s)));
                noc_semaphore_wait_min(slot_ptr, round * ARRIVALS_PER_CHILD);
                if (ADD_MODE && PER_SLOT_PUSH) {
                    cb_push_back(CB_IN_0, CH_TILES);
                    if (N_CH == 2) {
                        cb_push_back(CB_IN_1, CH_TILES);
                    }
                }
            }
            if (ADD_MODE) {
                // Pad the wave back up to the WHOLE CB so the write pointer wraps (see above).
                const uint32_t tail = PER_SLOT_PUSH ? (SLOTS - wave) * CH_TILES : SLOTS * CH_TILES;
                if (tail) {
                    cb_push_back(CB_IN_0, tail);
                    if (N_CH == 2) {
                        cb_push_back(CB_IN_1, tail);
                    }
                }
            }
        }

        // xfer mode has no compute kernel, so the tree's serialisation (a node cannot ship until its
        // own children have landed) is carried by this same-core semaphore instead of by the
        // accumulator CB handoff. Raised BEFORE this RISC-V's own ship so DM1 starts concurrently.
        if (!ADD_MODE) {
            noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], sem_tree_addr), 1);
            noc_async_atomic_barrier();
        }
    } else if (!ADD_MODE) {
        volatile tt_l1_ptr uint32_t* sem_tree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_tree_addr);
        noc_semaphore_wait_min(sem_tree_ptr, 1);
    }

    // ---- CHILD side: ship the channels this RISC-V owns into the parent's landing slot ----
    if (!is_root && send_mask != 0) {
        const bool own0 = (send_mask & 1u) != 0;
        const bool own1 = (send_mask & 2u) != 0;
        uint32_t src0 = local_addr;
        uint32_t src1 = local_addr + CH_BYTES;
        if (ADD_MODE) {
            if (own0) {
                cb_wait_front(final_cb_0, CH_TILES);
                src0 = get_read_ptr(final_cb_0);
            }
            if (own1) {
                cb_wait_front(final_cb_1, CH_TILES);
                src1 = get_read_ptr(final_cb_1);
            }
        }
        if (USE_INVITE) {
            noc_semaphore_wait_min(sem_go_ptr, 1);
        }
        const uint32_t slot_bytes = my_slot * SLOT_STRIDE;
        if (own0) {
            noc_async_write(src0, get_noc_addr(parent_x, parent_y, get_write_ptr(CB_IN_0) + slot_bytes), CH_BYTES);
        }
        if (own1) {
            noc_async_write(src1, get_noc_addr(parent_x, parent_y, get_write_ptr(CB_IN_1) + slot_bytes), CH_BYTES);
        }
        noc_async_write_barrier();
        // Signal the counter belonging to MY landing slot, so the parent knows WHICH slot is ready.
        noc_semaphore_inc(
            get_noc_addr(parent_x, parent_y, static_cast<uint32_t>(get_semaphore(SEM_DATA_BASE_ID + my_slot))), 1);
        // This remote atomic is the last NoC op on this RISC-V in xfer mode; without a flush the
        // firmware's exit-time ASSERT(ncrisc_noc_nonposted_atomics_flushed) trips.
        noc_async_atomic_barrier();
        if (ADD_MODE) {
            if (own0) {
                cb_pop_front(final_cb_0, CH_TILES);
            }
            if (own1) {
                cb_pop_front(final_cb_1, CH_TILES);
            }
        }
    }

    // ---- root: commit the answer into the resident result shard (correctness gate) ----
    if (IS_DM0 && is_root) {
        if (ADD_MODE) {
            cb_wait_front(final_cb_0, CH_TILES);
            const uint32_t rp0 = get_read_ptr(final_cb_0);
            noc_async_write(rp0, get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr), HALF_BYTES);
            if (N_CH == 2) {
                cb_wait_front(final_cb_1, CH_TILES);
                noc_async_write(get_read_ptr(final_cb_1),
                                get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr + HALF_BYTES), HALF_BYTES);
            } else {
                noc_async_write(rp0 + HALF_BYTES,
                                get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr + HALF_BYTES), HALF_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(final_cb_0, CH_TILES);
            if (N_CH == 2) {
                cb_pop_front(final_cb_1, CH_TILES);
            }
        } else {
            // xfer mode: prove the LAST child's bytes landed at the right address by echoing the
            // landing slot it wrote into. `last_slot` is host-known from the tree + SLOTS.
            const uint32_t off = last_slot * SLOT_STRIDE;
            const uint32_t base0 = get_write_ptr(CB_IN_0) + off;
            noc_async_write(base0, get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr), HALF_BYTES);
            if (N_CH == 2) {
                noc_async_write(get_write_ptr(CB_IN_1) + off,
                                get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr + HALF_BYTES), HALF_BYTES);
            } else {
                noc_async_write(base0 + HALF_BYTES,
                                get_noc_addr(my_x[noc_index], my_y[noc_index], result_addr + HALF_BYTES), HALF_BYTES);
            }
            noc_async_write_barrier();
        }
    }
}
"""


# ---------------------------------------------------------------------------
# The COMPUTE kernel (add mode only). Generated per (n_ch, slots, ch_tiles) with LITERAL CB indices
# and the wave/pop bookkeeping fully unrolled per fan-in, so the trip counts can never diverge from
# the reader's pushes (this op has hung twice from exactly that mismatch).
#
# CB DISCIPLINE: a LINEAR CHAIN of single-use CBs, not the op's true in-place `add<a,b,a>`. Each CB
# has exactly one producer and one consumer for the kernel's whole lifetime. (The sibling
# `reduce_tree_shape` bench measured a genuine device hang from in-place / ping-pong CB reuse in an
# isolated harness.) The add COUNT and per-tile work are identical either way, and the add sequence
# is byte-identical across every variant here, so this choice cannot bias the transport A/B.
# ---------------------------------------------------------------------------
def _compute_source(n_ch, slots, ch_tiles, eltwise_blk, max_fanin):
    chans = range(n_ch)
    cases = []
    for nc in range(max_fanin + 1):
        body = []
        prev = [CB_LOCAL[ch] for ch in chans]
        step = 0
        for c0 in range(0, nc, slots):
            wave = min(slots, nc - c0)
            for _c in range(c0, c0 + wave):
                step += 1
                for ch in chans:
                    nxt = CB_STEP[ch][step - 1]
                    body.append(
                        f"            add<blk_in({prev[ch]}), blk_in({CB_IN[ch]}), blk_out({nxt})>"
                        f"(blk_shape({ch_tiles}));"
                    )
                    prev[ch] = nxt
            if wave < slots:
                # Drain the slots this wave did NOT fill, so the per-CB pop TOTAL matches the
                # parent's whole-CB push exactly (the op does the same with rg_buf/ru_buf.pop_front).
                # `cb_pop_front` from api/compute/cb_api.h, not the CircularBuffer wrapper: the
                # wrapper lives behind `api/dataflow/circular_buffer.h`, a dataflow-only include that
                # a compute TU only sees by accident (via tilize_helpers.hpp) — compiling it directly
                # into a compute kernel fails with 'CircularBuffer was not declared'.
                for ch in chans:
                    body.append(f"            cb_pop_front({CB_IN[ch]}, {(slots - wave) * ch_tiles});")
        cases.append(
            "        case {}: {{\n{}\n            break;\n        }}".format(nc, "\n".join(body) or "            ;")
        )

    return f"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

using namespace compute_kernel_lib;

// PERF-1 BLOCKED ELTWISE (the shipped shape): `input(cb)`/`output(cb)` default to per-TILE
// wait/pop/reserve/push, which makes eltwise_chain silently clamp block_size to 1. `PerChunk` +
// `OperandKind::Block` + `EltwiseShape::tiles(n, blk)` is the graduated fast path — copied verbatim
// from moe_fused_swiglu_compute.cpp so the add cost this transport hides under is the REAL one.
constexpr auto blk_in(uint32_t cb) {{ return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }}
constexpr auto blk_out(uint32_t cb) {{ return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }}
ALWI auto blk_shape(uint32_t n) {{ return EltwiseShape::tiles(n, {eltwise_blk}); }}

void kernel_main() {{
    const uint32_t num_children = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup({CB_LOCAL[0]}, {CB_IN[0]}, {CB_STEP[0][0]});

    switch (num_children) {{
{chr(10).join(cases)}
        default: break;
    }}
}}
"""


# ---------------------------------------------------------------------------
# Program construction
# ---------------------------------------------------------------------------
#: NoC grid height used for torus-wrapped hop counting. Blackhole's NoC is a 17 x 12 torus (the
#: 11 x 10 worker grid is a harvested subset of it), so an edge's hop count in one direction is
#: `delta mod 12`, not `abs(delta)`.
NOC_GRID_Y = 12


def build(
    device, local_tensor, result_tensor, variant, hgroups, kgroups, half_tiles, eltwise_blk=8, torus_y=NOC_GRID_Y
):
    tree = reduce_tree(kgroups, hgroups, variant.orient)
    fan_in = tree_max_fanin(tree)
    slots = _pow2_slots(variant.slots, fan_in)
    n_ch = variant.n_ch
    ch_tiles = (2 * half_tiles) // n_ch
    add_mode = 1 if variant.mode == "add" else 0
    core_range = _core_range(hgroups, kgroups)

    owner = variant.resolved_owner()
    if variant.dir_noc:
        arrivals_per_child = 1  # every child ships all its channels from ONE RISC-V
    else:
        arrivals_per_child = len(set(owner))

    rt = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    compute_rt = ttnn.RuntimeArgs()
    slot_of = {}
    for node in tree.values():
        for c, ch in enumerate(node["children"]):
            slot_of[ch] = c % slots

    n_dir_noc0 = 0
    hops = {0: 0, 1: 0}
    for x in range(hgroups):
        for y in range(kgroups):
            info = tree[(x, y)]
            is_root = 1 if info["is_root"] else 0
            children = info["children"]
            nc = len(children)
            coords = []
            for cx, cy in children:
                vx, vy = _virt(device, cx, cy)
                coords += [vx, vy]
            if info["parent"] is not None:
                px, py = info["parent"]
                pvx, pvy = _virt(device, px, py)
            else:
                pvx, pvy = 0, 0

            if variant.dir_noc and not is_root:
                # SHORTEST-PATH, DIRECTION-MATCHED NoC. NOC_1 routes in the DECREASING physical
                # direction and NOC_0 in the increasing one, and BOTH wrap the torus (NOC_GRID_Y
                # rows, which includes the non-worker DRAM/PCIe/ETH rows). So the hop count of an
                # edge is `(child - parent) mod GRID_Y` on NOC_1 and `(parent - child) mod GRID_Y` on
                # NOC_0 — a RAW physical `parent_vy < child_vy` test gets the wrapped edges exactly
                # backwards. Measured: a single contention-free edge costs the SAME on either NoC
                # (3 140 vs 3 197 ns for 104 448 B), so this is NOT about one link being faster; it
                # is that a mismatched edge takes the LONG way round and 99 concurrent long paths
                # overlap into congestion (measured 1.92x, and the same 6x class that
                # `tensix_all_reduce_ring_transport` reports).
                _, my_vy = _virt(device, x, y)
                d_dec = (my_vy - pvy) % torus_y
                d_inc = (pvy - my_vy) % torus_y
                risc = 1 if d_dec <= d_inc else 0
                edge_owner = (risc,) * n_ch
                hops[risc] += min(d_dec, d_inc)
                n_dir_noc0 += 1 if risc == 0 else 0
            else:
                edge_owner = owner
                if not is_root:
                    _, my_vy = _virt(device, x, y)
                    d_dec = (my_vy - pvy) % torus_y
                    d_inc = (pvy - my_vy) % torus_y
                    for o in set(owner):
                        hops[o] += d_dec if o == 1 else d_inc

            masks = [0, 0]
            if not is_root:
                for ch in range(n_ch):
                    masks[edge_owner[ch]] |= 1 << ch

            final0 = CB_STEP[0][nc - 1] if nc > 0 else CB_LOCAL[0]
            final1 = CB_STEP[1][nc - 1] if nc > 0 else CB_LOCAL[1]
            last_slot = ((nc - 1) % slots) if nc > 0 else 0
            base = [
                local_tensor.buffer_address(),
                result_tensor.buffer_address(),
                is_root,
                nc,
                slot_of.get((x, y), 0),
                pvx,
                pvy,
                0,
                last_slot,
                final0,
                final1,
            ]
            for r in (0, 1):
                rt[r][x][y] = base[:7] + [masks[r]] + base[8:] + coords
            compute_rt[x][y] = [nc]

    n_step = max(1, fan_in)
    cbs = []
    pages = 0
    for ch in range(n_ch):
        cbs.append(_cb(CB_LOCAL[ch], core_range, ch_tiles))
        cbs.append(_cb(CB_IN[ch], core_range, slots * ch_tiles))
        pages += ch_tiles + slots * ch_tiles
        if add_mode:
            for s in range(n_step):
                cbs.append(_cb(CB_STEP[ch][s], core_range, ch_tiles))
            pages += n_step * ch_tiles
    l1_bytes = pages * BFP8_TILE_BYTES
    # The bytes this variant costs in the REAL op relative to the shipped shape: only the landing CBs
    # scale with `slots` (the op already owns everything else), so this is the number the coordinator
    # has to fit into its 143 360 B free budget.
    landing_l1_delta = (slots - 1) * n_ch * ch_tiles * BFP8_TILE_BYTES

    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=core_range, initial_value=0)]
    for s in range(MAX_FANIN):
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_DATA_BASE + s, core_ranges=core_range, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_TREE, core_ranges=core_range, initial_value=0))

    defines = (
        f"#define SEM_GO_ID {SEM_GO}\n#define SEM_DATA_BASE_ID {SEM_DATA_BASE}\n" f"#define SEM_TREE_ID {SEM_TREE}\n"
    )
    dm_source = _DM_KERNEL.replace("#include <stdint.h>", defines + "#include <stdint.h>", 1)

    def ct(is_dm0):
        return [
            1 if is_dm0 else 0,
            add_mode,
            n_ch,
            ch_tiles,
            slots,
            1 if variant.use_invite else 0,
            arrivals_per_child,
            CB_IN[0],
            CB_IN[1],
            CB_LOCAL[0],
            CB_LOCAL[1],
            half_tiles,
            1 if variant.per_slot_push else 0,
        ]

    kernels = [
        _kernel(
            dm_source,
            core_range,
            ct(True),
            rt[0],
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        _kernel(
            dm_source,
            core_range,
            ct(False),
            rt[1],
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
    ]
    if add_mode:
        kernels.append(
            _kernel(
                _compute_source(n_ch, slots, ch_tiles, eltwise_blk, n_step),
                core_range,
                [],
                compute_rt,
                compute_config(),
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    meta = {
        "fan_in": fan_in,
        "slots": slots,
        "n_ch": n_ch,
        "ch_tiles": ch_tiles,
        "arrivals_per_child": arrivals_per_child,
        "cb_l1_bytes": l1_bytes,
        "landing_l1_delta": landing_l1_delta,
        "n_edges_on_noc0": n_dir_noc0 if variant.dir_noc else sum(1 for o in owner if o == 0),
        # Total ROUTER HOPS the payload traverses on each NoC (torus-wrapped). This is the single
        # number that predicts every result in this bake-off.
        "hops": (hops[0], hops[1]),
        "tree": tree,
    }
    return descriptor, meta


def run(device, local_tensor, result_tensor, variant, hgroups, kgroups, half_tiles, eltwise_blk=8):
    descriptor, meta = build(
        device, local_tensor, result_tensor, variant, hgroups, kgroups, half_tiles, eltwise_blk=eltwise_blk
    )
    ttnn.generic_op([local_tensor, result_tensor], descriptor)
    return meta
