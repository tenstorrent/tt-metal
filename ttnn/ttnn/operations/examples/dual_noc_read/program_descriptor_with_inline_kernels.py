# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two INDEPENDENT DRAM operand streams, ONE core, and how many read engines fetch them.

A Tensix core has FIVE RISC-V processors, of which TWO move data: NCRISC (conventionally the
"reader") and BRISC (conventionally the "writer"). Each is bound to one NoC — NCRISC to NoC 0,
BRISC to NoC 1 — one processor, one port.

A fused kernel often needs two operands with no dependency between them: it computes `A op1 B` and
`A op2 C`, where B and C are different DRAM tensors and nothing requires fetching one before the
other. Written the natural way, ONE reader fetches both in series and BRISC does nothing for the
whole read phase — so every read queues behind a single data-movement processor.

The trick isolated here: give one operand to each data-movement RISC so both issue concurrently.
BRISC is free to take it during the read-heavy phase, before compute has produced anything to write.

It is NOT free, though: one RISC is one NoC, so handing an operand to BRISC also moves it onto
NoC 1 — and NoC 1 is the worse route for DRAM reads at spread core placements (NoC 0's east->south
routing disperses column-localized DRAM traffic; NoC 1's north->west concentrates it). Whether the
extra issue engine outweighs that depends on transaction size and core count: at a bf16 tile page on
a mid-size grid it does not, and the split is a REGRESSION. See README.md for the measured matrix.

  one_riscv     (BASELINE)  NCRISC reads A and B, both on NoC 0. BRISC idle.
  two_riscv                 NCRISC reads A on NoC 0; BRISC reads B on NoC 1. Each RISC OWNS its own
                            CB (its own reserve/push), so plain CB semantics order everything and
                            NO semaphore is needed. This is the cheap form — reach for it first.
  two_riscv_sem             Same split, but BRISC fills a slot the READER owns, and two local
                            same-core semaphores order the two RISCs (go = slot reserved,
                            done = bytes landed). This is what you need when the reader must do
                            something to the block after it lands (forward it, multicast it,
                            re-page it) and therefore has to keep ownership of the CB. It buys
                            the same concurrency and pays a handshake per block.

The SECOND axis is READS IN FLIGHT (`block`): each RISC issues `block` async reads back-to-back and
then waits on ONE barrier. Few in flight is latency-bound (each round trip is exposed); many in
flight lets the transfers pipeline. The baseline is given the FAIR treatment — it issues all 2*block
reads (A's and B's) before its single barrier, so it is never handicapped by extra barriers. The only
thing that differs between variants is WHICH data-movement RISC issues each read.

Isolation (read-only, no write traffic):
  * inputs A and B are DRAM interleaved — the reads under study;
  * the output C = A*B is L1 HEIGHT-SHARDED on the same single core and its CB is ALIASED to the
    tensor, so compute packs straight into the output buffer and NO kernel drains it to DRAM. The
    measured kernel therefore moves DRAM bytes in ONE direction only, and BRISC is free to be a
    pure second read engine (rather than splitting its time with writes);
  * compute is one `mul_tiles` per tile pair, identical in every variant — dummy math, there only
    to consume the operands;
  * ONE core, so the number is about this core's own data-movement capacity, not cross-core NoC
    contention.

What the win is actually relieving is measured, not assumed: it is predominantly the RISC-V's
COMMAND-ISSUE rate, not link bandwidth. See README.md ("What is actually being relieved").

Because nothing drains the output CB, a launch performs exactly one pass over the tiles — there is
no `kernel_iters` knob. Amortize launch overhead with the tile count instead (`--shape`).
"""

import ttnn

TILE = 32

# CB assignment. cb_a and cb_b are the two DRAM-streamed operands; cb_out is aliased to the
# L1-sharded output tensor, so compute's pack lands directly in the output buffer.
CB_A = 0
CB_B = 1
CB_OUT = 16

# Semaphore ids (indices into ProgramDescriptor.semaphores). Only two_riscv_sem uses them.
SEM_GO = 0  # reader -> BRISC: cb_b slot is reserved, go fill it
SEM_DONE = 1  # BRISC -> reader: bytes have landed in that slot

# Baseline first. The variants differ ONLY in which data-movement RISC issues each read.
#
# One RISC is one NoC: each data-movement processor is bound to a single port (NCRISC->NoC 0,
# BRISC->NoC 1), and firmware initializes only that port's per-RISC state
# (`noc_local_state_init(noc_index)`). So "add a RISC" and "add a port" are one and the same knob —
# there is no 2x2 of RISCs x ports to sweep, and the two contributions are not separable by A/B.
# The mechanism is therefore established by measurement instead: a transactions-vs-bytes sweep
# (holding bytes fixed while scaling command count) plus an independent NoC-model bound. Both say
# issue rate dominates. See README.md.
VARIANTS = ("one_riscv", "two_riscv", "two_riscv_sem")
BASELINE = "one_riscv"

# DIAGNOSTIC (not part of the ladder): the mirror of the baseline — BRISC reads BOTH operands, so
# ALL reads ride NoC 1 instead of NoC 0, with the SAME single-RISC issue load. Since a RISC and
# its port move together, this is the only way to see the port/route contribution on its own: any
# gap vs `one_riscv` is attributable to the NoC, not to issue rate. Matters at multi-core
# placements, where the two NoCs route DRAM traffic very differently.
DIAG_VARIANT = "one_riscv_brisc"
ALL_VARIANTS = VARIANTS + (DIAG_VARIANT,)

# Per-variant kernel wiring, passed to the kernels as explicit compile-time flags (clearer than a
# mode enum decoded in the kernel):
#   rd_b   — does NCRISC issue B's reads?
#   rd_own — does NCRISC own cb_b (reserve + push)?
#   wr_b   — does BRISC issue B's reads?
#   wr_own — does BRISC own cb_b?
#   sem    — use the two-semaphore handshake (reader keeps cb_b ownership)
# Each RISC always uses its own firmware-assigned NoC (NCRISC 0, BRISC 1) — see the note above.
_WIRING = {
    #                 rd_b rd_own  wr_b wr_own  sem
    "one_riscv": (1, 1, 0, 0, 0),
    "two_riscv": (0, 0, 1, 1, 0),
    "two_riscv_sem": (0, 1, 1, 0, 1),
    # diagnostic: NCRISC idle, BRISC reads A and B (all reads on NoC 1)
    "one_riscv_brisc": (0, 0, 1, 1, 0),
}

# Which RISC reads operand A. Only the diagnostic moves it off NCRISC.
_A_ON_WRITER = {"one_riscv": 0, "two_riscv": 0, "two_riscv_sem": 0, "one_riscv_brisc": 1}

# CB depth in blocks. Double-buffered so a RISC can prefetch block N+1 while compute drains N.
# two_riscv_sem's slot arithmetic depends on this exact value (see the writer kernel).
CB_DEPTH_BLOCKS = 2

LABEL = {
    "one_riscv": "NCRISC reads A+B, both on NoC0 (BRISC idle)",
    "two_riscv": "NCRISC:A/NoC0 + BRISC:B/NoC1, each owns its CB",
    "two_riscv_sem": "NCRISC:A/NoC0 + BRISC:B/NoC1, reader owns cb_b (2 sems)",
    "one_riscv_brisc": "DIAG: BRISC reads A+B, both on NoC1 (NCRISC idle)",
}

# The two factors, for the report: how many RISC-Vs issue reads, how many NoC ports carry them.
# RISCs and NoCs move together (1:1 firmware binding), so these are always equal — both are reported
# to keep the reader honest about the fact that the two factors are NOT independently varied.
RISCS = {"one_riscv": 1, "two_riscv": 2, "two_riscv_sem": 2, "one_riscv_brisc": 1}
NOCS = {"one_riscv": 1, "two_riscv": 2, "two_riscv_sem": 2, "one_riscv_brisc": 1}


# =============================================================================
# Reader (NCRISC / NoC 0) — always reads A; also reads B in the baseline; owns cb_b in the
# semaphore variant.
#
# CT: [page_bytes, block, mode, go_sem, done_sem, <A accessor>, <B accessor>]
# RT: [a_addr, b_addr, num_tiles]
# =============================================================================
_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t page_bytes  = get_compile_time_arg_val(0);
    constexpr uint32_t block       = get_compile_time_arg_val(1);
    constexpr uint32_t txn_bytes   = get_compile_time_arg_val(2);  // bytes per NoC transaction
    constexpr uint32_t reads_b     = get_compile_time_arg_val(3);  // does THIS RISC fetch B?
    constexpr uint32_t owns_b      = get_compile_time_arg_val(4);  // does THIS RISC reserve/push cb_b?
    constexpr uint32_t use_sem     = get_compile_time_arg_val(5);
    constexpr uint32_t reads_a     = get_compile_time_arg_val(6);  // 0 only in the NoC-1 diagnostic
    constexpr uint32_t go_sem_id   = get_compile_time_arg_val(7);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t txns_per_page = page_bytes / txn_bytes;
    constexpr auto a_args = TensorAccessorArgs<9>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t a_addr     = get_arg_val<uint32_t>(0);
    const uint32_t b_addr     = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles  = get_arg_val<uint32_t>(2);  // THIS core's tile count
    const uint32_t start_page = get_arg_val<uint32_t>(3);  // THIS core's first page

    const auto a_acc = TensorAccessor(a_args, a_addr, page_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, page_bytes);

    Semaphore<> go_sem(go_sem_id);
    Semaphore<> done_sem(done_sem_id);
    uint32_t seq = 0;

    if constexpr (!reads_a && !reads_b && !owns_b) {
        return;  // NoC-1 diagnostic: BRISC owns everything, NCRISC contributes nothing.
    }

    for (uint32_t p = 0; p < num_tiles; p += block) {
        cb_reserve_back(cb_a, block);
        if constexpr (owns_b) {
            cb_reserve_back(cb_b, block);
        }
        if constexpr (use_sem) {
            // Slot reserved -> release BRISC to fill it, CONCURRENT with our own read of A just
            // below. This is the whole point of the handshake.
            ++seq;
            go_sem.set(seq);
        }

        // Each tile page is fetched in `txns_per_page` transactions of txn_bytes. At the default
        // txn_bytes == page_bytes that is one read per tile; smaller txn_bytes moves the SAME bytes
        // in proportionally MORE transactions, which is the commands-vs-bytes discriminator.
        const uint32_t a_l1 = get_write_ptr(cb_a);
        for (uint32_t i = 0; i < block; ++i) {
            for (uint32_t t = 0; t < txns_per_page; ++t) {
                noc_async_read(
                    a_acc.get_noc_addr(start_page + p + i) + t * txn_bytes, a_l1 + i * page_bytes + t * txn_bytes, txn_bytes);
            }
        }
        if constexpr (reads_b) {
            // BASELINE: this RISC issues B's reads too, on its OWN (only) NoC. All of them go out
            // before the single barrier below, so this path is never handicapped by an extra
            // barrier — it just has one port to push everything through.
            const uint32_t b_l1 = get_write_ptr(cb_b);
            for (uint32_t i = 0; i < block; ++i) {
                for (uint32_t t = 0; t < txns_per_page; ++t) {
                    noc_async_read(
                        b_acc.get_noc_addr(start_page + p + i) + t * txn_bytes, b_l1 + i * page_bytes + t * txn_bytes, txn_bytes);
                }
            }
        }
        noc_async_read_barrier();

        if constexpr (use_sem) {
            done_sem.wait_min(seq);  // B's bytes are in the slot we reserved
        }
        cb_push_back(cb_a, block);
        if constexpr (owns_b) {
            cb_push_back(cb_b, block);
        }
    }
}
"""


# =============================================================================
# Writer (BRISC / NoC 1) — the second read engine. Present in EVERY variant (so the launch shape
# and kernel count are identical) but issues no transactions in the baseline.
#
# Despite the name, this kernel writes nothing: the output is L1-sharded and compute packs straight
# into it. `WriterConfigDescriptor` is used purely to land this kernel on BRISC, whose default NoC
# is 1 — so the plain `noc_async_read` calls below go out on the OTHER NoC from the reader's.
#
# CT: [page_bytes, block, mode, depth, go_sem, done_sem, <B accessor>]
# RT: [b_addr, num_tiles]
# =============================================================================
_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t page_bytes  = get_compile_time_arg_val(0);
    constexpr uint32_t block       = get_compile_time_arg_val(1);
    constexpr uint32_t txn_bytes   = get_compile_time_arg_val(2);  // bytes per NoC transaction
    constexpr uint32_t reads_b     = get_compile_time_arg_val(3);  // does THIS RISC fetch B?
    constexpr uint32_t owns_b      = get_compile_time_arg_val(4);  // does THIS RISC reserve/push cb_b?
    constexpr uint32_t use_sem     = get_compile_time_arg_val(5);
    constexpr uint32_t reads_a     = get_compile_time_arg_val(6);  // 1 only in the NoC-1 diagnostic
    constexpr uint32_t depth       = get_compile_time_arg_val(7);  // cb_b depth in blocks
    constexpr uint32_t go_sem_id   = get_compile_time_arg_val(8);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t txns_per_page = page_bytes / txn_bytes;
    constexpr auto b_args = TensorAccessorArgs<10>();
    constexpr auto a_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    if constexpr (!reads_b) {
        return;  // BASELINE: BRISC's NoC-1 port sits idle. This is the thing being fixed.
    }

    constexpr uint32_t cb_a = 0;
    const uint32_t b_addr     = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles  = get_arg_val<uint32_t>(1);  // THIS core's tile count
    const uint32_t start_page = get_arg_val<uint32_t>(2);  // THIS core's first page
    const uint32_t a_addr     = get_arg_val<uint32_t>(3);  // used only by the NoC-1 diagnostic
    const auto b_acc = TensorAccessor(b_args, b_addr, page_bytes);
    const auto a_acc = TensorAccessor(a_args, a_addr, page_bytes);

    if constexpr (owns_b) {
        // BRISC OWNS cb_b end to end — its own reserve/push. No semaphore: the CB's own
        // producer/consumer protocol already orders BRISC against compute, and compute waits on
        // both CBs independently. Cheapest correct form of the split.
        for (uint32_t p = 0; p < num_tiles; p += block) {
            cb_reserve_back(cb_b, block);
            if constexpr (reads_a) {
                cb_reserve_back(cb_a, block);
            }
            const uint32_t l1 = get_write_ptr(cb_b);
            for (uint32_t i = 0; i < block; ++i) {
                for (uint32_t t = 0; t < txns_per_page; ++t) {
                    noc_async_read(
                        b_acc.get_noc_addr(start_page + p + i) + t * txn_bytes, l1 + i * page_bytes + t * txn_bytes, txn_bytes);
                }
            }
            if constexpr (reads_a) {
                // Diagnostic only: A's reads go out on THIS RISC's NoC too, so all traffic is on NoC 1
                // with a single-RISC issue load — the mirror of the baseline.
                const uint32_t l1a = get_write_ptr(cb_a);
                for (uint32_t i = 0; i < block; ++i) {
                    for (uint32_t t = 0; t < txns_per_page; ++t) {
                        noc_async_read(
                            a_acc.get_noc_addr(start_page + p + i) + t * txn_bytes, l1a + i * page_bytes + t * txn_bytes, txn_bytes);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_b, block);
            if constexpr (reads_a) {
                cb_push_back(cb_a, block);
            }
        }
    } else {
        // two_riscv_sem: the READER owns cb_b (reserve + push); we only deposit bytes. A CB write
        // pointer is PER-RISC, so ours never advances (the reader does the pushing) — we must
        // reconstruct the live slot ourselves. Replicating the reader's cadence (one push per
        // block into a `depth`-slot ring) makes that pure modular arithmetic. This is why the
        // example REQUIRES num_tiles % block == 0: a short tail block would advance the reader's
        // pointer by less than a full slot and desynchronize this model.
        const uint32_t base = get_write_ptr(cb_b);
        const uint32_t slot_bytes = block * page_bytes;
        Semaphore<> go_sem(go_sem_id);
        Semaphore<> done_sem(done_sem_id);
        uint32_t seq = 0;
        for (uint32_t p = 0; p < num_tiles; p += block) {
            ++seq;
            go_sem.wait_min(seq);  // reader has reserved the slot
            const uint32_t l1 = base + ((seq - 1) % depth) * slot_bytes;
            for (uint32_t i = 0; i < block; ++i) {
                for (uint32_t t = 0; t < txns_per_page; ++t) {
                    noc_async_read(
                        b_acc.get_noc_addr(start_page + p + i) + t * txn_bytes, l1 + i * page_bytes + t * txn_bytes, txn_bytes);
                }
            }
            noc_async_read_barrier();
            done_sem.set(seq);  // reader may now push both CBs
        }
    }
}
"""


# =============================================================================
# Compute (unpack/math/pack TRISCs) — IDENTICAL in every variant, so no measured difference can
# come from here. One FPU eltwise multiply per tile pair: C = A * B.
#
# Works in DST-sized chunks (`chunk` <= 8 bf16 tiles) purely to keep the per-tile sync overhead
# off the critical path; the chunk is the same for every variant at a given block.
#
# `do_math=0` is a PAYLOAD ABLATION for perf diagnosis only: it drops the `mul_tiles` calls while
# keeping every CB wait/reserve/push/pop and the whole tile_regs + pack cycle intact. The output is
# then garbage (never correctness-checked), but the read pipeline and all its synchronization are
# unchanged — so it exposes the pure READ ceiling and reveals whether the FPU is masking the
# read-side win at large blocks.
#
# CT: [chunk, do_math]   RT: [num_tiles]
# =============================================================================
_COMPUTE = r"""
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t chunk = get_compile_time_arg_val(0);
    constexpr uint32_t do_math = get_compile_time_arg_val(1);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b);

    for (uint32_t t = 0; t < num_tiles; t += chunk) {
        const uint32_t c = (num_tiles - t) < chunk ? (num_tiles - t) : chunk;
        cb_wait_front(cb_a, c);
        cb_wait_front(cb_b, c);
        cb_reserve_back(cb_out, c);

        tile_regs_acquire();
        if constexpr (do_math) {
            for (uint32_t i = 0; i < c; ++i) {
                mul_tiles(cb_a, cb_b, i, i, i);
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < c; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, c);
        cb_pop_front(cb_a, c);
        cb_pop_front(cb_b, c);
    }
}
"""


# =============================================================================
# Host side
# =============================================================================
_DST_TILES = 8  # bf16 DST capacity (half-sync, no fp32 dest acc) — caps the compute chunk


def _core_grid(grid_x=1, grid_y=1):
    """A grid_x by grid_y rectangle anchored at (0,0). One CoreRange with ROW_MAJOR shard
    orientation, so shard index == y * grid_x + x — the same order the page assignment uses."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])


def create_output_memory_config(shape, grid_x=1, grid_y=1):
    """Height-shard the output across the grid so compute's pack lands directly in the output
    tensor and no kernel has to drain it to DRAM. Each core owns tiles_per_core tile-rows."""
    h, w = list(shape)
    num_cores = grid_x * grid_y
    if h % num_cores:
        raise ValueError(f"dual_noc_read example: output height {h} must divide over {num_cores} cores")
    return ttnn.create_sharded_memory_config(
        (h // num_cores, w),
        core_grid=_core_grid(grid_x, grid_y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def validate(a, b, block, txn_bytes=None, num_cores=1):
    for name, t in (("a", a), ("b", b)):
        shape = list(t.shape)
        if len(shape) != 2:
            raise ValueError(f"dual_noc_read example: {name} rank must be 2, got {len(shape)}")
        if t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"dual_noc_read example: {name} must be TILE_LAYOUT")
        if t.dtype != ttnn.bfloat16:
            raise ValueError(f"dual_noc_read example: {name} must be bfloat16, got {t.dtype}")
        if shape[0] % TILE or shape[1] % TILE:
            raise ValueError(f"dual_noc_read example: {name} H and W must be multiples of {TILE}, got {shape}")
        if t.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(f"dual_noc_read example: {name} must live in DRAM (the reads under study)")
    if list(a.shape) != list(b.shape):
        raise ValueError(f"dual_noc_read example: a and b must have the same shape, got {a.shape} vs {b.shape}")
    if block < 1:
        raise ValueError(f"dual_noc_read example: block must be >= 1, got {block}")

    h, w = list(a.shape)
    total_tiles = (h // TILE) * (w // TILE)
    if num_cores > 1:
        if total_tiles % num_cores:
            raise ValueError(f"dual_noc_read example: total tiles ({total_tiles}) must divide over {num_cores} cores")
        if (total_tiles // num_cores) % block:
            raise ValueError(
                f"dual_noc_read example: tiles per core ({total_tiles // num_cores}) must be "
                f"divisible by block ({block})"
            )
    if txn_bytes is not None:
        page_bytes = a.buffer_aligned_page_size()
        if txn_bytes < 32 or txn_bytes > page_bytes or page_bytes % txn_bytes:
            raise ValueError(
                f"dual_noc_read example: txn_bytes ({txn_bytes}) must be >= 32 and divide the "
                f"{page_bytes} B tile page"
            )
    if total_tiles % block:
        # two_riscv_sem reconstructs cb_b slot addresses with modular arithmetic that assumes
        # every block is full width; a short tail would desynchronize it. Required of all
        # variants so they run identical work.
        raise ValueError(
            f"dual_noc_read example: total tiles ({total_tiles}) must be divisible by block ({block}); "
            "the semaphore variant's slot arithmetic assumes full blocks"
        )
    return total_tiles


def compute_chunk(block):
    """DST-limited compute chunk. Same for every variant, so compute is held constant."""
    return min(block, _DST_TILES)


def l1_footprint_bytes(total_tiles, block, page_bytes):
    """cb_a + cb_b (double-buffered blocks) + cb_out (the whole L1-sharded output)."""
    return (2 * CB_DEPTH_BLOCKS * block + total_tiles) * page_bytes


def create_program_descriptor(a, b, output, *, variant, block, do_math=True, txn_bytes=None, grid_x=1, grid_y=1):
    if variant not in ALL_VARIANTS:
        raise ValueError(f"dual_noc_read example: variant must be one of {ALL_VARIANTS}, got {variant!r}")
    num_cores = grid_x * grid_y
    total_tiles = validate(a, b, block, txn_bytes, num_cores)
    rd_b, rd_own, wr_b, wr_own, sem = _WIRING[variant]
    a_on_writer = _A_ON_WRITER[variant]
    core_ranges = _core_grid(grid_x, grid_y)
    tiles_per_core = total_tiles // num_cores
    page_bytes = a.buffer_aligned_page_size()
    txn_bytes = page_bytes if txn_bytes is None else txn_bytes

    # Both operand CBs are identical in every variant — only WHICH RISC fills each one changes.
    def operand_cb(cb_id, tensor):
        return ttnn.CBDescriptor(
            total_size=CB_DEPTH_BLOCKS * block * page_bytes,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=tensor.dtype, page_size=page_bytes)
            ],
        )

    cbs = [
        operand_cb(CB_A, a),
        operand_cb(CB_B, b),
        # Aliased to the L1-sharded output: compute packs straight into the output buffer.
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output),
    ]

    reader_ct = [page_bytes, block, txn_bytes, rd_b, rd_own, sem, 1 - a_on_writer, SEM_GO, SEM_DONE]
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())

    writer_ct = [page_bytes, block, txn_bytes, wr_b, wr_own, sem, a_on_writer, CB_DEPTH_BLOCKS, SEM_GO, SEM_DONE]
    writer_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())

    # Contiguous page range per core, in the SAME row-major order the output height-sharding uses
    # (shard index == y * grid_x + x), so each core's reads land in its own output shard.
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    a_addr, b_addr = a.buffer_address(), b.buffer_address()
    for gy in range(grid_y):
        for gx in range(grid_x):
            start = (gy * grid_x + gx) * tiles_per_core
            reader_rt[gx][gy] = [a_addr, b_addr, tiles_per_core, start]
            writer_rt[gx][gy] = [b_addr, tiles_per_core, start, a_addr]
            compute_rt[gx][gy] = [tiles_per_core]

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC, default NoC 0
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # BRISC, default NoC 1
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=[compute_chunk(block), int(do_math)],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_DONE, core_ranges=core_ranges, initial_value=0),
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)


def dual_noc_read(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    variant: str = "two_riscv",
    block: int = 8,
    do_math: bool = True,
    txn_bytes: int = None,
    grid_x: int = 1,
    grid_y: int = 1,
) -> ttnn.Tensor:
    """C = A * B for two DRAM-interleaved bf16 tiled tensors, on one core, output L1-sharded.

    Args:
        variant: which RISCs fetch the operands.
            "one_riscv"     — NCRISC reads both A and B on NoC 0 (BASELINE; BRISC idle).
            "two_riscv"     — NCRISC reads A on NoC 0, BRISC reads B on NoC 1; each owns its CB.
            "two_riscv_sem" — same split, but the reader owns cb_b and two local semaphores order
                              the two RISCs (the form you need when the reader must post-process
                              the block).
        block: async reads issued per NoC barrier, per RISC. Must divide the total tile count.
        do_math: False drops the `mul_tiles` payload while keeping every CB handshake and the
            tile_regs/pack cycle — a perf-diagnosis ablation that exposes the pure read ceiling.
            The output is then GARBAGE; never correctness-check a do_math=False run.
        txn_bytes: bytes per NoC transaction. Default (None) = the whole tile page in one read.
            A smaller value must divide the page and moves the SAME bytes in proportionally MORE
            transactions — the knob that separates "cost per command" from "cost per byte".
        grid_x, grid_y: rectangular core grid anchored at (0,0). Default 1x1 (single core), which
            isolates one core's data-movement capacity. Larger grids add cross-core DRAM/NoC
            contention on top of the effect: the total tile count is split evenly across the grid
            in row-major order, so PER-CORE work shrinks as the grid grows unless the shape is
            scaled with it.

    Output is A*B for every variant (with do_math=True), in L1 (height-sharded on core (0,0)).
    """
    validate(a, b, block, txn_bytes, grid_x * grid_y)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        a.device(),
        create_output_memory_config(a.shape, grid_x, grid_y),
    )
    descriptor = create_program_descriptor(
        a,
        b,
        output,
        variant=variant,
        block=block,
        do_math=do_math,
        txn_bytes=txn_bytes,
        grid_x=grid_x,
        grid_y=grid_y,
    )
    return ttnn.generic_op([a, b, output], descriptor)
