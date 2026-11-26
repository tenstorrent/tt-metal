// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gate/up compute (TRISC) on a gate/up core: h = act(x @ W_gate, x @ W_up) for this core's slice of h's columns.
//
// Gate/up is two matmuls and an elementwise activation. The weights are interleaved so that one matmul does both: a
// row of a gate/up weight block holds this core's gate columns and the matching up columns side by side (gate tile p
// next to up tile p), so x @ block leaves each gate tile next to its up tile in DST, the two operands of one
// elementwise op together. The activation then runs on them in place, on the SFPU, driven by the PACK thread between
// the matmul and the pack, which leaves the MATH thread free to start the next matmul.
//
// Per sub-block: DST[row tile, gate/up tiles] = x[row tiles, K] @ W[K, gate/up tiles] over all K blocks (x and the
// weights streamed in), then act(gate, up) per pair, packed to the h CB as [h tiles x row tiles] (K-tile-major, the
// order the down cores read h in).
//
// Half-width gate/up (flat_options::half_width_gate_up): the weights interleave 16 columns (half a tile) of gate with
// the same 16 columns of up instead of whole 32-column tiles, so one output tile holds gate in faces 0 / 2 and up in
// faces 1 / 3, and the activation works face by face (glu_faces.hpp). A core's share of h is then counted in 16-column
// units instead of tiles, which splits the intermediate size evenly over more cores: 96 h tiles do not split over 64
// cores as gate/up tile pairs, but their 192 units do (3 a core, flat_options::gate_up_units), which spreads the
// gate/up work over every gate/up core.
//
// The schedule (CB 6, schedule_meta.hpp) gives each entry's sub-blocks and its gate/up weight load; a load's weight
// blocks are freed right after its last use.
//
// Named compile-time args (flat_options.hpp): x_block_k_tiles (K tiles per x block), row_tiles_per_sub_block,
// gate_up_blocks_per_expert (weight blocks per load = x blocks per sub-block), gate_up_block_tiles (tiles per weight
// block), gate_up_tile_pairs (gate/up tile pairs of this core), gate_up_weight_ring_blocks (the weight ring's
// blocks; 0: one load)
// Runtime args: 0 with 16-column units, 1 when this core's first unit is the right half of an h tile
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "flat_options.hpp"
#include "zz_probes.hpp"
#include "schedule_meta.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
// The activation: the named compile-time arg "expert_activation" (common/kernels/expert_activation.hpp, shared with
// the shared expert); its constants the expert_activation_*_bits named args (fp32 bit patterns).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#ifdef TRISC_PACK
#include "glu_faces.hpp"  // (the half-width activation, on the PACK thread's SFPU)
#endif

namespace {

constexpr expert_activation::Kind activation =
    static_cast<expert_activation::Kind>(get_named_compile_time_arg_val("expert_activation"));

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t weight_cb = tt::CBIndex::c_1;
constexpr uint32_t h_cb = tt::CBIndex::c_3;
constexpr uint32_t schedule_cb = tt::CBIndex::c_6;
constexpr uint32_t out_cb = tt::CBIndex::c_16;
constexpr uint32_t x_block_k_tiles = flat_options::x_block_k_tiles;
constexpr uint32_t row_tiles = flat_options::row_tiles_per_sub_block;
constexpr uint32_t blocks_per_load = flat_options::gate_up_blocks_per_expert;
constexpr uint32_t weight_block_tiles = flat_options::gate_up_block_tiles;
constexpr uint32_t tile_pairs = flat_options::gate_up_tile_pairs;
constexpr uint32_t ring_blocks =
    flat_options::gate_up_weight_ring_blocks ? flat_options::gate_up_weight_ring_blocks : blocks_per_load;
constexpr bool half_width = flat_options::half_width_gate_up;
constexpr bool units = flat_options::gate_up_units != 0;
// a row of a weight block (and of DST): this core's gate/up tiles
constexpr uint32_t row_width = units ? flat_options::gate_up_units : 2 * tile_pairs;
// K split: this core computes its group's tile_pairs pairs over its K part, keeps tile_pairs / k_split of them (part
// kpart's, RT 1) after folding the partner's partials in
constexpr uint32_t ks = flat_options::gate_up_k_split;
uint32_t kpart = 0;
// the h tiles this core produces per row tile
constexpr uint32_t h_row_tiles = units ? flat_options::gate_up_units_h_tiles : tile_pairs / ks;
// row tiles in DST at once: the whole sub-block, or with row passes row_pass_rows
constexpr bool row_passes = flat_options::row_pass_rows != 0;
constexpr uint32_t pass_row_tiles = row_passes ? flat_options::row_pass_rows : row_tiles;
static_assert(pass_row_tiles * row_width <= compute_kernel_lib::DEST_AUTO_LIMIT, "a pass's gate/up tiles fit DST");

// 16-column units: 1 when this core's first unit is the right half of an h tile (runtime arg 0)
uint32_t first_unit_offset = 0;

// The DST tile (within a row tile's row_width) that ends up holding this core's h tile `h_tile`: the gate tile of its
// pair, with half-width units the tile of its first unit.
FORCE_INLINE uint32_t h_tile_dest(uint32_t h_tile) { return h_tile == 0 ? 0 : 2 * h_tile - first_unit_offset; }

uint32_t matmul_rows = row_tiles;  // the row tiles the matmul is set up for

// custom_mm with a full [32, 32] in0 tile (flat_options gate_up_custom_mm): x [rows, K] tiles row-major at x_index,
// the weight block's K rows of row_width tiles, output tile (r, c) at DST r * row_width + c - matmul_block_k's
// layout, bit-identical results; the init does not depend on the row count
constexpr bool custom_mm = get_named_compile_time_arg_val("gate_up_custom_mm") != 0;

FORCE_INLINE void matmul_rows_init(uint32_t rows) {
    if (!custom_mm && rows != matmul_rows) {
        matmul_block_init(x_cb, weight_cb, false, row_width, rows, x_block_k_tiles);
        matmul_rows = rows;
    }
}

uint32_t weight_blocks_freed = 0;
uint32_t weight_blocks_landed = 0;  // stream blocks known to have landed (waited for once, not every pass)

// Waits for stream block `block` (in ring slot `ring_slot`) and returns its tile offset from the weight CB's read
// pointer, which sits at ring slot weight_blocks_freed % ring_blocks (blocks are freed by count). A block is waited for
// once: the unpack thread's waits are on the matmul's critical path.
ZZ_COUNTER(zz_x_wait);
ZZ_COUNTER(zz_w_wait);
ZZ_COUNTER(zz_h_wait);
ZZ_COUNTER(zz_acquire);
ZZ_COUNTER(zz_pack);
ZZ_COUNTER(zz_mm);     // (MATH: from the DEST acquire to the commit, a pass)
ZZ_COUNTER(zz_tiles);  // (tile matmuls)
FORCE_INLINE uint32_t weight_block(uint32_t block, uint32_t ring_slot) {
    if (block + 1 > weight_blocks_landed) {
        ZZ_WAIT(zz_w_wait, cb_wait_front(weight_cb, (block + 1 - weight_blocks_freed) * weight_block_tiles));
        weight_blocks_landed = block + 1;
    }
    return static_cast<uint32_t>(
               static_cast<int32_t>(ring_slot) - static_cast<int32_t>(weight_blocks_freed % ring_blocks)) *
           weight_block_tiles;
}

FORCE_INLINE void free_weight_block() {
    cb_pop_front(weight_cb, weight_block_tiles);
    ++weight_blocks_freed;
}

// The K steps of one x block: x row tiles [rows][x_block_k_tiles] from tile x_index on, times the weight block at
// w_index (row-major, row_width tiles a K row). One matmul_block_k: the unpacker unpacks the K steps and the math runs
// them in one LLK call - bit-identical to a matmul_block a step, without the per-step call overhead (the 2 x 2 tile
// output of a GLM-5.3 row pass is short on math per call). A full sub-block (or pass) takes the compile-time row
// count FULL, which lets the LLK unroll its per-tile loops; only a short last sub-block runs with the runtime one.
// The short form is out of line: the TRISCs fetch the K loop's code, and a second inlined copy slows the common one.
// (MEASUREMENT ONLY: gate_up_k_steps 0 - the per-step matmul_block loop, the A/B)
constexpr bool k_steps_call = get_named_compile_time_arg_val("gate_up_k_steps") != 0;
__attribute__((noinline)) void matmul_x_block_short(uint32_t x_index, uint32_t w_index, uint32_t rows) {
    if constexpr (custom_mm) {
        custom_mm_block(x_cb, weight_cb, x_index, w_index, 0, x_block_k_tiles, row_width, rows);
    } else if constexpr (k_steps_call) {
        matmul_block_k(
            x_cb, weight_cb, x_index, w_index, 0, false, row_width, rows, x_block_k_tiles, x_block_k_tiles, row_width);
    } else {
        for (uint32_t k = 0; k < x_block_k_tiles; ++k) {
            matmul_block(
                x_cb, weight_cb, x_index + k, w_index + k * row_width, 0, false, row_width, rows, x_block_k_tiles);
        }
    }
}

template <uint32_t FULL>
FORCE_INLINE void matmul_x_block(uint32_t x_index, uint32_t w_index, uint32_t rows) {
    if (rows == FULL) [[likely]] {
        if constexpr (custom_mm) {
            custom_mm_block(x_cb, weight_cb, x_index, w_index, 0, x_block_k_tiles, row_width, FULL);
        } else if constexpr (k_steps_call) {
            matmul_block_k(
                x_cb,
                weight_cb,
                x_index,
                w_index,
                0,
                false,
                row_width,
                FULL,
                x_block_k_tiles,
                x_block_k_tiles,
                row_width);
        } else {
            for (uint32_t k = 0; k < x_block_k_tiles; ++k) {
                matmul_block(
                    x_cb, weight_cb, x_index + k, w_index + k * row_width, 0, false, row_width, FULL, x_block_k_tiles);
            }
        }
    } else {
        matmul_x_block_short(x_index, w_index, rows);
    }
}

// The activation over DST tiles [0, tiles): each (gate, up) pair becomes its h tile, on the PACK thread's SFPU (after
// tile_regs_wait). Half-width: tiles t, t + 1 hold [gate | up] of an h tile's left and right 16 columns; both land in
// the h tile's DST tile.
// (MEASUREMENT ONLY: gate_up_skip_activation 1 - no activation, h is wrong; the A/B of its cost)
constexpr bool skip_activation = get_named_compile_time_arg_val("gate_up_skip_activation") != 0;
FORCE_INLINE void activate(uint32_t tiles) {
    if constexpr (skip_activation) {
        return;
    }
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    if constexpr (half_width) {
        // (one call site: two inlined copies of the SFPU body in this loop make the SFPI compiler's
        // rvtt_synth_renumber pass fail)
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint32_t unit = tile % row_width + first_unit_offset;  // this core's units, counted from an h tile
            const uint32_t h_dest = tile - tile % row_width + h_tile_dest(unit / 2);
            PACK((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_glu_faces,
                (DST_ACCUM_MODE, activation, 8),
                tile,
                tile,
                h_dest,
                VectorMode::C,
                unit & 1u)));
        }
    } else {
        for (uint32_t tile = 0; tile < tiles; tile += 2) {  // the (gate tile, up tile + 1) pair -> tile
            expert_activation::tile_pair<activation>(tile);
        }
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

// One sub-block: its `rows` row tiles holding tokens (the rest of the block is neither computed nor packed; its rows
// are never written out) over all K blocks, then the activation and the pack. The load's weight blocks start at
// stream block `first_block`, ring slot `first_slot`; `frees_weights`: the load's last use, its blocks go as used.
FORCE_INLINE void compute_sub_block(uint32_t first_block, uint32_t first_slot, bool frees_weights, uint32_t rows) {
    matmul_rows_init(rows);
    ZZ_WAIT(zz_acquire, tile_regs_acquire());
    for (uint32_t block = 0; block < blocks_per_load; ++block) {
        ZZ_WAIT(zz_x_wait, cb_wait_front(x_cb, row_tiles * x_block_k_tiles));
        matmul_x_block<row_tiles>(0, weight_block(first_block + block, first_slot + block), rows);
        cb_pop_front(x_cb, row_tiles * x_block_k_tiles);
        if (frees_weights) {  // free the block for the next load as soon as possible
            free_weight_block();
        }
    }
    tile_regs_commit();
    ZZ_WAIT(zz_h_wait, cb_reserve_back(h_cb, row_tiles * h_row_tiles));
    tile_regs_wait();
    [[maybe_unused]] const uint32_t zz_p0 = ZZ_NOW();
    activate(rows * row_width);
    pack_reconfig_data_format(out_cb, h_cb);
    for (uint32_t h_tile = 0; h_tile < h_row_tiles; ++h_tile) {  // [h tile][row tile]: K-tile-major
        for (uint32_t row = 0; row < row_tiles; ++row) {         // (rows past `rows`: whatever DST holds, never used)
            pack_tile(row < rows ? row * row_width + h_tile_dest(h_tile) : h_tile_dest(h_tile), h_cb);
        }
    }
    pack_reconfig_data_format(h_cb, out_cb);
    tile_regs_release();
    if constexpr (zz::on) {
        zz_pack += ZZ_NOW() - zz_p0;
    }
    cb_push_back(h_cb, row_tiles * h_row_tiles);
}

// K split, after a row pass's matmul (one row tile; DST: the group's 2 pairs, [gate, up] of pair 0 then pair 1, its
// partial sums over this core's K part): the partner's pair goes out (partial_out_cb, bf16; the receiver sends it),
// this core's own pair to partial_own_cb; then own + the partner's (partial_in_cb) folded in DST (fp32), the activation
// and the pack of this row's h tile. Each core sends before it waits for the partner's, so the pair cannot deadlock.
// (two halves: k_split_pack after each pass, k_split_fold one pass later - the partner's partial has a whole pass to
// arrive - and for a sub-block's last pass at its end)
constexpr uint32_t pout = flat_options::partial_out_cb, pin = flat_options::partial_in_cb,
                   pown = flat_options::partial_own_cb;
ZZ_COUNTER(zz_ks_pack);  // (PACK: the partials' pack, incl. its DEST wait)
ZZ_COUNTER(zz_ks_pin);   // (UNPACK: waiting for the partner's partial)
ZZ_COUNTER(zz_ks_fold);  // (MATH: the fold incl. the inits around it)
ZZ_COUNTER(zz_ks_h);     // (PACK: activation + h pack, incl. its DEST wait)
FORCE_INLINE void k_split_pack() {
    const uint32_t mine = 2 * kpart, theirs = 2 * (1 - kpart);
    [[maybe_unused]] const uint32_t zz0 = ZZ_NOW();
    WAYPOINT("KPDW");
    tile_regs_wait();
    WAYPOINT("KPOW");
    cb_reserve_back(pout, 2);
    WAYPOINT("KPMW");
    cb_reserve_back(pown, 2);
    WAYPOINT("KPGO");
    pack_reconfig_data_format(out_cb, pout);
    pack_tile<true>(theirs, pout, 0);
    pack_tile<true>(theirs + 1, pout, 1);
    pack_tile<true>(mine, pown, 0);
    pack_tile<true>(mine + 1, pown, 1);
    tile_regs_release();
    cb_push_back(pout, 2);
    cb_push_back(pown, 2);
    pack_reconfig_data_format(pout, out_cb);
    if constexpr (zz::on) {
        zz_ks_pack += ZZ_NOW() - zz0;
    }
}

// rows [row, row + n) (n 1 or 2: a DST half holds 2 rows' (gate, up)): DST 2 r / 2 r + 1 = own + the partner's.
// The unpacker's tile indices do not wrap at a CB's end: 2 rows only when they sit in consecutive slots of the
// partial rings (both advance a row a fold row, in step), never from the last slot (ks_slot_fit).
uint32_t ks_rows_folded = 0;  // (rows folded over the call: the partial rings' read slot)
FORCE_INLINE uint32_t ks_slot_fit(uint32_t want) {
    return want > 1 && ks_rows_folded % flat_options::partial_slots == flat_options::partial_slots - 1 ? 1 : want;
}
FORCE_INLINE void k_split_fold(uint32_t row, uint32_t n) {
    [[maybe_unused]] const uint32_t zz0 = ZZ_NOW();
    reconfig_data_format(pown, pin);
    add_init(pown, pin);
    WAYPOINT("KFMW");
    cb_wait_front(pown, 2 * n);
    WAYPOINT("KFIW");
    ZZ_WAIT(zz_ks_pin, cb_wait_front(pin, 2 * n));
    WAYPOINT("KFAW");
    tile_regs_acquire();
    WAYPOINT("KFGO");
    for (uint32_t t = 0; t < 2 * n; ++t) {
        add_tiles(pown, pin, t, t, t);
    }
    tile_regs_commit();
    cb_pop_front(pown, 2 * n);
    cb_pop_front(pin, 2 * n);
    ks_rows_folded += n;
    // back to the matmul (its sources, its init) for the next pass
    reconfig_data_format<SrcOrder::Reverse>(x_cb, weight_cb);
    // (no SrcB clear: it waits for both SrcB banks every pass)
    custom_mm_block_init_short<false, CustomMmDestLayout::Default, false>(x_cb, weight_cb, out_cb, row_width);
    if constexpr (zz::on) {
        zz_ks_fold += ZZ_NOW() - zz0;
    }
    [[maybe_unused]] const uint32_t zz1 = ZZ_NOW();
    WAYPOINT("KHDW");
    tile_regs_wait();
    WAYPOINT("KHGO");
    activate(2 * n);
    pack_reconfig_data_format(out_cb, h_cb);
    for (uint32_t r = 0; r < n; ++r) {
        pack_tile<true>(2 * r, h_cb, row + r);
    }
    pack_reconfig_data_format(h_cb, out_cb);
    tile_regs_release();
    if constexpr (zz::on) {
        zz_ks_h += ZZ_NOW() - zz1;
    }
}

// One sub-block in row passes, for when DST cannot hold all of a sub-block's gate/up tiles: each pass computes
// pass_row_tiles row tiles over all K blocks, as compute_sub_block does. The passes read the same weight blocks, which
// stay in their ring until the last pass frees them. x arrives in x sub-blocks of flat_options::x_row_tiles rows (a
// whole number of passes' rows: the x path's granularity, chosen apart from the passes' - the relays' per-block costs
// want few, large blocks; the ring's slack wants x freed early): a group of PPX passes shares an x sub-block's blocks,
// the group's first pass waits for them block by block (they arrive in order) and its last frees them, the passes
// between re-read them from their ring slots (stepped, not recomputed: the unpack thread's work per block is on the
// matmul's critical path). PPX 1: each pass waits for and frees its own blocks.
uint32_t x_blocks_freed = 0;  // (the x ring's read slot: x_blocks_freed % x_ring_blocks)
[[maybe_unused]] uint32_t zz_t_start = 0;  // (startup timeline: the kernel's start)
FORCE_INLINE void compute_sub_block_in_row_passes(
    uint32_t first_block, uint32_t first_slot, bool frees_weights, uint32_t rows) {
    constexpr uint32_t xr = flat_options::x_row_tiles, ppx = xr / pass_row_tiles;
    static_assert(xr % pass_row_tiles == 0 && row_tiles % xr == 0, "row passes: an x block holds whole passes' rows");
    constexpr uint32_t x_block_tiles = xr * x_block_k_tiles;
    constexpr uint32_t x_ring_blocks = flat_options::x_ring_blocks;
    const uint32_t passes = (rows + pass_row_tiles - 1) / pass_row_tiles;
    uint32_t front = 0;                       // (the ring slot of the group's first block)
    [[maybe_unused]] uint32_t ks_folded = 0;  // (K split: the sub-block's rows folded so far)
    ZZ_WAIT(zz_h_wait, cb_reserve_back(h_cb, row_tiles * h_row_tiles));
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t first_row = pass * pass_row_tiles;
        const uint32_t pass_rows = rows - first_row < pass_row_tiles ? rows - first_row : pass_row_tiles;
        const bool last_pass = pass + 1 == passes;
        const uint32_t in_group = pass % ppx;
        const bool first_g = in_group == 0, last_g = in_group + 1 == ppx || last_pass;
        if (first_g) {
            front = x_blocks_freed % x_ring_blocks;
        }
        int32_t slot_offset = 0;  // (block `block`'s slot - front, wrapped once past the ring's end)
        matmul_rows_init(pass_rows);
        ZZ_WAIT(zz_acquire, tile_regs_acquire());
        [[maybe_unused]] const uint32_t zz_m0 = ZZ_NOW();
        // The load's weight blocks sit at ring slots first_slot .. in order, and a load is freed whole (in the last
        // pass of its last sub-block, block by block as used): block b's offset from the read pointer is w_base +
        // b blocks, or w_base for every block while the pass frees them.
        const int32_t w_base =
            (static_cast<int32_t>(first_slot) - static_cast<int32_t>(weight_blocks_freed % ring_blocks)) *
            static_cast<int32_t>(weight_block_tiles);
        const bool freeing = last_pass && frees_weights;
        for (uint32_t block = 0; block < blocks_per_load; ++block) {
            if (first_g) {  // (block by block: grouped waits / frees cost the x chain its slack)
                WAYPOINT("KXWT");
                ZZ_WAIT(zz_x_wait, cb_wait_front(x_cb, last_g ? x_block_tiles : (block + 1) * x_block_tiles));
                WAYPOINT("KXGO");
            }
            const uint32_t x_offset = last_g ? 0 : static_cast<uint32_t>(slot_offset) * x_block_tiles;
            const uint32_t wb = first_block + block;
            if (wb + 1 > weight_blocks_landed) {
                WAYPOINT("KWWT");
                ZZ_WAIT(zz_w_wait, cb_wait_front(weight_cb, (wb + 1 - weight_blocks_freed) * weight_block_tiles));
                WAYPOINT("KWGO");
                if constexpr (zz::on) {  // (startup timeline: the first weight block / the first load's last block)
                    if (wb == 0) {
                        UNPACK((ZZ_EMIT("ZZ-T-W0", ZZ_NOW() - zz_t_start)));
                    }
                    if (wb + 1 == blocks_per_load) {
                        UNPACK((ZZ_EMIT("ZZ-T-WL", ZZ_NOW() - zz_t_start)));
                    }
                }
                weight_blocks_landed = wb + 1;
            }
            const uint32_t w_offset =
                static_cast<uint32_t>(freeing ? w_base : w_base + static_cast<int32_t>(block * weight_block_tiles));
            matmul_x_block<pass_row_tiles>(x_offset + in_group * pass_row_tiles * x_block_k_tiles, w_offset, pass_rows);
            if (last_g) {
                cb_pop_front(x_cb, x_block_tiles);
                ++x_blocks_freed;
            }
            if (freeing) {
                free_weight_block();
            }
            slot_offset = front + slot_offset + 1 == static_cast<int32_t>(x_ring_blocks)
                              ? slot_offset + 1 - static_cast<int32_t>(x_ring_blocks)
                              : slot_offset + 1;
        }
        tile_regs_commit();
        if constexpr (zz::on) {
            zz_mm += ZZ_NOW() - zz_m0;
            zz_tiles += blocks_per_load * x_block_k_tiles * pass_rows * row_width;
        }
        [[maybe_unused]] const uint32_t zz_p0 = ZZ_NOW();
        if constexpr (ks > 1) {
            // (rows fold in pairs, a pair once the pass after it is packed - its partner partials have had a pass to
            // arrive - the rest at the sub-block's end)
            k_split_pack();
            while (first_row >= ks_folded + 2) {
                const uint32_t n = ks_slot_fit(2);
                k_split_fold(ks_folded, n);
                ks_folded += n;
            }
            if (last_pass) {
                while (ks_folded <= first_row) {
                    const uint32_t n = ks_slot_fit(first_row + 1 - ks_folded < 2 ? 1 : 2);
                    k_split_fold(ks_folded, n);
                    ks_folded += n;
                }
            }
        } else {
            tile_regs_wait();
            activate(pass_rows * row_width);
            pack_reconfig_data_format(out_cb, h_cb);
            for (uint32_t h_tile = 0; h_tile < h_row_tiles; ++h_tile) {  // this pass's rows of each h tile
                for (uint32_t row = 0; row < pass_rows; ++row) {
                    pack_tile<true>(row * row_width + h_tile_dest(h_tile), h_cb, h_tile * row_tiles + first_row + row);
                }
            }
            pack_reconfig_data_format(h_cb, out_cb);
            tile_regs_release();
        }
        if constexpr (zz::on) {
            zz_pack += ZZ_NOW() - zz_p0;
        }
    }
    cb_push_back(h_cb, row_tiles * h_row_tiles);  // (rows past `rows`: never written, never used)
}

}  // namespace

void kernel_main() {
    if constexpr (units) {
        first_unit_offset = get_arg_val<uint32_t>(0);
    }
    if constexpr (ks > 1) {
        static_assert(
            ks == 1 || (custom_mm && row_passes && !units && tile_pairs == 2), "the K split: 2 whole pairs, custom_mm");
        kpart = get_arg_val<uint32_t>(1);
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, weight_cb, out_cb);
    if constexpr (custom_mm) {
        custom_mm_block_init(x_cb, weight_cb, out_cb, row_width);
    } else {
        matmul_block_init(x_cb, weight_cb, false, row_width, row_tiles, x_block_k_tiles);
    }
    if constexpr (half_width) {  // (every face-wise activation uses SiTU's sigmoid: its init)
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
    } else {
        expert_activation::init<activation>();
    }
    zz_t_start = ZZ_NOW();
    // The schedule page (schedule_meta.hpp, from gate_up_core_reader.cpp)
    cb_wait_front(schedule_cb, 1);
    [[maybe_unused]] const uint32_t zz_start = ZZ_NOW();
    const uint32_t entries = read_tile_value(schedule_cb, 0, schedule_page::kEntries);
    for (uint32_t entry = 0; entry < entries; ++entry) {
        const uint32_t sub_blocks = read_tile_value(schedule_cb, 0, schedule_page::kSubBlocks + entry);
        const uint32_t load_word = read_tile_value(schedule_cb, 0, schedule_page::kLoadWord + entry);
        const uint32_t last_row_tiles = read_tile_value(schedule_cb, 0, schedule_page::kLastRowTiles + entry);
        const uint32_t first_block = schedule_page::load_word_load(load_word) * blocks_per_load;
        const uint32_t first_slot = schedule_page::load_word_ring_region(load_word) * blocks_per_load;
        const bool last_use = schedule_page::load_word_last_use(load_word);
        [[maybe_unused]] const uint32_t zz_x0 = zz_x_wait;
        [[maybe_unused]] const uint32_t zz_w0 = zz_w_wait;
        [[maybe_unused]] const uint32_t zz_t0 = ZZ_NOW();
        for (uint32_t sub_block = 0; sub_block < sub_blocks; ++sub_block) {
            const bool last_sub_block = sub_block + 1 == sub_blocks;
            // row tiles holding tokens (none: one row tile of garbage, the down cores never write it out)
            const uint32_t token_rows = last_sub_block ? last_row_tiles : row_tiles;
            const uint32_t rows = token_rows ? (token_rows < row_tiles ? token_rows : row_tiles) : 1;
            const bool frees_weights = last_use && last_sub_block;
            if constexpr (row_passes) {
                compute_sub_block_in_row_passes(first_block, first_slot, frees_weights, rows);
            } else {
                compute_sub_block(first_block, first_slot, frees_weights, rows);
            }
        }
        // (per entry: entry << 24 | x wait, the weight wait, the entry's span, in 0.1 us)
        if (entry < 24) {  // (the profiler buffer)
            UNPACK((ZZ_EMIT("ZZ-GU-EX", (entry << 24) | ((zz_x_wait - zz_x0) / 135))));
            UNPACK((ZZ_EMIT("ZZ-GU-EW", (entry << 24) | ((zz_w_wait - zz_w0) / 135))));
            UNPACK((ZZ_EMIT("ZZ-GU-ET", (entry << 24) | ((ZZ_NOW() - zz_t0) / 135))));
        }
    }
    cb_pop_front(schedule_cb, 1);
    UNPACK((ZZ_EMIT("ZZ-GU-XWAIT", zz_x_wait)));
    UNPACK((ZZ_EMIT("ZZ-GU-WWAIT", zz_w_wait)));
    UNPACK((ZZ_EMIT("ZZ-GU-SPAN", ZZ_NOW() - zz_start)));
    PACK((ZZ_EMIT("ZZ-GU-HWAIT", zz_h_wait)));
    PACK((ZZ_EMIT("ZZ-GU-PACKTIME", zz_pack)));
    MATH((ZZ_EMIT("ZZ-GU-ACQUIRE", zz_acquire)));
    MATH((ZZ_EMIT("ZZ-GU-MM", zz_mm)));
    if constexpr (ks > 1) {
        PACK((ZZ_EMIT("ZZ-KS-PACK", zz_ks_pack)));
        PACK((ZZ_EMIT("ZZ-KS-H", zz_ks_h)));
        UNPACK((ZZ_EMIT("ZZ-KS-PIN", zz_ks_pin)));
        MATH((ZZ_EMIT("ZZ-KS-FOLD", zz_ks_fold)));
    }
    MATH((ZZ_EMIT("ZZ-GU-MMTILES", zz_tiles * 1350)));  // (x 1350: the report divides by it)
}
