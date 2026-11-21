// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// softmax compute (dim = -2, regime column_resident): one NEO of C. It sees its own slice of every STRIDED ring,
// so it reads like a single-threaded kernel over this NEO's `rounds` columns, in blocks of `block_cols`.
//
//   per block:  column_max_block     m[c]   = max_h (x + mask)          -> cb_col_max   (Float32, row 0)
//               column_expsum_block  1/s[c] = 1 / sum_h exp(x + mask - m) -> cb_col_recip (Float32, row 0)
//               normalize_block      out    = exp(x + mask - m) * (1/s)   -> cb_output_tiles
//
// The block's columns stay resident in cb_input_tiles across all three operations (popped at block end).
// The mask (0 on valid rows, NEG_BIG on padding rows) is added to the last tile-row of every column only, so the
// padding rows enter neither the max nor the sum, and come out as 0.
//
// RAW-LLK JUSTIFICATION (op_design.md, "Helpers considered and rejected"):
//  - compute_kernel_lib::reduce<MAX/SUM, REDUCE_COL>: its indexed input policies read other NEOs' tiles under a
//    thread plan (llk_unpack_AB_reduce_api.h:122-125 ignores the ring stride), and its only thread-safe policy
//    (WaitAndPopPerTile) pops the resident column that the next two operations re-read; its exact float path is
//    static_assert-ed out on Quasar (reduce_helpers_common.hpp:50).
//  - eltwise_chain: DEST accumulation exists only for FPU binaries (chain.hpp:303-312), DestReuseBinary has no
//    row broadcast (chain.hpp:579-583), and a chain owns the dst-sync window, so it cannot share a DEST section
//    with the sfpu_reduce / row-broadcast calls each section needs.
//  - BinaryMax chain element: minmax.hpp does not compile on Quasar (library gaps).
// So each block operation is written with the raw SFPU calls, which are exact in float32 on this part
// (library gaps, "An accurate float32 path exists"). Every input-ring change re-runs copy_init (Quasar programs
// the buffer descriptor at op init) with an unpack format reconfig; every output-ring change runs pack_init.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace {

constexpr uint32_t cb_input_tiles = dfb::cb_input_tiles;
constexpr uint32_t cb_h_mask = dfb::cb_h_mask;
constexpr uint32_t cb_col_max = dfb::cb_col_max;
constexpr uint32_t cb_col_recip = dfb::cb_col_recip;
constexpr uint32_t cb_output_tiles = dfb::cb_output_tiles;

constexpr uint32_t Ht = get_arg(args::Ht);
constexpr uint32_t block_cols = get_arg(args::block_cols);
constexpr bool has_mask = get_arg(args::has_mask) != 0;
constexpr bool exp_approx = get_arg(args::exp_approx) != 0;

constexpr uint32_t dest_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;

// Point the unpacker (and the math datacopy) at another input ring before copy_tile reads it.
FORCE_INLINE void load_from(uint32_t ring) {
    reconfig_data_format_srca(ring);
    copy_init(ring);
}

// SFPU inits are issued once per block operation, at its head, not per tile. On this part that holds across the
// alternation inside a section (tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/): exp_init, the add/sub/mul
// binary init and init_reduce program nothing beyond the shared SFPU init (ckernel_sfpu_exp.h exp_init,
// ckernel_sfpu_binary.h sfpu_binary_init, ckernel_sfpu_reduce.h init_reduce); binary_max re-records its replay
// body on every call and only adds ADDR_MOD_6 (ckernel_sfpu_binary_max_min.h); every sfpu_*_bcast_row call records
// its own binop (sfpu_binary_bcast.h, "nothing persists between calls"); recip_init's Newton constant
// (vConstFloatPrgm0) is written by no other op of its section. The math datacopy (copy_init) uses ADDR_MOD_0/1 only.

// Point the packer at another output ring before pack_tile writes it.
FORCE_INLINE void pack_to(uint32_t ring) {
    pack_reconfig_data_format(ring);
    pack_init(ring);
}

// Load tile `idx` of cb_input_tiles into DEST slot `d`; the last tile-row of a column gets the mask in slot `dmask`.
FORCE_INLINE void load_x(uint32_t idx, uint32_t ht, uint32_t d, uint32_t dmask) {
    copy_tile(cb_input_tiles, idx, d);
    if constexpr (has_mask) {
        if (ht == Ht - 1) {
            add_binary_tile(d, dmask, d);
        }
    }
}

// column_max_block: DEST slots 0 = running max, 1 = next tile, 2 = mask.
template <typename InRing, typename MaxRing>
void column_max_block(InRing& in, MaxRing& col_max, uint32_t bc) {
    constexpr uint32_t d_acc = 0, d_work = 1, d_mask = 2;
    static_assert(d_mask < dest_tiles, "column_max_block needs 3 DEST slots");
    pack_to(cb_col_max);
    if constexpr (has_mask) {
        add_binary_tile_init();
    }
    binary_max_tile_init();
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    for (uint32_t k = 0; k < bc; ++k) {
        tile_regs_acquire();
        if constexpr (has_mask) {
            load_from(cb_h_mask);
            copy_tile(cb_h_mask, 0, d_mask);
        }
        load_from(cb_input_tiles);
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            const uint32_t idx = k * Ht + ht;
            in.wait_front(idx + 1);  // cumulative: compute starts on the block's first tile
            const uint32_t d = ht == 0 ? d_acc : d_work;
            load_x(idx, ht, d, d_mask);
            if (ht > 0) {
                binary_max_tile(d_acc, d_work, d_acc);
            }
        }
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_COL>(d_acc);
        tile_regs_commit();

        col_max.reserve_back(1);
        tile_regs_wait();
        pack_tile(d_acc, cb_col_max);
        tile_regs_release();
        col_max.push_back(1);
    }
}

// column_expsum_block: DEST slots 0 = running sum, 1 = next exp tile, 2 = mask, 3 = m.
template <typename MaxRing, typename RecipRing>
void column_expsum_block(MaxRing& col_max, RecipRing& col_recip, uint32_t bc) {
    constexpr uint32_t d_acc = 0, d_work = 1, d_mask = 2, d_max = 3;
    static_assert(d_max < dest_tiles, "column_expsum_block needs 4 DEST slots");
    col_max.wait_front(bc);
    pack_to(cb_col_recip);
    sfpu_bcast_row_init();
    exp_tile_init<exp_approx>();
    add_binary_tile_init();
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    recip_tile_init();
    for (uint32_t k = 0; k < bc; ++k) {
        tile_regs_acquire();
        load_from(cb_col_max);
        copy_tile(cb_col_max, k, d_max);
        if constexpr (has_mask) {
            load_from(cb_h_mask);
            copy_tile(cb_h_mask, 0, d_mask);
        }
        load_from(cb_input_tiles);
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            const uint32_t d = ht == 0 ? d_acc : d_work;
            load_x(k * Ht + ht, ht, d, d_mask);
            sfpu_sub_bcast_row(d, d_max);
            exp_tile<exp_approx>(d);
            if (ht > 0) {
                add_binary_tile(d_acc, d_work, d_acc);
            }
        }
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(d_acc);
        recip_tile(d_acc);
        tile_regs_commit();

        col_recip.reserve_back(1);
        tile_regs_wait();
        pack_tile(d_acc, cb_col_recip);
        tile_regs_release();
        col_recip.push_back(1);
    }
}

// normalize_block: DEST slots 0 = m, 1 = 1/s, 2 = mask (when masked), then the output tiles of the section.
template <typename RecipRing, typename OutRing>
void normalize_block(RecipRing& col_recip, OutRing& out, uint32_t bc) {
    constexpr uint32_t d_max = 0, d_recip = 1, d_mask = 2;
    constexpr uint32_t d_first_out = has_mask ? 3 : 2;
    static_assert(d_first_out < dest_tiles, "normalize_block needs a DEST slot for its output");
    constexpr uint32_t outs_per_section = dest_tiles - d_first_out;
    col_recip.wait_front(bc);
    pack_to(cb_output_tiles);
    sfpu_bcast_row_init();
    exp_tile_init<exp_approx>();
    if constexpr (has_mask) {
        add_binary_tile_init();
    }
    for (uint32_t k = 0; k < bc; ++k) {
        for (uint32_t ht0 = 0; ht0 < Ht; ht0 += outs_per_section) {
            const uint32_t n = (Ht - ht0) < outs_per_section ? (Ht - ht0) : outs_per_section;
            tile_regs_acquire();
            load_from(cb_col_max);
            copy_tile(cb_col_max, k, d_max);
            load_from(cb_col_recip);
            copy_tile(cb_col_recip, k, d_recip);
            if constexpr (has_mask) {
                if (ht0 + n == Ht) {
                    load_from(cb_h_mask);
                    copy_tile(cb_h_mask, 0, d_mask);
                }
            }
            load_from(cb_input_tiles);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t d = d_first_out + i;
                load_x(k * Ht + ht0 + i, ht0 + i, d, d_mask);
                sfpu_sub_bcast_row(d, d_max);
                exp_tile<exp_approx>(d);
                sfpu_mul_bcast_row(d, d_recip);
            }
            tile_regs_commit();

            out.reserve_back(n);
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(d_first_out + i, cb_output_tiles);
            }
            tile_regs_release();
            out.push_back(n);
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t rounds = get_arg(args::rounds);  // columns of this NEO (the same for every NEO of the cluster)
    const uint32_t num_blocks = (rounds + block_cols - 1) / block_cols;

    DataflowBuffer in(dfb::cb_input_tiles);
    DataflowBuffer mask(dfb::cb_h_mask);
    DataflowBuffer col_max(dfb::cb_col_max);
    DataflowBuffer col_recip(dfb::cb_col_recip);
    DataflowBuffer out(dfb::cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_col_max, cb_output_tiles);

    if constexpr (has_mask) {
        mask.wait_front(1);  // held for the whole kernel
    }
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first = block * block_cols;
        const uint32_t bc = (rounds - first) < block_cols ? (rounds - first) : block_cols;  // last block may be short
        column_max_block(in, col_max, bc);
        column_expsum_block(col_max, col_recip, bc);
        normalize_block(col_recip, out, bc);
        in.pop_front(bc * Ht);
        col_max.pop_front(bc);
        col_recip.pop_front(bc);
    }
    if constexpr (has_mask) {
        mask.pop_front(1);
    }
}
