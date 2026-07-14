// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm (dataflow).
//
// Feeds the row-parallel streaming reduce. Per assigned tile-row the input is
// streamed TWICE over W in W_BLOCK_TILES-wide chunks (pass 1 = statistics,
// pass 2 = normalize); gamma (always ROW_MAJOR) is streamed alongside pass 2.
// The reduce scaler (1/W, plus a partial tile for non-tile-aligned W) is
// prepared once up front and never popped by the reader.
//
// HELPER SUBSTITUTION (documented): the ROW_MAJOR input/gamma stick reads are
// open-coded instead of dataflow_kernel_lib::read_sticks_for_tilize. Two reasons
// the helper cannot cover:
//   1. Non-tile-aligned-W tail: the last W-block's tile-width tail is
//      uninitialized L1; the masked SUM-reduce multiplies those columns by the
//      partial scaler (== 0), and 0 * NaN = NaN corrupts the per-row Sum(x^2).
//   2. Non-tile-aligned-H: partial-height tilize (tilize(1, <32)) mis-tilizes
//      real rows on device, so the reader instead emits a FULL 32-row tile-row
//      (real rows read, H-padding rows zeroed) and the compute always tilizes a
//      full tile — matching the TILE regime. read_sticks_for_tilize pushes only
//      the real rows, which would force the buggy partial tilize.
// Zeroing both the H-padding rows and the W tail gives clean full tiles.
// (TILE-layout inputs carry ttnn's zero implicit padding for free.)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

FORCE_INLINE void zero_l1_page(uint32_t l1, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < nbytes / 4; ++i) {
        p[i] = 0;
    }
}

// Emit `rows_to_push` CB pages (ROW granularity: 1 page per stick). The first
// `num_real_rows` rows are read from DRAM (`real_bytes` wide at byte `col_off`);
// any remaining rows are H-padding and are zeroed. The tile-width tail of a
// non-tile-aligned-W block (real_bytes < padded_bytes) is also zeroed.
//
// Zeroing serves two correctness needs: (1) tilize always sees a clean full
// 32-row tile (partial-height tilize mis-tilizes real rows — see reader header),
// (2) the masked reduce never multiplies uninitialized L1 by the partial scaler
// (0 * NaN = NaN). So H-padding rows reduce to 0 (discarded) and W-padding cols
// are exactly 0.
template <uint32_t cb, typename Acc>
FORCE_INLINE void read_wblock(
    const Acc& acc,
    uint32_t rows_to_push,
    uint32_t num_real_rows,
    uint32_t base_stick,
    uint32_t real_bytes,
    uint32_t padded_bytes,
    uint32_t col_off) {
    const bool w_tail = real_bytes < padded_bytes;
    for (uint32_t r = 0; r < rows_to_push; ++r) {
        cb_reserve_back(cb, 1);
        uint32_t l1 = get_write_ptr(cb);
        const bool is_real = r < num_real_rows;
        if (!is_real || w_tail) {
            zero_l1_page(l1, padded_bytes);
        }
        if (is_real) {
            noc_async_read(acc.get_noc_addr(base_stick + r, col_off), l1, real_bytes);
            noc_async_read_barrier();
        }
        cb_push_back(cb, 1);
    }
}

// Read one gamma W-block, dispatching on the gamma LAYOUT (independent of the
// input layout). Two legs, one per build:
//   * RM gamma  -> one stick into cb_gamma_rm (compute tilizes it, row-0 valid).
//   * TILE gamma -> WBT tiles straight into cb_gamma_tiles (already row-0-valid
//                   tiles from ttnn's [1,1,1,W] padded-to-[1,1,32,W] storage;
//                   compute skips the tilize). Raw page copy preserves the
//                   on-disk dtype, so cb_gamma_tiles carries gamma_dtype.
template <
    bool GAMMA_RM,
    uint32_t GAMMA_ELT,
    uint32_t WBLOCK_COLS,
    uint32_t GAMMA_PADDED_BYTES,
    uint32_t WBT,
    typename GArgs>
FORCE_INLINE void read_gamma_block(const GArgs& gamma_args, uint32_t gamma_addr, uint32_t b, uint32_t cols) {
    if constexpr (GAMMA_RM) {
        const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
        read_wblock<cb_gamma_rm>(g_acc, 1, 1, 0, cols * GAMMA_ELT, GAMMA_PADDED_BYTES, b * WBLOCK_COLS * GAMMA_ELT);
    } else {
        const auto g_acc = TensorAccessor(gamma_args, gamma_addr, get_tile_size(cb_gamma_tiles));
        for (uint32_t wt = 0; wt < WBT; ++wt) {
            uint32_t tile_id = b * WBT + wt;
            cb_reserve_back(cb_gamma_tiles, 1);
            noc_async_read_page(tile_id, g_acc, get_write_ptr(cb_gamma_tiles));
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, 1);
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_PARTIAL_W = get_compile_time_arg_val(2);
    constexpr uint32_t partial_w = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);
    constexpr uint32_t origin_W = get_compile_time_arg_val(5);
    constexpr uint32_t origin_H = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_image = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(9);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t input_elt = get_compile_time_arg_val(11);
    constexpr uint32_t gamma_elt = get_compile_time_arg_val(12);
    constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(13);
    constexpr auto input_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);

    // Reduce scaler = 1/W (SUM reduce -> mean). Non-tile-aligned W also emits a
    // partial tile that zeros positions beyond partial_w; the compute selects
    // it for the last W-tile of the last block via last_tile_at(1).
    //
    // HELPER NOTE: the partial-pair wrapper prepare_partial_reduce_scalers() is
    // stale against the current prepare_reduce_scaler() (it forwards a 4th
    // `compute_uses_reduce_tile` TEMPLATE arg, but the pool-type-aware
    // prepare_reduce_scaler now takes 3 template args) — a fresh compile fails.
    // So the full+partial pair is emitted directly with the (working) pool-type-
    // aware prepare_reduce_scaler: tile 0 = full (valid=32), tile 1 = partial
    // (valid=partial_w). This is exactly the pair the wrapper documents.
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f, TILE_W);
    if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f, partial_w);
    }

    constexpr uint32_t wblock_cols = W_BLOCK_TILES * TILE_W;
    constexpr uint32_t in_padded_bytes = wblock_cols * input_elt;
    constexpr uint32_t gamma_padded_bytes = wblock_cols * gamma_elt;

    if constexpr (IS_ROW_MAJOR) {
        const auto in_acc = TensorAccessor(input_args, input_addr);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            uint32_t tr = start_tile_row + t;
            uint32_t image = tr / tiles_per_image;
            uint32_t local_tr = tr - image * tiles_per_image;
            uint32_t base_stick = image * origin_H + local_tr * TILE_H;
            uint32_t num_rows = origin_H - local_tr * TILE_H;
            if (num_rows > TILE_H) {
                num_rows = TILE_H;
            }

            for (uint32_t pass = 0; pass < 2; ++pass) {
                for (uint32_t b = 0; b < num_w_blocks; ++b) {
                    uint32_t cols = origin_W - b * wblock_cols;
                    if (cols > wblock_cols) {
                        cols = wblock_cols;
                    }
                    // Full 32-row tile-row: num_rows real, rest zero-padded.
                    read_wblock<cb_input_rm>(
                        in_acc,
                        TILE_H,
                        num_rows,
                        base_stick,
                        cols * input_elt,
                        in_padded_bytes,
                        b * wblock_cols * input_elt);

                    if constexpr (HAS_GAMMA) {
                        if (pass == 1) {
                            read_gamma_block<
                                GAMMA_IS_ROW_MAJOR,
                                gamma_elt,
                                wblock_cols,
                                gamma_padded_bytes,
                                W_BLOCK_TILES>(gamma_args, gamma_addr, b, cols);
                        }
                    }
                }
            }
        }
    } else {
        // TILE regime: read tiles directly. Gamma is still ROW_MAJOR.
        uint32_t tile_bytes = get_tile_size(cb_input_tiles);
        const auto in_acc = TensorAccessor(input_args, input_addr, tile_bytes);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            uint32_t tr = start_tile_row + t;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                for (uint32_t b = 0; b < num_w_blocks; ++b) {
                    // DOUBLE_BUFFER (Refinement 3): reserve the whole W-block, issue
                    // all W_BLOCK_TILES async reads, then ONE barrier — instead of
                    // read-one/barrier/push per tile. The dominant win is coarsening
                    // the reader->compute CB handshake W_BLOCK_TILES-fold (the op was
                    // sync-bound on per-tile CB ping-pong, not NoC-bandwidth-bound).
                    // cb_input_tiles is 2*W_BLOCK_TILES deep and both push and the
                    // consumer's block wait are W_BLOCK_TILES-granular, so the reserved
                    // run at the CB write pointer never wraps.
                    cb_reserve_back(cb_input_tiles, W_BLOCK_TILES);
                    uint32_t wp = get_write_ptr(cb_input_tiles);
                    uint32_t base_tile = tr * Wt + b * W_BLOCK_TILES;
                    for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                        noc_async_read_page(base_tile + wt, in_acc, wp + wt * tile_bytes);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_input_tiles, W_BLOCK_TILES);
                    if constexpr (HAS_GAMMA) {
                        if (pass == 1) {
                            uint32_t cols = origin_W - b * wblock_cols;
                            if (cols > wblock_cols) {
                                cols = wblock_cols;
                            }
                            read_gamma_block<
                                GAMMA_IS_ROW_MAJOR,
                                gamma_elt,
                                wblock_cols,
                                gamma_padded_bytes,
                                W_BLOCK_TILES>(gamma_args, gamma_addr, b, cols);
                        }
                    }
                }
            }
        }
    }
}
