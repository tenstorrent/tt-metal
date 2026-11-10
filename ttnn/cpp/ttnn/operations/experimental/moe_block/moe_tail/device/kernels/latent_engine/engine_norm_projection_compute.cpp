// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// engine_norm_projection_compute: a latent engine core's compute. Per row block: my own rows' shares tilized into
// their landing slots; per step my share's squares, all of my K half in DEST (my finished columns kept, the partner's
// sent), the half's sum of squares, and the previous step's
//   output = (mine + the partner's) x rsqrt(sum of squares / latent_dim + eps) + shared expert
// (the latent rmsnorm folded: its weight is in W', the 1 / rms applied to the fp32 sums).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/matmul_fidelity.hpp"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/tilize.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t k_half_tiles = get_compile_time_arg_val(0), engine_columns = get_compile_time_arg_val(1);
constexpr uint32_t max_share_tiles = get_compile_time_arg_val(2);
// (fp32 bit patterns: 1 / latent_dim and the rmsnorm's eps)
constexpr uint32_t inverse_latent_dim_bits = get_compile_time_arg_val(3), norm_eps_bits = get_compile_time_arg_val(4);
constexpr uint32_t exchange_tiles = get_compile_time_arg_val(5);
// row blocks (the kernel's own loop) and the landing slots a block (the CB_LANDING pushes a block: own steps, then
// filler)
constexpr uint32_t row_blocks = get_compile_time_arg_val(6), max_steps = get_compile_time_arg_val(7);
// (moe_tail_program_factory's engine_cb)
constexpr uint32_t CB_UP_WEIGHTS = 0, CB_RING = 1, CB_ROW_SQUARES = 2, CB_MY_SQUARES = 3, CB_ONES = 4, CB_FINISHED = 5,
                   CB_EXCHANGE_SEND = 6, CB_SUM_OF_SQUARES = 7;
constexpr uint32_t CB_EXCHANGE_RECEIVE = 8, CB_SHARED_EXPERT = 9, CB_OUTPUT = 10, CB_LANDING = 11, CB_OWN_ROWS = 12;
// DEST: 8 fp32 tiles (fp32 accumulation, full sync - the host's ComputeConfigDescriptor). The squares use tile 0 for
// the running sum and 1 .. 7 for a batch of squared tiles.
constexpr uint32_t dest_tiles = 8, squares_batch_tiles = dest_tiles - 1;
// the output step's DEST: 0 the scale, 1 .. my finished columns (at most exchange_tiles - 1 = 3), 4 the partner's sum
// of squares then the shared expert's tiles, 5 .. the partner's partials
constexpr uint32_t dest_scale = 0, dest_mine = 1, dest_partner_sum = 4, dest_shared_expert = 4, dest_partner = 5;

// (the squares' row sums at HiFi4 whatever the kernel's fidelity: HiFi2 truncates them ~0.8% low; the up projection at
// the kernel's configured fidelity - common/kernels/matmul_fidelity.hpp)

void kernel_main() {
    // args: my column's tiles, my finished columns' offset in them and count, my share's K range, steps, own steps,
    // a 0, then the W' K range bounds after the first (the writer pushes W' range by range)
    const uint32_t column_tiles = get_arg_val<uint32_t>(0), finished_offset = get_arg_val<uint32_t>(1);
    const uint32_t finished_count = get_arg_val<uint32_t>(2);
    const uint32_t share_k_first = get_arg_val<uint32_t>(3), share_k_end = get_arg_val<uint32_t>(4);
    const uint32_t steps = get_arg_val<uint32_t>(5), own_steps = get_arg_val<uint32_t>(6);
    constexpr uint32_t w_k_bounds_arg = 8;
    uint32_t w_ranges_waited = 0, w_k_next = 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_RING, CB_UP_WEIGHTS, CB_FINISHED);
    for (uint32_t block = 0; block < row_blocks; ++block) {
        // my shares of my own rows tilized into their landing slots (no separate tilize: the combine's rows read once);
        // the reader pushes whole max_share_tiles blocks. (Not compute_kernel_lib::tilize: the share's width is a
        // runtime value smaller than the block it pushes / pops, which the helper's compile-time block width cannot
        // express.)
        for (uint32_t own_step = 0; own_step < own_steps; ++own_step) {
            DeviceZoneScopedN("TL-TILIZE");
            const uint32_t share_tiles = share_k_end - share_k_first;
            cb_wait_front(CB_OWN_ROWS, max_share_tiles);
            cb_reserve_back(CB_LANDING, max_share_tiles);
            reconfig_data_format_srca(CB_OWN_ROWS);
            pack_reconfig_data_format(CB_LANDING);
            tilize_init(CB_OWN_ROWS, share_tiles, CB_LANDING);
            tilize_block(CB_OWN_ROWS, share_tiles, CB_LANDING);
            tilize_uninit(CB_OWN_ROWS, CB_LANDING);
            cb_push_back(CB_LANDING, max_share_tiles);
            cb_pop_front(CB_OWN_ROWS, max_share_tiles);
        }
        // (the other steps' slots - landed by the gather: filler, so the CB's pointer is back at its base once the
        // writer pops the block's whole CB)
        if (own_steps < max_steps) {
            cb_reserve_back(CB_LANDING, (max_steps - own_steps) * max_share_tiles);
            cb_push_back(CB_LANDING, (max_steps - own_steps) * max_share_tiles);
        }
        if (block == 0) {
            cb_wait_front(CB_ONES, 1);  // (once a call: never popped)
        }
        // output = (mine + the partner's) x scale + shared expert for the oldest step not yet out - run behind the
        // next step's squares, matmul and sums (the partner's half and its sum of squares land meanwhile: the
        // exchange's round trip off the step's critical path)
        auto output_step = [&]() {
            DeviceZoneScopedN("TL-OUT");
            cb_wait_front(CB_EXCHANGE_RECEIVE, exchange_tiles);
            cb_wait_front(CB_SUM_OF_SQUARES, 1);
            cb_wait_front(CB_FINISHED, finished_count);
            cb_wait_front(CB_SHARED_EXPERT, finished_count);
            cb_reserve_back(CB_OUTPUT, finished_count);
            tile_regs_acquire();
            reconfig_data_format_srca(CB_SUM_OF_SQUARES);
            copy_init(CB_SUM_OF_SQUARES);
            copy_tile(CB_SUM_OF_SQUARES, 0, dest_scale);
            reconfig_data_format_srca(CB_EXCHANGE_RECEIVE);
            copy_init(CB_EXCHANGE_RECEIVE);
            copy_tile(CB_EXCHANGE_RECEIVE, exchange_tiles - 1, dest_partner_sum);
            for (uint32_t j = 0; j < finished_count; ++j) {
                copy_tile(CB_EXCHANGE_RECEIVE, j, dest_partner + j);
            }
            // scale = rsqrt((my half's + the partner's sum of squares) / latent_dim + eps)
            add_binary_tile_init();
            add_binary_tile(dest_scale, dest_partner_sum, dest_scale);
            binop_with_scalar_tile_init();
            mul_unary_tile(dest_scale, inverse_latent_dim_bits);
            add_unary_tile(dest_scale, norm_eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dest_scale);
            reconfig_data_format_srca(CB_FINISHED);
            copy_init(CB_FINISHED);
            for (uint32_t j = 0; j < finished_count; ++j) {
                copy_tile(CB_FINISHED, j, dest_mine + j);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < finished_count; ++j) {
                add_binary_tile(dest_mine + j, dest_partner + j, dest_mine + j);
            }
            mul_binary_tile_init();
            for (uint32_t j = 0; j < finished_count; ++j) {
                mul_binary_tile(dest_mine + j, dest_scale, dest_mine + j);
            }
            reconfig_data_format_srca(CB_SHARED_EXPERT);
            copy_init(CB_SHARED_EXPERT);
            for (uint32_t j = 0; j < finished_count; ++j) {
                copy_tile(CB_SHARED_EXPERT, j, dest_shared_expert + j);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < finished_count; ++j) {
                add_binary_tile(dest_mine + j, dest_shared_expert + j, dest_mine + j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUTPUT);
            for (uint32_t j = 0; j < finished_count; ++j) {
                pack_tile(dest_mine + j, CB_OUTPUT);
            }
            tile_regs_release();
            cb_push_back(CB_OUTPUT, finished_count);
            cb_pop_front(CB_EXCHANGE_RECEIVE, exchange_tiles);
            cb_pop_front(CB_SUM_OF_SQUARES, 1);
            cb_pop_front(CB_FINISHED, finished_count);
            cb_pop_front(CB_SHARED_EXPERT, finished_count);
        };
        for (uint32_t step = 0; step < steps; ++step) {
            cb_wait_front(CB_RING, k_half_tiles);
            {  // my share's squares accumulated elementwise in DEST (fp32, SFPU): sum_k z_k * z_k
                DeviceZoneScopedN("TL-SQ");
                cb_reserve_back(CB_MY_SQUARES, 1);
                reconfig_data_format_srca(CB_RING);
                pack_reconfig_data_format(CB_MY_SQUARES);
                tile_regs_acquire();
                for (uint32_t batch_first_k = share_k_first; batch_first_k < share_k_end;
                     batch_first_k += squares_batch_tiles) {
                    const uint32_t batch_tiles = share_k_end - batch_first_k < squares_batch_tiles
                                                     ? share_k_end - batch_first_k
                                                     : squares_batch_tiles;
                    copy_init(CB_RING);
                    for (uint32_t i = 0; i < batch_tiles; ++i) {
                        copy_tile(CB_RING, batch_first_k + i, 1 + i);
                    }
                    mul_binary_tile_init();
                    for (uint32_t i = 0; i < batch_tiles; ++i) {
                        mul_binary_tile(1 + i, 1 + i, 1 + i);
                    }
                    if (batch_first_k == share_k_first) {
                        copy_dest_values<DataFormat::Float32>(1, 0);
                    }
                    add_binary_tile_init();
                    for (uint32_t i = batch_first_k == share_k_first ? 1 : 0; i < batch_tiles; ++i) {
                        add_binary_tile(0, 1 + i, 0);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_MY_SQUARES);
                tile_regs_release();
                cb_push_back(CB_MY_SQUARES, 1);
            }
            {  // all of my half's K in DEST: my finished columns to CB_FINISHED, the partner's to CB_EXCHANGE_SEND
                DeviceZoneScopedN("TL-MM");
                reconfig_data_format(CB_UP_WEIGHTS, CB_RING);
                matmul_block_init_at<configured_fidelity>(CB_RING, CB_UP_WEIGHTS, 0, column_tiles, 1, k_half_tiles);
                cb_reserve_back(CB_FINISHED, finished_count);
                cb_reserve_back(CB_EXCHANGE_SEND, exchange_tiles);
                tile_regs_acquire();
                for (uint32_t k_tile = 0; k_tile < k_half_tiles; ++k_tile) {
                    // (block 0's step 0: W' K range by range, as each lands)
                    if (block == 0 && step == 0 && k_tile == w_k_next) {
                        w_k_next = get_arg_val<uint32_t>(w_k_bounds_arg + w_ranges_waited++);
                        cb_wait_front(CB_UP_WEIGHTS, w_k_next * column_tiles);
                    }
                    matmul_block_at<configured_fidelity>(
                        CB_RING, CB_UP_WEIGHTS, k_tile, k_tile * column_tiles, 0, false, column_tiles, 1, k_half_tiles);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_FINISHED);
                for (uint32_t j = 0; j < finished_count; ++j) {
                    pack_tile<true>(finished_offset + j, CB_FINISHED, j);
                }
                for (uint32_t column_tile = 0, exchange_tile = 0; column_tile < column_tiles; ++column_tile) {
                    if (column_tile < finished_offset || column_tile >= finished_offset + finished_count) {
                        pack_tile<true>(column_tile, CB_EXCHANGE_SEND, exchange_tile++);
                    }
                }
                tile_regs_release();
                cb_push_back(CB_FINISHED, finished_count);
                cb_pop_front(CB_RING, k_half_tiles);
            }
            // my half's sum of squares (the row's squares tiles @ ones, HiFi4, accumulated in DEST): to the partner and
            // mine
            {
                DeviceZoneScopedN("TL-S");
                cb_wait_front(CB_ROW_SQUARES, engine_columns);
                cb_reserve_back(CB_SUM_OF_SQUARES, 1);
                reconfig_data_format(CB_ONES, CB_ROW_SQUARES);
                matmul_block_init_at<MathFidelity::HiFi4>(CB_ROW_SQUARES, CB_ONES, 0, 1, 1, 1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < engine_columns; ++i) {
                    matmul_block_at<MathFidelity::HiFi4>(CB_ROW_SQUARES, CB_ONES, i, 0, 0, 0, 1, 1, 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_EXCHANGE_SEND);
                pack_tile<true>(0, CB_EXCHANGE_SEND, exchange_tiles - 1);
                pack_tile(0, CB_SUM_OF_SQUARES);
                tile_regs_release();
                cb_push_back(CB_EXCHANGE_SEND, exchange_tiles);
                cb_push_back(CB_SUM_OF_SQUARES, 1);
                cb_pop_front(CB_ROW_SQUARES, engine_columns);
            }
            if (step > 0) {
                output_step();  // (step - 1's)
            }
        }
        if (steps) {
            output_step();
        }
    }
}
