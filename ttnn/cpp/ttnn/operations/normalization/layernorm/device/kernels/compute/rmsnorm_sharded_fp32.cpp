// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Narrow Blackhole FP32 RMSNorm. Retain the ordinary sharded reader's one raw
// partial per core -> eight gathered partials -> one multicast inverse protocol.
void kernel_main() {
#ifndef IDLE_CORE
    constexpr uint32_t width_tiles = get_arg(args::block_w);
    constexpr uint32_t num_partials = get_arg(args::num_blocks_first_stage);
    static_assert(get_arg(args::block_h) == 1 && width_tiles == 28 && num_partials == 8);
    static_assert(get_arg(args::num_blocks_second_stage) == 0);
    static_assert(compute_kernel_lib::get_fp32_dest_acc_enabled());

    DataflowBuffer partial(dfb::ex_partial2);
    DataflowBuffer external(dfb::ex_external2);
    DataflowBuffer inverse_local(dfb::ex2pe);
    DataflowBuffer inverse(dfb::ex_global);
    DataflowBuffer gamma(dfb::gamma);
    DataflowBuffer output(dfb::out);
    compute_kernel_hw_startup(dfb::in0, dfb::scaler, dfb::out);

    // Input is a resident shard, not a streamed queue. Accumulate corresponding
    // columns in DEST before one SFPU row sum; no squared FP32 data enters SrcA/B.
    partial.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(dfb::in0);
    copy_init(dfb::in0);
    for (uint32_t tile = 0; tile < width_tiles; ++tile) {
        if (tile == 0) {
            copy_tile(dfb::in0, tile, 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 0);
        } else {
            copy_tile(dfb::in0, tile, 1);
            mul_binary_tile_init();
            mul_binary_tile(1, 1, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(dfb::ex_partial2);
    pack_tile(0, dfb::ex_partial2);
    tile_regs_release();
    partial.push_back(1);  // The reader owns the matching pop.

#ifdef IS_ALLGATHER_WORKER
    inverse_local.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(dfb::ex_external2);
    copy_init(dfb::ex_external2);
    for (uint32_t shard = 0; shard < num_partials; ++shard) {
        external.wait_front(1);
        if (shard == 0) {
            copy_tile(dfb::ex_external2, 0, 0);
        } else {
            copy_tile(dfb::ex_external2, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        external.pop_front(1);
    }
    // Only column zero is a statistic. Combine elementwise, never spatially
    // reduce the unspecified other columns of the local SFPU reduction output.
    binop_with_scalar_tile_init();
    mul_unary_tile(0, get_arg(args::inv_width_fp32));
    add_unary_tile(0, get_arg(args::eps_fp32));
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(dfb::ex2pe);
    pack_tile(0, dfb::ex2pe);
    tile_regs_release();
    inverse_local.push_back(1);  // The reader multicasts and pops this inverse.
#endif

    inverse.wait_front(1);
    gamma.wait_front(width_tiles);
    output.reserve_back(width_tiles);
    pack_reconfig_data_format(dfb::out);
    for (uint32_t tile = 0; tile < width_tiles; ++tile) {
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::in0);
        copy_init(dfb::in0);
        copy_tile(dfb::in0, tile, 0);
        reconfig_data_format_srca(dfb::ex_global);
        copy_init(dfb::ex_global);
        copy_tile(dfb::ex_global, 0, 1);
        reconfig_data_format_srca(dfb::gamma);
        copy_init(dfb::gamma);
        copy_tile(dfb::gamma, tile, 2);
        // Reinitialize both phases: row broadcast changes the column mask.
        // Column broadcast ignores all unspecified statistic columns; row
        // broadcast consumes only the row-zero BF16 gamma staged by the writer.
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(0, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();
    }
    output.push_back(width_tiles);
    inverse.pop_front(1);
    gamma.pop_front(width_tiles);

    // Keep the unchanged writer's constant-buffer lifecycle balanced. These
    // legacy BF16/packed scalers and epsilon are deliberately not used in math.
    DataflowBuffer scaler(dfb::scaler);
    scaler.wait_front(1);
    dummy_unpack(dfb::scaler);
    scaler.pop_front(1);
#ifdef IS_ALLGATHER_WORKER
    DataflowBuffer global_scaler(dfb::scaler_global);
    global_scaler.wait_front(1);
    dummy_unpack(dfb::scaler_global);
    global_scaler.pop_front(1);
#endif
    DataflowBuffer eps(dfb::eps);
    eps.wait_front(1);
    dummy_unpack(dfb::eps);
    eps.pop_front(1);
    // Output is directly backed by the tensor; the self-loop has no dataflow reader.
    output.wait_front(width_tiles);
    dummy_unpack(dfb::out);
    output.pop_front(width_tiles);
#endif
}
