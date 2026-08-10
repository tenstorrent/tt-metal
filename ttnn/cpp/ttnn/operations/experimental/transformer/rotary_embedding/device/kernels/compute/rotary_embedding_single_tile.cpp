// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for ttnn.experimental.rotary_embedding when head_dim == TILE_WIDTH
// (single tile along the W dimension). Uses an in-L1 transformation matrix to do
// HF-style rotate_half via matmul_tiles, since the inter-tile half-swap used for
// Wt >= 2 cannot express a sub-tile rotation.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

template <uint32_t in0_dfb_id, uint32_t out_dfb_id>
ALWI void UNTILIZE_ONE_TILE() {
    compute_kernel_lib::untilize<
        1,
        in0_dfb_id,
        out_dfb_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t in0_dfb_id, uint32_t out_dfb_id>
ALWI void TILIZE_ONE_TILE(uint32_t sync_dfb_id) {
    DataflowBuffer dfb_sync(sync_dfb_id);
    dfb_sync.wait_front(1);
    compute_kernel_lib::tilize<
        1,
        in0_dfb_id,
        out_dfb_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    dfb_sync.pop_front(1);
}

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t num_rows = get_compile_time_arg_val(8);

    DataflowBuffer dfb_in(in_dfb_id);
    DataflowBuffer dfb_trans_mat(trans_mat_dfb_id);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb_id);

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_dfb_id = get_compile_time_arg_val(9);
    constexpr uint32_t untilized_cos_sync_dfb_id = get_compile_time_arg_val(10);
    constexpr uint32_t untilized_sin_dfb_id = get_compile_time_arg_val(11);
    constexpr uint32_t untilized_sin_sync_dfb_id = get_compile_time_arg_val(12);
    constexpr uint32_t retilized_cos_dfb_id = get_compile_time_arg_val(13);
    constexpr uint32_t retilized_sin_dfb_id = get_compile_time_arg_val(14);

    compute_kernel_hw_startup(sin_dfb_id, sin_dfb_id, untilized_sin_dfb_id);
    UNTILIZE_ONE_TILE<sin_dfb_id, untilized_sin_dfb_id>();
    UNTILIZE_ONE_TILE<cos_dfb_id, untilized_cos_dfb_id>();
    reconfig_data_format_srca(cos_dfb_id, untilized_sin_dfb_id);
    pack_reconfig_data_format(untilized_cos_dfb_id, retilized_sin_dfb_id);
    TILIZE_ONE_TILE<untilized_sin_dfb_id, retilized_sin_dfb_id>(untilized_sin_sync_dfb_id);
    TILIZE_ONE_TILE<untilized_cos_dfb_id, retilized_cos_dfb_id>(untilized_cos_sync_dfb_id);
    constexpr uint32_t updated_cos_dfb_id = retilized_cos_dfb_id;
    constexpr uint32_t updated_sin_dfb_id = retilized_sin_dfb_id;
    constexpr auto trig_bcast = BroadcastDim::Row;
    constexpr auto trig_pop = PopPolicy::None;
#else
    constexpr uint32_t updated_cos_dfb_id = cos_dfb_id;
    constexpr uint32_t updated_sin_dfb_id = sin_dfb_id;
    constexpr auto trig_bcast = BroadcastDim::None;
    constexpr auto trig_pop = PopPolicy::PerTile;
#endif

    dfb_trans_mat.wait_front(onetile);
    compute_kernel_hw_startup(rotated_in_interm_dfb_id, updated_sin_dfb_id, sin_interm_dfb_id);
    for (uint32_t i = 0; i < num_rows; ++i) {
        // rotated = in @ trans_mat  (HF rotate_half on a single 32x32 tile)
        dfb_in.wait_front(onetile);
        reconfig_data_format(in_dfb_id, trans_mat_dfb_id);
        pack_reconfig_data_format(rotated_in_interm_dfb_id);
        matmul_init(in_dfb_id, trans_mat_dfb_id);

        tile_regs_acquire();
        matmul_tiles(in_dfb_id, trans_mat_dfb_id, 0, 0, 0);
        tile_regs_commit();

        dfb_rotated_in_interm.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, rotated_in_interm_dfb_id);
        tile_regs_release();

        dfb_rotated_in_interm.push_back(onetile);

        // sin_interim = rotated * sin  (chain waits+pops rotated_in_interm_dfb_id; sin held/streamed per mode)
        mul<input(rotated_in_interm_dfb_id),
            input(updated_sin_dfb_id, WaitPolicy::PerTile, trig_pop),
            output(sin_interm_dfb_id),
            trig_bcast>(EltwiseShape::tiles(onetile));

        // cos_interim = in * cos
        mul<input(in_dfb_id, WaitPolicy::None, PopPolicy::PerTile),
            input(updated_cos_dfb_id, WaitPolicy::PerTile, trig_pop),
            output(cos_interm_dfb_id),
            trig_bcast>(EltwiseShape::tiles(onetile));

        // out = cos_interim + sin_interim
        add<input(cos_interm_dfb_id), input(sin_interm_dfb_id), output(out_dfb_id)>(EltwiseShape::tiles(onetile));
    }
}
