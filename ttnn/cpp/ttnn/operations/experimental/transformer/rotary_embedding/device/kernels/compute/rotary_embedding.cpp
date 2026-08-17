// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

// out = input*cos + rotate_half(input)*sin. Decode mode reuses row-broadcast cos/sin
// instead of consuming one trig tile per output tile.
#ifdef DECODE_MODE
inline constexpr bool kDecodeMode = true;
#else
inline constexpr bool kDecodeMode = false;
#endif

template <uint32_t in0_dfb_id, uint32_t in1_dfb_id, uint32_t out_dfb_id>
ALWI void mul_tiles_chain(uint32_t in1_idx) {
    using namespace compute_kernel_lib;
    if constexpr (kDecodeMode) {
        eltwise_chain(
            IterationShape::one_tile(),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(in0_dfb_id, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
                input(
                    in1_dfb_id,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    OperandKind::Scalar,
                    DataFormatReconfig::Disabled,
                    compute_kernel_lib::TileOffset::Set)>{0u, in1_idx},
            PackTile<output(out_dfb_id, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
    } else {
        (void)in1_idx;
        mul<input(in0_dfb_id, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            input(in1_dfb_id, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            output(out_dfb_id, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>(
            IterationShape::one_tile());
    }
}

template <uint32_t num_tiles, uint32_t in0_dfb_id, uint32_t out_dfb_id>
ALWI void UNTILIZE_TILES() {
    compute_kernel_lib::untilize<
        num_tiles,
        in0_dfb_id,
        out_dfb_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t num_tiles, uint32_t in0_dfb_id, uint32_t out_dfb_id>
ALWI void TILIZE_ROWS(uint32_t sync_dfb_id) {
    DataflowBuffer sync_dfb(sync_dfb_id);
    sync_dfb.wait_front(num_tiles);
    compute_kernel_lib::tilize<
        num_tiles,
        in0_dfb_id,
        out_dfb_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    sync_dfb.pop_front(num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_dfb_id = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    DataflowBuffer dfb_scalar(scalar_dfb_id);
    dfb_scalar.wait_front(onetile);

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_dfb_id = get_compile_time_arg_val(12);
    constexpr uint32_t untilized_cos_sync_dfb_id = get_compile_time_arg_val(13);
    constexpr uint32_t untilized_sin_dfb_id = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_dfb_id = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_cos_dfb_id = get_compile_time_arg_val(16);
    constexpr uint32_t retilized_sin_dfb_id = get_compile_time_arg_val(17);
    compute_kernel_hw_startup(sin_dfb_id, scalar_dfb_id, untilized_sin_dfb_id);
    UNTILIZE_TILES<Wt, sin_dfb_id, untilized_sin_dfb_id>();
    UNTILIZE_TILES<Wt, cos_dfb_id, untilized_cos_dfb_id>();
    reconfig_data_format_srca(cos_dfb_id, untilized_sin_dfb_id);
    pack_reconfig_data_format(untilized_cos_dfb_id, retilized_sin_dfb_id);
    TILIZE_ROWS<Wt, untilized_sin_dfb_id, retilized_sin_dfb_id>(untilized_sin_sync_dfb_id);
    TILIZE_ROWS<Wt, untilized_cos_dfb_id, retilized_cos_dfb_id>(untilized_cos_sync_dfb_id);
    constexpr uint32_t updated_cos_dfb_id = retilized_cos_dfb_id;
    constexpr uint32_t updated_sin_dfb_id = retilized_sin_dfb_id;
#else
    compute_kernel_hw_startup(rotated_in_dfb_id, scalar_dfb_id, rotated_in_interm_dfb_id);
    constexpr uint32_t updated_cos_dfb_id = cos_dfb_id;
    constexpr uint32_t updated_sin_dfb_id = sin_dfb_id;
#endif
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            const uint32_t in1_idx = kDecodeMode ? j : 0;
            if (j < half_Wt) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::input(rotated_in_dfb_id),
                    compute_kernel_lib::input(
                        scalar_dfb_id,
                        compute_kernel_lib::BroadcastDim::Scalar,
                        compute_kernel_lib::WaitPolicy::None,
                        compute_kernel_lib::PopPolicy::None),
                    compute_kernel_lib::output(rotated_in_interm_dfb_id)>(
                    compute_kernel_lib::IterationShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_dfb_id, updated_sin_dfb_id);
                pack_reconfig_data_format(rotated_in_interm_dfb_id, sin_interm_dfb_id);
                mul_tiles_chain<rotated_in_interm_dfb_id, updated_sin_dfb_id, sin_interm_dfb_id>(in1_idx);
            } else {
                reconfig_data_format(rotated_in_dfb_id, updated_sin_dfb_id);
                pack_reconfig_data_format(out_dfb_id, sin_interm_dfb_id);
                mul_tiles_chain<rotated_in_dfb_id, updated_sin_dfb_id, sin_interm_dfb_id>(in1_idx);
            }

            mul_tiles_chain<in_dfb_id, updated_cos_dfb_id, cos_interm_dfb_id>(in1_idx);

            compute_kernel_lib::add<
                compute_kernel_lib::input(cos_interm_dfb_id),
                compute_kernel_lib::input(sin_interm_dfb_id),
                compute_kernel_lib::output(out_dfb_id)>(compute_kernel_lib::IterationShape::tiles(onetile));
        }
    }
}
