// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

template <bool kDecodeMode, uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb>
ALWI void mul_tiles_chain(uint32_t in1_idx) {
    using namespace compute_kernel_lib;
    if constexpr (kDecodeMode) {
        eltwise_chain(
            IterationShape::one_tile(),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(in0_cb, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
                // We don't pop in1 in decode which is sin/cos since we don't stream
                input(
                    in1_cb,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Scalar,
                    DataFormatReconfig::Disabled,
                    compute_kernel_lib::TileAddressing::Offset)>{0u, in1_idx},
            PackTile<output(out_cb, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
    } else {
        (void)in1_idx;
        mul<input(in0_cb, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            input(in1_cb, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            output(out_cb, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>(
            IterationShape::one_tile());
    }
}

template <uint32_t num_tiles, uint32_t in0_cb, uint32_t out_cb>
ALWI void UNTILIZE_TILES() {
    compute_kernel_lib::untilize<
        num_tiles,
        in0_cb,
        out_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t num_tiles, uint32_t in0_cb, uint32_t out_cb>
ALWI void TILIZE_ROWS(uint32_t sync_cb) {
    CircularBuffer cb_sync(sync_cb);
    cb_sync.wait_front(num_tiles);
    compute_kernel_lib::tilize<
        num_tiles,
        in0_cb,
        out_cb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    cb_sync.pop_front(num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);
    constexpr bool kDecodeMode = get_compile_time_arg_val(12) != 0;

    CircularBuffer cb_scalar(scalar_cb);
    cb_scalar.wait_front(onetile);

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_cb = get_compile_time_arg_val(13);
    constexpr uint32_t untilized_cos_sync_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(15);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(16);
    constexpr uint32_t retilized_cos_cb = get_compile_time_arg_val(17);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(18);
    compute_kernel_hw_startup(sin_cb, scalar_cb, untilized_sin_cb);
    UNTILIZE_TILES<Wt, sin_cb, untilized_sin_cb>();
    UNTILIZE_TILES<Wt, cos_cb, untilized_cos_cb>();
    reconfig_data_format_srca(cos_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_cos_cb, retilized_sin_cb);
    TILIZE_ROWS<Wt, untilized_sin_cb, retilized_sin_cb>(untilized_sin_sync_cb);
    TILIZE_ROWS<Wt, untilized_cos_cb, retilized_cos_cb>(untilized_cos_sync_cb);
    constexpr uint32_t updated_cos_cb = retilized_cos_cb;
    constexpr uint32_t updated_sin_cb = retilized_sin_cb;
#else
    compute_kernel_hw_startup(rotated_in_cb, scalar_cb, rotated_in_interm_cb);
    constexpr uint32_t updated_cos_cb = cos_cb;
    constexpr uint32_t updated_sin_cb = sin_cb;
#endif
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            const uint32_t in1_idx = kDecodeMode ? j : 0;
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                compute_kernel_lib::mul<
                    compute_kernel_lib::input(rotated_in_cb),
                    compute_kernel_lib::input(
                        scalar_cb,
                        compute_kernel_lib::BroadcastDim::Scalar,
                        compute_kernel_lib::WaitPolicy::None,
                        compute_kernel_lib::PopPolicy::None),
                    compute_kernel_lib::output(rotated_in_interm_cb)>(
                    compute_kernel_lib::IterationShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_cb, updated_sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                // Multiply rotated input by sin
                mul_tiles_chain<kDecodeMode, rotated_in_interm_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            } else {
                reconfig_data_format(rotated_in_cb, updated_sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                // Multiply rotated input by sin
                mul_tiles_chain<kDecodeMode, rotated_in_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            }

            // Multiply input by cos
            mul_tiles_chain<kDecodeMode, in_cb, updated_cos_cb, cos_interm_cb>(in1_idx);

            // Add applied sin/cos tensors
            compute_kernel_lib::add<
                compute_kernel_lib::input(cos_interm_cb),
                compute_kernel_lib::input(sin_interm_cb),
                compute_kernel_lib::output(out_cb)>(compute_kernel_lib::IterationShape::tiles(onetile));
        }
    }
}
