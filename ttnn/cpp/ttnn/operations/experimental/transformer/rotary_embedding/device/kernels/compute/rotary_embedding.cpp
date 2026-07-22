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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

#ifdef DECODE_MODE
inline constexpr bool kDecodeMode = true;
#else
inline constexpr bool kDecodeMode = false;
#endif

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb>
ALWI void mul_tiles_chain(uint32_t in1_idx) {
    using namespace compute_kernel_lib;
    if constexpr (kDecodeMode) {
        cb_wait_front(in1_cb, in1_idx + 1);
        eltwise_chain(
            EltwiseShape::single(),
            BinaryFpu<  // in0: chain owns wait(1)/pop(1)
                input(in0_cb, InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(
                    in1_cb,
                    InputLifecycle::CallerManaged,  // in1: held across the walk (TileOffset, no pop)
                    OperandKind::Scalar,
                    DataFormatReconfig::Disabled,
                    compute_kernel_lib::TileOffset::Set),
                BinaryFpuOp::Mul,
                BroadcastDim::Row>{0u, in1_idx},
            PackTile<output(out_cb, OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
    } else {
        (void)in1_idx;
        mul<input(in0_cb, InputLifecycle::Streaming, DataFormatReconfig::Disabled),
            input(in1_cb, InputLifecycle::Streaming, DataFormatReconfig::Disabled),
            output(out_cb, OutputLifecycle::Streaming, DataFormatReconfig::Disabled),
            BroadcastDim::None>(EltwiseShape::single());
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
    cb_wait_front(sync_cb, num_tiles);
    compute_kernel_lib::tilize<
        num_tiles,
        in0_cb,
        out_cb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    cb_pop_front(sync_cb, num_tiles);
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

    cb_wait_front(scalar_cb, onetile);

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_cb = get_compile_time_arg_val(12);
    constexpr uint32_t untilized_cos_sync_cb = get_compile_time_arg_val(13);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_cos_cb = get_compile_time_arg_val(16);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(17);
    binary_op_init_common(sin_cb, scalar_cb, untilized_sin_cb);
    UNTILIZE_TILES<Wt, sin_cb, untilized_sin_cb>();
    UNTILIZE_TILES<Wt, cos_cb, untilized_cos_cb>();
    reconfig_data_format_srca(cos_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_cos_cb, retilized_sin_cb);
    TILIZE_ROWS<Wt, untilized_sin_cb, retilized_sin_cb>(untilized_sin_sync_cb);
    TILIZE_ROWS<Wt, untilized_cos_cb, retilized_cos_cb>(untilized_cos_sync_cb);
    constexpr uint32_t updated_cos_cb = retilized_cos_cb;
    constexpr uint32_t updated_sin_cb = retilized_sin_cb;
#else
    binary_op_init_common(rotated_in_cb, scalar_cb, rotated_in_interm_cb);
    constexpr uint32_t updated_cos_cb = cos_cb;
    constexpr uint32_t updated_sin_cb = sin_cb;
#endif
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            const uint32_t in1_idx = kDecodeMode ? j : 0;
            if (j < half_Wt) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::input(rotated_in_cb),
                    compute_kernel_lib::input(scalar_cb, compute_kernel_lib::InputLifecycle::CallerManaged),
                    compute_kernel_lib::output(rotated_in_interm_cb),
                    compute_kernel_lib::BroadcastDim::Scalar>(compute_kernel_lib::EltwiseShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_cb, updated_sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                mul_tiles_chain<rotated_in_interm_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            } else {
                reconfig_data_format(rotated_in_cb, updated_sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                mul_tiles_chain<rotated_in_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            }

            mul_tiles_chain<in_cb, updated_cos_cb, cos_interm_cb>(in1_idx);

            compute_kernel_lib::add<
                compute_kernel_lib::input(cos_interm_cb),
                compute_kernel_lib::input(sin_interm_cb),
                compute_kernel_lib::output(out_cb)>(compute_kernel_lib::EltwiseShape::tiles(onetile));
        }
    }
}
