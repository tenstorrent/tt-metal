// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_input = 0;    // input(==tmp_pow_sum)
    constexpr uint32_t cb_decimal = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;  // output(==total_norm)

    constexpr uint32_t cb_x = 24;         // Sum[tmp_pow_sum](==x)
    constexpr uint32_t cb_xpow = 25;      // x^p
    constexpr uint32_t cb_logx = 26;      // log(x)
    constexpr uint32_t cb_exp_lxmd = 27;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x, cb_y);
    } else {
        binary_op_init_common(cb_logx, cb_decimal, cb_y);
    }

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            // Seed cb_x with first cb_input tile.
            // Original: copy_tile_init(cb_input) reconfigs srca; pack_tile no reconfig.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_input,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::PackTileReconfig::None>{});
        } else {
            // cb_x = cb_input + cb_x (in-place accumulator).
            // Original: add_tiles_init reconfigs srca/srcb; pack_tile no reconfig.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_input,
                    cb_x,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }
    }
    // x^p
    power_tile_to_cb(cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y, p, p_is_negative);
}
