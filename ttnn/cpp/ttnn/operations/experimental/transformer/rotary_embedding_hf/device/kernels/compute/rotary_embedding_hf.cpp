// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

// in0 * in1 -> out, plain per-iter streaming mul with NO data-format reconfig.
// The explicit reconfig_data_format / pack_reconfig_data_format calls preserved in
// the loop body set the src/pack formats; this mirrors the device-validated
// rotary_embedding.cpp non-DECODE `mul_tiles_chain` (BinaryDataFormatReconfig::None,
// PackTileReconfig::None). Replaces the old raw-LLK ALWI MUL_TILES helper.
template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb>
ALWI void mul_tiles_chain() {
    ckl::mul<
        in0_cb,
        in1_cb,
        out_cb,
        ckl::BroadcastDim::None,
        ckl::InputLifecycle::Streaming,
        ckl::InputLifecycle::Streaming,
        ckl::OutputLifecycle::Streaming,
        ckl::BinaryDataFormatReconfig::None,
        ckl::PackTileReconfig::None>(ckl::EltwiseShape::single());
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

    binary_op_init_common(rotated_in_cb, scalar_cb, rotated_in_interm_cb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1).
                // Original: reconfig_data_format(rotated_in_cb, scalar_cb) +
                //   pack_reconfig_data_format(rotated_in_interm_cb) -> replaced by the
                //   convenience default BinaryDataFormatReconfig::Input + PackTileReconfig::Output.
                //   rotated_in_cb InputLifecycle::Streaming; scalar_cb InputLifecycle::CallerManaged
                //   (waited once above, never popped); rotated_in_interm_cb OutputLifecycle::Streaming.
                ckl::mul<
                    rotated_in_cb,
                    scalar_cb,
                    rotated_in_interm_cb,
                    ckl::BroadcastDim::Scalar,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::CallerManaged>(ckl::EltwiseShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_cb, sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                // Multiply rotated input by sin (chain-based)
                mul_tiles_chain<rotated_in_interm_cb, sin_cb, sin_interm_cb>();
            } else {
                reconfig_data_format(rotated_in_cb, sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                // Multiply rotated input by sin (chain-based)
                mul_tiles_chain<rotated_in_cb, sin_cb, sin_interm_cb>();
            }

            // Multiply input by cos (chain-based)
            mul_tiles_chain<in_cb, cos_cb, cos_interm_cb>();

            // Add applied sin/cos tensors -> out_cb.
            // Original: reconfig_data_format_srca(rotated_in_cb, cos_interm_cb) +
            //   pack_reconfig_data_format(cos_interm_cb, out_cb) -> replaced by the convenience
            //   default Input/Output reconfig. cos_interm_cb/sin_interm_cb InputLifecycle::Streaming;
            //   out_cb OutputLifecycle::Streaming.
            ckl::add<cos_interm_cb, sin_interm_cb, out_cb>(ckl::EltwiseShape::tiles(onetile));
        }
    }
}
