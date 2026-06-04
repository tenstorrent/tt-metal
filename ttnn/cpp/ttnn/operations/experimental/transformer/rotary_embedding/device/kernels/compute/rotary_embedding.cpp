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

// Per-CB-constexpr chain wrapper for `mul(in0, in1) -> out`. Replaces the
// older raw-LLK ALWI MUL_TILES helper. All three CBs are compile-time
// (compile-time args) so this resolves to a single eltwise_chain call.
//
// DECODE_MODE branch: in1_cb is the retilized sin/cos held across the entire
// num_rows*Wt walk; we read tile at runtime offset `in1_idx` (= j) with
// bcast-rows, never popping in1. compute_kernel_lib::TileOffset::Set requires InputLifecycle::CallerManaged
// lifecycle, so the wait/reserve/pop/push are emitted externally.
//
// Non-DECODE_MODE branch: standard per-iter InputLifecycle::Streaming on both sides + plain
// mul_tiles (no bcast).
template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb>
ALWI void mul_tiles_chain(uint32_t in1_idx) {
    using namespace compute_kernel_lib;
#ifdef DECODE_MODE
    cb_wait_front(in0_cb, 1);
    cb_wait_front(in1_cb, in1_idx + 1);
    cb_reserve_back(out_cb, 1);
    eltwise_chain(
        1u,
        BinaryFpu<
            in0_cb,
            in1_cb,
            BinaryFpuOp::Mul,
            BroadcastDim::Row,
            BinaryDataFormatReconfig::None,
            InputLifecycle::CallerManaged,
            InputLifecycle::CallerManaged,
            OperandKind::Scalar,
            Dst::D0,
            OperandKind::Scalar,
            compute_kernel_lib::TileOffset::Unset,
            compute_kernel_lib::TileOffset::Set>{0u, in1_idx},
        PackTile<out_cb, Dst::D0, OutputLifecycle::CallerManaged, PackTileReconfig::None>{});
    cb_pop_front(in0_cb, 1);
    cb_push_back(out_cb, 1);
    // in1 NOT popped — held across the whole walk per DECODE_MODE contract.
#else
    (void)in1_idx;  // unused — in1 is per-iter streamed at index 0.
    eltwise_chain(
        1u,
        BinaryFpu<
            in0_cb,
            in1_cb,
            BinaryFpuOp::Mul,
            BroadcastDim::None,
            BinaryDataFormatReconfig::None,
            InputLifecycle::Streaming,
            InputLifecycle::Streaming,
            OperandKind::Scalar,
            Dst::D0,
            OperandKind::Scalar>{},
        PackTile<out_cb, Dst::D0, OutputLifecycle::Streaming, PackTileReconfig::None>{});
#endif
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
#ifdef DECODE_MODE
            const uint32_t in1_idx = j;
#else
            const uint32_t in1_idx = 0;
#endif
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1).
                // Reconfig audit: explicit reconfig_data_format(rotated_in_cb, scalar_cb) +
                //   mul_tiles_bcast_scalar_init_short reconfigs srca/srcb -> Input.
                //   Explicit pack_reconfig_data_format(rotated_in_interm_cb) -> Output.
                // Lifecycles: rotated_in_cb InputLifecycle::Streaming (wait+pop per iter); scalar_cb
                //   InputLifecycle::CallerManaged (waited once at line 85, never popped); rotated_in_interm_cb
                //   OutputLifecycle::Streaming.
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        rotated_in_cb,
                        scalar_cb,
                        compute_kernel_lib::BinaryFpuOp::Mul,
                        compute_kernel_lib::BroadcastDim::Scalar,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::CallerManaged,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        rotated_in_interm_cb,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
                reconfig_data_format_srcb(scalar_cb, updated_sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                // Multiply rotated input by sin (chain-based)
                mul_tiles_chain<rotated_in_interm_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            } else {
                reconfig_data_format(rotated_in_cb, updated_sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                // Multiply rotated input by sin (chain-based)
                mul_tiles_chain<rotated_in_cb, updated_sin_cb, sin_interm_cb>(in1_idx);
            }

            // Multiply input by cos (chain-based)
            mul_tiles_chain<in_cb, updated_cos_cb, cos_interm_cb>(in1_idx);

            // Add applied sin/cos tensors -> out_cb.
            // Reconfig audit: reconfig_data_format_srca(rotated_in_cb, cos_interm_cb)
            //   reconfigs srca to cos_interm_cb; add_tiles_init reconfigs srca/srcb to
            //   (cos_interm, sin_interm) -> Input. Explicit pack_reconfig to out_cb -> Output.
            // Lifecycles: cos_interm_cb/sin_interm_cb InputLifecycle::Streaming (per-iter wait+pop);
            //   out_cb OutputLifecycle::Streaming.
            compute_kernel_lib::add<cos_interm_cb, sin_interm_cb, out_cb>(onetile);
        }
    }
}
