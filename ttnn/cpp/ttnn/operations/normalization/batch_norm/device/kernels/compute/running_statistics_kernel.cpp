// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Chain replacement for moreh's `*_tiles_to_cb` helpers — emits a single-tile
// BinaryFpu + PackTile with Input+Output reconfig, matching the moreh helper's
// internal `*_init_with_dt` + `pack_tile_with_dt`. PopA/PopB select Streaming
// (wait+pop per call) vs HeldStream (wait, no pop) — matches the moreh helper's
// pop_a / pop_b runtime flags but expressed at compile time.
template <compute_kernel_lib::BinaryFpuOp Op, uint32_t CbA, uint32_t CbB, uint32_t CbOut, bool PopA, bool PopB>
ALWI void fpu_binary_to_cb_chain() {
    using namespace compute_kernel_lib;
    eltwise_chain(
        1,
        BinaryFpu<
            CbA,
            CbB,
            Op,
            BroadcastDim::None,
            BinaryDataFormatReconfig::Input,
            PopA ? Streaming : HeldStream,
            PopB ? Streaming : HeldStream,
            OperandKind::Scalar,
            Dst::D0,
            OperandKind::Scalar>{},
        PackTile<CbOut, Dst::D0, OutStreaming, PackTileReconfig::Output>{});
}

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);  // batch mean
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);   // batch var
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);      // old running mean tensor
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);       // old running var tensor
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);  // updated running mean tensor
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);   // updated running var tensor
    constexpr auto cb_momentum = get_compile_time_arg_val(9);              // momentum
    constexpr auto cb_one = get_compile_time_arg_val(10);                  // stores 1
    constexpr auto cb_tmp1 = get_compile_time_arg_val(11);                 // tmp 1
    constexpr auto cb_tmp2 = get_compile_time_arg_val(12);                 // tmp 2
    constexpr auto cb_tmp3 = get_compile_time_arg_val(13);                 // tmp 3

    // cb_out0 receives the "last computed stat": var if both, else mean.
    constexpr bool mean_packs_to_out0 = old_running_mean_has_value && !old_running_var_has_value;
    constexpr bool var_packs_to_out0 = old_running_var_has_value;

    CircularBuffer cb_out0_obj(cb_out0);
    CircularBuffer cb_momentum_obj(cb_momentum);
    CircularBuffer cb_one_obj(cb_one);

    binary_op_init_common(cb_batch_mean, cb_batch_var, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_one_obj.wait_front(1);
    cb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        //
        // Stages 1-3: 3 separate 1-tile chains via fpu_binary_to_cb_chain. Each
        // chain replaces a moreh `*_tiles_to_cb` helper (`*_init_with_dt` +
        // mul/sub_tiles + `pack_tile_with_dt`) one-for-one.
        //
        // Stage 4 (add cb_tmp2 + cb_tmp3) is inlined here as an explicit chain
        // because it needs TWO outputs: the per-stat updated CB (Streaming) AND
        // a mirror pack to cb_out0 (CallerManaged — kernel pushes cb_out0 once
        // per iter). Replaces add_tiles_to_cb + the trailing pack_tile(0,
        // cb_out0). The cb_out0 PackTile uses PackTileReconfig::None to
        // preserve the original plain pack_tile(0, cb_out0) (no pack_reconfig
        // before it in the pre-migration kernel).
        if constexpr (old_running_mean_has_value) {
            fpu_binary_to_cb_chain<BinaryFpuOp::Sub, cb_one, cb_momentum, cb_tmp1, false, false>();
            fpu_binary_to_cb_chain<BinaryFpuOp::Mul, cb_momentum, cb_batch_mean, cb_tmp2, false, true>();
            fpu_binary_to_cb_chain<BinaryFpuOp::Mul, cb_tmp1, cb_old_running_mean, cb_tmp3, true, true>();
            if constexpr (mean_packs_to_out0) {
                eltwise_chain(
                    onetile,
                    BinaryFpu<
                        cb_tmp2,
                        cb_tmp3,
                        BinaryFpuOp::Add,
                        BroadcastDim::None,
                        BinaryDataFormatReconfig::Input,
                        Streaming,
                        Streaming,
                        OperandKind::Scalar,
                        Dst::D0,
                        OperandKind::Scalar>{},
                    PackTile<cb_updated_running_mean, Dst::D0, OutStreaming, PackTileReconfig::Output>{},
                    PackTile<cb_out0, Dst::D0, OutCallerManaged, PackTileReconfig::None>{});
            } else {
                eltwise_chain(
                    onetile,
                    BinaryFpu<
                        cb_tmp2,
                        cb_tmp3,
                        BinaryFpuOp::Add,
                        BroadcastDim::None,
                        BinaryDataFormatReconfig::Input,
                        Streaming,
                        Streaming,
                        OperandKind::Scalar,
                        Dst::D0,
                        OperandKind::Scalar>{},
                    PackTile<cb_updated_running_mean, Dst::D0, OutStreaming, PackTileReconfig::Output>{});
            }
        }
        if constexpr (old_running_var_has_value) {
            fpu_binary_to_cb_chain<BinaryFpuOp::Sub, cb_one, cb_momentum, cb_tmp1, false, false>();
            fpu_binary_to_cb_chain<BinaryFpuOp::Mul, cb_momentum, cb_batch_var, cb_tmp2, false, true>();
            fpu_binary_to_cb_chain<BinaryFpuOp::Mul, cb_tmp1, cb_old_running_var, cb_tmp3, true, true>();
            // var_packs_to_out0 is always true when this branch runs.
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_tmp2,
                    cb_tmp3,
                    BinaryFpuOp::Add,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    Streaming,
                    Streaming,
                    OperandKind::Scalar,
                    Dst::D0,
                    OperandKind::Scalar>{},
                PackTile<cb_updated_running_var, Dst::D0, OutStreaming, PackTileReconfig::Output>{},
                PackTile<cb_out0, Dst::D0, OutCallerManaged, PackTileReconfig::None>{});
        }
        cb_out0_obj.push_back(1);
    }

    cb_one_obj.pop_front(1);
    cb_momentum_obj.pop_front(1);
}
