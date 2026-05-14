// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Streaming chain replacement for moreh's `*_tiles_to_cb` helpers: one FPU
// binary tile written to CbOut via BinaryFpu + PackTile, with input-and-output
// dtype reconfig matching `*_tiles_init_with_dt(CbA, CbB)` + `pack_tile_with_dt`.
// PopA/PopB choose WaitAndPop vs WaitNoPop.
template <
    compute_kernel_lib::BinaryFpuOp Op,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    bool PopA,
    bool PopB>
ALWI void fpu_binary_to_cb_chain() {
    using namespace compute_kernel_lib;
    using BinElt = BinaryFpu<
        CbA,
        CbB,
        CbOut,
        Op,
        BroadcastDim::None,
        BinaryDataFormatReconfig::InputAndOutput,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop>;
    using PackElt = PackTile<
        CbOut,
        Dst::D0,
        PackTilePolicy::PerTileReserveAndPush,
        PackTileIndexMode::FirstTile,
        PackTileReconfig::Output>;
    eltwise_chain(1, BinElt{}, PackElt{});
}

}  // namespace

void kernel_main() {
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

    experimental::CircularBuffer cb_out0_obj(cb_out0);
    experimental::CircularBuffer cb_momentum_obj(cb_momentum);
    experimental::CircularBuffer cb_one_obj(cb_one);

    binary_op_init_common(cb_batch_mean, cb_batch_var, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_one_obj.wait_front(1);
    cb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        tile_regs_acquire();
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        //
        // Migrated: the first three stages of each branch (sub_tiles_to_cb +
        // 2× mul_tiles_to_cb) are replaced with single-iter eltwise_chain calls
        // that emit BinaryFpu + PackTile with `InputAndOutput` reconfig. The
        // final add_tiles_to_cb stage is kept raw — its DEST[0] is read by the
        // outer `pack_tile(0, cb_out0)` below; a chain-managed acquire/release
        // would not preserve DEST[0] across the chain boundary.

        if constexpr (old_running_mean_has_value) {
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Sub, cb_one, cb_momentum, cb_tmp1, false, false>();
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Mul, cb_momentum, cb_batch_mean, cb_tmp2, false, true>();
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Mul, cb_tmp1, cb_old_running_mean, cb_tmp3, true, true>();
            add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_mean, 0, 0, 1, 1);  // cb_tmp2 + cb_tmp3
        }
        if constexpr (old_running_var_has_value) {
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Sub, cb_one, cb_momentum, cb_tmp1, false, false>();
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Mul, cb_momentum, cb_batch_var, cb_tmp2, false, true>();
            fpu_binary_to_cb_chain<compute_kernel_lib::BinaryFpuOp::Mul, cb_tmp1, cb_old_running_var, cb_tmp3, true, true>();
            add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_var, 0, 0, 1, 1);  // cb_tmp2 + cb_tmp3
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();
        cb_out0_obj.push_back(1);
    }

    cb_one_obj.pop_front(1);
    cb_momentum_obj.pop_front(1);
}
