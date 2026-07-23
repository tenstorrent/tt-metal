// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_gate = 1, cb_weight = 2, cb_tmp = 3;
constexpr uint32_t cb_stats = 4, cb_inv = 5, cb_norm = 6, cb_out = 7, cb_scaler = 8;

void square(uint32_t n) {
    cb_reserve_back(cb_tmp, n);
    pack_reconfig_data_format(cb_tmp);
    reconfig_data_format_srca(cb_x);
    copy_tile_to_dst_init_short(cb_x);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(cb_x, i, 0);
        copy_tile(cb_x, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tmp, i);
        tile_regs_release();
    }
    cb_push_back(cb_tmp, n);
}

void inverse_rms(uint32_t epsilon_bits, uint32_t inv_v_bits) {
    cb_reserve_back(cb_inv, 1);
    pack_reconfig_data_format(cb_inv);
    reconfig_data_format_srca(cb_stats);
    copy_tile_to_dst_init_short(cb_stats);
    tile_regs_acquire();
    copy_tile(cb_stats, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_v_bits);
    add_unary_tile(0, epsilon_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv, 0);
    tile_regs_release();
    cb_push_back(cb_inv, 1);
}

void normalize(uint32_t Vt) {
    cb_reserve_back(cb_norm, Vt);
    pack_reconfig_data_format(cb_norm);
    reconfig_data_format(cb_x, cb_inv);
    mul_bcast_cols_init_short(cb_x, cb_inv);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_x, cb_inv, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm, i);
        tile_regs_release();
    }
    cb_push_back(cb_norm, Vt);
}

void apply_weight(uint32_t Vt) {
    cb_reserve_back(cb_tmp, Vt);
    pack_reconfig_data_format(cb_tmp);
    reconfig_data_format(cb_norm, cb_weight);
    mul_bcast_rows_init_short(cb_norm, cb_weight);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_norm, cb_weight, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tmp, i);
        tile_regs_release();
    }
    cb_push_back(cb_tmp, Vt);
}

void activate_gate(uint32_t Vt) {
    cb_reserve_back(cb_norm, Vt);
    pack_reconfig_data_format(cb_norm);
    reconfig_data_format_srca(cb_gate);
    copy_tile_to_dst_init_short(cb_gate);
    sigmoid_tile_init();
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        copy_tile(cb_gate, i, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm, i);
        tile_regs_release();
    }
    cb_push_back(cb_norm, Vt);
}

void multiply_output(uint32_t Vt) {
    cb_reserve_back(cb_out, Vt);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format(cb_tmp, cb_norm);
    mul_tiles_init(cb_tmp, cb_norm);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles(cb_tmp, cb_norm, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, i);
        tile_regs_release();
    }
    cb_push_back(cb_out, Vt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(1);
    constexpr uint32_t inv_v_bits = get_compile_time_arg_val(2);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_scaler, cb_out);
    CircularBuffer(cb_weight).wait_front(Vt);
    CircularBuffer(cb_scaler).wait_front(1);
    for (uint32_t i = 0; i < count; i++) {
        CircularBuffer(cb_x).wait_front(Vt);
        CircularBuffer(cb_gate).wait_front(Vt);
        square(Vt);
        CircularBuffer(cb_tmp).wait_front(Vt);
        compute_kernel_lib::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_tmp,
            cb_scaler,
            cb_stats,
            compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        CircularBuffer(cb_stats).wait_front(1);
        CircularBuffer(cb_tmp).pop_front(Vt);
        inverse_rms(epsilon_bits, inv_v_bits);
        CircularBuffer(cb_inv).wait_front(1);
        normalize(Vt);
        CircularBuffer(cb_norm).wait_front(Vt);
        CircularBuffer(cb_x).pop_front(Vt);
        CircularBuffer(cb_inv).pop_front(1);
        CircularBuffer(cb_stats).pop_front(1);
        apply_weight(Vt);
        CircularBuffer(cb_tmp).wait_front(Vt);
        CircularBuffer(cb_norm).pop_front(Vt);
        activate_gate(Vt);
        CircularBuffer(cb_norm).wait_front(Vt);
        CircularBuffer(cb_gate).pop_front(Vt);
        multiply_output(Vt);
        CircularBuffer(cb_tmp).pop_front(Vt);
        CircularBuffer(cb_norm).pop_front(Vt);
    }
}
