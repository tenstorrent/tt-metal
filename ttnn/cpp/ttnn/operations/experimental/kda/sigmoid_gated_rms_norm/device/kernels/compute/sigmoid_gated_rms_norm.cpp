// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void square(uint32_t n, DataflowBuffer& tmp) {
    tmp.reserve_back(n);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::x, dfb::x);
    mul_init(dfb::x, dfb::x, false);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(dfb::x, dfb::x, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(n);
}

void inverse_rms(DataflowBuffer& inv) {
    inv.reserve_back(1);
    pack_reconfig_data_format(dfb::inv);
    reconfig_data_format(dfb::stats, dfb::epsilon);
    add_init(dfb::stats, dfb::epsilon);
    tile_regs_acquire();
    add_tiles(dfb::stats, dfb::epsilon, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::inv, 0);
    tile_regs_release();
    inv.push_back(1);
}

void scale_by_inverse_rms(uint32_t Vt, DataflowBuffer& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format(dfb::x, dfb::inv);
    mul_bcast_cols_init(dfb::x, dfb::inv);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(dfb::x, dfb::inv, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::norm, i);
        tile_regs_release();
    }
    norm.push_back(Vt);
}

void apply_weight(uint32_t Vt, DataflowBuffer& tmp) {
    tmp.reserve_back(Vt);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::norm, dfb::weight);
    mul_bcast_rows_init(dfb::norm, dfb::weight);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::norm, dfb::weight, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(Vt);
}

void activate_gate(uint32_t Vt, DataflowBuffer& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format_srca(dfb::gate);
    copy_tile_to_dst_init_short(dfb::gate);
    sigmoid_tile_init();
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        copy_tile(dfb::gate, i, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::norm, i);
        tile_regs_release();
    }
    norm.push_back(Vt);
}

void multiply_output(uint32_t Vt, DataflowBuffer& out) {
    out.reserve_back(Vt);
    pack_reconfig_data_format(dfb::out);
    reconfig_data_format(dfb::tmp, dfb::norm);
    mul_init(dfb::tmp, dfb::norm);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles(dfb::tmp, dfb::norm, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, i);
        tile_regs_release();
    }
    out.push_back(Vt);
}

template <uint32_t Vt>
TT_KERNEL void compute(uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::x, dfb::scaler, dfb::out);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer norm(dfb::norm);
    DataflowBuffer out(dfb::out);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer epsilon(dfb::epsilon);
    weight.wait_front(Vt);
    scaler.wait_front(1);
    epsilon.wait_front(1);
    for (uint32_t i = 0; i < wi_count; i++) {
        x.wait_front(Vt);
        gate.wait_front(Vt);
        square(Vt, tmp);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);
        inverse_rms(inv);
        inv.wait_front(1);
        scale_by_inverse_rms(Vt, norm);
        norm.wait_front(Vt);
        x.pop_front(Vt);
        inv.pop_front(1);
        stats.pop_front(1);
        apply_weight(Vt, tmp);
        tmp.wait_front(Vt);
        norm.pop_front(Vt);
        activate_gate(Vt, norm);
        norm.wait_front(Vt);
        gate.pop_front(Vt);
        multiply_output(Vt, out);
        tmp.pop_front(Vt);
        norm.pop_front(Vt);
    }
}
