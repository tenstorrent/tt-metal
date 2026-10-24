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

// Tiles per destination-register acquire: each stage packs a whole block of tiles per acquire, so its unpack, math and
// pack pipeline across the halves of DST instead of handing over one tile at a time.
constexpr uint32_t max_block_tiles = DST_ACCUM_MODE ? 4 : 8;

// Run `stage(tile, dst)` for tiles [0, n) in blocks of up to max_block_tiles per acquire and pack each block to `o`.
template <typename Stage>
void blocked(uint32_t n, uint32_t o, Stage stage) {
    for (uint32_t first = 0; first < n; first += max_block_tiles) {
        const uint32_t count = n - first < max_block_tiles ? n - first : max_block_tiles;
        tile_regs_acquire();
        for (uint32_t t = 0; t < count; t++) {
            stage(first + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < count; t++) {
            pack_tile(t, o, first + t);
        }
        tile_regs_release();
    }
}

template <typename DFB>
void square(uint32_t n, DFB& tmp) {
    tmp.reserve_back(n);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::x, dfb::x);
    mul_init(dfb::x, dfb::x, false);
    blocked(n, dfb::tmp, [](uint32_t i, uint32_t dst) { mul_tiles(dfb::x, dfb::x, i, i, dst); });
    tmp.push_back(n);
}

template <typename DFB>
void inverse_rms(DFB& inv) {
    inv.reserve_back(1);
    pack_reconfig_data_format(dfb::inv);
    reconfig_data_format(dfb::stats, dfb::epsilon);
    add_init(dfb::stats, dfb::epsilon);
    tile_regs_acquire();
    add_tiles(dfb::stats, dfb::epsilon, 0, 0, 0);
    // The row reduction leaves the mean square in column 0, the only column the broadcast reads.
    rsqrt_tile_init();
    rsqrt_tile(0, VectorMode::C);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::inv, 0);
    tile_regs_release();
    inv.push_back(1);
}

template <typename DFB>
void scale_by_inverse_rms(uint32_t Vt, DFB& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format(dfb::x, dfb::inv);
    mul_bcast_cols_init(dfb::x, dfb::inv);
    blocked(Vt, dfb::norm, [](uint32_t i, uint32_t dst) { mul_tiles_bcast_cols(dfb::x, dfb::inv, i, 0, dst); });
    norm.push_back(Vt);
}

template <typename DFB>
void apply_weight(uint32_t Vt, DFB& tmp) {
    tmp.reserve_back(Vt);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::norm, dfb::weight);
    mul_bcast_rows_init(dfb::norm, dfb::weight);
    blocked(Vt, dfb::tmp, [](uint32_t i, uint32_t dst) { mul_tiles_bcast_rows(dfb::norm, dfb::weight, i, i, dst); });
    tmp.push_back(Vt);
}

template <typename DFB>
void activate_gate(uint32_t Vt, DFB& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format_srca(dfb::gate);
    copy_init(dfb::gate);
    sigmoid_tile_init();
    blocked(Vt, dfb::norm, [](uint32_t i, uint32_t dst) {
        copy_tile(dfb::gate, i, dst);
        sigmoid_tile(dst);
    });
    norm.push_back(Vt);
}

template <typename DFB>
void multiply_output(uint32_t Vt, DFB& out) {
    out.reserve_back(Vt);
    pack_reconfig_data_format(dfb::out);
    reconfig_data_format(dfb::tmp, dfb::norm);
    mul_init(dfb::tmp, dfb::norm);
    blocked(Vt, dfb::out, [](uint32_t i, uint32_t dst) { mul_tiles(dfb::tmp, dfb::norm, i, i, dst); });
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
