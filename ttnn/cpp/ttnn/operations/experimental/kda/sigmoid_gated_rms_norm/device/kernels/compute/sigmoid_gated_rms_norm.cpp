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

// A core's items run as a group: each stage runs over all of the group's items, so its init and data-format
// reconfiguration are paid once per group and consecutive items' blocks pipeline through the unpacker, math and packer.
// A stage that reads the previous stage's output starts on an item the previous stage finished earlier, so no stage
// waits for a pack to land in L1 except where a gate sigmoid (independent of the RMS chain) runs in between. The
// stages hand items over Vt tiles at a time; x and the gate stay at the front of their buffers until the group ends.

template <uint32_t Vt, typename DFB>
void square(uint32_t items, DFB& x, DFB& tmp) {
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::x, dfb::x);
    mul_init(dfb::x, dfb::x, false);
    for (uint32_t j = 0; j < items; j++) {
        x.wait_front((j + 1) * Vt);
        tmp.reserve_back(Vt);
        const uint32_t base = j * Vt;
        blocked(Vt, dfb::tmp, [base](uint32_t i, uint32_t dst) { mul_tiles(dfb::x, dfb::x, base + i, base + i, dst); });
        tmp.push_back(Vt);
    }
}

template <typename DFB>
void inverse_rms(uint32_t items, DFB& stats, DFB& inv) {
    pack_reconfig_data_format(dfb::inv);
    reconfig_data_format(dfb::stats, dfb::epsilon);
    stats.wait_front(items);
    inv.reserve_back(items);
    for (uint32_t first = 0; first < items; first += max_block_tiles) {
        const uint32_t count = items - first < max_block_tiles ? items - first : max_block_tiles;
        add_init(dfb::stats, dfb::epsilon);
        tile_regs_acquire();
        for (uint32_t t = 0; t < count; t++) {
            add_tiles(dfb::stats, dfb::epsilon, first + t, 0, t);
        }
        // The row reduction leaves the mean square in column 0, the only column the broadcast reads.
        rsqrt_tile_init();
        for (uint32_t t = 0; t < count; t++) {
            rsqrt_tile(t, VectorMode::C);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < count; t++) {
            pack_tile(t, dfb::inv, first + t);
        }
        tile_regs_release();
    }
    inv.push_back(items);
    stats.pop_front(items);
}

template <uint32_t Vt, typename DFB>
void scale_by_inverse_rms(uint32_t items, DFB& inv, DFB& norm) {
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format(dfb::x, dfb::inv);
    mul_bcast_cols_init(dfb::x, dfb::inv);
    inv.wait_front(items);
    for (uint32_t j = 0; j < items; j++) {
        norm.reserve_back(Vt);
        const uint32_t base = j * Vt;
        blocked(Vt, dfb::norm, [base, j](uint32_t i, uint32_t dst) {
            mul_tiles_bcast_cols(dfb::x, dfb::inv, base + i, j, dst);
        });
        norm.push_back(Vt);
    }
    inv.pop_front(items);
}

template <uint32_t Vt, typename DFB>
void apply_weight(uint32_t items, DFB& norm, DFB& tmp) {
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::norm, dfb::weight);
    mul_bcast_rows_init(dfb::norm, dfb::weight);
    for (uint32_t j = 0; j < items; j++) {
        norm.wait_front(Vt);
        tmp.reserve_back(Vt);
        blocked(
            Vt, dfb::tmp, [](uint32_t i, uint32_t dst) { mul_tiles_bcast_rows(dfb::norm, dfb::weight, i, i, dst); });
        tmp.push_back(Vt);
        norm.pop_front(Vt);
    }
}

// sigmoid(gate) of item j into sig.
template <uint32_t Vt, typename DFB>
void activate_gate(uint32_t j, DFB& gate, DFB& sig) {
    pack_reconfig_data_format(dfb::sig);
    reconfig_data_format_srca(dfb::gate);
    copy_init(dfb::gate);
    sigmoid_tile_init();
    gate.wait_front((j + 1) * Vt);
    sig.reserve_back(Vt);
    const uint32_t base = j * Vt;
    blocked(Vt, dfb::sig, [base](uint32_t i, uint32_t dst) {
        copy_tile(dfb::gate, base + i, dst);
        sigmoid_tile(dst);
    });
    sig.push_back(Vt);
}

// out = weighted * sigmoid(gate) for the next `items` items, the sigmoids read from `sig_id` (the kernel's own sig, or
// the activated gate: both FP32, unpacked to the source registers).
template <uint32_t Vt, typename Id, typename DFB>
void multiply_output(uint32_t items, DFB& tmp, Id sig_id, DFB& sig, DFB& out) {
    pack_reconfig_data_format(dfb::out);
    reconfig_data_format(dfb::tmp, sig_id);
    mul_init(dfb::tmp, sig_id);
    for (uint32_t j = 0; j < items; j++) {
        tmp.wait_front(Vt);
        sig.wait_front(Vt);
        out.reserve_back(Vt);
        blocked(Vt, dfb::out, [sig_id](uint32_t i, uint32_t dst) { mul_tiles(dfb::tmp, sig_id, i, i, dst); });
        out.push_back(Vt);
        tmp.pop_front(Vt);
        sig.pop_front(Vt);
    }
}

// Gate sigmoids of a group before its RMS chain starts: they take about as long as the group's inputs take to land.
constexpr uint32_t leading_sigmoids = 2;

// A group's items that need a sigmoid lead it (raw of them); the rest read their activated gate (ACTIVATED_GATE).
template <uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t raw_count, uint32_t act_count) {
    compute_kernel_hw_startup(dfb::x, dfb::scaler, dfb::out);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer norm(dfb::norm);
    DataflowBuffer sig(dfb::sig);
    DataflowBuffer out(dfb::out);
    DataflowBuffer epsilon(dfb::epsilon);
#ifdef ACTIVATED_GATE
    DataflowBuffer activated(dfb::activated);
#endif
    const uint32_t wi_count = raw_count + act_count;
    for (uint32_t first = 0; first < wi_count; first += G) {
        const uint32_t items = wi_count - first < G ? wi_count - first : G;
        const uint32_t raw = first >= raw_count ? 0 : (raw_count - first < items ? raw_count - first : items);
        // The gate sigmoids fill the chain's waits: the first ones run while the group's inputs land (the gate comes
        // first, on the other NoC), the next one between the reduction and the inverse RMS that needs its last tile.
        // The last one runs after the products of the items whose sigmoids are ready, so their writes overlap it.
        uint32_t sigmoids = 0;
        while (sigmoids < raw && sigmoids < leading_sigmoids) {
            activate_gate<Vt>(sigmoids++, gate, sig);
        }
        square<Vt>(items, x, tmp);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(items, Vt));
        if (sigmoids < raw) {
            activate_gate<Vt>(sigmoids++, gate, sig);
        }
        if (first == 0) {
            epsilon.wait_front(1);
        }
        inverse_rms(items, stats, inv);
        scale_by_inverse_rms<Vt>(items, inv, norm);
        x.pop_front(items * Vt);
        if (first == 0) {
            weight.wait_front(Vt);
        }
        apply_weight<Vt>(items, norm, tmp);
        uint32_t products = 0;
        while (sigmoids < raw) {
            multiply_output<Vt>(sigmoids - products, tmp, dfb::sig, sig, out);
            products = sigmoids;
            activate_gate<Vt>(sigmoids++, gate, sig);
        }
        if (raw > products) {
            multiply_output<Vt>(raw - products, tmp, dfb::sig, sig, out);
        }
#ifdef ACTIVATED_GATE
        if (items > raw) {
            multiply_output<Vt>(items - raw, tmp, dfb::activated, activated, out);
        }
#endif
        if (raw > 0) {
            gate.pop_front(raw * Vt);
        }
    }
}
