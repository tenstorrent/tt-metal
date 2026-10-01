// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"

#include <cmath>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

namespace {

uint32_t fidelity_multiplier(MathFidelity fidelity) {
    switch (fidelity) {
        case MathFidelity::LoFi: return 1;
        case MathFidelity::HiFi2: return 2;
        case MathFidelity::HiFi3: return 3;
        default: return 4;
    }
}

}  // namespace

// Per-core roofline estimate, in cycles, of a blocked candidate: the largest of
//  - compute: the busiest core's tile products at the matrix engine's rate for the math fidelity (A tiles
//    shorter than 8 rows still take a full 8-row pass of the engine);
//  - NoC: the input bytes the busiest core receives (its rows of A and columns of B, once per output block
//    that uses them);
//  - DRAM: the bytes read from and written to DRAM in total (the mcast layouts read A once per output column
//    block and B once per output row block; Reuse reads A once and B once per M slice of a batch; the output
//    is written once), over the chip's bandwidth.
RooflineTerms roofline(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch) {
    const double a_bytes = in0_tile_bytes(p);
    const double b_bytes = in1_tile_bytes(p);
    const double engine_share = std::min(p.in0_tile_h, 8u) / 8.0;
    const double cycles_per_product = 2.0 * p.in0_tile_h * TILE_DIM * p.in1_tile_w *
                                      fidelity_multiplier(p.math_fidelity) / (hw.matmul_flops_per_cycle * engine_share);
    const double Kt = p.Kt;
    double products = 0;  // per core
    double received = 0;  // bytes per core
    double a_total = 0;   // tiles read from memory
    double b_total = 0;
    if (family == Family::Reuse) {
        const double cores = hw.grid.x * hw.grid.y;
        const double blocks = std::ceil(std::ceil(double(p.batch_a) * p.Mt / b.per_core_M) / cores);
        const double batches_per_block = std::max(1.0, double(b.per_core_M) / p.Mt);
        products = blocks * b.per_core_M * p.Nt * Kt;
        received = blocks * (b.per_core_M * Kt * a_bytes + batches_per_block * Kt * p.Nt * b_bytes);
        a_total = double(p.batch_a) * p.Mt * Kt;
        b_total = double(p.batch_b) * Kt * p.Nt * div_up(p.Mt, b.per_core_M);
    } else {
        // Unfused, the layout loops over the batch; in0 reuse (a broadcast A) keeps A resident across it
        const double loops = fuse_batch ? 1.0 : std::max(p.batch_a, p.batch_b);
        const double a_passes = double(b.per_core_N) / b.out_block_w;
        const double b_passes = double(b.per_core_M) / b.out_block_h;
        products = double(b.per_core_M) * b.per_core_N * Kt * loops;
        received = b.per_core_M * Kt * a_passes * (broadcasts_a(p) ? 1.0 : loops) * a_bytes +
                   b.per_core_N * Kt * b_passes * loops * b_bytes;
        a_total = double(p.batch_a) * p.Mt * Kt * a_passes;
        b_total = (p.batch_b > 1 ? double(p.batch_b) : loops) * Kt * p.Nt * b_passes;
    }
    const double out_total = double(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
    const double dram_bytes = (p.a.in_l1 ? 0.0 : a_total * a_bytes) + (p.b.in_l1 ? 0.0 : b_total * b_bytes) +
                              (p.out.in_l1 ? 0.0 : out_total * out_tile_bytes(p, p.out_format));
    return {products * cycles_per_product, received / hw.noc_bytes_per_cycle, dram_bytes / hw.dram_bytes_per_cycle};
}

std::optional<Estimate> RooflineEstimator::estimate(
    const MatmulDesc& p, const HardwareDesc& hw, const Candidate& c) const {
    return Estimate{
        .cycles = roofline(p, hw, c.family, c.blocking, c.fuse_batch).cycles(), .confidence = 0, .source = name()};
}

}  // namespace ttnn::operations::matmul::auto_config
