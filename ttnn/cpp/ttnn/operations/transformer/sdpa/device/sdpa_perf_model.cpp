// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_perf_model.hpp"
#include "ttnn/operation.hpp"
#include <cmath>

namespace ttnn::operations::transformer::sdpa {

namespace {

constexpr double kFlopsPerFma = 2.0;  // Each FMA is 2 FLOPS
// Wormhole and Blackhole have identical matmul throughput per cycle.
constexpr double kTensixMulAddsPerCycleLofi = 4096.0;

}  // namespace

int compute_sdpa_ideal_cycles_for_valid_pairs(
    uint32_t batch_size,
    uint32_t num_heads_q,
    double valid_pairs,
    uint32_t DH,
    uint32_t DV,
    tt::tt_metal::MathFidelity math_fidelity,
    int num_cores) {
    if (valid_pairs <= 0.0 || num_cores <= 0) {
        return 0;
    }

    const double fma_flops = kFlopsPerFma * valid_pairs * static_cast<double>(DH + DV) *
                             static_cast<double>(num_heads_q) * static_cast<double>(batch_size);

    // Ideal total cycles is ESTIMATED_FLOPS / IDEAL_THROUGHPUT.
    return static_cast<int>(std::ceil(
        (fma_flops / (static_cast<double>(num_cores) * kTensixMulAddsPerCycleLofi)) *
        static_cast<double>(tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(math_fidelity))));
}

int compute_sdpa_ideal_cycles(
    uint32_t batch_size,
    uint32_t num_heads_q,
    uint32_t Sq,
    uint32_t Sk,
    uint32_t DH,
    uint32_t DV,
    bool is_causal,
    tt::tt_metal::MathFidelity math_fidelity,
    int num_cores) {
    double valid_pairs = static_cast<double>(Sq) * static_cast<double>(Sk);

    // If causal, only half of the FMAs are actually performed.
    if (is_causal) {
        valid_pairs /= 2.0;
    }

    return compute_sdpa_ideal_cycles_for_valid_pairs(
        batch_size, num_heads_q, valid_pairs, DH, DV, math_fidelity, num_cores);
}

}  // namespace ttnn::operations::transformer::sdpa
