// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_perf_model.hpp"
#include "ttnn/operation.hpp"
#include <cmath>

namespace ttnn::operations::transformer::sdpa {

int compute_sdpa_ideal_cycles(
    uint32_t batch_size,
    uint32_t num_heads_q,
    uint32_t Sq,
    uint32_t Sk,
    uint32_t DH,
    uint32_t DV,
    bool is_causal,
    MathFidelity math_fidelity,
    int num_cores) {
    constexpr int64_t FLOPS_PER_FMA = 2;  // Each FMA is 2 FLOPS
    int64_t num_mul_adds = 0;

    // Q * K matmul: [B, NQH, Sq, DH] x [B, NKV, DH, Sk] -> [B, NQH, Sq, Sk]
    num_mul_adds += FLOPS_PER_FMA * DH * Sq * Sk * num_heads_q * batch_size;

    // attention * V matmul: [B, NQH, Sq, Sk] x [B, NKV, Sk, DV] -> [B, NQH, Sq, DV]
    num_mul_adds += FLOPS_PER_FMA * DV * Sq * Sk * num_heads_q * batch_size;

    // If causal, only half of the FMAs are actually performed
    if (is_causal) {
        num_mul_adds /= 2;
    }

    // Wormhole and Blackhole have identical matmul throughput per cycle
    const int tensix_mul_adds_per_cycle_lofi = 4096;

    // Ideal total cycles is ESTIMATED_FLOPS / IDEAL_THROUGHPUT
    return static_cast<int>(std::ceil(
        (static_cast<float>(num_mul_adds) / static_cast<float>(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        static_cast<float>(tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(math_fidelity))));
}

}  // namespace ttnn::operations::transformer::sdpa
