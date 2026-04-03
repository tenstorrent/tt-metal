// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::transformer::sdpa {

// Compute ideal clock cycles for SDPA operations based on FLOP count and hardware capabilities.
// This helper is shared across different SDPA operation variants for consistent performance modeling.
//
// Parameters:
//   batch_size   - Batch dimension
//   num_heads_q  - Number of query heads (determines matmul shape)
//   Sq           - Query sequence length (local for distributed variants)
//   Sk           - Key sequence length
//   DH           - Head dimension for Q/K
//   DV           - Head dimension for V (output)
//   is_causal    - If true, only half of FMAs are performed due to causal masking
//   math_fidelity - Compute fidelity affecting cycles per operation
//   num_cores    - Number of compute cores for parallelization
//
// Returns ideal_compute_cycles suitable for OpPerformanceModelGeneral constructor.
int compute_sdpa_ideal_cycles(
    uint32_t batch_size,
    uint32_t num_heads_q,
    uint32_t Sq,
    uint32_t Sk,
    uint32_t DH,
    uint32_t DV,
    bool is_causal,
    MathFidelity math_fidelity,
    int num_cores);

}  // namespace ttnn::operations::transformer::sdpa
