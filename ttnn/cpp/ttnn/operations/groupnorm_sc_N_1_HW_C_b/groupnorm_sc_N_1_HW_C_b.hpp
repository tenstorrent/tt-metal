// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// GroupNorm over an (N, 1, HW, C) channels-last activation.
//
//   input_tensor   DRAM interleaved (TILE or ROW_MAJOR; bfloat16, float32, bfloat8_b TILE) or L1 BLOCK_SHARDED
//                  (bfloat16, TILE or ROW_MAJOR)
//   num_groups     >= 1, divides C
//   gamma, beta    optional (1, 1, 1, C) affine; beta needs gamma
//   eps            > 0
//   in_place       BLOCK_SHARDED only: the result overwrites the input shard and the input is returned
//   output_tensor  preallocated output; must match the spec the op would allocate
//
// Refusals: std::invalid_argument (structural), UnsupportedAxisValue / ExcludedCell (support contract),
// PlanNotImplemented (no feasible plan).
Tensor groupnorm_sc_N_1_HW_C_b(
    const Tensor& input_tensor,
    int num_groups,
    const std::optional<Tensor>& gamma = std::nullopt,
    const std::optional<Tensor>& beta = std::nullopt,
    float eps = 1e-5f,
    bool in_place = false,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
