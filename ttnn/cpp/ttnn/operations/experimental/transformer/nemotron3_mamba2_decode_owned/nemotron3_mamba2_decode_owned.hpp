// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Mamba2 SSD decode step (Nemotron-3 Nano).
// Returns (ssm_state_out, y). ssm_state is mutated in place.
// See research/mm7_g1_mamba2_kernel_design.md §2 for the contract.
std::tuple<Tensor, Tensor> nemotron3_mamba2_decode_owned(
    const Tensor& x,
    const Tensor& z,
    const Tensor& dt,
    const Tensor& dt_bias,
    const Tensor& A_log,
    const Tensor& D,
    const Tensor& B_in,
    const Tensor& C_in,
    const Tensor& ssm_state,
    bool debug_fill = false,
    uint32_t debug_mode = 0,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_y = std::nullopt);

}  // namespace ttnn::experimental
