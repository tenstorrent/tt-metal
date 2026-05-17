// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    // tt-xla #4539 fix proposal: optional host-precomputed noise tensor.
    // Shape [32], bf16, ROW_MAJOR — one value per core (same convention as
    // k, p, temp). When provided, the writer kernel uses this value instead
    // of the kernel-side RNG output.
    const std::optional<Tensor>& noise = std::nullopt);

}  // namespace ttnn
