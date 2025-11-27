// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction::sampling {

struct ExecuteSampling {
    static Tensor invoke(
        const Tensor& input_values_tensor,
        const Tensor& input_indices_tensor,
        const Tensor& k,
        const Tensor& p,
        const Tensor& temp,
        const std::optional<uint32_t>& seed = std::nullopt,
        const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::reduction::sampling

namespace ttnn {

constexpr auto sampling =
    ttnn::register_operation<"ttnn::sampling", ttnn::operations::reduction::sampling::ExecuteSampling>();

}  // namespace ttnn
