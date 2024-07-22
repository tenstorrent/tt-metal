/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

[[maybe_unused]] std::vector<std::variant<std::monostate, Tensor, char *>> moreh_bmm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    std::optional<std::reference_wrapper<const Tensor>> input_grad = std::nullopt,
    std::optional<std::reference_wrapper<const Tensor>> mat2_grad = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
