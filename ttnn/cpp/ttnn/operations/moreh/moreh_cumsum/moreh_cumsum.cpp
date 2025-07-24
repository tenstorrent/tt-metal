// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cpp/ttnn/operations/reduction/cumsum/cumsum.hpp>
#include "moreh_cumsum.hpp"
namespace ttnn::operations::moreh::moreh_cumsum {
Tensor MorehCumsum::invoke(
    const Tensor& input,
    const int64_t dim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::cumsum(
        input, dim, input.dtype(), output, false, memory_config.has_value() ? *memory_config : input.memory_config());
}

Tensor MorehCumsumBackward::invoke(
    const Tensor& output_grad,
    const int64_t dim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::cumsum(
        output_grad,
        dim,
        output_grad.dtype(),
        input_grad,
        true,
        memory_config.has_value() ? *memory_config : output_grad.memory_config());
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
