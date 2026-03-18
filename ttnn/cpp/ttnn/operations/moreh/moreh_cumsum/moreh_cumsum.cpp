// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cpp/ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp>
#include "moreh_cumsum.hpp"

namespace ttnn {

Tensor moreh_cumsum(
    const Tensor& input,
    const int64_t dim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::cumsum(
        input, dim, input.dtype(), false, output, memory_config.has_value() ? *memory_config : input.memory_config());
}

Tensor moreh_cumsum_backward(
    const Tensor& output_grad,
    const int64_t dim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::cumsum(
        output_grad,
        dim,
        output_grad.dtype(),
        true,
        input_grad,
        memory_config.has_value() ? *memory_config : output_grad.memory_config());
}

}  // namespace ttnn
