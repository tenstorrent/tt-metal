// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::tilize_untilize {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteTilizeUntilize::invoke(
    const ttnn::Tensor& input,
    std::optional<MemoryConfig> output_memory_config,
    std::optional<DataType> output_dtype,
    const std::optional<MemoryConfig>& memory_config) {
    // Call the primitive device operation
    // The registered operation handles execution automatically
    return ttnn::prim::tilize_untilize(input, output_memory_config, output_dtype, memory_config);
}

}  // namespace ttnn::operations::reduction::tilize_untilize
