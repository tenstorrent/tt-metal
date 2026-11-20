// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::experimental {

// Route to the SFPU typecast; prim::copy only reinterprets bits for int<->float pairs (issue #58372).
ttnn::Tensor typecast(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::typecast(input_tensor, dtype, output_mem_config, optional_output_tensor);
}

}  // namespace ttnn::experimental
