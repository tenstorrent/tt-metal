// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::experimental {

// Datacopy-safe dtype pairs stay on prim::copy: all in-tree callers hit this path and the SFPU
// typecast has stricter validation (input/output memory layout must match, sharded page size must
// be L1-aligned, QSR rejects UINT16/UINT32 outputs). Unsafe pairs must go through the SFPU kernel
// since prim::copy only reinterprets bits for int<->float.
ttnn::Tensor typecast(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (ttnn::operations::data_movement::is_datacopy_safe_dtype_conversion(input_tensor.dtype(), dtype)) {
        return ttnn::prim::copy(
            input_tensor, output_mem_config.value_or(input_tensor.memory_config()), dtype, optional_output_tensor);
    }
    return ttnn::typecast(input_tensor, dtype, output_mem_config, optional_output_tensor);
}

}  // namespace ttnn::experimental
