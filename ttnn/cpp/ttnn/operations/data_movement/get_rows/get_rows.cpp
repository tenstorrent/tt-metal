// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "get_rows.hpp"

#include "device/get_rows_device_operation.hpp"

namespace ttnn {

Tensor get_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // torch.index_select raises IndexError for out-of-range indices; TT metal kernels cannot check
    // before NoC-reading a row, so the bounds check happens here, on the host, before dispatch. The
    // index vector is tiny (one uint32/int32 per output row) next to the rows being gathered.
    const auto num_input_rows = input_tensor.logical_shape()[0];

    const auto host_indices = indices_tensor.cpu();
    if (indices_tensor.dtype() == DataType::INT32) {
        for (const int32_t index : host_indices.to_vector<int32_t>()) {
            TT_FATAL(
                index >= 0 && static_cast<uint32_t>(index) < num_input_rows,
                "get_rows: index {} out of range for an input with {} rows (torch.index_select semantics)",
                index,
                num_input_rows);
        }
    } else {
        for (const uint32_t index : host_indices.to_vector<uint32_t>()) {
            TT_FATAL(
                index < num_input_rows,
                "get_rows: index {} out of range for an input with {} rows (torch.index_select semantics)",
                index,
                num_input_rows);
        }
    }

    return ttnn::prim::get_rows(input_tensor, indices_tensor, memory_config, optional_output_tensor);
}

}  // namespace ttnn
