// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice.hpp"
#include "device/nlp_kv_cache_load_slice_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    uint32_t seq_len_start,
    uint32_t seq_len_end,
    [[maybe_unused]] const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType =
        operations::experimental::transformer::nlp_kv_cache_load_slice::NlpKVCacheLoadSliceDeviceOperation;

    auto input_tensor_shape = input_tensor.padded_shape();
    auto dim0 = input_tensor_shape[0];
    auto dim1 = input_tensor_shape[1];
    auto head_dim = input_tensor_shape[3];

    ttnn::Shape output_tensor_start({
        0,
        0,
        seq_len_start,
        0,
    });

    ttnn::Shape output_tensor_end({
        dim0 - 1,
        dim1 - 1,
        seq_len_end - 1,
        head_dim - 1,
    });

    auto operation_attributes = OperationType::operation_attributes_t{output_tensor_start, output_tensor_end};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, preallocated_output};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
