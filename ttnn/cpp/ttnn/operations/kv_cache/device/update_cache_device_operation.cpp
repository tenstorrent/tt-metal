// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "update_cache_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
ttnn::operations::kv_cache::UpdateKVCacheOperation::tensor_return_value_t update_cache(
    const Tensor& cache,
    const Tensor& input,
    const uint32_t batch_idx,
    const uint32_t update_index,
    const uint32_t batch_offset,
    const UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::kv_cache::UpdateKVCacheOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .batch_idx = batch_idx,
            .update_idx = update_index,
            .batch_offset = batch_offset,
            .op_type = op_type,
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .cache = cache,
            .input = input});
}
}  // namespace ttnn::prim

namespace ttnn::operations::kv_cache {
}  // namespace ttnn::operations::kv_cache
