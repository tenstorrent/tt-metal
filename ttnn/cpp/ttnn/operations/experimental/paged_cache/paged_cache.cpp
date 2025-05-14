// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/paged_cache_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"

namespace ttnn {
namespace operations::experimental::paged_cache {

ttnn::Tensor PagedUpdateCacheOperation::invoke(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor = std::nullopt,
    const std::optional<bool> share_cache = std::nullopt,
    const std::optional<const Tensor>& page_table = std::nullopt,
    const uint32_t batch_offset = 0,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
    const bool share_cache_arg = share_cache.has_value() ? share_cache.value() : false;
    tt::tt_metal::operation::run(
        PagedUpdateCacheDeviceOperation{
            0,                               // .batch_idx_fallback (not used by UPDATE op type)
            std::nullopt,                    // .batch_idx_tensor_opt (not used by UPDATE op type)
            update_idxs,                     // .update_idxs
            batch_offset,                    // .batch_offset
            PagedUpdateCacheOpType::UPDATE,  // .op_type
            kernel_config_val,               // .compute_kernel_config
            share_cache_arg                  // .share_cache
        },
        {cache_tensor, input_tensor},
        {update_idxs_tensor, page_table});  // Optional inputs for UPDATE

    return cache_tensor;  // Updated cache tensor in-place
}

std::tuple<ttnn::Tensor, ttnn::Tensor> PagedFusedUpdateCacheOperation::invoke(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor = std::nullopt,
    const std::optional<bool> share_cache = std::nullopt,
    const std::optional<const Tensor>& page_table = std::nullopt,
    const uint32_t batch_offset = 0,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor1.device()->arch(), compute_kernel_config);
    const bool share_cache_arg = share_cache.has_value() ? share_cache.value() : false;
    tt::tt_metal::operation::run(
        PagedUpdateCacheDeviceOperation{
            0,                                     // .batch_idx_fallback (not used by FUSED_UPDATE op type)
            std::nullopt,                          // .batch_idx_tensor_opt (not used by FUSED_UPDATE op type)
            update_idxs,                           // .update_idxs
            batch_offset,                          // .batch_offset
            PagedUpdateCacheOpType::FUSED_UPDATE,  // .op_type
            kernel_config_val,                     // .compute_kernel_config
            share_cache_arg                        // .share_cache
        },
        {cache_tensor1, input_tensor1, cache_tensor2, input_tensor2},
        {update_idxs_tensor, page_table});  // Optional inputs for FUSED_UPDATE

    return {cache_tensor1, cache_tensor2};  // Updated cache tensors in-place
}

ttnn::Tensor PagedFillCacheOperation::invoke(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<const Tensor>& batch_idx_tensor,
    const uint32_t batch_idx_fallback,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);

    std::vector<std::optional<const Tensor>> optional_inputs_for_run;
    tt::tt_metal::operation::run(
        PagedUpdateCacheDeviceOperation{
            batch_idx_fallback,            // .batch_idx_fallback (used by FILL if tensor not present)
            batch_idx_tensor,              // .batch_idx_tensor_opt (used by FILL if present)
            {},                            // .update_idxs (empty for fill)
            0,                             // .batch_offset (0 for fill)
            PagedUpdateCacheOpType::FILL,  // .op_type
            kernel_config_val,             // .compute_kernel_config
            false  // .share_cache (false for fill, can be made a param if needed for future FILL variants)
        },
        {cache_tensor, input_tensor, page_table},  // Mandatory inputs for FILL
        {std::nullopt, std::nullopt});

    return cache_tensor;  // Updated cache tensor in-place
}

}  // namespace operations::experimental::paged_cache

}  // namespace ttnn
