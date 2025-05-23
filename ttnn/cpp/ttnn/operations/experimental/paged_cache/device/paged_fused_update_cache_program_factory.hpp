// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::paged_cache::detail {

tt::tt_metal::operation::ProgramWithCallbacks paged_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::optional<const Tensor>& update_idxs_tensor,
    const std::optional<const Tensor>& page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache);

tt::tt_metal::operation::ProgramWithCallbacks paged_tiled_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    std::optional<const Tensor> update_idxs_tensor,
    std::optional<const Tensor> page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache);

tt::tt_metal::operation::ProgramWithCallbacks paged_row_major_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    std::optional<const Tensor> update_idxs_tensor,
    std::optional<const Tensor> page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache);
}  // namespace ttnn::operations::experimental::paged_cache::detail
