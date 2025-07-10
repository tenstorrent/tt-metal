// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "paged_fused_update_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::detail {

using namespace tt::constants;
using namespace tt;

bool enable_fp32_dest_acc(
    const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

operation::ProgramWithCallbacks paged_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::optional<const Tensor>& update_idxs_tensor,
    const std::optional<const Tensor>& page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache) {
    // if input 1, input 2 are tiled call tiled program factory
    if (input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE) {
        return paged_tiled_fused_update_cache_multi_core(
            cache_tensor1,
            input_tensor1,
            cache_tensor2,
            input_tensor2,
            update_idxs_tensor,
            page_table,
            update_idxs,
            batch_offset,
            compute_kernel_config,
            share_cache);
    } else if (input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR) {
        return paged_row_major_fused_update_cache_multi_core(
            cache_tensor1,
            input_tensor1,
            cache_tensor2,
            input_tensor2,
            update_idxs_tensor,
            page_table,
            update_idxs,
            batch_offset,
            compute_kernel_config,
            share_cache);
    } else {
        TT_FATAL(false, "Error: input tensor1 and input tensor2 must be either both tiled or both row-major");
    }
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
