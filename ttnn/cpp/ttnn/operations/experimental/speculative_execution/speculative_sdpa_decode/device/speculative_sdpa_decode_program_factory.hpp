// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::experimental::speculative_execution::detail {

std::tuple<KernelHandle, KernelHandle> ccl_multi_core_with_workers(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const Tensor& output_tensor,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    CoreCoord ccl_core,
    uint32_t local_spec_result_input_ready_semaphore,
    uint32_t local_spec_result_input_ready_semaphore_wait_count,
    GlobalSemaphore& global_semaphore,
    uint32_t ccl_result_ready_semaphore_id,
    std::vector<uint32_t> result_ready_core_physical_xs,
    std::vector<uint32_t> result_ready_core_physical_ys);

operation::ProgramWithCallbacks speculative_sdpa_decode_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    std::optional<const Tensor> cur_pos_tensor,
    std::optional<const Tensor> page_table_tensor,
    std::optional<const Tensor> attn_mask,
    std::optional<const Tensor> priority_tensor,
    std::optional<const Tensor> other_priority_tensor,
    const Tensor& full_output_tensor,
    const Tensor& speculated_output_tensor,
    const Tensor& l2_dist_tensor,
    const Tensor& l2_norm_tensor,
    bool is_causal,
    const std::vector<uint32_t>& cur_pos_ids,
    std::optional<float> scale,
    std::optional<float> lambda,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    const uint32_t k_chunk_size,
    const uint32_t speculative_chunk_size,
    std::optional<bool> share_cache,
    // ccl related
    bool ccl_enabled,
    uint32_t num_devices,
    uint32_t device_index,
    ttnn::ccl::Topology topology,
    std::optional<GlobalSemaphore> global_semaphore,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device);

}  // namespace ttnn::operations::experimental::speculative_execution::detail
