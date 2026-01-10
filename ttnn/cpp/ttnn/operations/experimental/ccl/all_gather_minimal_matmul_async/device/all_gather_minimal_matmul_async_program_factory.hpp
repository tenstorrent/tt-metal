// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

namespace helpers {
void override_program_parameters(
    const ttnn::operations::experimental::all_gather_minimal_matmul_async::
        all_gather_minimal_matmul_async_override_variables_t& override_variables,
    const void* operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);
}

namespace detail {
ttnn::operations::experimental::all_gather_minimal_matmul_async::all_gather_minimal_matmul_async_override_variables_t
all_gather_minimal_matmul_async_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const AllGatherMinimalMatmulAsyncConfig>& config,
    const Tensor& mm_output_tensor,
    const Tensor& ag_output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const uint32_t chunks_per_sync,
    const uint32_t num_workers_per_direction,
    const uint32_t num_buffers_per_channel);

tt::tt_metal::operation::ProgramWithCallbacks all_gather_minimal_matmul_async_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const AllGatherMinimalMatmulAsyncConfig>& config,
    const Tensor& mm_output_tensor,
    const Tensor& ag_output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const uint32_t chunks_per_sync,
    const uint32_t num_workers_per_direction,
    const uint32_t num_buffers_per_channel);

}  // namespace detail
}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async
