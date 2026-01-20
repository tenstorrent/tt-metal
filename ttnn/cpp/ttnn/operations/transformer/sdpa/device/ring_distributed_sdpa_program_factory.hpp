// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/device_operation.hpp"

#include "ring_distributed_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

struct RingDistributedSdpaSharedVariables {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
};

struct RingDistributedSdpaMeshWorkloadFactory {
    using shared_variables_t = RingDistributedSdpaSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RingDistributedSDPAParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingDistributedSDPAInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingDistributedSDPAParams& operation_attributes,
        const RingDistributedSDPAInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const Tensor& input_tensor_q,
        const Tensor& input_tensor_k,
        const Tensor& input_tensor_v,
        const Tensor& output_tensor,
        uint32_t ring_size,
        uint32_t ring_id,
        std::optional<float> scale,
        std::size_t q_chunk_size,
        std::size_t k_chunk_size,
        DeviceComputeKernelConfig compute_kernel_config,
        std::optional<operations::transformer::SDPAProgramConfig> program_config,
        const std::optional<Tensor>& page_table = std::nullopt,
        std::optional<int64_t> chunk_start_idx = std::nullopt);
};

}  // namespace ttnn::prim
