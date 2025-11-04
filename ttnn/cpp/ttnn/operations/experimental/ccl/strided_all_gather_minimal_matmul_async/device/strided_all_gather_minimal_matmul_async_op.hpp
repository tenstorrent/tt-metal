// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn {

struct StridedAllGatherMinimalMatmulAsync {
    /* All Gather Params */
    const ttnn::StridedAllGatherAsync strided_all_gather_async_struct;

    /* Matmul Params */
    const operations::experimental::minimal_matmul::MinimalMatmulOp matmul_struct;

    /* Fusion Params */
    const CoreCoord all_gather_core_grid_offset;

    /* Physical Devices this op runs on*/
    std::vector<IDevice*> devices;

    /* General */
    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "all_gather_core_grid_offset");
    auto attribute_values() const {
        return std::forward_as_tuple(this->matmul_struct, this->all_gather_core_grid_offset);
    }
};

namespace ccl {
namespace strided_all_gather_minimal_matmul_async_detail {
StridedAllGatherMinimalMatmulAsync create_strided_all_gather_minimal_matmul_async_struct(
    const ttnn::StridedAllGatherAsync& strided_all_gather_struct_input,
    const operations::experimental::minimal_matmul::MinimalMatmulOp& matmul_struct_input,
    CoreCoord all_gather_core_grid_offset,
    const std::vector<IDevice*>& devices);
}  // namespace strided_all_gather_minimal_matmul_async_detail
}  // namespace ccl

tt::tt_metal::operation::ProgramWithCallbacks strided_all_gather_minimal_matmul_async_program(
    /* General Params */
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    IDevice* target_device,
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> tiles_per_chunk,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,

    /* Matmul Params */
    std::optional<const Tensor> bias,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    operations::experimental::minimal_matmul::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config);

namespace operations {
namespace experimental {
namespace ccl {

std::vector<Tensor> strided_all_gather_minimal_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    CoreCoord strided_all_gather_core_grid_offset,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config_ag = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
    std::optional<operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    const std::optional<const operations::experimental::minimal_matmul::MinimalMatmulConfig> config = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<uint32_t> tiles_per_chunk = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
