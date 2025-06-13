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
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn {

struct MatmulReduceScatterAsync {
    /* All Gather Params */
    const ttnn::ReduceScatterMinimalAsync reduce_scatter_minimal_async_struct;

    /* Matmul Params */
    const operations::matmul::Matmul matmul_struct;

    /* Fusion Params */
    const CoreCoord reduce_scatter_core_grid_offset;

    /* Physical Devices this op runs on*/
    std::vector<IDevice*> devices;

    /* General */
    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
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

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "reduce_scatter_core_grid_offset");
    auto attribute_values() const {
        return std::forward_as_tuple(this->matmul_struct, this->reduce_scatter_core_grid_offset);
    }
};

namespace ccl {
namespace matmul_reduce_scatter_async_detail {
MatmulReduceScatterAsync create_matmul_reduce_scatter_async_struct(
    const ttnn::ReduceScatterMinimalAsync& reduce_scatter_minimal_struct_input,
    const operations::matmul::Matmul& matmul_struct_input,
    const CoreCoord reduce_scatter_core_grid_offset,
    const std::vector<IDevice*>& devices);
}  // namespace matmul_reduce_scatter_async_detail
}  // namespace ccl

tt::tt_metal::operation::ProgramWithCallbacks matmul_reduce_scatter_async_multi_core_with_workers(
    /* General Params */
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_tensor,
    Tensor& reduce_scatter_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out);

namespace operations {
namespace experimental {
namespace ccl {

std::vector<Tensor> matmul_reduce_scatter_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<const Tensor>& bias = std::nullopt,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config_rs = std::nullopt,
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
    const bool transpose_a = false,
    const bool transpose_b = false,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
