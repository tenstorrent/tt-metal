// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

struct ReduceScatter {
    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::optional<size_t> user_defined_num_workers;
    const std::optional<size_t> user_defined_num_buffers_per_channel;
    const std::optional<uint32_t> cluster_axis;
    const std::vector<IDevice*> devices;
    const distributed::MeshDevice* mesh_device = nullptr;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const MeshCoordinate& mesh_coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

namespace ccl {

namespace reduce_scatter_detail {
tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const Tensor& input_tensors,
    const Tensor& output_tensors,
    ttnn::operations::binary::BinaryOpType reduce_op,
    uint32_t scatter_split_dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    chip_id_t target_device_id,
    std::optional<chip_id_t> receiver_device_id,
    std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel);
}

};  // namespace ccl

namespace operations::ccl {

Tensor reduce_scatter(
    const Tensor& input_tensor,
    int32_t dim,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

Tensor reduce_scatter(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

std::vector<Tensor> reduce_scatter(
    const std::vector<ttnn::Tensor>& input_tensors,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

}  // namespace operations::ccl

};  // namespace ttnn
