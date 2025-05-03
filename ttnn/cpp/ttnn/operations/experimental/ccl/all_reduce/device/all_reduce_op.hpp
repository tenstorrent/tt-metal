// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
namespace ttnn {

enum class AllReduceStrategy {
    AllGatherLocalReduce,
    ReduceScatterAllGather,
    // Fused,
    Invalid
};

struct AllReduce {
    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::optional<size_t> user_defined_num_workers;
    const std::optional<size_t> user_defined_num_buffers_per_channel;
    const std::vector<IDevice*> devices;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

namespace operations {
namespace experimental {
namespace ccl {
Tensor all_reduce(
    const Tensor& input_tensor,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);
std::vector<Tensor> all_reduce(
    const std::vector<Tensor>& input_tensors,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);
}  // namespace ccl
}  // namespace experimental
}  // namespace operations

};  // namespace ttnn
