// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
namespace ttnn {

struct AllReduce {
    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::optional<size_t> user_defined_num_workers;
    const std::optional<size_t> user_defined_num_buffers_per_channel;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};


namespace operations{
namespace experimental{
namespace ccl{
    Tensor all_reduce(
    const Tensor &input_tensor,
    const uint32_t scatter_split_dim,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);
} // namespace ccl
} // namespace experimental
} // namespace operations


};  // namespace ttnn
