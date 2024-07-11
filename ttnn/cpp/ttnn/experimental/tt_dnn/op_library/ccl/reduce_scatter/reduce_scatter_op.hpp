// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/experimental/tt_dnn/op_library/reduce/reduce_op.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace tt {
namespace tt_metal {

struct ReduceScatter {
    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const ttnn::utils::ccl::Topology topology;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor> &input_tensors,
    const uint32_t scatter_split_dim,
    ReduceOpMath reduce_op  = ReduceOpMath::SUM,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

namespace ccl {
namespace reduce_scatter_detail {
operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const std::vector<Tensor>& input_tensors,
    const std::vector<Tensor>& output_tensors,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
   ttnn::utils::ccl::Topology topology);
}
}; // namespace ccl

};  // namespace tt_metal
};  // namespace tt
