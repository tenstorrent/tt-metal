// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "tt_metal/host_api.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"


namespace ttnn {

void ReduceScatter::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size");
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size");
    }
}

std::vector<tt::tt_metal::Shape> ReduceScatter::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_legacy_shape();
    TT_ASSERT(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size");
    shape[this->scatter_dim] /= this->ring_size;
    return std::vector<tt::tt_metal::Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> ReduceScatter::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(false, "Sharded output is not supported for ReduceScatter");
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks ReduceScatter::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors,
        output_tensors,
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology);
}

static ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(ReduceOpMath reduce_op) {
    switch (reduce_op) {
        case ReduceOpMath::SUM: return ttnn::operations::binary::BinaryOpType::ADD;

        default: TT_FATAL("Reduce scatter only support reduce_op_type SUM"); return ttnn::operations::binary::BinaryOpType::ADD;
    }
}

namespace operations{
namespace ccl{
std::vector<Tensor> reduce_scatter_impl(
    const std::vector<Tensor>& input_tensors,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    const uint32_t scatter_dim,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const ttnn::ccl::Topology topology) {
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    std::vector<ReduceScatter> ops;
    ops.reserve(input_tensors.size());
    bool is_ring = topology ==ttnn::ccl::Topology::Ring;
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        bool is_last_chip_in_clockwise_direction = is_ring ? false : i == (input_tensors.size() - 1);
        bool is_last_chip_in_counter_clockwise_direction = is_ring ? false : i == 0;

        std::optional<chip_id_t> receiver_device_id =
            is_last_chip_in_clockwise_direction
                ? std::nullopt
                : std::optional<chip_id_t>(input_tensors[(i + 1) % input_tensors.size()].device()->id());
        std::optional<chip_id_t> sender_device_id =
            is_last_chip_in_counter_clockwise_direction
                ? std::nullopt
                : std::optional<chip_id_t>(input_tensors[i == 0 ? input_tensors.size() - 1 : i - 1].device()->id());
        ops.emplace_back(ttnn::ReduceScatter{
            binary_op_type,
            scatter_dim,
            num_links,
            static_cast<uint32_t>(input_tensors.size()),
            i,
            receiver_device_id,
            sender_device_id,
            output_mem_config,
            topology});
        output_tensors.push_back(operation::run(ops[i], {input_tensors.at(i)}).at(0));
    }
    return output_tensors;
}

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    const uint32_t scatter_dim,
    ReduceOpMath math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    return reduce_scatter_impl(
        input_tensors, binary_op_type, scatter_dim, num_links, output_mem_config,ttnn::ccl::Topology::Ring);
}
} // namespace ccl
} // namespace operations

};  // namespace ttnn
