// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"

#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
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
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ReduceScatter::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology);
}

static ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum:
            return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("Reduce scatter only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}

namespace operations{
namespace ccl{
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const uint32_t scatter_dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [binary_op_type, scatter_dim, num_links, output_mem_config, topology, devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            bool is_ring = topology ==ttnn::ccl::Topology::Ring;

            const auto& input_tensor = input_tensors.at(0);
            uint32_t num_devices = devices.size();
            uint32_t device_index = 0; // Initialize device index
            std::optional<chip_id_t> receiver_device_id = std::nullopt; // Initialize receiver device ID
            std::optional<chip_id_t> sender_device_id = std::nullopt; // Initialize sender device ID
            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices.at(i) == input_tensor.device()) {
                    bool is_last_chip_in_clockwise_direction = is_ring ? false : i == (input_tensors.size() - 1);
                    bool is_last_chip_in_counter_clockwise_direction = is_ring ? false : i == 0;
                    device_index = i;
                    receiver_device_id = devices.at((i + 1) % num_devices)->id(); // Next device in the ring
                    sender_device_id = devices.at((i + num_devices - 1) % num_devices)->id(); // Previous device in the ring
                    break;
                }
            }
            TT_FATAL(receiver_device_id != std::nullopt || sender_device_id != std::nullopt, "Error in reduce scatter op setup");

            return operation::run(
                ttnn::ReduceScatter{
                    binary_op_type,
                    scatter_dim,
                    num_links,
                    num_devices,
                    device_index,
                    receiver_device_id,
                    sender_device_id,
                    output_mem_config,
                    topology},
                {input_tensor});
        },
     {input_tensor},
     output_tensors);
    return output_tensors.at(0);
}

} // namespace ccl
} // namespace operations

};  // namespace ttnn
