// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_reduce/device/all_reduce_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <cstdint>

namespace ttnn {

void AllReduce::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(!t.is_sharded(), "Sharded tensors are not supported for all reduce currently");
    }
}

std::vector<ttnn::SimpleShape> AllReduce::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_logical_shape();
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllReduce::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks AllReduce::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        0,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

static ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum:
            return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("All reduce only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}

namespace operations{
namespace experimental{
namespace ccl{
Tensor all_reduce(
    const Tensor& input_tensor,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");
    TT_FATAL(topology == ttnn::ccl::Topology::Ring, "All Reduce op is currently supported only on Ring topology");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [binary_op_type, num_links, output_mem_config, topology, devices, user_defined_num_workers, user_defined_num_buffers_per_channel](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            TT_FATAL(input_tensors.size() >= 1, "All reduce op expects an input tensor but it received none");
            bool is_linear = topology == ttnn::ccl::Topology::Linear;

            const auto& input_tensor = input_tensors.at(0);
            uint32_t num_devices = devices.size();
            uint32_t device_index = 0; // Initialize device index
            std::optional<chip_id_t> receiver_device_id = std::nullopt; // Initialize receiver device ID
            std::optional<chip_id_t> sender_device_id = std::nullopt; // Initialize sender device ID
            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices.at(i) == input_tensor.device()) {

                    bool is_last_chip_in_clockwise_direction = is_linear && i == (num_devices - 1);
                    bool is_last_chip_in_counter_clockwise_direction = is_linear && i == 0;
                    device_index = i;
                    receiver_device_id = is_last_chip_in_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at((i + 1) % num_devices)->id());
                    sender_device_id = is_last_chip_in_counter_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at((i + num_devices - 1) % num_devices)->id());
                    break;
                }
            }
            TT_FATAL(receiver_device_id != std::nullopt || sender_device_id != std::nullopt, "Error in all reduce op setup");

            const auto& gathered_tensor = operation::run(
                create_all_gather_struct(input_tensor, 0, num_links, output_mem_config, user_defined_num_workers, user_defined_num_buffers_per_channel, devices, topology),
                {input_tensor});

            return {ttnn::sum(gathered_tensor.at(0))};
            },
     {input_tensor},
     output_tensors);

    return output_tensors.at(0);
}

} // namespace ccl
} // namespace experimental
} // namespace operations

};  // namespace ttnn
