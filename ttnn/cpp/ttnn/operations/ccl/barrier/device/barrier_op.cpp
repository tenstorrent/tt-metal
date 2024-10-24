// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/barrier/device/barrier_op.hpp"
#include "tt_metal/host_api.hpp"

#include <cstdint>

namespace ttnn {

void Barrier::validate(const std::vector<Tensor>& input_tensors) const {
    //Validate the input tensor
    TT_FATAL(this->topology == ccl::Topology::Ring, "We currently only support Ring topologies on this OP");
}

std::vector<SimpleShape> Barrier::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    //For a Barrier the output shape should match the input
    SimpleShape shape = input_tensors[0].get_logical_shape();
    return std::vector<SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> Barrier::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    //Tensor is unmodified, return what was passed in
    return input_tensors;
}

operation::ProgramWithCallbacks Barrier::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::barrier::detail::barrier_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->is_starting_core,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology);
}

namespace operations::ccl{

Tensor barrier(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology)
{
    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [output_mem_config, topology, devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            bool is_linear = (topology == ttnn::ccl::Topology::Linear);

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
            return operation::run(
                ttnn::Barrier{
                    device_index == 0,
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

} //namespace operations::ccl
} // namespace ttnn
