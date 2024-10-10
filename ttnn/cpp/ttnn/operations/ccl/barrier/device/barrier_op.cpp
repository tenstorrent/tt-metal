// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//There are two parts to barrier_op.cpp. First we want to add the requred functions
//to struct Barrier. Then we will generate the function referenced in the 
//Invoke function which uses a Barrier class variable

//Required: validate, compute_output_shapes, create_output_tensors, create_program

#include "ttnn/operations/ccl/barrier/device/barrier_op.hpp"
#include "tt_metal/host_api.hpp"

#include <cstdint>

namespace ttnn {

void Barrier::validate(const std::vector<Tensor>& input_tensors) const {
    //Create conditions which check that the input is of a correct type
    //Use TT_FATAL(condition_that_fails,"failure message")
    //for (auto const& t : input_tensors) {
        //This will run for all the input tensors
    //    ;
    //}
    TT_FATAL(this->topology == ccl::Topology::Ring, "We currently only support Ring topologies on this OP");
}

std::vector<tt::tt_metal::LegacyShape> Barrier::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    //For a Barrier the output shape should match the input
    auto shape = input_tensors[0].get_legacy_shape();
    return std::vector<tt::tt_metal::LegacyShape>(input_tensors.size(), shape);
}

std::vector<Tensor> Barrier::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    //I am not modifying the tensor in this function so I will just return what I was given
    return input_tensors;
}

operation::ProgramWithCallbacks Barrier::create_program(
    //The create program function which calls barrier_with_workers defined in device/host/barrier_full_worker_grid.cpp
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::barrier_detail::barrier_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->is_starting_core,
        this->num_samples,
        this->max_concurrent_samples,
        this->sample_page_size,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology);
}

namespace operations{

namespace ccl{
Tensor barrier(
    const Tensor& input_tensor,
    const uint32_t num_samples,
    const uint32_t max_concurrent_samples,
    const uint32_t sample_page_size,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology)
{
    //Host function to launch the OP called from Invoke in barrier.cpp

    //Get the workers
    auto devices = input_tensor.get_workers();

    //Split the job up between the tensors
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    
    //Define the launch_op function
    operation::launch_op(
        [num_samples,max_concurrent_samples,sample_page_size,output_mem_config, topology, devices](
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
                    num_samples,
                    max_concurrent_samples,
                    sample_page_size,
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
    //Return the first output tensor
    return output_tensors.at(0);
}
    
} //namespace ccl end

} //namespace operations end
} // namespace ttnn end