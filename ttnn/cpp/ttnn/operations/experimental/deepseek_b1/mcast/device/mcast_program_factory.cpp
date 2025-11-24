// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_device_operation.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

using namespace tt::constants;

namespace ttnn::operations::experimental::deepseek_b1::mcast {

McastDeviceOperation::McastProgramFactory::cached_program_t McastDeviceOperation::McastProgramFactory::create(
    const McastDeviceOperation::operation_attributes_t& operation_attributes,
    const McastDeviceOperation::tensor_args_t& tensor_args,
    McastDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get input and output shard specs
    const auto& input_shard_spec = input.memory_config().shard_spec().value();
    const auto& output_shard_spec = output_tensor.memory_config().shard_spec().value();

    // Get input core (single core)
    const auto input_core = input_shard_spec.grid.ranges()[0].start_coord;

    // Get output cores (multiple cores)
    const auto& output_cores = output_shard_spec.grid;
    const auto& output_core_range = output_cores.ranges()[0];

    tt::tt_metal::IDevice* device = input.device();
    auto mcast_dest_noc_start_core = device->worker_core_from_logical_core(output_core_range.start_coord);
    auto mcast_dest_noc_end_core = device->worker_core_from_logical_core(output_core_range.end_coord);

    // Calculate data size from shard shape
    // Shard shape is in elements [height, width]
    auto total_size = input.buffer()->aligned_size();

    // Get all cores (input + output) for semaphore allocation
    CoreRangeSet all_cores = output_cores;
    all_cores = all_cores.merge(CoreRangeSet(CoreRange(input_core, input_core)));

    // Create semaphores for multicast synchronization
    auto data_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    auto data_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Create sender kernel on input core
    const std::string sender_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/mcast/device/kernels/mcast_sender.cpp";

    bool is_part_of_receiver_grid = output_cores.contains(input_core);
    bool loopback = is_part_of_receiver_grid;

    // Named compile time args for MCAST (persistent multicast sender setup)
    std::unordered_map<std::string, uint32_t> sender_named_compile_args = {
        // MCAST: DEFINE_PERSISTENT_MCAST_SENDER_VARS
        {"mcast_dest_noc_start_x", mcast_dest_noc_start_core.x},
        {"mcast_dest_noc_start_y", mcast_dest_noc_start_core.y},
        {"mcast_dest_noc_end_x", mcast_dest_noc_end_core.x},
        {"mcast_dest_noc_end_y", mcast_dest_noc_end_core.y},
        {"mcast_num_cores", output_cores.num_cores()},
        {"mcast_loopback", loopback},
        {"mcast_is_part_of_receiver_grid", is_part_of_receiver_grid},
        {"mcast_data_sender_semaphore", data_sender_semaphore_id},
        {"mcast_data_receiver_semaphore", data_receiver_semaphore_id},
        // MCAST0: DEFINE_MCAST_SENDER_VARS
        {"mcast0_num_cores", output_cores.num_cores()},
        {"mcast0_data_size_bytes", total_size},
    };

    tt::tt_metal::KernelHandle sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_kernel_path,
        CoreRangeSet(CoreRange(input_core, input_core)),
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = operation_attributes.noc,
            .named_compile_args = sender_named_compile_args});

    // Create receiver kernels on output cores
    const std::string receiver_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/mcast/device/kernels/mcast_receiver.cpp";
    std::vector<uint32_t> receiver_compile_time_args = {data_receiver_semaphore_id};

    tt::tt_metal::NOC receiver_noc =
        operation_attributes.noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    tt::tt_metal::CreateKernel(
        program,
        receiver_kernel_path,
        output_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = receiver_noc,
            .compile_args = receiver_compile_time_args});

    // Set runtime arguments for sender kernel
    // Runtime args: [input_data_addr, mcast_receiver_data_addr]
    std::vector<uint32_t> sender_runtime_args = {
        input.buffer()->address(),
        output_tensor.buffer()->address(),
    };
    tt::tt_metal::SetRuntimeArgs(program, sender_kernel_id, input_core, sender_runtime_args);

    return {std::move(program), {.sender_kernel_id = sender_kernel_id, .input_core = input_core}};
}

void McastDeviceOperation::McastProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const McastDeviceOperation::operation_attributes_t& operation_attributes,
    const McastDeviceOperation::tensor_args_t& tensor_args,
    McastDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_return_value;

    auto& program = cached_program.program;
    const auto& sender_kernel_id = cached_program.shared_variables.sender_kernel_id;
    const auto& input_core = cached_program.shared_variables.input_core;

    // Update sender kernel runtime arguments
    auto& sender_runtime_args = GetRuntimeArgs(program, sender_kernel_id, input_core);
    sender_runtime_args[0] = input.buffer()->address();
    sender_runtime_args[1] = output.buffer()->address();
}

}  // namespace ttnn::operations::experimental::deepseek_b1::mcast
