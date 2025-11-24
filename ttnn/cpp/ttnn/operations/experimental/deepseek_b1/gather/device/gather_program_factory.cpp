// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/experimental/device.hpp>

using namespace tt::constants;

namespace ttnn::operations::experimental::deepseek_b1::gather {

// Mesh workload implementation - creates programs for all devices in the mesh
GatherDeviceOperation::GatherProgramFactory::cached_mesh_workload_t
GatherDeviceOperation::GatherProgramFactory::create_mesh_workload(
    const GatherDeviceOperation::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const GatherDeviceOperation::tensor_args_t& tensor_args,
    GatherDeviceOperation::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

// Create program for a specific mesh coordinate
GatherDeviceOperation::GatherProgramFactory::cached_program_t GatherDeviceOperation::GatherProgramFactory::create_at(
    const GatherDeviceOperation::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const GatherDeviceOperation::tensor_args_t& tensor_args,
    GatherDeviceOperation::tensor_return_value_t& tensor_return_value) {
    // For single-device or multi-device case, get the specific device
    auto mesh_device = tensor_args.input_tensor.device();
    tt::tt_metal::IDevice* target_device =
        mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.input_tensor.device();

    const Tensor& input = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get input and output shard specs
    const auto& input_shard_spec = input.memory_config().shard_spec().value();
    const auto& output_shard_spec = output_tensor.memory_config().shard_spec().value();

    // Get input cores (multiple cores)
    const auto& input_cores = input_shard_spec.grid;

    // Get output core (single core)
    const auto output_core = output_shard_spec.grid.ranges()[0].start_coord;
    auto noc_output_core = target_device->worker_core_from_logical_core(output_core);

    // Calculate data size from shard shape
    auto send_size = input.buffer()->aligned_size() / input_cores.num_cores();

    // Determine NOC routing for sender cores
    auto input_cores_vec = corerange_to_cores(
        input_cores, std::nullopt, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    std::vector<CoreCoord> noc_0_cores;
    std::vector<CoreCoord> noc_1_cores;

    if (operation_attributes.noc.has_value()) {
        if (*operation_attributes.noc == tt::tt_metal::NOC::NOC_0) {
            noc_0_cores = input_cores_vec;
        } else {
            noc_1_cores = input_cores_vec;
        }
    } else {
        for (const auto& core : input_cores_vec) {
            uint32_t noc_0_hop_distance = tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                target_device, core, output_core, tt::tt_metal::NOC::NOC_0);
            uint32_t noc_1_hop_distance = tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                target_device, core, output_core, tt::tt_metal::NOC::NOC_1);
            if (noc_0_hop_distance < noc_1_hop_distance) {
                noc_0_cores.push_back(core);
            } else {
                noc_1_cores.push_back(core);
            }
        }
    }

    CoreRangeSet noc0_cores_range = CoreRangeSet(noc_0_cores);
    CoreRangeSet noc1_cores_range = CoreRangeSet(noc_1_cores);

    // Get all cores for semaphore allocation
    CoreRangeSet all_cores = input_cores;
    all_cores = all_cores.merge(CoreRangeSet(CoreRange(output_core, output_core)));

    // Create semaphores
    const uint32_t noc0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    const uint32_t noc1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Create receiver kernel
    const std::string receiver_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gather/device/kernels/gather_receiver.cpp";
    std::vector<uint32_t> receiver_compile_time_args = {
        static_cast<uint32_t>(noc_0_cores.size()),
        static_cast<uint32_t>(noc_1_cores.size()),
        noc0_receiver_semaphore_id,
        noc1_receiver_semaphore_id,
    };

    tt::tt_metal::NOC receiver_noc = tt::tt_metal::NOC::NOC_0;
    if (noc0_cores_range.contains(output_core)) {
        receiver_noc = tt::tt_metal::NOC::NOC_1;
    } else {
        receiver_noc = tt::tt_metal::NOC::NOC_0;
    }

    tt::tt_metal::CreateKernel(
        program,
        receiver_kernel_path,
        CoreRangeSet(CoreRange(output_core, output_core)),
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = receiver_noc,
            .compile_args = receiver_compile_time_args});

    // Create sender kernels
    const std::string sender_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gather/device/kernels/gather_sender.cpp";
    std::vector<uint32_t> sender_compile_time_args = {
        noc_output_core.x,
        noc_output_core.y,
        send_size,
        0,  // semaphore (will be set per NOC)
    };

    tt::tt_metal::KernelHandle sender_noc0_kernel_id = 0, sender_noc1_kernel_id = 0;
    if (noc_0_cores.size() > 0) {
        sender_compile_time_args[3] = noc0_receiver_semaphore_id;
        sender_noc0_kernel_id = tt::tt_metal::CreateKernel(
            program,
            sender_kernel_path,
            noc0_cores_range,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
                .compile_args = sender_compile_time_args});
    }
    if (noc_1_cores.size() > 0) {
        sender_compile_time_args[3] = noc1_receiver_semaphore_id;
        sender_noc1_kernel_id = tt::tt_metal::CreateKernel(
            program,
            sender_kernel_path,
            noc1_cores_range,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
                .compile_args = sender_compile_time_args});
    }

    // Set runtime arguments for sender kernels
    for (uint32_t i = 0; i < input_cores_vec.size(); i++) {
        const auto& core = input_cores_vec[i];
        std::vector<uint32_t> sender_runtime_args = {
            input.buffer()->address(),
            output_tensor.buffer()->address(),
            i * send_size,
        };

        if (noc0_cores_range.contains(core)) {
            tt::tt_metal::SetRuntimeArgs(program, sender_noc0_kernel_id, core, sender_runtime_args);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, sender_noc1_kernel_id, core, sender_runtime_args);
        }
    }

    return {
        std::move(program),
        {.sender_noc0_kernel_id = sender_noc0_kernel_id,
         .sender_noc1_kernel_id = sender_noc1_kernel_id,
         .noc_0_cores = noc_0_cores,
         .noc_1_cores = noc_1_cores,
         .send_size = send_size}};
}

void GatherDeviceOperation::GatherProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_program,
    const GatherDeviceOperation::operation_attributes_t& operation_attributes,
    const GatherDeviceOperation::tensor_args_t& tensor_args,
    GatherDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_return_value;

    for (auto& [range, program] : cached_program.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());
        const auto& shared_vars = cached_program.shared_variables.at(range);

        const auto& sender_noc0_kernel_id = shared_vars.sender_noc0_kernel_id;
        const auto& sender_noc1_kernel_id = shared_vars.sender_noc1_kernel_id;
        const auto& noc_0_cores = shared_vars.noc_0_cores;
        const auto& noc_1_cores = shared_vars.noc_1_cores;

        // Update sender kernel runtime arguments
        auto& noc0_runtime_args_by_core = GetRuntimeArgs(program, sender_noc0_kernel_id);
        auto& noc1_runtime_args_by_core = GetRuntimeArgs(program, sender_noc1_kernel_id);
        for (const auto& core : noc_0_cores) {
            auto& noc0_runtime_args = noc0_runtime_args_by_core[core.x][core.y];
            noc0_runtime_args[0] = input.buffer()->address();
            noc0_runtime_args[1] = output.buffer()->address();
        }
        for (const auto& core : noc_1_cores) {
            auto& noc1_runtime_args = noc1_runtime_args_by_core[core.x][core.y];
            noc1_runtime_args[0] = input.buffer()->address();
            noc1_runtime_args[1] = output.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gather
