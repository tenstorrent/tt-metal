// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace ttnn::prim {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
// Helper function to compute core grid from device and operation attributes
CoreRangeSet compute_core_grid(
    const ManualSeedParams& operation_attributes, const IDevice* device, uint32_t& out_num_cores) {
    // Get device core grid
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    out_num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(out_num_cores, compute_with_storage_grid_size, true);

    // Override core grid if sub_core_grids is provided in operation attributes
    if (operation_attributes.sub_core_grids.has_value()) {
        core_grid = operation_attributes.sub_core_grids.value();
    }

    return core_grid;
}

// Helper function to describe a dataflow buffer holding a single tile-sized entry laid out like a tensor
DataflowBufferSpec make_tensor_dataflow_buffer(DFBSpecName unique_id, const MeshTensor& tensor) {
    // Dataflow buffer config
    const tt::DataFormat dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor.tensor_spec().data_type());
    const uint32_t tensor_tile_size = tensor.tensor_spec().tile().get_tile_size(dfb_data_format);

    return DataflowBufferSpec{
        .unique_id = std::move(unique_id),
        .entry_size = tensor_tile_size,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };
}

}  // anonymous namespace

ttnn::device_operation::ProgramArtifacts ManualSeedSingleSeedToAllCoresProgramFactory::create_program_artifacts(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& /*tensor_args*/, Tensor& /*output_tensor*/) {
    const KernelSpecName COMPUTE{"compute"};

    // Calculate core grid
    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);

    // Create compute kernel
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_path,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args = {{"seed", operation_attributes.seeds.value_or(0)}},
        .hw_config = ComputeHardwareConfig{},
    };

    ProgramSpec spec{
        .name = "manual_seed_single_seed_to_all_cores",
        .kernels = {compute},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {COMPUTE},
            .target_nodes = core_grid,
        }},
    };

    // The compute kernel takes no runtime arguments, so it needs no ProgramRunArgs entry.
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec)};
}

ttnn::device_operation::ProgramArtifacts ManualSeedSingleSeedSingleCoreProgramFactory::create_program_artifacts(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& /*tensor_args*/, Tensor& /*output_tensor*/) {
    const KernelSpecName COMPUTE{"compute"};

    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const auto& cores = corerange_to_cores(core_grid, num_cores, true);
    const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));
    CoreRangeSet chosen_core_ranges{CoreRange(core_chosen, core_chosen)};

    // Create compute kernel
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_path,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args = {{"seed", operation_attributes.seeds.value_or(0)}},
        .hw_config = ComputeHardwareConfig{},
    };

    ProgramSpec spec{
        .name = "manual_seed_single_seed_single_core",
        .kernels = {compute},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {COMPUTE},
            .target_nodes = chosen_core_ranges,
        }},
    };

    // The compute kernel takes no runtime arguments, so it needs no ProgramRunArgs entry.
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec)};
}

ttnn::device_operation::ProgramArtifacts ManualSeedSingleSeedSetCoresProgramFactory::create_program_artifacts(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& tensor_args, Tensor& /*output_tensor*/) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName USER_IDS_DFB{"user_ids"};
    const DFBSpecName KERNEL_COMMUNICATION_DFB{"kernel_communication"};
    const TensorParamName USER_IDS_TENSOR{"user_ids"};

    // Safety check
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSingleSeedSetCoresProgramFactory");

    uint32_t num_cores{};
    const CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Tensor config info
    const auto& user_ids_mesh = tensor_args.user_ids.value().mesh_tensor();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_mesh.logical_volume());

    // Create core kernels
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
        "reader_manual_seed_read_user_id.cpp";
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_single_seed_receive_user_id.cpp";

    // Create reader kernel
    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_path,
        .dfb_bindings =
            {// The user_ids buffer is a landing area for one NoC read: the reader fills it and then reads
             // the landed ids straight back out. No other kernel touches it, so the reader holds both ends.
             DFBBinding{
                 .dfb_spec_name = USER_IDS_DFB,
                 .accessor_name = "user_ids",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = USER_IDS_DFB,
                 .accessor_name = "user_ids",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             // The reader writes the match flag the compute kernel waits on.
             DFBBinding{
                 .dfb_spec_name = KERNEL_COMMUNICATION_DFB,
                 .accessor_name = "kernel_communication",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = USER_IDS_TENSOR,
            .accessor_name = "user_ids",
        }},
        .compile_time_args = {{"number_of_ids", number_of_ids}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"core_id", core_id}});
    }

    // Create compute kernel
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = compute_kernel_path,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = KERNEL_COMMUNICATION_DFB,
            .accessor_name = "kernel_communication",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .compile_time_args = {{"seed", operation_attributes.seeds.value_or(0)}},
        .hw_config = ComputeHardwareConfig{},
    };

    ProgramSpec spec{
        .name = "manual_seed_single_seed_set_cores",
        .kernels = {reader, compute},
        .dataflow_buffers =
            {make_tensor_dataflow_buffer(USER_IDS_DFB, user_ids_mesh),
             make_tensor_dataflow_buffer(KERNEL_COMMUNICATION_DFB, user_ids_mesh)},
        .tensor_parameters = {TensorParameter{.unique_id = USER_IDS_TENSOR, .spec = user_ids_mesh.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, COMPUTE},
            .target_nodes = core_grid,
        }},
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args)},
        .tensor_args = {{USER_IDS_TENSOR, user_ids_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts ManualSeedSetSeedsSetCoresProgramFactory::create_program_artifacts(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& tensor_args, Tensor& /*output_tensor*/) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName USER_IDS_DFB{"user_ids"};
    const DFBSpecName SEEDS_DFB{"seeds"};
    const DFBSpecName KERNEL_COMMUNICATION_DFB{"kernel_communication"};
    const TensorParamName USER_IDS_TENSOR{"user_ids"};
    const TensorParamName SEEDS_TENSOR{"seeds"};

    // Safety checks
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");
    TT_FATAL(
        tensor_args.seeds.has_value(), "seeds tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");

    uint32_t num_cores{};
    const CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Tensor config info
    const auto& user_ids_mesh = tensor_args.user_ids.value().mesh_tensor();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_mesh.logical_volume());

    const auto& seeds_mesh = tensor_args.seeds.value().mesh_tensor();

    // Create core kernels
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
        "reader_manual_seed_read_all_data.cpp";
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_receive_all_data.cpp";

    // Create reader kernel
    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_path,
        .dfb_bindings =
            {// The user_ids and seeds buffers are landing areas for one NoC read each: the reader fills
             // each one and then reads the landed data straight back out. No other kernel touches them,
             // so the reader holds both ends of each.
             DFBBinding{
                 .dfb_spec_name = USER_IDS_DFB,
                 .accessor_name = "user_ids",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = USER_IDS_DFB,
                 .accessor_name = "user_ids",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SEEDS_DFB,
                 .accessor_name = "seeds",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SEEDS_DFB,
                 .accessor_name = "seeds",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             // The reader writes the match flag and the selected seed that the compute kernel waits on.
             DFBBinding{
                 .dfb_spec_name = KERNEL_COMMUNICATION_DFB,
                 .accessor_name = "kernel_communication",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings =
            {TensorBinding{
                 .tensor_parameter_name = USER_IDS_TENSOR,
                 .accessor_name = "user_ids",
             },
             TensorBinding{
                 .tensor_parameter_name = SEEDS_TENSOR,
                 .accessor_name = "seeds",
             }},
        .compile_time_args = {{"number_of_ids", number_of_ids}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"core_id", core_id}});
    }

    // Create compute kernel
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = compute_kernel_path,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = KERNEL_COMMUNICATION_DFB,
            .accessor_name = "kernel_communication",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .hw_config = ComputeHardwareConfig{},
    };

    ProgramSpec spec{
        .name = "manual_seed_set_seeds_set_cores",
        .kernels = {reader, compute},
        .dataflow_buffers =
            {make_tensor_dataflow_buffer(USER_IDS_DFB, user_ids_mesh),
             make_tensor_dataflow_buffer(SEEDS_DFB, seeds_mesh),
             make_tensor_dataflow_buffer(KERNEL_COMMUNICATION_DFB, seeds_mesh)},
        .tensor_parameters =
            {TensorParameter{.unique_id = USER_IDS_TENSOR, .spec = user_ids_mesh.tensor_spec()},
             TensorParameter{.unique_id = SEEDS_TENSOR, .spec = seeds_mesh.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, COMPUTE},
            .target_nodes = core_grid,
        }},
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args)},
        .tensor_args = {{USER_IDS_TENSOR, user_ids_mesh}, {SEEDS_TENSOR, seeds_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
