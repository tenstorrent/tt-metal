// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr size_t kNumDirections = 4;

// The neighbor of each direction, in the {East, West, North, South} order of the writer's fabric connections;
// std::nullopt where the direction has no neighbor.
// precondition: `neighbors` lists the neighbors in that same order, one per true entry of `directions` (as
// common::get_neighbors returns them).
std::array<std::optional<ttnn::MeshCoordinate>, kNumDirections> neighbor_by_direction(
    const std::vector<ttnn::MeshCoordinate>& neighbors, const std::array<bool, kNumDirections>& directions) {
    std::array<std::optional<ttnn::MeshCoordinate>, kNumDirections> result;
    auto next_neighbor = neighbors.begin();
    for (size_t direction = 0; direction < kNumDirections; ++direction) {
        if (directions[direction]) {
            result[direction] = *next_neighbor++;
        }
    }
    return result;
}

// Build the ProgramSpec and run args for one device's slice of the all-to-all-combine
// workload.  Per-coord variation is real here: src_chip_id, the linearized mesh
// index and the fabric connections all depend on `mesh_coordinate`, so callers
// must build one program per coord and cannot reuse a single program across the mesh.
ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram build_combine_program(
    const AllToAllCombineDeviceOperation::operation_attributes_t& operation_attributes,
    const AllToAllCombineDeviceOperation::tensor_args_t& tensor_args,
    AllToAllCombineDeviceOperation::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tt::tt_metal::GlobalSemaphore& init_semaphore,
    const tt::tt_metal::GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;
    using tt::tt_metal::experimental::FabricConnectionSpec;
    using tt::tt_metal::experimental::FabricConnectionSpecName;
    using tt::tt_metal::experimental::FabricConnectionTarget;
    using tt::tt_metal::experimental::GlobalSemaphoreParamName;

    const KernelSpecName reader_kernel{"reader"};
    const KernelSpecName writer_kernel{"writer"};
    const DFBSpecName data_dfb{"data"};
    const DFBSpecName local_experts_dfb{"local_experts"};
    const DFBSpecName metadata_dfb{"metadata"};
    const ScratchpadSpecName mapping_scratchpad{"mapping"};
    const ScratchpadSpecName packet_header_scratchpad{"packet_header"};
    const TensorParamName input_param{"input"};
    const TensorParamName mapping_param{"mapping"};
    const TensorParamName metadata_param{"metadata"};
    const TensorParamName output_param{"output"};
    const GlobalSemaphoreParamName init_semaphore_param{"init_barrier_semaphore"};
    const GlobalSemaphoreParamName cross_device_semaphore_param{"final_barrier_semaphore"};
    // One fabric connection spec per direction, in the {East, West, North, South} order of the writer.
    const std::array<std::string, kNumDirections> direction_names = {"east", "west", "north", "south"};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    const auto input_dtype = input_tensor.dtype();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    const uint32_t batch_size = metadata_shape[1];
    const uint32_t seq_size = metadata_shape[2];
    const uint32_t selected_experts_k = metadata_shape[-1];
    const uint32_t experts = mapping_shape[-2];

    TT_FATAL(experts % num_devices == 0, "Currently assuming that experts are evenly split among devices");
    const uint32_t experts_per_device = experts / num_devices;

    const auto& input_spec = input_tensor.tensor_spec();
    const auto& mapping_spec = mapping_tensor.tensor_spec();
    const auto& metadata_spec = metadata_tensor.tensor_spec();

    const bool input_is_dram = input_tensor.memory_config().buffer_type() == BufferType::DRAM;

    const auto input_page_size_bytes = input_spec.compute_page_size_bytes();
    const auto mapping_page_size_bytes = mapping_spec.compute_page_size_bytes();
    const auto metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_input_page_size_bytes = tt::align(input_page_size_bytes, input_is_dram? dram_alignment:l1_alignment);
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());

    // Anything less will lead to deadlocks. It's clear why, TODO fix it.
    const uint32_t buffering_factor = experts_per_device;

    const auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);

    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    // perform work-split across cores
    uint32_t tokens_per_device = batch_size * seq_size;
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, operation_attributes.worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // scratch space to store and share indices of per device experts
    using local_experts_t = uint16_t;
    const auto aligned_local_expert_page_size_bytes =
        tt::align(experts_per_device * sizeof(local_experts_t), l1_alignment);
    const auto local_experts_dataformat = datatype_to_dataformat_converter(convert_to_data_type<local_experts_t>());

    // client interface
    // [0] data unicast; [1]/[2] bidirectional completion atomic-inc (pos/neg arcs); [1] reused for init send.
    constexpr auto num_headers = 3;

    Group<DataflowBufferSpec> dataflow_buffers = {
        // input sharded buffer
        DataflowBufferSpec{
            .unique_id = data_dfb,
            .entry_size = aligned_input_page_size_bytes,
            .num_entries = buffering_factor,
            .data_format_metadata = input_data_format,
        },
        DataflowBufferSpec{
            .unique_id = local_experts_dfb,
            .entry_size = aligned_local_expert_page_size_bytes,
            .num_entries = 1,
            .data_format_metadata = local_experts_dataformat,
        },
        // metadata page buffer
        DataflowBufferSpec{
            .unique_id = metadata_dfb,
            .entry_size = aligned_metadata_page_size_bytes,
            .num_entries = 1,
            .data_format_metadata = metadata_data_format,
        },
    };

    Group<ScratchpadSpec> scratchpads = {
        // full mapping buffer
        ScratchpadSpec{
            .unique_id = mapping_scratchpad,
            .size_per_node = aligned_mapping_page_size_bytes * 1,
        },
        ScratchpadSpec{
            .unique_id = packet_header_scratchpad,
            .size_per_node = CLIENT_INTERFACE_SIZE * num_headers,
        },
    };

    const uint32_t flat_mesh_idx = common::get_linearized_index(mesh_coordinate, mesh_view);

    const auto& axis = operation_attributes.axis;

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    // fabric routing info — enumerated for every device in the mesh in row-major order to
    // build the DEST_CHIP_ID / DEST_MESH_ID define arrays consumed by the writer kernel.
    // Mirrors the legacy use of `all_mesh_coordinates` (passed as tensor_coords.coords()),
    // which for these CCL ops covers the entire mesh.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord_fabric_node_id : mesh_view.get_fabric_node_ids()) {
        dest_mesh_id.push_back(*coord_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)coord_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] = common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);
    const auto neighbors_by_direction = neighbor_by_direction(neighbors, directions);

    KernelSpec::CompilerOptions::Defines writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
    };

    if (axis.has_value()) {
        writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis.value());
    }

    KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/"
                                        "reader_all_to_all_combine_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = local_experts_dfb,
                    .accessor_name = "local_experts",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = metadata_dfb,
                    .accessor_name = "metadata",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = data_dfb,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                // temp buffer just used by the reader
                ScratchpadBinding{
                    .scratchpad_spec_name = mapping_scratchpad,
                    .accessor_name = "mapping",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = input_param,
                    .accessor_name = "data",
                },
                TensorBinding{
                    .tensor_parameter_name = mapping_param,
                    .accessor_name = "mapping",
                },
                TensorBinding{
                    .tensor_parameter_name = metadata_param,
                    .accessor_name = "metadata",
                },
            },
        .compile_time_args =
            {
                {"num_local_experts", experts_per_device},
                {"batch_size", batch_size},
                {"seq_size", seq_size},
                {"num_mapping_pages", experts},  // same as experts
                {"linearized_mesh_coord", flat_mesh_idx},
                {"data_size_bytes", input_page_size_bytes},
                {"selected_experts_k", selected_experts_k},
                {"mapping_page_size_bytes", mapping_page_size_bytes},
                {"metadata_page_size_bytes", metadata_page_size_bytes},
                {"locally_reduced", operation_attributes.locally_reduced},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"token_start_idx", "token_end_idx"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::NOC_1,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };

    KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/writer_all_to_all_combine.cpp"),
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = metadata_dfb,
                    .accessor_name = "metadata",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = local_experts_dfb,
                    .accessor_name = "local_experts",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = data_dfb,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .scratchpad_bindings =
            {
                // packet headers
                ScratchpadBinding{
                    .scratchpad_spec_name = packet_header_scratchpad,
                    .accessor_name = "packet_header",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = output_param,
                    .accessor_name = "output",
                },
            },
        .compile_time_args =
            {
                {"batch_size", batch_size},
                {"seq_size", seq_size},
                {"selected_experts_k", selected_experts_k},
                {"num_local_experts", experts_per_device},
                {"num_devices", num_devices},
                {"src_chip_id", src_chip_id},
                {"data_size_bytes", input_page_size_bytes},
                {"alignment", l1_alignment},
                {"mesh_rows", mesh_view.num_rows()},
                {"mesh_cols", mesh_view.num_cols()},
                {"fabric_max_packet_size_bytes", max_packet_size_bytes},
                {"linearized_mesh_coord", common::get_linearized_index(mesh_coordinate, mesh_view)},
                {"topology", (uint32_t)topology},
                {"locally_reduced", operation_attributes.locally_reduced},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"token_start_idx", "token_end_idx"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::NOC_0,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };
    writer_kernel_spec.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = cross_device_semaphore_param,
            .accessor_name = "global_semaphore",
        },
        {
            .global_semaphore_parameter_name = init_semaphore_param,
            .accessor_name = "init_semaphore",
        },
    };

    std::vector<FabricConnectionSpec> fabric_connections;
    fabric_connections.reserve(kNumDirections);
    for (const auto& direction_name : direction_names) {
        fabric_connections.push_back(FabricConnectionSpec{
            .unique_id = FabricConnectionSpecName{direction_name},
            .src_fabric_node_id = fabric_node_id,
        });
        writer_kernel_spec.advanced_options.fabric_connection_bindings.push_back({
            .fabric_connection_spec_name = FabricConnectionSpecName{direction_name},
            .accessor_name = direction_name,
        });
    }

    KernelRunArgs reader_run_args{.kernel = reader_kernel};
    KernelRunArgs writer_run_args{.kernel = writer_kernel};

    uint32_t link_id = 0;
    uint32_t tokens_per_core_start = 0;
    log_debug(tt::LogOp, "Runtime arguments are being calculated for MeshCoordinate {}", mesh_coordinate);
    for (const auto& sender_core : sender_cores) {
        const uint32_t token_range_start = tokens_per_core_start;
        const uint32_t token_range_end = std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
        tokens_per_core_start = token_range_end;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            sender_core,
            {
                {"token_start_idx", token_range_start},
                {"token_end_idx", token_range_end},
            });

        log_debug(
            tt::LogOp,
            "Setting runtime args for core {}. It will operate on tokens {} to {}. Global semaphore address: {}",
            sender_core,
            token_range_start,
            token_range_end,
            (uint32_t)cross_device_semaphore.address());

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            sender_core,
            {
                {"token_start_idx", token_range_start},
                {"token_end_idx", token_range_end},
            });

        // One fabric connection per neighbor direction, on link `link_id`.
        for (size_t direction = 0; direction < kNumDirections; ++direction) {
            if (neighbors_by_direction[direction].has_value()) {
                fabric_connections[direction].targets[sender_core] = {
                    FabricConnectionTarget{
                        .dst_fabric_node_id = mesh_device->get_fabric_node_id(*neighbors_by_direction[direction]),
                        .link_idx = link_id,
                    },
                };
            }
        }
        link_id++;
    }

    ProgramSpec spec{
        .name = "all_to_all_combine",
        .kernels = {std::move(reader_kernel_spec), std::move(writer_kernel_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = input_param, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = mapping_param, .spec = mapping_tensor.tensor_spec()},
                TensorParameter{.unique_id = metadata_param, .spec = metadata_tensor.tensor_spec()},
                TensorParameter{.unique_id = output_param, .spec = output_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "sender",
                    .kernels = {reader_kernel, writer_kernel},
                    .target_nodes = sender_core_grid,
                },
            },
    };
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = init_semaphore_param, .memory_type = BufferType::L1},
        {.unique_id = cross_device_semaphore_param, .memory_type = BufferType::L1},
    };
    spec.advanced_options.fabric_connections = std::move(fabric_connections);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args.emplace(input_param, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(mapping_param, mapping_tensor.mesh_tensor());
    run_args.tensor_args.emplace(metadata_param, metadata_tensor.mesh_tensor());
    run_args.tensor_args.emplace(output_param, output_tensor.mesh_tensor());
    run_args.advanced_options.global_semaphore_args.emplace(init_semaphore_param, std::cref(init_semaphore));
    run_args.advanced_options.global_semaphore_args.emplace(
        cross_device_semaphore_param, std::cref(cross_device_semaphore));

    return {
        .range = ttnn::MeshCoordinateRange(mesh_coordinate),
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts
AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal;

    // Workload-scoped resources: allocate the two GlobalSemaphores once per
    // cache miss and run the cross-device Synchronize barrier here so it's
    // amortised across every per-coord program build below.  Storing them on
    // MeshWorkloadArtifacts::global_semaphores hands ownership to the program-cache so
    // they stay alive for the lifetime of the cached MeshWorkload.
    auto* mesh_device = tensor_args.input_tensor.device();
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    // Two slots up front: the programs' run args refer to the vector elements.
    artifacts.global_semaphores.reserve(2);
    artifacts.global_semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0));
    artifacts.global_semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0));
    const GlobalSemaphore& init_barrier_semaphore = artifacts.global_semaphores[0];
    const GlobalSemaphore& final_barrier_semaphore = artifacts.global_semaphores[1];
    tt::tt_metal::distributed::Synchronize(
        *mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    // Per-coord variation (src_chip_id, linearized mesh index, fabric connections)
    // means we MUST build one program per coord rather than reuse a single program across the mesh.
    artifacts.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        artifacts.programs.push_back(CMAKE_UNIQUE_NAMESPACE::build_combine_program(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            coord,
            init_barrier_semaphore,
            final_barrier_semaphore));
    }
    return artifacts;
}

}  // namespace ttnn::operations::ccl
