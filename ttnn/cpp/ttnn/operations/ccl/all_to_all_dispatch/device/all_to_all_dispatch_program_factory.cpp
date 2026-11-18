// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "ttnn/global_semaphore.hpp"
#include <limits>

namespace ttnn::operations::ccl {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.mesh_buffer().num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.mesh_buffer().page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) {
    return (uint32_t)tensor.mesh_buffer().aligned_page_size();
}

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis) {
    auto aligned_input_page_size = get_aligned_page_size(input_tensor);
    auto aligned_indices_page_size = get_aligned_page_size(indices_tensor);
    auto aligned_mapping_page_size = get_aligned_page_size(mapping_tensor);
    uint32_t tokens_per_device = get_num_rows(input_tensor);
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);

    auto mapping_pages = get_num_pages(mapping_tensor);

    auto mesh_view = input_tensor.device()->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    uint32_t dispatch_devices =
        axis.has_value() ? (axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()) : num_devices;

    constexpr uint32_t buffering_factor = 2;
    // Headers: payload, metadata (Ring: also the positive-arc completion credit), negative-arc completion credit.
    constexpr uint32_t num_packet_headers = 3;

    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    std::array<uint32_t, 6> cb_sizes = {
        buffering_factor * aligned_input_page_size,
        tokens_per_core * aligned_indices_page_size,
        mapping_pages * aligned_mapping_page_size,
        num_devices * tokens_per_core * sizeof(uint8_t),
        tokens_per_device * dispatch_devices * aligned_indices_page_size,
        num_packet_headers * packet_header_size_bytes,
    };

    std::array<uint32_t, 6> cb_page_sizes = {
        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        tokens_per_core * sizeof(uint8_t),
        aligned_indices_page_size,
        packet_header_size_bytes,
    };

    return {cb_sizes, cb_page_sizes};
}

}  // namespace detail

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

// Build the ProgramSpec and run args for one device's slice of the all-to-all-dispatch
// workload.  Per-coord variation is real here: src_chip_id, the linearized mesh
// index and the fabric connections all depend on `mesh_coordinate`, so callers
// must build one program per coord and cannot reuse a single program across the mesh.
ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram build_dispatch_program(
    const AllToAllDispatchDeviceOperation::operation_attributes_t& operation_attributes,
    const AllToAllDispatchDeviceOperation::tensor_args_t& tensor_args,
    AllToAllDispatchDeviceOperation::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tt::tt_metal::GlobalSemaphore& init_semaphore,
    const tt::tt_metal::GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using AllToAllTransferType = AllToAllDispatchDeviceOperation::AllToAllTransferType;
    using tt::tt_metal::experimental::FabricConnectionSpec;
    using tt::tt_metal::experimental::FabricConnectionSpecName;
    using tt::tt_metal::experimental::FabricConnectionTarget;
    using tt::tt_metal::experimental::GlobalSemaphoreParamName;

    const KernelSpecName reader_kernel{"reader"};
    const KernelSpecName writer_kernel{"writer"};
    const DFBSpecName input_dfb{"input"};
    const DFBSpecName indices_dfb{"indices"};
    const DFBSpecName mapping_dfb{"mapping"};
    const ScratchpadSpecName packet_header_scratchpad{"packet_header"};
    const ScratchpadSpecName send_preparation_buffer_scratchpad{"send_preparation_buffer"};
    const DFBSpecName metadata_buffer_dfb{"metadata_buffer"};
    const TensorParamName input_param{"input"};
    const TensorParamName indices_param{"indices"};
    const TensorParamName mapping_param{"mapping"};
    const TensorParamName output_param{"output"};
    const TensorParamName metadata_param{"metadata"};
    const GlobalSemaphoreParamName init_semaphore_param{"init_barrier_semaphore"};
    const GlobalSemaphoreParamName cross_device_semaphore_param{"final_barrier_semaphore"};
    // One fabric connection spec per direction, in the {East, West, North, South} order of the writer.
    const std::array<std::string, kNumDirections> direction_names = {"east", "west", "north", "south"};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& indices_tensor = tensor_args.expert_indices_tensor;
    const auto& mapping_tensor = tensor_args.expert_mapping_tensor;
    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);
    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    const auto [neighbors, directions] =
        common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);
    const auto neighbors_by_direction = neighbor_by_direction(neighbors, directions);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-2];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    log_debug(
        tt::LogOp,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    auto [dfb_sizes, dfb_page_sizes] =
        detail::get_cb_sizes(input_tensor, indices_tensor, mapping_tensor, num_links, operation_attributes.axis);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    const bool full_packet = operation_attributes.impl == AllToAllTransferType::FullPacket;

    Group<DataflowBufferSpec> dataflow_buffers = {
        // input sharded buffer
        DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = dfb_page_sizes[0],
            .num_entries = dfb_sizes[0] / dfb_page_sizes[0],
            .data_format_metadata = input_data_format,
        },
        // full indices buffer
        DataflowBufferSpec{
            .unique_id = indices_dfb,
            .entry_size = dfb_page_sizes[1],
            .num_entries = dfb_sizes[1] / dfb_page_sizes[1],
            .data_format_metadata = indices_data_format,
        },
        // full mapping buffer
        DataflowBufferSpec{
            .unique_id = mapping_dfb,
            .entry_size = dfb_page_sizes[2],
            .num_entries = dfb_sizes[2] / dfb_page_sizes[2],
            .data_format_metadata = mapping_data_format,
        },
    };
    Group<ScratchpadSpec> scratchpads = {
        // client interface
        ScratchpadSpec{
            .unique_id = packet_header_scratchpad,
            .size_per_node = dfb_sizes[5],
        },
        // book-keeping buffer to avoid sending the same token multiple times
        ScratchpadSpec{
            .unique_id = send_preparation_buffer_scratchpad,
            .size_per_node = dfb_sizes[3],
        },
    };
    if (full_packet) {
        // intermediate buffer for holding metadata before writing out to the device (for FullPacket impl)
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = metadata_buffer_dfb,
            .entry_size = dfb_page_sizes[4],
            .num_entries = dfb_sizes[4] / dfb_page_sizes[4],
            .data_format_metadata = mapping_data_format,
        });
    }

    // Enumerate the mesh in row-major order to populate the DEST_CHIP_ID / DEST_MESH_ID
    // define arrays consumed by the writer kernel.  Mirrors the legacy use of
    // tensor_coords.coords(), which for these CCL ops covers the entire mesh.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord_fabric_node_id : mesh_view.get_fabric_node_ids()) {
        dest_mesh_id.push_back(*coord_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)coord_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // The intermediate metadata buffer exists only in the FullPacket mode; FULL_PACKET selects the kernel code that
    // uses it.
    KernelSpec::CompilerOptions::Defines reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };
    if (full_packet) {
        reader_defines["FULL_PACKET"] = "1";
    }

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    KernelSpec::CompilerOptions::Defines writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
    };

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }
    if (full_packet) {
        writer_defines["FULL_PACKET"] = "1";
    }

    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = input_dfb,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = indices_dfb,
            .accessor_name = "indices",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = mapping_dfb,
            .accessor_name = "mapping",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = input_dfb,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = indices_dfb,
            .accessor_name = "indices",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = mapping_dfb,
            .accessor_name = "mapping",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    if (full_packet) {
        // The writers of every device fill the intermediate metadata buffer; the reader copies it to the metadata
        // tensor.
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = metadata_buffer_dfb,
            .accessor_name = "metadata_buffer",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = metadata_buffer_dfb,
            .accessor_name = "metadata_buffer",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/reader_all_to_all_dispatch.cpp"),
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = input_param,
                    .accessor_name = "input",
                },
                TensorBinding{
                    .tensor_parameter_name = indices_param,
                    .accessor_name = "indices",
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
                {"input_pages", input_pages},
                {"indices_pages", indices_pages},
                {"mapping_pages", mapping_pages},
                {"metadata_page_size", metadata_page_size},
                {"num_devices", num_devices},
                {"tokens_per_device", tokens_per_device},
                {"topology", (uint32_t)topology},
                {"src_mesh_id", src_mesh_id},
                {"src_chip_id", (uint32_t)src_chip_id},
                {"mesh_rows", mesh_view.num_rows()},
                {"mesh_cols", mesh_view.num_cols()},
                {"aligned_indices_page_size", aligned_indices_page_size},
                {"aligned_mapping_page_size", aligned_mapping_page_size},
                {"aligned_metadata_page_size", aligned_metadata_page_size},
                {"linearized_mesh_coord", linearized_mesh_coord},
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
                .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{},
            },
    };
    reader_kernel_spec.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = cross_device_semaphore_param,
            .accessor_name = "global_semaphore",
        },
    };

    KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp"),
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .scratchpad_bindings =
            {
                // packet headers, only touched by the writer
                ScratchpadBinding{
                    .scratchpad_spec_name = packet_header_scratchpad,
                    .accessor_name = "packet_header",
                },
                // send book-keeping, only touched by the writer
                ScratchpadBinding{
                    .scratchpad_spec_name = send_preparation_buffer_scratchpad,
                    .accessor_name = "send_preparation_buffer",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = output_param,
                    .accessor_name = "output",
                },
                TensorBinding{
                    .tensor_parameter_name = metadata_param,
                    .accessor_name = "metadata",
                },
            },
        .compile_time_args =
            {
                {"input_pages", input_pages},
                {"indices_pages", indices_pages},
                {"mapping_pages", mapping_pages},
                {"output_pages", output_pages},
                {"metadata_pages", metadata_pages},
                {"input_page_size", input_page_size},
                {"indices_page_size", indices_page_size},
                {"mapping_page_size", mapping_page_size},
                {"output_page_size", output_page_size},
                {"metadata_page_size", metadata_page_size},
                {"num_devices", num_devices},
                {"hidden_size", hidden_size},
                {"batch_size", batch_size},
                {"selected_experts_k", selected_experts_k},
                {"experts", experts},
                {"tokens_per_device", tokens_per_device},
                {"num_links", num_links},
                {"topology", (uint32_t)topology},
                {"src_mesh_id", src_mesh_id},
                {"src_chip_id", (uint32_t)src_chip_id},
                {"mesh_rows", mesh_view.num_rows()},
                {"mesh_cols", mesh_view.num_cols()},
                {"aligned_input_page_size", aligned_input_page_size},
                {"aligned_indices_page_size", aligned_indices_page_size},
                {"aligned_mapping_page_size", aligned_mapping_page_size},
                {"aligned_output_page_size", aligned_output_page_size},
                {"aligned_metadata_page_size", aligned_metadata_page_size},
                {"fabric_max_packet_size", (uint32_t)fabric_max_packet_size},
                {"alignment", l1_alignment},
                {"linearized_mesh_coord", linearized_mesh_coord},
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
                .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{},
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
            .src_fabric_node_id = src_fabric_node_id,
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
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            sender_core,
            {
                {"token_start_idx", token_range_start},
                {"token_end_idx", token_range_end},
            });

        // One fabric connection per neighbor direction, on link `link_id`.
        for (size_t direction = 0; direction < kNumDirections; ++direction) {
            if (!neighbors_by_direction[direction].has_value()) {
                continue;
            }
            const auto& neighbor_coordinate = *neighbors_by_direction[direction];
            log_debug(
                tt::LogOp,
                "Connection between mesh coord ({}, {}) and ({}, {}) at core {} will choose link_id: {} and handles "
                "token indices from {} to {}",
                mesh_coordinate[0],
                mesh_coordinate[1],
                neighbor_coordinate[0],
                neighbor_coordinate[1],
                sender_core,
                link_id,
                token_range_start,
                token_range_end);
            fabric_connections[direction].targets[sender_core] = {
                FabricConnectionTarget{
                    .dst_fabric_node_id = mesh_device->get_fabric_node_id(neighbor_coordinate),
                    .link_idx = link_id,
                },
            };
        }
        link_id++;
    }

    ProgramSpec spec{
        .name = "all_to_all_dispatch",
        .kernels = {std::move(reader_kernel_spec), std::move(writer_kernel_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = input_param, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = indices_param, .spec = indices_tensor.tensor_spec()},
                TensorParameter{.unique_id = mapping_param, .spec = mapping_tensor.tensor_spec()},
                TensorParameter{.unique_id = output_param, .spec = output_tensor.tensor_spec()},
                TensorParameter{.unique_id = metadata_param, .spec = metadata_tensor.tensor_spec()},
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
    run_args.tensor_args.emplace(indices_param, indices_tensor.mesh_tensor());
    run_args.tensor_args.emplace(mapping_param, mapping_tensor.mesh_tensor());
    run_args.tensor_args.emplace(output_param, output_tensor.mesh_tensor());
    run_args.tensor_args.emplace(metadata_param, metadata_tensor.mesh_tensor());
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
AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_mesh_workload_artifacts(
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
        artifacts.programs.push_back(CMAKE_UNIQUE_NAMESPACE::build_dispatch_program(
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
