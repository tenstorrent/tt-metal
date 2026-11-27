// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_program_factory.hpp"
#include "all_to_all_async_device_operation_types.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/utils/work_split.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/utils/math.hpp>
#include <algorithm>
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {
// Configuration constants
constexpr uint32_t MIN_CHUNK_GRANULARITY = 4;
constexpr uint32_t MAX_CHUNKS_PER_SHARD = 30;
constexpr uint32_t TRIPLE_BUFFER_MULTIPLIER = 3;
constexpr uint32_t PACKET_HEADER_BUFFER_SIZE = 8;

// Spec names. Prefixed: experimental/ccl builds as a unity build, which merges every factory's anonymous namespace.
const tt::tt_metal::KernelSpecName A2A_SENDER_READER{"sender_reader"};
const tt::tt_metal::KernelSpecName A2A_SENDER_WRITER{"sender_writer"};
const tt::tt_metal::KernelSpecName A2A_RECEIVER_WRITER{"receiver_writer"};
const tt::tt_metal::KernelSpecName A2A_RECEIVER_READER{"receiver_reader"};
// The sender cores' data buffer: the sender reader fills it from the input, the sender writer drains it into the
// fabric.
const tt::tt_metal::DFBSpecName A2A_SENDER_DATA{"sender_data"};
// The sender writer's packet headers (forward, backward and the semaphore increment).
const tt::tt_metal::DFBSpecName A2A_PACKET_HEADERS{"packet_headers"};
// The receiver cores' data buffer: the receiver reader fills it, the receiver writer drains it into the output.
const tt::tt_metal::DFBSpecName A2A_RECEIVER_DATA{"receiver_data"};
const tt::tt_metal::TensorParamName A2A_INPUT{"input"};
const tt::tt_metal::TensorParamName A2A_INTERMEDIATE{"intermediate"};
const tt::tt_metal::TensorParamName A2A_OUTPUT{"output"};
// The receiver core's count of chunks that have landed in the intermediate buffer; the sender writer increments it
// over the fabric, and the receiver reader waits on it.
const tt::tt_metal::GlobalSemaphoreParamName A2A_CHUNK_READY{"chunk_ready"};
const tt::tt_metal::experimental::FabricConnectionsSpecName A2A_FORWARD{"forward"};
const tt::tt_metal::experimental::FabricConnectionsSpecName A2A_BACKWARD{"backward"};

// A unity build merges anonymous namespaces across sources; CMake makes this one unique per source.
namespace CMAKE_UNIQUE_NAMESPACE {
// Same resolution as legacy append_fabric_connection_rt_args (tt_metal/fabric/fabric.cpp).
tt::tt_metal::experimental::FabricTargets fabric_target(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_fabric::FabricNodeId self,
    tt::tt_fabric::FabricNodeId peer) {
    using tt::tt_fabric::RoutingDirection;
    if (tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig(mesh_device))) {
        return tt::tt_metal::experimental::Fabric2DTargets{peer};
    }
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z}) {
        const auto neighbours = tt::tt_fabric::pipeline_get_chip_neighbors(mesh_device, self, direction);
        const auto it = neighbours.find(*peer.mesh_id);
        if (it != neighbours.end() && std::ranges::find(it->second, peer.chip_id) != it->second.end()) {
            return tt::tt_metal::experimental::Fabric1DTargets{direction};
        }
    }
    TT_THROW("Could not find any forwarding direction from {} to {}", self, peer);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

// Calculate optimal chunk parameters to keep num_chunks < MAX_CHUNKS_PER_SHARD
std::tuple<uint32_t, uint32_t, uint32_t> calculate_chunk_params(uint32_t pages_per_shard, uint32_t pages_per_packet) {
    uint32_t chunk_granularity = MIN_CHUNK_GRANULARITY;

    while (true) {
        const uint32_t chunk_tiles = chunk_granularity * pages_per_packet;
        const uint32_t num_chunks = tt::div_up(pages_per_shard, chunk_tiles);

        if (num_chunks < MAX_CHUNKS_PER_SHARD) {
            return {chunk_granularity, chunk_tiles, num_chunks};
        }
        chunk_granularity *= 2;
    }
}

// Create dataflow buffers for sender cores
void create_sender_buffers(
    tt::tt_metal::ProgramSpec& spec, uint32_t dfb_num_pages, uint32_t page_size, tt::DataFormat data_format) {
    // Main data buffer
    spec.dataflow_buffers.push_back(tt::tt_metal::DataflowBufferSpec{
        .unique_id = A2A_SENDER_DATA,
        .entry_size = page_size,
        .num_entries = dfb_num_pages,
        .data_format_metadata = data_format,
    });

    // Packet header buffer
    spec.dataflow_buffers.push_back(tt::tt_metal::DataflowBufferSpec{
        .unique_id = A2A_PACKET_HEADERS,
        .entry_size = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()),
        .num_entries = PACKET_HEADER_BUFFER_SIZE * 2,
        .data_format_metadata = tt::DataFormat::RawUInt32,
    });
}

// Create dataflow buffers for receiver cores
void create_receiver_buffer(
    tt::tt_metal::ProgramSpec& spec, uint32_t pages_per_packet, uint32_t page_size, tt::DataFormat data_format) {
    const uint32_t receiver_pages = pages_per_packet * TRIPLE_BUFFER_MULTIPLIER;

    spec.dataflow_buffers.push_back(tt::tt_metal::DataflowBufferSpec{
        .unique_id = A2A_RECEIVER_DATA,
        .entry_size = page_size,
        .num_entries = receiver_pages,
        .data_format_metadata = data_format,
    });
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> calculate_strides_and_offsets(
    uint32_t in_row_tiles, uint32_t in_col_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t in_dim) {
    uint32_t input_row_device_stride = 0;
    uint32_t input_col_device_stride = 0;
    uint32_t out_row_start = 0;
    uint32_t out_col_start = 0;
    uint32_t input_shard_row_tiles = in_row_tiles;
    uint32_t input_shard_col_tiles = in_col_tiles;

    if (in_dim == 2) {
        // out_dim == 3
        input_col_device_stride = in_col_tiles / ring_size;
        out_row_start = ring_index * in_row_tiles;
        input_shard_col_tiles = input_col_device_stride;
    } else if (in_dim == 3) {
        // out_dim == 2
        input_row_device_stride = in_row_tiles / ring_size;
        out_col_start = ring_index * in_col_tiles;
        input_shard_row_tiles = input_row_device_stride;
    }

    return std::make_tuple(
        input_row_device_stride,
        input_col_device_stride,
        out_row_start,
        out_col_start,
        input_shard_row_tiles,
        input_shard_col_tiles);
}

// The tensor and GlobalSemaphore arguments, supplied on the cache miss and again on every cache hit.
tt::tt_metal::ProgramRunArgs make_resource_run_args(
    const AllToAllAsyncParams& operation_attributes, const AllToAllAsyncInputs& tensor_args) {
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.tensor_args = {
        {A2A_INPUT, tensor_args.input_tensor.mesh_tensor()},
        {A2A_INTERMEDIATE, tensor_args.persistent_intermediate_buffer.mesh_tensor()},
        {A2A_OUTPUT, tensor_args.persistent_output_buffer.mesh_tensor()},
    };
    run_args.global_semaphore_args = {
        {A2A_CHUNK_READY, std::cref(operation_attributes.semaphore)},
    };
    return run_args;
}

// The program of one mesh coordinate.
ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram create_program_at(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    using namespace tt::tt_metal;

    log_debug(tt::LogOp, "DEBUG: create_descriptor is called");
    auto* mesh_device = tensor_args.input_tensor.device();
    std::optional<MeshCoordinate> forward_coord = std::nullopt;
    std::optional<MeshCoordinate> backward_coord = std::nullopt;
    uint32_t ring_index = operation_attributes.ring_size;  // Initialize device index

    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Ring,
        "DEBUG: topology: {}",
        operation_attributes.topology);

    const auto& mesh_view = tensor_args.input_tensor.device()->get_view();
    const auto ring_coords = mesh_view.get_ring_coordinates();
    const auto fabric_node_ids = mesh_view.get_ring_fabric_node_ids();

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (ring_coords.at(i) == mesh_coordinate) {
            ring_index = i;
            if (i != 0) {
                backward_coord = ring_coords.at(i - 1);
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_coord = ring_coords.at(operation_attributes.ring_size - 1);
                backward_fabric_node_id = fabric_node_ids.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_coord = ring_coords.at(i + 1);
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_coord = ring_coords.at(0);
                forward_fabric_node_id = fabric_node_ids.at(0);
            }
        }
    }

    TT_FATAL(ring_index < operation_attributes.ring_size, "DEBUG: ring_index: {}", ring_index);
    TT_FATAL(
        forward_coord.value() != backward_coord.value(),
        "DEBUG: forward and backward devices are the same: {}, {}",
        forward_coord.value(),
        backward_coord.value());
    TT_FATAL(
        forward_coord.value() != mesh_coordinate,
        "DEBUG: forward device is the same as target device: {}, {}",
        forward_coord.value(),
        mesh_coordinate);
    TT_FATAL(
        backward_coord.value() != mesh_coordinate,
        "DEBUG: backward device is the same as target device: {}, {}",
        backward_coord.value(),
        mesh_coordinate);

    // Implementation moved from all_to_all_async_minimal
    ProgramSpec spec{.name = "all_to_all_async"};
    MeshDevice* device = tensor_args.input_tensor.device();

    // Basic configuration
    const bool enable_async_output = false;
    [[maybe_unused]] const bool is_first_chip = ring_index == 0;
    [[maybe_unused]] const bool is_last_chip = ring_index == operation_attributes.ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        tensor_args.input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor};
    std::vector<Tensor> output_tensors = {
        tensor_args.persistent_intermediate_buffer, tensor_args.persistent_output_buffer};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, operation_attributes.topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ttnn::ccl::get_forward_backward_configuration(
        operation_attributes.ring_size, ring_index, operation_attributes.topology);

    log_trace(
        tt::LogOp,
        "ring_index: {}, num_targets_forward: {}, num_targets_backward: {}, dynamic_alternate: {}",
        ring_index,
        num_targets_forward,
        num_targets_backward,
        dynamic_alternate);

    TT_FATAL(!dynamic_alternate, "Dynamic alternate is not supported");

    /**
     * Get worker cores. 1 sender per link, ring_size receivers per link.
     *
     * It's ring_size rather than (ring_size-1) since this simplifies receiver
     * core coordinates. Device D always targets receiver core D on remote devices.
     */
    uint32_t num_senders_per_link = 1;
    uint32_t num_receivers_per_link = operation_attributes.ring_size;
    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links, num_senders_per_link, device, operation_attributes.sub_device_id);

    const auto [total_workers_core_range, total_workers_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        (num_senders_per_link + num_receivers_per_link),
        device,
        operation_attributes.sub_device_id);

    const auto receiver_worker_core_range = total_workers_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    TT_FATAL(
        receiver_worker_cores.size() == operation_attributes.num_links * num_receivers_per_link,
        "Expected {} receiver cores but got {}",
        operation_attributes.num_links * num_receivers_per_link,
        receiver_worker_cores.size());

    // Calculate buffer parameters
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t pages_per_packet = packet_size / page_size;
    const uint32_t dfb_pages = TRIPLE_BUFFER_MULTIPLIER * pages_per_packet;

    // Create buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    create_sender_buffers(spec, dfb_pages, page_size, data_format);
    create_receiver_buffer(spec, pages_per_packet, page_size, data_format);

    const auto [chunk_granularity, chunk_num_tiles, num_chunks_per_shard] = calculate_chunk_params(
        tensor_args.input_tensor.mesh_buffer().num_pages() /
            operation_attributes.ring_size,  // number of pages sent between each pair of devices
        pages_per_packet);

    const uint32_t contig_pages_advanced = pages_per_packet;  // Write 2 tiles per packet

    log_trace(
        tt::LogOp,
        "chunk_granularity: {}, chunk_num_tiles: {}, num_chunks_per_shard: {}",
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard);

    // Verify constraints
    TT_FATAL(num_chunks_per_shard < 32, "num_chunks_per_shard must be < 32, got {}", num_chunks_per_shard);
    TT_FATAL(chunk_granularity >= 4, "chunk_granularity must be >= 4, got {}", chunk_granularity);

    // Tensor Info
    const auto input_tensor_num_pages = tensor_args.input_tensor.mesh_buffer().num_pages();

    const uint32_t N_DRAM_BANKS = device->num_dram_channels();

    TT_FATAL(
        num_chunks_per_shard < 31,
        "num_chunks_per_shard: {}, chunk_num_tiles: {}, input_tensor_num_pages: {}",
        num_chunks_per_shard,
        chunk_num_tiles,
        input_tensor_num_pages);

    // Tensors
    const auto& input = tensor_args.input_tensor.mesh_tensor();
    const auto& intermediate = tensor_args.persistent_intermediate_buffer.mesh_tensor();
    const auto& output = tensor_args.persistent_output_buffer.mesh_tensor();
    spec.tensor_parameters = {
        TensorParameter{.unique_id = A2A_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = A2A_INTERMEDIATE, .spec = intermediate.tensor_spec()},
        TensorParameter{.unique_id = A2A_OUTPUT, .spec = output.tensor_spec()},
    };
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = A2A_CHUNK_READY}};

    // KERNEL CREATION
    // Reader
    KernelSpec worker_sender_reader_kernel{
        .unique_id = A2A_SENDER_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
            "interleaved_all_to_all_reader.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = A2A_SENDER_DATA,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A2A_INPUT, .accessor_name = "input"},
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"packet_size_in_pages", pages_per_packet},
                {"tensor0_page_size", op_config.get_page_size()},
                {"num_targets_forward_direction", num_targets_forward},
                {"num_targets_backward_direction", num_targets_backward},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "in_row_tiles",
                        "in_col_tiles",
                        "input_row_device_stride",
                        "input_col_device_stride",
                        "input_shard_row_tiles",
                        "input_shard_col_tiles",
                        "out_row_offset",
                        "out_col_offset",
                    },
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for ([[maybe_unused]] const auto& [name, arg] : worker_sender_reader_kernel.compile_time_args) {
        log_trace(tt::LogOp, "\t{}: {}", name, arg.to_uint32());
    }

    // Writer
    KernelSpec worker_sender_writer_kernel{
        .unique_id = A2A_SENDER_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
            "interleaved_all_to_all_writer.cpp",
        .dfb_bindings =
            {
                // The writer is the only kernel that touches the packet headers: it binds both ends.
                DFBBinding{
                    .dfb_spec_name = A2A_PACKET_HEADERS,
                    .accessor_name = "packet_headers",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = A2A_PACKET_HEADERS,
                    .accessor_name = "packet_headers",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = A2A_SENDER_DATA,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A2A_INTERMEDIATE, .accessor_name = "intermediate"},
                TensorBinding{.tensor_parameter_name = A2A_OUTPUT, .accessor_name = "output"},
            },
        .global_semaphore_bindings =
            {
                GlobalSemaphoreBinding{
                    .global_semaphore_parameter_name = A2A_CHUNK_READY,
                    .accessor_name = "chunk_ready",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"num_packet_headers_storable", PACKET_HEADER_BUFFER_SIZE},
                {"packet_size_in_pages", pages_per_packet},
                {"tensor0_page_size", op_config.get_page_size()},
                {"num_targets_forward_direction", num_targets_forward},
                {"num_targets_backward_direction", num_targets_backward},
                {"dynamic_alternate", dynamic_alternate},  // alternate
                {"chunk_granularity", chunk_granularity},  // granularity of signaling to receiver
                {"contig_pages_advanced", contig_pages_advanced},
                {"N_DRAM_BANKS", N_DRAM_BANKS},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "out_row_tiles",
                        "out_col_tiles",
                        "out_row_start",
                        "out_col_start",
                        "input_shard_row_tiles",
                        "input_shard_col_tiles",
                        "wait_output_semaphore",
                        "reset_global_semaphore",
                        "receiver_core_x",
                        "receiver_core_y",
                    },
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    for ([[maybe_unused]] const auto& [name, arg] : worker_sender_writer_kernel.compile_time_args) {
        log_trace(tt::LogOp, "\t{}: {}", name, arg.to_uint32());
    }

    // Create receiver kernels
    KernelSpec receiver_writer_kernel{
        .unique_id = A2A_RECEIVER_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
            "interleaved_all_to_all_receiver_writer.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = A2A_RECEIVER_DATA,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A2A_OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                // The kernel also reads an RTA of the same name (and the same value), whose local shadows this one.
                {"num_pages_per_packet_ct", pages_per_packet},
                {"chunk_granularity", chunk_granularity},
                {"chunk_num_tiles", chunk_num_tiles},
                {"num_chunks_per_shard", num_chunks_per_shard},
                {"page_size", op_config.get_page_size()},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "in_row_tiles",
                        "in_col_tiles",
                        "input_row_device_stride",
                        "input_col_device_stride",
                        "input_shard_row_tiles",
                        "input_shard_col_tiles",
                        "out_row_start",
                        "out_col_start",
                        "out_row_tiles",
                        "out_col_tiles",
                        "num_pages_per_packet",
                        "remote_device_ring_id",
                    },
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelSpec receiver_reader_kernel{
        .unique_id = A2A_RECEIVER_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
            "interleaved_all_to_all_receiver_reader.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = A2A_RECEIVER_DATA,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A2A_INPUT, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = A2A_INTERMEDIATE, .accessor_name = "intermediate"},
            },
        .global_semaphore_bindings =
            {
                GlobalSemaphoreBinding{
                    .global_semaphore_parameter_name = A2A_CHUNK_READY,
                    .accessor_name = "chunk_ready",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                // The kernel also reads an RTA of the same name (and the same value), whose local shadows this one.
                {"num_pages_per_packet_ct", pages_per_packet},
                {"chunk_granularity", chunk_granularity},
                {"chunk_num_tiles", chunk_num_tiles},
                {"num_chunks_per_shard", num_chunks_per_shard},
                {"page_size", op_config.get_page_size()},
                {"contig_pages_advanced", pages_per_packet},
                {"N_DRAM_BANKS", N_DRAM_BANKS},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "in_row_tiles",
                        "in_col_tiles",
                        "input_row_device_stride",
                        "input_col_device_stride",
                        "input_shard_row_tiles",
                        "input_shard_col_tiles",
                        "out_row_start",
                        "out_col_start",
                        "out_row_tiles",
                        "out_col_tiles",
                        "num_pages_per_packet",
                        "remote_device_ring_id",
                    },
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // Determine output shape and fracturing
    const auto& input_shape = tensor_args.input_tensor.padded_shape();
    const auto in_row_tiles = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto in_col_tiles = input_shape[3] / tt::constants::TILE_WIDTH;
    auto output_shape = tensor_args.persistent_output_buffer.padded_shape();
    const auto out_row_tiles = output_shape[2] / tt::constants::TILE_HEIGHT;
    const auto out_col_tiles = output_shape[3] / tt::constants::TILE_WIDTH;

    auto
        [input_row_device_stride,
         input_col_device_stride,
         out_row_start,
         out_col_start,
         input_shard_row_tiles,
         input_shard_col_tiles] =
            calculate_strides_and_offsets(
                in_row_tiles, in_col_tiles, operation_attributes.ring_size, ring_index, operation_attributes.in_dim);

    // Check that with this static partitioning, the final packet does not overflow the intermediate buffer
    const uint32_t packets_per_row = tt::div_up(input_shard_col_tiles, contig_pages_advanced);
    const uint32_t packets_per_device_shard = packets_per_row * input_shard_row_tiles;
    const uint32_t final_packet_id = packets_per_device_shard - 1;
    const uint32_t final_packet_global_id =
        (operation_attributes.ring_size - 2) + (final_packet_id * (operation_attributes.ring_size - 1));

    const uint32_t final_packet_first_tile_id =
        (final_packet_global_id % N_DRAM_BANKS) +
        (contig_pages_advanced * N_DRAM_BANKS * (final_packet_global_id / N_DRAM_BANKS));
    const uint32_t final_packet_second_tile_id = final_packet_first_tile_id + N_DRAM_BANKS;
    TT_FATAL(
        final_packet_second_tile_id < tensor_args.persistent_intermediate_buffer.mesh_buffer().num_pages(),
        "Final packet would overflow intermediate buffer. This is unexpected and dangerous. "
        "final_packet_first_tile_id: {}, final_packet_second_tile_id: {}, "
        "intermediate_buffer.mesh_buffer().num_pages(): "
        "{}",
        final_packet_first_tile_id,
        final_packet_second_tile_id,
        tensor_args.persistent_intermediate_buffer.mesh_buffer().num_pages());

    // Kernel Runtime Args
    ProgramRunArgs run_args = make_resource_run_args(operation_attributes, tensor_args);
    ProgramRunArgs::KernelRunArgs sender_reader_run_args{.kernel = A2A_SENDER_READER};
    ProgramRunArgs::KernelRunArgs sender_writer_run_args{.kernel = A2A_SENDER_WRITER};
    ProgramRunArgs::KernelRunArgs receiver_writer_run_args{.kernel = A2A_RECEIVER_WRITER};
    ProgramRunArgs::KernelRunArgs receiver_reader_run_args{.kernel = A2A_RECEIVER_READER};

    const uint32_t receiver_core_x = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).x;
    const uint32_t receiver_core_y = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).y;

    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    // The sender cores that open the fabric connections, and the link each one uses.
    Table<NodeCoord, uint32_t> fabric_links;

    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        AddRuntimeArgsForNode(
            sender_reader_run_args.runtime_arg_values,
            core,
            {
                {"in_row_tiles", in_row_tiles},
                {"in_col_tiles", in_col_tiles},
                {"input_row_device_stride", input_row_device_stride},
                {"input_col_device_stride", input_col_device_stride},
                {"input_shard_row_tiles", input_shard_row_tiles},
                {"input_shard_col_tiles", input_shard_col_tiles},
                {"out_row_offset", out_row_start},
                {"out_col_offset", out_col_start},
            });
        log_trace(tt::LogOp, "Reader Runtime Args:");
        log_trace(tt::LogOp, "\t{}", tensor_args.input_tensor.mesh_buffer().address());
        log_trace(tt::LogOp, "\t{}", in_row_tiles);
        log_trace(tt::LogOp, "\t{}", in_col_tiles);
        log_trace(tt::LogOp, "\t{}", input_row_device_stride);
        log_trace(tt::LogOp, "\t{}", input_col_device_stride);
        log_trace(tt::LogOp, "\t{}", input_shard_row_tiles);
        log_trace(tt::LogOp, "\t{}", input_shard_col_tiles);
        log_trace(tt::LogOp, "\t{}", out_row_start);
        log_trace(tt::LogOp, "\t{}", out_col_start);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output;
        bool reset_global_semaphore = (link == 0) && !enable_async_output;
        AddRuntimeArgsForNode(
            sender_writer_run_args.runtime_arg_values,
            core,
            {
                {"out_row_tiles", out_row_tiles},
                {"out_col_tiles", out_col_tiles},
                {"out_row_start", out_row_start},
                {"out_col_start", out_col_start},
                {"input_shard_row_tiles", input_shard_row_tiles},
                {"input_shard_col_tiles", input_shard_col_tiles},
                {"wait_output_semaphore", wait_output_semaphore},
                {"reset_global_semaphore", reset_global_semaphore},
                {"receiver_core_x", receiver_core_x},
                {"receiver_core_y", receiver_core_y},
            });

        log_trace(tt::LogOp, "Writer Runtime Args:");
        log_trace(tt::LogOp, "\t{}", tensor_args.persistent_intermediate_buffer.mesh_buffer().address());
        log_trace(tt::LogOp, "\t{}", tensor_args.persistent_output_buffer.mesh_buffer().address());
        for ([[maybe_unused]] const auto& [name, values] : sender_writer_run_args.runtime_arg_values) {
            log_trace(tt::LogOp, "\t{}: {}", name, values.find(core)->second.to_uint32());
        }
        fabric_links.emplace(core, link);
    }

    // The fabric connections: one one-target spec per side the device has, opened by every sender core on its link.
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    auto add_fabric_side = [&](const tt::tt_metal::experimental::FabricConnectionsSpecName& name,
                               const char* accessor_name,
                               tt::tt_fabric::FabricNodeId peer) {
        spec.advanced_options.fabric_connections.push_back(tt::tt_metal::experimental::FabricConnectionsSpec{
            .unique_id = name,
            .targets = CMAKE_UNIQUE_NAMESPACE::fabric_target(*mesh_device, sender_fabric_node_id, peer),
            .links = fabric_links,
        });
        worker_sender_writer_kernel.advanced_options.fabric_connection_bindings.push_back(
            tt::tt_metal::experimental::KernelAdvancedOptions::FabricConnectionBinding{
                .fabric_connections_spec_name = name,
                .accessor_name = accessor_name,
            });
    };
    if (forward_fabric_node_id.has_value()) {
        add_fabric_side(A2A_FORWARD, "forward", forward_fabric_node_id.value());
    }
    if (backward_fabric_node_id.has_value()) {
        add_fabric_side(A2A_BACKWARD, "backward", backward_fabric_node_id.value());
    }

    // Receiver args do not depend on link, so each receiver core is set once.
    for (uint32_t i = 0; i < receiver_worker_cores.size(); i++) {
        const auto core = receiver_worker_cores[i];
        // Compute strides and offsets for receiver, as if we are device at ring_index i.
        // This lets receiver mimic sender logic.
        auto
            [receiver_input_row_device_stride,
             receiver_input_col_device_stride,
             receiver_out_row_start,
             receiver_out_col_start,
             receiver_input_shard_row_tiles,
             receiver_input_shard_col_tiles] =
                calculate_strides_and_offsets(
                    in_row_tiles, in_col_tiles, operation_attributes.ring_size, i, operation_attributes.in_dim);

        // Set receiver runtime args
        AddRuntimeArgsForNode(
            receiver_reader_run_args.runtime_arg_values,
            core,
            {
                {"in_row_tiles", in_row_tiles},
                {"in_col_tiles", in_col_tiles},
                {"input_row_device_stride", receiver_input_row_device_stride},
                {"input_col_device_stride", receiver_input_col_device_stride},
                {"input_shard_row_tiles", receiver_input_shard_row_tiles},
                {"input_shard_col_tiles", receiver_input_shard_col_tiles},
                {"out_row_start", receiver_out_row_start},
                {"out_col_start", receiver_out_col_start},
                {"out_row_tiles", out_row_tiles},
                {"out_col_tiles", out_col_tiles},
                {"num_pages_per_packet", pages_per_packet},
                {"remote_device_ring_id", i},  // Receiver of device at ring_index i
            });

        AddRuntimeArgsForNode(
            receiver_writer_run_args.runtime_arg_values,
            core,
            {
                {"in_row_tiles", in_row_tiles},
                {"in_col_tiles", in_col_tiles},
                {"input_row_device_stride", receiver_input_row_device_stride},
                {"input_col_device_stride", receiver_input_col_device_stride},
                {"input_shard_row_tiles", receiver_input_shard_row_tiles},
                {"input_shard_col_tiles", receiver_input_shard_col_tiles},
                {"out_row_start", receiver_out_row_start},
                {"out_col_start", receiver_out_col_start},
                {"out_row_tiles", out_row_tiles},
                {"out_col_tiles", out_col_tiles},
                {"num_pages_per_packet", pages_per_packet},
                {"remote_device_ring_id", i},  // Receiver of device at ring_index i
            });
    }

    // Kernel order matches the legacy kernel order: sender reader, sender writer, receiver writer, receiver reader.
    spec.kernels = {
        std::move(worker_sender_reader_kernel),
        std::move(worker_sender_writer_kernel),
        std::move(receiver_writer_kernel),
        std::move(receiver_reader_kernel),
    };
    spec.work_units = {
        WorkUnitSpec{
            .name = "sender",
            .kernels = {A2A_SENDER_READER, A2A_SENDER_WRITER},
            .target_nodes = sender_worker_core_range,
        },
        WorkUnitSpec{
            .name = "receiver",
            .kernels = {A2A_RECEIVER_WRITER, A2A_RECEIVER_READER},
            .target_nodes = receiver_worker_core_range,
        },
    };
    run_args.kernel_run_args = {
        std::move(sender_reader_run_args),
        std::move(sender_writer_run_args),
        std::move(receiver_writer_run_args),
        std::move(receiver_reader_run_args),
    };

    return {
        .range = ttnn::MeshCoordinateRange(mesh_coordinate),
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // anonymous namespace

ttnn::device_operation::MeshWorkloadArtifacts AllToAllAsyncProgram::create_mesh_workload_artifacts(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    (void)tensor_return_value;
    // One program per coordinate, as the per-coordinate descriptor built.
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        artifacts.programs.push_back(create_program_at(operation_attributes, tensor_args, coord));
    }
    return artifacts;
}

tt::tt_metal::ProgramRunArgs AllToAllAsyncProgram::override_runtime_arguments(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRange& range) {
    (void)tensor_return_value;
    (void)range;
    // Every hit refreshes the input, intermediate and output buffers and the semaphore on every kernel that takes
    // them; the semaphore is excluded from the hash, so a hit may bring a different one.
    return make_resource_run_args(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim
