// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_program_factory.hpp"
#include "all_to_all_async_device_operation_types.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
// Configuration constants
constexpr uint32_t MIN_CHUNK_GRANULARITY = 4;
constexpr uint32_t MAX_CHUNKS_PER_SHARD = 30;
constexpr uint32_t TRIPLE_BUFFER_MULTIPLIER = 3;
constexpr uint32_t PACKET_HEADER_BUFFER_SIZE = 8;

// The names the per-coordinate programs and the cache-hit override share.
struct AllToAllAsyncSpecNames {
    tt::tt_metal::DFBSpecName sender_data_dfb{"sender_data"};
    tt::tt_metal::ScratchpadSpecName packet_header_scratchpad{"packet_header"};
    tt::tt_metal::DFBSpecName receiver_data_dfb{"receiver_data"};
    tt::tt_metal::TensorParamName input_param{"input"};
    tt::tt_metal::TensorParamName intermediate_param{"intermediate"};
    tt::tt_metal::TensorParamName output_param{"output"};
    tt::tt_metal::experimental::GlobalSemaphoreParamName semaphore_param{"semaphore"};
    tt::tt_metal::experimental::FabricConnectionSpecName forward_connection{"forward"};
    tt::tt_metal::experimental::FabricConnectionSpecName backward_connection{"backward"};
    tt::tt_metal::KernelSpecName sender_reader_kernel{"sender_reader"};
    tt::tt_metal::KernelSpecName sender_writer_kernel{"sender_writer"};
    tt::tt_metal::KernelSpecName receiver_writer_kernel{"receiver_writer"};
    tt::tt_metal::KernelSpecName receiver_reader_kernel{"receiver_reader"};
};

// The run args every dispatch supplies: the io tensors and the caller-supplied GlobalSemaphore.
tt::tt_metal::ProgramRunArgs make_dispatch_run_args(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    const Tensor& output_tensor) {
    const AllToAllAsyncSpecNames names;
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.tensor_args.emplace(names.input_param, tensor_args.input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(names.intermediate_param, tensor_args.persistent_intermediate_buffer.mesh_tensor());
    run_args.tensor_args.emplace(names.output_param, output_tensor.mesh_tensor());
    run_args.advanced_options.global_semaphore_args.emplace(
        names.semaphore_param, std::cref(operation_attributes.semaphore));
    return run_args;
}

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

// Create the dataflow buffer and the packet header scratchpad for sender cores
void create_sender_buffers(
    tt::tt_metal::ProgramSpec& spec, uint32_t dfb_num_pages, uint32_t page_size, tt::DataFormat data_format) {
    const AllToAllAsyncSpecNames names;
    // Main data buffer
    spec.dataflow_buffers.push_back(tt::tt_metal::DataflowBufferSpec{
        .unique_id = names.sender_data_dfb,
        .entry_size = page_size,
        .num_entries = dfb_num_pages,
        .data_format_metadata = data_format,
    });

    // Packet header buffer
    spec.scratchpads.push_back(tt::tt_metal::ScratchpadSpec{
        .unique_id = names.packet_header_scratchpad,
        .size_per_node = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()) *
                         (PACKET_HEADER_BUFFER_SIZE * 2),
    });
}

// Create dataflow buffer for receiver cores
void create_receiver_buffer(
    tt::tt_metal::ProgramSpec& spec, uint32_t pages_per_packet, uint32_t page_size, tt::DataFormat data_format) {
    const AllToAllAsyncSpecNames names;
    const uint32_t receiver_pages = pages_per_packet * TRIPLE_BUFFER_MULTIPLIER;

    spec.dataflow_buffers.push_back(tt::tt_metal::DataflowBufferSpec{
        .unique_id = names.receiver_data_dfb,
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

ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram build_all_to_all_program(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    const Tensor& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    using namespace tt::tt_metal;
    const AllToAllAsyncSpecNames names;

    log_debug(tt::LogOp, "DEBUG: build_all_to_all_program is called");
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

    spec.tensor_parameters = {
        TensorParameter{.unique_id = names.input_param, .spec = tensor_args.input_tensor.tensor_spec()},
        TensorParameter{
            .unique_id = names.intermediate_param, .spec = tensor_args.persistent_intermediate_buffer.tensor_spec()},
        TensorParameter{.unique_id = names.output_param, .spec = tensor_return_value.tensor_spec()},
    };
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = names.semaphore_param, .memory_type = BufferType::L1},
    };

    // KERNEL CREATION
    // Reader
    KernelSpec worker_sender_reader_kernel{
        .unique_id = names.sender_reader_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
                                        "interleaved_all_to_all_reader.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.sender_data_dfb,
                    .accessor_name = "sender_data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.input_param,
                    .accessor_name = "input",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"packet_size_in_pages", pages_per_packet},
                {"tensor0_page_size", op_config.get_page_size()},
                {"num_targets_forward_direction", static_cast<uint32_t>(num_targets_forward)},
                {"num_targets_backward_direction", static_cast<uint32_t>(num_targets_backward)},
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

    // Writer
    KernelSpec worker_sender_writer_kernel{
        .unique_id = names.sender_writer_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
                                        "interleaved_all_to_all_writer.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.sender_data_dfb,
                    .accessor_name = "sender_data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{
                    .scratchpad_spec_name = names.packet_header_scratchpad,
                    .accessor_name = "packet_header",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.intermediate_param,
                    .accessor_name = "intermediate",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"num_packet_headers_storable", PACKET_HEADER_BUFFER_SIZE},
                {"packet_size_in_pages", pages_per_packet},
                {"tensor0_page_size", op_config.get_page_size()},
                {"num_targets_forward_direction", static_cast<uint32_t>(num_targets_forward)},
                {"num_targets_backward_direction", static_cast<uint32_t>(num_targets_backward)},
                {"dynamic_alternate", static_cast<uint32_t>(dynamic_alternate)},
                {"chunk_granularity", chunk_granularity},  // granularity of signaling to receiver
                {"contig_pages_advanced", contig_pages_advanced},
                {"num_dram_banks", N_DRAM_BANKS},
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
    worker_sender_writer_kernel.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = names.semaphore_param,
            .accessor_name = "semaphore",
        },
    };
    worker_sender_writer_kernel.advanced_options.fabric_connection_bindings = {
        {
            .fabric_connection_spec_name = names.forward_connection,
            .accessor_name = "forward",
        },
        {
            .fabric_connection_spec_name = names.backward_connection,
            .accessor_name = "backward",
        },
    };

    // Create receiver kernels
    KernelSpec receiver_writer_kernel{
        .unique_id = names.receiver_writer_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
                                        "interleaved_all_to_all_receiver_writer.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.receiver_data_dfb,
                    .accessor_name = "receiver_data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.output_param,
                    .accessor_name = "output",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"packet_size_in_pages", pages_per_packet},
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
        .unique_id = names.receiver_reader_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
                                        "interleaved_all_to_all_receiver_reader.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.receiver_data_dfb,
                    .accessor_name = "receiver_data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.input_param,
                    .accessor_name = "input",
                },
                TensorBinding{
                    .tensor_parameter_name = names.intermediate_param,
                    .accessor_name = "intermediate",
                },
            },
        .compile_time_args =
            {
                {"my_ring_id", ring_index},
                {"ring_size", operation_attributes.ring_size},
                {"packet_size_in_pages", pages_per_packet},
                {"chunk_granularity", chunk_granularity},
                {"chunk_num_tiles", chunk_num_tiles},
                {"num_chunks_per_shard", num_chunks_per_shard},
                {"page_size", op_config.get_page_size()},
                {"contig_pages_advanced", pages_per_packet},
                {"num_dram_banks", N_DRAM_BANKS},
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
    receiver_reader_kernel.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = names.semaphore_param,
            .accessor_name = "semaphore",
        },
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
    const uint32_t receiver_core_x = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).x;
    const uint32_t receiver_core_y = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).y;

    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    tt::tt_metal::experimental::FabricConnectionSpec forward_fabric_connection{
        .unique_id = names.forward_connection,
        .src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate),
    };
    tt::tt_metal::experimental::FabricConnectionSpec backward_fabric_connection{
        .unique_id = names.backward_connection,
        .src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate),
    };

    KernelRunArgs reader_run_args{.kernel = names.sender_reader_kernel};
    KernelRunArgs writer_run_args{.kernel = names.sender_writer_kernel};
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        log_trace(tt::LogOp, "Reader Runtime Args:");
        log_trace(tt::LogOp, "\t{}", in_row_tiles);
        log_trace(tt::LogOp, "\t{}", in_col_tiles);
        log_trace(tt::LogOp, "\t{}", input_row_device_stride);
        log_trace(tt::LogOp, "\t{}", input_col_device_stride);
        log_trace(tt::LogOp, "\t{}", input_shard_row_tiles);
        log_trace(tt::LogOp, "\t{}", input_shard_col_tiles);
        log_trace(tt::LogOp, "\t{}", out_row_start);
        log_trace(tt::LogOp, "\t{}", out_col_start);
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
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

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output;
        bool reset_global_semaphore = (link == 0) && !enable_async_output;
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"out_row_tiles", out_row_tiles},
                {"out_col_tiles", out_col_tiles},
                {"out_row_start", out_row_start},
                {"out_col_start", out_col_start},
                {"input_shard_row_tiles", input_shard_row_tiles},
                {"input_shard_col_tiles", input_shard_col_tiles},
                {"wait_output_semaphore", static_cast<uint32_t>(wait_output_semaphore)},
                {"reset_global_semaphore", static_cast<uint32_t>(reset_global_semaphore)},
                {"receiver_core_x", receiver_core_x},
                {"receiver_core_y", receiver_core_y},
            });

        log_trace(tt::LogOp, "Writer Runtime Args:");
        log_trace(tt::LogOp, "\t{}", out_row_tiles);
        log_trace(tt::LogOp, "\t{}", out_col_tiles);
        log_trace(tt::LogOp, "\t{}", out_row_start);
        log_trace(tt::LogOp, "\t{}", out_col_start);
        log_trace(tt::LogOp, "\t{}", input_shard_row_tiles);
        log_trace(tt::LogOp, "\t{}", input_shard_col_tiles);
        log_trace(tt::LogOp, "\t{}", wait_output_semaphore);
        log_trace(tt::LogOp, "\t{}", reset_global_semaphore);
        log_trace(tt::LogOp, "\t{}", receiver_core_x);
        log_trace(tt::LogOp, "\t{}", receiver_core_y);
        if (forward_fabric_node_id.has_value()) {
            forward_fabric_connection.targets[core] = {
                tt::tt_metal::experimental::FabricConnectionTarget{
                    .dst_fabric_node_id = forward_fabric_node_id.value(),
                    .link_idx = link,
                },
            };
        }
        if (backward_fabric_node_id.has_value()) {
            backward_fabric_connection.targets[core] = {
                tt::tt_metal::experimental::FabricConnectionTarget{
                    .dst_fabric_node_id = backward_fabric_node_id.value(),
                    .link_idx = link,
                },
            };
        }
    }

    // Receiver args do not depend on link, so each receiver core is set once.
    KernelRunArgs receiver_reader_run_args{.kernel = names.receiver_reader_kernel};
    KernelRunArgs receiver_writer_run_args{.kernel = names.receiver_writer_kernel};
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

    spec.kernels = {
        std::move(worker_sender_reader_kernel),
        std::move(worker_sender_writer_kernel),
        std::move(receiver_writer_kernel),
        std::move(receiver_reader_kernel),
    };
    spec.work_units = {
        WorkUnitSpec{
            .name = "sender",
            .kernels = {names.sender_reader_kernel, names.sender_writer_kernel},
            .target_nodes = sender_worker_core_range,
        },
        WorkUnitSpec{
            .name = "receiver",
            .kernels = {names.receiver_writer_kernel, names.receiver_reader_kernel},
            .target_nodes = receiver_worker_core_range,
        },
    };
    spec.advanced_options.fabric_connections = {
        std::move(forward_fabric_connection),
        std::move(backward_fabric_connection),
    };

    ProgramRunArgs run_args = make_dispatch_run_args(operation_attributes, tensor_args, tensor_return_value);
    run_args.kernel_run_args = {
        std::move(reader_run_args),
        std::move(writer_run_args),
        std::move(receiver_writer_run_args),
        std::move(receiver_reader_run_args),
    };

    return {
        .range = ttnn::MeshCoordinateRange(mesh_coordinate),
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts AllToAllAsyncProgram::create_mesh_workload_artifacts(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    const auto coords = tensor_coords.coords();
    artifacts.programs.reserve(coords.size());
    // The ring index and the forward/backward neighbors depend on the coordinate, so each coordinate gets its own
    // program.
    for (const auto& coord : coords) {
        artifacts.programs.push_back(CMAKE_UNIQUE_NAMESPACE::build_all_to_all_program(
            operation_attributes, tensor_args, tensor_return_value, coord));
    }
    return artifacts;
}

tt::tt_metal::ProgramRunArgs AllToAllAsyncProgram::override_runtime_arguments(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRange& /*range*/) {
    // The semaphore is excluded from AllToAllAsyncDeviceOperation::compute_program_hash, so it is re-applied here on
    // every hit, together with the input, intermediate and output tensors.
    return CMAKE_UNIQUE_NAMESPACE::make_dispatch_run_args(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim
