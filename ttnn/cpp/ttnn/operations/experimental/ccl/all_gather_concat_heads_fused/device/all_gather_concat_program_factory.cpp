// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace ttnn::ccl;

namespace {

struct llama_config {
    CoreRange nlp_only_core_range_1 = CoreRange({1, 1}, {3, 1});  // cores that are used for NLP op only
    CoreRange nlp_only_core_range_2 = CoreRange({1, 2}, {2, 2});
    uint32_t num_cores_input_tensor = 8;
    std::array<CoreRange, 3> semaphore_mcast_ranges = {
        CoreRange({5, 9}, {6, 9}),  // cores waiting for all gather op to finish to start nlp op
        CoreRange({5, 0}, {6, 2}),
        CoreRange({5, 4}, {6, 7})};

    uint32_t num_semaphore_ranges = 3;
    uint32_t concat_num_cores = 16;
    uint32_t num_tiles_reshard = 2;
};

// Builds the ProgramDescriptor for one coord.  ring_index, forward/backward
// neighbours vary with the coord; the rest mirrors the legacy create_at body
// verbatim.  Dynamic CBs (output tensor & globally-allocated cb_q_output) are
// expressed via CBDescriptor::buffer so the framework patches the address on
// every dispatch.
ProgramDescriptor build_program_descriptor(
    const AllGatherConcatParams& operation_attributes,
    const AllGatherConcatInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& temp_tensor = tensor_args.buffer_tensor;
    auto& output_tensor = tensor_return_value;

    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Input tensor must be on a MeshDevice");

    const auto& mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    IDevice* target_device = mesh_device->get_device(mesh_coordinate);

    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                        : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    const auto fabric_node_ids = (operation_attributes.cluster_axis == 0)
                                     ? mesh_view.get_fabric_node_ids_on_column(mesh_coordinate[1])
                                     : mesh_view.get_fabric_node_ids_on_row(mesh_coordinate[0]);

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t ring_index = 0;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_fabric_node_id = fabric_node_ids.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_fabric_node_id = fabric_node_ids.at(0);
            }
        }
    }

    const bool enable_async_output_tensor = false;

    auto ring_core_ranges = output_tensor.shard_spec().value().grid.ranges();

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == operation_attributes.ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        target_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    std::vector<Tensor> temp_tensors = {temp_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, temp_tensors, operation_attributes.topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ::ttnn::ccl::get_forward_backward_configuration(
            operation_attributes.ring_size, ring_index, operation_attributes.topology);

    // To overlap NLP local data with all gather, we divide the batches for each device into:
    //      - local batch (starts with start_local)
    //      - remote batches
    //          - batch 1 (from batch_start_1 to batch end_1)
    //          - batch 2 (from batch_start_2 to batch end_2) if applicable
    LineTopology line_topology(operation_attributes.ring_size, ring_index);
    const size_t num_targets_right = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
    const size_t num_targets_left = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);

    uint32_t batch_size = 1;
    uint32_t batch_start_1 = 8;
    uint32_t batch_end_1 = 32;
    uint32_t batch_start_2 = 0;
    uint32_t batch_end_2 = 0;
    uint32_t start_local = 0;

    if (num_targets_right == 2 && num_targets_left == 1) {
        batch_size = 2;
        batch_start_1 = 0;
        batch_end_1 = 8;
        batch_start_2 = 16;
        batch_end_2 = 32;
        start_local = 8;
    } else if (num_targets_right == 1 && num_targets_left == 2) {
        batch_size = 2;
        batch_start_1 = 0;
        batch_end_1 = 16;
        batch_start_2 = 24;
        batch_end_2 = 32;
        start_local = 16;
    } else if (num_targets_right == 0 && num_targets_left == 3) {
        batch_size = 1;
        batch_start_1 = 0;
        batch_end_1 = 24;
        start_local = 24;
    }

    // Get worker cores, assuming 1 worker per link
    auto [sender_worker_core_range, sender_worker_cores] =
        llama_specific::get_custom_worker_core_placement(operation_attributes.num_links);

    // Tensor Info
    const uint32_t logical_dim_2 = std::min(input_tensor.logical_shape()[2], operation_attributes.num_heads);
    const auto input_tensor_num_pages =
        input_tensor.logical_shape()[0] * input_tensor.logical_shape()[1] * logical_dim_2;
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = logical_dim_2;

    const auto output_interm_tensor_cores = temp_tensor.memory_config().shard_spec()->grid;
    const auto output_interm_tensor_shard_num_pages = logical_dim_2;
    const auto row_size = input_tensor.padded_shape()[-1] / 2 * output_tensor.element_size();

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);

    // concat info
    uint32_t single_tile_size = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()));

    uint32_t first_phase = 1;
    auto q_shard_spec = output_tensor.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto in_shard_spec = temp_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;

    ProgramDescriptor desc;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        (input_tensor_num_pages / operation_attributes.num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes}},
    });

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes}},
    });

    // Globally-allocated CB pointing at the output tensor's buffer.  Setting
    // CBDescriptor::buffer wires the framework's dynamic-CB patcher: on cache
    // hits the CB address is updated from output_tensor.buffer() directly.
    uint32_t q_output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tensor.padded_shape()[-2] * row_size,
        .core_ranges = q_cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(q_output_cb_index), .data_format = df, .page_size = single_tile_size}},
        .buffer = output_tensor.buffer(),
    });

    uint32_t pre_tilize_cb_index = tt::CBIndex::c_17;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tensor.padded_shape()[-2] * row_size,
        .core_ranges = q_cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(pre_tilize_cb_index),
            .data_format = df,
            .page_size = single_tile_size}},
    });

    llama_config llama_configuration;
    std::vector<CoreRange> q_cores_vector;
    uint32_t range_count = 0;
    for (auto cr : ring_core_ranges) {
        q_cores_vector.push_back(cr);
        range_count++;
        if (range_count == llama_configuration.concat_num_cores) {
            break;
        }
    }
    const auto& q_cores_updated = CoreRangeSet(q_cores_vector);
    std::vector<CoreRange> sem_cores_vector;
    sem_cores_vector.push_back(CoreRange(sender_worker_cores[0], sender_worker_cores[0]));
    range_count = 0;
    for (auto cr : ring_core_ranges) {
        sem_cores_vector.push_back(cr);
        range_count++;
        if (range_count == llama_configuration.concat_num_cores) {
            break;
        }
    }
    const auto& sem_cores_updated = CoreRangeSet(sem_cores_vector);
    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc =
        operation_attributes.use_noc1_only ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores);
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_x_coords.push_back(mesh_device->worker_core_from_logical_core(in_cores_vec[i]).x);
        noc_y_coords.push_back(mesh_device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    // create concat semaphore for each link (reserve slot ids; framework allocates real semaphores)
    const uint32_t concat_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = concat_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = sem_cores_updated,
        .initial_value = 0,
    });
    const uint32_t concat_semaphore_id2 = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = concat_semaphore_id2,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = sem_cores_updated,
        .initial_value = 0,
    });

    std::vector<uint32_t> concat_reader_ct_args = {
        pre_tilize_cb_index,
        first_phase,
        in_num_cores,
        batch_size,
        batch_start_1,
        batch_end_1,
        batch_start_2,
        batch_end_2,
        start_local,
        input_tensor_shard_shape[1] * input_tensor.element_size(),
        output_tensor_shard_shape[1] * output_tensor.element_size(),
    };

    KernelDescriptor concat_reader_kernel_desc;
    concat_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_concat_reader.cpp";
    concat_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    concat_reader_kernel_desc.core_ranges = q_cores_updated;
    concat_reader_kernel_desc.compile_time_args = std::move(concat_reader_ct_args);
    concat_reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
        .noc_mode = operation_attributes.use_noc1_only ? NOC_MODE::DM_DYNAMIC_NOC : NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(concat_reader_kernel_desc));
    const KernelHandle concat_reader_kernel_id = desc.kernels.size() - 1;

    std::vector<uint32_t> tilize_ct_args = {
        q_output_cb_index,
    };
    KernelDescriptor tilize_writer_kernel_desc;
    tilize_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "tilize_writer.cpp";
    tilize_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    tilize_writer_kernel_desc.core_ranges = q_cores_updated;
    tilize_writer_kernel_desc.compile_time_args = std::move(tilize_ct_args);
    tilize_writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = writer_noc,
        .noc_mode = operation_attributes.use_noc1_only ? NOC_MODE::DM_DYNAMIC_NOC : NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(tilize_writer_kernel_desc));

    KernelDescriptor tilize_compute_kernel_desc;
    tilize_compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/tilize_compute.cpp";
    tilize_compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    tilize_compute_kernel_desc.core_ranges = q_cores_updated;
    tilize_compute_kernel_desc.compile_time_args = {1, 2, tt::CBIndex::c_17, tt::CBIndex::c_16};
    tilize_compute_kernel_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(tilize_compute_kernel_desc));

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> all_gather_reader_ct_args = {
        ring_index,     // my_chip_id
        src0_cb_index,  // cb0_id
        op_config.get_page_size(),
    };  // tensor0_page_size};

    KernelDescriptor worker_sender_reader_kernel_desc;
    worker_sender_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_all_gather_concat_reader.cpp";
    worker_sender_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_reader_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_reader_kernel_desc.compile_time_args = std::move(all_gather_reader_ct_args);
    worker_sender_reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
        .noc_mode = operation_attributes.use_noc1_only ? NOC_MODE::DM_DYNAMIC_NOC : NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_reader_kernel_desc));
    const KernelHandle worker_sender_reader_kernel_id = desc.kernels.size() - 1;

    // Writer
    uint32_t out_ready_sem_wait_value =
        (dynamic_alternate ? (operation_attributes.ring_size + 1) : operation_attributes.ring_size) *
        operation_attributes.num_links;
    std::vector<uint32_t> all_gather_writer_ct_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        dynamic_alternate,                // alternate
        llama_configuration.num_semaphore_ranges,
        out_ready_sem_wait_value};

    KernelDescriptor worker_sender_writer_kernel_desc;
    worker_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_all_gather_concat_writer.cpp";
    worker_sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_writer_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_writer_kernel_desc.compile_time_args = std::move(all_gather_writer_ct_args);
    worker_sender_writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = writer_noc,
        .noc_mode = operation_attributes.use_noc1_only ? NOC_MODE::DM_DYNAMIC_NOC : NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_writer_kernel_desc));
    const KernelHandle worker_sender_writer_kernel_id = desc.kernels.size() - 1;

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_interm_tensor_cores, std::nullopt, true);
    auto cores_per_device =
        output_cores_vec.size() + operation_attributes.ring_size - (1 / operation_attributes.ring_size);
    uint32_t start_core_index_for_device = output_cores_vec.size() / operation_attributes.ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;

    TT_FATAL(
        output_cores_vec.size() % operation_attributes.ring_size == 0 || output_cores_vec.size() == 1,
        "output sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution scheme",
        output_cores_vec.size(),
        operation_attributes.ring_size);
    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);

    log_trace(tt::LogOp, "output_cores_this_device: {}", output_cores_this_device);

    std::vector<uint32_t> nlp_local_core_x;
    std::vector<uint32_t> nlp_local_core_y;
    for (uint32_t k = 0; k < llama_configuration.num_cores_input_tensor; k++) {
        auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[k]);
        nlp_local_core_x.push_back(this_core.x);
        nlp_local_core_y.push_back(this_core.y);
    }

    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / operation_attributes.num_links;
        uint32_t remainder = input_tensor_num_pages % operation_attributes.num_links;
        bool add_remainder = link == operation_attributes.num_links - 1;
        uint32_t input_tile_id_start = link * base_pages_per_worker;
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + (add_remainder * remainder);

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t output_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % output_interm_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / output_interm_tensor_shard_num_pages;
             i < (input_tile_id_end + output_interm_tensor_shard_num_pages - 1) / output_interm_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args.  input_tensor buffer base address is a
        // Buffer* binding so the framework patches it on cache hits.
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.reserve(6 + 2 * input_tensor_cores_x.size());
        reader_rt_args.push_back(input_tensor.buffer());  // tensor_address0
        reader_rt_args.push_back(operation_attributes.semaphore.address());
        reader_rt_args.push_back(static_cast<uint32_t>(input_tensor_shard_num_pages));
        reader_rt_args.push_back(worker_num_tiles_to_read);                            // num_tiles_to_read
        reader_rt_args.push_back(input_first_core_tile_start_offset);                  // first_core_tile_start_offset
        reader_rt_args.push_back(static_cast<uint32_t>(input_tensor_cores_x.size()));  // num_cores
        for (uint32_t v : input_tensor_cores_x) {
            reader_rt_args.push_back(v);
        }
        for (uint32_t v : input_tensor_cores_y) {
            reader_rt_args.push_back(v);
        }
        desc.kernels[worker_sender_reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

        // Set writer runtime args.  temp_tensor buffer base address is a
        // Buffer* binding so the framework patches it on cache hits.
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(temp_tensor.buffer());                      // tensor_address0
        writer_rt_args.push_back(operation_attributes.semaphore.address());  // out_ready_sem_bank_addr
        writer_rt_args.push_back(static_cast<uint32_t>(input_tensor_shard_num_pages));
        writer_rt_args.push_back(worker_num_tiles_to_read);                             // num_tiles_to_read
        writer_rt_args.push_back(output_first_core_tile_start_offset);                  // first_core_tile_start_offset
        writer_rt_args.push_back(static_cast<uint32_t>(output_tensor_cores_x.size()));  // num_cores
        writer_rt_args.push_back(static_cast<uint32_t>(wait_output_semaphore));         // wait_output_semaphore
        writer_rt_args.push_back(static_cast<uint32_t>(reset_global_semaphore));        // reset_global_semaphore
        writer_rt_args.push_back(static_cast<uint32_t>(drain_sync_core.x));
        writer_rt_args.push_back(static_cast<uint32_t>(drain_sync_core.y));
        writer_rt_args.push_back(concat_semaphore_id);
        writer_rt_args.push_back(concat_semaphore_id2);

        auto sem_mcast_ranges = CoreRangeSet(llama_configuration.semaphore_mcast_ranges);
        std::vector<uint32_t> mcast_start_x;
        std::vector<uint32_t> mcast_start_y;
        std::vector<uint32_t> mcast_end_x;
        std::vector<uint32_t> mcast_end_y;

        for (const auto& range : sem_mcast_ranges.ranges()) {
            auto start_core = mesh_device->worker_core_from_logical_core(range.start_coord);
            auto end_core = mesh_device->worker_core_from_logical_core(range.end_coord);
            if (writer_noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(start_core, end_core);
            }
            mcast_start_x.push_back(start_core.x);
            mcast_start_y.push_back(start_core.y);
            mcast_end_x.push_back(end_core.x);
            mcast_end_y.push_back(end_core.y);
        }
        for (uint32_t v : output_tensor_cores_x) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : output_tensor_cores_y) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : mcast_start_x) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : mcast_start_y) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : mcast_end_x) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : mcast_end_y) {
            writer_rt_args.push_back(v);
        }

        // Fabric connection args need a temporary uint32_t vector since the
        // helper appends raw values.
        std::vector<uint32_t> writer_rt_args_extra;
        writer_rt_args_extra.push_back(static_cast<uint32_t>(forward_fabric_node_id.has_value()));
        if (forward_fabric_node_id.has_value()) {
            const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                target_device_fabric_node_id, forward_fabric_node_id.value(), link, desc, core, writer_rt_args_extra);
        }
        writer_rt_args_extra.push_back(static_cast<uint32_t>(backward_fabric_node_id.has_value()));
        if (backward_fabric_node_id.has_value()) {
            const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                target_device_fabric_node_id, backward_fabric_node_id.value(), link, desc, core, writer_rt_args_extra);
        }
        for (uint32_t v : writer_rt_args_extra) {
            writer_rt_args.push_back(v);
        }

        desc.kernels[worker_sender_writer_kernel_id].emplace_runtime_args(core, writer_rt_args);
    }

    /* rt for concat kernels */
    for (uint32_t i = 0; i < llama_configuration.concat_num_cores; ++i) {
        uint32_t second_half_core = 1;
        if (i % 2 == 0) {
            second_half_core = 0;
        }
        // in_tile_offset_by_batch is the start address of each batch in the input tile. The first face_h batches are in
        // the upper half of the tile and rest are in the lower half of tile.
        const auto& core = cores[i];
        std::vector<uint32_t> input_cores_x;
        std::vector<uint32_t> input_cores_y;
        std::array<uint32_t, 8> kernel_core_noc_x = {19, 20, 21, 19, 20, 21, 19, 20};
        std::array<uint32_t, 8> kernel_core_noc_y = {18, 18, 18, 19, 19, 19, 20, 20};
        for (uint32_t k = 0; k < llama_configuration.num_cores_input_tensor; k++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[k]);
            input_cores_x.push_back(this_core.x);
            input_cores_y.push_back(this_core.y);
            TT_ASSERT(
                this_core.x == kernel_core_noc_x[k] && this_core.y == kernel_core_noc_y[k],
                "This op should run on a TG machine");
        }
        bool is_worker_core = core.x == 1 && core.y == 0;
        is_worker_core = is_worker_core || (core.x == 2 && core.y == 0);
        is_worker_core = is_worker_core || (core.x == 3 && core.y == 0);
        if (!is_worker_core) {
            KernelDescriptor::RTArgList reader_runtime_args;
            reader_runtime_args.reserve(6 + (2 * in_num_cores));
            // q_start_addr (temp_tensor base) and input_tensor base are
            // Buffer* bindings — patched per-dispatch.
            reader_runtime_args.push_back(temp_tensor.buffer());
            reader_runtime_args.push_back(input_tensor.buffer());
            reader_runtime_args.push_back(concat_semaphore_id);
            reader_runtime_args.push_back(concat_semaphore_id2);
            for (uint32_t v : noc_x_coords) {
                reader_runtime_args.push_back(v);
            }
            for (uint32_t v : noc_y_coords) {
                reader_runtime_args.push_back(v);
            }
            reader_runtime_args.push_back(second_half_core);
            reader_runtime_args.push_back(i / 2);

            desc.kernels[concat_reader_kernel_id].emplace_runtime_args(core, reader_runtime_args);
        }
    }

    return desc;
}

}  // namespace

WorkloadDescriptor AllGatherConcatMeshWorkloadFactory::create_workload_descriptor(
    const AllGatherConcatParams& operation_attributes,
    const AllGatherConcatInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());

    for (const auto& coord : coords) {
        ProgramDescriptor desc =
            build_program_descriptor(operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
