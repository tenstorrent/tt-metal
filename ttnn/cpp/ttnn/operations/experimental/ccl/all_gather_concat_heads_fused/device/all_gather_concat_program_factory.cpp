// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ranges>
#include <optional>
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace ttnn::ccl;

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

AllGatherConcatMeshWorkloadFactory::cached_mesh_workload_t AllGatherConcatMeshWorkloadFactory::create_mesh_workload(
    const AllGatherConcatParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

AllGatherConcatMeshWorkloadFactory::cached_program_t AllGatherConcatMeshWorkloadFactory::create_at(
    const AllGatherConcatParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
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

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t ring_index = 0;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_device = devices.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    tt::tt_metal::Program program{};
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

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        (input_tensor_num_pages / operation_attributes.num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    uint32_t q_output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_q_output_config =
        tt::tt_metal::CircularBufferConfig(output_tensor.padded_shape()[-2] * row_size, {{q_output_cb_index, df}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_q_output = tt::tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t pre_tilize_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig cb_pre_tilize_config =
        tt::tt_metal::CircularBufferConfig(output_tensor.padded_shape()[-2] * row_size, {{pre_tilize_cb_index, df}})
            .set_page_size(pre_tilize_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, q_cores, cb_pre_tilize_config);

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
    // create concat semaphore for each link
    uint32_t concat_semaphore_id = tt::tt_metal::CreateSemaphore(program, sem_cores_updated, 0);
    uint32_t concat_semaphore_id2 = tt::tt_metal::CreateSemaphore(program, sem_cores_updated, 0);

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

    auto concat_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_concat_reader.cpp",
        q_cores_updated,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = concat_reader_ct_args});

    std::vector<uint32_t> tilize_ct_args = {
        q_output_cb_index,
    };
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "tilize_writer.cpp",
        q_cores_updated,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = tilize_ct_args});

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/tilize_compute.cpp",
        q_cores_updated,
        tt::tt_metal::ComputeConfig{.compile_args = {1, 2, tt::CBIndex::c_17, tt::CBIndex::c_16}});

    // KERNEL CREATION
    // Reader

    std::vector<uint32_t> all_gather_reader_ct_args = {
        ring_index,     // my_chip_id
        src0_cb_index,  // cb0_id
        op_config.get_page_size(),
    };  // tensor0_page_size};

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_all_gather_concat_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = all_gather_reader_ct_args});

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

    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_all_gather_concat_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = all_gather_writer_ct_args});

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

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            operation_attributes.semaphore.address(),
            input_tensor_shard_num_pages,
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        std::vector<uint32_t> writer_rt_args = {
            temp_tensor.buffer()->address(),           // tensor_address0
            operation_attributes.semaphore.address(),  // out_ready_sem_bank_addr (absolute address)
            input_tensor_shard_num_pages,
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,
            drain_sync_core.y,
            concat_semaphore_id,
            concat_semaphore_id2,
        };

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
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());

        writer_rt_args.insert(writer_rt_args.end(), mcast_start_x.begin(), mcast_start_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_y.begin(), mcast_start_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_x.begin(), mcast_end_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_y.begin(), mcast_end_y.end());

        log_trace(tt::LogOp, "Writer Runtime Args:");
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto target_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto forward_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_device_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto target_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto backward_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_device_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    /* rt for concat kernels*/
    uint32_t q_start_addr = temp_tensor.buffer()->address();
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
        if (is_worker_core == 0) {
            std::vector<uint32_t> reader_runtime_args;
            reader_runtime_args.reserve(6 + (2 * in_num_cores));
            reader_runtime_args = {
                q_start_addr, input_tensor.buffer()->address(), concat_semaphore_id, concat_semaphore_id2};

            reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
            reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());
            reader_runtime_args.push_back(second_half_core);
            reader_runtime_args.push_back(i / 2);

            tt::tt_metal::SetRuntimeArgs(program, concat_reader_kernel_id, core, reader_runtime_args);
        }
    }
    uint32_t num_concat_worker_cores = llama_configuration.concat_num_cores;

    AllGatherConcatSharedVariables shared_vars{
        .sender_worker_cores = sender_worker_cores,
        .num_concat_worker_cores = num_concat_worker_cores,
        .cb_q_output = cb_q_output,
        .cores = cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .concat_reader_kernel_id = concat_reader_kernel_id,
    };

    return cached_program_t{std::move(program), std::move(shared_vars)};
}

void AllGatherConcatMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherConcatParams& operation_attributes,
    const AllGatherConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& temp_tensor = tensor_args.buffer_tensor;
    const auto& output = tensor_return_value;

    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);

        auto* dst_buffer_query = output.buffer();
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_q_output, *dst_buffer_query);

        auto semaphore = operation_attributes.semaphore;
        log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

        // update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);

        uint32_t q_base_addr = temp_tensor.buffer()->address();
        uint32_t q_start_addr = q_base_addr;
        for (const auto& core : shared_vars.sender_worker_cores) {
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();
            worker_reader_sender_runtime_args[1] = semaphore.address();

            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = q_start_addr;
            worker_writer_sender_runtime_args[1] = semaphore.address();
        }

        for (uint32_t i = 0; i < shared_vars.num_concat_worker_cores; ++i) {
            const auto& core = shared_vars.cores[i];
            auto& concat_reader_runtime_args = GetRuntimeArgs(program, shared_vars.concat_reader_kernel_id, core);
            concat_reader_runtime_args[0] = q_start_addr;
            concat_reader_runtime_args[1] = input.buffer()->address();
        }
    }
}

}  // namespace ttnn::experimental::prim
