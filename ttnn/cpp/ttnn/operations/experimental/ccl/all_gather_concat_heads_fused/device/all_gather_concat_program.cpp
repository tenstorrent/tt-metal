// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/fabric.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

struct llama_config {
    CoreRange sem_drain_core = CoreRange({1, 0}, {1, 0});
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

uint32_t get_tile_offset_by_batch(uint32_t i, uint32_t face_h, uint32_t sub_tile_line_bytes) {
    if (i / 2 < face_h) {
        return i / 2 * sub_tile_line_bytes;
    }
    return (i / 2 + face_h) * sub_tile_line_bytes;
}
tt::tt_metal::operation::ProgramWithCallbacks all_gather_concat_llama_sharded(
    const Tensor& input_tensor,
    const Tensor& temp_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const uint32_t num_heads) {
    tt::tt_metal::Program program{};
    ttnn::MeshDevice* mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;

    TensorSpec output_intermediate_tensor_spec = temp_tensor.tensor_spec();
    auto output_interm_padded_shape = output_intermediate_tensor_spec.padded_shape();
    auto ring_core_ranges = output_tensor.shard_spec().value().grid.ranges();

    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
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
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, temp_tensors, topology);
    LineTopology line_topology(ring_size, ring_index);
    const size_t num_targets_forward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    const size_t num_targets_backward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);

    // To overlap NLP local data with all gather, we divide the batches for each device into:
    //      - local batch (starts with start_local)
    //      - remote batches
    //          - batch 1 (from batch_start_1 to batch end_1)
    //          - batch 2 (from batch_start_2 to batch end_2) if applicable
    uint32_t batch_size = 1;
    uint32_t batch_start_1 = 8;
    uint32_t batch_end_1 = 32;
    uint32_t batch_start_2 = 0;
    uint32_t batch_end_2 = 0;
    uint32_t start_local = 0;

    if (num_targets_forward == 2 && num_targets_backward == 1) {
        batch_size = 2;
        batch_start_1 = 0;
        batch_end_1 = 8;
        batch_start_2 = 16;
        batch_end_2 = 32;
        start_local = 8;
    } else if (num_targets_forward == 1 && num_targets_backward == 2) {
        batch_size = 2;
        batch_start_1 = 0;
        batch_end_1 = 16;
        batch_start_2 = 24;
        batch_end_2 = 32;
        start_local = 16;
    } else if (num_targets_forward == 0 && num_targets_backward == 3) {
        batch_size = 1;
        batch_start_1 = 0;
        batch_end_1 = 24;
        start_local = 24;
    }

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    auto sender_worker_core_range = CoreRangeSet(CoreRange({1, 0}, {num_links, 0}));
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range, num_links, true);
    // Tensor Info
    const uint32_t logical_dim_2 = input_tensor.get_logical_shape()[2];
    const auto input_tensor_num_pages =
        input_tensor.get_logical_shape()[0] * input_tensor.get_logical_shape()[1] * logical_dim_2;
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = logical_dim_2;

    const auto output_interm_tensor_cores = temp_tensor.memory_config().shard_spec()->grid;
    const auto output_interm_tensor_shard_shape = temp_tensor.memory_config().shard_spec()->shape;
    const auto output_interm_tensor_shard_num_pages = logical_dim_2;
    const auto row_size = input_tensor.get_padded_shape()[-1] / 2 * output_tensor.element_size();

    tt::log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    tt::log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    tt::log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);

    // concat info
    const auto& input_concat_shape = temp_tensor.get_padded_shape();
    const uint32_t head_dim = input_concat_shape[-1];
    const uint32_t batch = input_concat_shape[1];
    uint32_t single_tile_size =
        tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype()));

    auto tile_shape = temp_tensor.get_tensor_spec().tile().get_tile_shape();
    auto tile_h = tile_shape[0];
    auto tile_w = tile_shape[1];
    auto tile_hw = tile_h * tile_w;

    auto face_shape = temp_tensor.get_tensor_spec().tile().get_face_shape();
    auto face_h = face_shape[0];
    auto face_w = face_shape[1];
    auto face_hw = face_h * face_w;
    uint32_t first_phase = 1;
    uint32_t second_phase = 2;

    const uint32_t head_tiles = head_dim / tile_w;
    const uint32_t head_size = head_tiles * single_tile_size;
    const uint32_t tile_size = head_size / head_tiles;

    uint32_t element_size = temp_tensor.element_size();
    uint32_t sub_tile_line_bytes = face_w * element_size;
    auto q_shard_spec = output_tensor.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = temp_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;
    auto in_num_tiles = in_shard_spec.shape[0] * in_shard_spec.shape[1] / TILE_HW;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_config().channel_buffer_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        input_tensor_num_pages / num_links +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    uint32_t q_output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_q_output_config =
        tt::tt_metal::CircularBufferConfig(output_tensor.get_padded_shape()[-2] * row_size, {{q_output_cb_index, df}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_q_output = tt::tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t pre_tilize_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig cb_pre_tilize_config =
        tt::tt_metal::CircularBufferConfig(output_tensor.get_padded_shape()[-2] * row_size, {{pre_tilize_cb_index, df}})
            .set_page_size(pre_tilize_cb_index, single_tile_size);
    auto cb_pre_tilize = tt::tt_metal::CreateCircularBuffer(program, q_cores, cb_pre_tilize_config);

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
    sem_cores_vector.push_back(llama_configuration.sem_drain_core);
    range_count = 0;
    for (auto cr : ring_core_ranges) {
        sem_cores_vector.push_back(cr);
        range_count++;
        if (range_count == llama_configuration.concat_num_cores) {
            break;
        }
    }
    const auto& sem_cores_updated = CoreRangeSet(sem_cores_vector);
    uint32_t q_base_addr = temp_tensor.buffer()->address();
    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = tt::tt_metal::NOC::NOC_0;

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);
    const uint32_t num_tiles_per_core_concat = (head_tiles * batch) / in_num_cores;

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
            .compile_args = concat_reader_ct_args});

    std::vector<uint32_t> tilize_ct_args = {
        q_output_cb_index,
    };
    auto tilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "tilize_writer.cpp",
        q_cores_updated,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = tilize_ct_args});

    auto tilize_compute_kernel_id = tt::tt_metal::CreateKernel(
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
            .compile_args = all_gather_reader_ct_args});

    // Writer
    uint32_t out_ready_sem_wait_value = ring_size * num_links;
    std::vector<uint32_t> all_gather_writer_ct_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
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
            .compile_args = all_gather_writer_ct_args});

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_interm_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - 1 / ring_size;
    uint32_t start_core_index_for_device = output_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;

    TT_FATAL(
        output_cores_vec.size() % ring_size == 0 || output_cores_vec.size() == 1,
        "output sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution scheme",
        output_cores_vec.size(),
        ring_size);
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

    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        bool add_remainder = link == num_links - 1;
        uint32_t input_tile_id_start = link * base_pages_per_worker;
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + add_remainder * remainder;

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

        tt::log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        tt::log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        tt::log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        tt::log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        tt::log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        tt::log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        tt::log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        tt::log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        tt::log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
            TT_ASSERT(drain_sync_core.x == 19 && drain_sync_core.y == 18, "This op should run on a TG machine");
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            semaphore.address(),
            input_tensor_shard_num_pages,
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        std::vector<uint32_t> writer_rt_args = {
            temp_tensor.buffer()->address(),  // tensor_address0
            semaphore.address(),              // out_ready_sem_bank_addr (absolute address)
            input_tensor_shard_num_pages,
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
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
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_device->id(), forward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_device->id(), backward_device.value()->id(), link, program, {core}, writer_rt_args);
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
        uint32_t in_tile_offset_by_batch = get_tile_offset_by_batch(i, face_h, sub_tile_line_bytes);

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
            reader_runtime_args.reserve(6 + 2 * in_num_cores);
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
    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         semaphore,
         sender_worker_cores,
         num_concat_worker_cores,
         cb_q_output,
         cores,
         concat_reader_kernel_id,
         face_h,
         sub_tile_line_bytes](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& temp_tensor = input_tensors[1];
            const auto& output = output_tensors[0];
            auto dst_buffer_query = output.buffer();

            UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);

            auto semaphore = static_cast<const ttnn::AllGatherConcat*>(operation)->semaphore;
            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);

            uint32_t q_base_addr = temp_tensor.buffer()->address();
            uint32_t q_start_addr = q_base_addr;
            for (const auto& core : sender_worker_cores) {
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = semaphore.address();

                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = q_start_addr;
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }

            for (uint32_t i = 0; i < num_concat_worker_cores; ++i) {
                const auto& core = cores[i];
                auto& concat_reader_runtime_args = GetRuntimeArgs(program, concat_reader_kernel_id, core);
                concat_reader_runtime_args[0] = q_start_addr;
                concat_reader_runtime_args[1] = input.buffer()->address();
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
