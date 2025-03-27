// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
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

void append_fabric_connection_rt_arguments(
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& connection,
    const CoreCoord& core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& writer_rt_args) {
    writer_rt_args.push_back(connection.has_value());
    if (connection.has_value()) {
        auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
        append_worker_to_fabric_edm_sender_rt_args(
            connection.value(),
            sender_worker_flow_control_semaphore_id,
            sender_worker_teardown_semaphore_id,
            sender_worker_buffer_index_semaphore_id,
            writer_rt_args);
    }
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_concat_llama_sharded(
    const Tensor& input_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    Tensor& temp_tensor,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode,
    const uint32_t num_heads) {
    tt::tt_metal::Program program{};
    const bool enable_async_output_tensor = false;
    TT_FATAL(
        enable_persistent_fabric_mode,
        "only persistent fabric mode is supported for all_gather_concat_llama_post_binary_matmul");

    IDevice* device = input_tensor.device();
    TensorSpec output_intermediate_tensor_spec = temp_tensor.tensor_spec();
    auto output_interm_padded_shape = output_intermediate_tensor_spec.padded_shape();

    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            forward_device.value_or(nullptr),
            backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric_mode,
            num_links);

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

    auto sender_worker_core_range = CoreRangeSet(CoreRange({1, 0}, {3, 0}));
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range, 3, true);
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;

    const auto output_interm_tensor_cores = temp_tensor.memory_config().shard_spec->grid;
    const auto output_interm_tensor_shard_shape = temp_tensor.memory_config().shard_spec->shape;
    const auto output_interm_tensor_shard_num_pages =
        output_interm_tensor_shard_shape[0] * output_interm_tensor_shard_shape[1] / TILE_HW;

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

    const uint32_t head_tiles = head_dim / tile_w;
    const uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = temp_tensor.element_size();
    uint32_t sub_tile_line_bytes = face_w * element_size;
    auto q_shard_spec = output_tensor.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = temp_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;
    auto in_num_tiles = in_shard_spec.shape[0] * in_shard_spec.shape[1] / TILE_HW;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
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
        tt::tt_metal::CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, df}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_q_output = tt::tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t tmp_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig tmp_cb_config =
        tt::tt_metal::CircularBufferConfig(2, {{tmp_cb_index, df}}).set_page_size(tmp_cb_index, 2);
    auto tmp_cb_handle = CreateCircularBuffer(program, q_cores, tmp_cb_config);

    uint32_t q_base_addr = temp_tensor.buffer()->address();
    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    auto core_range_1 = CoreRange({1, 1}, {3, 1});
    auto core_range_2 = CoreRange({1, 2}, {2, 2});
    const auto& q_cores_updated = CoreRangeSet(std::vector{core_range_1, core_range_2});
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = tt::tt_metal::NOC::NOC_0;

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores);
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_x_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).x);
        noc_y_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    // create concat semaphore for each link
    uint32_t concat_semaphore_id = tt::tt_metal::CreateSemaphore(program, q_cores, 0);

    std::vector<uint32_t> concat_reader_ct_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        1,  // read the first phase
        in_num_cores,
        face_h,
        face_hw,
        tmp_cb_index,
        batch_size,
        batch_start_1,
        batch_end_1,
        batch_start_2,
        batch_end_2,
        start_local};

    auto concat_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_concat_reader.cpp",
        q_cores_updated,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = concat_reader_ct_args});

    concat_reader_ct_args[6] = 2;
    auto concat_reader_2_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/"
        "llama_concat_reader.cpp",
        q_cores_updated,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = concat_reader_ct_args});

    // KERNEL CREATION
    // Reader

    std::vector<uint32_t> all_gather_reader_ct_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        1,  // read the first phase
        in_num_cores,
        face_h,
        face_hw,
        tmp_cb_index,
        batch_size,
        batch_start_1,
        batch_end_1,
        batch_start_2,
        batch_end_2,
        start_local};

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
    std::vector<uint32_t> all_gather_writer_ct_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        2,  // read the second phase
        in_num_cores,
        face_h,
        face_hw,
        tmp_cb_index,
        batch_size,
        batch_start_1,
        batch_end_1,
        batch_start_2,
        batch_end_2,
        start_local};

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
    for (uint32_t k = 0; k < 8; k++) {
        auto this_core = device->worker_core_from_logical_core(input_cores_vec[k]);
        nlp_local_core_x.push_back(this_core.x);
        nlp_local_core_y.push_back(this_core.y);
    }

    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

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
            auto this_core = device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / output_interm_tensor_shard_num_pages;
             i < (input_tile_id_end + output_interm_tensor_shard_num_pages - 1) / output_interm_tensor_shard_num_pages;
             i++) {
            auto this_core = device->worker_core_from_logical_core(output_cores_this_device[i]);
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
            drain_sync_core = device->worker_core_from_logical_core(core);
        }
        std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> forward_fabric_connection =
            !forward_device.has_value()
                ? std::nullopt
                : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> backward_fabric_connection =
            !backward_device.has_value()
                ? std::nullopt
                : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            0,
            input_tensor.buffer()->address(),  // tensor_address0
            semaphore.address(),
            // input_tensor_shard_num_pages,        // num_tiles_per_core
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
        reader_rt_args[0] = reader_rt_args.size();
        uint32_t in_tile_offset_by_batch_r =
            link < face_h ? link * sub_tile_line_bytes : (link + face_h) * sub_tile_line_bytes;
        reader_rt_args.push_back(in_tile_offset_by_batch_r);
        reader_rt_args.push_back(temp_tensor.buffer()->address());
        for (auto nocx : noc_x_coords) {
            reader_rt_args.push_back(nocx);
        }
        for (auto nocy : noc_y_coords) {
            reader_rt_args.push_back(nocy);
        }
        // reader_rt_args.push_back(ring_size * num_links);  // sem target value
        // reader_rt_args.push_back(drain_sync_core.x);
        // reader_rt_args.push_back(drain_sync_core.y);
        reader_rt_args.push_back(link == 0);
        reader_rt_args.push_back(concat_semaphore_id);
        reader_rt_args.insert(reader_rt_args.end(), nlp_local_core_x.begin(), nlp_local_core_x.end());
        reader_rt_args.insert(reader_rt_args.end(), nlp_local_core_y.begin(), nlp_local_core_y.end());

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        std::vector<uint32_t> writer_rt_args = {
            0,
            temp_tensor.buffer()->address(),  // tensor_address0
            semaphore.address(),              // out_ready_sem_bank_addr (absolute address)
            // output_interm_tensor_shard_num_pages,  // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            // drain_sync_core.x,                     // out_ready_sem_noc0_x
            // drain_sync_core.y,                     // out_ready_sem_noc0_y
            // out_ready_sem_wait_value,              // out_ready_sem_wait_value
            concat_semaphore_id,
        };
        auto sem_mcast_range_1 = CoreRange({2, 0}, {3, 0});
        auto sem_mcast_range_2 = CoreRange({1, 1}, {3, 1});
        auto sem_mcast_range_3 = CoreRange({1, 2}, {2, 2});
        auto sem_mcast_ranges = CoreRangeSet(std::vector{sem_mcast_range_1, sem_mcast_range_2, sem_mcast_range_3});
        std::vector<uint32_t> mcast_start_x;
        std::vector<uint32_t> mcast_start_y;
        std::vector<uint32_t> mcast_end_x;
        std::vector<uint32_t> mcast_end_y;

        for (const auto& range : sem_mcast_ranges.ranges()) {
            auto start_core = device->worker_core_from_logical_core(range.start_coord);
            auto end_core = device->worker_core_from_logical_core(range.end_coord);
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
        append_fabric_connection_rt_arguments(forward_fabric_connection, core, program, writer_rt_args);
        append_fabric_connection_rt_arguments(backward_fabric_connection, core, program, writer_rt_args);
        writer_rt_args[0] = writer_rt_args.size();
        uint32_t in_tile_offset_by_batch =
            link < face_h ? link * sub_tile_line_bytes : (link + face_h) * sub_tile_line_bytes;

        writer_rt_args.push_back(in_tile_offset_by_batch);
        writer_rt_args.push_back(temp_tensor.buffer()->address());
        for (auto nocx : noc_x_coords) {
            writer_rt_args.push_back(nocx);
        }
        for (auto nocy : noc_y_coords) {
            writer_rt_args.push_back(nocy);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    /* rt for concat kernels*/
    uint32_t q_start_addr = temp_tensor.buffer()->address();
    for (uint32_t i = 0; i < num_cores; ++i) {
        // in_tile_offset_by_batch is the start address of each batch in the input tile. The first face_h batches are in
        // the upper half of the tile and rest are in the lower half of tile.
        uint32_t in_tile_offset_by_batch = i < face_h ? i * sub_tile_line_bytes : (i + face_h) * sub_tile_line_bytes;

        const auto& core = cores[i];
        std::vector<uint32_t> input_cores_x;
        std::vector<uint32_t> input_cores_y;
        for (uint32_t k = 0; k < 8; k++) {
            auto this_core = device->worker_core_from_logical_core(input_cores_vec[k]);
            input_cores_x.push_back(this_core.x);
            input_cores_y.push_back(this_core.y);
        }
        bool is_worker_core = core.x == 1 && core.y == 0;
        is_worker_core = is_worker_core || (core.x == 2 && core.y == 0);
        is_worker_core = is_worker_core || (core.x == 3 && core.y == 0);
        if (is_worker_core == 0) {
            std::vector<uint32_t> reader_runtime_args;
            reader_runtime_args.reserve(3 + 2 * in_num_cores + 21);
            reader_runtime_args = {in_tile_offset_by_batch, q_start_addr, concat_semaphore_id};
            reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
            reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());
            reader_runtime_args.push_back(input_tensor.buffer()->address());

            tt::tt_metal::SetRuntimeArgs(program, concat_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, concat_reader_2_kernel_id, core, reader_runtime_args);
        }
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         semaphore,
         sender_worker_cores,
         num_cores,
         cb_q_output,
         cores,
         temp_tensor,
         concat_reader_kernel_id,
         concat_reader_2_kernel_id,
         face_h,
         sub_tile_line_bytes](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
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
            uint32_t idx = 0;
            for (const auto& core : sender_worker_cores) {
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[1] = input.buffer()->address();
                worker_reader_sender_runtime_args[2] = semaphore.address();
                uint32_t concat_args_index = worker_reader_sender_runtime_args[0];
                uint32_t in_tile_offset_by_batch =
                    idx < face_h ? idx * sub_tile_line_bytes : (idx + face_h) * sub_tile_line_bytes;

                worker_reader_sender_runtime_args[concat_args_index] = in_tile_offset_by_batch;
                worker_reader_sender_runtime_args[concat_args_index + 1] = q_start_addr;

                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[1] = q_start_addr;
                worker_writer_sender_runtime_args[2] = semaphore.address();
                uint32_t concat_args_index_2 = worker_writer_sender_runtime_args[0];

                worker_writer_sender_runtime_args[concat_args_index_2] = in_tile_offset_by_batch;
                worker_writer_sender_runtime_args[concat_args_index_2 + 1] = q_start_addr;
                idx++;
            }

            for (uint32_t i = 0; i < num_cores; ++i) {
                const auto& core = cores[i];
                bool is_worker_core = core.x == 1 && core.y == 0;
                is_worker_core = is_worker_core || (core.x == 2 && core.y == 0);
                is_worker_core = is_worker_core || (core.x == 3 && core.y == 0);
                if (is_worker_core == 0) {
                    uint32_t in_tile_offset_by_batch =
                        i < face_h ? i * sub_tile_line_bytes : (i + face_h) * sub_tile_line_bytes;
                    auto& concat_reader_runtime_args = GetRuntimeArgs(program, concat_reader_kernel_id, core);
                    concat_reader_runtime_args[0] = in_tile_offset_by_batch;
                    concat_reader_runtime_args[1] = q_start_addr;
                    concat_reader_runtime_args[concat_reader_runtime_args.size() - 1] = input.buffer()->address();

                    auto& concat_reader_2_runtime_args = GetRuntimeArgs(program, concat_reader_2_kernel_id, core);
                    concat_reader_2_runtime_args[0] = in_tile_offset_by_batch;
                    concat_reader_2_runtime_args[1] = q_start_addr;
                    concat_reader_2_runtime_args[concat_reader_2_runtime_args.size() - 1] = input.buffer()->address();
                }
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
