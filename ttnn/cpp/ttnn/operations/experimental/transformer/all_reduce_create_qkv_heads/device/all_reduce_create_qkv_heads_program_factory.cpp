// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "all_reduce_create_qkv_heads_program_factory.hpp"
#include <tt-metalium/fabric.hpp>

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks all_reduce_create_qkv_heads_minimal_multi_core_with_workers(
    const std::vector<Tensor>& input_tensors,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::vector<Tensor>& output_tensors,
    const DataType dtype,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim) {
    tt::tt_metal::Program program{};

    const Tensor& input_tensor = input_tensors[0];
    const Tensor& buffer_tensor = input_tensors[1];
    const Tensor& batch_offset_tensor = input_tensors[2];
    Tensor& output_tensor = output_tensors[0];
    Tensor& q_output_tensor = output_tensors[1];
    Tensor& k_output_tensor = output_tensors[2];
    Tensor& v_output_tensor = output_tensors[3];

    auto mesh_device = input_tensor.mesh_device();
    // For qkv heads fuse

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);

    const uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    const uint32_t head_tiles = head_dim / tt::constants::TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = output_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = q_output_tensor.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / tt::constants::TILE_HW;
    const auto k_shard_spec = k_output_tensor.shard_spec().value();
    const auto k_cores = k_shard_spec.grid;
    const auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / tt::constants::TILE_HW;
    const auto v_shard_spec = v_output_tensor.shard_spec().value();
    const auto v_cores = v_shard_spec.grid;
    const auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / tt::constants::TILE_HW;
    const auto in_shard_spec = output_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;
    const auto in_num_tiles = in_shard_spec.shape[0] * in_shard_spec.shape[1] / tt::constants::TILE_HW;
    uint32_t batch_offset_index_stick_size = 0;
    // auto qk_cores = q_cores;

    auto qk_cores_set = std::set<CoreRange>();
    qk_cores_set.insert(q_cores.ranges().begin(), q_cores.ranges().end());
    qk_cores_set.insert(k_cores.ranges().begin(), k_cores.ranges().end());
    auto qk_cores = CoreRangeSet(qk_cores_set);

    //  Create CBs for reader/writer for batch_offset
    uint32_t batch_offset_cb_index_reader = tt::CBIndex::c_15;

    tt::DataFormat cb_batch_offset_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(batch_offset_tensor.dtype());
    uint32_t single_batch_offset_tile_size = tt::tt_metal::detail::TileSize(cb_batch_offset_data_format);
    batch_offset_index_stick_size = batch_offset_tensor.buffer()->aligned_page_size();

    tt::tt_metal::CircularBufferConfig cb_batch_offset_config_reader =
        tt::tt_metal::CircularBufferConfig(
            single_batch_offset_tile_size, {{batch_offset_cb_index_reader, cb_batch_offset_data_format}})
            .set_page_size(batch_offset_cb_index_reader, 1);
    tt::tt_metal::CreateCircularBuffer(
        program, output_tensor.memory_config().shard_spec()->grid, cb_batch_offset_config_reader);

    uint32_t q_base_addr = q_output_tensor.buffer()->address();
    uint32_t k_base_addr = k_output_tensor.buffer()->address();
    uint32_t v_base_addr = v_output_tensor.buffer()->address();

    // cores for q
    const uint32_t q_num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& q_cores_vector = corerange_to_cores(q_cores, q_num_cores, true);

    // cores for k
    const uint32_t k_num_cores = k_cores.num_cores();  // number of cores of the output
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for v
    const uint32_t v_num_cores = v_cores.num_cores();  // number of cores of the output
    const auto& v_cores_vector = corerange_to_cores(v_cores, v_num_cores, true);

    TT_FATAL(
        q_num_cores == k_num_cores && k_num_cores == v_num_cores,
        "Output q/k/v must have the same number of cores, q_num_cores: {}, k_num_cores: {}, v_num_cores: {}",
        q_num_cores,
        k_num_cores,
        v_num_cores);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    auto in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> qcores_noc_x_coords, qcores_noc_y_coords;
    std::vector<uint32_t> kcores_noc_x_coords, kcores_noc_y_coords;
    std::vector<uint32_t> vcores_noc_x_coords, vcores_noc_y_coords;
    qcores_noc_x_coords.reserve(q_cores_vector.size());
    qcores_noc_y_coords.reserve(q_cores_vector.size());
    kcores_noc_x_coords.reserve(k_cores_vector.size());
    kcores_noc_y_coords.reserve(k_cores_vector.size());
    vcores_noc_x_coords.reserve(v_cores_vector.size());
    vcores_noc_y_coords.reserve(v_cores_vector.size());
    for (uint32_t i = 0; i < q_cores_vector.size(); i++) {
        auto worker_core = mesh_device->worker_core_from_logical_core(q_cores_vector[i]);
        qcores_noc_x_coords.push_back(worker_core.x);
        qcores_noc_y_coords.push_back(worker_core.y);
    }
    for (uint32_t i = 0; i < k_cores_vector.size(); i++) {
        auto worker_core = mesh_device->worker_core_from_logical_core(k_cores_vector[i]);
        kcores_noc_x_coords.push_back(worker_core.x);
        kcores_noc_y_coords.push_back(worker_core.y);
    }
    for (uint32_t i = 0; i < v_cores_vector.size(); i++) {
        auto worker_core = mesh_device->worker_core_from_logical_core(v_cores_vector[i]);
        vcores_noc_x_coords.push_back(worker_core.x);
        vcores_noc_y_coords.push_back(worker_core.y);
    }

    uint32_t process_qv = 1, process_k = 0;

    // End of qkv heads fuse

    // TODO: Remove this once we have a way to get the number of cores per link
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        target_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors_ccl = {input_tensor};
    std::vector<Tensor> output_tensors_ccl = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors_ccl, output_tensors_ccl, topology);
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == ccl::Topology::Linear) {
        LineTopology line_topology(ring_size, ring_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        if (ring_index % 2 == 0) {
            std::swap(num_targets_forward, num_targets_backward);
        }
    }
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages =
        input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / tt::constants::TILE_HW;
    const auto num_input_cores = input_tensor_cores.num_cores();
    const auto output_tensor_num_pages = output_tensor.buffer()->num_pages();
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages =
        output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / tt::constants::TILE_HW;
    const auto num_output_cores = output_tensor_cores.num_cores();

    // Get worker cores, assuming 1 worker per link
    std::optional<CoreRangeSet> reserved_cores = output_tensor_cores;
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores_fuse(num_links, num_workers_per_link, mesh_device, sub_device_id, reserved_cores);

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = input_tensor_num_pages;  // TODO: Reduce this to double-buffer packet-size?
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_df = tt::tt_metal::datatype_to_dataformat_converter(dtype);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Reduction kernel setup
    auto all_cores = output_tensor_cores.merge(sender_worker_core_range);
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    // TODO: Currently does not support output shards being split across multiple links
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    std::vector<uint32_t> num_output_cores_in_link(num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_cores_this_link = std::min(output_cores_per_link, num_output_cores - num_assigned_cores);
        output_corerangeset_per_link.emplace_back(
            cores_to_corerangeset(std::vector<CoreCoord>(
                                      output_cores_vec.begin() + num_assigned_cores,
                                      output_cores_vec.begin() + num_assigned_cores + num_cores_this_link))
                .merge_ranges());
        num_output_cores_in_link[link] = num_cores_this_link;
        num_assigned_cores += num_cores_this_link;
    }

    // Create output tensor page splits
    std::vector<uint32_t> output_tensor_pages_in_link(num_links, 0);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link[link] = num_pages_this_link;
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    /*
        Overview of algorithm:

        - Ouput: each link gets assigned a start and end core index, since multiple links
            may have to read different offesets within a shard on the same core
        - First, assign all the necessary cores needed for a link. This may result in the link
            containing extra pages. This will result in an overflow, which is used to detect
            the tile offset (within a shard) for the next link
        - Once you have the start_core_idx, the end_core_idx is calculated by
            getting the upper bound on the number of cores needed to read the pages assigned
            to the link, accounting for the tile offset. This calculation is done by dividing
            the upper bound on the number of pages assigned to this link
            (num_pages_this_link + input_tensor_tile_offset) by the number of pages in a shard.
            This gives the number of cores needed to read the pages assigned to this link.
        - If an overflow is detected, then the start_core_idx for the next link is set
            to the end_core_idx of the current link. Ie, 2 links read from the same core
    */
    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link(num_links, 0);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];

        // Get offset based on previous overflow
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link[link] = input_tensor_tile_offset;

        uint32_t end_core_idx = std::min(
            start_core_idx + tt::div_up(num_pages_this_link + input_tensor_tile_offset, input_tensor_shard_num_pages),
            num_input_cores);

        // Num pages allocated based on number of input cores selected for this link
        uint32_t num_pages_allocated =
            (end_core_idx - start_core_idx) * input_tensor_shard_num_pages - input_tensor_tile_offset;

        // Update overflow
        num_pages_overflow = num_pages_allocated - num_pages_this_link;

        // Store core indices
        input_cores_idx_per_link[link] = {start_core_idx, end_core_idx};

        // Set start index based on overflow
        if (num_pages_overflow > 0) {
            start_core_idx = end_core_idx - 1;
        } else {
            start_core_idx = end_core_idx;
        }
    }

    // Create reduction semaphores for each link
    std::vector<uint32_t> reduction_semaphore_ids(num_links, 0);
    for (uint32_t link = 0; link < num_links; link++) {
        reduction_semaphore_ids[link] = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_CB_size = reduction_CB_tiles * reduction_CB_single_tile_size;

    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig reduction_cb_config =
        tt::tt_metal::CircularBufferConfig(reduction_CB_size, {{reduction_cb_index, df}})
            .set_page_size(reduction_cb_index, reduction_CB_single_tile_size)
            .set_globally_allocated_address(*buffer_tensor.buffer());
    auto cb_reduction = tt::tt_metal::CreateCircularBuffer(program, all_cores, reduction_cb_config);

    /* out cb */
    uint32_t out_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(output_df);
    uint32_t out_CB_tiles = output_tensor_shard_num_pages;
    uint32_t out_CB_size = out_CB_tiles * out_CB_single_tile_size;

    uint32_t out_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{out_cb_index, output_df}})
            .set_page_size(out_cb_index, out_CB_single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());  // TODO: Remove once new cb attached for output
    auto cb_out = tt::tt_metal::CreateCircularBuffer(
        program, output_tensor_cores, out_cb_config);  // TODO: This should be the output cores instead

    // Create reduction dataflow kernel
    std::vector<uint32_t> reader_compile_time_args = {
        reduction_cb_index,  // reduction_cb_index
        reduction_CB_tiles,  // total_num_reduction_tiles
        // qkv heads reader compile time args
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        1,  // read the first phase
        in_num_cores,
        q_num_cores,
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader,
        out_cb_index,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        reduction_cb_index,  // reduction_cb_index
        reduction_CB_tiles,  // total_num_reduction_tiles
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        2,  // read the second phase
        in_num_cores,
        q_num_cores,
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader,
        out_cb_index,
    };

    auto reduction_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args);
    auto reduction_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args);

    auto reduction_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/"
        "reduction_receiver.cpp",
        output_tensor_cores,
        reduction_reader_kernel_config);

    auto reduction_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/"
        "reduction_receiver.cpp",
        output_tensor_cores,
        reduction_writer_kernel_config);

    // Create reduction dataflow kernel
    auto reduction_kernel_config = tt::tt_metal::ComputeConfig{};
    reduction_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        out_cb_index,        // out_cb_index
    };
    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/"
        "reduction.cpp",
        output_tensor_cores,
        reduction_kernel_config);
    std::vector<uint32_t> reduction_kernel_rt_args = {
        ring_size,                      // num_blocks
        output_tensor_shard_num_pages,  // block_num_tiles
    };
    tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, output_tensor_cores, reduction_kernel_rt_args);

    // Now prepare rt args for the reader and writer kernels

    std::vector<uint32_t> reader_writer_runtime_args_template;
    reader_writer_runtime_args_template.reserve(7 + 2 * q_num_cores + 2 * k_num_cores + 2 * v_num_cores);
    reader_writer_runtime_args_template = {
        q_base_addr,
        k_base_addr,
        v_base_addr,
        batch_offset_tensor.buffer()->address(),
        0,
        output_tensor_shard_num_pages};
    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), qcores_noc_x_coords.begin(), qcores_noc_x_coords.end());
    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), qcores_noc_y_coords.begin(), qcores_noc_y_coords.end());

    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), kcores_noc_x_coords.begin(), kcores_noc_x_coords.end());
    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), kcores_noc_y_coords.begin(), kcores_noc_y_coords.end());

    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), vcores_noc_x_coords.begin(), vcores_noc_x_coords.end());
    reader_writer_runtime_args_template.insert(
        reader_writer_runtime_args_template.end(), vcores_noc_y_coords.begin(), vcores_noc_y_coords.end());
    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = tt::tt_metal::NOC::NOC_0;
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/"
        "worker_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_args});

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/"
        "worker_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_args});

    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];

        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];
        uint32_t output_first_core_tile_start_offset = 0;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_cores_idx_per_link[link].first; i < input_cores_idx_per_link[link].second; i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = output_cores_per_link * link;
             i < output_cores_per_link * link + num_output_cores_in_link[link];
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_vec[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
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
        std::vector<uint32_t> mcast_start_x;
        std::vector<uint32_t> mcast_start_y;
        std::vector<uint32_t> mcast_end_x;
        std::vector<uint32_t> mcast_end_y;

        uint32_t num_mcast_cores = 0;
        for (const auto& range : output_corerangeset_per_link[link].ranges()) {
            auto start_core = mesh_device->worker_core_from_logical_core(range.start_coord);
            auto end_core = mesh_device->worker_core_from_logical_core(range.end_coord);
            num_mcast_cores += (end_core.x - start_core.x + 1) * (end_core.y - start_core.y + 1);
            bool mcast_range_contains_self =
                start_core.x <= core.x && core.x <= end_core.x && start_core.y <= core.y && core.y <= end_core.y;
            if (mcast_range_contains_self) {
                num_mcast_cores -= 1;
            }
            if (writer_noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(start_core, end_core);
            }
            mcast_start_x.push_back(start_core.x);
            mcast_start_y.push_back(start_core.y);
            mcast_end_x.push_back(end_core.x);
            mcast_end_y.push_back(end_core.y);
        }

        uint32_t out_ready_sem_wait_value = ring_size;
        std::vector<uint32_t> writer_rt_args = {
            reduction_cb_index,                   // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            num_mcast_cores,                      // num_mcast_cores
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            reduction_semaphore_ids[link],        // reduction_semaphore_id
            mcast_start_x.size(),                 // num_mcast_ranges
            link,                                 // link
        };
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

        // Set reduction worker runtime args
        std::vector<uint32_t> reduction_reader_rt_args(reader_writer_runtime_args_template);
        std::vector<uint32_t> reduction_writer_rt_args(reader_writer_runtime_args_template);
        reduction_reader_rt_args.push_back(reduction_semaphore_ids[link]);
        reduction_writer_rt_args.push_back(reduction_semaphore_ids[link]);
        tt::tt_metal::SetRuntimeArgs(
            program, reduction_reader_kernel_id, output_corerangeset_per_link[link], reduction_reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(
            program, reduction_writer_kernel_id, output_corerangeset_per_link[link], reduction_writer_rt_args);
    }

    auto& reduction_reader_args_by_core = GetRuntimeArgs(program, reduction_reader_kernel_id);
    auto& reduction_writer_args_by_core = GetRuntimeArgs(program, reduction_writer_kernel_id);

    for (uint32_t i = 0; i < in_num_cores; i++) {
        const auto& core = in_cores_vec[i];
        auto& reader_args = reduction_reader_args_by_core[core.x][core.y];
        reader_args[4] = i;
        auto& writer_args = reduction_writer_args_by_core[core.x][core.y];
        writer_args[4] = i;
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         sender_worker_cores,
         cb_out,
         cb_reduction,
         output_cores_vec,
         reduction_reader_kernel_id,
         reduction_writer_kernel_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& buffer_tensor = input_tensors[1];
            const auto& batch_tensor = input_tensors[2];
            const auto& output = output_tensors[0];
            const auto& q_output = output_tensors[1];
            const auto& k_output = output_tensors[2];
            const auto& v_output = output_tensors[3];
            auto q_base_addr = q_output.buffer()->address();
            auto k_base_addr = k_output.buffer()->address();
            auto v_base_addr = v_output.buffer()->address();
            auto batch_base_addr = batch_tensor.buffer()->address();

            auto semaphore = static_cast<const ttnn::AllReduceCreateQkvHeads*>(operation)->semaphore;

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }

            auto& reduction_reader_args_by_core = GetRuntimeArgs(program, reduction_reader_kernel_id);
            auto& reduction_writer_args_by_core = GetRuntimeArgs(program, reduction_writer_kernel_id);

            for (uint32_t i = 0; i < output_cores_vec.size(); i++) {
                const auto& core = output_cores_vec[i];
                auto& reader_args = reduction_reader_args_by_core[core.x][core.y];
                reader_args[0] = q_base_addr;
                reader_args[1] = k_base_addr;
                reader_args[2] = v_base_addr;
                reader_args[3] = batch_base_addr;
                auto& writer_args = reduction_writer_args_by_core[core.x][core.y];
                writer_args[0] = q_base_addr;
                writer_args[1] = k_base_addr;
                writer_args[2] = v_base_addr;
                writer_args[3] = batch_base_addr;
            }
            UpdateDynamicCircularBufferAddress(program, cb_out, *output.buffer());
            UpdateDynamicCircularBufferAddress(program, cb_reduction, *buffer_tensor.buffer());
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
