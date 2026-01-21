// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace detail {

uint32_t strided_all_gather_async_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    return (num_workers_per_direction + num_mux_cores_per_direction_per_link) * num_directions_per_link;
}

uint32_t strided_default_workers(
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    auto d_id = mesh_device.get_sub_device_ids().at(0);
    auto core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, d_id);
    uint32_t num_cores = core_range_set.num_cores();
    // Above 4 workers we start getting performance drops, so we limit to 4 workers or less, depending on the number of
    // available cores This was determined by the sweep
    // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
    ttnn::SmallVector<uint32_t> candidate_worker_counts;
    // if per link data moved is greater than 0.25 MB, we search greedily for 4 workers, otherwise we search greedily
    // for 2 workers. for ring, half the data is moved per link, so we divide by 2
    double data_moved_per_link_bytes = double(output_data_size_bytes) * (ring_size - 1) / ring_size / num_links /
                                       (topology == ttnn::ccl::Topology::Ring ? 2 : 1);
    if (data_moved_per_link_bytes > double(0.25 * 1024 * 1024)) {
        candidate_worker_counts = {4, 2, 1};
    } else {
        candidate_worker_counts = {2, 1};
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * strided_all_gather_async_core_count_per_link(
                            worker_count, num_directions_per_link, num_mux_cores_per_direction_per_link);
        if (num_cores >= core_count) {
            log_trace(
                tt::LogOp,
                "data_moved_per_link_bytes: {} and worker_count: {}",
                data_moved_per_link_bytes,
                worker_count);
            return worker_count;
        }
    }
    TT_THROW(
        "Not enough cores available on the subdevice or device for the requested match the number of links {}",
        num_links);
}

void strided_fabric_mux_connection_ct_args(
    const bool is_termination_master,
    const CoreCoord& mux_virtual_core,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    uint32_t worker_id,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& writer_ct_args) {
    writer_ct_args.push_back(is_termination_master);
    writer_ct_args.push_back(mux_virtual_core.x);
    writer_ct_args.push_back(mux_virtual_core.y);
    writer_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_status_address());
    writer_ct_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
}

void strided_fabric_mux_connection_rt_args(
    const bool& mux_connection_valid,
    const CoreCoord& worker_logical_core,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
    worker_rt_args.push_back(num_workers_per_direction);
}
}  // namespace detail

StridedAllGatherAsyncProgramFactory::cached_mesh_workload_t StridedAllGatherAsyncProgramFactory::create_mesh_workload(
    const StridedAllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void StridedAllGatherAsyncProgramFactory::override_runtime_arguments_per_program(
    const shared_variables_t& shared_variables,
    tt::tt_metal::Program& program,
    const StridedAllGatherAsyncParams& attributes,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& reader_kernel_ids = shared_variables.reader_kernel_ids;
    const auto& writer_kernel_ids = shared_variables.writer_kernel_ids;
    const auto& all_cores = shared_variables.all_cores;
    const auto& num_links = shared_variables.num_links;
    const auto& num_directions_per_link = shared_variables.num_directions_per_link;
    const auto& num_workers_per_direction = shared_variables.num_workers_per_direction;
    const auto& num_mux_cores_per_direction_per_link = shared_variables.num_mux_cores_per_direction_per_link;
    const auto& num_cores_per_link = shared_variables.num_cores_per_link;

    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;

    // update senders
    uint32_t core_idx = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                auto out_ready_semaphore = attributes.semaphore.at(dir);
                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = output.buffer()->address();
                worker_reader_sender_runtime_args[9] = out_ready_semaphore.address();
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = output.buffer()->address();
                worker_writer_sender_runtime_args[11] = out_ready_semaphore.address();

                core_idx++;
            }
        }
    }
}

void StridedAllGatherAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const StridedAllGatherAsyncParams& attributes,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        override_runtime_arguments_per_program(shared_variables, program, attributes, tensor_args, output_tensor);
    }
}

ttnn::device_operation::CachedProgram<StridedAllGatherAsyncProgramFactory::shared_variables_t>
StridedAllGatherAsyncProgramFactory::create_at(
    const StridedAllGatherAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    tt::tt_metal::Program program{};
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> empty_fused_op_signaler;
    return {
        std::move(program),
        strided_all_gather_async_minimal_default_helper(
            program,
            tensor_args.input_tensor,
            mesh_coordinate,
            forward_coord,
            backward_coord,
            output_tensor,
            attributes.dim,
            attributes.num_links,
            attributes.ring_size,
            device_index,
            attributes.topology,
            attributes.semaphore,
            empty_fused_op_signaler,
            false,
            attributes.tiles_per_chunk,
            attributes.num_workers_per_link,
            attributes.num_buffers_per_channel,
            attributes.mm_cores_y,
            attributes.mm_block_ht,
            attributes.mm_block_wt,
            CoreCoord(0, 0))};
}

StridedAllGatherAsyncProgramFactory::shared_variables_t
StridedAllGatherAsyncProgramFactory::strided_all_gather_async_minimal_default_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t /*dim*/,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
    bool read_local_slice_from_input,
    std::optional<uint32_t> tiles_per_chunk,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt,
    const CoreCoord core_grid_offset) {
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device not found");

    // op hyperparams
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    // Get worker cores
    // 2 senders (reader + writer) per direction (forward, reverse_order) per link
    uint32_t output_data_size_bytes = output_tensor.buffer()->size();
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(detail::strided_default_workers(
        *mesh_device,
        topology,
        output_data_size_bytes,
        num_links,
        ring_size,
        num_directions_per_link,
        num_mux_cores_per_direction_per_link));
    uint32_t num_cores_per_link = detail::strided_all_gather_async_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Need a separate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, mesh_device);

    const auto [all_core_range, all_cores] =
        ttnn::ccl::choose_worker_cores(num_links, num_cores_per_link, mesh_device, std::nullopt, core_grid_offset);
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    std::vector<CoreCoord> sender_forward_cores;
    std::vector<CoreCoord> sender_backward_cores;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_cores.push_back(worker_core);
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_cores.push_back(worker_core);
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // for bfloat8_b, tile_num_per_link=6, we would need to send 2 packages, but they can be of size 3 instead of 4
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t cb_num_pages = 3 * num_tiles_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

    uint32_t batch_head_size = input_tensor_shape[0] * input_tensor_shape[1];

    uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
    TT_FATAL(!(input_tensor_shape[3] % TILE_WIDTH), "Input tensor width must be a multiple of TILE_WIDTH");
    TT_FATAL(!(output_tensor_shape[3] % TILE_WIDTH), "Output tensor width must be a multiple of TILE_WIDTH");
    uint32_t TILE_WIDTH = 32;

    uint32_t input_tensor_Wt = input_tensor_shape[3] / TILE_WIDTH;
    uint32_t input_tensor_Ht = input_tensor_shape[2] / TILE_WIDTH;

    uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;
    uint32_t output_tensor_Ht = output_tensor_shape[2] / TILE_WIDTH;

    uint32_t tiles_per_chunk_val = tiles_per_chunk.value_or(0);
    uint32_t mm_cores_y_val = mm_cores_y.value_or(0);
    uint32_t mm_block_ht_val = mm_block_ht.value_or(0);
    uint32_t mm_block_wt_val = mm_block_wt.value_or(0);
    if (fuse_op) {
        tiles_per_chunk_val = mm_cores_y_val * mm_block_ht_val * mm_block_wt_val;
    }

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    // KERNEL CREATION
    /* All gather fusion */
    std::vector<std::vector<uint32_t>> device_chunk_widths(ring_size);
    std::vector<uint32_t> device_k_block_counts(ring_size, 0);
    uint32_t padded_K_tiles = tt::round_up(output_tensor_Wt, mm_block_wt_val);
    uint32_t K_blocks = padded_K_tiles / mm_block_wt_val;

    uint32_t curr_device = 0;
    uint32_t curr_device_end = input_tensor_Wt - 1;
    uint32_t device_max_chunks = 0;
    for (uint32_t k_block_iter = 0; k_block_iter < K_blocks; k_block_iter++) {
        uint32_t curr_k_block_start = k_block_iter * mm_block_wt_val;
        uint32_t curr_k_block_end = ((k_block_iter + 1) * mm_block_wt_val) - 1;
        if (curr_k_block_end < curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_k_block_end - curr_k_block_start + 1);
        } else if (curr_k_block_end == curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_k_block_end - curr_k_block_start + 1);
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        } else if (curr_k_block_end > curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_device_end - curr_k_block_start + 1);
            if (curr_device + 1 < ring_size) {
                device_k_block_counts[curr_device + 1]++;
                device_chunk_widths[curr_device + 1].push_back(curr_k_block_end - curr_device_end);
            }
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        }
    }
    for (uint32_t d = 0; d < ring_size; d++) {
        device_max_chunks = std::max(device_max_chunks, (uint32_t)device_chunk_widths[d].size());
    }

    if (fuse_op) {
        fused_op_signaler_forward->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_forward_cores);
        fused_op_signaler_backward->init_all_gather(
            program, mesh_device, sender_backward_core_ranges, sender_backward_cores);
        fused_op_signaler_sender_workers->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_forward_cores);
    }

    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            // Fabrix mux kernel
            uint32_t mux_core_offset = (link * num_cores_per_link) +
                                       (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
            auto num_full_size_channels = num_workers_per_direction;
            auto num_header_only_channels = 0;
            size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
                num_full_size_channels,
                num_header_only_channels,
                num_buffers_full_size_channels,
                0,
                buffer_size_bytes_full_size_channel,
                mux_base_l1_address);

            const bool mux_connection_valid =
                (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
            if (mux_connection_valid) {
                auto mux_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    {mux_logical_core},
                    tt::tt_metal::DataMovementConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                CoreCoord supplemental_core = all_cores
                    [(link * num_cores_per_link) +
                     ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction)) +
                     num_mux_cores_per_direction_per_link + worker];
                CoreCoord opposite_core_coord = mesh_device->worker_core_from_logical_core(supplemental_core);

                uint32_t global_worker_id = (link * num_workers_per_direction) + worker;
                uint32_t global_worker_count = num_links * num_workers_per_direction;
                uint32_t base_pages_per_worker = single_batch_head_num_pages / global_worker_count;
                uint32_t remainder = single_batch_head_num_pages % global_worker_count;
                uint32_t tiles_per_core = base_pages_per_worker + ((global_worker_id < remainder) ? 1 : 0);

                // Reader
                std::vector<uint32_t> sender_reader_compile_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_slices_forward_direction
                    num_targets_backward,             // num_slices_backward_direction
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    fuse_op,                          // fused op
                    tiles_per_chunk_val,
                    global_worker_count,
                    global_worker_id,
                };
                tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
                tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_reader_compile_args);
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/"
                    "minimal_default_reader.cpp",
                    {core},
                    tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),   // input_tensor_address
                    output_tensor.buffer()->address(),  // output_tensor_address
                    input_tensor_Wt,                    // width in tiles of the output shard
                    input_tensor_Ht,                    // height in tiles of the output shard
                    output_tensor_Wt,                   // width in tiles of entire output
                    batch_head_size,                    // product of the first two dims
                    global_worker_id,                   //
                    tiles_per_core,                     //
                    ring_size,                          // ring_size
                    semaphore.at(dir).address(),        // out_ready_semaphore_forward
                    mm_block_wt_val,
                    mm_block_ht_val,
                    mm_cores_y_val};
                reader_rt_args.push_back(device_max_chunks);
                for (uint32_t d = 0; d < ring_size; d++) {
                    reader_rt_args.push_back(device_k_block_counts[d]);
                    reader_rt_args.push_back(device_chunk_widths[d].size());
                    for (unsigned int width : device_chunk_widths[d]) {
                        reader_rt_args.push_back(width);
                    }
                }
                if (fuse_op) {
                    if (dir) {
                        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            1);
                    } else {
                        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            0);
                    }
                }

                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer
                std::vector<uint32_t> sender_writer_compile_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_targets_forward_direction
                    num_targets_backward,             // num_targets_backward_direction
                    fuse_op,                          // fused op
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    tiles_per_chunk_val,
                    global_worker_count,
                    global_worker_id,
                };
                detail::strided_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_compile_args);
                if (dir) {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
                } else {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
                }
                tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/"
                    "minimal_default_writer.cpp",
                    {core},
                    tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                std::vector<uint32_t> writer_rt_args = {
                    output_tensor.buffer()->address(),  // output_tensor_address
                    input_tensor_Wt,                    // width in tiles of the input shard
                    input_tensor_Ht,                    // height in tiles of the input shard
                    output_tensor_Wt,                   // width in tiles of entire output
                    output_tensor_Ht,                   // height in tiles of entire output
                    batch_head_size,                    // product of the first two dims
                    global_worker_id,                   //
                    tiles_per_core,                     //
                    virtual_core.x,                     // out_ready_sem_noc0_x
                    virtual_core.y,                     // out_ready_sem_noc0_y
                    ring_size,                          // ring_size
                    semaphore.at(dir).address(),        // out_ready_semaphore_forward
                    opposite_core_coord.x,
                    opposite_core_coord.y,
                    mm_block_wt_val,
                    mm_block_ht_val,
                    mm_cores_y_val,
                    read_local_slice_from_input};
                writer_rt_args.push_back(device_max_chunks);
                for (uint32_t d = 0; d < ring_size; d++) {
                    writer_rt_args.push_back(device_k_block_counts[d]);
                    writer_rt_args.push_back(device_chunk_widths[d].size());
                    for (unsigned int width : device_chunk_widths[d]) {
                        writer_rt_args.push_back(width);
                    }
                }
                detail::strided_fabric_mux_connection_rt_args(
                    mux_connection_valid,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    writer_rt_args);
                if (fuse_op) {
                    fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                        writer_rt_args,
                        num_workers_per_direction * num_links,
                        worker + (link * num_workers_per_direction),
                        2);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
            }
        }
    }

    return {
        reader_kernel_ids,
        writer_kernel_ids,
        all_cores,
        num_links,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

}  // namespace ttnn::experimental::prim
