// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
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

namespace ttnn {

CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    if (available_cores.num_cores() < num_workers_preferred) {
        log_warning(
            tt::LogOp,
            "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
            "performance loss.",
            num_workers_preferred,
            num_workers_per_link,
            num_links,
            available_cores.num_cores());
    }
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                sender_worker_core_range =
                    sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
            if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                break;
            }
        }
        if (sender_worker_core_range.num_cores() == num_workers_preferred) {
            break;
        }
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

namespace operations::experimental::ccl::all_reduce_async {

AllReduceAsyncMeshWorkloadFactory::cached_mesh_workload_t AllReduceAsyncMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllReduceAsyncMeshWorkloadFactory::cached_program_t AllReduceAsyncMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& coord,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;

    log_debug(tt::LogOp, "all_reduce_async create_program at physical coordinate {} is called", coord);

    uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);

    auto input_tensor_shape = input_tensor.padded_shape();

    auto input_tensor_memory_config = input_tensor.memory_config();
    auto output_tensor_memory_config = output_tensor.memory_config();
    [[maybe_unused]] uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec()->grid.num_cores();
    [[maybe_unused]] uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec()->grid.num_cores();

    log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    log_debug(
        tt::LogOp,
        "input_tensor_memory_config.shard_spec()->shape: {}",
        input_tensor_memory_config.shard_spec()->shape);
    log_debug(
        tt::LogOp,
        "output_tensor_memory_config.shard_spec()->shape: {}",
        output_tensor_memory_config.shard_spec()->shape);

    log_debug(tt::LogOp, "Running TG Llama specific all_reduce_async_minimal_multi_core_with_workers");
    // previously parameters from all_reduce_async_minimal_multi_core_with_workers
    const auto& output_dtype = operation_attributes.dtype;
    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& topology = operation_attributes.topology;
    const auto& semaphore = operation_attributes.semaphore;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    const auto& use_noc1_only = operation_attributes.use_noc1_only;
    const auto& use_optimal_ccl_for_llama = operation_attributes.use_optimal_ccl_for_llama;

    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = use_noc1_only ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = device_index == 0;
    [[maybe_unused]] bool is_last_chip = device_index == ring_size - 1;
    log_trace(
        tt::LogOp, "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}", coord, is_first_chip, is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, device_index, topology, true);
    auto [forward_args, backward_args] = ttnn::ccl::get_forward_backward_line_mcast_configuration(
        topology, coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // Tensor Info
    [[maybe_unused]] const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto num_input_cores = input_tensor_cores.num_cores();
    const auto output_tensor_num_pages = output_tensor.buffer()->num_pages();
    // Get only cores that have actual data
    const auto& output_tensor_original_corerangeset = output_tensor.memory_config().shard_spec()->grid;
    const auto& cores_with_data = output_tensor.buffer()->buffer_distribution_spec()->cores_with_data();

    // filter output_tensor_cores to only include cores that have data and preserve original order
    CoreRangeSet output_tensor_cores;
    if (cores_with_data.size() == output_tensor_original_corerangeset.num_cores()) {
        output_tensor_cores = output_tensor_original_corerangeset;
    } else {
        std::vector<CoreRange> output_core_ranges;
        output_core_ranges.reserve(cores_with_data.size());
        for (const auto& coord : cores_with_data) {
            output_core_ranges.emplace_back(coord, coord);
        }
        output_tensor_cores = CoreRangeSet(output_core_ranges);
    }
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;
    const auto num_output_cores = output_tensor_cores.num_cores();

    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));

    std::vector<CoreRange> output_cores;
    for (const auto& cr : sub_device_cores.ranges()) {
        const auto intersection = output_tensor_cores.intersection(cr);
        if (!intersection.empty()) {
            output_cores.push_back(intersection.bounding_box());
        }
    }
    // output_cores_all is the bounding box of the output_tensor_cores but respecting boundaries of subdevice grids
    CoreRangeSet output_cores_all(output_cores);

    CoreRangeSet reserved_cores = output_cores_all;
    auto available_cores = sub_device_cores.subtract(reserved_cores);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    CoreRangeSet sender_worker_core_range;
    std::vector<CoreCoord> sender_worker_cores;
    std::tie(sender_worker_core_range, sender_worker_cores) =
        use_optimal_ccl_for_llama ? llama_specific::get_custom_worker_core_placement(num_links)
                                  : ar_choose_worker_cores(num_links, num_workers_per_link, available_cores);

    constexpr bool has_work = true;

    // output_cores_unused is the cores that should do no work
    auto output_cores_unused = output_cores_all.subtract(output_tensor_cores);
    // all_cores is both sender and worker cores
    auto all_cores = output_cores_all.merge(sender_worker_core_range);

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
    uint32_t cb_num_pages = tt::div_up(output_tensor_cores.num_cores(), num_links) * output_tensor_shard_num_pages;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_df = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Reduction kernel setup
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
            ((end_core_idx - start_core_idx) * input_tensor_shard_num_pages) - input_tensor_tile_offset;

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
    auto reduction_reader_kernel_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = use_noc1_only ? tt::tt_metal::NOC::NOC_1 : reader_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
    reduction_reader_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        reduction_CB_tiles,  // total_num_reduction_tiles
    };
    auto reduction_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "reduction_receiver.cpp",
        output_cores_all,
        reduction_reader_kernel_config);
    if (!output_cores_unused.empty()) {
        tt::tt_metal::SetRuntimeArgs(program, reduction_reader_kernel_id, output_cores_unused, {!has_work, 0, 0, 0});
    }

    // Create reduction dataflow kernel
    auto reduction_kernel_config = tt::tt_metal::ComputeConfig{};
    reduction_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        out_cb_index,        // out_cb_index
    };
    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp",
        output_cores_all,
        reduction_kernel_config);
    tt::tt_metal::SetRuntimeArgs(
        program, reduction_kernel_id, output_tensor_cores, {1, ring_size, output_tensor_shard_num_pages});
    if (!output_cores_unused.empty()) {
        tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, output_cores_unused, {!has_work, 0, 0});
    }

    // Reader
    std::vector<uint32_t> reader_compile_args = {
        device_index,               // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .noc_mode =
                use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = reader_compile_args});

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        device_index,                     // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
    };
    writer_compile_args.insert(writer_compile_args.end(), forward_args.begin(), forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), backward_args.begin(), backward_args.end());
    log_trace(tt::LogOp, "Writer Compile Args:");
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .noc_mode =
                use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
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
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
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
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto forward_device_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }

        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto backward_device_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        // Set reduction worker runtime args
        std::vector<uint32_t> reduction_reader_rt_args = {
            has_work,
            reduction_semaphore_ids[link],  // reduction_semaphore_id
            semaphore.address(),            // global semaphore_address
            out_ready_sem_wait_value,       // out_ready_sem_wait_value
        };
        tt::tt_metal::SetRuntimeArgs(
            program, reduction_reader_kernel_id, output_corerangeset_per_link[link], reduction_reader_rt_args);
    }

    return {
        std::move(program),
        shared_variables_t{
            .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
            .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
            .reduction_reader_kernel_id = reduction_reader_kernel_id,
            .sender_worker_cores = sender_worker_cores,
            .output_tensor_cores = output_tensor_cores,
            .cb_out = cb_out,
            .cb_reduction = cb_reduction}};
}

void AllReduceAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        const auto& input_tensor = tensor_args.input_tensor;
        const auto& buffer_tensor = tensor_args.buffer_tensor;

        auto semaphore = operation_attributes.semaphore;

        // update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);
        auto& reduction_reader_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.reduction_reader_kernel_id);
        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input_tensor.buffer()->address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[1] = semaphore.address();
        }
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_out, *output_tensor.buffer());
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_reduction, *buffer_tensor.buffer());
        for (const auto& cr : shared_vars.output_tensor_cores.ranges()) {
            for (const auto& core : corerange_to_cores(cr, std::nullopt, true)) {
                auto& reduction_reader_runtime_args = reduction_reader_runtime_args_by_core[core.x][core.y];
                reduction_reader_runtime_args[2] = semaphore.address();
            }
        }
    }
}

}  // namespace operations::experimental::ccl::all_reduce_async

}  // namespace ttnn
