// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
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

}  // namespace ttnn

namespace ttnn::experimental::prim {

namespace {

// Descriptor kernel indices, fixed by push order in create_descriptor.
constexpr uint32_t kReductionReader = 0;
constexpr uint32_t kReductionCompute = 1;
constexpr uint32_t kWorkerReader = 2;
constexpr uint32_t kWorkerWriter = 3;

static_assert(ttnn::ccl::AllReduceReaderCommonArgs::input == 0);
static_assert(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore == 0);

tt::tt_metal::KernelDescriptor::RTArgList uint32_rt_args(const std::vector<uint32_t>& args) {
    tt::tt_metal::KernelDescriptor::RTArgList list;
    list.reserve(args.size());
    for (uint32_t value : args) {
        list.push_back(value);
    }
    return list;
}

void emplace_uint32_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel, const CoreCoord& core, const std::vector<uint32_t>& args) {
    kernel.emplace_runtime_args(core, uint32_rt_args(args));
}

void emplace_uint32_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel, const CoreRangeSet& cores, const std::vector<uint32_t>& args) {
    const auto list = uint32_rt_args(args);
    for (const auto& core : tt::tt_metal::corerange_to_cores(cores, std::nullopt, true)) {
        kernel.emplace_runtime_args(core, list);
    }
}

}  // namespace

tt::tt_metal::ProgramDescriptor AllReduceAsyncMeshWorkloadFactory::create_descriptor(
    const AllReduceAsyncParams& operation_attributes,
    const AllReduceAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "AllReduceAsyncMeshWorkloadFactory::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

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

    tt::tt_metal::ProgramDescriptor desc;
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
        coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

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
    output_cores.reserve(sub_device_cores.ranges().size());
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
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // Reduction kernel setup
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    // TODO: Currently does not support output shards being split across multiple links
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    output_corerangeset_per_link.reserve(num_links);
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
    std::vector<uint32_t> output_tensor_pages_in_link;
    output_tensor_pages_in_link.reserve(num_links);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link.push_back(num_pages_this_link);
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    /*
        Overview of algorithm:

        - Output: each link gets assigned a start and end core index, since multiple links
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
    std::vector<uint32_t> input_tensor_tile_offset_per_link;
    input_tensor_tile_offset_per_link.reserve(num_links);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];

        // Get offset based on previous overflow
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link.push_back(input_tensor_tile_offset);

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
    std::vector<uint32_t> reduction_semaphore_ids;
    reduction_semaphore_ids.reserve(num_links);
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t sem_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
        reduction_semaphore_ids.push_back(sem_id);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_CB_size = reduction_CB_tiles * reduction_CB_single_tile_size;

    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = reduction_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduction_cb_index),
            .data_format = df,
            .page_size = reduction_CB_single_tile_size,
        }}},
        .buffer = buffer_tensor.buffer(),
    });

    /* out cb */
    uint32_t out_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(output_df);
    uint32_t out_CB_tiles = output_tensor_shard_num_pages;
    uint32_t out_CB_size = out_CB_tiles * out_CB_single_tile_size;

    uint32_t out_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = output_tensor_cores,  // TODO: This should be the output cores instead
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_index),
            .data_format = output_df,
            .page_size = out_CB_single_tile_size,
        }}},
        .buffer = output_tensor.buffer(),  // TODO: Remove once new cb attached for output
    });

    const std::vector<uint32_t> no_work_rt_args = {static_cast<uint32_t>(!has_work), 0u, 0u};

    // Create reduction dataflow kernel
    tt::tt_metal::KernelDescriptor reduction_reader_kernel_desc;
    reduction_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "reduction_receiver.cpp";
    reduction_reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reduction_reader_kernel_desc.core_ranges = output_cores_all;
    reduction_reader_kernel_desc.compile_time_args = {
        reduction_cb_index,  // reduction_cb_index
        reduction_CB_tiles,  // total_num_reduction_tiles
    };
    reduction_reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = use_noc1_only ? tt::tt_metal::NOC::NOC_1 : reader_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(reduction_reader_kernel_desc));
    if (!output_cores_unused.empty()) {
        emplace_uint32_runtime_args(desc.kernels[kReductionReader], output_cores_unused, no_work_rt_args);
    }

    // Create reduction dataflow kernel
    tt::tt_metal::KernelDescriptor reduction_compute_kernel_desc;
    reduction_compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp";
    reduction_compute_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reduction_compute_kernel_desc.core_ranges = output_cores_all;
    reduction_compute_kernel_desc.compile_time_args = {
        reduction_cb_index,  // reduction_cb_index
        out_cb_index,        // out_cb_index
    };
    tt::tt_metal::ComputeConfigDescriptor reduction_compute_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (operation_attributes.fp32_dest_acc) {
        // fp32 dest accumulation -> ring sum independent of ETH arrival order.
        reduction_compute_config.fp32_dest_acc_en = true;
        reduction_compute_config.dst_full_sync_en = true;
    }
    reduction_compute_kernel_desc.config = reduction_compute_config;
    desc.kernels.push_back(std::move(reduction_compute_kernel_desc));
    const std::vector<uint32_t> reduction_compute_rt_args = {1u, ring_size, output_tensor_shard_num_pages};
    emplace_uint32_runtime_args(desc.kernels[kReductionCompute], output_tensor_cores, reduction_compute_rt_args);
    if (!output_cores_unused.empty()) {
        emplace_uint32_runtime_args(desc.kernels[kReductionCompute], output_cores_unused, no_work_rt_args);
    }

    // Reader
    std::vector<uint32_t> reader_compile_args = {
        device_index,               // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    tt::tt_metal::KernelDescriptor worker_sender_reader_kernel_desc;
    worker_sender_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_reader.cpp";
    worker_sender_reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_reader_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    worker_sender_reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_reader_kernel_desc));

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
    tt::tt_metal::KernelDescriptor worker_sender_writer_kernel_desc;
    worker_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_writer.cpp";
    worker_sender_writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_writer_kernel_desc.core_ranges = sender_worker_core_range;
    worker_sender_writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    worker_sender_writer_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = writer_noc,
        .noc_mode = use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    desc.kernels.push_back(std::move(worker_sender_writer_kernel_desc));

    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];

        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];
        uint32_t output_first_core_tile_start_offset = 0;

        const uint32_t num_input_cores_in_link =
            input_cores_idx_per_link[link].second - input_cores_idx_per_link[link].first;
        std::vector<uint32_t> input_tensor_cores_x;
        input_tensor_cores_x.reserve(num_input_cores_in_link);
        std::vector<uint32_t> input_tensor_cores_y;
        input_tensor_cores_y.reserve(num_input_cores_in_link);
        std::vector<uint32_t> output_tensor_cores_x;
        output_tensor_cores_x.reserve(num_output_cores_in_link[link]);
        std::vector<uint32_t> output_tensor_cores_y;
        output_tensor_cores_y.reserve(num_output_cores_in_link[link]);
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
        emplace_uint32_runtime_args(desc.kernels[kWorkerReader], core, reader_rt_args);

        // Set writer runtime args
        const size_t num_mcast_ranges = output_corerangeset_per_link[link].ranges().size();
        std::vector<uint32_t> mcast_start_x;
        mcast_start_x.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_start_y;
        mcast_start_y.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_end_x;
        mcast_end_x.reserve(num_mcast_ranges);
        std::vector<uint32_t> mcast_end_y;
        mcast_end_y.reserve(num_mcast_ranges);

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
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                target_fabric_node_id, forward_device_fabric_node_id, link, desc, core, writer_rt_args);
        }

        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto backward_device_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                target_fabric_node_id, backward_device_fabric_node_id, link, desc, core, writer_rt_args);
        }

        emplace_uint32_runtime_args(desc.kernels[kWorkerWriter], core, writer_rt_args);

        // Set reduction worker runtime args
        std::vector<uint32_t> reduction_reader_rt_args = {
            has_work,
            reduction_semaphore_ids[link],  // reduction_semaphore_id
            out_ready_sem_wait_value,       // out_ready_sem_wait_value
        };
        emplace_uint32_runtime_args(
            desc.kernels[kReductionReader], output_corerangeset_per_link[link], reduction_reader_rt_args);
    }

    desc.kernels[kWorkerReader].emplace_common_runtime_args({input_tensor.buffer()});
    desc.kernels[kWorkerWriter].emplace_common_runtime_args(
        {static_cast<uint32_t>(semaphore.address())});  // smuggled-rta-ok: caller GlobalSemaphore excluded from the
                                                        // program-cache key; re-applied by override_runtime_arguments
    desc.kernels[kReductionReader].emplace_common_runtime_args(
        {static_cast<uint32_t>(semaphore.address())});  // smuggled-rta-ok: caller GlobalSemaphore excluded from the
                                                        // program-cache key; re-applied by override_runtime_arguments
    return desc;
}

void AllReduceAsyncMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const AllReduceAsyncParams& operation_attributes,
    const AllReduceAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    (void)coord;
    tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerReader)[ttnn::ccl::AllReduceReaderCommonArgs::input] =
        tensor_args.input_tensor.buffer()->address();
    tt::tt_metal::GetCommonRuntimeArgs(program, kWorkerWriter)[ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore] =
        static_cast<uint32_t>(operation_attributes.semaphore.address());
    tt::tt_metal::GetCommonRuntimeArgs(program, kReductionReader)[ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore] =
        static_cast<uint32_t>(operation_attributes.semaphore.address());

    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& indices = cb->buffer_indices();
        if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_1))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *tensor_args.buffer_tensor.buffer());
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_2))) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb->id(), *output_tensor.buffer());
        }
    }
}

}  // namespace ttnn::experimental::prim
