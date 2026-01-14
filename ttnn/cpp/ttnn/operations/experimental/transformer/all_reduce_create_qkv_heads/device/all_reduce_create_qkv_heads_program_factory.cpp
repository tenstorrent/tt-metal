// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads_program_factory.hpp"
#include "all_reduce_create_qkv_heads_device_operation.hpp"

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>

#include <algorithm>
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

namespace ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads::program {

using namespace ttnn::ccl;

namespace {

// Helper function to convert cores to corerangeset
CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(std::move(core_ranges));
}

// Helper function to choose worker cores for fused operations
std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores_fuse(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<CoreRangeSet>& reserved_core_range = std::nullopt) {
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    auto available_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    if (reserved_core_range.has_value()) {
        available_cores = available_cores.subtract(*reserved_core_range);
    }
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

}  // namespace

AllReduceCreateQkvHeadsMeshWorkloadFactory::cached_mesh_workload_t
AllReduceCreateQkvHeadsMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
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

ttnn::device_operation::CachedProgram<AllReduceCreateQkvHeadsSharedVariables>
AllReduceCreateQkvHeadsMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    log_debug(tt::LogOp, "AllReduceCreateQkvHeadsMeshWorkloadFactory::create_at called");

    tt::tt_metal::Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;
    const auto& batch_offset_tensor = tensor_args.batch_offset_tensor;
    auto& output_tensor = tensor_return_value.all_reduce;
    auto& q_output_tensor = tensor_return_value.q;
    auto& k_output_tensor = tensor_return_value.k;
    auto& v_output_tensor = tensor_return_value.v;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto* const target_device = mesh_device->get_device(mesh_coord);
    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coord[1])
                                        : mesh_view.get_devices_on_row(mesh_coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices.at(i) == target_device) {
            device_index = i;
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
    log_debug(tt::LogOp, "Running TG Llama specific all_reduce_create_qkv_heads_minimal_multi_core_with_workers");
    log_debug(
        tt::LogOp,
        "AllReduceCreateQkvHeads: device_index={}, ring_size={}, num_links={}, topology={}",
        device_index,
        operation_attributes.ring_size,
        operation_attributes.num_links,
        static_cast<int>(operation_attributes.topology));

    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc =
        operation_attributes.use_noc1_only ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // For qkv heads fuse
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(operation_attributes.dtype);

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t head_tiles = operation_attributes.head_dim / tt::constants::TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = output_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = q_output_tensor.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto k_shard_spec = k_output_tensor.shard_spec().value();
    const auto k_cores = k_shard_spec.grid;
    const auto v_shard_spec = v_output_tensor.shard_spec().value();
    const auto v_cores = v_shard_spec.grid;
    const auto in_shard_spec = output_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;
    uint32_t batch_offset_index_stick_size = 0;

    auto qk_cores_set = std::set<CoreRange>();
    qk_cores_set.insert(q_cores.ranges().begin(), q_cores.ranges().end());
    qk_cores_set.insert(k_cores.ranges().begin(), k_cores.ranges().end());
    auto qk_cores = CoreRangeSet(qk_cores_set);

    // Create CBs for reader/writer for batch_offset
    uint32_t batch_offset_cb_index_reader = tt::CBIndex::c_15;

    tt::DataFormat cb_batch_offset_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(batch_offset_tensor.dtype());
    uint32_t single_batch_offset_tile_size = tt::tile_size(cb_batch_offset_data_format);
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

    for (auto core : q_cores_vector) {
        auto worker_core = mesh_device->worker_core_from_logical_core(core);
        qcores_noc_x_coords.push_back(worker_core.x);
        qcores_noc_y_coords.push_back(worker_core.y);
    }
    for (auto core : k_cores_vector) {
        auto worker_core = mesh_device->worker_core_from_logical_core(core);
        kcores_noc_x_coords.push_back(worker_core.x);
        kcores_noc_y_coords.push_back(worker_core.y);
    }
    for (auto core : v_cores_vector) {
        auto worker_core = mesh_device->worker_core_from_logical_core(core);
        vcores_noc_x_coords.push_back(worker_core.x);
        vcores_noc_y_coords.push_back(worker_core.y);
    }

    // End of qkv heads fuse

    // TODO: Remove this once we have a way to get the number of cores per link
    [[maybe_unused]] bool is_first_chip = device_index == 0;
    [[maybe_unused]] bool is_last_chip = device_index == operation_attributes.ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        target_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors_ccl = {input_tensor};
    std::vector<Tensor> output_tensors_ccl = {output_tensor};
    const auto& op_config =
        ttnn::ccl::CCLOpConfig(input_tensors_ccl, output_tensors_ccl, operation_attributes.topology);
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;

    if (operation_attributes.topology == Topology::Linear) {
        LineTopology line_topology(operation_attributes.ring_size, device_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (operation_attributes.topology == Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(operation_attributes.ring_size - 1, 2);
        num_targets_backward = operation_attributes.ring_size - 1 - num_targets_forward;
        if (device_index % 2 == 0) {
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
    const auto [sender_worker_core_range, sender_worker_cores] = choose_worker_cores_fuse(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        reserved_cores);

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
    tt::DataFormat output_df = tt::tt_metal::datatype_to_dataformat_converter(operation_attributes.dtype);

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
    auto all_cores = output_tensor_cores.merge(sender_worker_core_range);
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    // TODO: Currently does not support output shards being split across multiple links
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    std::vector<uint32_t> num_output_cores_in_link(operation_attributes.num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), operation_attributes.num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
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
    std::vector<uint32_t> output_tensor_pages_in_link(operation_attributes.num_links, 0);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link[link] = num_pages_this_link;
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    /*
        Overview of algorithm:

        - Output: each link gets assigned a start and end core index, since multiple links
            may have to read different offsets within a shard on the same core
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
    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(operation_attributes.num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link(operation_attributes.num_links, 0);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
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
    std::vector<uint32_t> reduction_semaphore_ids(operation_attributes.num_links, 0);
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        reduction_semaphore_ids[link] = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * operation_attributes.ring_size;
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
        reduction_cb_index,
        reduction_CB_tiles,  // total_num_reduction_tiles
        // qkv heads reader compile time args
        element_size,
        sub_tile_line_bytes,
        head_size,
        operation_attributes.num_heads,
        operation_attributes.num_kv_heads,
        head_tiles,
        1,  // read the first phase
        in_num_cores,
        q_num_cores,
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader,
        out_cb_index,
    };
    tt::tt_metal::TensorAccessorArgs(batch_offset_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        reduction_cb_index,
        reduction_CB_tiles,  // total_num_reduction_tiles
        element_size,
        sub_tile_line_bytes,
        head_size,
        operation_attributes.num_heads,
        operation_attributes.num_kv_heads,
        head_tiles,
        2,  // read the second phase
        in_num_cores,
        q_num_cores,
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader,
        out_cb_index,
    };
    tt::tt_metal::TensorAccessorArgs(batch_offset_tensor.buffer()).append_to(writer_compile_time_args);

    auto reduction_reader_kernel_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = operation_attributes.use_noc1_only ? tt::tt_metal::NOC::NOC_1 : reader_noc,
        .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                       : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(reader_compile_time_args)};

    auto reduction_writer_kernel_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = operation_attributes.use_noc1_only ? tt::tt_metal::NOC::NOC_1 : writer_noc,
        .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                       : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(writer_compile_time_args)};

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

    // Create reduction compute kernel
    auto reduction_kernel_config = tt::tt_metal::ComputeConfig{};
    reduction_kernel_config.compile_args = {
        reduction_cb_index,
        out_cb_index,
    };
    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/compute/"
        "reduction.cpp",
        output_tensor_cores,
        reduction_kernel_config);
    std::vector<uint32_t> reduction_kernel_rt_args = {
        operation_attributes.ring_size,  // num_blocks
        output_tensor_shard_num_pages,   // block_num_tiles
    };
    tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, output_tensor_cores, reduction_kernel_rt_args);

    // Now prepare rt args for the reader and writer kernels

    std::vector<uint32_t> reader_writer_runtime_args_template;
    reader_writer_runtime_args_template.reserve(7 + (2 * q_num_cores) + (2 * k_num_cores) + (2 * v_num_cores));
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

    // Reader
    std::vector<uint32_t> reader_compile_args = {
        device_index,               // my_chip_id
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
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
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
    log_trace(tt::LogOp, "Writer Compile Args:");
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/"
        "worker_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .noc_mode = operation_attributes.use_noc1_only ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC
                                                           : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = writer_compile_args});

    // Kernel Runtime Args
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
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

        uint32_t out_ready_sem_wait_value = operation_attributes.ring_size;
        std::vector<uint32_t> writer_rt_args = {
            reduction_cb_index,                        // tensor_address0
            operation_attributes.semaphore.address(),  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,             // num_tiles_per_core
            worker_num_tiles_to_read,                  // num_tiles_to_read
            output_first_core_tile_start_offset,       // first_core_tile_start_offset
            output_tensor_cores_x.size(),              // num_cores
            num_mcast_cores,                           // num_mcast_cores
            drain_sync_core.x,                         // out_ready_sem_noc0_x
            drain_sync_core.y,                         // out_ready_sem_noc0_y
            out_ready_sem_wait_value,                  // out_ready_sem_wait_value
            reduction_semaphore_ids[link],             // reduction_semaphore_id
            mcast_start_x.size(),                      // num_mcast_ranges
            link,                                      // link
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

    return ttnn::device_operation::CachedProgram<shared_variables_t>{
        std::move(program),
        {.worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
         .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
         .sender_worker_cores = sender_worker_cores,
         .cb_out = cb_out,
         .cb_reduction = cb_reduction,
         .output_cores_vec = output_cores_vec,
         .reduction_reader_kernel_id = reduction_reader_kernel_id,
         .reduction_writer_kernel_id = reduction_writer_kernel_id}};
}

void AllReduceCreateQkvHeadsMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;
    const auto& batch_tensor = tensor_args.batch_offset_tensor;
    const auto& output = tensor_return_value.all_reduce;
    const auto& q_output = tensor_return_value.q;
    const auto& k_output = tensor_return_value.k;
    const auto& v_output = tensor_return_value.v;

    auto q_base_addr = q_output.buffer()->address();
    auto k_base_addr = k_output.buffer()->address();
    auto v_base_addr = v_output.buffer()->address();
    auto batch_base_addr = batch_tensor.buffer()->address();

    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(mesh_coord_range);

        // Update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);
        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[1] = operation_attributes.semaphore.address();
        }

        auto& reduction_reader_args_by_core = GetRuntimeArgs(program, shared_vars.reduction_reader_kernel_id);
        auto& reduction_writer_args_by_core = GetRuntimeArgs(program, shared_vars.reduction_writer_kernel_id);

        for (const auto& core : shared_vars.output_cores_vec) {
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
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_out, *output.buffer());
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_reduction, *buffer_tensor.buffer());
    }
}

}  // namespace ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads::program
