// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor_impl.hpp"

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
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

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::fused::normalization {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
inline bool is_dram(const Tensor& input_tensor) {
    return input_tensor.memory_config().buffer_type() == BufferType::DRAM;
}
inline bool is_dram(const std::optional<const Tensor>& input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// computes layernorm(a+*b)*gamma
// if b is nullptr it's treated as zero (no addition)

operation::ProgramWithCallbacks frmsnorm_pre_multi_core_sharded(
    const Tensor& a,                       // input
    const std::optional<const Tensor>& b,  // residual
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    // New Parameters
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const uint32_t dim = 3;
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / TILE_WIDTH;

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::MeshDevice* mesh_device = a.mesh_device();
    tt::tt_metal::Program program{};
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    uint32_t output_page_size = 0;
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    if (output.get_layout() == Layout::TILE) {
        output_page_size = output.tensor_spec().tile().get_tile_size(out_data_format);
    } else {
        output_page_size = output.buffer()->page_size();
    }

    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == ccl::Topology::Linear) {
        ccl::LineTopology line_topology(ring_size, ring_index);
        num_targets_forward =
            line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
        num_targets_backward =
            line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        if (ring_index % 2 == 0) {
            std::swap(num_targets_forward, num_targets_backward);
        }
    }

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    ShardSpec shard_spec = a.shard_spec().value();
    CoreRangeSet all_cores = shard_spec.grid;
    const auto& cores = corerange_to_cores(all_cores, all_cores.num_cores(), true);

    // Tensor Info
    const auto input_tensor_cores = a.memory_config().shard_spec()->grid;
    const auto output_tensor_cores = output.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_config().channel_buffer_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = output_page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        (num_links == 1) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), compute_kernel_config);

    if (!dst_full_sync_en) {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 4,
                "subblock_wt={}, but subblock width must less than 4 tiles in fp32 mode when dst_full_sync_en is false",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles when dst_full_sync_en is false",
                subblock_wt);
        }
    } else {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles in fp32 mode when dst_full_sync_en is true",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 16,
                "subblock_wt={}, but subblock width must less than 16 tiles when dst_full_sync_en is true",
                subblock_wt);
        }
    }
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    // tile sizes
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::log_debug("in_data_format: {}", in_data_format);
    tt::log_debug("out_data_format: {}", out_data_format);
    tt::log_debug("cb_data_format: {}", cb_data_format);
    tt::log_debug("math_fidelity: {}", math_fidelity);
    tt::log_debug("math_approx_mode: {}", math_approx_mode);
    tt::log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shape = a.get_padded_shape();
    uint32_t M = a.volume() / shape[-1];
    uint32_t K = shape[-1];
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = TILE_HEIGHT;
    uint32_t num_blocks = 0;

    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};
    std::optional<CoreCoord> grid_offset = std::nullopt;
    if (bbox.start_coord.x != 0 || bbox.start_coord.y != 0) {
        grid_offset = bbox.start_coord;
    }
    num_blocks = shard_spec.num_cores();

    // two-stage reduce
    bool use_two_stage_reduce = false;
    // only do this for row/col dim are full length
    if (grid_size.x > 1 && grid_size.x <= mesh_device->compute_with_storage_grid_size().x &&
        grid_size.y > 1) {  // row major and multiple rows
        use_two_stage_reduce = true;
    }
    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();

    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    CoreRangeSet all_worker_and_storage_cores = all_storage_cores.merge(a.shard_spec().value().grid);
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), true);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)mesh_device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)mesh_device->worker_core_from_logical_core(core).y);

        tt::log_debug(
            "Storage core: ({}, {}), physical coords: ({}, {})",
            core.x,
            core.y,
            storage_core_noc_x.back(),
            storage_core_noc_y.back());
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = block_wt;
    // post_all_gather_stats_block_tiles
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;

    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex_external_CB_size = tt::div_up(Kt, block_wt) * single_tile_size;
    uint32_t xmm2_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex2pe_CB_size = single_tile_size;
    uint32_t stats_cb_size = 0;
    uint32_t stats_reduced_cb_size = 0;
    // output buffer size
    uint32_t out_CB_size;
    out_CB_size = out_single_tile_size;
    uint32_t out_reshard_CB_size = out_CB_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    // define core ranges
    bool use_mcast = num_blocks > 1;

    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t num_cores_all_to_all = 1;
    uint32_t num_cores_all_to_all_second_stage = 0;
    uint32_t num_blocks_first_stage = num_blocks;
    uint32_t num_blocks_second_stage = 0;
    if (use_two_stage_reduce) {
        num_blocks_first_stage = num_cores_x;
        num_cores_all_to_all = num_cores_y;
        num_blocks_second_stage = num_cores_y;
    }
    // change tt::CBIndex external size
    if (use_two_stage_reduce) {
        ex_external_CB_size = (num_blocks_first_stage + num_blocks_second_stage - 1) * single_tile_size;
    }
    uint32_t num_none_all_to_all_workers = num_blocks - num_cores_all_to_all;

    CoreCoord start_core = {0, 0};
    CoreRange sender_cores(start_core, start_core);
    CoreRangeSet all_to_all_cores;
    CoreRangeSet all_to_all_workers_except_sender;
    CoreRangeSet not_all_to_all_workers;
    uint32_t num_cores_x_mcast, num_cores_y_mcast;
    sender_cores = {start_core, start_core};
    CoreCoord all_core_grid_size;
    CoreCoord none_core_grid_size;
    if (use_two_stage_reduce) {
        all_core_grid_size = {1, num_cores_y};
        none_core_grid_size = {num_cores_x - 1, num_cores_y};
    } else {
        all_core_grid_size = grid_size;
        none_core_grid_size = grid_size;
    }
    all_to_all_cores = num_cores_to_corerangeset(start_core, num_cores_all_to_all, all_core_grid_size, true);
    if (use_mcast) {
        CoreCoord all_start_core;
        CoreCoord end_core = sender_cores.end_coord;
        if (use_two_stage_reduce) {
            if (end_core.x == all_core_grid_size.x - 1) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        } else {
            if (end_core.x == bbox.end_coord.x) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        }
        all_to_all_workers_except_sender =
            num_cores_to_corerangeset(all_start_core, num_cores_all_to_all - 1, all_core_grid_size, true);
    }
    if (num_none_all_to_all_workers > 0) {
        if (use_two_stage_reduce) {
            CoreCoord none_start_core = {all_core_grid_size.x, sender_cores.end_coord.y};
            CoreCoord none_end_core = {num_cores_x - 1, num_cores_y - 1};
            CoreRange none_core_range = CoreRange(none_start_core, none_end_core);
            not_all_to_all_workers = CoreRangeSet(none_core_range);
        } else {
            CoreCoord none_start_core;
            CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end_coord;
            if (end_core.x == bbox.end_coord.x) {
                none_start_core = {0, end_core.y + 1};
            } else {
                none_start_core = {end_core.x + 1, end_core.y};
            }
            not_all_to_all_workers =
                num_cores_to_corerangeset(none_start_core, num_none_all_to_all_workers, none_core_grid_size, true);
        }
    }
    num_cores_x_mcast = num_cores_x;
    num_cores_y_mcast = num_cores_y;
    auto applyStartOffset = [](const CoreRangeSet& input_set, const CoreCoord& grid_offset) -> CoreRangeSet {
        if (input_set.empty()) {
            return input_set;
        }

        std::vector<CoreRange> new_ranges;
        new_ranges.reserve(input_set.size());

        for (const CoreRange& range : input_set.ranges()) {
            CoreCoord new_start = {range.start_coord.x + grid_offset.x, range.start_coord.y + grid_offset.y};
            CoreCoord new_end = {range.end_coord.x + grid_offset.x, range.end_coord.y + grid_offset.y};
            new_ranges.emplace_back(new_start, new_end);
        }

        return CoreRangeSet(std::move(new_ranges));
    };
    if (grid_offset.has_value()) {
        start_core = {start_core.x + grid_offset.value().x, start_core.y + grid_offset.value().y};
        sender_cores = {
            {sender_cores.start_coord.x + start_core.x, sender_cores.start_coord.y + start_core.y},
            {sender_cores.end_coord.x + start_core.x, sender_cores.end_coord.y + start_core.y}};
        all_to_all_cores = applyStartOffset(all_to_all_cores, grid_offset.value());
        all_to_all_workers_except_sender = applyStartOffset(all_to_all_workers_except_sender, grid_offset.value());
        not_all_to_all_workers = applyStartOffset(not_all_to_all_workers, grid_offset.value());
    }
    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_second_stage_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // Create circular buffers

    // in1 sharded

    CBHandle cb_in1 = 0;
    CBHandle cb_add_out = 0;

    // in2 scaler
    uint32_t in2_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig in2_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in2_cb_index, bfloat16_tile_size);
    auto cb_in2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig in4_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in4_cb_index, bfloat16_tile_size);
    auto cb_in4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // ex_partial2
    uint32_t ex_cb_partial2_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig ex_cb_partial2_config =
        tt::tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial2_index, cb_data_format}})
            .set_page_size(ex_cb_partial2_index, single_tile_size);
    auto cb_ex_partial2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial2_config);
    // ex2
    uint32_t ex2_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig ex2_cb_config =
        tt::tt_metal::CircularBufferConfig(ex_CB_size, {{ex2_cb_index, cb_data_format}})
            .set_page_size(ex2_cb_index, single_tile_size);
    auto cb_ex2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_config);

    // ex_external2
    uint32_t ex_cb_external2_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig ex_cb_external2_config =
        tt::tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external2_index, cb_data_format}})
            .set_page_size(ex_cb_external2_index, single_tile_size);
    auto cb_ex_external2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external2_config);

    CBHandle cb_stats = 0;

    // in0 sharded
    uint32_t in0_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig in0_cb_config =
        tt::tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size)
            .set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);

    // x
    uint32_t x_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig x_cb_config =
        tt::tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt::tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);

    // out
    uint32_t cb_to_allgather_writer = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{cb_to_allgather_writer, out_data_format}})
            .set_page_size(cb_to_allgather_writer, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_8;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle = CreateCircularBuffer(program, all_cores, cb_reserved_packet_header_config);

    uint32_t add_out_cb_index = tt::CBIndex::c_9;
    uint32_t in1_cb_index = tt::CBIndex::c_10;

    if (b) {
        tt::tt_metal::CircularBufferConfig add_out_cb_config =
            tt::tt_metal::CircularBufferConfig(in1_CB_size, {{add_out_cb_index, in_data_format}})
                .set_page_size(add_out_cb_index, in_single_tile_size)
                .set_globally_allocated_address(*a.buffer());
        cb_add_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, add_out_cb_config);

        tt::tt_metal::CircularBufferConfig in1_cb_config =
            tt::tt_metal::CircularBufferConfig(
                in1_CB_size, {{in1_cb_index, tt::tt_metal::datatype_to_dataformat_converter(b.value().get_dtype())}})
                .set_page_size(in1_cb_index, in_single_tile_size)
                .set_globally_allocated_address(*b.value().buffer());
        cb_in1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in1_cb_config);
    }

    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)add_out_cb_index,
        (std::uint32_t)ex_cb_external2_index};
    std::vector<uint32_t> reader_mcast_receiver_all_to_all_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)1,
        (std::uint32_t)num_cores_x_mcast,
        (std::uint32_t)num_cores_y_mcast,
        (std::uint32_t)use_two_stage_reduce,
        (std::uint32_t)num_blocks_first_stage,
        (std::uint32_t)num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)ex_cb_external2_index};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)0,
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)ex_cb_partial2_index,
        (std::uint32_t)ex2_cb_index,
        (std::uint32_t)ex_cb_external2_index};

    std::vector<uint32_t> writer_compile_time_args = {
        1,  // Gets overwritten in not all to all workers
        in2_cb_index,
        in4_cb_index,
        cb_to_allgather_writer,
        // all gather parameters
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_pages_per_packet,             // packet_size_in_pages
        output_page_size,                 // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        num_links};
    // compute kernel compile time args
    std::vector<uint32_t> all_to_all_except_top_compute_compile_time_args = {
        num_blocks_first_stage,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_wt,
        fp32_dest_acc_en,
        num_blocks_second_stage,
        in2_cb_index,
        in4_cb_index,
        ex_cb_partial2_index,
        ex2_cb_index,
        add_out_cb_index,
        ex_cb_external2_index,
        cb_to_allgather_writer,
        x_cb_index,
        in1_cb_index,
        in0_cb_index};
    std::vector<uint32_t> not_all_to_all_compute_compile_time_args = {
        num_blocks_first_stage,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        0,
        block_wt,
        fp32_dest_acc_en,
        num_blocks_second_stage,
        in2_cb_index,
        in4_cb_index,
        ex_cb_partial2_index,
        ex2_cb_index,
        add_out_cb_index,
        ex_cb_external2_index,
        cb_to_allgather_writer,
        x_cb_index,
        in1_cb_index,
        in0_cb_index};

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(mesh_device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(mesh_device->arch());

    // reader kernel

    std::string sender_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "reader_mcast_sender_unary_sharded_rms.cpp";
    std::string reciever_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "reader_mcast_receiver_unary_sharded_rms.cpp";

    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        sender_reader_kernel_file,
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args});
    KernelHandle reader_mcast_receiver_kernels_id_all_to_all = -1;
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id_all_to_all = CreateKernel(
            program,
            reciever_reader_kernel_file,
            all_to_all_workers_except_sender,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_all_to_all_compile_time_args});
    }
    if (num_none_all_to_all_workers > 0) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            reciever_reader_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args});
    }

    // writer kernel + all gather kernel
    std::string writer_kernel =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "writer_unary_sharded_rms.cpp";
    writer_compile_time_args.at(0) = 1;
    auto writer_mcast_sender_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_to_all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_compile_time_args});
    KernelHandle writer_mcast_receiver_kernels_id = -1;
    if (num_none_all_to_all_workers > 0) {
        writer_compile_time_args.at(0) = 0;
        writer_mcast_receiver_kernels_id = CreateKernel(
            program,
            writer_kernel,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = writer_noc,
                .compile_args = writer_compile_time_args});
    }

    // compute kernel
    std::string compute_kernel_file;
    compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/"
        "rms_sharded_pre.cpp";
    KernelHandle compute_kernels_id = -1;
    auto compute_kernels_id_all_to_all = CreateKernel(
        program,
        compute_kernel_file,
        all_to_all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = all_to_all_except_top_compute_compile_time_args});
    if (num_none_all_to_all_workers > 0) {
        compute_kernels_id = CreateKernel(
            program,
            compute_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = not_all_to_all_compute_compile_time_args});
    }

    // Get AG worker cores

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    writer_kernel_ids.reserve(cores.size());
    float winv = 1.0f / block_w;                                                               // bcast-w scaler
    float cinv = (1.0f / num_blocks);                                                          // bcast-cores scaler
    float cinv_one = 1.0f;  // bcast-cores scaler for all-to-all cores not on first row/col
    auto bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    auto bfloat_cinv_value_one = bfloat16(cinv_one);
    uint32_t packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_value_one, bfloat_cinv_value_one});
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;

    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    CoreCoord core_start_offset = grid_offset.value_or(CoreCoord{0, 0});
    for (uint32_t core_idx_x = core_start_offset.x; core_idx_x < num_cores_x + core_start_offset.x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(mesh_device->worker_core_from_logical_core({core_idx_x, core_start_offset.y}).x);
    }
    for (uint32_t core_idx_y = core_start_offset.y; core_idx_y < num_cores_y + core_start_offset.y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(mesh_device->worker_core_from_logical_core({core_start_offset.x, core_idx_y}).y);
    }

    uint32_t last_core_width_index = 0;
    last_core_width_index = cores.size() - 1;

    // For write back calculation
    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    // All gather RT prep

    CoreCoord drain_sync_core;
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - 1 / ring_size;
    uint32_t start_core_index_for_device = output_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;
    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        tt::log_debug("core: {}, {}", core.x, core.y);

        uint32_t width_index = 0;
        width_index = i;

        uint32_t width_index_two_stage = width_index % num_blocks_first_stage;

        uint32_t all_to_all_worker_tile_offset_size_bytes;
        if (use_two_stage_reduce) {
            all_to_all_worker_tile_offset_size_bytes = (width_index_two_stage)*single_tile_size;
        } else {
            all_to_all_worker_tile_offset_size_bytes = (width_index)*single_tile_size;
        }

        uint32_t num_reduce_tiles_per_block_h = block_wt;
        // account for padding
        if (width_index == last_core_width_index) {
            num_reduce_tiles_per_block_h = Kt - last_core_width_index * block_wt;
        }

        std::vector<uint32_t> compute_args{num_reduce_tiles_per_block_h};
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            compute_args.push_back(1);
            compute_args.push_back((uint32_t)use_two_stage_reduce);
            bool is_second_stage_reader;
            if (use_two_stage_reduce) {
                is_second_stage_reader = width_index < 1;
            } else {
                is_second_stage_reader = false;
            }
            compute_args.push_back((uint32_t)is_second_stage_reader);
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id_all_to_all, core, compute_args);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
        }

        if (width_index == 0) {
            CoreCoord mcast_start, mcast_end;
            CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
            CoreCoord bottom_right_core = {
                (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
            auto top_left_core_physical = mesh_device->worker_core_from_logical_core(top_left_core);
            auto bottom_right_core_physical = mesh_device->worker_core_from_logical_core(bottom_right_core);
            mcast_start = top_left_core_physical;
            mcast_end = bottom_right_core_physical;
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            std::vector<uint32_t> mcast_sender_args;
            mcast_sender_args.push_back(mcast_start.x);
            mcast_sender_args.push_back(mcast_start.y);
            mcast_sender_args.push_back(mcast_end.x);
            mcast_sender_args.push_back(mcast_end.y);
            mcast_sender_args.push_back(core.x - start_core.x);
            mcast_sender_args.push_back(core.y - start_core.y);
            mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);
        } else if (
            (not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> mcast_receiver_args;
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            bool is_second_stage_reader;
            if (use_two_stage_reduce and width_index < 1) {
                is_second_stage_reader = true;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            } else {
                is_second_stage_reader = false;
                mcast_receiver_args.push_back((uint32_t)is_second_stage_reader);
            }
            mcast_receiver_args.push_back(core.x - start_core.x);
            mcast_receiver_args.push_back(core.y - start_core.y);
            mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            tt::tt_metal::SetRuntimeArgs(
                program, reader_mcast_receiver_kernels_id_all_to_all, core, mcast_receiver_args);
        } else {
            std::vector<uint32_t> mcast_receiver_args;
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
            mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
        }
        // Set all gather runtime args

        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        // all_gather_rts Start at RT index 3 of writer
        std::vector<uint32_t> all_gather_rts = {
            semaphore.address(),        // out_ready_sem_bank_addr (absolute address)
            out_ready_sem_wait_value,   // out_ready_sem_wait_value
            output.buffer()->address()  // tensor_address0
        };

        if (i < num_links) {
            // Add RT values for the all gather core to all_gather_rts
            // Will be appended to the end of writer rt args
            uint32_t base_pages_per_worker = 1 / num_links;
            uint32_t remainder = 1 % num_links;
            uint32_t input_tile_id_start = i * base_pages_per_worker + std::min(i, remainder);
            uint32_t input_tile_id_end = (i + 1) * base_pages_per_worker + std::min(i + 1, remainder);
            uint32_t output_first_core_tile_start_offset =
                (ring_index + input_tile_id_start) % output_tensor_shard_num_pages;
            std::vector<uint32_t> output_tensor_cores_x;
            std::vector<uint32_t> output_tensor_cores_y;
            for (uint32_t i = input_tile_id_start / output_tensor_shard_num_pages;
                 i < (input_tile_id_end + output_tensor_shard_num_pages - 1) / output_tensor_shard_num_pages;
                 i++) {
                auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
                output_tensor_cores_x.push_back(this_core.x);
                output_tensor_cores_y.push_back(this_core.y);
            }
            if (i == 0) {
                // drain sync core is the first worker core
                drain_sync_core = mesh_device->worker_core_from_logical_core(core);
            }

            std::vector<uint32_t> base_rt_args = {
                output_first_core_tile_start_offset,  // first_core_tile_start_offset
                output_tensor_cores_x.size(),         // num_cores
                drain_sync_core.x,                    // out_ready_sem_noc0_x
                drain_sync_core.y                     // out_ready_sem_noc0_y
            };
            all_gather_rts.insert(all_gather_rts.end(), base_rt_args.begin(), base_rt_args.end());
            all_gather_rts.insert(all_gather_rts.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
            all_gather_rts.insert(all_gather_rts.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());

            all_gather_rts.push_back(forward_device.has_value());
            if (forward_device.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device->id(), forward_device.value()->id(), i, program, {core}, all_gather_rts);
            }

            all_gather_rts.push_back(backward_device.has_value());
            if (backward_device.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device->id(), backward_device.value()->id(), i, program, {core}, all_gather_rts);
            }
        }
        // Set writer runtime args

        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> writer_mcast_sender_args;
            if (use_two_stage_reduce && (!(width_index < 1))) {
                writer_mcast_sender_args.push_back(packed_winv_value);
                writer_mcast_sender_args.push_back(packed_cinv_value_one);
            } else {
                writer_mcast_sender_args.push_back(packed_winv_value);
                writer_mcast_sender_args.push_back(packed_cinv_value);
            }
            writer_mcast_sender_args.push_back(i);  // Core ID to limit number of cores to do all gather on
            writer_mcast_sender_args.insert(
                writer_mcast_sender_args.end(), all_gather_rts.begin(), all_gather_rts.end());
            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_sender_kernels_id, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_mcast_sender_kernels_id);
        } else {
            std::vector<uint32_t> writer_mcast_receiver_args;
            writer_mcast_receiver_args.push_back(packed_winv_value);
            writer_mcast_receiver_args.push_back(packed_cinv_value);
            writer_mcast_receiver_args.push_back(i);  // Core ID to limit number of cores to do all gather on
            writer_mcast_receiver_args.insert(
                writer_mcast_receiver_args.end(), all_gather_rts.begin(), all_gather_rts.end());
            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_receiver_kernels_id, core, writer_mcast_receiver_args);
            writer_kernel_ids.push_back(writer_mcast_receiver_kernels_id);
        }
    }

    auto override_runtime_arguments_callback =
        [writer_kernel_ids,
         writer_mcast_sender_kernels_id,
         writer_mcast_receiver_kernels_id,
         num_none_all_to_all_workers,
         cb_in0,
         cb_in1,
         cb_stats,
         cb_add_out,
         cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto src_buffer_a = input_tensors.at(0).buffer();
            const auto b_tensor = optional_input_tensors.at(0);
            const auto dst_buffer = output_tensors.at(0).buffer();
            auto& writer_sender_args_by_core = GetRuntimeArgs(program, writer_mcast_sender_kernels_id);
            auto& writer_receiver_args_by_core = num_none_all_to_all_workers > 0
                                                     ? GetRuntimeArgs(program, writer_mcast_receiver_kernels_id)
                                                     : writer_sender_args_by_core;
            auto semaphore = static_cast<const RMSAllGather*>(operation)->semaphore;

            for (uint32_t i = 0; i < cores.size(); ++i) {
                const CoreCoord& core = cores[i];

                const auto writer_kernel_id = writer_kernel_ids.at(i);

                if (writer_kernel_id == writer_mcast_sender_kernels_id) {
                    auto& runtime_args = writer_sender_args_by_core[core.x][core.y];
                    runtime_args[3] = semaphore.address();
                    runtime_args[5] = dst_buffer->address();
                } else if (writer_kernel_id == writer_mcast_receiver_kernels_id) {
                    auto& runtime_args = writer_receiver_args_by_core[core.x][core.y];
                    runtime_args[3] = semaphore.address();
                    runtime_args[5] = dst_buffer->address();
                }
            }
            UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);

            if (b_tensor.has_value()) {
                UpdateDynamicCircularBufferAddress(program, cb_in1, *b_tensor.value().buffer());
                UpdateDynamicCircularBufferAddress(program, cb_add_out, *src_buffer_a);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks frmsnorm_post_multi_core_sharded(
    const Tensor& a,                           // input
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& stats,  // stats
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    const GlobalSemaphore& semaphore,
    const uint32_t ring_size,
    const uint32_t num_links) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / TILE_WIDTH;
    bool skip_write_back = output.shard_spec().value() == a.shard_spec().value();

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    // convert data format
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    if (!dst_full_sync_en) {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 4,
                "subblock_wt={}, but subblock width must less than 4 tiles in fp32 mode when dst_full_sync_en is false",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles when dst_full_sync_en is false",
                subblock_wt);
        }
    } else {
        if (fp32_dest_acc_en) {
            TT_FATAL(
                subblock_wt <= 8,
                "subblock_wt={}, but subblock width must less than 8 tiles in fp32 mode when dst_full_sync_en is true",
                subblock_wt);
        } else {
            TT_FATAL(
                subblock_wt <= 16,
                "subblock_wt={}, but subblock width must less than 16 tiles when dst_full_sync_en is true",
                subblock_wt);
        }
    }

    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format =
        gamma.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().get_dtype())
                          : tt::DataFormat::Float16_b;
    // tile sizes
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_single_tile_size = tt::tt_metal::detail::TileSize(gamma_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::log_debug("in_data_format: {}", in_data_format);
    tt::log_debug("out_data_format: {}", out_data_format);
    tt::log_debug("cb_data_format: {}", cb_data_format);
    tt::log_debug("gamma_cb_data_format: {}", gamma_cb_data_format);
    tt::log_debug("math_fidelity: {}", math_fidelity);
    tt::log_debug("math_approx_mode: {}", math_approx_mode);
    tt::log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shape = a.get_padded_shape();
    uint32_t M = a.volume() / shape[-1];
    uint32_t K = shape[-1];
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = 1 * TILE_HEIGHT;
    uint32_t num_blocks = 0;
    ShardSpec shard_spec = a.shard_spec().value();

    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1};
    std::optional<CoreCoord> grid_offset = std::nullopt;
    if (bbox.start_coord.x != 0 || bbox.start_coord.y != 0) {
        grid_offset = bbox.start_coord;
    }
    num_blocks = shard_spec.num_cores();

    // two-stage reduce
    bool use_two_stage_reduce = false;
    if (grid_size.x > 1 && grid_size.x <= device->compute_with_storage_grid_size().x &&
        grid_size.y > 1) {  // row major and multiple rows
        use_two_stage_reduce = true;
    }
    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();
    bool output_row_wise = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    CoreRangeSet all_worker_and_storage_cores = all_storage_cores.merge(a.shard_spec().value().grid);
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), true);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);

        tt::log_debug(
            "Storage core: ({}, {}), physical coords: ({}, {})",
            core.x,
            core.y,
            storage_core_noc_x.back(),
            storage_core_noc_y.back());
    }

    // b, gamma addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t num_rows_per_all_to_all_worker = 1;
    uint32_t in0_block_tiles = block_wt;
    // post_all_gather_stats_block_tiles
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().get_padded_shape()[-1] / TILE_WIDTH;
        num_distributed_devices = post_all_gather_stats_block_tiles;
    }

    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_block_tiles * gamma_single_tile_size / 1;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t xmm2_CB_size = in0_block_tiles * single_tile_size / 1;
    uint32_t ex2pe_CB_size = single_tile_size;
    uint32_t stats_cb_size = 0;
    uint32_t stats_reduced_cb_size = 0;
    stats_cb_size = post_all_gather_stats_block_tiles * single_tile_size;
    stats_reduced_cb_size = single_tile_size;
    // output buffer size
    uint32_t out_CB_size;
    out_CB_size = in0_block_tiles * out_single_tile_size;
    uint32_t out_reshard_CB_size = out_CB_size;
    if (!skip_write_back) {
        out_reshard_CB_size = block_wt_resharded * 1 * out_single_tile_size;
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_blocks > 1;

    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t num_cores_all_to_all = 1;
    uint32_t num_cores_all_to_all_second_stage = 0;
    uint32_t num_blocks_first_stage = num_blocks;
    uint32_t num_blocks_second_stage = 0;
    if (use_two_stage_reduce) {
        num_blocks_first_stage = num_cores_x;
        num_cores_all_to_all_second_stage = num_cores_y;
        num_cores_all_to_all = num_cores_y;
        num_blocks_second_stage = num_cores_all_to_all_second_stage;
    }
    uint32_t num_none_all_to_all_workers = num_blocks - num_cores_all_to_all;

    CoreCoord start_core = {0, 0};
    CoreRangeSet all_cores = shard_spec.grid;
    CoreRange sender_cores(start_core, start_core);
    CoreRangeSet all_to_all_cores;
    CoreRangeSet all_to_all_workers_except_sender;
    CoreRangeSet not_all_to_all_workers;
    sender_cores = {start_core, start_core};
    CoreCoord all_core_grid_size;
    CoreCoord none_core_grid_size;
    if (use_two_stage_reduce) {
        all_core_grid_size = {1, num_cores_y};
        none_core_grid_size = {num_cores_x - 1, num_cores_y};
    } else {
        all_core_grid_size = grid_size;
        none_core_grid_size = grid_size;
    }
    all_to_all_cores = num_cores_to_corerangeset(start_core, num_cores_all_to_all, all_core_grid_size, true);
    if (use_mcast) {
        CoreCoord all_start_core;
        CoreCoord end_core = sender_cores.end_coord;
        if (use_two_stage_reduce) {
            if (end_core.x == all_core_grid_size.x - 1) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        } else {
            if (end_core.x == bbox.end_coord.x) {
                all_start_core = {0, end_core.y + 1};
            } else {
                all_start_core = {end_core.x + 1, end_core.y};
            }
        }
        all_to_all_workers_except_sender =
            num_cores_to_corerangeset(all_start_core, num_cores_all_to_all - 1, all_core_grid_size, true);
    }
    if (num_none_all_to_all_workers > 0) {
        if (use_two_stage_reduce) {
            CoreCoord none_start_core = {all_core_grid_size.x, sender_cores.end_coord.y};
            CoreCoord none_end_core = {num_cores_x - 1, num_cores_y - 1};
            CoreRange none_core_range = CoreRange(none_start_core, none_end_core);
            not_all_to_all_workers = CoreRangeSet(none_core_range);
        } else {
            CoreCoord none_start_core;
            CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end_coord;
            if (end_core.x == bbox.end_coord.x) {
                none_start_core = {0, end_core.y + 1};
            } else {
                none_start_core = {end_core.x + 1, end_core.y};
            }
            not_all_to_all_workers =
                num_cores_to_corerangeset(none_start_core, num_none_all_to_all_workers, none_core_grid_size, true);
        }
    }
    auto applyStartOffset = [](const CoreRangeSet& input_set, const CoreCoord& grid_offset) -> CoreRangeSet {
        if (input_set.empty()) {
            return input_set;
        }

        std::vector<CoreRange> new_ranges;
        new_ranges.reserve(input_set.size());

        for (const CoreRange& range : input_set.ranges()) {
            CoreCoord new_start = {range.start_coord.x + grid_offset.x, range.start_coord.y + grid_offset.y};
            CoreCoord new_end = {range.end_coord.x + grid_offset.x, range.end_coord.y + grid_offset.y};
            new_ranges.emplace_back(new_start, new_end);
        }

        return CoreRangeSet(std::move(new_ranges));
    };
    if (grid_offset.has_value()) {
        start_core = {start_core.x + grid_offset.value().x, start_core.y + grid_offset.value().y};
        sender_cores = {
            {sender_cores.start_coord.x + start_core.x, sender_cores.start_coord.y + start_core.y},
            {sender_cores.end_coord.x + start_core.x, sender_cores.end_coord.y + start_core.y}};
        all_to_all_cores = applyStartOffset(all_to_all_cores, grid_offset.value());
        all_to_all_workers_except_sender = applyStartOffset(all_to_all_workers_except_sender, grid_offset.value());
        not_all_to_all_workers = applyStartOffset(not_all_to_all_workers, grid_offset.value());
    }
    // Mcast args
    auto reduce_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }

    // Create circular buffers

    // ex_global
    uint32_t ex_global_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig ex_global_cb_config =
        tt::tt_metal::CircularBufferConfig(ex_global_CB_size, {{ex_global_cb_index, cb_data_format}})
            .set_page_size(ex_global_cb_index, single_tile_size);
    auto cb_ex_global = tt::tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);

    // out
    uint32_t output_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, out_data_format}})
            .set_page_size(output_cb_index, out_single_tile_size);
    if (skip_write_back) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*output.buffer());
    }
    CBHandle cb_output = 0;
    cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // gamma
    uint32_t in5_cb_index = tt::CBIndex::c_2;
    if (gamma.has_value()) {
        tt::tt_metal::CircularBufferConfig in5_cb_config =
            tt::tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_cb_data_format}})
                .set_page_size(in5_cb_index, gamma_single_tile_size);
        auto cb_in5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }

    // in0 sharded
    uint32_t in0_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig in0_cb_config =
        tt::tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
            .set_page_size(in0_cb_index, in_single_tile_size)
            .set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);

    // in3 eps
    uint32_t in3_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig in3_cb_config =
        tt::tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in3_cb_index, bfloat16_tile_size);
    auto cb_in3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);

    // in4 scaler-c
    uint32_t in4_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig in4_cb_config =
        tt::tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in4_cb_index, bfloat16_tile_size);
    auto cb_in4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);

    // x
    uint32_t x_cb_index;
    x_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig x_cb_config =
        tt::tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
            .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt::tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);

    uint32_t output_reshard_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig output_reshard_cb_config =
        tt::tt_metal::CircularBufferConfig(out_reshard_CB_size, {{output_reshard_cb_index, out_data_format}})
            .set_page_size(output_reshard_cb_index, out_single_tile_size);
    CBHandle cb_output_reshard = 0;
    if (!skip_write_back) {
        output_reshard_cb_config = output_reshard_cb_config.set_globally_allocated_address(*output.buffer());
        cb_output_reshard = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_reshard_cb_config);
    }

    // cb_var
    uint32_t cb_var_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_var_config =
        tt::tt_metal::CircularBufferConfig(ex_global_CB_size, {{cb_var_index, cb_data_format}})
            .set_page_size(cb_var_index, single_tile_size);
    auto cb_var_global = tt::tt_metal::CreateCircularBuffer(program, sender_cores, cb_var_config);

    // cb_stats_reduced
    uint32_t cb_stats_reduced_index;
    cb_stats_reduced_index = tt::CBIndex::c_9;
    tt::tt_metal::CircularBufferConfig stats_reduced_cb_config =
        tt::tt_metal::CircularBufferConfig(stats_reduced_cb_size, {{cb_stats_reduced_index, cb_data_format}})
            .set_page_size(cb_stats_reduced_index, single_tile_size);
    auto cb_stats_reduced = tt::tt_metal::CreateCircularBuffer(program, sender_cores, stats_reduced_cb_config);

    // cb_stats
    uint32_t cb_stats_index;
    cb_stats_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig stats_cb_config =
        tt::tt_metal::CircularBufferConfig(stats_cb_size, {{cb_stats_index, cb_data_format}})
            .set_page_size(cb_stats_index, single_tile_size)
            .set_globally_allocated_address(*stats.value().buffer());
    auto cb_stats = tt::tt_metal::CreateCircularBuffer(program, sender_cores, stats_cb_config);

    const auto& cores = corerange_to_cores(all_cores, all_cores.num_cores(), true);

    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)num_blocks,
        (std::uint32_t)single_tile_size,
        (std::uint32_t)cb_stats_reduced_index,
        (std::uint32_t)ex_global_cb_index};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_sender_semaphore_id, ex_global_cb_index};

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());

    if (!skip_write_back) {
        reader_noc = NOC::NOC_0;
        writer_noc = NOC::NOC_1;
    }

    // reader kernel
    std::string sender_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "reader_mcast_sender_unary_sharded_rms_post.cpp";
    std::string reciever_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "reader_mcast_receiver_unary_sharded_rms_post.cpp";

    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        sender_reader_kernel_file,
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
            .compile_args = reader_mcast_sender_compile_time_args,
            .defines = reader_mcast_sender_defines});
    KernelHandle reader_mcast_receiver_kernels_id_all_to_all = -1;
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id_all_to_all = CreateKernel(
            program,
            reciever_reader_kernel_file,
            all_to_all_workers_except_sender,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }
    if (num_none_all_to_all_workers > 0) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            reciever_reader_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .compile_args = reader_mcast_receiver_compile_time_args,
                .defines = reader_mcast_receiver_defines});
    }

    // writer defines
    std::map<string, string> writer_defines;
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    // writer compile time args
    std::vector<uint32_t> writer_compile_time_args = {
        1,  // is_all_to_all_worker
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)is_dram(gamma),
        (std::uint32_t)block_wt,
        output_reshard_cb_index,
        output_cb_index,
        in3_cb_index,
        in4_cb_index,
        in5_cb_index};

    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().get_padded_shape()[-1] * gamma.value().element_size();
        writer_compile_time_args.push_back(gamma_stick_size);
    } else {
        writer_compile_time_args.push_back(0);
    }

    writer_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);

    // write back compile time args
    writer_compile_time_args.push_back(block_wt * out_single_tile_size);  // out_tensor_stride_w_bytes
    writer_compile_time_args.push_back(
        block_wt_resharded * out_single_tile_size);  // out_reshard_tensor_stride_w_bytes: how many bytes to skip to get
                                                     // to the next data chunk
    // writer kernel
    std::string writer_kernel;
    writer_kernel =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/"
        "writer_unary_sharded_rms_post.cpp";
    auto writer_mcast_sender_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_to_all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});
    KernelHandle writer_mcast_receiver_kernels_id = -1;
    if (num_none_all_to_all_workers > 0) {
        writer_compile_time_args.at(0) = 0;
        writer_mcast_receiver_kernels_id = CreateKernel(
            program,
            writer_kernel,
            not_all_to_all_workers,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = writer_noc,
                .compile_args = writer_compile_time_args,
                .defines = writer_defines});
    }
    // compute kernel compile time args
    std::vector<uint32_t> all_to_all_except_top_compute_compile_time_args = {
        num_blocks_first_stage,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_wt,
        fp32_dest_acc_en,
        num_blocks_second_stage,
        output_cb_index,
        cb_stats_index,
        in0_cb_index,
        in3_cb_index,
        in4_cb_index,
        cb_var_index,
        x_cb_index,
        in5_cb_index,
        cb_stats_reduced_index,
        ex_global_cb_index};
    std::vector<uint32_t> not_all_to_all_compute_compile_time_args = {
        num_blocks_first_stage,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        0,
        block_wt,
        fp32_dest_acc_en,
        num_blocks_second_stage,
        output_cb_index,
        cb_stats_index,
        in0_cb_index,
        in3_cb_index,
        in4_cb_index,
        cb_var_index,
        x_cb_index,
        in5_cb_index,
        cb_stats_reduced_index,
        ex_global_cb_index};
    // compute kernel
    std::string compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/"
        "rms_sharded_post.cpp";
    KernelHandle compute_kernels_id = -1;
    auto compute_kernels_id_all_to_all = CreateKernel(
        program,
        compute_kernel_file,
        all_to_all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = all_to_all_except_top_compute_compile_time_args});
    if (num_none_all_to_all_workers > 0) {
        compute_kernels_id = CreateKernel(
            program,
            compute_kernel_file,
            not_all_to_all_workers,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = not_all_to_all_compute_compile_time_args});
    }

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    writer_kernel_ids.reserve(cores.size());
    float winv = 1.0f / block_w;                    // bcast-w scaler
    float cinv = (1.0f / num_distributed_devices);  // bcast-cores scaler
    float cinv_one = 1.0f;                          // bcast-cores scaler for all-to-all cores not on first row/col
    auto bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    auto bfloat_cinv_value_one = bfloat16(cinv_one);
    uint32_t packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_value_one, bfloat_cinv_value_one});
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;

    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    CoreCoord core_start_offset = grid_offset.value_or(CoreCoord{0, 0});
    for (uint32_t core_idx_x = core_start_offset.x; core_idx_x < num_cores_x + core_start_offset.x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, core_start_offset.y}).x);
    }
    for (uint32_t core_idx_y = core_start_offset.y; core_idx_y < num_cores_y + core_start_offset.y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({core_start_offset.x, core_idx_y}).y);
    }

    // For write back calculation
    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        tt::log_debug("core: {}, {}", core.x, core.y);

        uint32_t width_index = 0;
        width_index = i;

        uint32_t width_index_two_stage = width_index % num_blocks_first_stage;

        uint32_t gamma_tile_start_id = width_index * block_wt;

        std::vector<uint32_t> compute_args{};
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            bool is_second_stage_reader;
            if (use_two_stage_reduce) {
                is_second_stage_reader = width_index < 1;
            } else {
                is_second_stage_reader = false;
            }
            compute_args.push_back((uint32_t)(!(use_two_stage_reduce && (!is_second_stage_reader))));
            compute_args.push_back((uint32_t)num_distributed_devices);
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id_all_to_all, core, compute_args);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
        }

        if (width_index == 0) {
            CoreCoord mcast_start, mcast_end;
            CoreCoord top_left_core = {(std::size_t)start_core.x, (std::size_t)start_core.y};
            CoreCoord bottom_right_core = {
                (std::size_t)start_core.x + num_cores_x - 1, (std::size_t)start_core.y + num_cores_y - 1};
            auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
            auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
            mcast_start = top_left_core_physical;
            mcast_end = bottom_right_core_physical;
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            std::vector<uint32_t> mcast_sender_args;
            mcast_sender_args.push_back(mcast_start.x);
            mcast_sender_args.push_back(mcast_start.y);
            mcast_sender_args.push_back(mcast_end.x);
            mcast_sender_args.push_back(mcast_end.y);
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);
        } else if (
            (not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> mcast_receiver_args = {};
            tt::tt_metal::SetRuntimeArgs(
                program, reader_mcast_receiver_kernels_id_all_to_all, core, mcast_receiver_args);
        } else {
            std::vector<uint32_t> mcast_receiver_args = {};
            tt::tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
        }

        // Compute write back rt args

        std::vector<uint32_t> write_back_writer_args;

        uint32_t num_storage_cores = all_storage_cores.num_cores();

        write_back_writer_args.push_back(
            current_storage_core_offset * out_single_tile_size);  // storage_core_start_offset

        uint32_t current_worker_num_segments_to_write_back = 0;
        uint32_t worker_core_current_offset = 0;

        while (worker_core_current_offset <
               block_wt) {  // Continue until all worker core data has been written to corresponding storage cores
            uint32_t num_tiles_available_at_current_storage_core = block_wt_resharded - current_storage_core_offset;
            uint32_t num_tiles_left_on_current_worker_core = block_wt - worker_core_current_offset;
            uint32_t num_tiles_to_write_back =
                std::min(num_tiles_left_on_current_worker_core, num_tiles_available_at_current_storage_core);
            current_worker_num_segments_to_write_back += 1;

            tt::log_debug(
                "New segment for worker core {}, Worker core offset: {}, Storage core offset: {}, Num tiles to "
                "write "
                "back: {}",
                i,
                worker_core_current_offset,
                current_storage_core_offset,
                num_tiles_to_write_back);

            write_back_writer_args.push_back(num_tiles_to_write_back);                   // num_bytes_to_write_back
            write_back_writer_args.push_back(storage_core_noc_x[current_storage_core]);  // current_storage_core_noc_x
            write_back_writer_args.push_back(storage_core_noc_y[current_storage_core]);  // current_storage_core_noc_y
            worker_core_current_offset += num_tiles_to_write_back;
            current_storage_core_offset += num_tiles_to_write_back;

            if (current_storage_core_offset >= block_wt_resharded) {
                current_storage_core += 1;        // Move to next storage core
                current_storage_core_offset = 0;  // Reset offset on new storage core

                TT_ASSERT(
                    current_storage_core <= num_storage_cores,
                    "current_storage_core {} is exceeding number of storage cores {}",
                    current_storage_core,
                    num_storage_cores);
            }
        }
        TT_ASSERT(
            worker_core_current_offset == block_wt,
            "All worker core data should be written, but worker_core_current_offset {} != block_wt {}",
            worker_core_current_offset,
            block_wt);

        write_back_writer_args.insert(write_back_writer_args.begin(), current_worker_num_segments_to_write_back);

        // Set writer runtime args
        if ((not use_two_stage_reduce and width_index < num_cores_all_to_all) or
            (use_two_stage_reduce and width_index_two_stage < 1)) {
            std::vector<uint32_t> writer_mcast_sender_args;
            if (use_two_stage_reduce) {
                if (width_index < 1) {
                    writer_mcast_sender_args.push_back(packed_cinv_value);
                    writer_mcast_sender_args.push_back(packed_winv_value);
                } else {
                    writer_mcast_sender_args.push_back(packed_cinv_value_one);
                    writer_mcast_sender_args.push_back(packed_winv_value);
                }
            } else {
                writer_mcast_sender_args.push_back(packed_cinv_value);
                writer_mcast_sender_args.push_back(packed_winv_value);
            }
            writer_mcast_sender_args.push_back(e.u);
            writer_mcast_sender_args.push_back(gamma_dram_addr);
            writer_mcast_sender_args.push_back(gamma_tile_start_id);

            // Add args for write back (reshard)
            writer_mcast_sender_args.insert(
                writer_mcast_sender_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());

            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_sender_kernels_id, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_mcast_sender_kernels_id);
        } else {
            std::vector<uint32_t> writer_mcast_receiver_args;
            writer_mcast_receiver_args.push_back(packed_cinv_value);
            writer_mcast_receiver_args.push_back(packed_winv_value);
            writer_mcast_receiver_args.push_back(e.u);
            writer_mcast_receiver_args.push_back(gamma_dram_addr);
            writer_mcast_receiver_args.push_back(gamma_tile_start_id);

            // Add args for write back (reshard)
            writer_mcast_receiver_args.insert(
                writer_mcast_receiver_args.end(), write_back_writer_args.begin(), write_back_writer_args.end());

            tt::tt_metal::SetRuntimeArgs(program, writer_mcast_receiver_kernels_id, core, writer_mcast_receiver_args);
            writer_kernel_ids.push_back(writer_mcast_receiver_kernels_id);
        }
    }

    auto override_runtime_arguments_callback =
        [writer_kernel_ids,
         writer_mcast_sender_kernels_id,
         writer_mcast_receiver_kernels_id,
         num_none_all_to_all_workers,
         cb_in0,
         cb_stats,
         cb_output,
         cb_output_reshard,
         cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto src_buffer_a = input_tensors.at(0).buffer();
            const auto gamma_tensor = optional_input_tensors.at(1);
            const auto stats_tensor = optional_input_tensors.at(2);
            const auto dst_buffer = output_tensors.at(0).buffer();
            bool skip_write_back =
                output_tensors.at(0).shard_spec().value() == input_tensors.at(0).shard_spec().value();
            UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);
            if (!skip_write_back) {
                UpdateDynamicCircularBufferAddress(program, cb_output_reshard, *dst_buffer);
            } else {
                UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            }
            if (stats_tensor.has_value()) {
                const auto stats_buffer = optional_input_tensors.at(2).value().buffer();
                UpdateDynamicCircularBufferAddress(program, cb_stats, *stats_buffer);
            }

            auto& writer_sender_args_by_core = GetRuntimeArgs(program, writer_mcast_sender_kernels_id);
            auto& writer_receiver_args_by_core = num_none_all_to_all_workers > 0
                                                     ? GetRuntimeArgs(program, writer_mcast_receiver_kernels_id)
                                                     : writer_sender_args_by_core;

            const auto gamma_address = gamma_tensor.has_value() ? gamma_tensor.value().buffer()->address() : 0;

            for (uint32_t i = 0; i < cores.size(); ++i) {
                const CoreCoord& core = cores[i];

                const auto writer_kernel_id = writer_kernel_ids.at(i);

                if (writer_kernel_id == writer_mcast_sender_kernels_id) {
                    auto& runtime_args = writer_sender_args_by_core[core.x][core.y];
                    runtime_args[3] = gamma_address;

                } else if (writer_kernel_id == writer_mcast_receiver_kernels_id) {
                    auto& runtime_args = writer_receiver_args_by_core[core.x][core.y];
                    runtime_args[3] = gamma_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::fused::normalization
