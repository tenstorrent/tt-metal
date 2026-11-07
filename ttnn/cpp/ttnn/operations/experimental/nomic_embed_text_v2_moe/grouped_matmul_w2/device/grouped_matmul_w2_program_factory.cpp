// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w2_device_operation.hpp"

#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/grouped_matmul_w1_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w2/device/kernels/";

enum CircularBufferIndex : uint8_t {
    kInputRing = tt::CBIndex::c_0,
    kInputCounts = tt::CBIndex::c_1,
    kFirstHalfRing = tt::CBIndex::c_2,
    kSecondHalfRing = tt::CBIndex::c_3,
    kOutputRing = tt::CBIndex::c_4,
    kSchedule = tt::CBIndex::c_5,
    kWeightCounts = tt::CBIndex::c_6,
};

enum SemaphoreId : uint32_t {
    kInputArrived,
    kWeightGranted,
    kWeightGrantedOdd,
    kWeightArrived,
    kForwardFlag,
    // One input grant counter a ring slot.
    kInputGrantedBase,
};

// Weight reads in flight a share sender keeps: with column lanes the forwarders pace the stream and more reads only
// contend (8x512: 103.8 us at 4, 106.7 at 8); without them the stream is the call (2x512: 78.7 at 8, 80.2 at 4).
constexpr uint32_t kWeightDepth = 4;
constexpr uint32_t kWeightDepthRowLanesOnly = 8;

tt::tt_metal::CBDescriptor ring(
    const CoreRangeSet& cores,
    uint8_t index,
    tt::DataFormat format,
    uint32_t page_bytes,
    uint32_t num_pages,
    const std::optional<tt::tt_metal::TileDescriptor>& tile = std::nullopt) {
    return tt::tt_metal::CBDescriptor{
        .total_size = page_bytes * num_pages,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index,
            .data_format = format,
            .page_size = page_bytes,
            .tile = tile,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor GroupedMatmulW2DeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;
    using tt::tt_metal::Tile;
    using tt::tt_metal::TileDescriptor;

    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    auto* counts_buffer = tensor_args.expert_counts.buffer();
    auto* device = input.device();
    const auto grid = device->compute_with_storage_grid_size();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    constexpr uint32_t kTile = tt::constants::TILE_HEIGHT;
    const uint32_t rows = input.logical_shape()[2];
    const uint32_t k_tiles = input.logical_shape()[3] / kTile;
    const uint32_t n_tiles = attributes.weight_n / kTile;
    const uint32_t input_blocks = rows / kTile;
    const uint32_t num_experts = tensor_args.expert_counts.logical_shape()[1];
    const auto map = grouped_matmul_w2_map(input_blocks, grid);
    const auto chunks =
        grouped_weight_chunks(k_tiles, n_tiles, num_experts, false, attributes.weight_chunk_tiles, grid.x);
    const uint32_t share_tiles = chunks.share_n_tiles;
    const uint32_t slice_tiles = k_tiles * share_tiles;
    const uint32_t half_tiles = slice_tiles / 2;
    const uint32_t input_piece_tiles = k_tiles / kGroupedMatmulW2InputPieces;

    const auto compute_list = map.compute_cores();
    std::vector<CoreCoord> dedicated;
    for (uint32_t lane = 0; lane < map.num_lanes; ++lane) {
        if (map.dedicated_sender(lane)) {
            dedicated.push_back(map.input_sender(lane));
        }
    }
    const CoreRangeSet compute_cores(ttsl::Span<const CoreCoord>(compute_list.data(), compute_list.size()));
    std::vector<CoreCoord> all_list = compute_list;
    all_list.insert(all_list.end(), dedicated.begin(), dedicated.end());
    const CoreRangeSet all_cores(ttsl::Span<const CoreCoord>(all_list.data(), all_list.size()));

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const Tile plain_tile({kTile, kTile});
    const Tile weight_tile({kTile, kTile}, /*transpose_tile=*/true);
    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(weight.dtype());
    const auto output_format = tt::DataFormat::Float16_b;
    const uint32_t input_page_bytes = tt::round_up(plain_tile.get_tile_size(input_format), dram_alignment);
    const uint32_t weight_page_bytes = tt::round_up(weight_tile.get_tile_size(weight_format), dram_alignment);
    const uint32_t output_tile_bytes = plain_tile.get_tile_size(output_format);
    const uint32_t counts_page_bytes = static_cast<uint32_t>(counts_buffer->aligned_page_size());
    const uint32_t schedule_bytes =
        tt::round_up(num_experts * static_cast<uint32_t>(sizeof(uint32_t)), tt::tt_metal::hal::get_l1_alignment());

    // The input ring and its counts page first on every core, so the ring sits at the same address on the dedicated
    // senders as on the compute cores.
    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(ring(
        all_cores,
        kInputRing,
        input_format,
        input_page_bytes,
        kGroupedMatmulW2InputSlots * input_piece_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(all_cores, kInputCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));
    descriptor.cbs.push_back(ring(
        compute_cores,
        kFirstHalfRing,
        weight_format,
        weight_page_bytes,
        attributes.first_slots * half_tiles,
        TileDescriptor{weight_tile}));
    descriptor.cbs.push_back(ring(
        compute_cores,
        kSecondHalfRing,
        weight_format,
        weight_page_bytes,
        attributes.second_slots * half_tiles,
        TileDescriptor{weight_tile}));
    descriptor.cbs.push_back(ring(
        compute_cores,
        kOutputRing,
        output_format,
        output_tile_bytes,
        kGroupedMatmulW2OutputSlots * share_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(compute_cores, kSchedule, tt::DataFormat::UInt32, schedule_bytes, 1));
    descriptor.cbs.push_back(ring(compute_cores, kWeightCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));
    for (uint32_t id = 0; id < kInputGrantedBase + kGroupedMatmulW2InputSlots; ++id) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    const auto noc = [&](const CoreCoord& core) { return device->worker_core_from_logical_core(core); };

    KernelDescriptor weight_kernel;
    weight_kernel.kernel_source = kKernelDirectory + "dataflow/weight_stream.cpp";
    weight_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    weight_kernel.core_ranges = compute_cores;
    weight_kernel.named_compile_time_args = {
        {"num_experts", num_experts},
        {"num_lanes", map.num_lanes},
        {"slice_tiles", slice_tiles},
        {"piece_tiles", attributes.weight_chunk_tiles},
        {"first_slots", attributes.first_slots},
        {"second_slots", attributes.second_slots},
        {"shard_tiles", chunks.shard_tiles},
        {"bank_sharers", chunks.bank_sharers},
        {"depth", map.column_lanes() ? kWeightDepth : kWeightDepthRowLanesOnly},
        {"counts_page_bytes", counts_page_bytes},
        {"first_half_cb", kFirstHalfRing},
        {"second_half_cb", kSecondHalfRing},
        {"n_tiles", share_tiles},
        {"output_cb", kOutputRing},
        {"counts_cb", kWeightCounts},
        {"schedule_cb", kSchedule},
        {"granted", kWeightGranted},
        {"granted_odd", kWeightGrantedOdd},
        {"arrived", kWeightArrived},
        {"forward_flag", kForwardFlag},
    };
    TensorAccessorArgs(*weight.buffer()).append_to(weight_kernel.compile_time_args);
    TensorAccessorArgs(*counts_buffer).append_to(weight_kernel.compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(weight_kernel.compile_time_args);
    weight_kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};

    KernelDescriptor input_kernel;
    input_kernel.kernel_source = kKernelDirectory + "dataflow/input_stream.cpp";
    input_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    input_kernel.core_ranges = all_cores;
    input_kernel.named_compile_time_args = {
        {"num_experts", num_experts},
        {"num_lanes", map.num_lanes},
        {"lane_cores", map.lane_cores},
        {"k_tiles", k_tiles},
        {"n_tiles", share_tiles},
        {"input_pieces", kGroupedMatmulW2InputPieces},
        {"input_slots", kGroupedMatmulW2InputSlots},
        {"counts_page_bytes", counts_page_bytes},
        {"input_cb", kInputRing},
        {"output_cb", kOutputRing},
        {"counts_cb", kInputCounts},
        {"granted_base", kInputGrantedBase},
        {"arrived", kInputArrived},
    };
    TensorAccessorArgs(*output.buffer()).append_to(input_kernel.compile_time_args);
    TensorAccessorArgs(*counts_buffer).append_to(input_kernel.compile_time_args);
    input_kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

    KernelDescriptor compute;
    compute.kernel_source = kKernelDirectory + "compute/grouped_matmul_w2.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = compute_cores;
    compute.named_compile_time_args = {
        {"num_experts", num_experts},
        {"k_tiles", k_tiles},
        {"n_tiles", share_tiles},
        {"input_cb", kInputRing},
        {"first_half_cb", kFirstHalfRing},
        {"second_half_cb", kSecondHalfRing},
        {"output_cb", kOutputRing},
        {"schedule_cb", kSchedule},
        {"piece_tiles", attributes.weight_chunk_tiles},
        {"input_piece_tiles", input_piece_tiles},
    };
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    const uint32_t no_multicast[5] = {0, 0, 0, 0, 0};
    for (uint32_t lane = 0; lane < map.num_lanes; ++lane) {
        const CoreCoord sender = map.input_sender(lane);
        const bool column_lane = lane >= GroupedMatmulW2Map::kRowLanes;
        // NOC1 multicasts start at the corner nearest the sender: west along a row, north up a column.
        const uint32_t receivers = map.lane_cores - (map.dedicated_sender(lane) ? 0 : 1);
        const CoreCoord lane_start =
            column_lane ? map.lane_core(lane, map.lane_cores - 2) : map.lane_core(lane, receivers - 1);
        const CoreCoord lane_end = map.lane_core(lane, 0);
        const CoreCoord start = noc(lane_start);
        const CoreCoord end = noc(lane_end);
        std::vector<uint32_t> lane_coords;
        for (uint32_t index = 0; index < map.lane_cores; ++index) {
            const CoreCoord core = noc(map.lane_core(lane, index));
            lane_coords.push_back(core.x);
            lane_coords.push_back(core.y);
        }
        const CoreCoord sender_noc = noc(sender);

        std::vector<CoreCoord> input_cores;
        for (uint32_t index = 0; index < map.lane_cores; ++index) {
            input_cores.push_back(map.lane_core(lane, index));
        }
        if (map.dedicated_sender(lane)) {
            input_cores.push_back(sender);
        }
        for (const auto& core : input_cores) {
            // 0: a receiver; 1: the lane's sender, computing; 2: a dedicated sender.
            const uint32_t role = core == sender ? (map.dedicated_sender(lane) ? 2u : 1u) : 0u;
            KernelDescriptor::RTArgList args;
            args.push_back(input.buffer());
            args.push_back(output.buffer());
            args.push_back(counts_buffer);
            args.append(
                {input_blocks,
                 lane,
                 role == 2 ? 0u : map.share_of(core),
                 role,
                 role == 0 ? 1u : 0u,
                 static_cast<uint32_t>(sender_noc.x),
                 static_cast<uint32_t>(sender_noc.y)});
            if (role == 0) {
                args.append({no_multicast[0], no_multicast[1], no_multicast[2], no_multicast[3], no_multicast[4]});
            } else {
                args.append(
                    {static_cast<uint32_t>(start.x),
                     static_cast<uint32_t>(start.y),
                     static_cast<uint32_t>(end.x),
                     static_cast<uint32_t>(end.y),
                     receivers});
            }
            for (uint32_t value : lane_coords) {
                args.push_back(value);
            }
            input_kernel.emplace_runtime_args(core, args);
        }

        for (uint32_t index = 0; index < map.lane_cores; ++index) {
            const CoreCoord core = map.lane_core(lane, index);
            const uint32_t share = map.share_of(core);
            const bool is_sender = !column_lane && core.y == 0;
            const bool forwards = map.column_lanes() && !column_lane && core == map.forwarder(share);
            const CoreCoord upstream = noc(column_lane ? map.forwarder(share) : CoreCoord(share, 0));
            KernelDescriptor::RTArgList args;
            args.push_back(weight.buffer());
            args.push_back(counts_buffer);
            args.push_back(output.buffer());
            args.append(
                {input_blocks,
                 lane,
                 share,
                 is_sender ? 1u : 0u,
                 core == sender ? 1u : 0u,
                 static_cast<uint32_t>(upstream.x),
                 static_cast<uint32_t>(upstream.y)});
            if (is_sender) {
                const CoreCoord column_start = noc(CoreCoord(share, 1));
                const CoreCoord column_end = noc(CoreCoord(share, grid.y - 1));
                args.append(
                    {static_cast<uint32_t>(column_start.x),
                     static_cast<uint32_t>(column_start.y),
                     static_cast<uint32_t>(column_end.x),
                     static_cast<uint32_t>(column_end.y),
                     static_cast<uint32_t>(grid.y - 1)});
            } else {
                args.append({no_multicast[0], no_multicast[1], no_multicast[2], no_multicast[3], no_multicast[4]});
            }
            if (forwards) {
                const uint32_t row = map.forward_row(share);
                const CoreCoord forward_start = noc(CoreCoord(map.num_shares, row));
                const CoreCoord forward_end = noc(CoreCoord(grid.x - 1, row));
                args.append(
                    {static_cast<uint32_t>(forward_start.x),
                     static_cast<uint32_t>(forward_start.y),
                     static_cast<uint32_t>(forward_end.x),
                     static_cast<uint32_t>(forward_end.y),
                     static_cast<uint32_t>(grid.x - map.num_shares)});
            } else {
                args.append({no_multicast[0], no_multicast[1], no_multicast[2], no_multicast[3], no_multicast[4]});
            }
            weight_kernel.emplace_runtime_args(core, args);
        }
    }

    descriptor.kernels.push_back(std::move(weight_kernel));
    descriptor.kernels.push_back(std::move(input_kernel));
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
