// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w1_device_operation.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/kernels/";

enum CircularBufferIndex : uint8_t {
    kInputRing = tt::CBIndex::c_0,
    kWeightRing = tt::CBIndex::c_1,
    kOutputRing = tt::CBIndex::c_2,
    kSchedule = tt::CBIndex::c_3,
    kInputCounts = tt::CBIndex::c_4,
    kWeightCounts = tt::CBIndex::c_5,
    kOnes = tt::CBIndex::c_6,
    kStaging = tt::CBIndex::c_7,
};

enum SemaphoreId : uint32_t {
    kInputGranted,
    kInputArrived,
    kWeightGranted,
    kWeightArrived,
    kSenderFlag,
    kWeightGrantedOdd,
    kForwardFlag,
};

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

tt::tt_metal::ProgramDescriptor GroupedMatmulW1DeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;
    using tt::tt_metal::Tile;
    using tt::tt_metal::TileDescriptor;

    const auto layout = compute_layout(attributes, tensor_args);
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    // Without counts (one expert, every row) the kernels never read them; the input holds their place.
    const bool has_counts = tensor_args.expert_counts.has_value();
    auto* counts_buffer = has_counts ? tensor_args.expert_counts->buffer() : input.buffer();
    // Without a bias tensor the kernels never read one; the weight holds its place.
    auto* bias_buffer = tensor_args.bias.has_value() ? tensor_args.bias->buffer() : weight.buffer();
    auto* device = input.device();
    const bool row_major_output = attributes.output_layout == tt::tt_metal::Layout::ROW_MAJOR;
    const bool per_expert_lanes = attributes.per_expert_lanes;
    const bool chunked = attributes.weight_chunk_tiles > 0;
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    // Core (share, lane) computes, a row lane a core row. A share's weight sender is its top core (NOC0 south). A row
    // lane's input sender is its east core (NOC1 west) or, with dedicated senders, the cores east of the compute grid
    // in turn. In the folded layout the columns east of the row lanes' senders are lanes too, share s at row s, sent
    // from below (NOC1 north) by the cores under them, and share s's forwarder in its column passes each weight piece
    // to row s.
    const uint32_t num_lanes = layout.num_lanes;
    const uint32_t num_shares = layout.num_columns;
    const uint32_t num_senders = layout.input_senders;
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t row_lanes = layout.folded ? static_cast<uint32_t>(grid.y) : num_lanes;
    const uint32_t column_lanes = num_lanes - row_lanes;
    const uint32_t column_x0 = num_shares + layout.row_lane_senders;
    struct Lane {
        std::vector<CoreCoord> cores;  // by share
        std::vector<CoreCoord> senders;
        CoreCoord start;  // the input multicast's rectangle, start nearest the sender
        CoreCoord end;
        uint32_t receivers = 0;
    };
    std::vector<Lane> lanes(num_lanes);
    for (uint32_t lane = 0; lane < row_lanes; ++lane) {
        auto& plan = lanes[lane];
        for (uint32_t share = 0; share < num_shares; ++share) {
            plan.cores.emplace_back(share, lane);
        }
        const uint32_t dedicated = layout.folded ? layout.row_lane_senders : num_senders;
        for (uint32_t sender = 0; sender < dedicated; ++sender) {
            plan.senders.emplace_back(num_shares + sender, lane);
        }
        plan.start = CoreCoord(dedicated > 0 ? num_shares - 1 : std::max(num_shares, 2u) - 2, lane);
        plan.end = CoreCoord(0, lane);
        plan.receivers = dedicated > 0 ? num_shares : num_shares - 1;
    }
    for (uint32_t column = 0; column < column_lanes; ++column) {
        auto& plan = lanes[row_lanes + column];
        const uint32_t x = column_x0 + column;
        for (uint32_t share = 0; share < num_shares; ++share) {
            plan.cores.emplace_back(x, share);
        }
        for (uint32_t sender = 0; sender < num_senders; ++sender) {
            plan.senders.emplace_back(x, num_shares + sender);
        }
        plan.start = CoreCoord(x, num_shares - 1);
        plan.end = CoreCoord(x, 0);
        plan.receivers = num_shares;
    }
    const CoreRange row_cores(CoreCoord(0, 0), CoreCoord(num_shares - 1, row_lanes - 1));
    std::optional<CoreRange> column_cores;
    std::optional<CoreRangeSet> sender_cores;
    if (layout.folded) {
        column_cores = CoreRange(CoreCoord(column_x0, 0), CoreCoord(column_x0 + column_lanes - 1, num_shares - 1));
        sender_cores = CoreRangeSet{CoreRange(
            CoreCoord(column_x0, num_shares), CoreCoord(column_x0 + column_lanes - 1, num_shares + num_senders - 1))};
        if (layout.row_lane_senders > 0) {
            sender_cores = sender_cores->merge(CoreRangeSet{CoreRange(
                CoreCoord(num_shares, 0), CoreCoord(num_shares + layout.row_lane_senders - 1, row_lanes - 1))});
        }
    } else if (num_senders > 0) {
        sender_cores =
            CoreRangeSet{CoreRange(CoreCoord(num_shares, 0), CoreCoord(num_shares + num_senders - 1, num_lanes - 1))};
    }
    const CoreRangeSet compute_cores = column_cores.has_value()
                                           ? CoreRangeSet{std::vector<CoreRange>{row_cores, *column_cores}}
                                           : CoreRangeSet{row_cores};
    const CoreRangeSet all_cores = sender_cores.has_value() ? compute_cores.merge(*sender_cores) : compute_cores;

    // Tiles a dtype's DRAM pages hold are padded to the DRAM alignment, and the rings use the same stride, so a tile
    // lands where a read of its page puts it.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const Tile plain_tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
    const Tile weight_tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}, attributes.transpose_weight);
    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(weight.dtype());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.dtype);
    const uint32_t input_page_bytes = tt::round_up(plain_tile.get_tile_size(input_format), dram_alignment);
    const uint32_t weight_page_bytes = tt::round_up(weight_tile.get_tile_size(weight_format), dram_alignment);
    const uint32_t output_tile_bytes = plain_tile.get_tile_size(output_format);
    // A word an expert; with the tail split, word 1 holds the tail unit.
    const uint32_t schedule_bytes = tt::round_up(
        std::max(layout.num_experts, 2u) * static_cast<uint32_t>(sizeof(uint32_t)),
        tt::tt_metal::hal::get_l1_alignment());
    const uint32_t input_blocks = layout.rows / tt::constants::TILE_HEIGHT;
    // The counts page, a word an expert, read whole by every data-movement kernel.
    const uint32_t counts_page_bytes =
        has_counts ? static_cast<uint32_t>(counts_buffer->aligned_page_size()) : tt::tt_metal::hal::get_l1_alignment();

    // An L1-interleaved input: the sender steps the bank and address itself. A BLOCK_SHARDED one (the split norm's)
    // holds each shard's part of a row contiguously: one read a run.
    const auto& input_memory = input.memory_config();
    uint32_t input_banks = 0;
    if (input_memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        input_memory.buffer_type() == tt::tt_metal::BufferType::L1) {
        input_banks = static_cast<uint32_t>(device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1));
    }
    uint32_t input_run = 0;
    if (input_memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED &&
        input_memory.shard_spec().has_value()) {
        const uint32_t run = input_memory.shard_spec()->shape[1] / tt::constants::TILE_WIDTH;
        input_run = run > 0 && layout.k_tiles % run == 0 ? run : 0;
    }
    if (input_memory.is_sharded() && input_memory.nd_shard_spec().has_value() &&
        input_memory.memory_layout() != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        const uint32_t run = input_memory.nd_shard_spec()->shard_shape[-1] / tt::constants::TILE_WIDTH;
        input_run = run > 0 && layout.k_tiles % run == 0 ? run : 0;
    }

    // The input ring and both counts pages first on every core, so the ring sits at the same address on the
    // dedicated senders as on the compute cores.
    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(ring(
        all_cores,
        kInputRing,
        input_format,
        input_page_bytes,
        layout.input_slots * layout.k_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(all_cores, kInputCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));
    descriptor.cbs.push_back(ring(all_cores, kWeightCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));
    descriptor.cbs.push_back(ring(
        compute_cores,
        kWeightRing,
        weight_format,
        weight_page_bytes,
        layout.weight_slots * layout.slot_tiles,
        TileDescriptor{weight_tile}));
    descriptor.cbs.push_back(ring(
        compute_cores,
        kOutputRing,
        output_format,
        output_tile_bytes,
        layout.output_slots * layout.column_n_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(compute_cores, kSchedule, tt::DataFormat::UInt32, schedule_bytes, 1));
    if (layout.has_bias) {
        descriptor.cbs.push_back(
            ring(compute_cores, kOnes, input_format, input_page_bytes, 1, TileDescriptor{plain_tile}));
    }
    if (sender_cores.has_value()) {
        descriptor.cbs.push_back(ring(
            *sender_cores,
            kStaging,
            input_format,
            input_page_bytes,
            layout.staging_blocks * layout.k_tiles,
            TileDescriptor{plain_tile}));
    }
    for (uint32_t id :
         {kInputGranted, kInputArrived, kWeightGranted, kWeightArrived, kSenderFlag, kWeightGrantedOdd, kForwardFlag}) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    const auto noc = [&](uint32_t x, uint32_t y) { return device->worker_core_from_logical_core(CoreCoord(x, y)); };

    // A plain TILE weight's strides: (1, 1, K, E * N) holds tile (k, n) of expert e at row k, column e * N + n; read
    // transposed, (1, 1, N, E * K) at row n, column e * K + k.
    const uint32_t expert_tiles = layout.num_experts * (attributes.transpose_weight ? layout.k_tiles : layout.n_tiles);
    const uint32_t subblocks_max = layout.column_n_tiles / layout.subblock_w;

    // The weight stream's pieces; a plain weight's slice is read in pieces of a slot's width of tiles.
    const uint32_t piece_tiles = chunked ? layout.chunks.chunk_tiles : layout.column_n_tiles;

    KernelDescriptor weight_kernel;
    weight_kernel.kernel_source = kKernelDirectory + "dataflow/weight_stream.cpp";
    weight_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    weight_kernel.core_ranges = compute_cores;
    weight_kernel.named_compile_time_args = {
        {"num_experts", layout.num_experts},
        {"num_lanes", num_lanes},
        {"per_expert_lanes", per_expert_lanes ? 1u : 0u},
        {"k_tiles", layout.k_tiles},
        {"has_bias", layout.has_bias ? 1u : 0u},
        {"subblock_w", layout.subblock_w},
        {"slot_tiles", layout.slot_tiles},
        {"slots", layout.weight_slots},
        {"chunked", chunked ? 1u : 0u},
        {"chunk_tiles", piece_tiles},
        {"max_chunks", layout.chunks.max_chunks},
        {"shard_tiles", layout.chunks.shard_tiles},
        {"bank_sharers", layout.chunks.bank_sharers},
        {"stride_k", attributes.transpose_weight ? 1u : expert_tiles},
        {"stride_n", attributes.transpose_weight ? expert_tiles : 1u},
        {"stride_expert", attributes.transpose_weight ? layout.k_tiles : layout.n_tiles},
        {"output_n_tiles", layout.n_tiles},
        {"depth", layout.weight_depth},
        {"counts_page_bytes", counts_page_bytes},
        {"has_counts", has_counts ? 1u : 0u},
        {"tail_split", layout.tail_split ? 1u : 0u},
        {"subblocks_max", subblocks_max},
        {"weight_cb", kWeightRing},
        {"counts_cb", kWeightCounts},
        {"schedule_cb", kSchedule},
        {"granted", kWeightGranted},
        {"granted_odd", kWeightGrantedOdd},
        {"forward_flag", kForwardFlag},
        {"arrived", kWeightArrived},
    };
    TensorAccessorArgs(*weight.buffer()).append_to(weight_kernel.compile_time_args);
    TensorAccessorArgs(*counts_buffer).append_to(weight_kernel.compile_time_args);
    TensorAccessorArgs(*bias_buffer).append_to(weight_kernel.compile_time_args);
    weight_kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};

    // One instance a sender count: the row lanes', and the column lanes' in the folded layout.
    auto make_input_kernel = [&](const CoreRangeSet& cores, uint32_t lane_senders) {
        KernelDescriptor input_kernel;
        input_kernel.kernel_source = kKernelDirectory + "dataflow/input_stream.cpp";
        input_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        input_kernel.core_ranges = cores;
        input_kernel.named_compile_time_args = {
            {"num_experts", layout.num_experts},
            {"num_lanes", num_lanes},
            {"per_expert_lanes", per_expert_lanes ? 1u : 0u},
            {"k_tiles", layout.k_tiles},
            {"counts_page_bytes", counts_page_bytes},
            {"has_counts", has_counts ? 1u : 0u},
            {"input_cb", kInputRing},
            {"output_cb", kOutputRing},
            {"output_n_tiles", layout.n_tiles},
            {"row_major_output", row_major_output ? 1u : 0u},
            {"input_banks", input_banks},
            {"input_run", input_run},
            {"num_senders", lane_senders},
            {"subblock_outer", layout.subblock_outer ? 1u : 0u},
            {"subblock_w", layout.subblock_w},
            {"group_blocks", layout.group_blocks},
            {"tail_split", layout.tail_split ? 1u : 0u},
            {"subblocks_max", subblocks_max},
            {"counts_cb", kInputCounts},
            {"granted", kInputGranted},
            {"arrived", kInputArrived},
            {"has_bias", layout.has_bias ? 1u : 0u},
            {"ones_cb", kOnes},
        };
        TensorAccessorArgs(*input.buffer()).append_to(input_kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(input_kernel.compile_time_args);
        TensorAccessorArgs(*counts_buffer).append_to(input_kernel.compile_time_args);
        input_kernel.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
        return input_kernel;
    };
    KernelDescriptor input_kernel =
        make_input_kernel(CoreRangeSet{row_cores}, layout.folded ? layout.row_lane_senders : num_senders);
    std::optional<KernelDescriptor> column_input_kernel;
    if (column_cores.has_value()) {
        column_input_kernel = make_input_kernel(CoreRangeSet{*column_cores}, num_senders);
    }

    std::optional<KernelDescriptor> sender_kernel;
    if (sender_cores.has_value()) {
        sender_kernel.emplace();
        sender_kernel->kernel_source = kKernelDirectory + "dataflow/input_sender.cpp";
        sender_kernel->source_type = KernelDescriptor::SourceType::FILE_PATH;
        sender_kernel->core_ranges = *sender_cores;
        sender_kernel->named_compile_time_args = {
            {"num_experts", layout.num_experts},
            {"num_lanes", num_lanes},
            {"k_tiles", layout.k_tiles},
            {"counts_page_bytes", counts_page_bytes},
            {"has_counts", has_counts ? 1u : 0u},
            {"num_senders", num_senders},
            {"staging_slots", layout.staging_blocks},
            {"input_slots", layout.input_slots},
            {"input_banks", input_banks},
            {"input_run", input_run},
            {"input_cb", kInputRing},
            {"staging_cb", kStaging},
            {"counts_cb", kInputCounts},
            {"granted", kInputGranted},
            {"arrived", kInputArrived},
            {"flag", kSenderFlag},
        };
        TensorAccessorArgs(*input.buffer()).append_to(sender_kernel->compile_time_args);
        TensorAccessorArgs(*counts_buffer).append_to(sender_kernel->compile_time_args);
        sender_kernel->config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    }

    KernelDescriptor compute;
    compute.kernel_source = kKernelDirectory + "compute/grouped_matmul_w1.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = compute_cores;
    compute.named_compile_time_args = {
        {"num_experts", layout.num_experts},
        {"k_tiles", layout.k_tiles},
        {"slot_n_tiles", layout.column_n_tiles},
        {"subblock_w", layout.subblock_w},
        {"row_major_output", row_major_output ? 1u : 0u},
        {"transpose_weight", attributes.transpose_weight ? 1u : 0u},
        {"input_cb", kInputRing},
        {"weight_cb", kWeightRing},
        {"output_cb", kOutputRing},
        {"schedule_cb", kSchedule},
        {"has_bias", layout.has_bias ? 1u : 0u},
        {"ones_cb", kOnes},
        {"subblock_outer", layout.subblock_outer ? 1u : 0u},
        {"tail_split", layout.tail_split ? 1u : 0u},
        {"group_blocks", layout.group_blocks},
        {"chunk_tiles", piece_tiles},
    };
    if (attributes.activation.has_value()) {
        const auto params = attributes.activation->get_params();
        compute.defines.emplace_back("SFPU_ACTIVATION", "1");
        compute.named_compile_time_args.emplace_back(
            "activation_type", static_cast<uint32_t>(ttnn::operations::matmul::KernelActivation::GELU_TANH));
        compute.named_compile_time_args.emplace_back(
            "activation_param0", params.empty() ? 0u : static_cast<uint32_t>(params[0]));
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // A share's column of row-lane cores: its sender on top, receivers below. In the folded layout the forwarder
    // (share, share) passes each piece east along its own row to the share's column-lane cores. A forward to another
    // row goes round the NOC0 torus; with every share's forward doing so they would cross the same links and wait on
    // each other. Share 0's sender is (0, 0), so its forwarder is in the first row below the shares', whose links
    // eastward no other forward uses (from (0, 1) it shared row 1 with share 1's: 133 against 45 us forwarding).
    auto forwarder_row = [&](uint32_t share) { return share == 0 ? num_shares : share; };
    const CoreCoord no_core(0, 0);
    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        const auto& plan = lanes[lane];
        const bool column_lane = lane >= row_lanes;
        // A NOC1 multicast starts at the corner nearest its sender.
        const CoreCoord lane_start = noc(plan.start.x, plan.start.y);
        const CoreCoord lane_end = noc(plan.end.x, plan.end.y);
        const CoreCoord lane_east = noc(plan.cores.back().x, plan.cores.back().y);
        for (uint32_t share = 0; share < num_shares; ++share) {
            const CoreCoord core = plan.cores[share];
            const uint32_t n_tiles = share == num_shares - 1 ? layout.last_column_n_tiles : layout.column_n_tiles;
            const uint32_t slice_tiles = (layout.k_tiles + (layout.has_bias ? 1u : 0u)) * n_tiles;
            // Per-expert lanes: every core is its own weight sender, with no receivers.
            const bool weight_sender = per_expert_lanes || (!column_lane && lane == 0);
            const bool forwarder = layout.folded && !column_lane && lane == forwarder_row(share);
            const CoreCoord grant_to = column_lane ? noc(share, forwarder_row(share)) : noc(share, 0);
            const CoreCoord column_start = noc(share, std::min(1u, row_lanes - 1));
            const CoreCoord column_end = noc(share, row_lanes - 1);
            const CoreCoord forward_start = forwarder ? noc(column_x0, share) : no_core;
            const CoreCoord forward_end = forwarder ? noc(column_x0 + column_lanes - 1, share) : no_core;
            weight_kernel.emplace_runtime_args(
                core,
                {weight.buffer(),
                 counts_buffer,
                 bias_buffer,
                 input_blocks,
                 lane,
                 share,
                 share * layout.column_n_tiles,
                 slice_tiles,
                 weight_sender ? 1u : 0u,
                 static_cast<uint32_t>(grant_to.x),
                 static_cast<uint32_t>(grant_to.y),
                 static_cast<uint32_t>(column_start.x),
                 static_cast<uint32_t>(column_start.y),
                 static_cast<uint32_t>(column_end.x),
                 static_cast<uint32_t>(column_end.y),
                 per_expert_lanes ? 0u : row_lanes - 1,
                 static_cast<uint32_t>(forward_start.x),
                 static_cast<uint32_t>(forward_start.y),
                 static_cast<uint32_t>(forward_end.x),
                 static_cast<uint32_t>(forward_end.y),
                 forwarder ? column_lanes : 0u});

            KernelDescriptor::RTArgList input_args;
            input_args.push_back(input.buffer());
            input_args.push_back(output.buffer());
            input_args.push_back(counts_buffer);
            input_args.append(
                {input_blocks,
                 lane,
                 share * layout.column_n_tiles,
                 n_tiles,
                 plan.senders.empty() && share == num_shares - 1 ? 1u : 0u,
                 static_cast<uint32_t>(lane_east.x),
                 static_cast<uint32_t>(lane_east.y),
                 static_cast<uint32_t>(lane_start.x),
                 static_cast<uint32_t>(lane_start.y),
                 static_cast<uint32_t>(lane_end.x),
                 static_cast<uint32_t>(lane_end.y),
                 plan.receivers});
            for (const auto& sender : plan.senders) {
                const CoreCoord sender_core = noc(sender.x, sender.y);
                input_args.append({static_cast<uint32_t>(sender_core.x), static_cast<uint32_t>(sender_core.y)});
            }
            (column_lane ? *column_input_kernel : input_kernel).emplace_runtime_args(core, input_args);
            compute.emplace_runtime_args(core, {n_tiles});
        }
        for (uint32_t sender = 0; sender < plan.senders.size(); ++sender) {
            sender_kernel->emplace_runtime_args(
                plan.senders[sender],
                {input.buffer(),
                 counts_buffer,
                 input_blocks,
                 lane,
                 sender,
                 static_cast<uint32_t>(lane_start.x),
                 static_cast<uint32_t>(lane_start.y),
                 static_cast<uint32_t>(lane_end.x),
                 static_cast<uint32_t>(lane_end.y),
                 plan.receivers});
        }
    }

    descriptor.kernels.push_back(std::move(weight_kernel));
    descriptor.kernels.push_back(std::move(input_kernel));
    if (column_input_kernel.has_value()) {
        descriptor.kernels.push_back(std::move(*column_input_kernel));
    }
    if (sender_kernel.has_value()) {
        descriptor.kernels.push_back(std::move(*sender_kernel));
    }
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
