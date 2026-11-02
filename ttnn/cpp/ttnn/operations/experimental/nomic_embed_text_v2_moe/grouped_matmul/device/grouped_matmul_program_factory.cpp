// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_device_operation.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul/device/kernels/";

// Two slots per ring: the next block or expert loads while the current one computes.
constexpr uint32_t kRingSlots = 2;

enum CircularBufferIndex : uint8_t {
    kInputRing = tt::CBIndex::c_0,
    kWeightRing = tt::CBIndex::c_1,
    kOutputRing = tt::CBIndex::c_2,
    kSchedule = tt::CBIndex::c_3,
    kInputCounts = tt::CBIndex::c_4,
    kWeightCounts = tt::CBIndex::c_5,
};

enum SemaphoreId : uint32_t {
    kInputSender,
    kInputReceiver,
    kWeightSender,
    kWeightReceiver,
    kWeightSlotFree,
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

tt::tt_metal::ProgramDescriptor GroupedMatmulDeviceOperation::create_descriptor(
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
    const auto& counts = tensor_args.expert_counts;
    auto* device = input.device();
    const bool row_major_output = attributes.output_layout == tt::tt_metal::Layout::ROW_MAJOR;
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    // Core (column, lane): the lane's input sender is column 0, the column's weight sender lane 0. With
    // per-expert lanes every core reads its own weights, so there is no weight sender.
    const bool per_expert_lanes = attributes.per_expert_lanes;
    const uint32_t num_lanes = layout.num_lanes;
    const uint32_t num_columns = layout.num_columns;
    const CoreCoord last_core(num_columns - 1, num_lanes - 1);
    const CoreRangeSet all_cores{CoreRange(CoreCoord(0, 0), last_core)};

    // Tiles a dtype's DRAM pages hold are padded to the DRAM alignment, and the rings use the same
    // stride, so a tile lands where a read of its page puts it.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const Tile plain_tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
    const Tile weight_tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}, attributes.transpose_weight);
    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(weight.dtype());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.dtype);
    const uint32_t input_page_bytes = tt::round_up(plain_tile.get_tile_size(input_format), dram_alignment);
    const uint32_t weight_page_bytes = tt::round_up(weight_tile.get_tile_size(weight_format), dram_alignment);
    const uint32_t output_tile_bytes = plain_tile.get_tile_size(output_format);
    const uint32_t counts_page_bytes = static_cast<uint32_t>(counts.buffer()->aligned_page_size());
    // The compute kernel's schedule: a word per expert.
    const uint32_t schedule_bytes = tt::round_up(
        layout.num_experts * static_cast<uint32_t>(sizeof(uint32_t)), tt::tt_metal::hal::get_l1_alignment());
    // The kernels cut the regions at the input's blocks.
    const uint32_t input_blocks = layout.rows / tt::constants::TILE_HEIGHT;

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(ring(
        all_cores,
        kInputRing,
        input_format,
        input_page_bytes,
        kRingSlots * layout.k_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(
        all_cores,
        kWeightRing,
        weight_format,
        weight_page_bytes,
        kRingSlots * layout.k_tiles * layout.column_n_tiles,
        TileDescriptor{weight_tile}));
    descriptor.cbs.push_back(ring(
        all_cores,
        kOutputRing,
        output_format,
        output_tile_bytes,
        kRingSlots * layout.column_n_tiles,
        TileDescriptor{plain_tile}));
    descriptor.cbs.push_back(ring(all_cores, kSchedule, tt::DataFormat::UInt32, schedule_bytes, 1));
    descriptor.cbs.push_back(ring(all_cores, kInputCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));
    descriptor.cbs.push_back(ring(all_cores, kWeightCounts, tt::DataFormat::UInt32, counts_page_bytes, 1));

    for (uint32_t id : {kInputSender, kInputReceiver, kWeightSender, kWeightReceiver, kWeightSlotFree}) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // RISCV_1 on NOC1 reads the input, RISCV_0 on NOC0 the weights, as ttnn.matmul's 2D kernels do.
    // A NOC1 multicast runs from the far corner of its rectangle to the near one.
    const auto input_kernel = [&](bool is_sender) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/input_reader.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = is_sender ? CoreRangeSet{CoreRange(CoreCoord(0, 0), CoreCoord(0, num_lanes - 1))}
                                       : CoreRangeSet{CoreRange(CoreCoord(1, 0), last_core)};
        kernel.named_compile_time_args = {
            {"is_sender", is_sender ? 1u : 0u},
            {"per_expert_lanes", per_expert_lanes ? 1u : 0u},
            {"num_experts", layout.num_experts},
            {"num_lanes", num_lanes},
            {"k_tiles", layout.k_tiles},
            {"num_receivers", num_columns - 1},
            {"counts_page_bytes", counts_page_bytes},
            {"input_cb", kInputRing},
            {"counts_cb", kInputCounts},
            {"sender_semaphore", kInputSender},
            {"receiver_semaphore", kInputReceiver},
        };
        TensorAccessorArgs(*input.buffer()).append_to(kernel.compile_time_args);
        TensorAccessorArgs(*counts.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
        return kernel;
    };
    const auto weight_kernel = [&](bool is_sender) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/weight_reader_writer.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        if (per_expert_lanes) {
            kernel.core_ranges = all_cores;
        } else {
            kernel.core_ranges = is_sender ? CoreRangeSet{CoreRange(CoreCoord(0, 0), CoreCoord(num_columns - 1, 0))}
                                           : CoreRangeSet{CoreRange(CoreCoord(0, 1), last_core)};
        }
        // Expert e's tile (k, n) in the weight's pages: (1, 1, K, E * N) holds it at row k, column
        // e * N + n; read transposed, (1, 1, N, E * K) at row n, column e * K + k.
        const uint32_t expert_tiles =
            layout.num_experts * (attributes.transpose_weight ? layout.k_tiles : layout.n_tiles);
        kernel.named_compile_time_args = {
            {"is_sender", is_sender ? 1u : 0u},
            {"per_expert_lanes", per_expert_lanes ? 1u : 0u},
            {"num_experts", layout.num_experts},
            {"num_lanes", num_lanes},
            {"k_tiles", layout.k_tiles},
            {"slot_n_tiles", layout.column_n_tiles},
            {"num_receivers", num_lanes - 1},
            {"counts_page_bytes", counts_page_bytes},
            {"weight_stride_k", attributes.transpose_weight ? 1u : expert_tiles},
            {"weight_stride_n", attributes.transpose_weight ? expert_tiles : 1u},
            {"weight_stride_expert", attributes.transpose_weight ? layout.k_tiles : layout.n_tiles},
            {"output_n_tiles", layout.n_tiles},
            {"row_major_output", row_major_output ? 1u : 0u},
            {"subblock_w", layout.subblock_w},
            {"weight_cb", kWeightRing},
            {"output_cb", kOutputRing},
            {"counts_cb", kWeightCounts},
            {"schedule_cb", kSchedule},
            {"sender_semaphore", kWeightSender},
            {"receiver_semaphore", kWeightReceiver},
            {"slot_free_semaphore", kWeightSlotFree},
        };
        TensorAccessorArgs(*weight.buffer()).append_to(kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);
        TensorAccessorArgs(*counts.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        return kernel;
    };

    // A grid one core wide or tall has no receivers on that side. With per-expert lanes there is no
    // weight sender: one weight kernel, the receivers', runs on every core and reads its own weights.
    KernelDescriptor input_sender = input_kernel(true);
    std::optional<KernelDescriptor> input_receiver;
    if (num_columns > 1) {
        input_receiver = input_kernel(false);
    }
    std::optional<KernelDescriptor> weight_sender;
    std::optional<KernelDescriptor> weight_receiver;
    if (!per_expert_lanes) {
        weight_sender = weight_kernel(true);
    }
    if (per_expert_lanes || num_lanes > 1) {
        weight_receiver = weight_kernel(false);
    }

    KernelDescriptor compute;
    compute.kernel_source = kKernelDirectory + "compute/grouped_matmul.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
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
    };
    if (attributes.activation.has_value()) {
        const auto params = attributes.activation->get_params();
        compute.defines.emplace_back("SFPU_ACTIVATION", "1");
        compute.named_compile_time_args.emplace_back(
            "activation_type", static_cast<uint32_t>(ttnn::operations::matmul::KernelActivation::GELU_TANH));
        compute.named_compile_time_args.emplace_back(
            "activation_param0", !params.empty() && params[0] != 0.0f ? 1u : 0u);
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        for (uint32_t column = 0; column < num_columns; ++column) {
            const CoreCoord core(column, lane);
            const auto lane_sender = device->worker_core_from_logical_core(CoreCoord(0, lane));
            const auto lane_first =
                device->worker_core_from_logical_core(CoreCoord(std::min(1u, num_columns - 1), lane));
            const auto lane_last = device->worker_core_from_logical_core(CoreCoord(num_columns - 1, lane));
            const auto column_sender = device->worker_core_from_logical_core(CoreCoord(column, 0));
            const auto column_first =
                device->worker_core_from_logical_core(CoreCoord(column, std::min(1u, num_lanes - 1)));
            const auto column_last = device->worker_core_from_logical_core(CoreCoord(column, num_lanes - 1));
            const uint32_t n_tiles = column == num_columns - 1 ? layout.last_column_n_tiles : layout.column_n_tiles;

            // NOC1 multicasts from the far end of the lane, NOC0 from the near end of the column.
            (column == 0 ? input_sender : *input_receiver)
                .emplace_runtime_args(
                    core,
                    {input.buffer(),
                     counts.buffer(),
                     input_blocks,
                     lane,
                     static_cast<uint32_t>(lane_sender.x),
                     static_cast<uint32_t>(lane_sender.y),
                     static_cast<uint32_t>(lane_last.x),
                     static_cast<uint32_t>(lane_last.y),
                     static_cast<uint32_t>(lane_first.x),
                     static_cast<uint32_t>(lane_first.y)});
            (lane == 0 && weight_sender.has_value() ? *weight_sender : *weight_receiver)
                .emplace_runtime_args(
                    core,
                    {weight.buffer(),
                     output.buffer(),
                     counts.buffer(),
                     input_blocks,
                     lane,
                     column * layout.column_n_tiles,
                     n_tiles,
                     static_cast<uint32_t>(column_sender.x),
                     static_cast<uint32_t>(column_sender.y),
                     static_cast<uint32_t>(column_first.x),
                     static_cast<uint32_t>(column_first.y),
                     static_cast<uint32_t>(column_last.x),
                     static_cast<uint32_t>(column_last.y)});
            compute.emplace_runtime_args(core, {n_tiles});
        }
    }

    descriptor.kernels.push_back(std::move(input_sender));
    if (input_receiver.has_value()) {
        descriptor.kernels.push_back(std::move(*input_receiver));
    }
    if (weight_sender.has_value()) {
        descriptor.kernels.push_back(std::move(*weight_sender));
    }
    if (weight_receiver.has_value()) {
        descriptor.kernels.push_back(std::move(*weight_receiver));
    }
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
