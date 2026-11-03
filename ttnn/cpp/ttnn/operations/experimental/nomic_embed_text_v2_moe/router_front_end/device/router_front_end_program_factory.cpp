// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "router_front_end_device_operation.hpp"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/common/device/page_multicast.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/router_front_end/device/kernels/";

// Two slots of a block's input tiles, scores and rows: the reader fetches the next block and the writer drains
// the last while compute works on this one.
constexpr uint32_t kSlots = 2;

// The softmax's buffers take ttnn.softmax's formats for a float32 input under fp32 accumulation, so the
// probabilities come out as its do.
enum CircularBufferIndex : uint8_t {
    kInput = tt::CBIndex::c_0,       // a block's input tiles
    kWeight = tt::CBIndex::c_1,      // the router weight, all of K
    kLogits = tt::CBIndex::c_2,      // input @ weight
    kMaskPadded = tt::CBIndex::c_3,  // 0 on the experts' columns, -inf past them
    kMasked = tt::CBIndex::c_4,      // the logits plus the mask
    kMax = tt::CBIndex::c_5,         // each row's max
    kMaxScaler = tt::CBIndex::c_6,
    kExps = tt::CBIndex::c_7,  // exp(masked - max)
    kSumScaler = tt::CBIndex::c_8,
    kRecipSum = tt::CBIndex::c_9,  // 1 / each row's sum
    kScores = tt::CBIndex::c_10,
    kRows = tt::CBIndex::c_11,  // a block's rows, row-major
};

constexpr uint32_t kWeightReadySemaphore = 0;

tt::tt_metal::CBDescriptor buffer(
    const CoreRangeSet& cores, uint8_t index, tt::DataFormat format, uint32_t page_bytes, uint32_t num_pages) {
    return tt::tt_metal::CBDescriptor{
        .total_size = page_bytes * num_pages,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index,
            .data_format = format,
            .page_size = page_bytes,
        }}},
    };
}

// A core's blocks, [first, first + count).
struct CoreBlocks {
    CoreCoord core;
    uint32_t first = 0;
    uint32_t count = 0;
};

// Blocks [first, first + num_blocks) over the cores in order, the first num_blocks % cores one more each.
void split_blocks(
    const std::vector<CoreCoord>& cores, uint32_t first, uint32_t num_blocks, std::vector<CoreBlocks>& assignment) {
    const uint32_t num_cores = std::min(static_cast<uint32_t>(cores.size()), num_blocks);
    const uint32_t base = num_blocks / num_cores;
    const uint32_t extra = num_blocks % num_cores;
    for (uint32_t index = 0; index < num_cores; ++index) {
        const uint32_t count = base + (index < extra ? 1 : 0);
        assignment.push_back({cores[index], first, count});
        first += count;
    }
}

// An input in the split layer norm's layout, block-sharded column-major with its tile rows on the shard grid's
// columns, has each grid column take the blocks its shard column holds, so a block's reads stay in one column:
// at 4096 tokens the first block's reads took 5.2 against 10.2 us spread over the grid. Otherwise the cores take
// consecutive blocks in row-major order.
std::vector<CoreBlocks> assign_blocks(const Tensor& input, uint32_t num_blocks, const CoreCoord& grid) {
    std::vector<CoreBlocks> assignment;
    const auto& memory_config = input.memory_config();
    const auto& shard_spec = memory_config.shard_spec();
    if (memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED && shard_spec.has_value() &&
        shard_spec->orientation == tt::tt_metal::ShardOrientation::COL_MAJOR && shard_spec->grid.ranges().size() == 1) {
        const uint32_t shard_blocks = shard_spec->shape[0] / tt::constants::TILE_HEIGHT;
        uint32_t column = shard_spec->grid.ranges().front().start_coord.x;
        for (uint32_t first = 0; first < num_blocks; first += shard_blocks, ++column) {
            std::vector<CoreCoord> cores;
            for (uint32_t y = 0; y < grid.y; ++y) {
                cores.emplace_back(column, y);
            }
            split_blocks(cores, first, std::min(shard_blocks, num_blocks - first), assignment);
        }
        return assignment;
    }
    split_blocks(
        corerange_to_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}), std::nullopt, /*row_wise=*/true),
        0,
        num_blocks,
        assignment);
    return assignment;
}

}  // namespace

tt::tt_metal::ProgramDescriptor RouterFrontEndDeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;

    const auto layout = compute_layout(attributes, tensor_args);
    auto& [scores, rows] = output;
    auto* device = tensor_args.input.device();
    const std::vector<CoreBlocks> assignment =
        assign_blocks(tensor_args.input, layout.num_blocks, device->compute_with_storage_grid_size());
    std::vector<CoreRange> core_ranges;
    for (const CoreBlocks& item : assignment) {
        core_ranges.emplace_back(item.core);
    }
    const CoreRangeSet cores = CoreRangeSet().merge(core_ranges);

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    const uint32_t hidden_tiles = layout.hidden_tiles;
    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(buffer(cores, kInput, tt::DataFormat::Float16_b, bf16_tile_bytes, kSlots * hidden_tiles));
    descriptor.cbs.push_back(buffer(cores, kWeight, tt::DataFormat::Float32, fp32_tile_bytes, hidden_tiles));
    for (const uint8_t index : {kLogits, kMasked, kMax, kMaxScaler, kExps, kSumScaler, kRecipSum}) {
        descriptor.cbs.push_back(buffer(cores, index, tt::DataFormat::Float32, fp32_tile_bytes, 1));
    }
    descriptor.cbs.push_back(buffer(cores, kMaskPadded, tt::DataFormat::Float16_b, bf16_tile_bytes, 1));
    descriptor.cbs.push_back(buffer(cores, kScores, tt::DataFormat::Float32, fp32_tile_bytes, kSlots));
    // A block's 32 rows take hidden_tiles tiles' bytes; the untilize writes them a tile-sized page at a time.
    descriptor.cbs.push_back(buffer(cores, kRows, tt::DataFormat::Float16_b, bf16_tile_bytes, kSlots * hidden_tiles));

    // The writer multicasts the weight, on the NoC the input reads leave free, from a core with the fewest blocks.
    constexpr NOC kWriterNoc = NOC::NOC_1;
    const CoreCoord weight_sender =
        std::min_element(assignment.rbegin(), assignment.rend(), [](const CoreBlocks& a, const CoreBlocks& b) {
            return a.count < b.count;
        })->core;
    const std::vector<MulticastRectangle> rectangles =
        assignment.size() > 1 ? multicast_rectangles(*device, cores, weight_sender, kWriterNoc)
                              : std::vector<MulticastRectangle>{};
    const bool weight_multicast = !rectangles.empty();
    if (weight_multicast) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = kWeightReadySemaphore,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = cores,
            .initial_value = 0,
        });
    }

    KernelDescriptor reader;
    reader.kernel_source = kKernelDirectory + "dataflow/router_front_end_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.named_compile_time_args = {
        {"hidden_tiles", hidden_tiles},
        {"input_cb", kInput},
        {"max_scaler_cb", kMaxScaler},
        {"sum_scaler_cb", kSumScaler},
    };
    TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader.compile_time_args);
    reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};

    KernelDescriptor writer;
    writer.kernel_source = kKernelDirectory + "dataflow/router_front_end_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.named_compile_time_args = {
        {"hidden_tiles", hidden_tiles},
        {"num_experts", layout.num_experts},
        {"weight_cb", kWeight},
        {"mask_cb", kMaskPadded},
        {"scores_cb", kScores},
        {"rows_cb", kRows},
        {"weight_multicast", weight_multicast ? 1u : 0u},
        {"weight_ready_semaphore", kWeightReadySemaphore},
    };
    TensorAccessorArgs(*tensor_args.weight.buffer()).append_to(writer.compile_time_args);
    TensorAccessorArgs(*scores.buffer()).append_to(writer.compile_time_args);
    TensorAccessorArgs(*rows.buffer()).append_to(writer.compile_time_args);
    writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = kWriterNoc};

    KernelDescriptor compute;
    compute.kernel_source = kKernelDirectory + "compute/router_front_end.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.named_compile_time_args = {
        {"hidden_tiles", hidden_tiles},
        {"input_cb", kInput},
        {"weight_cb", kWeight},
        {"logits_cb", kLogits},
        {"mask_cb", kMaskPadded},
        {"masked_cb", kMasked},
        {"max_cb", kMax},
        {"max_scaler_cb", kMaxScaler},
        {"exps_cb", kExps},
        {"sum_scaler_cb", kSumScaler},
        {"recip_sum_cb", kRecipSum},
        {"scores_cb", kScores},
        {"rows_cb", kRows},
    };
    compute.defines.emplace_back("EXP_APPROX", math_approx_mode ? "1" : "0");
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    const std::vector<uint32_t> rectangle_args = multicast_rectangle_args(rectangles);
    for (const auto& [core, first, count] : assignment) {
        reader.emplace_runtime_args(core, {tensor_args.input.buffer(), first, count});
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> writer_args{
            tensor_args.weight.buffer(),
            scores.buffer(),
            rows.buffer(),
            first,
            count,
            layout.num_tokens,
            core == weight_sender ? 1u : 0u};
        writer_args.insert(writer_args.end(), rectangle_args.begin(), rectangle_args.end());
        writer.emplace_runtime_args(core, writer_args);
        compute.emplace_runtime_args(core, {count});
    }

    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
