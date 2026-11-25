// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_combine_device_operation.hpp"

#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/common/device/page_multicast.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/routed_combine/device/kernels/";

// Two slots: the reader gathers the next unit's rows while compute works on this one.
constexpr uint32_t kSlots = 2;

// The reader's row reads in flight: batches of kBatchReads under kBatchDepth transaction ids. At 4096
// tokens 8 x 2 beat 16 x 2, 4 x 4 and 8 x 3.
constexpr uint32_t kBatchReads = 8;
constexpr uint32_t kBatchDepth = 2;

enum CircularBufferIndex : uint8_t {
    kRows = tt::CBIndex::c_0,
    kTiles = tt::CBIndex::c_1,
    kWeights = tt::CBIndex::c_2,
    kBias = tt::CBIndex::c_3,
    kOutput = tt::CBIndex::c_4,
    kIndex = tt::CBIndex::c_5,
};

constexpr uint32_t kBiasReadySemaphore = 0;

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

}  // namespace

tt::tt_metal::ProgramDescriptor RoutedCombineDeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;

    const auto layout = compute_layout(attributes, tensor_args);
    auto* device = tensor_args.expert_rows.device();
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet cores = tt::tt_metal::num_cores_to_corerangeset(layout.num_cores, grid, /*row_wise=*/true);

    const DeviceComputeKernelConfig& compute_config = attributes.compute_kernel_config;
    const uint32_t dest_tiles = get_dest_reg_count(compute_config);

    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t unit_tiles = layout.top_k * layout.chunk_tiles;

    // The bias goes out from the last core, which takes the fewest units.
    const std::vector<CoreCoord> core_list = corerange_to_cores(cores, std::nullopt, /*row_wise=*/true);
    const CoreCoord bias_sender = core_list.back();
    const std::vector<MulticastRectangle> rectangles =
        layout.combine && layout.num_cores > 1 ? multicast_rectangles(*device, cores, bias_sender, NOC::NOC_0)
                                               : std::vector<MulticastRectangle>{};
    const bool bias_multicast = !rectangles.empty();

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(buffer(cores, kRows, tt::DataFormat::Float16_b, bf16_tile_bytes, kSlots * unit_tiles));
    // A gather tilizes straight into the output: no tiles, weights or bias to hold.
    if (layout.combine) {
        descriptor.cbs.push_back(buffer(cores, kTiles, tt::DataFormat::Float16_b, bf16_tile_bytes, unit_tiles));
        descriptor.cbs.push_back(
            buffer(cores, kWeights, tt::DataFormat::Float32, fp32_tile_bytes, kSlots * layout.top_k));
        descriptor.cbs.push_back(buffer(cores, kBias, tt::DataFormat::Float16_b, bf16_tile_bytes, layout.hidden_tiles));
    }
    descriptor.cbs.push_back(
        buffer(cores, kOutput, tt::DataFormat::Float16_b, bf16_tile_bytes, kSlots * layout.chunk_tiles));
    descriptor.cbs.push_back(
        buffer(cores, kIndex, tt::DataFormat::UInt32, tt::constants::TILE_HEIGHT * sizeof(uint32_t), layout.top_k));

    KernelDescriptor reader;
    reader.kernel_source = kKernelDirectory + "dataflow/combine_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.named_compile_time_args = {
        {"combine", layout.combine ? 1u : 0u},
        {"top_k", layout.top_k},
        {"hidden_tiles", layout.hidden_tiles},
        {"chunk_tiles", layout.chunk_tiles},
        {"token_blocks", layout.padded_tokens / tt::constants::TILE_HEIGHT},
        {"rows_cb", kRows},
        {"weights_cb", kWeights},
        {"bias_cb", kBias},
        {"index_cb", kIndex},
        {"bias_multicast", bias_multicast ? 1u : 0u},
        {"slots", kSlots},
        {"batch_reads", kBatchReads},
        {"batch_depth", kBatchDepth},
        {"bias_ready_semaphore", kBiasReadySemaphore},
    };
    TensorAccessorArgs(*tensor_args.expert_rows.buffer()).append_to(reader.compile_time_args);
    TensorAccessorArgs(*tensor_args.combine_index.buffer()).append_to(reader.compile_time_args);
    // A gather reads neither; stand-ins keep the accessors' compile-time layout.
    for (const std::optional<Tensor>* operand : {&tensor_args.combine_weights, &tensor_args.bias}) {
        (operand->has_value() ? TensorAccessorArgs(*(*operand)->buffer())
                              : TensorAccessorArgs::create_dram_interleaved())
            .append_to(reader.compile_time_args);
    }
    reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};

    KernelDescriptor writer;
    writer.kernel_source = kKernelDirectory + "dataflow/combine_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.named_compile_time_args = {
        {"hidden_tiles", layout.hidden_tiles},
        {"chunk_tiles", layout.chunk_tiles},
        {"output_cb", kOutput},
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

    KernelDescriptor compute;
    compute.kernel_source = kKernelDirectory + "compute/combine.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.named_compile_time_args = {
        {"combine", layout.combine ? 1u : 0u},
        {"top_k", layout.top_k},
        {"hidden_tiles", layout.hidden_tiles},
        {"chunk_tiles", layout.chunk_tiles},
        {"dest_tiles", dest_tiles},
        {"rows_cb", kRows},
        {"tiles_cb", kTiles},
        {"weights_cb", kWeights},
        {"bias_cb", kBias},
        {"output_cb", kOutput},
    };
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_config);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    if (bias_multicast) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = kBiasReadySemaphore,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = cores,
            .initial_value = 0,
        });
    }
    const std::vector<uint32_t> rectangle_args = multicast_rectangle_args(rectangles);

    // Core c takes units [first, first + count), the first num_units % num_cores cores one more.
    const uint32_t base_units = layout.num_units / layout.num_cores;
    const uint32_t extra_units = layout.num_units % layout.num_cores;
    uint32_t core_index = 0;
    uint32_t first_unit = 0;
    for (const CoreCoord& core : core_list) {
        const uint32_t count = base_units + (core_index++ < extra_units ? 1 : 0);
        if (layout.combine) {
            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> args{
                tensor_args.expert_rows.buffer(),
                tensor_args.combine_index.buffer(),
                tensor_args.combine_weights->buffer(),
                tensor_args.bias->buffer(),
                first_unit,
                count,
                core == bias_sender ? 1u : 0u};
            args.insert(args.end(), rectangle_args.begin(), rectangle_args.end());
            reader.emplace_runtime_args(core, args);
        } else {
            reader.emplace_runtime_args(
                core,
                {tensor_args.expert_rows.buffer(),
                 tensor_args.combine_index.buffer(),
                 0u,
                 0u,
                 first_unit,
                 count,
                 0u,
                 0u});
        }
        writer.emplace_runtime_args(core, {output.buffer(), first_unit, count});
        compute.emplace_runtime_args(core, {first_unit, count});
        first_unit += count;
    }

    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
