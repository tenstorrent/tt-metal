// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "update_cache_multi_core_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

ProgramDescriptor UpdateCacheMultiCoreProgramFactory::create_descriptor(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto update_idx = operation_attributes.update_idx;
    const auto batch_offset = operation_attributes.batch_offset;
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config is required");
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();

    tt::DataFormat cache_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::tt_metal::IDevice* device = input_tensor.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_cb_data_format);

    uint32_t Wt = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Width size after untilize
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor.padded_shape()[-1] * sizeof(::bfloat16);

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);

    uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.padded_shape()[0];
    uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.padded_shape()[1];

    uint32_t B = input_tensor.padded_shape()[-2];
    uint32_t Bcache = cache_tensor.padded_shape()[0];
    const uint32_t granularity = std::min(static_cast<uint32_t>(2), Bcache);  // granularity = 2 best for performance
    uint32_t num_batched_heads = input_tensor.padded_shape()[1] * B / tt::constants::TILE_HEIGHT;
    uint32_t tile_update_offset = update_idx % tt::constants::TILE_HEIGHT * Wbytes;
    uint32_t batch_read_offset = batch_offset * Wbytes;  // Offset to read from input tensor

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_batched_heads_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_batched_heads_per_core_group_2 = 0;
        num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_batched_heads_per_core_group_1,
            num_batched_heads_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major);
        num_input_tiles = 2 * Wt;  // double buffered
    }

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();

    // ---- Build the ProgramDescriptor ----

    ProgramDescriptor desc;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_cache_tiles = 2 * granularity * Wt;  // double buffered
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cache_tiles * cache_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cache_cb_data_format,
            .page_size = cache_single_tile_size,
        }}},
    });

    // For sharded inputs, set CBDescriptor::buffer so the framework refreshes the dynamic
    // CB address (equivalent to the old set_globally_allocated_address +
    // UpdateDynamicCircularBufferAddress pair).
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = shard_spec.has_value() ? src_buffer : nullptr,
    });

    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    uint32_t interm1_cb_index = tt::CBIndex::c_25;

    uint32_t num_interm_tiles = 2 * granularity * Wt;  // double buffered
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * interm_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                .data_format = interm_cb_data_format,
                .page_size = interm_single_tile_size,
            },
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm1_cb_index),
                .data_format = interm_cb_data_format,
                .page_size = interm_single_tile_size,
            },
        }},
    });

    uint32_t interm2_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * interm_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(interm2_cb_index),
            .data_format = interm_cb_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    // Output is same tensor as cache input, so cb/tile size is same
    uint32_t output_cb_index = tt::CBIndex::c_16;

    // Must buffer all tiles for a single head
    uint32_t num_output_tiles = B * Wt;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * cache_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cache_cb_data_format,
            .page_size = cache_single_tile_size,
        }}},
    });

    const uint32_t u_range = std::min(static_cast<uint32_t>(32), Bcache);
    const uint32_t u_count = u_range / granularity;

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index, (std::uint32_t)src1_cb_index, (std::uint32_t)granularity, (std::uint32_t)u_count};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::map<std::string, std::string> reader_kernel_defines_map;
    if (shard_spec.has_value()) {
        reader_kernel_defines_map["INPUT_SHARDED"] = "1";
    }
    KernelDescriptor::Defines reader_kernel_defines;
    for (auto& kv : reader_kernel_defines_map) {
        reader_kernel_defines.emplace_back(kv.first, kv.second);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = reader_kernel_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)interm0_cb_index,
        (std::uint32_t)interm1_cb_index,
        (std::uint32_t)interm2_cb_index,
        (std::uint32_t)granularity,
        (std::uint32_t)u_count};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel(s) — group_1 has num_batched_heads_per_core_group_1, optional group_2
    // gets a second compute kernel with the group_2 count baked into compile-time args.
    std::vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        interm0_cb_index,
        interm1_cb_index,
        interm2_cb_index,
        output_cb_index,
        num_batched_heads_per_core_group_1,
        Wt,
        granularity,
        u_count};
    const auto make_compute_config = [&]() {
        return ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };
    };

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args;
    compute_desc_g1.config = make_compute_config();

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_g2 = compute_kernel_args;
        compute_kernel_args_g2[6] = num_batched_heads_per_core_group_2;
        KernelDescriptor desc_g2;
        desc_g2.kernel_source = "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp";
        desc_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        desc_g2.core_ranges = core_group_2;
        desc_g2.compile_time_args = std::move(compute_kernel_args_g2);
        desc_g2.config = make_compute_config();
        compute_desc_g2 = std::move(desc_g2);
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    // Per-core runtime args. We push raw buffer addresses (uint32_t) rather than Buffer*
    // because cache_start_id and tile_update_offset depend on operation_attributes
    // (update_idx) which UpdateKVCacheOperation::compute_program_hash deliberately
    // excludes from the program-cache key. With buffer_bindings empty, the framework
    // uses the descriptor-rebuild slow path on cache hits, which correctly re-derives
    // the per-core ids from the new update_idx every dispatch.
    uint32_t cache_tile_idx = update_idx / tt::constants::TILE_HEIGHT * Wt;
    uint32_t cache_start_id = 0;
    uint32_t input_start_id = 0;
    uint32_t batch_start_id = 0;
    uint32_t total_batched_heads = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_batched_heads_per_core;
        if (i < g1_numcores) {
            num_batched_heads_per_core = num_batched_heads_per_core_group_1;
        } else {
            num_batched_heads_per_core = num_batched_heads_per_core_group_2;
        }
        input_start_id = total_batched_heads * Wt;
        batch_start_id = (total_batched_heads * TILE_HEIGHT) % B;
        // Batch Offset + Head Offset + Index Offset
        cache_start_id = batch_start_id * cache_batch_num_tiles +
                         ((total_batched_heads * tt::constants::TILE_HEIGHT) / B) * cache_head_num_tiles;
        cache_start_id += cache_tile_idx;
        reader_desc.emplace_runtime_args(
            core,
            {dst_buffer->address(),
             src_buffer->address(),
             Wt,
             Bcache,
             num_batched_heads_per_core,
             cache_total_num_tiles,
             cache_batch_num_tiles,
             cache_head_num_tiles,
             cache_start_id,
             input_start_id,
             batch_start_id});

        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer->address(),
             Wt,
             Bcache,
             num_batched_heads_per_core,
             cache_total_num_tiles,
             cache_batch_num_tiles,
             cache_head_num_tiles,
             cache_start_id,
             batch_start_id,
             Wbytes,
             tile_update_offset,
             batch_read_offset});
        total_batched_heads += num_batched_heads_per_core;
    }

    // Stable kernel order (reader, writer, compute, [optional second compute]) — this is
    // the kernel index the framework uses when applying runtime args on cache hits.
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
