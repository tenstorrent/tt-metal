// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "update_cache_multi_core_program_factory.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor UpdateCacheMultiCoreProgramFactory::create_descriptor(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*output_tensor*/) {
    ProgramDescriptor desc;

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

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    CBDescriptor src1_cb_desc{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    };
    if (shard_spec.has_value()) {
        src1_cb_desc.buffer = input_tensor.buffer();
    }
    desc.cbs.push_back(std::move(src1_cb_desc));

    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    uint32_t interm1_cb_index = tt::CBIndex::c_25;

    uint32_t num_interm_tiles = 2 * granularity * Wt;  // double buffered
    // interm0 / interm1 share a single CB (same total_size) but expose two buffer indices.
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

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();

    const uint32_t u_range = std::min(static_cast<uint32_t>(32), Bcache);
    const uint32_t u_count = u_range / granularity;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index, (std::uint32_t)src1_cb_index, (std::uint32_t)granularity, (std::uint32_t)u_count};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)interm0_cb_index,
        (std::uint32_t)interm1_cb_index,
        (std::uint32_t)interm2_cb_index,
        (std::uint32_t)granularity,
        (std::uint32_t)u_count};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_kernel_defines;
    if (shard_spec.has_value()) {
        reader_kernel_defines.emplace_back("INPUT_SHARDED", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_kernel_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
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

    KernelDescriptor compute_desc_group_1;
    compute_desc_group_1.kernel_source = "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp";
    compute_desc_group_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_group_1.core_ranges = core_group_1;
    compute_desc_group_1.compile_time_args = std::move(compute_kernel_args_group_1);
    compute_desc_group_1.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    desc.kernels.push_back(std::move(compute_desc_group_1));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            src0_cb_index,
            src1_cb_index,
            interm0_cb_index,
            interm1_cb_index,
            interm2_cb_index,
            output_cb_index,
            num_batched_heads_per_core_group_2,
            Wt,
            granularity,
            u_count};

        KernelDescriptor compute_desc_group_2;
        compute_desc_group_2.kernel_source =
            "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp";
        compute_desc_group_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_group_2.core_ranges = core_group_2;
        compute_desc_group_2.compile_time_args = std::move(compute_kernel_args_group_2);
        compute_desc_group_2.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        desc.kernels.push_back(std::move(compute_desc_group_2));
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

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
            {dst_buffer,
             src_buffer,
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
            {dst_buffer,
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

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
