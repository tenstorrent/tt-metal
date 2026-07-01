// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "fill_cache_multi_core_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

ProgramDescriptor FillCacheMultiCoreProgramFactory::create_descriptor(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    // TODO: For interleaved and kv_heads > 1, we assert that each core only gets 1 tile along seq_len
    // For sharded, each core gets shard_shape[0] number of tiles along seq_len.
    // For either case, assume that work doesn't spill over to next head, so we just increment by Wt within
    // reader/writer
    uint32_t num_blocks_of_work = input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / TILE_HEIGHT;

    uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t input_Ht = input_tensor.padded_shape()[-2] / TILE_HEIGHT;  // seq_len
    uint32_t cache_HtWt = cache_tensor.padded_shape()[-2] * Wt / TILE_HEIGHT;
    uint32_t cache_CHtWt = cache_tensor.padded_shape()[1] * cache_HtWt;
    uint32_t update_idxt = update_idx / TILE_HEIGHT;
    uint32_t start_idx = (batch_idx * cache_CHtWt) + (update_idxt * Wt);
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_blocks_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
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
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
        num_input_tiles = 2;  // double buffered
    }

    // ---- Build the ProgramDescriptor ----

    ProgramDescriptor desc;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();

    uint32_t src0_cb_index = 0;
    // For sharded inputs, set CBDescriptor::buffer so the framework refreshes the dynamic
    // CB address (equivalent to the old set_globally_allocated_address +
    // UpdateDynamicCircularBufferAddress pair).
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = shard_spec.has_value() ? src_buffer : nullptr,
    });

    uint32_t output_cb_index = src0_cb_index;

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args;
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
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = reader_kernel_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    // Per-core runtime args. We push raw buffer addresses (uint32_t) rather than Buffer*
    // because the per-core cache_start_id derives from operation_attributes (batch_idx,
    // update_idx) which UpdateKVCacheOperation::compute_program_hash deliberately excludes
    // from the program-cache key. With buffer_bindings empty the framework uses the
    // descriptor-rebuild slow path on cache hits, which correctly re-derives
    // cache_start_id every dispatch.
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                src_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written * Wt,
            });

        const uint32_t cache_start_id = start_idx                                       // user batch start
                                        + (num_blocks_written / input_Ht * cache_HtWt)  // cache head offset
                                        + ((num_blocks_written % input_Ht) * Wt);       // seq_len offset

        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer->address(),
                num_blocks_per_core * Wt,
                cache_start_id,
            });
        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
