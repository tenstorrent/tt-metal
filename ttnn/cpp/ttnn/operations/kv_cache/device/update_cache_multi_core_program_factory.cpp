// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "update_cache_multi_core_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

UpdateCacheMultiCoreProgramFactory::cached_program_t UpdateCacheMultiCoreProgramFactory::create(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*output_tensor*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto update_idx = operation_attributes.update_idx;
    const auto batch_offset = operation_attributes.batch_offset;
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config is required");
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();

    Program program{};

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
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_cache_tiles * cache_single_tile_size, {{src0_cb_index, cache_cb_data_format}})
            .set_page_size(src0_cb_index, cache_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, input_single_tile_size);
    if (shard_spec.has_value()) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*input_tensor.buffer());
    }
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    uint32_t interm1_cb_index = tt::CBIndex::c_25;

    uint32_t num_interm_tiles = 2 * granularity * Wt;  // double buffered
    std::map<uint8_t, tt::DataFormat> interim_data_format_spec = {
        {interm0_cb_index, interm_cb_data_format}, {interm1_cb_index, interm_cb_data_format}};
    tt::tt_metal::CircularBufferConfig cb_interm0_config =
        tt::tt_metal::CircularBufferConfig(num_interm_tiles * interm_single_tile_size, interim_data_format_spec)
            .set_page_size(interm0_cb_index, interm_single_tile_size)
            .set_page_size(interm1_cb_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_interm0_config);

    uint32_t interm2_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig cb_interm2_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * interm_single_tile_size, {{interm2_cb_index, interm_cb_data_format}})
            .set_page_size(interm2_cb_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_interm2_config);

    // Output is same tensor as cache input, so cb/tile size is same
    uint32_t output_cb_index = tt::CBIndex::c_16;

    // Must buffer all tiles for a single head
    uint32_t num_output_tiles = B * Wt;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * cache_single_tile_size, {{output_cb_index, cache_cb_data_format}})
            .set_page_size(output_cb_index, cache_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

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

    std::map<std::string, std::string> reader_kernel_defines;
    if (shard_spec.has_value()) {
        reader_kernel_defines["INPUT_SHARDED"] = "1";
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_kernel_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

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

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    if (!core_group_2.ranges().empty()) {
        compute_kernel_args[6] = num_batched_heads_per_core_group_2;
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    uint32_t cache_tile_idx = update_idx / tt::constants::TILE_HEIGHT * Wt;
    uint32_t cache_start_id = 0;
    uint32_t input_start_id = 0;
    uint32_t batch_start_id = 0;
    uint32_t total_batched_heads = 0;
    std::vector<uint32_t> cache_start_ids;
    cache_start_ids.reserve(num_cores);
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
        cache_start_ids.push_back(cache_start_id);
        cache_start_id += cache_tile_idx;
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
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

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
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

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .unary_reader_kernel_id = unary_reader_kernel_id,
            .unary_writer_kernel_id = unary_writer_kernel_id,
            .cores = cores,
            .Wbytes = Wbytes,
            .Wt = Wt,
            .cache_start_ids = cache_start_ids,
            .cb_src1 = cb_src1,
        }};
}

void UpdateCacheMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KvCacheParams& operation_attributes,
    const KvCacheInputs& tensor_args,
    Tensor& /*output_tensor*/) {
    auto& program = cached_program.program;
    const auto Wbytes = cached_program.shared_variables.Wbytes;
    const auto Wt = cached_program.shared_variables.Wt;
    const auto& cache_start_ids = cached_program.shared_variables.cache_start_ids;
    const auto& cb_src1 = cached_program.shared_variables.cb_src1;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const auto update_idx = operation_attributes.update_idx;

    uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
    uint32_t cache_tile_idx = update_idx / TILE_HEIGHT * Wt;

    auto* src_buffer = tensor_args.input.buffer();

    auto* dst_buffer = tensor_args.cache.buffer();

    if (tensor_args.input.is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer);
    }

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t curr_cache_start_id = cache_start_ids[i] + cache_tile_idx;
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = src_buffer->address();
            runtime_args[8] = curr_cache_start_id;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[7] = curr_cache_start_id;
            runtime_args[10] = tile_update_offset;
        }
    }
}

}  // namespace ttnn::prim
