// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "fill_cache_multi_core_program_factory.hpp"
#include "update_cache_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::kv_cache {

using namespace tt::constants;

operation::ProgramWithCallbacks fill_cache_multi_core(
    const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx, const uint32_t update_idx) {
    Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

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
    uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    std::optional<ShardSpec> shard_spec = input_tensor.shard_spec();

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

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    if (shard_spec.has_value()) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*input_tensor.buffer());
    }
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    std::map<std::string, std::string> reader_kernel_defines;
    if (shard_spec.has_value()) {
        reader_kernel_defines["INPUT_SHARDED"] = "1";
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_kernel_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written * Wt,
            });

        const uint32_t cache_start_id = start_idx                                     // user batch start
                                        + num_blocks_written / input_Ht * cache_HtWt  // cache head offset
                                        + (num_blocks_written % input_Ht) * Wt;       // seq_len offset

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_blocks_per_core * Wt,
                cache_start_id,
            });
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           cb_src0,
                                           cores,
                                           g1_numcores,
                                           core_group_1,
                                           num_blocks_per_core_group_1,
                                           core_group_2,
                                           num_blocks_per_core_group_2,
                                           Wt,
                                           input_Ht,
                                           cache_HtWt,
                                           cache_CHtWt](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto batch_idx = static_cast<const UpdateCache*>(operation)->batch_idx;
        const auto update_idx = static_cast<const UpdateCache*>(operation)->update_idx;

        uint32_t update_idxt = update_idx / TILE_HEIGHT;
        uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;

        auto src_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = input_tensors.at(0).buffer();

        if (input_tensors.at(1).is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        }

        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); i++) {
            const CoreCoord& core = cores.at(i);
            uint32_t num_blocks_per_core = 0;
            if (i < g1_numcores) {
                num_blocks_per_core = num_blocks_per_core_group_1;
            } else {
                num_blocks_per_core = num_blocks_per_core_group_2;
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                const uint32_t cache_start_id = start_idx                                     // user batch start
                                                + num_blocks_written / input_Ht * cache_HtWt  // cache head offset
                                                + (num_blocks_written % input_Ht) * Wt;       // seq_len offset

                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[2] = cache_start_id;
            }
            num_blocks_written += num_blocks_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::kv_cache
