// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/paged_update_cache/paged_update_cache_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include <stdint.h>

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks paged_update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const std::vector<uint32_t> update_idxs, const uint32_t batch_offset, DeviceComputeKernelConfig compute_kernel_config) {
    Program program{};

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.get_dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt_metal::Device *device = input_tensor.device();

    bool fp32_dest_acc_en;
    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
            fp32_dest_acc_en = input_cb_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    uint32_t Wt = cache_tensor.get_legacy_shape()[-1] / TILE_WIDTH;

    // Width size after untilize
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.get_legacy_shape()[-1] * sizeof(float) : cache_tensor.get_legacy_shape()[-1] * sizeof(bfloat16);

    log_debug("cache_cb_data_format: {}", cache_cb_data_format);
    log_debug("input_cb_data_format: {}", input_cb_data_format);
    log_debug("interm_cb_data_format: {}", interm_cb_data_format);
    log_debug("Wbytes: {}", Wbytes);
    log_debug("Wt: {}", Wt);


    uint32_t cache_total_num_tiles = cache_tensor.volume() / TILE_HW;
    uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.get_legacy_shape()[0];
    // uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.get_legacy_shape()[1];

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    uint32_t B = input_tensor.get_legacy_shape()[1];
    uint32_t num_heads = cache_tensor.get_legacy_shape()[1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_batched_heads_per_core;

    CoreRangeSet all_cores({});

    std::optional<ShardSpec> shard_spec = input_tensor.shard_spec();

    uint32_t num_input_tiles;

    row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    all_cores = shard_spec.value().grid;
    num_cores = all_cores.num_cores();
    num_batched_heads_per_core = shard_spec.value().shape[0] / TILE_HEIGHT;
    num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    auto bbox = all_cores.bounding_box();
    num_cores_x = bbox.end_coord.x + 1;
    num_cores_y = bbox.end_coord.y + 1;

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_cache_tiles = 2 * Wt; // double buffered
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_cache_tiles * cache_single_tile_size, {{src0_cb_index, cache_cb_data_format}})
		.set_page_size(src0_cb_index, cache_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src1_cb_index, input_cb_data_format}})
		.set_page_size(src1_cb_index, input_single_tile_size);
    if (shard_spec.has_value()) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*input_tensor.buffer());
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    uint32_t interm0_cb_index = CB::c_intermed0;
    uint32_t interm1_cb_index = CB::c_intermed1;

    uint32_t num_interm_tiles = 2 * Wt; // double buffered
    std::map<uint8_t, tt::DataFormat> interim_data_format_spec = {
        {interm0_cb_index, interm_cb_data_format},
        {interm1_cb_index, interm_cb_data_format}
    };
    tt_metal::CircularBufferConfig cb_interm0_config = tt_metal::CircularBufferConfig(num_interm_tiles * interm_single_tile_size, interim_data_format_spec)
		.set_page_size(interm0_cb_index, interm_single_tile_size)
        .set_page_size(interm1_cb_index, interm_single_tile_size);
    auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm0_config);

    uint32_t interm2_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig cb_interm2_config = tt_metal::CircularBufferConfig(num_interm_tiles * interm_single_tile_size, {{interm2_cb_index, interm_cb_data_format}})
		.set_page_size(interm2_cb_index, interm_single_tile_size);
    auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm2_config);

    // Output is same tensor as cache input, so cb/tile size is same
    uint32_t output_cb_index = CB::c_out0;

    // Must buffer all tiles for a single head
    uint32_t num_output_tiles = B * Wt;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * cache_single_tile_size, {{output_cb_index, cache_cb_data_format}})
		.set_page_size(output_cb_index, cache_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) output_cb_index,
        (std::uint32_t) interm0_cb_index,
        (std::uint32_t) interm1_cb_index,
        (std::uint32_t) interm2_cb_index,
    };


    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/paged_update_cache/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/paged_update_cache/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        interm0_cb_index,
        interm1_cb_index,
        interm2_cb_index,
        output_cb_index,
        Wt,
    };

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/paged_update_cache/kernels/compute/update_cache.cpp",
        all_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args}
    );

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; ++i) {
        const CoreCoord &core = cores.at(i);
        const uint32_t update_idx = update_idxs.at(i);
        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        // Offset to write into untilized cache
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                dst_buffer->address(),
                Wt, cache_start_id
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                Wt, cache_start_id, Wbytes, tile_update_offset_B
            }
        );
    }

    auto override_runtime_arguments_callback = [
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        cores,
        Wbytes,
        Wt,
        cb_src1,
        cache_batch_num_tiles
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        const std::vector<uint32_t> update_idxs = static_cast<const PagedUpdateCache*>(operation)->update_idxs;



        auto src_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = input_tensors.at(0).buffer();

        if (input_tensors.at(1).is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer);
        }

        auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

        for (uint32_t i = 0, num_tiles_read = 0; i < cores.size(); ++i){
            const uint32_t update_idx = update_idxs.at(i);
            // Cache tile info
            const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
            const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
            // Offset to write into untilized cache
            uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

            const CoreCoord &core = cores.at(i);

            {
                auto &runtime_args = reader_args_by_core.at(core.x).at(core.y);
                runtime_args[0] = dst_buffer->address();
                runtime_args[2] = cache_start_id;
            }

            {
                auto &runtime_args = writer_args_by_core.at(core.x).at(core.y);
                runtime_args[0] = dst_buffer->address();
                runtime_args[2] = cache_start_id;
                runtime_args[4] = tile_update_offset_B;
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}   // namespace primary
}   // namespace operations
}   // namespace tt
