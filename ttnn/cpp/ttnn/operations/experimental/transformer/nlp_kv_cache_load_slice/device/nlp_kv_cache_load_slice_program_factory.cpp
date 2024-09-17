// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "nlp_kv_cache_load_slice_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_unpad_runtime_args_tile_sharded(
    const Tensor &input_tensor,
    Tensor &output_tensor,
    const tt::tt_metal::LegacyShape &output_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_cores_x,
    uint32_t num_tiles_per_core) {
    auto input_buffer = input_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();

    vector<uint32_t> common_reader_kernel_args = {input_buffer->address(), 0};

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_id = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, Shape(output_tensor_start));
    const uint32_t num_tiles_shifted_per_core = input_shape[-2] * input_shape[-1] / TILE_HW;

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        // reader and writer kernel args
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[1] = start_id;
        vector<uint32_t> writer_kernel_args = {
            num_tiles_per_core,
        };
        ret_val[i] = {reader_kernel_args, writer_kernel_args};

        start_id += num_tiles_shifted_per_core;
    }

    return ret_val;
}

operation::ProgramWithCallbacks multi_core_nlp_kv_cache_load_slice(
    const Tensor &a, Tensor &output, const tt::tt_metal::LegacyShape &output_tensor_start, const tt::tt_metal::LegacyShape &output_tensor_end) {
    const tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();
    const tt::tt_metal::LegacyShape input_shape = a.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto shard_spec = output.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto num_cores_total = all_cores.num_cores();
    auto core_range = *all_cores.ranges().begin();
    auto num_cores_x = core_range.grid_size().x;
    uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
    auto num_tiles_per_core = num_units_per_shard_height * num_units_per_shard_width;

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_tiles_per_core;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // Shared reader and writer config setup
    uint32_t num_unpadded_tiles_head_dim = output_shape[-1] / TILE_WIDTH;
    uint32_t num_unpadded_tiles_seqlen_dim = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_tiles_seqlen_dim =
        (input_shape[-2] / TILE_HEIGHT - num_unpadded_tiles_seqlen_dim) * (input_shape[-1] / TILE_WIDTH);

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // Reader
    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_is_dram,
                                                      (std::uint32_t)num_tiles_per_core,
                                                      (std::uint32_t)num_unpadded_tiles_head_dim,
                                                      (std::uint32_t)num_unpadded_tiles_seqlen_dim,
                                                      (std::uint32_t)num_padded_tiles_seqlen_dim,
                                                      (std::uint32_t)num_cores_total};
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto all_runtime_args = get_unpad_runtime_args_tile_sharded(
        a, output, output_tensor_start, num_cores_total, num_cores_x, num_tiles_per_core);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        // Reader
        // Reader runtime args
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);

        // Writer
        // Writer runtime args
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0](
                                              const void *operation,
                                              Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        auto dst_tensor_buffer = dst_tensor.buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *dst_tensor_buffer);

        auto shard_spec = dst_tensor.shard_spec().value();
        auto all_cores = shard_spec.grid;
        auto num_cores_total = all_cores.num_cores();
        auto core_range = *all_cores.ranges().begin();
        auto num_cores_x = core_range.grid_size().x;
        uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        auto num_tiles_per_core = num_units_per_shard_height * num_units_per_shard_width;

        const auto tensor_start = static_cast<const ttnn::operations::data_movement::SliceDeviceOperation *>(operation)->slice_start;
        auto all_runtime_args = get_unpad_runtime_args_tile_sharded(
            src_tensor, dst_tensor, tensor_start, num_cores_total, num_cores_x, num_tiles_per_core);

        for (uint32_t i = 0; i < num_cores_total; i++) {
            CoreCoord core = {i % num_cores_x, i / num_cores_x};
            { SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first); }
            { SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second); }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // ttnn::operations::experimental::transformer
