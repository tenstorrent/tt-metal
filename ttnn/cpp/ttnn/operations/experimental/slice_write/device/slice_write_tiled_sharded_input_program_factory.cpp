// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_tiled_sharded_input_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/padded_slice/device/padded_slice_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::detail;

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t cb_input_index = 0;
constexpr uint32_t kSliceWriteTiledWriterAddressArgIdx = 0;

CBDescriptor make_slice_write_tiled_cb(
    uint32_t cb_index,
    const CoreRangeSet& core_ranges,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat data_format,
    Buffer* buffer = nullptr) {
    return CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    };
}

void emplace_slice_write_tiled_writer_args(
    KernelDescriptor& kernel, const CoreCoord& core, const std::vector<uint32_t>& args, Buffer* output_buffer) {
    KernelDescriptor::RTArgList runtime_args;
    runtime_args.reserve(args.size());
    for (uint32_t arg_idx = 0; arg_idx < args.size(); ++arg_idx) {
        if (arg_idx == kSliceWriteTiledWriterAddressArgIdx) {
            runtime_args.push_back(output_buffer);
        } else {
            runtime_args.push_back(args[arg_idx]);
        }
    }
    kernel.emplace_runtime_args(core, runtime_args);
}

SliceWriteRuntimeArgs get_slice_write_runtime_args_tiled_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores) {
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto actual_input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < actual_input_shape.rank(); i++) {
        actual_input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    auto output_shape = output_tensor.padded_shape();
    log_debug(tt::LogOp, "Slice Write Output Shape: {}", output_shape);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    auto shard_spec = input_tensor.shard_spec().value();
    auto input_cores = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input_tensor);

    std::uint32_t num_dims = static_cast<std::uint32_t>(actual_input_shape.rank());
    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> size_till_end(num_dims);

    num_input_tiles_per_dim[0] = tt::div_up(actual_input_shape[-1], (TILE_WIDTH * num_cores_channels));
    num_input_tiles_per_dim[1] = tt::div_up(actual_input_shape[-2], TILE_HEIGHT);

    num_output_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH) - num_input_tiles_per_dim[0];
    num_output_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) - num_input_tiles_per_dim[1];
    num_output_tiles_per_dim[1] *= tt::div_up(output_shape[-1], TILE_WIDTH);

    uint32_t num_tiles_per_channel = num_input_tiles_per_dim[0];

    log_debug(
        tt::LogOp,
        "Output Start : {}, Output End : {}, Actual Input Shape : {}, \n Input Shape : {}, Output Shape : {}",
        output_tensor_start,
        output_tensor_end,
        actual_input_shape,
        input_shape,
        output_shape);

    accumulated_total_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];

    uint32_t output_channel_tiles = accumulated_total_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[0] = num_input_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[1] = num_input_tiles_per_dim[1] * accumulated_input_total_tiles_per_dim[0];
    for (int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = actual_input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_tiles_per_dim[i - 1];
        num_input_tiles_per_dim[i] = num_unpadded_dim;
        num_output_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_tiles_per_dim[i] = num_total_dim * accumulated_total_tiles_per_dim[i - 1];
        accumulated_input_total_tiles_per_dim[i] = num_unpadded_dim * accumulated_input_total_tiles_per_dim[i - 1];
    }

    log_debug(
        tt::LogOp,
        "Slice Write Input Tiles {}, Output Tiles {}, Acc Output Tiles {}, Acc Input Tiles {}",
        num_input_tiles_per_dim,
        num_output_tiles_per_dim,
        accumulated_total_tiles_per_dim,
        accumulated_input_total_tiles_per_dim);

    using namespace tt::tt_metal::experimental;
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    log_debug(tt::LogOp, "Output Buffer address: {}", output_buffer->address());
    std::vector<uint32_t> common_writer_kernel_args = {
        0u, input_single_tile_size, input_single_tile_size, input_single_tile_size, num_dims, 0, 0, 0, 0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());

    auto num_cores_total = cores.size();

    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard Height {} should be a multiple of tile height",
        shard_spec.shape[0]);
    const auto num_tiles_nhw_per_core = shard_spec.shape[0] / TILE_HEIGHT;

    SliceWriteRuntimeArgs ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(output_tensor, output_tensor_start);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        } else if (is_width_sharded) {
            core_h_index = 0;
            core_w_index = core_index;
        }

        const uint32_t num_sticks_read = core_h_index * num_tiles_nhw_per_core;
        const uint32_t width_offset = core_w_index * num_tiles_per_channel;

        const uint32_t channels_tiles_this_core = std::min(output_channel_tiles - width_offset, num_tiles_per_channel);
        id_per_dim[0] = 0;
        uint32_t unpadded_written = num_sticks_read;
        uint32_t start_id = id_per_dim[0] + start_offset + width_offset;
        int max_num_tiles_this_core = 0;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_tiles_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_tiles_per_dim[j]) {
                // Handle edge case where last dimension is completely written
                id_per_dim[j] = num_input_tiles_per_dim[j];
            }
            unpadded_written = unpadded_written / num_input_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_tiles_per_dim[j - 1];
            size_till_end[j] = num_input_tiles_per_dim[j] - id_per_dim[j] - ((j == 1) ? 0 : 1);
            max_num_tiles_this_core += size_till_end[j] * accumulated_input_total_tiles_per_dim[j - 1];
        }
        WriterKernelArgs writer_kernel_args = common_writer_kernel_args;

        uint32_t num_tiles_this_core = std::min<uint32_t>(
            num_tiles_nhw_per_core * num_tiles_per_channel, std::max<int>(max_num_tiles_this_core, 0));

        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {}, Channel Tiles : {}, Max Tiles: {}, Num Tiles: "
            "{} for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            channels_tiles_this_core,
            max_num_tiles_this_core,
            num_tiles_this_core,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset] = 1;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());
        writer_kernel_args.push_back(num_tiles_per_channel - channels_tiles_this_core);

        ReaderKernelArgs reader_kernel_args = {num_tiles_this_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}
}  // namespace

ProgramDescriptor SliceWriteTiledShardedInputProgramFactory::create_descriptor(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& output_tensor_end = operation_attributes.slice_end;

    ProgramDescriptor desc;
    const auto& input_padded_shape = input.padded_shape();

    auto input_shape = input.logical_shape();
    auto output_shape = output.logical_shape();
    auto actual_input_shape = input_shape;
    for (int index = 0; index < input_shape.rank(); index++) {
        actual_input_shape[index] = output_tensor_end[index] - output_tensor_start[index];
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    log_debug(tt::LogOp, "Slice Write Input Shape : {} ,Actual Input Shape: {}", input_shape, input_shape);
    TT_FATAL(input.dtype() == output.dtype(), "Input & output should have the same dtype");
    TT_FATAL(output_tensor_start[-1] == 0, "Slice write expects output start for the last dimension to be 0");
    TT_FATAL(
        output_tensor_start[-2] % TILE_HEIGHT == 0,
        "Slice write expects output start for the second last dimension to be a multiple of tile height");

    TT_FATAL(
        input_padded_shape[-2] % TILE_HEIGHT == 0,
        "Slice write expects input shape for the second last dimension to be a multiple of tile height");

    TT_FATAL(
        input.layout() == Layout::TILE,
        "Slice write expects input tensor to be in TILE layout, got {}",
        input.layout());
    TT_FATAL(
        output.layout() == Layout::TILE,
        "Slice write expects output tensor to be in TILE layout, got {}",
        output.layout());

    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");

    auto shard_spec = input.shard_spec().value();
    auto input_cores = shard_spec.grid;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    log_debug(tt::LogOp, "Input cores = {}", input_cores);
    log_debug(tt::LogOp, "Input shard spec = {}", shard_spec);

    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Slice write needs tiled inputs, where the shard height {} is a multiple of tile height {}",
        shard_spec.shape[0],
        TILE_HEIGHT);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    auto* input_buffer = input.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer should be allocated on device");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_tiles_channel_per_core = shard_spec.shape[1] / TILE_HEIGHT;

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input);

    TT_FATAL(
        input_cb_data_format == output_cb_data_format,
        "Input & output should have the same data format, {} , {}",
        input_cb_data_format,
        output_cb_data_format);

    desc.cbs.push_back(make_slice_write_tiled_cb(
        cb_input_index,
        input_cores,
        input_single_tile_size,
        num_tiles_height_per_core * num_tiles_channel_per_core,
        input_cb_data_format,
        input_buffer));

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, 0};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args_vec);
    KernelDescriptor::Defines writer_defines;
    if (num_tiles_channel_per_core * TILE_WIDTH * num_cores_channels > output_shape[-1]) {
        writer_defines.emplace_back("UNPAD_INPUT_WIDTH", "1");
    }
    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = input_cores;
    reader_kernel.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_writer_interleaved.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = input_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_time_args_vec);
    writer_kernel.defines = std::move(writer_defines);
    writer_kernel.config = WriterConfigDescriptor{};

    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    auto all_runtime_args = get_slice_write_runtime_args_tiled_sharded_input(
        input, output, output_tensor_start, output_tensor_end, iter_cores);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        reader_kernel.runtime_args.emplace_back(core, all_runtime_args[i].first);
        emplace_slice_write_tiled_writer_args(writer_kernel, core, all_runtime_args[i].second, dst_buffer);
        i++;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

}  // namespace ttnn::experimental::prim
