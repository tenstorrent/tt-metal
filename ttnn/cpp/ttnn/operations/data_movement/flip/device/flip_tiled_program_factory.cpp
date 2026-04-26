// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

static uint32_t get_tile_num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    return input_tensor.padded_shape().volume() / (tile_shape[0] * tile_shape[1]);
}

static ttnn::SmallVector<uint32_t> get_tile_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); i++) {
        if (i == shape.rank() - 1) {
            tiled_shape.push_back(shape[i] / tile_shape[1]);
        } else if (i == shape.rank() - 2) {
            tiled_shape.push_back(shape[i] / tile_shape[0]);
        } else {
            tiled_shape.push_back(shape[i]);
        }
    }
    return tiled_shape;
}

static ttnn::SmallVector<uint32_t> get_tile_strides(const ttnn::SmallVector<uint32_t>& tiled_shape) {
    ttnn::SmallVector<uint32_t> strides(tiled_shape.size());
    strides[tiled_shape.size() - 1] = 1;
    for (int i = (int)tiled_shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tiled_shape[i + 1];
    }
    return strides;
}

}  // namespace detail

FlipDeviceOperation::MultiCoreTiled::cached_program_t FlipDeviceOperation::MultiCoreTiled::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    Program program{};

    uint32_t rank = input_tensor.logical_shape().rank();
    uint32_t element_size = input_tensor.element_size();
    uint32_t num_tiles = detail::get_tile_num_tiles(input_tensor);

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    auto tiled_shape = detail::get_tile_tiled_shape(input_tensor);
    auto tile_strides = detail::get_tile_strides(tiled_shape);

    std::vector<uint32_t> dims_to_flip(rank, 0);
    for (const auto& d : operation_attributes.dims) {
        dims_to_flip[d] = 1;
    }

    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_grid, num_tiles);

    DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t tile_size_bytes = input_tensor.tensor_spec().tile().get_tile_size(data_format);

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * tile_size_bytes, {{CBIndex::c_0, data_format}})
            .set_page_size(CBIndex::c_0, tile_size_bytes));

    std::vector<uint32_t> reader_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"rank", rank},
        {"element_size", element_size},
        {"tile_height", tile_shape[0]},
        {"tile_width", tile_shape[1]},
        {"face_height", face_shape[0]},
        {"face_width", face_shape[1]},
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/reader_interleaved_tiled.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/writer_interleaved_tiled.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};
    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0};

    reader_runtime_args.insert(reader_runtime_args.end(), tiled_shape.begin(), tiled_shape.end());
    reader_runtime_args.insert(reader_runtime_args.end(), tile_strides.begin(), tile_strides.end());
    reader_runtime_args.insert(reader_runtime_args.end(), dims_to_flip.begin(), dims_to_flip.end());

    uint32_t start_tile = 0;
    uint32_t end_tile = 0;
    auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    for (const auto& [ranges, tiles_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                end_tile += tiles_per_core;

                reader_runtime_args[1] = start_tile;
                reader_runtime_args[2] = end_tile;
                SetRuntimeArgs(program, reader_id, core, reader_runtime_args);

                writer_runtime_args[1] = start_tile;
                writer_runtime_args[2] = end_tile;
                SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

                start_tile += tiles_per_core;
            }
        }
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_id, .unary_writer_kernel_id = writer_id, .core_range = all_cores},
    };
}

void FlipDeviceOperation::MultiCoreTiled::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& all_cores = cached_program.shared_variables.core_range;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    for (const auto& core : corerange_to_cores(all_cores, std::nullopt)) {
        tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core)[0] = src_buffer->address();
        tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core)[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement
