// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

static uint32_t get_tile_volume(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    return tile_shape[0] * tile_shape[1];
}

static uint32_t get_num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.padded_shape();
    auto tile_vol = get_tile_volume(input_tensor);
    return shape.volume() / tile_vol;
}

static ttnn::SmallVector<uint32_t> get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    return tiled_shape;
}

static ttnn::SmallVector<uint32_t> get_tile_strides(const ttnn::SmallVector<uint32_t>& shape) {
    ttnn::SmallVector<uint32_t> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
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

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;

    uint32_t rank = input_tensor.logical_shape().rank();
    uint32_t element_size = input_tensor.element_size();
    uint32_t num_tiles = detail::get_num_tiles(input_tensor);
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    ttnn::SmallVector<uint32_t> input_tiled_shape = detail::get_tiled_shape(input_tensor);
    ttnn::SmallVector<uint32_t> input_tile_strides = detail::get_tile_strides(input_tiled_shape);

    auto dims = operation_attributes.dims;
    std::vector<uint32_t> dims_to_flip(rank, 0);
    for (const auto& d : dims) {
        dims_to_flip[d] = 1;
    }

    // ------------------------------------------------------------------------
    // 1) Split work to all available cores
    // ------------------------------------------------------------------------
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_grid, num_tiles);

    // ------------------------------------------------------------------------
    // 2) Create circular buffer
    // ------------------------------------------------------------------------
    DataFormat input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_page_size =
        input_tensor.tensor_spec().tile().get_tile_size(datatype_to_dataformat_converter(input_tensor.dtype()));

    uint32_t num_input_pages_to_read = 2;  // double buffering
    uint32_t cb_size = num_input_pages_to_read * input_page_size;

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(cb_size, {{CBIndex::c_0, input_data_format}})
            .set_page_size(CBIndex::c_0, input_page_size));

    // ------------------------------------------------------------------------
    // 3) Set compile time arguments for kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"src_is_dram", (uint32_t)src_is_dram},
        {"rank", rank},
        {"element_size", element_size},
        {"tile_height", tile_shape[0]},
        {"tile_width", tile_shape[1]},
        {"face_height", face_shape[0]},
        {"face_width", face_shape[1]},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {{"dst_is_dram", (uint32_t)dst_is_dram}};

    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // ------------------------------------------------------------------------
    // 4) Create kernels
    // ------------------------------------------------------------------------
    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "reader_interleaved_tiled.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "writer_interleaved_tiled.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    // ------------------------------------------------------------------------
    // 5) Set runtime arguments for kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_runtime_args = {input_tensor.buffer()->address(), 0, 0};
    std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), 0, 0};

    reader_runtime_args.insert(reader_runtime_args.end(), input_tiled_shape.begin(), input_tiled_shape.end());
    reader_runtime_args.insert(reader_runtime_args.end(), input_tile_strides.begin(), input_tile_strides.end());
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
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::data_movement
