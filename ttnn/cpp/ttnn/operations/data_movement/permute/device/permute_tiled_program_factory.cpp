// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t tile_volume(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    return tile_shape[0] * tile_shape[1];
}

uint32_t num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.padded_shape();
    auto tile_vol = tile_volume(input_tensor);
    return shape.volume() / tile_vol;
}

uint32_t tile_size(const ttnn::Tensor& input_tensor) {
    auto dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    return tt::tile_size(dataformat);
}

ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
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
    auto res = ttnn::Shape(tiled_shape);
    return res;
}

ttnn::SmallVector<uint32_t> get_strides(const ttnn::Shape& shape) {
    ttnn::SmallVector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    for (int i = shape.rank() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Function to compute the inverse of a permutation
ttnn::SmallVector<uint32_t> get_inverse_permutation(const ttnn::SmallVector<uint32_t>& perm) {
    // Get the size of the permutation
    size_t n = perm.size();

    // Create a vector for the inverse permutation
    ttnn::SmallVector<uint32_t> inverse_permutation(n);

    // Validate the input permutation
    ttnn::SmallVector<bool> seen(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (perm[i] >= n || seen[perm[i]]) {
            TT_FATAL(false, "Invalid permutation: duplicate or out of range value");
        }
        seen[perm[i]] = true;
        inverse_permutation[perm[i]] = static_cast<uint32_t>(i);
    }

    return inverse_permutation;
}

uint32_t get_buffer_alignment(const ttnn::Tensor& tensor) {
    return (
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                                         : tt::tt_metal::hal::get_l1_alignment());
}

}  // namespace detail

PermuteDeviceOperation::MultiCoreTileInvariant::cached_program_t PermuteDeviceOperation::MultiCoreTileInvariant::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::tile_size(input_tensor);

    uint32_t num_tiles = detail::num_tiles(tensor_return_value);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_pages_to_read = 2;

    uint32_t rank = operation_attributes.dims.size();
    bool swap_hw = operation_attributes.dims[rank - 2] == rank - 1 && operation_attributes.dims[rank - 1] == rank - 2;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;
    if (swap_hw) {
        uint32_t src1_cb_index = tt::CBIndex::c_16;
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(
                num_input_pages_to_read * input_page_size, {{src1_cb_index, cb_data_format}})
                .set_page_size(src1_cb_index, input_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
        output_cb_index = src1_cb_index;
    }

    std::vector<uint32_t> reader_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"rank", rank},
        {"page_size", input_page_size},
        {"num_tiles", num_tiles},
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_tiled_invariant.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t compute_kernel_id = 0;
    if (swap_hw) {
        std::vector<uint32_t> compute_kernel_args = {};
        bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args,
            });
    }

    // think of tensor as its tiled shape rather than its logical shape
    auto output_tiled_shape = detail::get_tiled_shape(tensor_return_value);
    auto input_tiled_shape = detail::get_tiled_shape(input_tensor);
    auto output_shape_view = output_tiled_shape.view();

    // read is less expensive than write, so read in order of output tensor, get relevant pre-permutation input tiles,
    // and then write it out to determine index in input tensor we need the input strides
    auto input_tile_strides = detail::get_strides(input_tiled_shape);

    // we also need the inverse permutation to map back to input tensor
    auto inv_perm = detail::get_inverse_permutation(operation_attributes.dims);

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};

    reader_runtime_args.insert(reader_runtime_args.end(), output_shape_view.begin(), output_shape_view.end());
    reader_runtime_args.insert(reader_runtime_args.end(), inv_perm.begin(), inv_perm.end());
    reader_runtime_args.insert(reader_runtime_args.end(), input_tile_strides.begin(), input_tile_strides.end());

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0};

    std::vector<uint32_t> compute_runtime_args = {0};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_tile = 0;
    uint32_t num_tiles_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }
        uint32_t end_tile = start_tile + num_tiles_per_core;
        reader_runtime_args[1] = start_tile;
        reader_runtime_args[2] = end_tile;

        writer_runtime_args[1] = num_tiles_per_core;     // for some reason num_tiles comes first in writer unary
        writer_runtime_args[2] = start_tile;             // start tile is second in writer unary

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        if (swap_hw) {
            compute_runtime_args[0] = num_tiles_per_core;  // number of tiles transposed
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        }
        start_tile = end_tile;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .core_range = all_cores}};
}

void PermuteDeviceOperation::MultiCoreTileInvariant::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();

        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

PermuteDeviceOperation::MultiCoreTileRowInvariant::cached_program_t
PermuteDeviceOperation::MultiCoreTileRowInvariant::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const float pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    auto dims = operation_attributes.dims;

    auto input_shape = input_tensor.logical_shape();

    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    auto padded_output_shape = output_tensor.padded_shape();
    uint32_t rank = operation_attributes.dims.size();
    bool swap_hw = dims[rank - 1] == rank - 2;

    if (swap_hw) {
        for (uint32_t i = 0; i < rank; i++) {
            if (dims[i] == rank - 2) {
                dims[i] = rank - 1;
            } else if (dims[i] == rank - 1) {
                dims[i] = rank - 2;
            }
        }
        std::swap(tile_shape[0], tile_shape[1]);
        std::swap(input_shape[rank - 2], input_shape[rank - 1]);
        std::swap(padded_output_shape[rank - 2], padded_output_shape[rank - 1]);
        std::swap(face_shape[0], face_shape[1]);
    }

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::tile_size(input_tensor);

    uint32_t num_tiles = detail::num_tiles(input_tensor);
    uint32_t num_output_tiles = detail::num_tiles(tensor_return_value);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t padding_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = src0_cb_index;

    uint32_t num_input_pages_to_read = 2;

    uint32_t padded_num_tensor_tiles =
        num_output_tiles / (padded_output_shape[rank - 2] / tile_shape[0]);  // only last row of Xt should have padding

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_H = input_shape[dims[rank - 2]];
    uint32_t element_size = input_tensor.element_size();

    bool needs_padding = (output_H % tile_shape[1] != 0);
    if (needs_padding) {
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(face_shape[1] * element_size, {{padding_cb_index, cb_data_format}})
                .set_page_size(padding_cb_index, face_shape[1] * element_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    }
    if (swap_hw) {
        uint32_t src2_cb_index = tt::CBIndex::c_16;
        tt::tt_metal::CircularBufferConfig cb_src2_config =
            tt::tt_metal::CircularBufferConfig(
                num_input_pages_to_read * input_page_size, {{src2_cb_index, cb_data_format}})
                .set_page_size(src2_cb_index, input_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
        output_cb_index = src2_cb_index;
    }
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (output_H % tile_shape[1] != 0) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = face_shape[1] / num_packed_values;
        switch (input_tensor.dtype()) {
            case DataType::INT32:
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    uint32_t h_in_dest = 0;
    for (uint32_t i = 0; i < rank; i++) {
        if (dims[i] == rank - 2) {
            h_in_dest = i;
            break;
        }
    }

    uint32_t accumulated_outer_dims = 1;
    for (uint32_t i = 0; i < rank - 2; i++) {
        accumulated_outer_dims *= input_shape[i];
    }

    std::vector<uint32_t> reader_compile_time_args = {};

    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"num_writes", num_writes},
        {"padding_val_packed", padding_val_packed},
        {"needs_padding", needs_padding},
        {"swap_hw", swap_hw},
        {"H", input_shape[rank - 1]},
        {"W", input_shape[rank - 2]},
        {"accumulated_outer_dims", accumulated_outer_dims},
        {"tile_height", tile_shape[1]},
        {"tile_width", tile_shape[0]},
    };

    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    uint32_t compute_kernel_id = 0;
    if (swap_hw) {
        std::vector<uint32_t> compute_kernel_args = {};
        bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args,
            });
    }

    std::vector<uint32_t> writer_compile_time_args = {};

    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"element_size", element_size},
        {"output_cb_index", output_cb_index},
        {"output_H", output_H},
        {"H", input_shape[rank - 2]},
        {"W", input_shape[rank - 1]},
        {"tile_height", tile_shape[0]},
        {"tile_width", tile_shape[1]},
        {"face_height", face_shape[0]},
        {"face_width", face_shape[1]},
        {"needs_padding", needs_padding},
        {"rank", rank},
        {"h_in_dest", h_in_dest},
    };

    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_tiled_row_invariant.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    auto input_shape_view = input_shape.view();

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0, 0, 0};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(writer_runtime_args.end(), dims.begin(), dims.end());

    std::vector<uint32_t> compute_runtime_args = {0};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_tile = 0;
    uint32_t num_tiles_per_core = 0;
    uint32_t start_tile_padding = 0;
    uint32_t num_tiles_per_core_padding = 0;
    uint32_t end_tile_padding = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }
        if (needs_padding) {
            if (padded_core_group_1.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_1;
            } else if (padded_core_group_2.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_2;
            } else {
                // no-op
                num_tiles_per_core_padding = 0;
            }
        }
        uint32_t end_tile = start_tile + num_tiles_per_core;
        reader_runtime_args[1] = num_tiles_per_core;
        reader_runtime_args[2] = start_tile;

        writer_runtime_args[1] = start_tile;  // for some reason num_tiles comes first in writer unary
        writer_runtime_args[2] = end_tile;    // start tile is second in writer unary
        if (needs_padding) {
            end_tile_padding = start_tile_padding + num_tiles_per_core_padding;
            writer_runtime_args[3] = start_tile_padding;
            writer_runtime_args[4] = end_tile_padding;
        }
        if (swap_hw) {
            compute_runtime_args[0] = num_tiles_per_core;  // number of tiles transposed
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

        start_tile = end_tile;
        start_tile_padding = end_tile_padding;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .core_range = all_cores}};
}

void PermuteDeviceOperation::MultiCoreTileRowInvariant::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

PermuteDeviceOperation::MultiCoreTiledGeneric::cached_program_t PermuteDeviceOperation::MultiCoreTiledGeneric::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * The algorithm is as follows:
     * 1. Read in blocks of data along the X and W dimensions (XW blocks, W is contiguous)
     *  a. TILE_HEIGHT rows along X with TILE_WIDTH elements across W
     * 2. Tilize, transpose, and untilize the data into a WX block
     * 3. Write out all the data in WX block to its correct position in the permuted output tensor buffer
     *  a. We write out on face/subtile line at a time
     *  a. X is the output width dimension, but it's tiled so we can only write out face/subtile line at a time
     * 4. Repeat until all XW blocks are processed
     * 5. If X is not a multiple of TILE_WIDTH, we pad the last face/subtile line with the pad value
     * 6. If Y is not a multiple of TILE_HEIGHT, we pad the last set of tiles on the Y dimension with the pad value
     *
     */

    using namespace tt;
    using namespace tt::tt_metal;
    const float pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto& dims = operation_attributes.dims;
    uint32_t rank = dims.size();
    auto& output_tensor = tensor_return_value;
    const auto& output_shape = output_tensor.logical_shape();
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};
    uint32_t logical_volume = input_shape.volume();
    uint32_t num_rows = logical_volume / input_shape[rank - 1];
    uint32_t y_dim_index_in_input = dims[rank - 2];

    uint32_t x_dim_index_in_input = dims[rank - 1];
    uint32_t x = input_shape[x_dim_index_in_input];
    uint32_t y = input_shape[y_dim_index_in_input];
    uint32_t w = input_shape[rank - 1];

    // X is the new width so we need to pad it to the target tile_shape[1]
    uint32_t x_block_size = tile_shape[1];
    uint32_t w_block_size = tile_shape[1];

    uint32_t element_size = input_tensor.element_size();
    uint32_t X_p = x_block_size * ((x + x_block_size - 1) / x_block_size);
    uint32_t W_p = tile_shape[1] * ((w + tile_shape[1] - 1) / tile_shape[1]);
    uint32_t H_p = tile_shape[0] * ((input_shape[rank - 2] + tile_shape[0] - 1) / tile_shape[0]);
    uint32_t H_t = H_p / tile_shape[0];
    uint32_t W_t = W_p / tile_shape[1];

    uint32_t subtile_line_bytes = face_shape[1] * element_size;
    uint32_t read_alignment = detail::get_buffer_alignment(input_tensor);
    uint32_t misalignment = read_alignment > subtile_line_bytes ? read_alignment - subtile_line_bytes : 0;

    uint32_t permuted_w_dim = 0;  // Will hold the position of w_dim in the permuted array
    for (uint32_t i = 0; i < rank; ++i) {
        if (dims[i] == rank - 1) {
            permuted_w_dim = i;
            break;
        }
    }

    uint32_t w_blocks = W_p / w_block_size;
    uint32_t x_blocks = X_p / x_block_size;

    uint32_t num_faces_w = tile_shape[1] / face_shape[1];

    uint32_t padded_xw_volume = X_p * W_p;
    for (uint32_t i = 0; i < rank - 1; i++) {
        if (i == x_dim_index_in_input) {
            continue;
        }
        padded_xw_volume *= input_shape[i];
    }

    uint32_t xw_blocks = padded_xw_volume / (tile_shape[0] * tile_shape[1]);

    bool needs_x_padding = (x % tile_shape[1] != 0);
    bool needs_y_padding =
        (y % tile_shape[0] != 0);  // if H is not moved, we could just keep existing implicit padding instead of
                                   // re-padding, but it complicates logic, may be worth investigating in the future
    bool needs_padding = needs_x_padding or needs_y_padding;

    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;

    uint32_t padding_cb_index = tt::CBIndex::c_3;

    if (needs_padding) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = face_shape[1] / num_packed_values;

        switch (input_tensor.dtype()) {
            case DataType::INT32:
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    // Faces with real data in the final tile along the width dimension, divided up
    uint32_t final_tile_real_w = w % tile_shape[1];
    uint32_t final_tile_real_faces_w =
        w % tile_shape[1] == 0 ? num_faces_w : ((final_tile_real_w + face_shape[1] - 1) / face_shape[1]);

    uint32_t final_tile_real_x = x % tile_shape[1];
    uint32_t final_tile_real_faces_x =
        needs_x_padding
            ? num_faces_w
            : (final_tile_real_x == 0 ? num_faces_w : ((final_tile_real_x + face_shape[1] - 1) / face_shape[1]));

    uint32_t num_output_tiles = detail::num_tiles(tensor_return_value);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src2_cb_index = tt::CBIndex::c_2;

    uint32_t num_input_pages_to_read = 2;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    // CoreCoord compute_with_storage_grid_size = {1u, 1u};
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, xw_blocks);

    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[rank - 2] /
                                                           tile_shape[0]);  // only last row of Xt should have padding
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::tile_size(tensor_return_value) + misalignment;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt::tt_metal::CircularBufferConfig cb_src2_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);

    if (needs_y_padding) {
        tt::tt_metal::CircularBufferConfig cb_padding_cfg =
            tt::tt_metal::CircularBufferConfig(face_shape[1] * element_size, {{padding_cb_index, cb_data_format}})
                .set_page_size(padding_cb_index, face_shape[1] * element_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_padding_cfg);
    }

    uint32_t non_x_rows = num_rows / x;

    std::vector<uint32_t> reader_compile_time_args = {};

    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"rank", rank},
        {"page_size", input_page_size},
        {"element_size", element_size},
        {"tile_height", tile_shape[0]},
        {"tile_width", tile_shape[1]},
        {"face_height", face_shape[0]},
        {"face_width", face_shape[1]},
        {"x_dim_index_in_input", x_dim_index_in_input},
        {"X", x},
        {"W", w},
        {"H", input_shape[rank - 2]},
        {"X_p", X_p},
        {"W_p", W_p},
        {"H_p", H_p},
        {"H_t", H_t},
        {"W_t", W_t},
        {"final_tile_real_w", final_tile_real_w},
        {"final_tile_real_faces_w", final_tile_real_faces_w},
        {"xw_blocks", xw_blocks},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"num_writes", num_writes},
        {"padding_val_packed", padding_val_packed},
        {"needs_x_padding", needs_x_padding},
        {"needs_y_padding", needs_y_padding},
        {"rows_per_x", non_x_rows},
        {"misalignment", misalignment},
        {"read_alignment", read_alignment},
    };

    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_tiled_generic.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};

    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_tiled.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
        });

    std::vector<uint32_t> writer_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"rank", rank},
        {"page_size", input_page_size},
        {"element_size", element_size},
        {"tile_height", tile_shape[0]},
        {"tile_width", tile_shape[1]},
        {"face_height", face_shape[0]},
        {"face_width", face_shape[1]},
        {"x_dim_index_in_input", x_dim_index_in_input},
        {"X", x},
        {"W", w},
        {"Y", output_shape[rank - 2]},
        {"X_p", X_p},
        {"W_p", W_p},
        {"rows_per_x", non_x_rows},
        {"H_t", H_t},
        {"W_t", W_t},
        {"final_tile_real_x", final_tile_real_x},
        {"final_tile_real_faces_x", final_tile_real_faces_x},
        {"xw_blocks", xw_blocks},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"needs_y_padding", needs_y_padding},
        {"permuted_w_dim", permuted_w_dim},
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_tiled_generic.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    auto input_shape_view = input_shape.view();

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};
    reader_runtime_args.insert(reader_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    reader_runtime_args.insert(reader_runtime_args.end(), dims.begin(), dims.end());

    std::vector<uint32_t> compute_runtime_args = {0, 0};

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0, 0, 0};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(writer_runtime_args.end(), dims.begin(), dims.end());

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    uint32_t num_tiles_per_core_padding = 0;
    uint32_t start_tile_padding = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }
        if (needs_padding) {
            if (padded_core_group_1.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_1;
            } else if (padded_core_group_2.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_2;
            } else {
                // no-op
                num_tiles_per_core_padding = 0;
            }
        }

        uint32_t end_block = start_block + num_blocks_per_core;
        reader_runtime_args[1] = start_block;
        reader_runtime_args[2] = end_block;

        compute_runtime_args[0] = start_block;
        compute_runtime_args[1] = end_block;

        writer_runtime_args[1] = start_block;
        writer_runtime_args[2] = end_block;

        if (needs_padding) {
            uint32_t end_tile_padding = start_tile_padding + num_tiles_per_core_padding;
            writer_runtime_args[3] = start_tile_padding;
            writer_runtime_args[4] = end_tile_padding;
            start_tile_padding = end_tile_padding;
        }

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        start_block = end_block;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .core_range = all_cores}};
}

void PermuteDeviceOperation::MultiCoreTiledGeneric::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement
