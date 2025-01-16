// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t tile_volume(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    return tile_shape[0] * tile_shape[1];
}

uint32_t num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.get_padded_shape();
    auto tile_vol = tile_volume(input_tensor);
    return shape.volume() / tile_vol;
}

uint32_t tile_size(const ttnn::Tensor& input_tensor) { return tile_volume(input_tensor) * input_tensor.element_size(); }

ttnn::SimpleShape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.get_padded_shape();
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
    auto res = ttnn::SimpleShape(tiled_shape);
    return res;
}

ttnn::SmallVector<uint32_t> get_strides(const ttnn::SimpleShape& shape) {
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

}  // namespace detail

PermuteDeviceOperation::MultiCoreTileInvariant::cached_program_t PermuteDeviceOperation::MultiCoreTileInvariant::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_page_size = detail::tile_size(input_tensor);

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_page_size = detail::tile_size(tensor_return_value);

    uint32_t num_tiles = detail::num_tiles(tensor_return_value);

    tt::tt_metal::IDevice* device = input_tensor.device();

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
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;
    if (swap_hw) {
        uint32_t src1_cb_index = tt::CBIndex::c_16;
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(
                num_input_pages_to_read * input_page_size, {{src1_cb_index, cb_data_format}})
                .set_page_size(src1_cb_index, input_page_size);
        auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
        output_cb_index = src1_cb_index;
    }

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, rank, input_page_size, num_tiles};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_tiled_invariant.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, (std::uint32_t)dst_is_dram};

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
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void PermuteDeviceOperation::MultiCoreTileInvariant::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
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
    const std::optional<float> pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.get_logical_shape();
    const auto& dims = operation_attributes.dims;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_page_size = detail::tile_size(input_tensor);

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_page_size = detail::tile_size(tensor_return_value);

    uint32_t num_tiles = detail::num_tiles(input_tensor);
    uint32_t num_output_tiles = detail::num_tiles(tensor_return_value);

    tt::tt_metal::IDevice* device = input_tensor.device();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t padding_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = src0_cb_index;

    uint32_t num_input_pages_to_read = 2;

    uint32_t rank = operation_attributes.dims.size();

    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();

    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.get_padded_shape()[rank - 2] /
                                                           tile_shape[0]);  // only last row of Xt should have padding

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    // CoreCoord compute_with_storage_grid_size = {1u, 1u};
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
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t output_H = input_shape[dims[rank - 2]];
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t element_size = input_tensor.element_size();

    const auto& face_shape = input_tensor.get_tensor_spec().tile().get_face_shape();

    bool needs_padding = (output_H % tile_shape[1] != 0) && pad_value.has_value();
    if (needs_padding) {
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(face_shape[1] * element_size, {{padding_cb_index, cb_data_format}})
                .set_page_size(padding_cb_index, face_shape[1] * element_size);
        auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    }
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (pad_value.has_value()) {
        if (output_H % tile_shape[1] != 0) {
            uint32_t num_packed_values = sizeof(uint32_t) / element_size;
            num_writes = face_shape[1] / num_packed_values;
            if (input_tensor.get_dtype() == DataType::BFLOAT16) {
                padding_val_packed =
                    pack_two_bfloat16_into_uint32({bfloat16(pad_value.value()), bfloat16(pad_value.value())});
            } else if (num_packed_values == 2) {
                padding_val_packed =
                    static_cast<uint32_t>(pad_value.value()) | (static_cast<uint32_t>(pad_value.value()) << 16);
            } else {
                padding_val_packed = std::bit_cast<uint32_t>(pad_value.value());
            }
        }
    }

    uint32_t h_in_dest = 0;
    for (uint32_t i = 0; i < rank; i++) {
        if (dims[i] == rank - 2) {
            h_in_dest = i;
            break;
        }
    }

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_is_dram, num_writes, padding_val_packed, (uint32_t)needs_padding};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)dst_is_dram,
        element_size,
        output_cb_index,
        output_H,
        input_shape[rank - 2],
        input_shape[rank - 1],
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1],
        (uint32_t)needs_padding,
        rank,
        h_in_dest};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_tiled_row_invariant.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto input_shape_view = input_shape.view();

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0, 0, 0};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(
        writer_runtime_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());

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

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

        start_tile = end_tile;
        start_tile_padding = end_tile_padding;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void PermuteDeviceOperation::MultiCoreTileRowInvariant::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
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
