// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"

#include "ttnn/operations/math.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::transpose::program {

namespace {

void set_runtime_args_hc_tiled_interleaved(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    bool is_create,
    const CoreRange& total_cores) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] /
                                                           tile_shape[0]);  // only last row of Ct should have padding

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;
    auto cores = corerange_to_cores(all_cores, std::nullopt);

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    // Need to set runtime args for all cores, not just the ones doing work.
    for (const auto& core : total_cores) {
        uint32_t num_tiles_per_core;
        uint32_t padded_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        if (padded_core_group_1.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_1;
        } else if (padded_core_group_2.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_2;
        } else {
            padded_tiles_per_core = 0;
        }

        uint32_t end_idx = start_idx + num_tiles_per_core;
        uint32_t padded_end_idx = padded_start_idx + padded_tiles_per_core;
        if (is_create) {
            SetRuntimeArgs(program, reader_kernel_id, core, {input_buffer->address(), num_tiles_per_core, start_idx});

            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_buffer->address(), start_idx, end_idx, padded_start_idx, padded_end_idx});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            writer_args[0] = output_buffer->address();
        }
        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }
}

}  // namespace

TransposeHCTiledInterleavedProgramFactory::cached_program_t TransposeHCTiledInterleavedProgramFactory::create(
    const transpose::TransposeParams& operation_attributes,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    Program program = Program();
    auto tile = input_tensor.tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    uint32_t C = input_tensor.logical_shape()[1];
    bool needs_padding = (C % tile_shape[1] != 0) && pad_value.has_value();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t padding_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb_src0_config = CircularBufferConfig(2 * single_tile_size, {{src0_cb_index, cb_data_format}})
                                              .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, total_cores, cb_src0_config);

    if (needs_padding) {
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(face_shape[1] * input_tensor.element_size(), {{padding_cb_index, cb_data_format}})
                .set_page_size(padding_cb_index, face_shape[1] * input_tensor.element_size());
        CreateCircularBuffer(program, total_cores, cb_src1_config);
    }

    Buffer* src_buffer = input_tensor.buffer();
    uint32_t element_size = input_tensor.element_size();
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];

    if (pad_value.has_value() && C % tile_shape[1] != 0) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = face_shape[1] / num_packed_values;
        if (input_tensor.dtype() == DataType::BFLOAT16) {
            padding_val_packed =
                pack_two_bfloat16_into_uint32({bfloat16(pad_value.value()), bfloat16(pad_value.value())});
        } else if (num_packed_values == 2) {
            padding_val_packed =
                static_cast<uint32_t>(pad_value.value()) | (static_cast<uint32_t>(pad_value.value()) << 16);
        } else {
            padding_val_packed = std::bit_cast<uint32_t>(pad_value.value());
        }
    }

    std::vector<uint32_t> reader_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"num_writes", num_writes},
        {"padding_val_packed", padding_val_packed},
        {"needs_padding", needs_padding},
        {"swap_hw", 0u},
        {"H", 1u},
        {"W", 1u},
        {"accumulated_outer_dims", 1u},
        {"tile_height", 1u},
        {"tile_width", 1u},
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    Buffer* dst_buffer = output_tensor.buffer();
    std::vector<uint32_t> writer_compile_time_args = {
        element_size,
        tt::CBIndex::c_0,
        C,
        H,
        W,
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1],
        static_cast<uint32_t>(needs_padding)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    set_runtime_args_hc_tiled_interleaved(
        program, reader_kernel_id, writer_kernel_id, input_tensor, output_tensor, true, total_cores);

    return {std::move(program), {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id}};
}

void TransposeHCTiledInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const transpose::TransposeParams& /*operation_attributes*/,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto compute_with_storage_grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    set_runtime_args_hc_tiled_interleaved(
        program,
        shared_variables.reader_kernel_id,
        shared_variables.writer_kernel_id,
        tensor_args.input,
        tensor_return_value,
        false,
        total_cores);
}

}  // namespace ttnn::operations::data_movement::transpose::program
