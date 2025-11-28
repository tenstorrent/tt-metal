// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_program_factory.hpp"

#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::program {

namespace {

void set_runtime_args_hc_tiled(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2,
    bool is_create) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 Wt,
                 H,
                 Ct,
                 HW_bytes,
                 CHW_bytes,
                 num_tiles_read,
                 num_tiles_per_core,
                 num_tiles_read / CtHWt * CHW_bytes,
                 h,
                 h / TILE_HEIGHT * Wt,
                 ct,
                 ct * TILE_HEIGHT * HW_bytes,
                 num_tiles_read % Wt});

            SetRuntimeArgs(
                program, writer_kernel_id, core, {output_buffer->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = Wt;
            reader_args[2] = H;
            reader_args[3] = Ct;
            reader_args[4] = HW_bytes;
            reader_args[5] = CHW_bytes;
            reader_args[6] = num_tiles_read;
            reader_args[7] = num_tiles_per_core;
            reader_args[8] = num_tiles_read / CtHWt * CHW_bytes;
            reader_args[9] = h;
            reader_args[10] = h / TILE_HEIGHT * Wt;
            reader_args[11] = ct;
            reader_args[12] = ct * TILE_HEIGHT * HW_bytes;
            reader_args[13] = num_tiles_read % Wt;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        num_tiles_read += num_tiles_per_core;
    }
}

void set_runtime_args_hc_rm(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_2,
    bool is_create) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;

        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 num_sticks_per_core_read,
                 num_read_per_barrier,
                 curr_sticks_read,
                 curr_c,
                 curr_h,
                 curr_n});

            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_buffer->address(), num_sticks_per_core_read, num_read_per_barrier, curr_sticks_write});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = num_sticks_per_core_read;
            reader_args[2] = num_read_per_barrier;
            reader_args[3] = curr_sticks_read;
            reader_args[4] = curr_c;
            reader_args[5] = curr_h;
            reader_args[6] = curr_n;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_sticks_per_core_read;
            writer_args[2] = num_read_per_barrier;
            writer_args[3] = curr_sticks_write;
        }

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }
}

void set_runtime_args_hc_tiled_interleaved(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    bool is_create) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] / tile_shape[0]);

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
    for (const auto& core : cores) {
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

TransposeHCProgramFactory::cached_program_t TransposeHCProgramFactory::create(
    const transpose::operation_attributes_t& operation_attributes,
    const transpose::tensor_args_t& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;

    bool is_tiled_interleaved = input_tensor.layout() == Layout::TILE && !input_tensor.is_sharded();

    if (is_tiled_interleaved) {
        // Tiled interleaved path
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

        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * single_tile_size, {{src0_cb_index, cb_data_format}})
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
            program, reader_kernel_id, writer_kernel_id, input_tensor, output_tensor, true);

        return {
            std::move(program),
            {.reader_kernel_id = reader_kernel_id,
             .writer_kernel_id = writer_kernel_id,
             .core_group_1 = {},
             .core_group_2 = {},
             .num_cores_total = 0,
             .num_cores_y = 0,
             .num_tiles_per_core_group_1 = 0,
             .num_tiles_per_core_group_2 = 0,
             .is_row_major = false,
             .is_tiled_interleaved = true}};
    }

    // Row-major or tiled non-interleaved path
    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    const auto& a_shape = input_tensor.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t NCH = N * C * H;
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    Program program = CreateProgram();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NCH : num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();

    uint32_t src0_cb_index = 0;
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    if (row_major) {
        auto num_sticks = num_tiles_per_core_group_1 > num_tiles_per_core_group_2 ? num_tiles_per_core_group_1
                                                                                  : num_tiles_per_core_group_2;
        auto stick_size = W * input_tensor.element_size();
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_sticks * stick_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, stick_size);
        CreateCircularBuffer(program, total_cores, cb_src0_config);
    } else {
        uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, total_cores, cb_src0_config);

        if (misaligned) {
            uint32_t src1_cb_index = 1;
            CircularBufferConfig cb_src1_config = CircularBufferConfig(alignment, {{src1_cb_index, cb_data_format}})
                                                      .set_page_size(src1_cb_index, alignment);
            CreateCircularBuffer(program, total_cores, cb_src1_config);
        }
    }

    Buffer* src0_buffer = input_tensor.buffer();
    std::vector<uint32_t> reader_compile_time_args;
    if (row_major) {
        reader_compile_time_args.push_back(N);
        reader_compile_time_args.push_back(H);
        reader_compile_time_args.push_back(C);
        reader_compile_time_args.push_back(W * input_tensor.element_size());
        reader_compile_time_args.push_back(W * input_tensor.element_size());
    } else {
        reader_compile_time_args.push_back(sub_tile_line_bytes);
        reader_compile_time_args.push_back(cb_data_format == tt::DataFormat::Float32 ? 1 : 0);
        reader_compile_time_args.push_back(alignment);
    }
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    if (row_major) {
        writer_compile_time_args.push_back(W * input_tensor.element_size());
        writer_compile_time_args.push_back(W * input_tensor.element_size());
    }
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_hc_interleaved_partitioned.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        row_major
            ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
              "writer_unary_transpose_hc_interleaved_start_id_rm.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    if (row_major) {
        set_runtime_args_hc_rm(
            program,
            reader_kernel_id,
            writer_kernel_id,
            input_tensor,
            output_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2,
            true);
    } else {
        set_runtime_args_hc_tiled(
            program,
            reader_kernel_id,
            writer_kernel_id,
            input_tensor,
            output_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2,
            true);
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores_total = num_cores_total,
         .num_cores_y = num_cores_y,
         .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
         .num_tiles_per_core_group_2 = num_tiles_per_core_group_2,
         .is_row_major = row_major,
         .is_tiled_interleaved = false}};
}

void TransposeHCProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const transpose::operation_attributes_t& operation_attributes,
    const transpose::tensor_args_t& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    if (shared_variables.is_tiled_interleaved) {
        set_runtime_args_hc_tiled_interleaved(
            program,
            shared_variables.reader_kernel_id,
            shared_variables.writer_kernel_id,
            tensor_args.input,
            tensor_return_value,
            false);
    } else if (shared_variables.is_row_major) {
        set_runtime_args_hc_rm(
            program,
            shared_variables.reader_kernel_id,
            shared_variables.writer_kernel_id,
            tensor_args.input,
            tensor_return_value,
            shared_variables.num_cores_total,
            shared_variables.num_cores_y,
            shared_variables.core_group_1,
            shared_variables.num_tiles_per_core_group_1,
            shared_variables.core_group_2,
            shared_variables.num_tiles_per_core_group_2,
            false);
    } else {
        set_runtime_args_hc_tiled(
            program,
            shared_variables.reader_kernel_id,
            shared_variables.writer_kernel_id,
            tensor_args.input,
            tensor_return_value,
            shared_variables.num_cores_total,
            shared_variables.num_cores_y,
            shared_variables.core_group_1,
            shared_variables.num_tiles_per_core_group_1,
            shared_variables.core_group_2,
            shared_variables.num_tiles_per_core_group_2,
            false);
    }
}

}  // namespace ttnn::operations::data_movement::program
