// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>

#include "moreh_getitem_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

struct IndexInfo {
    bool is_defined;
    bool is_dram;
    uint32_t address;
    uint32_t unit_size;
};

namespace ttnn::operations::moreh::moreh_getitem {
MorehGetItemOperation::MorehGetItemTilizedFactory::cached_program_t
MorehGetItemOperation::MorehGetItemTilizedFactory::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::operations::primary;

    auto input = tensor_args.input;
    auto index_tensors = tensor_args.index_tensors;
    auto output = output_tensor;
    auto index_dims = operation_attributes.index_dims;
    auto memory_config = operation_attributes.memory_config;
    auto TILE_HEIGHT = constants::TILE_HEIGHT;
    auto TILE_WIDTH = constants::TILE_WIDTH;
    // auto core_range = operation_attributes.core_range;
    auto device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange allCores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    auto core_range = allCores;

    auto input_shape = input.get_shape();
    auto input_shape_without_padding = input_shape.value.without_padding();
    auto output_shape = output.get_shape();
    auto output_shape_without_padding = output_shape.value.without_padding();

    std::array<uint32_t, 5> new_input_shape{};
    std::array<uint32_t, 5> new_output_shape{};
    std::array<uint32_t, 5> new_input_padded_shape{};
    std::array<uint32_t, 5> new_output_padded_shape{};

    new_input_shape.fill(1);
    new_input_padded_shape.fill(1);
    auto input_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < input_shape.rank(); index++) {
        new_input_shape[index + input_dim_offset] = input_shape_without_padding[index];
        new_input_padded_shape[index + input_dim_offset] = input_shape.value[index];
    }

    new_output_shape.fill(1);
    new_output_padded_shape.fill(1);
    auto output_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < output_shape.rank(); index++) {
        new_output_shape[index + output_dim_offset] = output_shape_without_padding[index];
        new_output_padded_shape[index + output_dim_offset] = output_shape.value[index];
    }

    ttnn::Shape input_5d_shape(new_input_shape, new_input_padded_shape);
    ttnn::Shape output_5d_shape(new_output_shape, new_output_padded_shape);

    bool is_w_index_exist = false;
    for (auto dim : index_dims) {
        if (dim + input_dim_offset == 4) {
            is_w_index_exist = true;
        }
    }

    auto input_5d_shape_without_padding = input_5d_shape.value.without_padding();
    auto output_5d_shape_without_padding = output_5d_shape.value.without_padding();

    auto index_layout = index_tensors.front().get_layout();
    bool is_row_major_index = (index_layout == Layout::ROW_MAJOR);

    if (is_w_index_exist) {
        // compute index info
        IndexInfo index_info[5] = {0};

        for (uint32_t i = 0; i < index_tensors.size(); i++) {
            auto dim = index_dims[i] + input_dim_offset;
            auto index = index_tensors[i];

            index_info[dim].is_defined = true;
            index_info[dim].address = index.buffer()->address();
            index_info[dim].is_dram = is_dram(index);
            index_info[dim].unit_size = index.element_size();
        }

        uint32_t index_size = index_tensors[0].get_shape().value.without_padding()[-1];

        uint32_t input_unit_size = input.element_size();
        uint32_t output_unit_size = output.element_size();

        uint32_t alignment_size = 32;
        uint32_t num_elements_per_alignment = alignment_size / output_unit_size;
        uint32_t num_units =
            output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
            output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
            ((output_5d_shape_without_padding[4] + num_elements_per_alignment - 1) / num_elements_per_alignment);

        uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
        uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
                split_work_to_cores(core_range, num_units);

        Program program = Program();

        // create circular buffers
        auto src_cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
        auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].get_dtype());
        auto output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());

        auto src_cb_index = CB::c_in0;
        auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
        auto cb_src0_config = CircularBufferConfig(rounded_input_page_size, {{src_cb_index, src_cb_data_format}})
                                  .set_page_size(src_cb_index, rounded_input_page_size);
        auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

        for (uint32_t dim = 0; dim < 5; dim++) {
            if (!index_info[dim].is_defined)
                continue;

            auto src1_cb_index = CB::c_in1 + dim;
            auto index_page_size = 1024 * 4;
            auto cb_index_config = CircularBufferConfig(index_page_size, {{src1_cb_index, index_cb_data_format}})
                                       .set_page_size(src1_cb_index, index_page_size);
            auto cb_src1 = CreateCircularBuffer(program, all_cores, cb_index_config);
        }

        auto out_cb0_index = CB::c_out0;
        auto rounded_output_page_size = round_up_to_mul32(output_unit_size);
        auto cb_out0_config = CircularBufferConfig(rounded_output_page_size, {{out_cb0_index, output_cb_data_format}})
                                  .set_page_size(out_cb0_index, rounded_output_page_size);
        auto cb_out0 = CreateCircularBuffer(program, all_cores, cb_out0_config);

        auto out_cb1_index = CB::c_out1;
        auto cb_out1_config = CircularBufferConfig(rounded_output_page_size, {{out_cb1_index, output_cb_data_format}})
                                  .set_page_size(out_cb1_index, rounded_output_page_size);
        auto cb_out1 = CreateCircularBuffer(program, all_cores, cb_out1_config);

        // create read/wrtie kernel
        auto src_is_dram = is_dram(input);
        auto dst_is_dram = is_dram(output);

        std::map<string, string> reader_defines;
        std::map<string, string> writer_defines;

        if (is_row_major_index) {
            reader_defines["ROW_MAJOR_INDEX"] = 1;
        } else {
            reader_defines["TILIZE_INDEX"] = 1;
        }

        auto reader_kernel_id = CreateReadKernel(
            program,
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "reader_moreh_getitem_tilize_w.cpp",
            all_cores,
            {
                src_is_dram,
                index_info[0].is_dram,
                index_info[1].is_dram,
                index_info[2].is_dram,
                index_info[3].is_dram,
                index_info[4].is_dram,
            },
            reader_defines);
        auto writer_kernel_id = CreateWriteKernel(
            program,
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "writer_moreh_getitem_tilize_w.cpp",
            all_cores,
            {dst_is_dram},
            writer_defines);

        uint32_t face_width = 16;
        uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
        uint32_t num_alignment_width = div_up(output_5d_shape_without_padding[4], num_elements_per_alignment);
        uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

        uint32_t input_num_tile_c = input_5d_shape.value[1];
        uint32_t input_num_tile_d = input_5d_shape.value[2];
        uint32_t input_num_tile_height = input_5d_shape.value[3] / TILE_HEIGHT;
        uint32_t input_num_tile_width = input_5d_shape.value[4] / TILE_WIDTH;
        uint32_t input_noc_id_stride_h = input_num_tile_width;
        uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
        uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
        uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

        uint32_t output_num_tile_c = output_5d_shape.value[1];
        uint32_t output_num_tile_d = output_5d_shape.value[2];
        uint32_t output_num_tile_height = output_5d_shape.value[3] / TILE_HEIGHT;
        uint32_t output_num_tile_width = output_5d_shape.value[4] / TILE_WIDTH;

        uint32_t output_noc_id_stride_h = output_num_tile_width;
        uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
        uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
        uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

        uint32_t input_stick_idx_stride_w = 1;
        uint32_t input_stick_idx_stride_h = input_num_stick_width;
        uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape.value.without_padding()[3];
        uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape.value.without_padding()[2];
        uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape.value.without_padding()[1];

        // Set Runtime Args
        auto core_x_offset = core_range.start_coord.x;
        auto core_y_offset = core_range.start_coord.y;

        uint32_t g1_numcores = core_group_1.num_cores();
        uint32_t g2_numcores = core_group_2.num_cores();

        uint32_t start_id = 0;
        for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
            CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
            uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

            vector<uint32_t> reader_args = {
                // buffers
                input.buffer()->address(),
                index_info[0].address,
                index_info[1].address,
                index_info[2].address,
                index_info[3].address,
                index_info[4].address,

                // input
                input_stick_idx_stride_n,
                input_stick_idx_stride_c,
                input_stick_idx_stride_d,
                input_stick_idx_stride_h,
                input_stick_idx_stride_w,
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_num_stick_width,
                input_noc_id_stride_n,
                input_noc_id_stride_c,
                input_noc_id_stride_d,
                input_noc_id_stride_h,

                input_5d_shape_without_padding[0],
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_5d_shape_without_padding[4],

                // index
                index_info[0].is_defined,
                index_info[1].is_defined,
                index_info[2].is_defined,
                index_info[3].is_defined,
                index_info[4].is_defined,
                index_info[0].unit_size,
                index_info[1].unit_size,
                index_info[2].unit_size,
                index_info[3].unit_size,
                index_info[4].unit_size,
                index_size,

                // output
                output_5d_shape_without_padding[0],
                output_5d_shape_without_padding[1],
                output_5d_shape_without_padding[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                input.element_size(),
                num_elements_per_alignment,
                num_alignment_width,
            };

            vector<uint32_t> writer_args = {
                // buffers
                output.buffer()->address(),

                // output
                output_5d_shape_without_padding[1],
                output_5d_shape_without_padding[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_noc_id_stride_n,
                output_noc_id_stride_c,
                output_noc_id_stride_d,
                output_noc_id_stride_h,
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                output_unit_size,
                output.element_size(),
                num_elements_per_alignment,
                num_alignment_width,
            };

            SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

            start_id += num_units_per_core;
        }
        return {
            std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h, index_dims, input_dim_offset}};

    } else {
        // compute index info

        IndexInfo index_info[5] = {0};

        for (uint32_t i = 0; i < index_tensors.size(); i++) {
            auto dim = index_dims[i] + input_dim_offset;
            auto index = index_tensors[i];

            index_info[dim].is_defined = true;
            index_info[dim].address = index_tensors[i].buffer()->address();
            index_info[dim].is_dram = is_dram(index_tensors[i]);
            index_info[dim].unit_size = index.get_shape().value[-1] * index.element_size();
        }
        uint32_t index_size = index_tensors[0].get_shape().value.without_padding()[-1];

        uint32_t input_unit_size = 16 * input.element_size();
        uint32_t output_unit_size = 16 * output.element_size();

        uint32_t num_units = output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
                             output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
                             ((output_5d_shape_without_padding[4] + 15) / 16);

        uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
        uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
                split_work_to_cores(core_range, num_units);

        Program program = Program();

        // create circular buffers
        auto src_cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
        auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].get_dtype());
        auto output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());

        auto src_cb_index = CB::c_in0;
        auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
        auto cb_src0_config = CircularBufferConfig(rounded_input_page_size, {{src_cb_index, src_cb_data_format}})
                                  .set_page_size(src_cb_index, rounded_input_page_size);
        auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

        for (uint32_t dim = 0; dim < 5; dim++) {
            if (!index_info[dim].is_defined)
                continue;

            auto src1_cb_index = CB::c_in1 + dim;
            // auto index_page_size = round_up_to_mul32(index_info[dim].unit_size);
            auto index_page_size = 1024 * 4;
            auto cb_index_config = CircularBufferConfig(index_page_size, {{src1_cb_index, index_cb_data_format}})
                                       .set_page_size(src1_cb_index, index_page_size);
            auto cb_src1 = CreateCircularBuffer(program, all_cores, cb_index_config);
        }

        auto out_cb_index = CB::c_out0;
        auto rounded_output_page_size = round_up_to_mul32(input_unit_size);
        auto cb_out0_config = CircularBufferConfig(rounded_input_page_size, {{out_cb_index, output_cb_data_format}})
                                  .set_page_size(out_cb_index, rounded_input_page_size);
        auto cb_out0 = CreateCircularBuffer(program, all_cores, cb_out0_config);

        // create read/wrtie kernel
        auto src_is_dram = is_dram(input);
        auto dst_is_dram = is_dram(output);

        std::map<string, string> reader_defines;
        std::map<string, string> writer_defines;

        if (is_row_major_index) {
            reader_defines["ROW_MAJOR_INDEX"] = 1;
        } else {
            reader_defines["TILIZE_INDEX"] = 1;
        }

        auto reader_kernel_id = CreateReadKernel(
            program,
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "reader_moreh_getitem_tilize.cpp",
            all_cores,
            {
                src_is_dram,
                index_info[0].is_dram,
                index_info[1].is_dram,
                index_info[2].is_dram,
                index_info[3].is_dram,
                index_info[4].is_dram,
            },
            reader_defines);
        auto writer_kernel_id = CreateWriteKernel(
            program,
            "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
            "writer_moreh_getitem_tilize.cpp",
            all_cores,
            {dst_is_dram},
            writer_defines);

        uint32_t face_width = 16;
        uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
        uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

        uint32_t input_num_tile_c = input_5d_shape.value[1];
        uint32_t input_num_tile_d = input_5d_shape.value[2];
        uint32_t input_num_tile_height = input_5d_shape.value[3] / TILE_HEIGHT;
        uint32_t input_num_tile_width = input_5d_shape.value[4] / TILE_WIDTH;
        uint32_t input_noc_id_stride_h = input_num_tile_width;
        uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
        uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
        uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

        uint32_t output_num_tile_c = output_5d_shape.value[1];
        uint32_t output_num_tile_d = output_5d_shape.value[2];
        uint32_t output_num_tile_height = output_5d_shape.value[3] / TILE_HEIGHT;
        uint32_t output_num_tile_width = output_5d_shape.value[4] / TILE_WIDTH;

        uint32_t output_noc_id_stride_h = output_num_tile_width;
        uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
        uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
        uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

        uint32_t input_stick_idx_stride_w = 1;
        uint32_t input_stick_idx_stride_h = input_num_stick_width;
        uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape.value.without_padding()[3];
        uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape.value.without_padding()[2];
        uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape.value.without_padding()[1];

        // Set Runtime Args
        auto core_x_offset = core_range.start_coord.x;
        auto core_y_offset = core_range.start_coord.y;
        uint32_t g1_numcores = core_group_1.num_cores();
        uint32_t g2_numcores = core_group_2.num_cores();

        uint32_t start_id = 0;
        for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
            CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
            uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

            vector<uint32_t> reader_args = {
                // buffers
                input.buffer()->address(),
                index_info[0].address,
                index_info[1].address,
                index_info[2].address,
                index_info[3].address,
                index_info[4].address,

                // input
                input_stick_idx_stride_n,
                input_stick_idx_stride_c,
                input_stick_idx_stride_d,
                input_stick_idx_stride_h,
                input_stick_idx_stride_w,
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_noc_id_stride_n,
                input_noc_id_stride_c,
                input_noc_id_stride_d,
                input_noc_id_stride_h,
                input_num_stick_width,

                input_5d_shape_without_padding[0],
                input_5d_shape_without_padding[1],
                input_5d_shape_without_padding[2],
                input_5d_shape_without_padding[3],
                input_5d_shape_without_padding[4],

                // index
                index_info[0].is_defined,
                index_info[1].is_defined,
                index_info[2].is_defined,
                index_info[3].is_defined,
                index_info[4].is_defined,
                index_info[0].unit_size,
                index_info[1].unit_size,
                index_info[2].unit_size,
                index_info[3].unit_size,
                index_info[4].unit_size,
                index_size,

                // output
                output_5d_shape[0],
                output_5d_shape[1],
                output_5d_shape[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                input_unit_size,
                input.element_size(),
            };
            vector<uint32_t> writer_args = {
                // buffers
                output.buffer()->address(),

                // output
                output_5d_shape_without_padding[1],
                output_5d_shape_without_padding[2],
                output_5d_shape_without_padding[3],
                output_5d_shape_without_padding[4],
                output_noc_id_stride_n,
                output_noc_id_stride_c,
                output_noc_id_stride_d,
                output_noc_id_stride_h,
                output_num_stick_width,

                // etc
                start_id,
                num_units_per_core,
                output_unit_size,
                output.element_size(),
            };

            SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

            start_id += num_units_per_core;
        }

        return {
            std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h, index_dims, input_dim_offset}};
    }
}

void MorehGetItemOperation::MorehGetItemTilizedFactory::override_runtime_arguments(
    cached_program_t &cached_program,
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
    auto &program = cached_program.program;
    auto &reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto &writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;
    auto index_dims = cached_program.shared_variables.index_dims;
    auto input_dim_offset = cached_program.shared_variables.input_dim_offset;

    TT_ASSERT(tensor_return_value.buffer()->size() == 1);

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = tensor_return_value.buffer();
    auto index_tensors = tensor_args.index_tensors;
    IndexInfo index_info[5] = {0};
    for (uint32_t i = 0; i < index_dims.size(); i++) {
        auto dim = index_dims[i] + input_dim_offset;
        auto index_buffer = index_tensors[i];

        index_info[dim].address = index_buffer.buffer()->address();
    }

    for (uint32_t icore = 0; icore < num_cores; icore++) {
        CoreCoord core = {icore / core_h, icore % core_h};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = index_info[0].address;
            runtime_args[2] = index_info[1].address;
            runtime_args[3] = index_info[2].address;
            runtime_args[4] = index_info[3].address;
            runtime_args[5] = index_info[4].address;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_getitem
