// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_getitem/moreh_getitem_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

struct IndexInfo
{
    bool is_defined;
    bool is_dram;
    uint32_t address;
    uint32_t unit_size;
};

operation::ProgramWithCallbacks moreh_getitem_rm(
    const Tensor &input,
    const std::vector<Tensor> &index_tensors,
    const std::vector<uint32_t> &index_dims,
    const Tensor &output,
    const CoreRange core_range) {

    log_debug(LogTest, "moreh_getitem_rm");

    auto input_shape = input.get_legacy_shape();
    auto output_shape = output.get_legacy_shape();

    std::array<uint32_t, 5> new_input_shape{};
    std::array<uint32_t, 5> new_output_shape{};
    new_input_shape.fill(1);
    new_output_shape.fill(1);

    auto input_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < input_shape.rank(); index++) {
        new_input_shape[index + input_dim_offset] = input_shape[index];
    }
    auto output_dim_offset = 5 - output_shape.rank();
    for (auto index = 0; index < output_shape.rank(); index++) {
        new_output_shape[index + output_dim_offset] = output_shape[index];
    }
    Shape input_5d_shape(new_input_shape);
    Shape output_5d_shape(new_output_shape);

    uint32_t index_start_dim = index_dims.front();
    uint32_t index_end_dim = index_dims.back();

    Tensor input_5d = input;
    input_5d = input_5d.reshape(input_5d_shape);

    auto input_5d_shape_without_padding = input_5d_shape.without_padding();

    IndexInfo index_info[5] = {0};

    for (uint32_t i = 0 ; i < index_tensors.size(); i++) {
        auto dim = index_dims[i] + input_dim_offset;
        auto index = index_tensors.at(i);

        index_info[dim].is_defined = true;
        index_info[dim].address = index_tensors.at(i).buffer()->address();
        index_info[dim].is_dram = is_dram(index_tensors.at(i));
        index_info[dim].unit_size = index.get_legacy_shape()[-1] * index.element_size();
    }

    uint32_t index_size = index_tensors.front().get_legacy_shape()[-1];

    uint32_t input_unit_size = input_5d_shape[-1] * input_5d.element_size();
    uint32_t output_unit_size = input_unit_size;

    // split work
    uint32_t num_units = output.volume() / output_shape[-1];
    log_debug(LogTest, "num_units {}", num_units);

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(core_range, num_units);

    Program program = Program();

    // create circular buffers
    auto src_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    auto index_cb_data_format = tt_metal::datatype_to_dataformat_converter(index_tensors.at(0).get_dtype());
    auto output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto src_cb_index = tt::CB::c_in0;
    auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
    auto cb_src0_config = tt_metal::CircularBufferConfig(rounded_input_page_size, {{src_cb_index, src_cb_data_format}})
                              .set_page_size(src_cb_index, rounded_input_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    for (uint32_t dim = 0; dim < 5; dim++) {
        if (!index_info[dim].is_defined)
            continue;

        auto src1_cb_index = tt::CB::c_in1 + dim;
        auto index_page_size = round_up_to_mul32(index_info[dim].unit_size);
        auto cb_index_config = tt_metal::CircularBufferConfig(index_page_size, {{src1_cb_index, index_cb_data_format}})
                                   .set_page_size(src1_cb_index, index_page_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_index_config);
    }

    auto out_cb_index = tt::CB::c_out0;
    auto rounded_output_page_size = round_up_to_mul32(input_unit_size);
    auto cb_out0_config =
        tt_metal::CircularBufferConfig(rounded_input_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, rounded_input_page_size);
    auto cb_out0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_out0_config);


    // create read/wrtie kernel
    auto src_is_dram = is_dram(input_5d);
    auto dst_is_dram = is_dram(output);

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_getitem/moreh_getitem_rm/kernels/reader_moreh_getitem.cpp",
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
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_getitem/moreh_getitem_rm/kernels/writer_moreh_getitem.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    uint32_t input_stick_idx_stride_h = 1;
    uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape.without_padding()[3];
    uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape.without_padding()[2];
    uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape.without_padding()[1];

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
            input_5d.buffer()->address(),
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
            index_start_dim,
            index_end_dim,

            // output
            output_5d_shape[0],
            output_5d_shape[1],
            output_5d_shape[2],
            output_5d_shape[3],
            output_5d_shape[4],

            // etc
            start_id,
            num_units_per_core,
            input_unit_size,
        };

        vector<uint32_t> writer_args = {
            // buffer
            output.buffer()->address(),

            // output
            output_unit_size,

            // etc
            start_id,
            num_units_per_core,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        start_id += num_units_per_core;
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = reader_kernel_id, writer_kernel_id = writer_kernel_id, num_cores, core_h, index_dims, input_dim_offset](
            const Program &program,
            const std::vector<Buffer *> &input_buffers,
            const std::vector<Buffer *> &output_buffers) {
            TT_ASSERT(output_buffers.size() == 1);

            auto src_buffer = input_buffers.at(0);
            auto dst_buffer = output_buffers.at(0);

            IndexInfo index_info[5] = {0};

            for (uint32_t i = 0; i < index_dims.size(); i++) {
                auto dim = index_dims[i] + input_dim_offset;
                auto index_buffer = input_buffers.at(i + 1);

                index_info[dim].address = index_buffer->address();
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
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
