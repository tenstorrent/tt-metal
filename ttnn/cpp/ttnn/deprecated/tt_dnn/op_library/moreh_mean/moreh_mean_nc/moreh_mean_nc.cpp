// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_mean/moreh_mean_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_mean_nc(
    const Tensor &input,
    const Tensor &output,
    int64_t dim,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = input.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    const auto single_tile_size = detail::TileSize(cb_data_format);

    const auto &input_shape = input.get_legacy_shape();
    const auto &input_shape_without_padding = input_shape.without_padding();

    const auto Ht = input_shape[-2] / TILE_HEIGHT;
    const auto Wt = input_shape[-1] / TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];

    const auto rank = input_shape.rank();
    auto input_tile_stride = HtWt;
    for (int i = dim + 1; i < rank - 2; i++) {
        input_tile_stride *= input_shape[i];
    }

    uint32_t inner_size = 1;
    for (int i = dim + 1; i < rank - 2; i++) {
        inner_size *= input_shape[i];
    }

    const auto units_to_divide = output.volume() / TILE_HW;

    log_debug(
        LogTest,
        "dim {} num_reduce_input_tile {} input_tile_offset {}, units_to_divide {}",
        dim,
        num_reduce_input_tile,
        input_tile_stride,
        units_to_divide);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, 2},        // input
            {CB::c_in1, 1},        // zero
            {CB::c_in2, 1},        // scaler
            {CB::c_intermed0, 1},  // accumulated mean
            {CB::c_out0, 2},       // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_mean/kernels/reader_moreh_mean_nc.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_mean/kernels/writer_moreh_mean_nc.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_mean/kernels/moreh_mean_nc.cpp";
    std::map<string, string> compute_defines;
    const std::vector<uint32_t> compute_args_group_1{units_per_core_group_1};
    const std::vector<uint32_t> compute_args_group_2{units_per_core_group_2};

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }
    auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, units_per_core_group_1, compute_args_group_1},
            {core_group_2, units_per_core_group_2, compute_args_group_2},
        },
        ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            // TODO(hyungsuk): change preserve_fp32_precision from false to fp32_dest_acc_en after fix #10337
            // .preserve_fp32_precision = fp32_dest_acc_en,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .defines = compute_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};

        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_ids[0], core, {num_reduce_input_tile, units_per_core});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_ids[1], core, {num_reduce_input_tile, units_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_reduce_input_tile,
             units_per_core,
             input_tile_stride,
             tile_offset,
             static_cast<uint32_t>(is_dram(input)),
             HtWt,
             inner_size});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(), units_per_core, tile_offset, static_cast<uint32_t>(is_dram(output))});

        tile_offset += units_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores, core_h](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto *input_buffer = input_tensors.at(0).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / core_h, i % core_h};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
