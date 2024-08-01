// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_test_impl(
    const Tensor &input, const Tensor &output, const DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device{output.device()};
    auto program{CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_tiles{output.volume() / TILE_HW};
    tt::DataFormat data_format{tt_metal::datatype_to_dataformat_converter(output.get_dtype())};
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid{device->compute_with_storage_grid_size()};
    const auto num_cores_y{grid.y};

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_tiles);

    log_debug(LogOp, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogOp, "num_tiles_per_core_group_1 : {}", num_output_tiles_per_core_group_1);
    log_debug(LogOp, "num_tiles_per_core_group_2 : {}", num_output_tiles_per_core_group_2);
    log_debug(LogOp, "num_tiles {}", num_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{2};
    const uint32_t out0_t{2};

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_out0, out0_t},
            {CB::c_intermed0, out0_t, tt::DataFormat::Float16_b},
            {CB::c_intermed1, out0_t, tt::DataFormat::UInt32},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(is_dram(input))};
    const std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(output))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/reader_test.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/writer_test.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> compute_defines;

    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/moreh_test.cpp";
    std::vector<uint32_t> compute_args_group_1 = {
        num_output_tiles_per_core_group_1,  // num_output_tiles
    };

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_output_tiles_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {
        num_output_tiles_per_core_group_2,  // num_output_tiles
        };

        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_output_tiles_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr{input.buffer()->address()};
    const auto output_addr{output.buffer()->address()};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_to_be_used; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_1_id, core, {num_output_tiles_per_core, num_tiles_written});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_output_tiles_per_core, num_tiles_written});
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {input_addr, num_output_tiles_per_core, num_tiles_written});

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output_addr, num_output_tiles_per_core, num_tiles_written});

        num_tiles_written += num_output_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *input_buffer = input_tensors.at(0).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
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
