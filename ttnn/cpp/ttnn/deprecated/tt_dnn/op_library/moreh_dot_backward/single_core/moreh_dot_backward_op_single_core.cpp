// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_dot_backward_single_core(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad) {
    Program program{};
    CoreCoord core = {0, 0};
    const uint32_t core_num = 1;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    Buffer *src0_buffer = output_grad.buffer();
    Buffer *src1_buffer = input.buffer();
    Buffer *src2_buffer = other.buffer();

    uint32_t num_tiles = input.volume() / TILE_HW;
    float scaler = 1.0f;
    const auto &a_shape_wo_padding = input.get_legacy_shape().without_padding();
    uint32_t pad_h = a_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (TILE_WIDTH) : (pad_w);

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;   // output_grad
    const uint32_t in1_t = 2;   // input
    const uint32_t in2_t = 2;   // other
    const uint32_t out0_t = 2;  // input_grad
    const uint32_t out1_t = 2;  // other_grad

    CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_in1, in1_t},
            {CB::c_in2, in2_t},
            {CB::c_out0, out0_t},
            {CB::c_out1, out1_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    bool has_input_grad = input_grad.has_value();
    bool has_other_grad = other_grad.has_value();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)is_dram(src0_buffer), (std::uint32_t)is_dram(src1_buffer), (std::uint32_t)is_dram(src2_buffer)};

    bool dst0_is_dram = false;
    bool dst1_is_dram = false;
    uint32_t dst0_address = 0;
    uint32_t dst1_address = 0;

    if (has_input_grad) {
        const auto &input_grad_tensor = input_grad.value();
        Buffer *dst0_buffer = input_grad_tensor.buffer();
        TT_ASSERT(dst0_buffer != nullptr, "input_grad buffer should be allocated on device!");
        dst0_is_dram = is_dram(dst0_buffer);
        dst0_address = dst0_buffer->address();
    }

    if (has_other_grad) {
        const auto &other_grad_tensor = other_grad.value();
        Buffer *dst1_buffer = other_grad_tensor.buffer();
        TT_ASSERT(dst1_buffer != nullptr, "other_grad buffer should be allocated on device!");
        dst1_is_dram = is_dram(dst1_buffer);
        dst1_address = dst1_buffer->address();
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)CB::c_out0,
        (std::uint32_t)CB::c_out1,
        (std::uint32_t)dst0_is_dram,
        (std::uint32_t)dst1_is_dram,
    };

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot_backward/kernels/reader_moreh_dot_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot_backward/kernels/writer_moreh_dot_backward.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> compute_defines;

    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot_backward/kernels/moreh_dot_backward.cpp";
    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core, core_num, compute_kernel_args}, compute_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {(std::uint32_t)has_input_grad,
         (std::uint32_t)has_other_grad,
         src0_buffer->address(),
         src1_buffer->address(),
         src2_buffer->address(),
         num_tiles,
         0});

    SetRuntimeArgs(
        program, compute_kernel_id, core, {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, num_tiles});

    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, dst0_address, dst1_address, num_tiles, 0});

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, compute_kernel_id](
            const void *operation,
            const Program &program,
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<Tensor> &output_tensors) {
            const auto &output_grad = input_tensors.at(0);
            const auto &input = input_tensors.at(1);
            const auto &other = input_tensors.at(2);
            const auto &input_grad = optional_input_tensors.at(0);
            const auto &other_grad = optional_input_tensors.at(1);

            bool has_input_grad = input_grad.has_value();
            bool has_other_grad = other_grad.has_value();

            Buffer *src0_buffer = output_grad.buffer();
            Buffer *src1_buffer = input.buffer();
            Buffer *src2_buffer = other.buffer();

            uint32_t dst0_address = 0;
            uint32_t dst1_address = 0;

            if (has_input_grad) {
                const auto &input_grad_tensor = input_grad.value();
                Buffer *dst0_buffer = input_grad_tensor.buffer();
                dst0_address = dst0_buffer->address();
            }

            if (has_other_grad) {
                const auto &other_grad_tensor = other_grad.value();
                Buffer *dst1_buffer = other_grad_tensor.buffer();
                dst1_address = dst1_buffer->address();
            }

            CoreCoord core = {0, 0};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
                runtime_args[2] = src0_buffer->address();
                runtime_args[3] = src1_buffer->address();
                runtime_args[4] = src2_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
                runtime_args[2] = dst0_address;
                runtime_args[3] = dst1_address;
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
