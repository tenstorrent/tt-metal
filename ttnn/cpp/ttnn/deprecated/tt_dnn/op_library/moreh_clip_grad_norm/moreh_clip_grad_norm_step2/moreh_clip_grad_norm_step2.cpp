// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_clip_grad_norm_step2_impl(const Tensor& tmp_pow_sum,
                                                                float norm_type,
                                                                const Tensor& total_norm) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = tmp_pow_sum.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_tiles = tmp_pow_sum.volume() / tt::constants::TILE_HW;

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(1.0f / norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord single_core = {0, 0};

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==tmp_pow_sum)
    const uint32_t in1_t = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    const uint32_t out0_t = 1;  // output(==total_norm)

    const uint32_t im0_t = 1;  // Sum[tmp_pow_sum](==x)
    const uint32_t im1_t = 1;  // x^p
    const uint32_t im2_t = 1;  // log(x)
    const uint32_t im3_t = 1;  // exp(log(x) * decimal)

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(total_norm.get_dtype());

    CreateCircularBuffer(program,
                         single_core,
                         cb_data_format,
                         {
                             {CB::c_in0, in0_t},        // input(==tmp_pow_sum)
                             {CB::c_in1, in1_t},        // decimal
                             {CB::c_out0, out0_t},      // output(==total_norm)
                             {CB::c_intermed0, im0_t},  // Sum[tmp_pow_sum](==x)
                             {CB::c_intermed1, im1_t},  // x^p
                             {CB::c_intermed2, im2_t},  // log(x)
                             {CB::c_intermed3, im3_t},  // exp(log(x) * decimal)
                         });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "reader_moreh_clip_grad_norm_step2.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "writer_moreh_clip_grad_norm_step2.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, single_core);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, single_core);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "moreh_clip_grad_norm_step2_kernel.cpp";

    const auto compute_kernels_id = CreateComputeKernel(program, compute_kernel_file, {single_core, num_tiles});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = tmp_pow_sum.buffer()->address();
    const auto output_addr = total_norm.buffer()->address();

    // reader
    const std::vector<uint32_t> reader_runtime_args{
        input_addr, static_cast<uint32_t>(is_dram(tmp_pow_sum)), num_tiles, *reinterpret_cast<uint32_t*>(&decimal)};
    SetRuntimeArgs(program, reader_kernels_id, single_core, reader_runtime_args);

    // writer
    const std::vector<uint32_t> writer_runtime_args{output_addr, static_cast<uint32_t>(is_dram(total_norm))};
    SetRuntimeArgs(program, writer_kernels_id, single_core, writer_runtime_args);

    // compute
    const std::vector<uint32_t> compute_runtime_args{num_tiles, p, static_cast<uint32_t>(p_is_negative)};
    SetRuntimeArgs(program, compute_kernels_id, single_core, compute_runtime_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           compute_kernels_id = compute_kernels_id,
                                           single_core = single_core](const void* operation,
                                                                      Program& program,
                                                                      const std::vector<Tensor>& input_tensors,
                                                                      const std::vector<std::optional<const Tensor>>&,
                                                                      const std::vector<Tensor>&) {
        const auto norm_type = static_cast<const MorehClipGradNormStep2*>(operation)->norm_type;

        auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(1.0f / norm_type);

        const auto input_address = input_tensors.at(0).buffer()->address();
        const auto output_address = input_tensors.at(1).buffer()->address();

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, single_core);
            runtime_args[0] = input_address;
            runtime_args[3] = *reinterpret_cast<uint32_t*>(&decimal);
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, single_core);
            runtime_args[0] = output_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, compute_kernels_id, single_core);
            runtime_args[1] = p;
            runtime_args[2] = static_cast<uint32_t>(p_is_negative);
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
