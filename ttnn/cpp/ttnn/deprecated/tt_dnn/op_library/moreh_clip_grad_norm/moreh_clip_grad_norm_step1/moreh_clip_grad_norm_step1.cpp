// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
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

operation::ProgramWithCallbacks moreh_clip_grad_norm_step1_impl(const std::vector<Tensor>& inputs,
                                                                float norm_type,
                                                                uint32_t tile_offset_of_tmp_pow_sum,
                                                                const Tensor& tmp_pow_sum) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = tmp_pow_sum.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    std::vector<std::pair<uint32_t, uint32_t>> origin_hw_vec;
    origin_hw_vec.reserve(num_inputs);

    for (uint32_t j = 0; j < num_inputs; ++j) {
        const auto& input_shape_without_padding = inputs.at(j).get_legacy_shape().without_padding();
        origin_hw_vec.emplace_back(input_shape_without_padding[2], input_shape_without_padding[3]);
    }

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    const auto [num_cores_to_be_used,
                all_cores,
                core_group_1,
                core_group_2,
                num_inputs_per_core_group_1,
                num_inputs_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_inputs);
    TT_ASSERT(core_group_2.ranges().empty());
    TT_ASSERT(num_inputs_per_core_group_1 == 1);
    TT_ASSERT(num_inputs_per_core_group_2 == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==x)
    const uint32_t in1_t = 1;  // one
    const uint32_t in2_t = 1;  // decimal
    const uint32_t in3_t = 2;  // mask_h_w

    const uint32_t out0_t = 1;  // output(==y)

    const uint32_t im0_t = 1;  // |x|
    const uint32_t im1_t = 1;  // |x|^p
    const uint32_t im2_t = 1;  // Add[|x|^p * exp(log(|x|) * decimal)]
    const uint32_t im3_t = 1;  // log(|x|)
    const uint32_t im4_t = 1;  // exp(log(|x|) * decimal)
    const uint32_t im5_t = 1;  // |x|^p * exp(log(|x|) * decimal)

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(tmp_pow_sum.get_dtype());

    CreateCircularBuffer(program,
                         core_group_1,
                         cb_data_format,
                         {
                             {CB::c_in0, in0_t},        // input(==x)
                             {CB::c_in1, in1_t},        // one
                             {CB::c_in2, in2_t},        // decimal
                             {CB::c_in3, in3_t},        // mask_h_w
                             {CB::c_out0, out0_t},      // output(==y)
                             {CB::c_intermed0, im0_t},  // |x|
                             {CB::c_intermed1, im1_t},  // |x|^p
                             {CB::c_intermed2, im2_t},  // Add[|x|^p * exp(log(|x|) * decimal)]
                             {CB::c_intermed3, im3_t},  // log(|x|)
                             {CB::c_intermed4, im4_t},  // exp(log(|x|) * decimal)
                             {CB::c_intermed5, im5_t},  // |x|^p * exp(log(|x|) * decimal)
                         });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/kernels/"
        "reader_moreh_clip_grad_norm_step1.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/kernels/"
        "writer_moreh_clip_grad_norm_step1.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, core_group_1);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, core_group_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/kernels/"
        "moreh_clip_grad_norm_step1_kernel.cpp";

    const auto compute_kernels_id =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_inputs_per_core_group_1}, compute_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_addr = tmp_pow_sum.buffer()->address();

    uint32_t tile_offset = tile_offset_of_tmp_pow_sum;
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = input.volume() / tt::constants::TILE_HW;
        const auto [origin_h, origin_w] = origin_hw_vec.at(i);

        // reader
        const std::vector<uint32_t> reader_runtime_args{input_addr,
                                                        static_cast<uint32_t>(is_dram(input)),
                                                        num_tiles,
                                                        *reinterpret_cast<uint32_t*>(&decimal),
                                                        origin_h,
                                                        origin_w};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr, static_cast<uint32_t>(is_dram(tmp_pow_sum)), tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_tiles,
            p,
            static_cast<uint32_t>(p_is_negative),
            origin_h,
            origin_w,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);

        tile_offset++;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           compute_kernels_id = compute_kernels_id,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>&) {
        const auto norm_type = static_cast<const MorehClipGradNormStep1*>(operation)->norm_type;

        auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(norm_type);

        auto output_buffer = optional_input_tensors.at(0).value().buffer();
        const auto output_address = output_buffer->address();

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = input_tensors.at(i).buffer()->address();
                runtime_args[3] = *reinterpret_cast<uint32_t*>(&decimal);
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = output_address;
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, compute_kernels_id, core);
                runtime_args[1] = p;
                runtime_args[2] = static_cast<uint32_t>(p_is_negative);
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
