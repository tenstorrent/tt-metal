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
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_clip_grad_norm_step3_impl(
    const std::vector<Tensor>& inputs, const Tensor& clip_coef_clamped) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = inputs.at(0).device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
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
    const uint32_t in0_t = 1;  // input(inplace)
    const uint32_t in1_t = 1;  // clip_coef_clamped

    const uint32_t out0_t = 1;  // output(inplace)

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(inputs.at(0).get_dtype());

    CreateCircularBuffer(
        program,
        core_group_1,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input(inplace)
            {CB::c_in1, in1_t},    // clip_coef_clamped
            {CB::c_out0, out0_t},  // output(inplace)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "reader_moreh_clip_grad_norm_step3.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "writer_moreh_clip_grad_norm_step3.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, core_group_1);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, core_group_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "moreh_clip_grad_norm_step3_kernel.cpp";

    const auto compute_kernels_id =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_inputs_per_core_group_1});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto clip_coef_clamped_addr = clip_coef_clamped.buffer()->address();
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = input.volume() / tt::constants::TILE_HW;

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            clip_coef_clamped_addr,
            static_cast<uint32_t>(is_dram(clip_coef_clamped)),
            num_tiles};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{input_addr, static_cast<uint32_t>(is_dram(input)), num_tiles};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{num_tiles};
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_addresses_callback =
        [reader_kernels_id = reader_kernels_id,
         writer_kernels_id = writer_kernels_id,
         num_cores_to_be_used = num_cores_to_be_used,
         num_cores_y = num_cores_y](
            const Program& program, const std::vector<Buffer*>& input_buffers, const std::vector<Buffer*>&) {
            auto clip_coef_clamped_buffer = input_buffers.at(input_buffers.size() - 1);
            const auto clip_coef_clamped_address = clip_coef_clamped_buffer->address();

            for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};

                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                    runtime_args[0] = input_buffers.at(i)->address();
                    runtime_args[2] = clip_coef_clamped_address;
                }

                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                    runtime_args[0] = input_buffers.at(i)->address();
                }
            }
        };

    return {.program = std::move(program), .override_addresses_callback = override_addresses_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
