// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_dot/moreh_dot_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_dot_single_core(const Tensor &a, const Tensor &b, Tensor &output) {
    Program program{};
    CoreCoord core = {0, 0};
    const uint32_t core_num = 1;

    DataFormat cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;
    float scaler = 1.0f;
    const auto &a_shape_wo_padding = a.get_legacy_shape().without_padding();
    uint32_t pad_h = a_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (TILE_WIDTH) : (pad_w);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;   // a
    const uint32_t in1_t = 2;   // b
    const uint32_t in2_t = 1;   // scaler
    const uint32_t out0_t = 2;  // out
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_in1, in1_t},
            {CB::c_in2, in2_t},
            {CB::c_out0, out0_t},
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)is_dram(src0_buffer),
        (std::uint32_t)is_dram(src1_buffer),
        *reinterpret_cast<uint32_t *>(&scaler)};

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)CB::c_out0, (std::uint32_t)is_dram(dst_buffer)};

    const auto reader_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot/single_core/kernels/reader_moreh_dot.cpp";
    const auto writer_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot/single_core/kernels/writer_moreh_dot.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_dot/single_core/kernels/moreh_dot.cpp";
    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core, core_num, compute_kernel_args}, compute_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_buffer->address(), src1_buffer->address(), num_tiles, 0, mask_h, mask_w});
    SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles, 1});
    SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), 1, 0});

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, compute_kernel_id](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        uint32_t num_tiles = input_tensors.at(0).volume() / TILE_HW;

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[1] = src_buffer_b->address();
            runtime_args[2] = num_tiles;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
            runtime_args[0] = num_tiles;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = 1;
        }
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
