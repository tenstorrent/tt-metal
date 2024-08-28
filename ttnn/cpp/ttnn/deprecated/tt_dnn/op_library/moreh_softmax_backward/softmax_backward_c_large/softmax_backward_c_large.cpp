// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_softmax_backward_c_large(const Tensor &output, const Tensor &output_grad, const Tensor &input_grad, uint32_t dim, const CoreRange core_range, const MorehSoftmaxBackwardOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    log_info(LogTest, "Large tensor algorithm selected");

    // split work
    auto shape = input_grad.get_legacy_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    uint32_t num_tiles = input_grad.volume() / shape[dim] / H / W * Ht * Wt;

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_range, num_tiles);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // y
            {CB::c_in1, 2},        // dy
            {CB::c_out0, 2},       // dx
            {CB::c_intermed0, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // y * dy
            {CB::c_intermed1, 2, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // sum(y * dy)
            {CB::c_intermed2, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // dy - sum
        });

    // create read/wrtie kernel
    bool y_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dy_is_dram = output_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dx_is_dram = input_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        compute_defines["LOG"] = 1;
        reader_defines["LOG"] = 1;
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    auto reader_kernel_id = CreateReadKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/kernels/reader_moreh_softmax_backward_c.cpp", all_cores, {y_is_dram, dy_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/kernels/writer_moreh_softmax_backward_c.cpp", all_cores, {dx_is_dram}, writer_defines);

    auto outer_stride = Ht * Wt;
    for(int i = dim ; i < shape.rank() - 2; i++ ) {
        outer_stride *= shape[i];
    }
    auto dim_size = shape[dim];
    auto inner_size = outer_stride / dim_size;

    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/kernels/moreh_softmax_backward_c_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, dim_size}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, dim_size}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        vector<uint32_t> reader_args = {
            output.buffer()->address(),
            output_grad.buffer()->address(),
            num_tiles_per_core, tile_offset,
            outer_stride, inner_size,
            dim_size};

        vector<uint32_t> writer_args = {input_grad.buffer()->address(), num_tiles_per_core, tile_offset,
            outer_stride, inner_size,
            dim_size};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
