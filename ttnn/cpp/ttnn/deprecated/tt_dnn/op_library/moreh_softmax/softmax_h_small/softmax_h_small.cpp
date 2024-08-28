// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_h_small_available(const Tensor &tensor, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto h = tensor.get_legacy_shape()[-2];
    int32_t Ht = (h + TILE_HEIGHT - 1) / TILE_HEIGHT;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt_metal::detail::TileSize(data_format);
    auto intermed_tile_size = tt_metal::detail::TileSize(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;   // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Ht * tile_size;   // output;

    cb_usage += Ht * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Ht * intermed_tile_size;   // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (L1_UNRESERVED_BASE + cb_usage <= L1_512KB);
}

operation::ProgramWithCallbacks moreh_softmax_h_small(const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    log_info(LogTest, "Small tensor algorithm selected");
    // split work
    auto shape = input.get_legacy_shape();
    auto H = shape[-2];
    auto W = shape[-1];

    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    auto num = input.volume() / H / W;
    uint32_t num_cols_tiles = num * Wt;
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_range, num_cols_tiles);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, Ht},         // input
            {CB::c_in1, 1},         // mask
            {CB::c_in2, 1},          // scaler
            {CB::c_out0, Ht},        // output
            {CB::c_intermed0, Ht, intermed_data_format},  // exp(x)
            {CB::c_intermed1, 1, intermed_data_format},   // reduce
            {CB::c_intermed2, 1, intermed_data_format},     // max
            {CB::c_intermed3, Ht, intermed_data_format},     // x - max
            {CB::c_intermed4, 1, intermed_data_format}     // tmp
        });

    // create read/wrtie kernel
    bool src_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/reader_moreh_softmax_h.cpp", all_cores, {src_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/writer_moreh_softmax_h.cpp", all_cores, {dst_is_dram}, writer_defines);

    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxOp::SOFTMAX || op == MorehSoftmaxOp::LOGSOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["LOG"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/moreh_softmax_h.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Ht}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Ht}},
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

        float scaler = 1.0f;
        uint32_t mask_h = shape.without_padding()[-2] % TILE_HEIGHT;
        if(mask_h == 0) mask_h = TILE_HEIGHT;
        vector<uint32_t> reader_args = {
            input.buffer()->address(), num_tiles_per_core, tile_offset, Ht, Wt, *reinterpret_cast<uint32_t *>(&scaler), mask_h};

        vector<uint32_t> writer_args = {output.buffer()->address(), num_tiles_per_core, tile_offset, Ht, Wt};

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
