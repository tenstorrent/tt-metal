// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory_general_w_small.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <utility>

namespace ttnn::prim {

SoftmaxProgramFactoryGeneralWSmall::cached_program_t SoftmaxProgramFactoryGeneralWSmall::create(
    const SoftmaxParams& attributes, const SoftmaxInputs& tensor_args, Tensor& output_tensor) {
    log_debug(tt::LogMetal, "SoftmaxProgramFactoryGeneralWSmall selected");

    tt::tt_metal::Program program{};

    // Constants
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& compute_kernel_config = attributes.compute_kernel_config;
    auto* const device = input_tensor.device();
    const auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    const auto shape = input_tensor.padded_shape();
    const auto H = shape[-2];
    const auto W = shape[-1];
    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    // Work split
    auto num = input_tensor.physical_volume() / H / W;
    uint32_t num_kernel_rows = num * Ht;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        operations::split_work_to_cores_wt_core_range(core_range, num_kernel_rows);

    auto arch = input_tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input_tensor.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    // Circular buffers
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    operations::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, Wt},                         // input
            {tt::CBIndex::c_1, 1},                          // mask
            {tt::CBIndex::c_2, 1},                          // scaler
            {tt::CBIndex::c_16, Wt},                        // output
            {tt::CBIndex::c_24, Wt, intermed_data_format},  // exp(x)
            {tt::CBIndex::c_25, 1, intermed_data_format},   // reduce
            {tt::CBIndex::c_26, 1, intermed_data_format},   // max
            {tt::CBIndex::c_27, Wt, intermed_data_format},  // x - max
            {tt::CBIndex::c_28, 1, intermed_data_format}    // tmp
        });

    // Data movement kernels
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> reader_ct_args = {static_cast<uint32_t>(input_tensor.dtype() == DataType::FLOAT32)};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct_args);
    const auto reader_kernel_id = operations::CreateReadKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/reader_moreh_softmax_w.cpp",
        all_cores,
        reader_ct_args,
        reader_defines);
    std::vector<uint32_t> writer_ct_args = {};
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);
    const auto writer_kernel_id = operations::CreateWriteKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/writer_moreh_softmax_w.cpp",
        all_cores,
        writer_ct_args,
        writer_defines);

    // Kernel constants
    std::map<std::string, std::string> compute_defines;
    compute_defines["SOFTMAX"] = "1";
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Compute kernels
    operations::CreateComputeKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_w.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Wt}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Wt}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Runtime args
    const auto core_x_offset = core_range.start_coord.x;
    const auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        float scaler = 1.0f;
        uint32_t mask_w = input_tensor.logical_shape()[-1] % tt::constants::TILE_WIDTH;
        if (mask_w == 0) {
            mask_w = tt::constants::TILE_WIDTH;
        }
        const std::vector<uint32_t> reader_args = {
            input_tensor.buffer()->address(),
            num_tiles_per_core,
            tile_offset,
            Wt,
            *reinterpret_cast<uint32_t*>(&scaler),
            mask_w};

        const std::vector<uint32_t> writer_args = {
            output_tensor.buffer()->address(), num_tiles_per_core, tile_offset, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core * Wt;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

}  // namespace ttnn::prim
