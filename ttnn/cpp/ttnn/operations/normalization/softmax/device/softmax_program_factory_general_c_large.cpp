// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory_general_c_large.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <utility>

namespace ttnn::prim {

SoftmaxProgramFactoryGeneralCLarge::cached_program_t SoftmaxProgramFactoryGeneralCLarge::create(
    const SoftmaxParams& attributes, const SoftmaxInputs& tensor_args, Tensor& output_tensor) {
    log_debug(tt::LogMetal, "SoftmaxProgramFactoryGeneralCLarge selected");

    tt::tt_metal::Program program{};

    // Constants
    const auto& input = tensor_args.input_tensor;
    const auto dim = static_cast<int>(static_cast<unsigned char>(attributes.dim));
    const auto& compute_kernel_config = attributes.compute_kernel_config;
    auto* const device = input.device();
    const auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    const auto shape = input.padded_shape();
    const auto H = shape[-2];
    const auto W = shape[-1];
    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    // Work split
    const uint32_t num_tiles = input.physical_volume() / shape[dim] / H / W * Ht * Wt;
    const uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        operations::split_work_to_cores_wt_core_range(core_range, num_tiles);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    // Circular buffers
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    operations::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 2},                         // input
            {tt::CBIndex::c_16, 2},                        // output
            {tt::CBIndex::c_24, 1, intermed_data_format},  // exp(x)
            {tt::CBIndex::c_25, 1, intermed_data_format},  // recips
            {tt::CBIndex::c_26, 2, intermed_data_format},  // add
            {tt::CBIndex::c_27, 1},                        // max
            {tt::CBIndex::c_28, 1, intermed_data_format},  // tmp
        });

    // Data movement kernels
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> reader_ct_args = {};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    const auto reader_kernel_id = operations::CreateReadKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/reader_moreh_softmax_c_large.cpp",
        all_cores,
        reader_ct_args,
        reader_defines);
    std::vector<uint32_t> writer_ct_args = {};
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);
    const auto writer_kernel_id = operations::CreateWriteKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/writer_moreh_softmax_c_large.cpp",
        all_cores,
        writer_ct_args,
        writer_defines);

    auto outer_stride = Ht * Wt;
    for (int i = dim; i < shape.rank() - 2; i++) {
        outer_stride *= shape[i];
    }
    const auto dim_size = shape[dim];
    const auto inner_size = outer_stride / dim_size;

    // Kernel defines
    std::map<std::string, std::string> compute_defines;
    compute_defines["SOFTMAX"] = "1";
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Comput kernel
    operations::CreateComputeKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_c_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, dim_size}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, dim_size}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Runtime Args
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

        const std::vector<uint32_t> reader_args = {
            input.buffer()->address(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size};

        const std::vector<uint32_t> writer_args = {
            output_tensor.buffer()->address(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

}  // namespace ttnn::prim
