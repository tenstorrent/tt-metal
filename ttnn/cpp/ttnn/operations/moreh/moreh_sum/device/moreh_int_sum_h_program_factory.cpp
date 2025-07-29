// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_sum_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sum {
MorehSumOperation::MorehSumHIntFactory::cached_program_t MorehSumOperation::MorehSumHIntFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto input = tensor_args.input;
    const auto& output = output_tensor;

    auto memory_config = operation_attributes.memory_config;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    tt::tt_metal::IDevice* device{input.device()};
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};

    const auto cb_data_format{datatype_to_dataformat_converter(output.dtype())};
    const auto& shape{input.padded_shape()};

    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);
    uint32_t Wt{W / tt::constants::TILE_WIDTH};
    uint32_t Ht{H / tt::constants::TILE_HEIGHT};
    uint32_t HtWt{Ht * Wt};
    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    auto num_cols{other_dims_product * Wt};

    // check mask for h-dim
    const auto& input_shape_without_padding{input.logical_shape()};
    const auto origin_H{input_shape_without_padding[-2]};
    const bool do_mask_h{(origin_H % tt::constants::TILE_HEIGHT) != 0};
    const auto mask_h{do_mask_h ? origin_H % tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    if (!fp32_dest_acc_en) {
        log_warning(tt::LogOp, "fp32_dest_acc_en should be set for integer sum");
        fp32_dest_acc_en = true;
    }
    log_debug(tt::LogOp, "do_mask_h {} mask_h {}", do_mask_h, mask_h);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid{device->compute_with_storage_grid_size()};
    const auto num_cores_y{grid.y};

    const uint32_t in0_t{2};        // input
    const uint32_t in1_t{1};        // mask
    const uint32_t intermed0_t{1};  // accumalated sum
    const uint32_t out0_t{2};       // output
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_cols);

    log_debug(
        tt::LogOp,
        "num_tiles {}, num_cols {}, num_cols_per_core_group_1 {}, num_cols_per_core_group_2 {}",
        num_tiles,
        num_cols,
        num_cols_per_core_group_1,
        num_cols_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},         // input
            {tt::CBIndex::c_1, in1_t},         // mask
            {tt::CBIndex::c_24, intermed0_t},  // accumalated sum
            {tt::CBIndex::c_16, out0_t},       // output
        });
    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(is_dram(input)), Ht, Wt};
    std::map<std::string, std::string> reader_defines{};
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
    }
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(output))};
    const auto reader_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_int_sum_h.cpp"};
    const auto writer_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp"};
    const auto reader_kernel_id{
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines)};
    const auto writer_kernel_id{CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args)};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{
        num_cols_per_core_group_1,  // num_cols
        Ht,                         // Ht
        origin_H};

    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_int_sum_h.cpp"};
    CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id{std::nullopt};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_cols_per_core_group_2,  // num_cols
            Ht,                         // Ht
            origin_H};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }
    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core{0};
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_cols_read / Wt * HtWt + num_cols_read % Wt,
             num_cols_read % Wt,
             num_cols_per_core,
             mask_h});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_cols_per_core,  // number of tiles to write
                num_cols_read       // output tile start index
            });

        num_cols_read += num_cols_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}

void MorehSumOperation::MorehSumHIntFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    log_debug(tt::LogOp, "{}:{} args_callback ", __func__, __LINE__);
    auto src_dram_buffer = tensor_args.input.buffer();
    auto dst_dram_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_sum
