// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_sum_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sum {
MorehSumOperation::MorehSumWIntFactory::cached_program_t MorehSumOperation::MorehSumWIntFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto input = tensor_args.input;
    auto output = output_tensor;

    auto output_mem_config = operation_attributes.output_mem_config;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    tt::tt_metal::Device* device{input.device()};
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format{datatype_to_dataformat_converter(output.get_dtype())};
    const auto shape{input.get_padded_shape()};

    const auto [W, H, other_dims_product] = tt::operations::primary::extract_spatial_dims(shape);
    uint32_t Wt{W / tt::constants::TILE_WIDTH};
    uint32_t Ht{H / tt::constants::TILE_HEIGHT};
    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;
    auto num_rows{other_dims_product * Ht};

    // check mask for w-dim
    const auto input_shape_without_padding{input.get_logical_shape()};
    const auto origin_W{input_shape_without_padding[-1]};
    const bool do_mask_w{(origin_W % tt::constants::TILE_WIDTH) != 0};
    const auto mask_w{do_mask_w ? origin_W % tt::constants::TILE_WIDTH : tt::constants::TILE_WIDTH};

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
    log_debug(tt::LogOp, "do_mask_w {} mask_w {}", do_mask_w, mask_w);

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
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(
        tt::LogOp,
        "num_tiles {}, num_rows {}, num_rows_per_core_group_1 {}, num_rows_per_core_group_2 {}",
        num_tiles,
        num_rows,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CB::c_in0, in0_t},              // input
            {tt::CB::c_in1, in1_t},              // mask
            {tt::CB::c_intermed0, intermed0_t},  // accumalated sum
            {tt::CB::c_out0, out0_t},            // output
        });
    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(tt::operations::primary::is_dram(input))};
    std::map<string, string> reader_defines{};
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(tt::operations::primary::is_dram(output))};
    const auto reader_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/reader_moreh_int_sum_w.cpp"};
    const auto writer_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_int_sum_w.cpp"};
    const auto reader_kernel_id{tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines)};
    const auto writer_kernel_id{
        tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args)};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,  // num_rows
        Wt,                         // Wt
        origin_W};

    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file{
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_int_sum_w.cpp"};
    const auto compute_kernel_1_id = tt::operations::primary::CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_rows_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id{std::nullopt};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,  // num_rows
            Wt,                         // Wt
            origin_W};
        compute_kernel_2_id = tt::operations::primary::CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    uint32_t out_dim_divider{Wt};
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core{0};
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        uint32_t num_tensor_tiles_per_core{num_rows_per_core * Wt};
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_tensor_tiles_per_core,
             tile_offset,  // tile index of row to start reading from
             mask_w});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                tile_offset / out_dim_divider                 // output tile start index
            });

        tile_offset += num_tensor_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}

void MorehSumOperation::MorehSumWIntFactory::override_runtime_arguments(
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
    auto src_dram_buffer{tensor_args.input.buffer()};
    auto dst_dram_buffer{tensor_return_value.buffer()};

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
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
