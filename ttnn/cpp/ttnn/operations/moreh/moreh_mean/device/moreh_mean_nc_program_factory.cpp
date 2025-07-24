// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_mean_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_mean {
MorehMeanOperation::MorehMeanNCFactory::cached_program_t MorehMeanOperation::MorehMeanNCFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto input = tensor_args.input;
    auto dim = operation_attributes.dim;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    auto device = input.device();
    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    const auto cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& input_shape_without_padding = input.logical_shape();

    const auto Ht = input_shape[-2] / constants::TILE_HEIGHT;
    const auto Wt = input_shape[-1] / constants::TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];

    const auto rank = input_shape.rank();
    auto input_tile_stride = HtWt;
    for (int i = dim + 1; i < rank - 2; i++) {
        input_tile_stride *= input_shape[i];
    }

    uint32_t inner_size = 1;
    for (int i = dim + 1; i < rank - 2; i++) {
        inner_size *= input_shape[i];
    }

    const auto units_to_divide = output.physical_volume() / constants::TILE_HW;

    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, units_to_divide);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, 2},   // input
            {CBIndex::c_1, 1},   // zero
            {CBIndex::c_2, 1},   // scaler
            {CBIndex::c_24, 1},  // accumulated mean
            {CBIndex::c_16, 2},  // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    const auto reader_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_nc.cpp";
    const auto writer_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_nc.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp";
    std::map<std::string, std::string> compute_defines;
    const std::vector<uint32_t> compute_args_group_1{units_per_core_group_1};
    const std::vector<uint32_t> compute_args_group_2{units_per_core_group_2};

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, units_per_core_group_1, compute_args_group_1},
            {core_group_2, units_per_core_group_2, compute_args_group_2},
        },
        ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .defines = compute_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};

        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_ids[0], core, {num_reduce_input_tile, units_per_core});
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_ids[1], core, {num_reduce_input_tile, units_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_reduce_input_tile,
             units_per_core,
             input_tile_stride,
             tile_offset,
             static_cast<uint32_t>(is_dram(input)),
             HtWt,
             inner_size});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(), units_per_core, tile_offset, static_cast<uint32_t>(is_dram(output))});

        tile_offset += units_per_core;
    }
    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

void MorehMeanOperation::MorehMeanNCFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;

    auto src_buffer_address = tensor_args.input.buffer()->address();
    auto dst_buffer_address = tensor_return_value.buffer()->address();

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer_address;
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_mean
