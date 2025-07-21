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
MorehMeanOperation::MorehMeanHFactory::cached_program_t MorehMeanOperation::MorehMeanHFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto input = tensor_args.input;
    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);
    const auto& shape = input.padded_shape();

    auto device = input.device();
    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    uint32_t W = shape[-1], H = shape[-2];

    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    // check mask for h-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_H = input_shape_without_padding[-2];
    const bool do_mask_h = (origin_H % constants::TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_H % constants::TILE_HEIGHT : constants::TILE_HEIGHT;

    auto program = CreateProgram();

    auto units_to_divide = input.physical_volume() / W / H * Wt;

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, units_to_divide);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    // create circular buffers
    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CBIndex::c_0, num_input_tiles},                   // input
            {CBIndex::c_2, 1},                                 // scaler
            {CBIndex::c_3, 1},                                 // mask
            {CBIndex::c_24, 1, fp32_dest_acc_en_data_format},  //
            {CBIndex::c_25, 1},                                //
            {CBIndex::c_16, 1},                                // output
        });

    float scaler = 1.0f / origin_H;
    auto bfloat_scaler_value = *(new class bfloat16(scaler));
    auto packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(input)), Ht, Wt, HtWt, packed_scaler_value};

    std::map<std::string, std::string> reader_defines;
    reader_defines["REDUCE_SCALER"] = "1";
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
    }
    const auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_h.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(CBIndex::c_16), static_cast<uint32_t>(is_dram(output))};

    const auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_unary_interleaved_start_id.cpp",
        all_cores,
        writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ///////////////////////////////////////////////////////////////////////////
    std::string compute_kernel_name = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp";
    auto reduce_op = ReduceOpMath::SUM;
    auto reduce_dim = ReduceOpDim::H;
    std::map<std::string, std::string> compute_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = 1;
        unpack_to_dest_mode[tt::CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        Ht,                      // Ht
        units_per_core_group_1,  // Wt
        1,                       // NC
        origin_H};
    std::vector<uint32_t> compute_kernel_args_group_2 = {
        Ht,                      // Ht
        units_per_core_group_2,  // Wt
        1,                       // NC
        origin_H};

    auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_name,
        {
            {core_group_1, units_per_core_group_1, compute_kernel_args_group_1},
            {core_group_2, units_per_core_group_2, compute_kernel_args_group_2},
        },
        ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .defines = compute_defines});

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core = 0;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             tile_offset / Wt * HtWt + tile_offset % Wt,
             tile_offset % Wt,
             units_per_core,
             mask_h});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                units_per_core,  // number of tiles to write
                tile_offset      // output tile start index
            });
        tile_offset += units_per_core;
    }
    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}
void MorehMeanOperation::MorehMeanHFactory::override_runtime_arguments(
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
