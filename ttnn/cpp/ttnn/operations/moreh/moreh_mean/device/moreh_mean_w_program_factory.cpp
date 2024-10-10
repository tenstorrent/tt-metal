// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "common/bfloat16.hpp"
#include "moreh_mean_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_mean {
MorehMeanOperation::MorehMeanWFactory::cached_program_t MorehMeanOperation::MorehMeanWFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::operations::primary;

    auto input = tensor_args.input;
    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);
    const auto shape = input.get_shape();

    auto device = input.device();
    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    uint32_t W = shape.value[-1], H = shape.value[-2];
    uint32_t HW = H * W;

    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;

    // check mask for w-dim
    const auto input_shape_without_padding = shape.value.without_padding();
    const auto origin_W = input_shape_without_padding[-1];
    const bool do_mask_w = (origin_W % constants::TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % constants::TILE_WIDTH : constants::TILE_WIDTH;

    auto program = CreateProgram();

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto units_to_divide = input.volume() / W / H * Ht;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    // create circular buffers
    tt::DataFormat data_format = datatype_to_dataformat_converter(input.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, num_input_tiles},                        // input
            {CB::c_in2, 1},                                      // scalar
            {CB::c_in3, 1},                                      // mask
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},  //
            {CB::c_intermed1, 1},                                //
            {CB::c_out0, 1},                                     // output
        });

    float scaler = 1.0f / origin_W;
    auto bfloat_scaler_value = *(new class bfloat16(scaler));
    auto packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(is_dram(input)), packed_scaler_value};

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(CB::c_out0), static_cast<uint32_t>(is_dram(output))};

    std::map<string, string> reader_defines{};
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }
    const auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_w.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);

    const auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_unary_interleaved_start_id.cpp",
        all_cores,
        writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ///////////////////////////////////////////////////////////////////////////
    string compute_kernel_name = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_w.cpp";
    auto reduce_op = ReduceOpMath::SUM;
    auto reduce_dim = ReduceOpDim::W;
    std::map<string, string> compute_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }
    vector<uint32_t> compute_kernel_args_group_1 = {
        units_per_core_group_1,  // Ht
        Wt,                      // Wt
        1,                       // NC
        origin_W,
    };
    vector<uint32_t> compute_kernel_args_group_2 = {
        units_per_core_group_2,  // Ht
        Wt,                      // Wt
        1,                       // NC
        origin_W,
    };
    vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

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

    uint32_t out_dim_divider = Wt;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = units_per_core * Wt;
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
    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

void MorehMeanOperation::MorehMeanWFactory::override_runtime_arguments(
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
