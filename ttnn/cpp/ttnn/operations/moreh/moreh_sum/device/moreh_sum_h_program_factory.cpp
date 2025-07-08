// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_sum_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_sum {
MorehSumOperation::MorehSumHFactory::cached_program_t MorehSumOperation::MorehSumHFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto input = tensor_args.input;
    const auto& output = output_tensor;

    auto memory_config = operation_attributes.memory_config;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    tt::tt_metal::ReduceOpMath reduce_op = tt::tt_metal::ReduceOpMath::SUM;
    tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::H;
    float scaler = 1.0f;

    const auto& shape = input.padded_shape();
    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);

    uint32_t Wt = W / tt::constants::TILE_WIDTH;
    uint32_t Ht = H / tt::constants::TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    // check mask for h-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_H = input_shape_without_padding[-2];
    const bool do_mask_h = (origin_H % tt::constants::TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_H % tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat mask_h_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t mask_h_single_tile_size = tt::tt_metal::detail::TileSize(mask_h_cb_data_format);
    tt::DataFormat intermed_cb_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat intermed1_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t intermed_single_tile_size = tt::tt_metal::detail::TileSize(intermed_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cols = other_dims_product * Wt;

    const CoreRange all_core_range(
        {0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        split_work_to_cores_wt_core_range(all_core_range, num_cols);

    std::string compute_kernel_name =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp";

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    CBHandle cb_src1 = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t scaler_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(1 * scaler_single_tile_size, {{scaler_cb_index, scaler_cb_data_format}})
            .set_page_size(scaler_cb_index, scaler_single_tile_size);
    auto cb_scaler = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    tt::tt_metal::CircularBufferConfig cb_mask_h_config =
        tt::tt_metal::CircularBufferConfig(mask_h_single_tile_size, {{tt::CBIndex::c_3, mask_h_cb_data_format}})
            .set_page_size(tt::CBIndex::c_3, mask_h_single_tile_size);
    auto cb_mask_h = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_mask_h_config);

    tt::tt_metal::CircularBufferConfig cb_intermed0_config =
        tt::tt_metal::CircularBufferConfig(intermed_single_tile_size, {{tt::CBIndex::c_24, intermed_cb_data_format}})
            .set_page_size(tt::CBIndex::c_24, intermed_single_tile_size);
    auto cb_intermed0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    tt::tt_metal::CircularBufferConfig cb_intermed1_config =
        tt::tt_metal::CircularBufferConfig(intermed_single_tile_size, {{tt::CBIndex::c_25, intermed1_cb_data_format}})
            .set_page_size(tt::CBIndex::c_25, intermed_single_tile_size);
    auto cb_intermed1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::KernelHandle reader_kernel_id;
    auto bfloat_scaler_value = *(new class bfloat16(scaler));
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram, Ht, Wt, HtWt, packed_scaler_value};

    std::map<std::string, std::string> reader_defines;
    reader_defines["REDUCE_SCALER"] = "1";
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
    }
    reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_sum_h.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    tt::tt_metal::KernelHandle writer_kernel_id;

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_sum_h.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    std::map<std::string, std::string> reduce_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        reduce_defines["FP32_DEST_ACC_EN"] = "1";
    }

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        Ht,                         // Ht
        num_cols_per_core_group_1,  // Wt
        1,                          // NC
        origin_H};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto reduce_compute_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            Ht,                         // Ht
            num_cols_per_core_group_2,  // Wt
            1,                          // NC
            origin_H};

        auto reduce_compute_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            compute_kernel_name,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = reduce_defines});
    }

    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_cols_read / Wt * HtWt + num_cols_read % Wt,
             num_cols_read % Wt,
             num_cols_per_core,
             mask_h});

        tt::tt_metal::SetRuntimeArgs(
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

void MorehSumOperation::MorehSumHFactory::override_runtime_arguments(
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
    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_sum
