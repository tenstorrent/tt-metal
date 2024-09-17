// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_sum_nc_impl(const Tensor &input, const Tensor &output, int64_t dim,const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = input.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    const auto single_tile_size = detail::TileSize(cb_data_format);

    const auto &input_shape = input.get_legacy_shape();
    const auto &input_shape_without_padding = input_shape.without_padding();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] = extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(dim));
    const auto num_reduce_input_tile = input_shape[dim];
    const auto num_output_tiles = output.volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);

    log_debug(LogOp, "reduce_tile_size {} inner_tile_size {} Ht {} Wt {}", reduce_tile_size, inner_tile_size, Ht, Wt);
    log_debug(
        LogOp, "dim {} num_reduce_input_tile {} num_output_tiles {}", dim, num_reduce_input_tile, num_output_tiles);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;        // input
    const uint32_t in1_t = 1;        // zero
    const uint32_t intermed0_t = 1;  // accumulated sum
    const uint32_t out0_t = 2;       // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},              // input
            {CB::c_in1, in1_t},              // zero
            {CB::c_intermed0, intermed0_t, (fp32_dest_acc_en) ? tt::DataFormat::Float32: cb_data_format},
            {CB::c_out0, out0_t},            // output
        });
    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args =
             {static_cast<uint32_t>(is_dram(input))} ;
    std::map<string, string> reader_defines;
    reader_defines["USE_FPU"] = "1";
    std::vector<uint32_t> writer_compile_time_args =
             {static_cast<uint32_t>(is_dram(output))} ;
    const auto reader_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_nc_impl/kernels/reader_moreh_sum_nc.cpp";
    const auto writer_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_nc_impl/kernels/writer_moreh_sum_nc.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1, num_reduce_input_tile};
    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    // set preserve_fp32_precision to the same value as fp32_dest_acc_en
    // bool preserve_fp32_precision = fp32_dest_acc_en;
    vector<PreserveFP32Target> preserve_fp32_precision(NUM_CIRCULAR_BUFFERS, PreserveFP32Target::Disabled);
    auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_nc_impl/kernels/moreh_sum_nc.cpp";
    if (device->arch() == tt::ARCH::GRAYSKULL) {
        compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_nc_impl/kernels/moreh_sum_nc_gs.cpp";
    }
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        preserve_fp32_precision);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2, num_reduce_input_tile};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            preserve_fp32_precision);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_reduce_input_tile,
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(dim),
             reduce_tile_size,
             inner_tile_size
             });

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            { output.buffer()->address(), num_tiles_per_core, tile_offset
            });

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *input_buffer = input_tensors.at(0).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
