// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/universal_kernel.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

using namespace tt::constants;
using uint32_t = std::uint32_t;
namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reduce_multi_core_h(
    const Tensor& a,
    Tensor& output,
    ReduceOpMath reduce_op,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    float scaler) {
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_cols = NC * Wt;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);

    tt::DataFormat src0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat dst_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    uint32_t chunk_size = ttnn::get_dest_reg_count(compute_kernel_config);

    tt::DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    uint32_t scaler_cb_index = CBIndex::c_4;
    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(1 * scaler_single_tile_size, {{scaler_cb_index, scaler_cb_data_format}})
            .set_page_size(scaler_cb_index, scaler_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    MathConfig math_config = MathConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    auto universal_config_base = UniversalKernelConfigBuilder(math_config)
                                     .set_defines(reduce_op_utils::get_defines(reduce_op, ReduceOpDim::H))
                                     .add_compile_time_arg("Ht", Ht)
                                     .add_compile_time_arg("Wt", Wt)
                                     .add_compile_time_arg("HtWt", HtWt)
                                     .add_compile_time_arg("row_chunk", chunk_size)
                                     .add_compile_time_arg("packed_scaler_value", packed_scaler_value)
                                     .add_runtime_arg("start_write_page_id", 0)
                                     .add_runtime_arg("col_start_tile_id", 0)
                                     .add_runtime_arg("curr_col_in_batch", 0)
                                     .add_buffer("src0", a.buffer(), src0_data_format)
                                     .add_buffer("out", output.buffer(), dst_data_format);

    size_t start_write_page_id_idx = universal_config_base.get_runtime_arg_idx("start_write_page_id");
    size_t col_start_tile_id_idx = universal_config_base.get_runtime_arg_idx("col_start_tile_id");
    size_t curr_col_in_batch_idx = universal_config_base.get_runtime_arg_idx("curr_col_in_batch");
    size_t buffer_addresses_start_idx = universal_config_base.buffer_addresses_start_runtime_arg_idx();

    auto universal_config_group_1 = universal_config_base;
    universal_config_group_1.add_compile_time_arg("num_cols_per_core_group", num_cols_per_core_group_1);
    auto universal_config_group_2 = universal_config_base;
    universal_config_group_2.add_compile_time_arg("num_cols_per_core_group", num_cols_per_core_group_2);

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
        core_group_1,
        universal_config_group_1);

    if (!core_group_2.ranges().empty()) {
        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
            core_group_2,
            universal_config_group_2);
    }

    const auto& cores =
        grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        tt_metal::UpdateRuntimeArgs(program, core, [=](uint32_t* runtime_args) {
            runtime_args[start_write_page_id_idx] = num_cols_read;
            runtime_args[col_start_tile_id_idx] = num_cols_read / Wt * HtWt + num_cols_read % Wt;
            runtime_args[curr_col_in_batch_idx] = num_cols_read % Wt;
        });

        num_cols_read += num_cols_per_core;
    }

    auto override_runtime_arguments_callback = [cores = cores, buffer_addresses_start_idx](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer_address = input_tensors.at(0).buffer()->address();
        auto dst_buffer_address = output_tensors.at(0).buffer()->address();
        for (const auto& core : cores) {
            tt_metal::UpdateRuntimeArgs(program, core, [=](uint32_t* runtime_args) {
                runtime_args[buffer_addresses_start_idx] = src_buffer_address;
                runtime_args[buffer_addresses_start_idx + 1] = dst_buffer_address;
            });
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
