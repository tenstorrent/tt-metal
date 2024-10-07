// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

using namespace tt::constants;
using uint32_t = std::uint32_t;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reduce_multi_core_w(
    const Tensor &a,
    Tensor &output,
    ReduceOpMath reduce_op,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config,
    float scaler) {
    const auto shape = a.get_legacy_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);

    auto program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt_metal::detail::TileSize(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_rows = NC * Ht;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(num_input_tiles * scaler_single_tile_size, {{CB::c_in2, scaler_cb_data_format}})
            .set_page_size(CB::c_in2, scaler_single_tile_size);
    auto cb_scaler = tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bfloat16 bfloat_scaler_value = bfloat16(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    tt_metal::Buffer *src_buffer = a.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, packed_scaler_value};
    tt_metal::Buffer *dst_buffer = output.buffer();
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    std::map<string, string> reduce_defines = reduce_op_utils::get_defines(reduce_op, ReduceOpDim::W);
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reduce_defines));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, reduce_defines));

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_rows_per_core_group_1,  // Ht
        Wt,                         // Wt
        1,                          // NC
    };

    auto reduce_compute_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args_group_1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_rows_per_core_group_2,  // Ht
            Wt,                         // Wt
            1,                          // NC
        };

        auto reduce_compute_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args_group_2,
                .defines = reduce_defines});
    }

    uint32_t out_dim_divider = Wt;
    const auto &cores =
        grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord &core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                a.buffer()->address(),
                num_tensor_tiles_per_core,
                num_tiles_read  // tile index of row to start reading from
            });

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                num_tiles_read / out_dim_divider              // output tile start index
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, cores](
                                              const ProgramHandle program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);

        auto &reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        auto &writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
        for (const auto &core : cores) {
            {
                auto &runtime_args = reader_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_dram_buffer->address();
            }

            {
                auto &runtime_args = writer_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_dram_buffer->address();
            }
        }
    };

    return {program, override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
