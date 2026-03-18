// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "welford_reduce_program_factory.hpp"
#include "ttnn/operations/reduction/generic/device/welford_reduce_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cmath>

namespace ttnn::prim {

WelfordReduceProgramFactory::cached_program_t WelfordReduceProgramFactory::create(
    const WelfordReduceParams& operation_attributes, const Tensor& tensor_arg, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Shape& shape = tensor_arg.padded_shape();

    // TODO: Add support for more dimensions.
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];
    uint32_t NC = 1;
    const uint32_t tile_height = tensor_arg.tensor_spec().tile().get_height();
    const uint32_t tile_width = tensor_arg.tensor_spec().tile().get_width();

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_arg.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = tensor_arg.device();

// Work division:
// - Work is split by rows of the tile grid. Each core processes one or more complete rows;
//   each row is a contiguous block of Wt tiles along the W dimension (the compute kernel
//   reduces each row of tiles to one output tile).
// - Example: 4D tensor with shape (N=2, C=1, H=64, W=128) and assuming 32x32 tile size.
//     Wt = 4, Ht = 2
//     Tile grid for each (N,C) slice (2 rows, 4 tiles per row):
//
//          W (tile index) 
//          0    1    2    3
//     H 0 [0]  [1]  [2]  [3]   ← row 0 of the tile grid (4 tiles → reduce to 1)
//       1 [4]  [5]  [6]  [7]   ← row 1 of the tile grid (4 tiles → reduce to 1)
//
//     The minimum any core will process is Wt = 4 tiles (i.e. one row of the tile grid).
//     There are Ht = 2 rows of tiles for each (N,C) slice. Since there are N*C = 2 slices,
//     in total, there are N * C * Ht = 2 * 1 * 2 = 4 rows of tiles to be distributed among cores.

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_rows_of_tiles = NC * Ht;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_of_tiles_per_core_group_1, num_rows_of_tiles_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_of_tiles_per_core_group_1, num_rows_of_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows_of_tiles);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_of_tiles_per_core_group_1, num_rows_of_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows_of_tiles);
    }

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * scaler_single_tile_size, {{CBIndex::c_2, scaler_cb_data_format}})
            .set_page_size(CBIndex::c_2, scaler_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bfloat16 bfloat_scaler_value = bfloat16::truncate(operation_attributes.scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    tt_metal::Buffer* src_buffer = tensor_arg.buffer();
    std::vector<uint32_t> reader_compile_time_args = {packed_scaler_value};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    tt_metal::Buffer* dst_buffer = tensor_return_value.buffer();
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reduce_defines));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, reduce_defines));

    std::vector<uint32_t> compute_compile_args = {
        Wt,
        W,
        tile_width,
    };

    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_var_std_welford.cpp");

    tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_args,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        tt_metal::CreateKernel(
            program,
            compute_kernel,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_args,
                .defines = reduce_defines});
    }

    uint32_t out_dim_divider = Wt;
    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_of_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_of_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                num_tensor_tiles_per_core
            });

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void WelfordReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const WelfordReduceParams& /*operation_attributes*/,
    const Tensor& tensor_arg,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    auto* src_dram_buffer = tensor_arg.buffer();
    auto* dst_dram_buffer = tensor_return_value.buffer();

    auto& reader_runtime_args_by_core =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id);
    auto& writer_runtime_args_by_core =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id);
    for (const auto& core : cached_program.shared_variables.cores) {
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
