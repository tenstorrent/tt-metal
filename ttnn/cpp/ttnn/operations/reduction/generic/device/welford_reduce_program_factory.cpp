// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/reduction/generic/device/welford_reduce_device_operation_types.hpp"
#include "welford_reduce_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

WelfordReduceProgramFactory::cached_program_t WelfordReduceProgramFactory::create(
    const WelfordReduceParams& operation_attributes, const Tensor& tensor_arg, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Shape& padded_shape = tensor_arg.padded_shape();
    const Shape& logical_shape = tensor_arg.logical_shape();

    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];
    uint32_t W_padded = padded_shape[-1];
    uint32_t H_padded = padded_shape[-2];
    TT_FATAL(
        H_padded > 0 && W_padded > 0,
        "Padded H and W dimensions must be non-zero, got H_padded={}, W_padded={}",
        H_padded,
        W_padded);
    // Product of all dimensions except the last two (H, W).
    // Named NC by convention even though tensor may have arbitrary rank.
    uint32_t NC = tensor_arg.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = tensor_arg.tensor_spec().tile().get_height();
    const uint32_t tile_width = tensor_arg.tensor_spec().tile().get_width();

    uint32_t Wt = W_padded / tile_width;
    uint32_t Ht = H_padded / tile_height;
    uint32_t HtWt = Ht * Wt;

    const bool reduce_w = (operation_attributes.reduce_dim == ReduceOpDim::W);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_arg.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    // Scalar datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = tensor_arg.device();

    // Work division:
    // - W-reduce: Work is split by rows of the tile grid (NC * Ht work units).
    //   Each core processes one or more complete rows of Wt tiles.
    //   Each row of tiles is a contiguous block of Wt tiles along the W dimension (the compute kernel
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

    // - H-reduce: Similar to above, but for the H dimension. Work is split by columns of
    //   the tile grid (NC * Wt work units).
    //   Each core processes one or more complete columns of Ht tiles → 1 output tile per column.

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_work_units = reduce_w ? NC * Ht : NC * Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_work_units_per_core_group_1, num_work_units_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_work_units);
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_work_units);
    }

    CBIndex input_cb_index = CBIndex::c_0;
    uint32_t input_tiles_per_cb = 2;
    tt_metal::CircularBufferConfig input_cb_config =
        tt_metal::CircularBufferConfig(
            input_tiles_per_cb * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);

    CBIndex scalar_cb_index = CBIndex::c_2;

    tt_metal::CircularBufferConfig scalar_cb_config =
        tt_metal::CircularBufferConfig(scalar_single_tile_size, {{scalar_cb_index, scalar_cb_data_format}})
            .set_page_size(scalar_cb_index, scalar_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, scalar_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t output_tiles_per_cb = 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            output_tiles_per_cb * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // cb_var (c_19): W-reduce only -- scratch buffer for variance tile between
    // the two transpose steps (Welford produces row-oriented results that must
    // be transposed back to column orientation).
    if (reduce_w) {
        CBIndex scratch_cb_index = CBIndex::c_19;
        // It stores temporary data from the DST register, so data format is the same as the DST register.
        tt::DataFormat scratch_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        uint32_t scratch_single_tile_size = tt::tile_size(scratch_cb_data_format);
        tt_metal::CircularBufferConfig scratch_cb_config =
            tt_metal::CircularBufferConfig(scratch_single_tile_size, {{scratch_cb_index, scratch_cb_data_format}})
                .set_page_size(scratch_cb_index, scratch_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, scratch_cb_config);
    }

    // cb_scaled (c_20): both W-reduce and H-reduce need this when do_scale is
    // true.  mul_tiles_bcast_scalar_init_short reconfigures the FPU math
    // pipeline, so the scaled result must be packed to this intermediate CB
    // and read back before the SFPU Welford operation.
    bool do_scale = (operation_attributes.scalar != 1.0f);
    if (do_scale) {
        CBIndex scaled_cb_index = CBIndex::c_20;
        tt_metal::CircularBufferConfig scaled_cb_config =
            tt_metal::CircularBufferConfig(input_single_tile_size, {{scaled_cb_index, input_cb_data_format}})
                .set_page_size(scaled_cb_index, input_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, scaled_cb_config);
    }

    bfloat16 bfloat_scalar_value = bfloat16::truncate(operation_attributes.scalar);
    uint32_t packed_scalar_value = pack_two_bfloat16_into_uint32({bfloat_scalar_value, bfloat_scalar_value});

    tt_metal::Buffer* input_buffer = tensor_arg.buffer();
    tt_metal::Buffer* output_buffer = tensor_return_value.buffer();

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);

    // --- Reader kernel ---
    tt_metal::KernelHandle reader_kernel_id;
    if (reduce_w) {
        // W-reduce: sequential reader reads tiles row by row
        std::vector<uint32_t> reader_compile_time_args = {packed_scalar_value};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reduce_defines));
    } else {
        // H-reduce: column-partitioned reader reads tiles column by column
        std::vector<uint32_t> reader_compile_time_args = {Ht, Wt, HtWt, /*row_chunk=*/1, packed_scalar_value};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    // --- Writer kernel (same for both paths) ---
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, reduce_defines));

    // --- Compute kernel ---
    bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);

    // W-reduce compile args: {Wt, W, tile_width, do_scale, correction, is_std}
    // H-reduce compile args: {Ht, H, tile_height, do_scale, correction, is_std}
    std::vector<uint32_t> compute_compile_args = {
        reduce_w ? Wt : Ht,
        reduce_w ? W : H,
        reduce_w ? tile_width : tile_height,
        static_cast<uint32_t>(do_scale),
        static_cast<uint32_t>(operation_attributes.correction),
        static_cast<uint32_t>(is_std),
    };

    const std::string compute_kernel =
        reduce_w ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp"
                 : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";

    tt_metal::KernelHandle compute_kernel_id_group_1 = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_args,
            .defines = reduce_defines});

    tt_metal::KernelHandle compute_kernel_id_group_2 = 0;
    if (!core_group_2.ranges().empty()) {
        compute_kernel_id_group_2 = tt_metal::CreateKernel(
            program,
            compute_kernel,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_args,
                .defines = reduce_defines});
    }

    // --- Runtime args per core ---
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

    if (reduce_w) {
        // W-reduce: each work unit is one row of Wt tiles
        uint32_t input_tiles_offset = 0;
        uint32_t output_tiles_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_work_units_per_core = 0;
            if (core_group_1.contains(core)) {
                num_work_units_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_work_units_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            uint32_t num_output_tiles_per_core = num_work_units_per_core;
            tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {tensor_arg.buffer()->address(), num_input_tiles_per_core, input_tiles_offset});
            tt_metal::SetRuntimeArgs(
                program,
                core_group_1.contains(core) ? compute_kernel_id_group_1 : compute_kernel_id_group_2,
                core,
                {num_work_units_per_core});
            tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {tensor_return_value.buffer()->address(), num_output_tiles_per_core, output_tiles_offset});
            input_tiles_offset += num_input_tiles_per_core;
            output_tiles_offset += num_output_tiles_per_core;
        }
    } else {
        // H-reduce: each work unit is one column of Ht tiles
        // Reader args: {src_addr, col_start_tile_id, curr_col_in_batch, num_cols}
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            if (core_group_1.contains(core)) {
                num_cols_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {tensor_arg.buffer()->address(),
                 (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),
                 num_cols_read % Wt,
                 num_cols_per_core});
            tt_metal::SetRuntimeArgs(
                program,
                core_group_1.contains(core) ? compute_kernel_id_group_1 : compute_kernel_id_group_2,
                core,
                {num_cols_per_core});
            tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {tensor_return_value.buffer()->address(), num_cols_per_core, num_cols_read});
            num_cols_read += num_cols_per_core;
        }
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
