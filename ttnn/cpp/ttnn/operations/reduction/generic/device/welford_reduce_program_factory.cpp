// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "welford_reduce_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor WelfordReduceDeviceOperation::WelfordReduceProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_arg,
    tensor_return_value_t& tensor_return_value) {
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
    const bool reduce_h = (operation_attributes.reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (operation_attributes.reduce_dim == ReduceOpDim::HW);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);

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
    //     Wt = 4, Ht = 2, so there are in total 2 * 1 * 4 * 2 = 16 tiles.
    //     Tiles are stored in memory in row-major order: tile 0, tile 1, tile 2,..., tile 15.
    //     tile 0 corresponds to N = 0, C = 0, Ht = 0, Wt = 0, which is simply denoted as (0, 0, 0, 0).
    //     tile 1 corresponds to (0, 0, 0, 1), and so on.
    //     Tile grid for each (N,C) slice (2 rows, 4 tiles per row):
    //
    //           Wt (tile index)
    //           0    1    2    3
    //     Ht 0 [0]  [1]  [2]  [3]   ← row 0 of the tile grid (4 tiles → reduce to 1)
    //        1 [4]  [5]  [6]  [7]   ← row 1 of the tile grid (4 tiles → reduce to 1)
    //
    //     The minimum any core will process is Wt = 4 tiles (i.e. one row of the tile grid).
    //     There are Ht = 2 rows of tiles for each (N,C) slice. Since there are N*C = 2 slices,
    //     in total, there are N * C * Ht = 2 * 1 * 2 = 4 rows of tiles to be distributed among cores.

    // - H-reduce: Similar to above, but for the H dimension. Work is split by columns of
    //   the tile grid (NC * Wt work units).
    //   Each core processes one or more complete columns of Ht tiles → 1 output tile per column.
    //
    // - HW-reduce: Work is split by output elements.
    //   An "output element" is a single output tile, which contains one scalar value that is the result
    //   of reducing all dimensions that were requested to be reduced (other tile elements are padding).
    //   Each core produces one or more output elements.
    // - Example: 5D tensor (3, 4, 8, 64, 128), 32×32 tiles, reducing dims {2, 3, 4}.
    //     The host dispatch (generic_reductions.cpp) permutes all reduction dims to the end;
    //     here the permutation is identity since dims 2,3,4 are already trailing.
    //     The last two reduction dims (3,4) become H and W.  The extra reduction dim 2
    //     (size 8) folds into the NC batch → NC = 3 × 4 × 8 = 96, reduce_batch_size = 8.
    //
    //     NC slices are laid out in row-major order of the non-H/W dims:
    //       slice  0: (0,0,0)   slice  1: (0,0,1)  ...  slice  7: (0,0,7)
    //       slice  8: (0,1,0)   slice  9: (0,1,1)  ...  slice 15: (0,1,7)
    //       ...
    //       slice 88: (2,3,0)   slice 89: (2,3,1)  ...  slice 95: (2,3,7)
    //
    //     reduce_batch_size must equal 8 (the product of extra reduction dims) because
    //     each output element must fully reduce dim 2.  Slices 0–7 are the 8 values along
    //     dim 2 for (dim0=0, dim1=0); the writer Welford-combines all 8 and writes a final
    //     variance scalar.  A smaller reduce_batch_size (e.g. 2) would only combine 2 of
    //     the 8 slices, producing a partial result.  The writer applies Bessel's
    //     correction and the compute kernel applies sqrt for std, so the
    //     intermediate Welford state (mean, M2, count) is lost — there is no
    //     way to recombine those final scalars afterwards.
    //
    //     Total work units = NC / reduce_batch_size = 96 / 8 = 12
    //     (one per (dim0, dim1) pair: 3 × 4 = 12).

    const uint32_t reduce_batch_size = operation_attributes.reduce_batch_size;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / reduce_batch_size) : (NC * Wt));
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

    ProgramDescriptor desc;

    CBIndex input_cb_index = CBIndex::c_0;
    uint32_t input_tiles_per_cb = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_tiles_per_cb * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    CBIndex scalar_cb_index = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scalar_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scalar_cb_index),
            .data_format = scalar_cb_data_format,
            .page_size = scalar_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t output_tiles_per_cb = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tiles_per_cb * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // cb_var (c_19): W-reduce only -- scratch buffer for variance tile between
    // the two transpose steps (Welford produces row-oriented results that must
    // be transposed back to column orientation).
    if (reduce_w) {
        CBIndex scratch_cb_index = CBIndex::c_19;
        // It stores temporary data from the DST register, so data format is the same as the DST register.
        tt::DataFormat scratch_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        uint32_t scratch_single_tile_size = tt::tile_size(scratch_cb_data_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(scratch_cb_index),
                .data_format = scratch_cb_data_format,
                .page_size = scratch_single_tile_size,
            }}},
        });
    }

    // cb_scaled (c_20): only W-reduce needs this when do_scale is true.
    // transpose_wh_tile is an unpack operation that reads from a CB, not
    // DST, so the FPU mul result must be packed to this intermediate CB.
    // H and HW reduce don't need the transpose.
    bool do_scale = (operation_attributes.scalar != 1.0f);
    if (do_scale && reduce_w) {
        CBIndex scaled_cb_index = CBIndex::c_20;
        desc.cbs.push_back(CBDescriptor{
            .total_size = input_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(scaled_cb_index),
                .data_format = input_cb_data_format,
                .page_size = input_single_tile_size,
            }}},
        });
    }

    // cb_partial (c_21): HW-reduce only -- holds per-column mean+var tile pairs
    // from the compute kernel, consumed by the writer kernel.
    // Uses Float32 format to preserve precision from DST accumulators.
    if (reduce_hw) {
        CBIndex partial_cb_index = CBIndex::c_21;
        tt::DataFormat partial_cb_data_format = tt::DataFormat::Float32;
        uint32_t partial_single_tile_size = tt::tile_size(partial_cb_data_format);
        // Reserve space for 4 tiles to enable double buffering (since compute kernel packs 2 tiles at a time).
        desc.cbs.push_back(CBDescriptor{
            .total_size = 4 * partial_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(partial_cb_index),
                .data_format = partial_cb_data_format,
                .page_size = partial_single_tile_size,
            }}},
        });

        // cb_combined (c_22): HW-reduce only -- holds the combined scalar result
        // (one Float32 tile per output) written by the writer kernel after
        // W-combining all per-column partials and applying Bessel's correction.
        // The compute kernel reads this tile, applies sqrt_tile for std, and
        // re-packs it to cb_out in the correct output data format (the packer
        // hardware is required for BFLOAT8_B conversion).
        CBIndex combined_cb_index = CBIndex::c_22;
        tt::DataFormat combined_cb_data_format = tt::DataFormat::Float32;
        uint32_t combined_single_tile_size = tt::tile_size(combined_cb_data_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = combined_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(combined_cb_index),
                .data_format = combined_cb_data_format,
                .page_size = combined_single_tile_size,
            }}},
        });
    }

    tt_metal::Buffer* input_buffer = tensor_arg.buffer();
    tt_metal::Buffer* output_buffer = tensor_return_value.buffer();

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);
    reduce_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";

    // --- Reader kernel ---
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scalar);
    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};

    if (reduce_h || reduce_hw) {
        // H-reduce and HW-reduce: column-partitioned reader reads tiles column by column.
        // Welford processes one column at a time (SFPU can only track one running
        // mean/M2 state), so the reader must deliver tiles in strict column-major
        // order: all Ht tiles of column 0, then all Ht tiles of column 1, etc.
        std::vector<uint32_t> reader_compile_time_args = {Ht, Wt, HtWt, scaler_bits, /*use_welford=*/1};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
    } else {
        // W-reduce: sequential reader reads tiles row by row.
        std::vector<uint32_t> reader_compile_time_args = {scaler_bits};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
    }

    // --- Compute + Writer kernels ---
    bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);

    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};

    if (reduce_hw) {
        if (operation_attributes.correction) {
            TT_FATAL(
                H * W * reduce_batch_size >= 2,
                "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
                H * W * reduce_batch_size);
        }

        // HW-reduce: custom writer that combines partial stats and constructs output tile.
        std::vector<uint32_t> writer_compile_time_args = {
            Wt, W, tile_width, H, static_cast<uint32_t>(operation_attributes.correction), reduce_batch_size};
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "writer_welford_hw.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
        // Note: HW writer does not pass reduce_defines (matches original behavior).
    } else {
        // W-reduce and H-reduce: generic tile writer.
        std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
        writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    }

    std::vector<uint32_t> compute_compile_args;
    std::string compute_kernel;

    if (reduce_hw) {
        // HW-reduce compile args: {Ht, H, tile_height, Wt, do_scale, reduce_batch_size, is_std}
        compute_compile_args = {
            Ht,
            H,
            tile_height,
            Wt,
            static_cast<uint32_t>(do_scale),
            reduce_batch_size,
            static_cast<uint32_t>(is_std),
        };
        compute_kernel = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
    } else {
        if (operation_attributes.correction) {
            uint32_t reduce_size = reduce_w ? W : H;
            TT_FATAL(
                reduce_size >= 2,
                "Bessel's correction requires at least 2 elements along the reduction dimension, got {}",
                reduce_size);
        }

        // W-reduce compile args: {Wt, W, tile_width, do_scale, correction, is_std}
        // H-reduce compile args: {Ht, H, tile_height, do_scale, correction, is_std}
        compute_compile_args = {
            reduce_w ? Wt : Ht,
            reduce_w ? W : H,
            reduce_w ? tile_width : tile_height,
            static_cast<uint32_t>(do_scale),
            static_cast<uint32_t>(operation_attributes.correction),
            static_cast<uint32_t>(is_std),
        };
        compute_kernel = reduce_w
                             ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp"
                             : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
    }

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_compile_args;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        KernelDescriptor d;
        d.kernel_source = compute_kernel;
        d.source_type = KernelDescriptor::SourceType::FILE_PATH;
        d.core_ranges = core_group_2;
        d.compile_time_args = compute_compile_args;
        d.defines = {reduce_defines.begin(), reduce_defines.end()};
        d.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(d);
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
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_work_units_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_work_units_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            uint32_t num_output_tiles_per_core = num_work_units_per_core;
            reader_desc.emplace_runtime_args(core, {tensor_arg.buffer(), num_input_tiles_per_core, input_tiles_offset});
            (in_g1 ? compute_desc_g1 : *compute_desc_g2)
                .runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_work_units_per_core});
            writer_desc.emplace_runtime_args(
                core, {tensor_return_value.buffer(), num_output_tiles_per_core, output_tiles_offset});
            input_tiles_offset += num_input_tiles_per_core;
            output_tiles_offset += num_output_tiles_per_core;
        }
    } else if (reduce_hw) {
        // HW-reduce: each work unit is one output element, produced from
        // reduce_batch_size consecutive NC slices (Ht * Wt tiles each).
        // Reader uses the column-partitioned reader with
        // num_cols = Wt * nc_slices_per_core so the compute kernel's
        // for wt: for ht: loop order sees column-major tile order.
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        TT_FATAL(
            NC % reduce_batch_size == 0, "NC ({}) must be divisible by reduce_batch_size ({})", NC, reduce_batch_size);
        uint32_t nc_slice_offset = 0;
        uint32_t output_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_outputs_per_core = 0;
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_outputs_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_outputs_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            // Total NC slices this core will process
            uint32_t nc_slices_per_core = num_outputs_per_core * reduce_batch_size;
            // Reader: read all columns for all NC slices assigned to this core.
            uint32_t num_cols = Wt * nc_slices_per_core;
            uint32_t col_start_tile_id = nc_slice_offset * HtWt;
            reader_desc.emplace_runtime_args(
                core,
                {tensor_arg.buffer(),
                 col_start_tile_id,
                 /*curr_col_in_batch=*/0u,
                 num_cols});
            // Compute: runtime arg is total NC slices (not num_outputs).
            (in_g1 ? compute_desc_g1 : *compute_desc_g2)
                .runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{nc_slices_per_core});
            // Writer: runtime args are {dst_addr, NC_per_core, output_tile_start_id}.
            // NC_per_core is total NC slices; the writer uses reduce_batch_size
            // (compile-time) to determine how many to group per output.
            writer_desc.emplace_runtime_args(core, {tensor_return_value.buffer(), nc_slices_per_core, output_offset});
            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        // H-reduce: each work unit is one column of Ht tiles
        // Reader args: {src_addr, col_start_tile_id, curr_col_in_batch, num_cols}
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_cols_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            reader_desc.emplace_runtime_args(
                core,
                {tensor_arg.buffer(),
                 (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),
                 num_cols_read % Wt,
                 num_cols_per_core});
            (in_g1 ? compute_desc_g1 : *compute_desc_g2)
                .runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_cols_per_core});
            writer_desc.emplace_runtime_args(core, {tensor_return_value.buffer(), num_cols_per_core, num_cols_read});
            num_cols_read += num_cols_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
