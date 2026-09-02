// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "welford_reduce_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <cstdint>
#include <variant>

namespace ttnn::prim {

namespace {

// compute_g2 exists only when the work split leaves a second core group. override_runtime_arguments
// cannot see the Program, and naming a kernel it lacks is fatal, so it recomputes the decision here;
// create_program_artifacts asserts the two agree.
bool welford_has_second_core_group(const WelfordReduceParams& attrs, const tt::tt_metal::MeshTensor& a) {
    using namespace tt::tt_metal;
    // Mirrors create_program_artifacts: the tensor may be any rank, so NC is the product of every
    // dimension except the last two, taken from the physical volume.
    const uint32_t W_padded = a.padded_shape()[-1];
    const uint32_t H_padded = a.padded_shape()[-2];
    const uint32_t NC = a.physical_volume() / (H_padded * W_padded);
    const uint32_t Ht = H_padded / a.tensor_spec().tile().get_height();
    const uint32_t Wt = W_padded / a.tensor_spec().tile().get_width();

    const bool reduce_w = attrs.reduce_dim == ReduceOpDim::W;
    const bool reduce_hw = attrs.reduce_dim == ReduceOpDim::HW;
    const uint32_t num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / attrs.reduce_batch_size) : (NC * Wt));

    // Must use the same overload as create_program_artifacts: the grid-size CoreCoord form, not a
    // CoreRange built from it (a size is not an inclusive end coordinate).
    CoreRangeSet group_2;
    if (attrs.sub_core_grids.has_value()) {
        group_2 = std::get<3>(split_work_to_cores(*attrs.sub_core_grids, num_work_units));
    } else {
        group_2 = std::get<3>(split_work_to_cores(a.mutable_device().compute_with_storage_grid_size(), num_work_units));
    }
    return !group_2.ranges().empty();
}

}  // namespace

ttnn::device_operation::ProgramArtifacts
WelfordReduceDeviceOperation::WelfordReduceProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_arg,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_arg.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    const Shape& padded_shape = input.padded_shape();
    const Shape& logical_shape = input.logical_shape();

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
    uint32_t NC = input.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();

    uint32_t Wt = W_padded / tile_width;
    uint32_t Ht = H_padded / tile_height;
    uint32_t HtWt = Ht * Wt;

    const bool reduce_w = (operation_attributes.reduce_dim == ReduceOpDim::W);
    const bool reduce_h = (operation_attributes.reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (operation_attributes.reduce_dim == ReduceOpDim::HW);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device().arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    // Float32 input on the welford path requires fp32_dest_acc_en=true as a prerequisite for
    // UnpackToDest (set below). UnpackToDest is what bypasses the unpacker's
    // Float32 → TF32 truncation in SrcA; fp32_dest_acc_en provides the 32-bit DEST that
    // UnpackToDest writes into. Without fp32 DEST, UnpackToDest can't be enabled
    // and inputs are silently truncated to TF32 (10 mantissa bits) on the way through SrcA.
    TT_FATAL(
        !(input_cb_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "ttnn.std/var with Float32 input requires fp32_dest_acc_en=true in the compute kernel "
        "config; otherwise precision is silently lost in the unpacker format conversion.");

    // Match the scalar buffer's data format to the input. When the input is FP32, the scalar must
    // also be FP32: mul_tiles_bcast_scalar reads the input as SrcA and the scalar as SrcB, and a
    // format/stride mismatch between the two operands would cause the unpacker to silently
    // produce zeros into DEST.
    tt::DataFormat scalar_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Float32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);

    // For variance output (is_std=false) to bf16, the scratch buffers (var for W-reduce,
    // combined for HW-reduce) do not need to be wider than bf16: there is no math between
    // the scratch read-back and the final pack to output, so bf16-rounding once at the
    // pack to scratch vs once at the pack to output produces a bit-identical result.
    // For std output, sqrt sits between the read-back and the output pack, so keep the
    // scratch at the wider precision to avoid quantizing the variance before sqrt
    // (which could shift the std output by up to one bf16-ULP on elements whose post-
    // sqrt value straddles a bf16 rounding boundary).
    bool narrow_scratch_to_bf16 = !is_std && dst_cb_data_format == tt::DataFormat::Float16_b;

    tt_metal::IDevice* device = &input.mutable_device();

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

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: this factory shares a unity-build translation unit with the reduce
    // factories, so no anonymous-namespace constants are introduced.
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName SCALAR_DFB{"scalar"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName VAR_DFB{"var"};
    const DFBSpecName PARTIAL_DFB{"partial"};
    const DFBSpecName COMBINED_DFB{"combined"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = reduce_w ? "welford_reduce_w" : (reduce_hw ? "welford_reduce_hw" : "welford_reduce_h");

    // ---- Dataflow buffers ----
    // Input buffer. The unpack_modes entry below makes it UnpackToDest for FP32 input so the welford
    // SFPU intake (copy_tile / transpose_tile) reads via the precision-preserving unpack-to-DEST path
    // instead of the FPU SrcA path (which would truncate FP32 to TF32). The user scalar is applied as
    // an SFPU post-multiplication on the reduced output, not by pre-scaling the input -- see
    // post_mul_scaler below.
    constexpr uint32_t input_tiles_per_cb = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_tiles_per_cb,
        .data_format_metadata = input_cb_data_format,
    });

    // The reader fills one scalar entry on every reduce dim, but no welford compute kernel reads it
    // (the user scalar is applied post-reduction in the compute kernel instead), so the reader is
    // this buffer's only toucher.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALAR_DFB,
        .entry_size = scalar_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scalar_cb_data_format,
    });

    constexpr uint32_t output_tiles_per_cb = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = output_tiles_per_cb,
        .data_format_metadata = dst_cb_data_format,
    });

    // var: W-reduce only -- scratch buffer for the variance tile between
    // the two transpose steps (Welford produces row-oriented results that must
    // be transposed back to column orientation).
    tt::DataFormat scratch_cb_data_format = tt::DataFormat::Float16_b;
    if (reduce_w) {
        // Float32 only when the DST register is fp32 and we are not narrowing the scratch
        // to the output dtype (variance output to bf16 -- see narrow_scratch_to_bf16 above);
        // bf16 otherwise.
        scratch_cb_data_format =
            (fp32_dest_acc_en && !narrow_scratch_to_bf16) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = VAR_DFB,
            .entry_size = tt::tile_size(scratch_cb_data_format),
            .num_entries = 1,
            .data_format_metadata = scratch_cb_data_format,
        });
    }

    // Post-reduction scaling: the reduction always runs unscaled (the precise
    // UnpackToDest path), and the user scalar is applied to the small-magnitude result via
    // SFPU mul_unary_tile inside the compute kernel, which skips an identity multiplier.
    // Pre-scaling the input (the old do_scale path) read the input via the FPU SrcA operand at TF32
    // precision and collapsed large-offset inputs to a constant before the multiply. The
    // post-multiplier follows var(s*x)=s^2 var(x) and std(s*x)=|s| std(x):
    //   var: scalar^2   std: |scalar|.
    const float post_mul_scaler =
        is_std ? std::abs(operation_attributes.scalar) : operation_attributes.scalar * operation_attributes.scalar;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(post_mul_scaler);

    // partial: HW-reduce only -- holds per-column mean+var tile pairs
    // from the compute kernel, consumed by the writer kernel.
    // Uses Float32 format to preserve precision from DST accumulators.
    tt::DataFormat combined_cb_data_format = tt::DataFormat::Float32;
    if (reduce_hw) {
        tt::DataFormat partial_cb_data_format = tt::DataFormat::Float32;
        // Reserve space for 4 tiles to enable double buffering (since compute kernel packs 2 tiles at a time).
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PARTIAL_DFB,
            .entry_size = tt::tile_size(partial_cb_data_format),
            .num_entries = 4,
            .data_format_metadata = partial_cb_data_format,
        });

        // combined: HW-reduce only -- holds the combined scalar result
        // (one tile per output) written by the writer kernel after W-combining
        // all per-column partials and applying Bessel's correction.
        // The compute kernel reads this tile, applies sqrt_tile for std, and
        // re-packs it to the output buffer in the correct output data format (the packer
        // hardware is required for BFLOAT8_B conversion).
        // Float32 unless we can safely narrow to bf16.
        combined_cb_data_format = narrow_scratch_to_bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = COMBINED_DFB,
            .entry_size = tt::tile_size(combined_cb_data_format),
            .num_entries = 1,
            .data_format_metadata = combined_cb_data_format,
        });
    }

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);
    reduce_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    // Enables the SFPU post-multiplication of the reduced output by the user scalar in the
    // compute kernel (see post_mul_scaler above). Only the compute kernel reads this; the
    // reader/writer ignore it.
    // --- Reader kernel ---
    // The reduction always runs unscaled and no welford compute kernel reads this buffer, so the
    // reader only ever fills it with the identity.
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f);
    std::string reader_source;
    KernelSpec::CompileTimeArgs reader_ct_args;
    Group<std::string> reader_rta_names;
    if (reduce_h || reduce_hw) {
        // H-reduce and HW-reduce: column-partitioned reader reads tiles column by column.
        // Welford processes one column at a time (SFPU can only track one running
        // mean/M2 state), so the reader must deliver tiles in strict column-major
        // order: all Ht tiles of column 0, then all Ht tiles of column 1, etc.
        // enable_fp32_sfpu=0: Welford never uses the fp32-SFPU reduce path (use_welford=1 forces
        // row_chunk=1). The arg keeps this reader's CT-arg set in lockstep with the reduce factories.
        reader_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp";
        reader_ct_args = {
            {"Ht", Ht},
            {"Wt", Wt},
            {"HtWt", HtWt},
            {"scaler_bits", scaler_bits},
            {"use_welford", 1u},
            {"enable_fp32_sfpu", 0u},
        };
        reader_rta_names = {"col_start_tile_id", "curr_col_in_batch", "num_cols"};
    } else {
        // W-reduce: sequential reader reads tiles row by row.
        reader_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp";
        reader_ct_args = {{"scaler_bits", scaler_bits}};
        reader_rta_names = {"num_tiles", "start_id"};
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = reader_source,
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reduce_defines)},
        // The two readers are shared with the Reduce factories, so their accessor names are the
        // shared kernels' vocabulary (in0 / scaler), not this factory's local naming.
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Self-loop: no welford compute kernel reads the scalar entry, so the reader is its
                // only toucher.
                DFBBinding{
                    .dfb_spec_name = SCALAR_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALAR_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = std::move(reader_ct_args),
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // --- Writer kernel ---
    std::string writer_source;
    KernelSpec::CompileTimeArgs writer_ct_args;
    Group<std::string> writer_rta_names;
    // Only the HW writer applies Bessel's correction; the W/H paths use the generic unary writer.
    Group<std::string> writer_common_rta_names;
    Group<DFBBinding> writer_dfb_bindings;
    KernelSpec::CompilerOptions::Defines writer_defines;

    if (reduce_hw) {
        if (operation_attributes.correction) {
            TT_FATAL(
                H * W * reduce_batch_size >= 2,
                "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
                H * W * reduce_batch_size);
        }

        // HW-reduce: custom writer that combines partial stats and constructs the output tile.
        writer_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "writer_welford_hw.cpp";
        writer_ct_args = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"H", H},
            {"reduce_batch_size", reduce_batch_size},
            {"combined_is_bf16", static_cast<uint32_t>(narrow_scratch_to_bf16)},
        };
        writer_rta_names = {"NC_per_core", "output_tile_start_id"};
        writer_common_rta_names = {"correction"};
        writer_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = PARTIAL_DFB,
                .accessor_name = "partial",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            // The writer *produces* the combined scalar entry that compute reads back — the reverse
            // of the usual writer direction.
            DFBBinding{
                .dfb_spec_name = COMBINED_DFB,
                .accessor_name = "combined",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        // Note: HW writer does not pass reduce_defines (matches original behavior).
    } else {
        // W-reduce and H-reduce: generic tile writer — the Metal 2.0 fork of the eltwise/unary
        // writer. Its binding vocabulary (dfb::out, tensor::dst, RTAs num_pages / start_id) is the
        // fork's, not this op's.
        writer_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp";
        writer_rta_names = {"num_pages", "start_id"};
        writer_dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer_defines = KernelSpec::CompilerOptions::Defines(reduce_defines);
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = writer_source,
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args = std::move(writer_ct_args),
        .runtime_arg_schema =
            {.runtime_arg_names = std::move(writer_rta_names), .common_runtime_arg_names = writer_common_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // --- Compute kernels ---
    std::string compute_kernel;
    KernelSpec::CompileTimeArgs compute_ct_args;
    std::string compute_rta_name;

    if (reduce_hw) {
        compute_ct_args = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"Wt", Wt},
            {"reduce_batch_size", reduce_batch_size},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
        compute_kernel = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
        compute_rta_name = "NC_per_core";
    } else {
        if (operation_attributes.correction) {
            uint32_t reduce_size = reduce_w ? W : H;
            TT_FATAL(
                reduce_size >= 2,
                "Bessel's correction requires at least 2 elements along the reduction dimension, got {}",
                reduce_size);
        }

        compute_ct_args = {
            {reduce_w ? "Wt" : "Ht", reduce_w ? Wt : Ht},
            {reduce_w ? "W" : "H", reduce_w ? W : H},
            {reduce_w ? "tile_width" : "tile_height", reduce_w ? tile_width : tile_height},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
        if (reduce_w) {
            // welford_fp32_input gates the transpose re-init / welford PreserveStats recovery in the
            // W-reduce compute kernel's wt-inner loop, needed because transpose_tile's unpack-to-DEST
            // path clobbers the welford SFPU replay buffer on FP32 input. H- and HW-reduce kernels
            // read the input via copy_tile (no transpose) and don't need this flag.
            compute_ct_args.emplace(
                "welford_fp32_input", static_cast<uint32_t>(input_cb_data_format == tt::DataFormat::Float32 ? 1 : 0));
        }
        compute_kernel = reduce_w
                             ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp"
                             : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
        compute_rta_name = reduce_w ? "NCHt" : "NCWt";
    }

    // Legacy resolved a TTNN ComputeKernelConfig but forwarded only math_fidelity,
    // fp32_dest_acc_en and unpack_to_dest_mode onto ComputeConfigDescriptor, leaving
    // math_approx_mode and dst_full_sync_en at the *Metal* descriptor defaults (both false).
    // Reproduce that exactly: the TTNN helper would otherwise carry the caller's math_approx_mode
    // into sfpu_precision_mode and the caller's dst_full_sync_en into double_buffer_dest, silently
    // changing precision / Dest buffering. (DST_SYNC_FULL is still passed as a *define*, exactly as
    // legacy did.)
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // std::visit rather than a Gen1-only get_if: to_compute_hardware_config yields a
    // ComputeGen2Config on Quasar, and the fields set below exist on both generations. The
    // explicit-unpack-mode requirement in particular is enforced generation-agnostically, so a
    // Gen1-only branch would leave FP32 + 32-bit-Dest programs failing ProgramSpec validation there.
    std::visit(
        [&](auto& compute_cfg) {
            compute_cfg.sfpu_precision_mode = Precision::Precise;  // legacy math_approx_mode = false
            compute_cfg.double_buffer_dest = true;                 // legacy dst_full_sync_en = false
            // For Float32 input with fp32_dest_acc_en, force unpack-to-dest so that
            // the unpacker writes full fp32 to DEST instead of routing through SrcA, which would
            // downcast to TF32, losing precision and even leading to large-mean fp32 variance
            // silently collapsing to ~0 due to TF32 truncation wiping the bits that are different
            // between nearby samples.
            //
            // Apply this to every Float32 buffer the compute kernel reads back via copy_tile /
            // transpose_tile:
            //   - Input: needed on all three reduction paths (H, W, HW) with FP32 input. The Welford
            //     SFPU intake reads it directly via copy_tile/transpose_tile, so UnpackToDest
            //     preserves the full FP32 into DEST (there is no input pre-scaling -- see post_mul_scaler).
            //   - W-reduce only: var -- the variance tile is read back after the initial
            //     transpose to undo it.
            //   - HW-reduce only: combined -- the variance tile is read back after the
            //     writer-side cross-core re-reduction.
            if (input_cb_data_format == tt::DataFormat::Float32) {
                compute_cfg.unpack_modes.emplace(IN_DFB, UnpackMode::UnpackToDest);
            }
            if (reduce_w && fp32_dest_acc_en && !narrow_scratch_to_bf16) {
                compute_cfg.unpack_modes.emplace(VAR_DFB, UnpackMode::UnpackToDest);
            }
            if (reduce_hw && fp32_dest_acc_en && !narrow_scratch_to_bf16) {
                compute_cfg.unpack_modes.emplace(COMBINED_DFB, UnpackMode::UnpackToDest);
            }
            // Legacy left every other entry at Default (= UnpackToSrc). Metal 2.0 nonetheless requires an
            // explicit mode for every Float32 buffer this kernel consumes under a 32-bit Dest register,
            // so state the legacy value for those.
            auto require_explicit_unpack_mode = [&](const DFBSpecName& name, tt::DataFormat format) {
                if (fp32_dest_acc_en && format == tt::DataFormat::Float32) {
                    compute_cfg.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
                }
            };
            require_explicit_unpack_mode(IN_DFB, input_cb_data_format);
            if (reduce_w) {
                require_explicit_unpack_mode(VAR_DFB, scratch_cb_data_format);
            }
            if (reduce_hw) {
                require_explicit_unpack_mode(COMBINED_DFB, combined_cb_data_format);
            }
        },
        compute_hw);

    auto make_compute = [&](const KernelSpecName& unique_id) {
        Group<DFBBinding> dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN_DFB,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (reduce_w) {
            // Self-loop: compute packs the variance tile into var and reads it straight back to
            // transpose it; nothing else touches it.
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = VAR_DFB,
                .accessor_name = "var",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = VAR_DFB,
                .accessor_name = "var",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (reduce_hw) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = PARTIAL_DFB,
                .accessor_name = "partial",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = COMBINED_DFB,
                .accessor_name = "combined",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel,
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options =
                {.defines = KernelSpec::CompilerOptions::Defines(reduce_defines),
                 .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(dfb_bindings),
            .compile_time_args = compute_ct_args,
            .runtime_arg_schema =
                {.runtime_arg_names = {compute_rta_name},
                 .common_runtime_arg_names = {"post_mul_scaler_bits", "correction"}},
            .hw_config = compute_hw,
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1));
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    TT_FATAL(
        has_core_group_2 == welford_has_second_core_group(operation_attributes, input),
        "Welford second-core-group predicate disagrees with the work split; the cache-hit override "
        "would name a compute kernel the program does not have");
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_G2));
    }

    // ---- Work units (placement) ----
    // Reader and writer belong to both work units, so their derived node set is the union of the two
    // core groups (the legacy `all_cores`), while each core group hosts its own compute instance.
    spec.work_units.push_back(
        WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        spec.work_units.push_back(
            WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // --- Runtime args per node ---
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

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = COMPUTE_G2};

    auto add_compute_rta = [&](bool in_g1, const CoreCoord& core, uint32_t value) {
        AddRuntimeArgsForNode(
            (in_g1 ? compute_g1_run_args : compute_g2_run_args).runtime_arg_values, core, {{compute_rta_name, value}});
    };

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
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_tiles", num_input_tiles_per_core}, {"start_id", input_tiles_offset}});
            add_compute_rta(in_g1, core, num_work_units_per_core);
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_pages", num_output_tiles_per_core}, {"start_id", output_tiles_offset}});
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
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"col_start_tile_id", col_start_tile_id}, {"curr_col_in_batch", 0u}, {"num_cols", num_cols}});
            // Compute: runtime arg is total NC slices (not num_outputs).
            add_compute_rta(in_g1, core, nc_slices_per_core);
            // Writer: NC_per_core is total NC slices; the writer uses reduce_batch_size
            // (compile-time) to determine how many to group per output.
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"NC_per_core", nc_slices_per_core}, {"output_tile_start_id", output_offset}});
            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        // H-reduce: each work unit is one column of Ht tiles
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
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                 {"curr_col_in_batch", num_cols_read % Wt},
                 {"num_cols", num_cols_per_core}});
            add_compute_rta(in_g1, core, num_cols_per_core);
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}});
            num_cols_read += num_cols_per_core;
        }
    }

    const KernelRunArgs::CommonRuntimeArgValues compute_common_args{
        {"post_mul_scaler_bits", post_mul_scaler_bits},
        {"correction", static_cast<uint32_t>(operation_attributes.correction)}};
    if (reduce_hw) {
        writer_run_args.common_runtime_arg_values = {
            {"correction", static_cast<uint32_t>(operation_attributes.correction)}};
    }
    compute_g1_run_args.common_runtime_arg_values = compute_common_args;

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_g1_run_args));
    if (has_core_group_2) {
        compute_g2_run_args.common_runtime_arg_values = compute_common_args;
        run_args.kernel_run_args.push_back(std::move(compute_g2_run_args));
    }

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs
WelfordReduceDeviceOperation::WelfordReduceProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_arg,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt::tt_metal::experimental;
    const auto& input = tensor_arg.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    // Names must match create_program_artifacts.
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    // compute_program_hash excludes `scalar` and `correction`, so calls differing only in those
    // share this program and must re-apply them here. Every other arg is shape-derived and hashed.
    const bool is_std = operation_attributes.math_op == tt::tt_metal::ReduceOpMath::STD;
    const float post_mul_scaler =
        is_std ? std::abs(operation_attributes.scalar) : operation_attributes.scalar * operation_attributes.scalar;
    const KernelRunArgs::CommonRuntimeArgValues compute_common_args{
        {"post_mul_scaler_bits", std::bit_cast<uint32_t>(post_mul_scaler)},
        {"correction", static_cast<uint32_t>(operation_attributes.correction)}};

    ProgramRunArgs params;
    if (operation_attributes.reduce_dim == tt::tt_metal::ReduceOpDim::HW) {
        // Only the HW writer applies Bessel's correction.
        params.kernel_run_args.push_back(KernelRunArgs{
            .kernel = WRITER,
            .common_runtime_arg_values = {{"correction", static_cast<uint32_t>(operation_attributes.correction)}}});
    }
    params.kernel_run_args.push_back(
        KernelRunArgs{.kernel = COMPUTE_G1, .common_runtime_arg_values = compute_common_args});
    if (welford_has_second_core_group(operation_attributes, input)) {
        params.kernel_run_args.push_back(
            KernelRunArgs{.kernel = COMPUTE_G2, .common_runtime_arg_values = compute_common_args});
    }

    params.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input});
    params.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output});
    return params;
}

}  // namespace ttnn::prim
