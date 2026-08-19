// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <variant>

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
ReduceDeviceOperation::ReduceMultiCoreHProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const bool rm_path = operation_attributes.row_major_h_dense_path;
    const auto& shape = a.padded_shape();
    const auto& logical_shape = a.logical_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    uint32_t Wt = tt::div_up(W, tile_width);
    uint32_t Ht = tt::div_up(H, tile_height);
    uint32_t HtWt = Ht * Wt;

    if (rm_path) {
        validate_rm_preconditions(
            a, output, operation_attributes.math_op, operation_attributes.negate, ReduceOpDim::H, "Reduce H");
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device().arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = &a.mutable_device();

    // This path runs on the input's shard grid and aliases the input and output CBs onto the
    // tensors' own buffers; CBs live in L1, so both sides must be width-sharded there.
    bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              a.memory_config().is_l1() && output.memory_config().is_l1();

    // Populate the RM-only locals (chunk sizes, page bytes, padding identity, datum sizes) into
    // a single struct so the per-site formulas don't drift between this factory and the W one.
    // tt::datum_size(...) inside make_rm_plan throws for block-float formats; guard the call
    // behind rm_path since validate_rm_preconditions already gates the RM branch to BF16/FP32.
    RmPlan plan{};
    if (rm_path) {
        plan = make_rm_plan(
            shape,
            logical_shape,
            tile_height,
            tile_width,
            src0_cb_data_format,
            dst_cb_data_format,
            operation_attributes.math_op,
            ReduceOpDim::H);
    }

    // H-axis split geometry: every slice reduces a uniform `slice_Ht` tiles, the last one's overhang
    // past Ht_rm identity-padded by the reader. Clamped to Ht_rm so no slice is empty.
    const uint32_t num_h_slices = rm_path ? std::min(std::max(operation_attributes.num_h_slices, 1u), plan.Ht_rm) : 1;
    const uint32_t slice_Ht = rm_path ? tt::div_up(plan.Ht_rm, num_h_slices) : 0;
    // compute_output_specs sizes the output's H from the unclamped attribute, so the clamp above must
    // be a no-op; the host already bounds num_h_slices by Ht_rm.
    TT_FATAL(
        !rm_path || operation_attributes.num_h_slices <= plan.Ht_rm,
        "Reduce H (dense RM): num_h_slices {} exceeds Ht_rm {}; the output spec and the kernels would disagree",
        operation_attributes.num_h_slices,
        plan.Ht_rm);

    uint32_t chunk_size = use_width_sharding ? 1 : ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;

    // Int32 max/min/sum use the SFPU reduce path; fp32 SUM only for the accurate mean opt-in.
    const bool is_sfpu_reduce =
        use_sfpu_reduce_path(a.dtype(), operation_attributes.math_op, operation_attributes.use_sfpu_reduce);
    const bool use_fpu_negate = operation_attributes.negate && !is_sfpu_reduce;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // One output tile column per (nc, slice, wt) of the (N, C, num_h_slices, W) result.
    auto num_cols = NC * num_h_slices * Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
    }
    TT_FATAL(num_cores > 0, "Reduce H requires at least one worker core");

    // Current sharding only supports width, and that input and output are sharded
    if (use_width_sharding) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / tile_width);
        num_cols_per_core_group_2 = 0;
    }

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the reduce factory .cpp files land in the same unity-build
    // translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName IN_DFB{"in"};  // tiled/sharded: reduce input; dense-RM: tilize output
    const DFBSpecName SRC1_DFB{"src1"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName ACC_DFB{"acc"};
    const DFBSpecName INEG_DFB{"ineg"};
    const DFBSpecName RM_DFB{"rm"};
    const DFBSpecName CLEAR_VALUE_DFB{"clear_value"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = rm_path ? "reduce_multi_core_h_dense_rm"
                        : (use_width_sharding ? "reduce_multi_core_h_width_sharded" : "reduce_multi_core_h");

    // ---- Dataflow buffers ----
    if (rm_path) {
        // Buffer entries are per-row (see make_rm_plan); hold 2 slabs worth of rows so the reader can
        // produce one slab while compute drains the previous one (compute_kernel_lib::tilize waits
        // for up to TILE_HEIGHT entries per block).
        const uint32_t num_rm_pages = 2 * plan.rm_rows_per_tile;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_DFB,
            .entry_size = plan.rm_staging_page_size,
            .num_entries = num_rm_pages,
            .data_format_metadata = src0_cb_data_format,
        });

        // The reader's private padding-identity template: it fills the entry and then re-reads it as
        // a NoC source, so it is this buffer's only toucher.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = CLEAR_VALUE_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = src0_cb_data_format,
        });

        // reduce_rm.cpp accumulates partial reductions across H chunks into the accumulator.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = plan.wt_tiles_per_chunk,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    if (rm_path) {
        // RM compute kernel expects up to wt_tiles_per_chunk * ht_tiles_per_chunk tiles in flight
        // (NC fan-out is pinned to 1 in the RM compute contract).
        const uint32_t num_input_tiles = std::max(2U, plan.wt_tiles_per_chunk * plan.ht_tiles_per_chunk);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
        });
    } else if (use_width_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
        constexpr uint32_t num_input_tiles = 2;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
        });

        // A view onto the resident input shard rather than its own L1 allocation: the reader reads
        // the shard in place through this buffer.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC1_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_shard_tiles,
            .data_format_metadata = src0_cb_data_format,
            .borrowed_from = INPUT_TENSOR,
        });
    } else {
        uint32_t num_input_tiles = use_fpu_negate ? chunk_size : 2;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
        });
    }

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format,
    });

    if (rm_path) {
        const uint32_t num_output_tiles = std::max(2U, plan.wt_tiles_per_chunk);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
        });
    } else if (use_width_sharding) {
        uint32_t num_output_tiles = output.shard_spec().value().numel() / tile_hw;
        // A view onto the resident output shard: the packer writes the result in place, and the
        // writer's wait/pop is only a readiness handshake.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = OUTPUT_TENSOR,
        });
    } else {
        uint32_t num_output_tiles = use_fpu_negate ? chunk_size : 2;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
        });
    }
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    // Packed fp32 scalar passed to the compute kernel for mul_unary_tile post-reduction scaling.
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    if (use_fpu_negate) {
        // The reduce_h_neg kernel pushes ntiles tiles per inner-loop iteration
        // via push_back(ntiles).  The FIFO write pointer only wraps when
        // wr_ptr exactly reaches fifo_limit, so it is not enough for the buffer
        // size to be a multiple of each individual push size — the cumulative
        // offset across the inner Ht loop must also wrap to 0 at the end of
        // each nc iteration.  Per nc, the kernel advances wr_ptr by
        // Ht * Wt_per_core regardless of how that splits into chunk_size and
        // partial pushes, so sizing the buffer at Ht * Wt_per_core makes the
        // trajectory land on fifo_limit exactly.  For two core groups, the
        // single-buffer option uses Ht * lcm(Wt_g1, Wt_g2) so the same allocation
        // works for both groups.
        const uint32_t compute_Wt_g1 =
            use_width_sharding ? (NC == 0 ? 0 : num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
        const uint32_t compute_Wt_g2 = use_width_sharding ? 0 : num_cols_per_core_group_2;
        uint32_t per_nc_advance = 0;
        if (compute_Wt_g2 == 0) {
            per_nc_advance = compute_Wt_g1;
        } else if (compute_Wt_g1 == 0) {
            per_nc_advance = compute_Wt_g2;
        } else {
            per_nc_advance = std::lcm(compute_Wt_g1, compute_Wt_g2);
        }
        TT_FATAL(
            per_nc_advance > 0,
            "Negate H reduce: per-core Wt resolved to 0 (g1={}, g2={}, NC={}, sharded={})",
            compute_Wt_g1,
            compute_Wt_g2,
            NC,
            use_width_sharding);
        // Compute in uint64_t to mirror h_reduce_negate_fits_in_l1 and avoid
        // uint32_t overflow before the L1 fit check / buffer sizing.
        const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;

        // L1 fit check: acc and ineg are each sized at negate_cb_tiles.  If the
        // combined allocation would not fit in the available L1 budget, the caller
        // is expected to fall back to external negation — see
        // ttnn::prim::h_reduce_negate_fits_in_l1 in common.cpp, which mirrors this
        // calculation.
        const uint64_t per_cb_bytes = negate_cb_tiles * dst_single_tile_size;
        const uint64_t negate_cb_bytes = 2ull * per_cb_bytes;
        const auto lowest_address = device->lowest_occupied_compute_l1_address();
        uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
        const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        TT_FATAL(
            max_l1_space > base_addr,
            "Negate H reduce: L1 base allocator address {} >= lowest occupied address {}; no room for buffers",
            base_addr,
            max_l1_space);
        max_l1_space -= base_addr;
        TT_FATAL(
            negate_cb_bytes <= max_l1_space,
            "Negate H reduce: acc + ineg ({} B for {} tiles) would not fit in available L1 ({} B). "
            "Caller must use h_reduce_negate_fits_in_l1 to choose the external-negate fallback.",
            negate_cb_bytes,
            negate_cb_tiles,
            max_l1_space);
        // num_entries is uint32_t; the L1 fit check above already bounds negate_cb_tiles by the
        // per-core L1 budget (well under 4 GiB), but assert the narrowing explicitly so any future
        // budget change surfaces here instead of producing a silently-truncated buffer size.
        TT_FATAL(
            per_cb_bytes <= std::numeric_limits<uint32_t>::max(),
            "Negate H reduce: per-buffer size {} B exceeds uint32_t range",
            per_cb_bytes);
        const uint32_t negate_num_entries = static_cast<uint32_t>(negate_cb_tiles);

        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = negate_num_entries,
            .data_format_metadata = dst_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = negate_num_entries,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    // Declared on every config. On the width-sharded path no kernel builds a TensorAccessor on
    // them; they are named by the two borrowed-memory buffers' `borrowed_from` instead, which the
    // spec validator accepts as a use of the parameter.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    // Accurate fp32: route Float32 through the SFPU (needs 32-bit DEST)
    const bool fp32_sfpu_reduce = is_sfpu_reduce && a.dtype() == DataType::FLOAT32 && fp32_dest_acc_en;
    // A bf16 input packed into an FP32 partial needs the packer reconfigured, not just the unpacker.
    if (rm_path && dst_cb_data_format != src0_cb_data_format) {
        reduce_defines["REDUCE_RM_MIXED_FORMAT"] = "1";
    }
    // Write the compute-packed tiles whole instead of extracting their reduced row into RM pages.
    if (rm_path && output.layout() == Layout::TILE) {
        reduce_defines["REDUCE_RM_TILE_OUTPUT"] = "1";
    }

    // ---- Reader kernel ----
    std::string reader_source;
    KernelSpec::CompileTimeArgs reader_ct_args;
    Group<std::string> reader_rta_names;
    Group<DFBBinding> reader_dfb_bindings;
    Group<TensorBinding> reader_tensor_bindings;
    std::map<std::string, std::string> reader_defines_map = reduce_defines;

    if (rm_path) {
        reader_source = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp";
        reader_ct_args = build_rm_reader_ct_args(plan, scaler_bits, num_h_slices, slice_Ht);
        reader_rta_names = {"rt_count", "rt_start"};
        reader_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = RM_DFB,
                .accessor_name = "rm",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .accessor_name = "scaler",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            // Self-loop: the reader both fills the identity template and re-reads it.
            DFBBinding{
                .dfb_spec_name = CLEAR_VALUE_DFB,
                .accessor_name = "clear_value",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = CLEAR_VALUE_DFB,
                .accessor_name = "clear_value",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        reader_tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}};
    } else if (use_width_sharding) {
        reader_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp";
        reader_ct_args = {
            {"scaler_bits", scaler_bits},
            {"enable_fp32_sfpu", fp32_sfpu_reduce ? 1u : 0u},
        };
        reader_rta_names = {"num_tiles", "Wt", "Ht", "batch", "row_size_bytes", "batch_size_bytes"};
        reader_defines_map["REDUCE_SCALER"] = "1";
        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        reader_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN_DFB,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .accessor_name = "scaler",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            // Self-loop: the reader reserves the whole borrowed input shard and re-reads it in place
            // as the NoC source; nothing else touches it.
            DFBBinding{
                .dfb_spec_name = SRC1_DFB,
                .accessor_name = "in1",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SRC1_DFB,
                .accessor_name = "in1",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        // No tensor binding: the input is reached through the borrowed SRC1_DFB, not an accessor.
    } else {
        reader_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp";
        reader_ct_args = {
            {"Ht", Ht},
            {"Wt", Wt},
            {"HtWt", HtWt},
            {"scaler_bits", scaler_bits},
            {"use_welford", 0u},
            {"enable_fp32_sfpu", fp32_sfpu_reduce ? 1u : 0u},
        };
        reader_rta_names = {"col_start_tile_id", "curr_col_in_batch", "num_cols"};
        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        reader_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN_DFB,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .accessor_name = "scaler",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        reader_tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}};
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = reader_source,
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reader_defines_map)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = std::move(reader_ct_args),
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- Writer kernel ----
    // On the tiled and width-sharded paths these are the Metal 2.0 forks of the eltwise/unary and
    // data_movement/sharded writers; their binding vocabulary is the forks', not this op's.
    std::string writer_source;
    KernelSpec::CompileTimeArgs writer_ct_args;
    Group<std::string> writer_rta_names;
    Group<TensorBinding> writer_tensor_bindings;
    KernelSpec::CompilerOptions::Defines writer_defines;

    if (rm_path) {
        writer_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_rm_scalar.cpp";
        // One writer for both layouts; tile_output picks whole-tile pages over (nc, slice) RM pages.
        writer_ct_args =
            build_rm_writer_ct_args(plan, operation_attributes.output_layout == Layout::TILE, num_h_slices);
        writer_rta_names = {"rt_count", "rt_start"};
        writer_tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}};
        writer_defines = KernelSpec::CompilerOptions::Defines(reduce_defines);
    } else if (use_width_sharding) {
        writer_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_metal2.cpp";
        writer_rta_names = {"num_units"};
        // No tensor binding: the output is written in place through the borrowed OUT_DFB.
    } else {
        writer_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp";
        writer_rta_names = {"num_pages", "start_id"};
        writer_tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}};
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = writer_source,
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args = std::move(writer_ct_args),
        .runtime_arg_schema = {.runtime_arg_names = std::move(writer_rta_names)},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- Compute kernels (one per core group) ----
    // Legacy resolved a TTNN ComputeKernelConfig and forwarded math_fidelity, fp32_dest_acc_en,
    // dst_full_sync_en and unpack_to_dest_mode onto ComputeConfigDescriptor, leaving
    // math_approx_mode at the *Metal* descriptor default (false). Reproduce that: the TTNN helper
    // would otherwise carry the caller's math_approx_mode into sfpu_precision_mode, silently
    // changing SFPU precision. (Unlike the other three factories, this one *does* forward
    // dst_full_sync_en, so double_buffer_dest keeps the helper's inverted value.)
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // std::visit rather than a Gen1-only get_if: to_compute_hardware_config yields a
    // ComputeGen2Config on Quasar, and the fields set below exist on both generations. The
    // explicit-unpack-mode requirement in particular is enforced generation-agnostically, so a
    // Gen1-only branch would leave FP32 + 32-bit-Dest programs failing ProgramSpec validation there.
    // (Unlike the other three factories, this one forwarded dst_full_sync_en, so double_buffer_dest
    // is left at whatever the caller's config resolved to.)
    std::visit(
        [&](auto& compute_cfg) {
            compute_cfg.sfpu_precision_mode = Precision::Precise;  // legacy math_approx_mode = false
            if (fp32_sfpu_reduce) {
                // Legacy: unpack_to_dest_mode[src0_cb_index] = UnpackToDestFp32 — unpacks the reduce
                // input straight into the fp32 DEST, bypassing the SrcA tf32 truncation. The RM
                // path's chunk accumulator (legacy c_5) gets the same treatment so partials
                // round-trip in fp32.
                compute_cfg.unpack_modes.emplace(IN_DFB, UnpackMode::UnpackToDest);
                if (rm_path) {
                    compute_cfg.unpack_modes.emplace(ACC_DFB, UnpackMode::UnpackToDest);
                }
            }
            // Legacy left every other entry at Default (= UnpackToSrc). Metal 2.0 nonetheless requires an
            // explicit mode for every Float32 buffer this kernel consumes under a 32-bit Dest register,
            // so state the legacy value for those.
            auto require_explicit_unpack_mode = [&](const DFBSpecName& name, tt::DataFormat format) {
                if (fp32_dest_acc_en && format == tt::DataFormat::Float32) {
                    compute_cfg.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
                }
            };
            require_explicit_unpack_mode(IN_DFB, src0_cb_data_format);
            require_explicit_unpack_mode(SCALER_DFB, scaler_cb_data_format);
            if (rm_path) {
                require_explicit_unpack_mode(RM_DFB, src0_cb_data_format);
                require_explicit_unpack_mode(ACC_DFB, dst_cb_data_format);
            } else if (use_fpu_negate) {
                require_explicit_unpack_mode(ACC_DFB, dst_cb_data_format);
                require_explicit_unpack_mode(INEG_DFB, dst_cb_data_format);
            }
        },
        compute_hw);

    // For width-sharding, num_cols_per_core_group_1 == NC * shard_Wt. Expose (shard_Wt, NC)
    // to the compute kernel so its (nc, wt_chunk, ht, wt_in_chunk) iteration matches the
    // reader's per-batch tile layout.
    uint32_t compute_Wt = use_width_sharding ? (num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
    uint32_t compute_NC = use_width_sharding ? NC : 1;

    // MIN on an SFPU path uses the base reduce.cpp kernel (negate=false); fast-mode float/bf16 MIN
    // uses -MAX(-x) in reduce_h_neg.
    const std::string compute_kernel =
        rm_path ? std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp")
                : std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
                      (operation_attributes.negate ? "_h_neg" : "") + ".cpp";

    // reduce_rm.cpp reads its per-core output count as a runtime arg on the H path only; the define
    // below is what lets the kernel's H branch reference that name at all (the W factory declares no
    // compute runtime args, so the `args::` token does not exist there).
    std::map<std::string, std::string> compute_defines_map = reduce_defines;
    if (rm_path) {
        compute_defines_map["REDUCE_RM_H_PATH"] = "1";
    }
    if (use_fpu_negate) {
        // reduce_h_neg.cpp is the FPU -MAX(-x) path; the acc / ineg scratch buffers it uses are
        // bound only in this case. The kernel gates its references to those buffers on this define,
        // because `if constexpr` cannot suppress name lookup.
        compute_defines_map["REDUCE_FPU_NEGATE"] = "1";
    }

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t group_compute_Wt, uint32_t group_compute_NC) {
        Group<DFBBinding> dfb_bindings;
        KernelSpec::CompileTimeArgs ct_args;
        Group<std::string> rta_names;
        if (rm_path) {
            // RM kernel takes per-core counts via runtime args, so both groups share compile args.
            // The compute kernel's H loop bound is per-slice: slice_Ht == plan.Ht_rm when unsplit.
            ct_args = build_rm_compute_ct_args(plan, slice_Ht, post_mul_scaler_bits, fp32_sfpu_reduce);
            rta_names = {"num_output_tiles_local", "output_tiles_seen"};
            dfb_bindings = {
                DFBBinding{
                    .dfb_spec_name = RM_DFB,
                    .accessor_name = "rm",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // tilize writes the tiled staging buffer and reduce drains it, both inside this
                // kernel — a self-loop.
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "tile_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "tile_in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALER_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The chunk accumulator round-trips through L1 inside this kernel — a self-loop.
                DFBBinding{
                    .dfb_spec_name = ACC_DFB,
                    .accessor_name = "acc",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ACC_DFB,
                    .accessor_name = "acc",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            };
        } else {
            ct_args = {
                {"Ht", Ht},
                {"Wt", group_compute_Wt},
                {"NC", group_compute_NC},
                // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
                {"post_mul_scaler_bits", post_mul_scaler_bits},
                // enable_fp32_sfpu: route Float32 through the SFPU
                {"enable_fp32_sfpu", fp32_sfpu_reduce ? 1u : 0u},
            };
            dfb_bindings = {
                DFBBinding{
                    .dfb_spec_name = IN_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALER_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            };
            if (use_fpu_negate) {
                // Self-loops: the compute kernel is the only toucher of acc / ineg.
                dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = ACC_DFB,
                    .accessor_name = "acc",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                });
                dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = ACC_DFB,
                    .accessor_name = "acc",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                });
                dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = INEG_DFB,
                    .accessor_name = "ineg",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                });
                dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = INEG_DFB,
                    .accessor_name = "ineg",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                });
            }
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel,
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options =
                {.defines = KernelSpec::CompilerOptions::Defines(compute_defines_map),
                 .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(dfb_bindings),
            .compile_time_args = std::move(ct_args),
            .runtime_arg_schema = {.runtime_arg_names = std::move(rta_names)},
            .hw_config = compute_hw,
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1, compute_Wt, compute_NC));
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        uint32_t compute_Wt_group_2 = use_width_sharding ? (num_cols_per_core_group_2 / NC) : num_cols_per_core_group_2;
        uint32_t compute_NC_group_2 = use_width_sharding ? NC : 1;
        spec.kernels.push_back(make_compute(COMPUTE_G2, compute_Wt_group_2, compute_NC_group_2));
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

    // ---- Runtime args per node ----
    std::vector<CoreCoord> cores;
    if (rm_path) {
        // RM compute kernel iterates cores in row-wise order; match it so per-core counts line up.
        cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    } else if (operation_attributes.sub_core_grids.has_value()) {
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
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = COMPUTE_G2};

    if (rm_path) {
        for (uint32_t i = 0, output_tiles_seen = 0; i < num_cores; i++) {
            const CoreCoord& core = cores[i];
            uint32_t num_output_tiles_local = 0;
            if (core_group_1.contains(core)) {
                num_output_tiles_local = num_cols_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_output_tiles_local = num_cols_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"rt_count", num_output_tiles_local}, {"rt_start", output_tiles_seen}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"rt_count", num_output_tiles_local}, {"rt_start", output_tiles_seen}});
            if (core_group_1.contains(core)) {
                AddRuntimeArgsForNode(
                    compute_g1_run_args.runtime_arg_values,
                    core,
                    {{"num_output_tiles_local", num_output_tiles_local}, {"output_tiles_seen", output_tiles_seen}});
            } else if (has_core_group_2) {
                AddRuntimeArgsForNode(
                    compute_g2_run_args.runtime_arg_values,
                    core,
                    {{"num_output_tiles_local", num_output_tiles_local}, {"output_tiles_seen", output_tiles_seen}});
            } else {
                TT_THROW("Reduce H (dense RM): core in core_group_2 but no second compute kernel");
            }

            output_tiles_seen += num_output_tiles_local;
            if (i == num_cores - 1) {
                TT_FATAL(
                    output_tiles_seen == num_cols,
                    "Reduce H (dense RM) assigned {} output tile columns across cores, expected {}",
                    output_tiles_seen,
                    num_cols);
            }
        }
    } else if (use_width_sharding) {
        TT_FATAL(NC != 0, "Batch size NC must be non-zero (shape[0]={}, shape[1]={})", shape[0], shape[1]);
        uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        uint32_t shard_batch_size = shard_row_size * Ht;
        // Width-sharded path: iterate the actual shard core set (all_cores), not the
        // grid_to_cores sequence — sharded grids may not start at (0,0).
        for (const auto& range : all_cores.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    CoreCoord core{x, y};
                    AddRuntimeArgsForNode(
                        reader_run_args.runtime_arg_values,
                        core,
                        {{"num_tiles", num_cols_per_core_group_1 * Ht},
                         {"Wt", shard_Wt},
                         {"Ht", Ht},
                         {"batch", NC},
                         {"row_size_bytes", shard_row_size},
                         {"batch_size_bytes", shard_batch_size}});
                    AddRuntimeArgsForNode(
                        writer_run_args.runtime_arg_values, core, {{"num_units", num_cols_per_core_group_1}});
                }
            }
        }
    } else {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
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
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                 {"curr_col_in_batch", num_cols_read % Wt},
                 {"num_cols", num_cols_per_core}});

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {// number of tiles to write
                 {"num_pages", num_cols_per_core},
                 // output tile start index
                 {"start_id", num_cols_read}});
            num_cols_read += num_cols_per_core;
            if (i == num_cores - 1) {
                TT_FATAL(
                    num_cols_read == num_cols,
                    "Reduce H assigned {} columns across cores, expected {}",
                    num_cols_read,
                    num_cols);
            }
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    if (rm_path) {
        run_args.kernel_run_args.push_back(std::move(compute_g1_run_args));
        if (has_core_group_2) {
            run_args.kernel_run_args.push_back(std::move(compute_g2_run_args));
        }
    }

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{a});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
