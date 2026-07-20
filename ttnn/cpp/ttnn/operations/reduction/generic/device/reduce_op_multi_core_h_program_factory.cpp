// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreHProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
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

    bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

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

    // H-axis split: partition the reduction axis into `num_h_slices` contiguous tile ranges so the
    // op can use NC*Wt*num_h_slices cores instead of NC*Wt. Each slice reduces `slice_Ht` tiles
    // (compile-time uniform; the last slice's overhang past Ht_rm is identity-padded to 0 by the
    // reader). Clamped to Ht_rm so slices are never empty. 1 = normal H-reduce (output H=1).
    const uint32_t num_h_slices = rm_path ? std::min(std::max(operation_attributes.num_h_slices, 1u), plan.Ht_rm) : 1;
    const uint32_t slice_Ht = rm_path ? tt::div_up(plan.Ht_rm, num_h_slices) : 0;

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
    // Each (nc, slice, wt) triple is one output tile column of the (N, C, num_h_slices, W) result.
    // num_h_slices == 1 (all non-split paths) reduces this to the classic NC*Wt.
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

    ProgramDescriptor desc;

    if (rm_path) {
        constexpr uint32_t cb_rm = CBIndex::c_24;
        // CB pages are per-row (see make_rm_plan); hold 2 slabs worth of rows so the reader can
        // produce one slab while compute drains the previous one (compute_kernel_lib::tilize waits
        // for up to TILE_HEIGHT pages per block).
        const uint32_t num_rm_pages = 2 * plan.rm_rows_per_tile;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_rm_pages * plan.rm_staging_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_rm),
                .data_format = src0_cb_data_format,
                .page_size = plan.rm_staging_page_size,
            }}},
        });

        constexpr uint32_t cb_clear_value = CBIndex::c_4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_clear_value),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });

        // reduce_rm.cpp accumulates partial reductions across H chunks into cb_acc (c_5).
        constexpr uint32_t cb_acc = CBIndex::c_5;
        desc.cbs.push_back(CBDescriptor{
            .total_size = plan.wt_tiles_per_chunk * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_acc),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t src1_cb_index = CBIndex::c_1;
    if (rm_path) {
        // RM compute kernel expects up to wt_tiles_per_chunk * ht_tiles_per_chunk tiles in flight
        // (NC fan-out is pinned to 1 in the RM compute contract).
        const uint32_t num_input_tiles = std::max(2U, plan.wt_tiles_per_chunk * plan.ht_tiles_per_chunk);
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_input_tiles * src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });
    } else if (use_width_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
        uint32_t num_input_tiles = 2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_input_tiles * src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = num_shard_tiles * src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
            .tensor = &a,
        });
    } else {
        uint32_t num_input_tiles = use_fpu_negate ? chunk_size : 2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_input_tiles * src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });
    }

    uint32_t scaler_cb_index = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scaler_cb_index),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = CBIndex::c_3;
    if (rm_path) {
        const uint32_t num_output_tiles = std::max(2U, plan.wt_tiles_per_chunk);
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    } else if (use_width_sharding) {
        uint32_t num_output_tiles = output.shard_spec().value().numel() / tile_hw;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
            .tensor = &output,
        });
    } else {
        uint32_t num_output_tiles = use_fpu_negate ? chunk_size : 2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    // Packed fp32 scalar passed to the compute kernel for mul_unary_tile post-reduction scaling.
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    if (use_fpu_negate) {
        // The reduce_h_neg kernel pushes ntiles tiles per inner-loop iteration
        // via push_back(ntiles).  The CB FIFO write pointer only wraps when
        // wr_ptr exactly reaches fifo_limit, so it is not enough for the CB
        // size to be a multiple of each individual push size — the cumulative
        // offset across the inner Ht loop must also wrap to 0 at the end of
        // each nc iteration.  Per nc, the kernel advances wr_ptr by
        // Ht * Wt_per_core regardless of how that splits into chunk_size and
        // partial pushes, so sizing the CB at Ht * Wt_per_core makes the
        // trajectory land on fifo_limit exactly.  For two core groups, the
        // single-CB option uses Ht * lcm(Wt_g1, Wt_g2) so the same allocation
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
        // uint32_t overflow before the L1 fit check / CB sizing.
        const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;

        // L1 fit check: c_4 (acc) and c_5 (ineg) are each sized at
        // negate_cb_tiles.  If the combined allocation would not fit in the
        // available L1 budget, the caller is expected to fall back to external
        // negation — see ttnn::prim::h_reduce_negate_fits_in_l1 in common.cpp,
        // which mirrors this calculation.
        const uint64_t per_cb_bytes = negate_cb_tiles * dst_single_tile_size;
        const uint64_t negate_cb_bytes = 2ull * per_cb_bytes;
        const auto lowest_address = device->lowest_occupied_compute_l1_address();
        uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
        const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        TT_FATAL(
            max_l1_space > base_addr,
            "Negate H reduce: L1 base allocator address {} >= lowest occupied address {}; no room for CBs",
            base_addr,
            max_l1_space);
        max_l1_space -= base_addr;
        TT_FATAL(
            negate_cb_bytes <= max_l1_space,
            "Negate H reduce: cb_acc + cb_ineg ({} B for {} tiles) would not fit in available L1 ({} B). "
            "Caller must use h_reduce_negate_fits_in_l1 to choose the external-negate fallback.",
            negate_cb_bytes,
            negate_cb_tiles,
            max_l1_space);
        // CBDescriptor.total_size is uint32_t; the L1 fit check above already
        // bounds per_cb_bytes by the per-core L1 budget (well under 4 GiB),
        // but assert the narrowing explicitly so any future budget change
        // surfaces here instead of producing a silently-truncated CB size.
        TT_FATAL(
            per_cb_bytes <= std::numeric_limits<uint32_t>::max(),
            "Negate H reduce: per-CB size {} B exceeds uint32_t total_size range",
            per_cb_bytes);
        const uint32_t per_cb_total_size = static_cast<uint32_t>(per_cb_bytes);

        uint32_t acc_cb_index = CBIndex::c_4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = per_cb_total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(acc_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });

        uint32_t ineg_cb_index = CBIndex::c_5;
        desc.cbs.push_back(CBDescriptor{
            .total_size = per_cb_total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(ineg_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    // Accurate fp32 mean: route Float32 SUM through the SFPU (needs 32-bit DEST)
    const bool fp32_sfpu_reduce = is_sfpu_reduce && a.dtype() == DataType::FLOAT32 && fp32_dest_acc_en;
    // RM reduce packs into a dst-format CB; when that differs from the input format (e.g. bf16 input
    // reduced into an FP32 partial for the H-axis-split stage 1) the packer must be reconfigured too.
    if (rm_path && dst_cb_data_format != src0_cb_data_format) {
        reduce_defines["REDUCE_RM_MIXED_FORMAT"] = "1";
    }

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // UnpackToDestFp32 unpacks c_0 straight into the fp32 DEST, bypassing the SrcA tf32 truncation.
    if (fp32_sfpu_reduce) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};

    if (rm_path) {
        std::vector<uint32_t> reader_compile_time_args =
            build_rm_reader_ct_args(plan, scaler_bits, a, ReduceOpDim::H, num_h_slices, slice_Ht);

        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_rm.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
        reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    } else if (use_width_sharding) {
        std::vector<uint32_t> reader_compile_time_args = {
            src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits, fp32_sfpu_reduce ? 1u : 0u};
        std::map<std::string, std::string> reader_defines;
        reader_defines["REDUCE_SCALER"] = "1";
        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_defines.insert(reduce_defines.begin(), reduce_defines.end());
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
        reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    } else {
        std::vector<uint32_t> reader_compile_time_args = {
            Ht, Wt, HtWt, scaler_bits, /*use_welford=*/0, fp32_sfpu_reduce ? 1u : 0u};
        TensorAccessorArgs(a).append_to(reader_compile_time_args);

        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        std::map<std::string, std::string> reader_defines;
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_defines.insert(reduce_defines.begin(), reduce_defines.end());

        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
        reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    }

    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};

    if (rm_path) {
        std::vector<uint32_t> writer_compile_time_args = build_rm_writer_ct_args(plan, output, ReduceOpDim::H);

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "writer_reduce_rm_scalar.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
        writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    } else if (use_width_sharding) {
        std::vector<uint32_t> writer_ct_args = {
            output_cb_index,
        };
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
        writer_desc.compile_time_args = writer_ct_args;
    } else {
        std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
        TensorAccessorArgs(output).append_to(writer_compile_time_args);

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
    }
    // For width-sharding, num_cols_per_core_group_1 == NC * shard_Wt. Expose (shard_Wt, NC)
    // to the compute kernel so its (nc, wt_chunk, ht, wt_in_chunk) iteration matches the
    // reader's per-batch tile layout.
    uint32_t compute_Wt = use_width_sharding ? (num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
    uint32_t compute_NC = use_width_sharding ? NC : 1;
    // reduce_rm.cpp (H path) expects {Ht, Wt, nc_per_reduce, post_mul_bits, wt_chunk, ht_chunk};
    // reduce.cpp / reduce_h_neg.cpp expect {Ht, Wt, NC, post_mul_bits}.
    std::vector<uint32_t> compute_kernel_args_group_1;
    if (rm_path) {
        // Per-slice tile count: the compute kernel reduces `slice_Ht` tiles per output (its H loop
        // bound). Equals plan.Ht_rm when num_h_slices == 1, so the non-split path is unchanged.
        compute_kernel_args_group_1 = build_rm_compute_ct_args(plan, slice_Ht, post_mul_scaler_bits);
    } else {
        compute_kernel_args_group_1 = {
            Ht,                          // Ht
            compute_Wt,                  // Wt
            compute_NC,                  // NC
            post_mul_scaler_bits,        // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
            fp32_sfpu_reduce ? 1u : 0u,  // enable_fp32_sfpu: route Float32 SUM through the SFPU
        };
    }

    // MIN on Int32 uses -MAX(-x) in reduce_h_neg.
    const std::string compute_kernel =
        rm_path ? std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp")
                : std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
                      (operation_attributes.negate ? "_h_neg" : "") + ".cpp";

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2;
        if (rm_path) {
            // RM kernel takes per-core counts via runtime args, so both groups share compile args.
            compute_kernel_args_group_2 = compute_kernel_args_group_1;
        } else {
            uint32_t compute_Wt_group_2 =
                use_width_sharding ? (num_cols_per_core_group_2 / NC) : num_cols_per_core_group_2;
            uint32_t compute_NC_group_2 = use_width_sharding ? NC : 1;
            compute_kernel_args_group_2 = {
                Ht,                          // Ht
                compute_Wt_group_2,          // Wt
                compute_NC_group_2,          // NC
                post_mul_scaler_bits,        // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
                fp32_sfpu_reduce ? 1u : 0u,  // enable_fp32_sfpu: route Float32 SUM through the SFPU
            };
        }

        KernelDescriptor d;
        d.kernel_source = compute_kernel;
        d.source_type = KernelDescriptor::SourceType::FILE_PATH;
        d.core_ranges = core_group_2;
        d.compile_time_args = compute_kernel_args_group_2;
        d.defines = {reduce_defines.begin(), reduce_defines.end()};
        d.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        compute_desc_g2 = std::move(d);
    }

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
            reader_desc.emplace_runtime_args(
                core,
                {
                    a,
                    num_output_tiles_local,
                    output_tiles_seen,
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    output,
                    num_output_tiles_local,
                    output_tiles_seen,
                });
            if (core_group_1.contains(core)) {
                compute_desc_g1.emplace_runtime_args(core, {num_output_tiles_local, output_tiles_seen});
            } else if (compute_desc_g2.has_value()) {
                compute_desc_g2->emplace_runtime_args(core, {num_output_tiles_local, output_tiles_seen});
            } else {
                TT_THROW("Reduce H (dense RM): core in core_group_2 but no second compute descriptor");
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
        KernelDescriptor::CoreRuntimeArgs reader_rt_args = {
            num_cols_per_core_group_1 * Ht, shard_Wt, Ht, NC, shard_row_size, shard_batch_size};
        KernelDescriptor::CoreRuntimeArgs writer_rt_args = {num_cols_per_core_group_1};
        // Width-sharded path: iterate the actual slice core set (all_cores), not the
        // grid_to_cores sequence — sharded grids may not start at (0,0).
        for (const auto& range : all_cores.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    CoreCoord core{x, y};
                    reader_desc.runtime_args.emplace_back(core, reader_rt_args);
                    writer_desc.runtime_args.emplace_back(core, writer_rt_args);
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
            reader_desc.emplace_runtime_args(
                core, {a, (num_cols_read / Wt * HtWt) + (num_cols_read % Wt), num_cols_read % Wt, num_cols_per_core});

            writer_desc.emplace_runtime_args(
                core,
                {
                    output,
                    num_cols_per_core,  // number of tiles to write
                    num_cols_read       // output tile start index
                });
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

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
