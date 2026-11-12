// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

// [Quasar bring-up: reduce-W metal2]  LEGACY ProgramDescriptor factory, now reachable ONLY for the
// row-major dense path (operation_attributes.row_major_w_dense_path). Its three kernels have no
// Metal 2.0 twin, so it is kept verbatim rather than ported; the TILE path moved to the Metal 2.0
// ReduceMultiCoreWProgramFactory below. On Quasar this path still aborts in CreateDataMovementKernel,
// exactly as it did before this change.
tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreWRmProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const bool rm_path = operation_attributes.row_major_w_dense_path;
    const auto& shape = a.padded_shape();
    const auto& logical_shape = a.logical_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = tt::div_up(W, tile_width);
    uint32_t Ht = tt::div_up(H, tile_height);

    if (rm_path) {
        validate_rm_preconditions(
            a, output, operation_attributes.math_op, operation_attributes.negate, ReduceOpDim::W, "Reduce W");
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

    // Populate the RM-only locals (chunk sizes, page bytes, padding identity, datum sizes) into
    // a single struct so the per-site formulas don't drift between this factory and the H one.
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
            ReduceOpDim::W);
    }

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // RM splits NC*H_logical row-wise so each core gets contiguous logical rows; tile path
    // keeps the existing NC*Ht slicing.
    const uint32_t num_rows = rm_path ? (NC * plan.H_logical) : (NC * Ht);
    constexpr bool k_split_rows_row_wise = true;
    const bool split_row_wise = rm_path;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows, split_row_wise);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, split_row_wise);
    }
    TT_FATAL(num_cores > 0, "Reduce W requires at least one worker core");

    ProgramDescriptor desc;

    if (rm_path) {
        constexpr uint32_t cb_rm = tt::CBIndex::c_24;
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

        constexpr uint32_t cb_clear_value = tt::CBIndex::c_4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_clear_value),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });

        // reduce_rm.cpp accumulates partial reductions across W chunks into cb_acc (c_5).
        constexpr uint32_t cb_acc = tt::CBIndex::c_5;
        desc.cbs.push_back(CBDescriptor{
            .total_size = plan.ht_tiles_per_chunk * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_acc),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    if (rm_path) {
        num_input_tiles = std::max(num_input_tiles, plan.wt_tiles_per_chunk);
    }
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
        .total_size = scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    std::vector<uint32_t> reader_compile_time_args;
    if (rm_path) {
        reader_compile_time_args =
            build_rm_reader_ct_args(plan, std::bit_cast<uint32_t>(operation_attributes.scaler), a, ReduceOpDim::W);
    } else {
        reader_compile_time_args = {std::bit_cast<uint32_t>(operation_attributes.scaler)};
        TensorAccessorArgs(a).append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args;
    if (rm_path) {
        writer_compile_time_args = build_rm_writer_ct_args(plan, output, ReduceOpDim::W);
    } else {
        writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
        TensorAccessorArgs(output).append_to(writer_compile_time_args);
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils_qsr::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        rm_path ? "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/dataflow/"
                  "reader_unary_reduce_rm.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/dataflow/"
                  "reader_unary_reduce_universal_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = rm_path ? "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/"
                                          "kernels/dataflow/writer_reduce_rm_scalar.cpp"
                                        : "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/"
                                          "kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // For RM, per-group counts are logical rows; the compute kernel expects tile-row counts.
    const uint32_t ht_per_core_group_1 =
        rm_path ? (num_rows_per_core_group_1 + plan.rm_rows_per_tile - 1) / plan.rm_rows_per_tile
                : num_rows_per_core_group_1;
    const uint32_t ht_per_core_group_2 =
        rm_path ? (num_rows_per_core_group_2 + plan.rm_rows_per_tile - 1) / plan.rm_rows_per_tile
                : num_rows_per_core_group_2;

    // reduce_rm.cpp expects {Ht, Wt, nc_per_reduce, post_mul_bits, wt_chunk, ht_chunk};
    // reduce.cpp expects {Ht, Wt, NC, post_mul_bits}.
    std::vector<uint32_t> compute_kernel_args_group_1;
    if (rm_path) {
        compute_kernel_args_group_1 = build_rm_compute_ct_args(plan, ht_per_core_group_1, post_mul_scaler_bits);
    } else {
        compute_kernel_args_group_1 = {
            ht_per_core_group_1,   // Ht
            Wt,                    // Wt
            1,                     // NC
            post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
        };
    }

    // MIN (negate) and INT32 reduce are rejected in validate() on Quasar (negative_tile / sfpu_reduce
    // are unported), so only the plain MAX/SUM reduce.cpp compute kernel is selected here.
    const std::string compute_kernel =
        rm_path
            ? std::string(
                  "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/compute/reduce_rm.cpp")
            : std::string(
                  "ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/compute/reduce.cpp");

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2;
        if (rm_path) {
            compute_kernel_args_group_2 = build_rm_compute_ct_args(plan, ht_per_core_group_2, post_mul_scaler_bits);
        } else {
            compute_kernel_args_group_2 = {
                ht_per_core_group_2,   // Ht
                Wt,                    // Wt
                1,                     // NC
                post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
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
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        compute_desc_g2 = std::move(d);
    }

    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    uint32_t out_dim_divider = Wt;
    std::vector<CoreCoord> cores;
    if (rm_path) {
        // Match the row-wise split so core[i] receives the i-th contiguous row chunk.
        cores = corerange_to_cores(all_cores, std::nullopt, k_split_rows_row_wise);
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
    TT_FATAL(num_rows == 0 || !cores.empty(), "Non-zero reduce workload requires non-empty core list");
    for (uint32_t i = 0, num_tiles_read = 0, num_rows_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        if (rm_path) {
            // RM split distributes logical rows, so num_rows_per_core IS the logical-row count.
            // Use raw addresses (not Buffer*) so mesh program-cache fast paths re-apply per-core args.
            reader_desc.emplace_runtime_args(
                core,
                {
                    a,
                    num_rows_per_core,
                    num_rows_read,
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    output,
                    num_rows_per_core,
                    num_rows_read,
                });
            num_rows_read += num_rows_per_core;
        } else {
            uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
            reader_desc.emplace_runtime_args(
                core,
                {
                    a,
                    num_tensor_tiles_per_core,
                    num_tiles_read  // tile index of row to start reading from
                });

            writer_desc.emplace_runtime_args(
                core,
                {
                    output,
                    num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                    num_tiles_read / out_dim_divider              // output tile start index
                });
            num_tiles_read += num_tensor_tiles_per_core;
        }
        if (i == num_cores - 1) {
            if (rm_path) {
                TT_FATAL(
                    num_rows_read == num_rows,
                    "Reduce W (dense RM) assigned {} logical rows across cores, expected {}",
                    num_rows_read,
                    num_rows);
            } else {
                TT_FATAL(
                    num_tiles_read == num_rows * Wt,
                    "Reduce W assigned {} input tiles, expected {}",
                    num_tiles_read,
                    num_rows * Wt);
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

// ============================================================================
// [Quasar bring-up: reduce-W metal2]  Metal 2.0 port of the multi-core W reduce
// factory -- SCOPED to the TILE (non row-major-dense) path.
//
// Why: the legacy factory above builds a ProgramDescriptor whose reader/writer carry
// Reader/WriterConfigDescriptor{}. On Gen2 that reaches CreateDataMovementKernel and
// TT_FATALs with "DataMovementKernel is not supported on Quasar" (kernel.hpp:451), so
// every ReduceMean / ReduceMax that lands on the W-parallelised strategy aborts.
//
// The port is host-only: all three Metal 2.0 kernels this path needs already exist in
// this directory and are already exercised by the sibling single-core-HW factory --
//   dataflow/reader_unary_reduce_universal_start_id_metal2.cpp
//   dataflow/writer_unary_interleaved_start_id_metal2.cpp
//   compute/reduce_metal2.cpp
// so reduce_op_single_core_hw_program_factory.cpp (same three kernels, one core) and
// reduce_op_multi_core_h_program_factory.cpp (same three resources, multi-core work
// split) are between them a line-for-line template.
//
// NOT ported: the row-major dense path (rm_path). Its three kernels
// (reader_unary_reduce_rm.cpp, writer_reduce_rm_scalar.cpp, compute/reduce_rm.cpp) have
// no Metal 2.0 twin, and a device-kernel rewrite is out of scope here. That path keeps
// the legacy ProgramDescriptor factory above, now reachable as
// ReduceMultiCoreWRmProgramFactory and selected in reduce_op_device_operation.cpp when
// operation_attributes.row_major_w_dense_path is set. Nothing about its behaviour
// changes. (It is opt-in via reduce_op.cpp's use_row_major_support, which defaults to
// false and which no in-tree caller sets, so in practice only the path below runs.)
//
// negate (MIN-via-negate) and INT32 reduce are rejected for every path in
// validate_on_program_cache_miss() on Quasar, so no fused-negate / SFPU variant is
// needed here.
ttnn::device_operation::ProgramArtifacts
ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = tt::div_up(W, tile_width);
    uint32_t Ht = tt::div_up(H, tile_height);

    // The row-major dense branch has no Metal 2.0 kernels; it is routed to the legacy
    // factory by select_program_factory, so reaching here with it set is a wiring bug.
    TT_FATAL(
        !operation_attributes.row_major_w_dense_path,
        "Reduce MultiCoreW Metal 2.0 port does not cover the row-major dense path "
        "(it has no Metal 2.0 kernels); ReduceMultiCoreWRmProgramFactory handles it.");
    TT_FATAL(
        !operation_attributes.negate,
        "Reduce MultiCoreW Metal 2.0 port does not cover negate (MIN); negative_tile is unported on Quasar.");
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device().arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    IDevice* device = &a.mutable_device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Tile path: one unit of work is one tile-row (Wt input tiles -> 1 output tile).
    const uint32_t num_rows = NC * Ht;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
    }
    TT_FATAL(num_cores > 0, "Reduce W requires at least one worker core");

    // For MAX (and, on Quasar, SUM/AVG) with a non-unity scalar the device reduces with
    // scaler=1.0 and the user scalar is applied afterwards by the compute kernel; see reduce_op.cpp.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- Resource names ----
    const DFBSpecName IN{"in"};          // legacy c_0
    const DFBSpecName SCALER{"scaler"};  // legacy c_2
    const DFBSpecName OUT{"out"};        // legacy c_3
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = src0_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = src0_cb_data_format};
    DataflowBufferSpec scaler_dfb{
        .unique_id = SCALER,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format};
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format};

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reduce defines ----
    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils_qsr::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : reduce_defines) {
        compute_defines.emplace(k, v);
    }
    // The reader needs REDUCE_OP / REDUCE_DIM for prepare_reduce_scaler<>.
    KernelSpec::CompilerOptions::Defines reader_defines = compute_defines;

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/");

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_reduce_universal_start_id_metal2.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"scaler_bits", scaler_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // One compute kernel per work-split group. For a W reduce the per-core quantity is the
    // number of tile-rows (Ht); Wt is the whole reduced extent and is the same on every core.
    auto make_compute = [&](const KernelSpecName& id, uint32_t compute_Ht) {
        return KernelSpec{
            .unique_id = id,
            .source = kdir / "compute/reduce_metal2.cpp",
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"Ht", compute_Ht}, {"Wt", Wt}, {"NC", 1u}, {"post_mul_scaler_bits", post_mul_scaler_bits}},
            .hw_config = ttnn::to_compute_hardware_config(
                device->arch(),
                ttnn::ComputeKernelConfig{
                    .math_fidelity = math_fidelity,
                    .math_approx_mode = false,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en}),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_G1, num_rows_per_core_group_1));
    work_units.push_back(
        WorkUnitSpec{.name = "reduce_w_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
        work_units.push_back(
            WorkUnitSpec{.name = "reduce_w_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args (mirror the legacy interleaved work-distribution loop) ----
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
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);

    const uint32_t out_dim_divider = Wt;
    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {
                {"num_tiles", num_tensor_tiles_per_core},
                {"start_id", num_tiles_read},  // tile index of the row to start reading from
            });
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"num_pages", num_tensor_tiles_per_core / out_dim_divider},
                {"start_id", num_tiles_read / out_dim_divider},
            });
        num_tiles_read += num_tensor_tiles_per_core;
        if (i == num_cores - 1) {
            TT_FATAL(
                num_tiles_read == num_rows * Wt,
                "Reduce W assigned {} input tiles, expected {}",
                num_tiles_read,
                num_rows * Wt);
        }
    }

    ProgramSpec spec{
        .name = "reduce_multi_core_w",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, scaler_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)}};
    run_args.tensor_args = {{INPUT, a}, {OUTPUT, output}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
