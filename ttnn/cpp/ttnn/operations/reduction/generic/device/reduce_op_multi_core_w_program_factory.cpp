// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <map>

namespace ttnn::prim {

namespace {
// desc.kernels push order at both exits of create_descriptor. The group-2 compute kernel exists
// only when the work split leaves a second core group.
enum : uint32_t { kWReaderIdx = 0, kWWriterIdx = 1, kWComputeG1Idx = 2, kWComputeG2Idx = 3 };

// Height-sharded fast path: each core reduces its own L1-resident shard, aliased as a CB, so the
// reader/writer runtime args carry no tensor and slot 0 there is a tile count, not an address.
bool w_uses_height_sharding(bool rm_path, const tt::tt_metal::MeshTensor& a, const tt::tt_metal::MeshTensor& output) {
    using tt::tt_metal::TensorMemoryLayout;
    if (rm_path || a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED ||
        output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED || !a.shard_spec().has_value() ||
        !output.shard_spec().has_value()) {
        return false;
    }
    const auto& shape = a.padded_shape();
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t NC = shape[0] * shape[1];
    const uint32_t Ht = tt::div_up(shape[2], tile_height);
    const uint32_t shard_Ht = a.shard_spec()->shape[0] / tile_height;
    return a.shard_spec()->grid == output.shard_spec()->grid &&
           a.shard_spec()->shape[0] == output.shard_spec()->shape[0] &&
           a.shard_spec()->orientation == output.shard_spec()->orientation &&
           shard_Ht * a.shard_spec()->grid.num_cores() == NC * Ht;
}

// True when the work split leaves a second core group, i.e. create_descriptor pushed kWComputeG2Idx.
// create_descriptor derives num_rows the same way, below.
bool w_has_second_compute_group(
    const ReduceDeviceOperation::operation_attributes_t& attrs, const tt::tt_metal::MeshTensor& a) {
    const bool rm_path = attrs.row_major_w_dense_path;
    const auto& shape = a.padded_shape();
    const uint32_t NC = shape[0] * shape[1];
    const uint32_t Ht = tt::div_up(shape[2], a.tensor_spec().tile().get_height());
    const uint32_t num_rows = rm_path ? (NC * a.logical_shape()[2]) : (NC * Ht);
    const auto split =
        attrs.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(*attrs.sub_core_grids, num_rows, rm_path)
            : tt::tt_metal::split_work_to_cores(a.mutable_device().compute_with_storage_grid_size(), num_rows, rm_path);
    return !std::get<3>(split).ranges().empty();
}
}  // namespace

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create_descriptor(
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
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    uint32_t Wt = tt::div_up(W, tile_width);
    uint32_t Ht = tt::div_up(H, tile_height);

    // Height-sharded fast path: each core reduces its own L1-resident shard locally instead of
    // gathering tiles over the NoC. Needs matching shard grid, height and orientation, plus shards
    // that tile the tensor exactly; anything else falls through to the generic path.
    const uint32_t shard_Ht = a.shard_spec().has_value() ? a.shard_spec()->shape[0] / tile_height : 0;
    const bool use_height_sharding = w_uses_height_sharding(rm_path, a, output);

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

    // Height-sharded: pin the worker set to the shard grid and give each core exactly the rows
    // (tile-rows) in its own shard. Each row reduces over the full Wt columns -> one output tile.
    if (use_height_sharding) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_Ht;
        num_rows_per_core_group_2 = 0;
    }

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
    uint32_t src1_cb_index = tt::CBIndex::c_1;
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

    // Height-sharded: alias the resident input shard as c_1 so the reader can stream it locally.
    if (use_height_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
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
    }

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
    // Height-sharded: alias the resident output shard as c_3 (write results in place). Otherwise a
    // 2-tile scratch CB feeding the interleaved writer.
    const uint32_t num_output_tiles = use_height_sharding ? output.shard_spec().value().numel() / tile_hw : 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .tensor = use_height_sharding ? &output : nullptr,
    });

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.scaler_mode == ScalerMode::PostMul;
    // Both scalars ride as common runtime arg 0 (reader: scaler, compute: post-mul), keeping them
    // out of the program hash and out of the kernel binaries.
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // Int32 max/min/sum use the SFPU reduce path; fp32 SUM only for the accurate mean opt-in.
    const bool is_sfpu_reduce =
        use_sfpu_reduce_path(a.dtype(), operation_attributes.math_op, operation_attributes.use_sfpu_reduce);
    const bool use_fpu_negate = operation_attributes.negate && !is_sfpu_reduce;

    std::vector<uint32_t> reader_compile_time_args;
    if (rm_path) {
        reader_compile_time_args = build_rm_reader_ct_args(plan, a, ReduceOpDim::W);
    } else if (use_height_sharding) {
        reader_compile_time_args = {src0_cb_index, src1_cb_index, CBIndex::c_2};
    } else {
        TensorAccessorArgs(a).append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args;
    if (rm_path) {
        writer_compile_time_args = build_rm_writer_ct_args(plan, output, ReduceOpDim::W);
    } else {
        writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
        if (!use_height_sharding) {  // interleaved writer also needs the tensor accessor
            TensorAccessorArgs(output).append_to(writer_compile_time_args);
        }
    }

    if (use_fpu_negate) {
        uint32_t acc_cb_index = tt::CBIndex::c_4;
        uint32_t num_acc_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_acc_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(acc_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });

        uint32_t inv_cb_index = tt::CBIndex::c_5;
        uint32_t num_inv_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_inv_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(inv_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    // Accurate fp32: route Float32 through the SFPU (needs 32-bit DEST)
    const bool fp32_sfpu_reduce = is_sfpu_reduce && a.dtype() == DataType::FLOAT32 && fp32_dest_acc_en;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // UnpackToDestFp32 unpacks c_0 (and RM cb_acc) into the fp32 DEST, bypassing SrcA tf32.
    if (fp32_sfpu_reduce) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        if (rm_path) {
            unpack_to_dest_mode[CBIndex::c_5] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        rm_path ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp"
        : use_height_sharding ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                "reader_unary_reduce_input_rows_partitioned_sharded.cpp"
                              : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                "reader_unary_reduce_universal_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    // The sharded reader prepares the scaler tile itself (gated on REDUCE_SCALER).
    auto reader_defines = reduce_defines;
    if (use_height_sharding) {
        reader_defines["REDUCE_SCALER"] = "1";
    }
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        rm_path ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_rm_scalar.cpp"
        : use_height_sharding
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
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

    // reduce_rm.cpp expects {Ht, Wt, nc_per_reduce, wt_chunk, ht_chunk, fp32};
    // reduce.cpp / reduce_w_neg.cpp expect {Ht, Wt, NC, fp32}. post_mul is common runtime arg 0.
    std::vector<uint32_t> compute_kernel_args_group_1;
    if (rm_path) {
        compute_kernel_args_group_1 = build_rm_compute_ct_args(plan, ht_per_core_group_1, fp32_sfpu_reduce);
    } else {
        compute_kernel_args_group_1 = {
            ht_per_core_group_1,         // Ht
            Wt,                          // Wt
            1,                           // NC
            fp32_sfpu_reduce ? 1u : 0u,  // enable_fp32_sfpu: route Float32 through the SFPU
        };
    }

    // MIN on an SFPU path uses the base reduce.cpp kernel (negate=false); fast-mode float/bf16 MIN
    // uses -MAX(-x) in reduce_w_neg.
    const std::string compute_kernel =
        rm_path ? std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp")
                : std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
                      (operation_attributes.negate ? "_w_neg" : "") + ".cpp";

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
            compute_kernel_args_group_2 = build_rm_compute_ct_args(plan, ht_per_core_group_2, fp32_sfpu_reduce);
        } else {
            compute_kernel_args_group_2 = {
                ht_per_core_group_2,         // Ht
                Wt,                          // Wt
                1,                           // NC
                fp32_sfpu_reduce ? 1u : 0u,  // enable_fp32_sfpu: route Float32 through the SFPU
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
    if (use_height_sharding) {
        // Each core streams its resident shard (shard_Ht rows x Wt cols) into c_0 and reduces each
        // row over Wt -> shard_Ht output tiles written in place to the aliased output shard.
        // Every core owns an identical local shard, so all cores get the same args.
        // The generic path's coverage check is past the early return, so restate it here.
        TT_FATAL(
            shard_Ht * num_cores == num_rows,
            "Height-sharded reduce W: {} shard rows across {} cores must cover all {} tile-rows",
            shard_Ht,
            num_cores,
            num_rows);
        KernelDescriptor::CoreRuntimeArgs reader_rt_args = {shard_Ht * Wt};
        KernelDescriptor::CoreRuntimeArgs writer_rt_args = {shard_Ht};
        for (const CoreCoord& core : corerange_to_cores(all_cores)) {
            reader_desc.runtime_args.emplace_back(core, reader_rt_args);
            writer_desc.runtime_args.emplace_back(core, writer_rt_args);
        }
        reader_desc.emplace_common_runtime_args({scaler_bits});
        compute_desc_g1.emplace_common_runtime_args({post_mul_scaler_bits});
        if (compute_desc_g2.has_value()) {
            compute_desc_g2->emplace_common_runtime_args({post_mul_scaler_bits});
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        desc.kernels.push_back(std::move(compute_desc_g1));
        if (compute_desc_g2.has_value()) {
            desc.kernels.push_back(std::move(*compute_desc_g2));
        }
        return desc;
    }
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

    reader_desc.emplace_common_runtime_args({scaler_bits});
    compute_desc_g1.emplace_common_runtime_args({post_mul_scaler_bits});
    if (compute_desc_g2.has_value()) {
        compute_desc_g2->emplace_common_runtime_args({post_mul_scaler_bits});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

void ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Only the scalars and the buffer addresses vary per dispatch; per-core counts and offsets
    // derive from the hashed shape and grid, so a hit already holds them.
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    tt::tt_metal::GetCommonRuntimeArgs(program, kWReaderIdx)[0] = std::bit_cast<uint32_t>(operation_attributes.scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);
    tt::tt_metal::GetCommonRuntimeArgs(program, kWComputeG1Idx)[0] = post_mul_scaler_bits;
    if (w_has_second_compute_group(operation_attributes, a)) {
        tt::tt_metal::GetCommonRuntimeArgs(program, kWComputeG2Idx)[0] = post_mul_scaler_bits;
    }

    if (w_uses_height_sharding(operation_attributes.row_major_w_dense_path, a, output)) {
        // The shards reach the kernels as aliased CBs, not as runtime-arg addresses.
        patch_cached_cb_address(program, tt::CBIndex::c_1, a);
        patch_cached_cb_address(program, tt::CBIndex::c_3, output);
    } else {
        patch_cached_arg_on_all_cores(program, kWReaderIdx, 0, a.mesh_buffer().get_reference_buffer()->address());
        patch_cached_arg_on_all_cores(program, kWWriterIdx, 0, output.mesh_buffer().get_reference_buffer()->address());
    }
}

}  // namespace ttnn::prim
