// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <variant>

namespace ttnn::prim {

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
    const bool use_height_sharding =
        !rm_path && a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
        output.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && a.shard_spec().has_value() &&
        output.shard_spec().has_value() && a.shard_spec()->grid == output.shard_spec()->grid &&
        a.shard_spec()->shape[0] == output.shard_spec()->shape[0] &&
        a.shard_spec()->orientation == output.shard_spec()->orientation &&
        shard_Ht * a.shard_spec()->grid.num_cores() == NC * Ht;

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
    spec.name = rm_path              ? "reduce_multi_core_w_dense_rm"
                : use_height_sharding ? "reduce_multi_core_w_height_sharded"
                                      : "reduce_multi_core_w";

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // Int32 max/min/sum use the SFPU reduce path; fp32 SUM only for the accurate mean opt-in.
    const bool is_sfpu_reduce =
        use_sfpu_reduce_path(a.dtype(), operation_attributes.math_op, operation_attributes.use_sfpu_reduce);
    const bool use_fpu_negate = operation_attributes.negate && !is_sfpu_reduce;

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

        // reduce_rm.cpp accumulates partial reductions across W chunks into the accumulator.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = plan.ht_tiles_per_chunk,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    uint32_t num_input_tiles = 2;
    if (rm_path) {
        num_input_tiles = std::max(num_input_tiles, plan.wt_tiles_per_chunk);
    }
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_cb_data_format,
    });

    if (use_height_sharding) {
        // A view onto the resident input shard rather than its own L1 allocation: the reader reads
        // the shard in place through this buffer.
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC1_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_shard_tiles,
            .data_format_metadata = src0_cb_data_format,
            .borrowed_from = INPUT_TENSOR,
        });
    }

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format,
    });

    if (use_height_sharding) {
        // A view onto the resident output shard: the packer writes the result in place, and the
        // writer's wait/pop is only a readiness handshake.
        uint32_t num_output_tiles = output.shard_spec().value().numel() / tile_hw;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = OUTPUT_TENSOR,
        });
    } else {
        constexpr uint32_t num_output_tiles = 2;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    if (use_fpu_negate) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    // Accurate fp32: route Float32 through the SFPU (needs 32-bit DEST)
    const bool fp32_sfpu_reduce = is_sfpu_reduce && a.dtype() == DataType::FLOAT32 && fp32_dest_acc_en;

    // ---- Reader kernel ----
    Group<DFBBinding> reader_dfb_bindings;
    Group<TensorBinding> reader_tensor_bindings;
    KernelSpec::CompileTimeArgs reader_ct_args;
    Group<std::string> reader_rta_names;
    std::string reader_source;
    std::map<std::string, std::string> reader_defines_map = reduce_defines;
    if (rm_path) {
        reader_source = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp";
        reader_ct_args = build_rm_reader_ct_args(plan, std::bit_cast<uint32_t>(operation_attributes.scaler));
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
    } else if (use_height_sharding) {
        reader_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_input_rows_partitioned_sharded.cpp";
        reader_ct_args = {{"scaler_bits", std::bit_cast<uint32_t>(operation_attributes.scaler)}};
        reader_rta_names = {"num_tiles"};
        // The sharded reader prepares the scaler tile itself (gated on REDUCE_SCALER).
        reader_defines_map["REDUCE_SCALER"] = "1";
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
            "reader_unary_reduce_universal_start_id.cpp";
        reader_ct_args = {{"scaler_bits", std::bit_cast<uint32_t>(operation_attributes.scaler)}};
        reader_rta_names = {"num_tiles", "start_id"};
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
    // On the tiled and height-sharded paths these are the Metal 2.0 forks of the eltwise/unary and
    // data_movement/sharded writers; their binding vocabulary is the forks', not this op's.
    std::string writer_source;
    KernelSpec::CompileTimeArgs writer_ct_args;
    Group<std::string> writer_rta_names;
    Group<TensorBinding> writer_tensor_bindings;
    if (rm_path) {
        writer_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_rm_scalar.cpp";
        writer_ct_args = build_rm_writer_ct_args(plan);
        writer_rta_names = {"rt_count", "rt_start"};
        writer_tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}};
    } else if (use_height_sharding) {
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
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reduce_defines)},
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
    // Legacy resolved a TTNN ComputeKernelConfig but forwarded only math_fidelity,
    // fp32_dest_acc_en and unpack_to_dest_mode onto ComputeConfigDescriptor, leaving
    // math_approx_mode and dst_full_sync_en at the *Metal* descriptor defaults (both false).
    // Reproduce that exactly: the TTNN helper would otherwise carry the caller's math_approx_mode
    // into sfpu_precision_mode and the caller's dst_full_sync_en into double_buffer_dest, silently
    // changing precision / Dest buffering.
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // std::visit rather than a Gen1-only get_if: to_compute_hardware_config yields a
    // ComputeGen2Config on Quasar, and the fields set below exist on both generations. The
    // explicit-unpack-mode requirement in particular is enforced generation-agnostically, so a
    // Gen1-only branch would leave FP32 + 32-bit-Dest programs failing ProgramSpec validation there.
    std::visit(
        [&](auto& compute_cfg) {
            compute_cfg.sfpu_precision_mode = Precision::Precise;  // legacy math_approx_mode = false
            compute_cfg.double_buffer_dest = true;                 // legacy dst_full_sync_en = false
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

    // For RM, per-group counts are logical rows; the compute kernel expects tile-row counts.
    const uint32_t ht_per_core_group_1 =
        rm_path ? (num_rows_per_core_group_1 + plan.rm_rows_per_tile - 1) / plan.rm_rows_per_tile
                : num_rows_per_core_group_1;
    const uint32_t ht_per_core_group_2 =
        rm_path ? (num_rows_per_core_group_2 + plan.rm_rows_per_tile - 1) / plan.rm_rows_per_tile
                : num_rows_per_core_group_2;

    // MIN on an SFPU path uses the base reduce.cpp kernel (negate=false); fast-mode float/bf16 MIN
    // uses -MAX(-x) in reduce_w_neg.
    const std::string compute_kernel =
        rm_path ? std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp")
                : std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
                      (operation_attributes.negate ? "_w_neg" : "") + ".cpp";

    std::map<std::string, std::string> compute_defines_map = reduce_defines;
    if (use_fpu_negate) {
        // reduce_w_neg.cpp is the FPU -MAX(-x) path; the acc / ineg scratch buffers it uses are
        // bound only in this case. The kernel gates its references to those buffers on this define,
        // because `if constexpr` cannot suppress name lookup.
        compute_defines_map["REDUCE_FPU_NEGATE"] = "1";
    }

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t ht_per_core_group) {
        Group<DFBBinding> dfb_bindings;
        KernelSpec::CompileTimeArgs ct_args;
        if (rm_path) {
            ct_args = build_rm_compute_ct_args(plan, ht_per_core_group, post_mul_scaler_bits, fp32_sfpu_reduce);
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
                {"Ht", ht_per_core_group},
                {"Wt", Wt},
                {"NC", 1u},
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
            .hw_config = compute_hw,
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1, ht_per_core_group_1));
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_G2, ht_per_core_group_2));
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
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    uint32_t out_dim_divider = Wt;

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    if (use_height_sharding) {
        // Each core streams its resident shard (shard_Ht rows x Wt cols) into the input buffer and
        // reduces each row over Wt -> shard_Ht output tiles written in place to the borrowed output
        // shard. Every core owns an identical local shard, so all cores get the same args.
        TT_FATAL(
            shard_Ht * num_cores == num_rows,
            "Height-sharded reduce W: {} shard rows across {} cores must cover all {} tile-rows",
            shard_Ht,
            num_cores,
            num_rows);
        // Height-sharded path: iterate the actual shard core set (all_cores), not the grid_to_cores
        // sequence — sharded grids may not start at (0,0).
        for (const CoreCoord& core : corerange_to_cores(all_cores)) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_tiles", shard_Ht * Wt}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_units", shard_Ht}});
        }
    } else {
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
            cores =
                grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
        }
        TT_FATAL(
            cores.size() == num_cores,
            "Resolved core list size {} must match split num_cores {}",
            cores.size(),
            num_cores);
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
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"rt_count", num_rows_per_core}, {"rt_start", num_rows_read}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"rt_count", num_rows_per_core}, {"rt_start", num_rows_read}});
                num_rows_read += num_rows_per_core;
            } else {
                uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"num_tiles", num_tensor_tiles_per_core},
                     // tile index of row to start reading from
                     {"start_id", num_tiles_read}});

                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {// number of tiles to write
                     {"num_pages", num_tensor_tiles_per_core / out_dim_divider},
                     // output tile start index
                     {"start_id", num_tiles_read / out_dim_divider}});
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
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{a});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
