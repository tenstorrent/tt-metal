// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <ttnn/metal_v2_artifacts.hpp>

#include <bit>
#include <string>
#include <utility>
#include <cstdint>

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
SoftmaxDeviceOperation::SoftmaxShardedProgramFactoryAttentionOptimized::create_program_artifacts(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryAttentionOptimizedSharded selected");

    const auto& input_tensor = tensor_args.input_tensor;
    auto* device = input_tensor.device();
    const auto arch = device->arch();

    TT_FATAL(
        input_tensor.is_sharded() && input_tensor.shard_spec().has_value(),
        "Input tensor must be sharded when using SoftmaxShardedMultiCoreProgramConfig");

    const bool has_mask = tensor_args.mask.has_value();
    // is_causal_mask without an actual mask is contradictory and unsupported. Fail cleanly here rather
    // than letting a CAUSAL_MASK-without-FUSED_SCALE_MASK kernel reach JIT and fail opaquely (the host
    // used to throw std::bad_optional_access on this input before the mask-arg reads were guarded).
    TT_FATAL(
        has_mask || !attributes.is_causal_mask,
        "Causal-mask softmax requires an attention mask (is_causal_mask=true with no mask is unsupported).");

    const tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config);

    const tt::DataFormat out0_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const tt::DataFormat mask_cb_data_format =
        has_mask ? datatype_to_dataformat_converter(tensor_args.mask->dtype()) : tt::DataFormat::Float16_b;
    const tt::DataFormat fused_attention_scale_cb_data_format = tt::DataFormat::Float16_b;
    const tt::DataFormat max_scaler_cb_data_format =
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const tt::DataFormat sum_scaler_cb_data_format =
        im_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    log_debug(tt::LogOp, "in0_cb_data_format: {}", in0_cb_data_format);
    log_debug(tt::LogOp, "out0_cb_data_format: {}", out0_cb_data_format);
    log_debug(tt::LogOp, "mask_cb_data_format: {}", mask_cb_data_format);
    log_debug(tt::LogOp, "im_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "fused_attention_scale_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "max_scaler_cb_data_format: {}", max_scaler_cb_data_format);
    log_debug(tt::LogOp, "sum_scaler_cb_data_format: {}", sum_scaler_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shard_orient = input_tensor.shard_spec().value().orientation;
    const auto& shape = input_tensor.padded_shape();
    const std::uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const std::uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const std::uint32_t tile_hw = input_tensor.tensor_spec().tile().get_tile_hw();
    std::uint32_t num_cores_per_batch = (shape[1] * shape[2] * shape[3]) / (input_tensor.shard_spec().value().shape[0] *
                                                                            input_tensor.shard_spec().value().shape[1]);

    std::uint32_t mask_H = shape[2];
    if (has_mask) {
        mask_H = tensor_args.mask->padded_shape()[2];
    }
    std::uint32_t mask_Ht = mask_H / tile_height;

    TT_FATAL(
        std::holds_alternative<SoftmaxShardedMultiCoreProgramConfig>(attributes.program_config),
        "Invalid softmax sharded program config for given tensor and sharding shape");
    SoftmaxShardedMultiCoreProgramConfig program_config =
        std::get<SoftmaxShardedMultiCoreProgramConfig>(attributes.program_config);
    std::uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;

    // single tile sizes
    std::uint32_t im_tile_size = tt::tile_size(im_cb_data_format);
    std::uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);
    std::uint32_t out0_tile_size = tt::tile_size(out0_cb_data_format);
    std::uint32_t mask_tile_size = tt::tile_size(mask_cb_data_format);
    std::uint32_t fused_attention_scale_tile_size = tt::tile_size(fused_attention_scale_cb_data_format);
    std::uint32_t max_scaler_tile_size = tt::tile_size(max_scaler_cb_data_format);
    std::uint32_t sum_scaler_tile_size = tt::tile_size(sum_scaler_cb_data_format);

    const bool mask_sharded = has_mask && tensor_args.mask->is_sharded();
    // c_3 is a borrowed-memory DFB the compute reads directly (self-loop) only when the *default*
    // sharded reader is selected — that reader alone honors SHARDED_CAUSAL_MASK and skips the mask read.
    // The hw-dims and row-major readers always stream the mask themselves (c_3 reader-produced, 1P+1C),
    // so the resident path must exclude them or those readers would reference args/DFBs the host withholds.
    const bool mask_sharded_resident =
        has_mask && attributes.is_causal_mask && mask_sharded && !attributes.is_scale_causal_mask_hw_dims_softmax;
    // hw_dims_only_causal_mask does not support RM Layout atm
    const bool use_row_major_kernel = has_mask && tensor_args.mask->layout() == tt::tt_metal::Layout::ROW_MAJOR;

    // c_3 (attn mask) entry count — mirrors legacy in3_CB_size / mask_tile_size.
    std::uint32_t attn_num_entries = program_config.block_w;
    if (attributes.is_causal_mask) {
        if (mask_sharded) {
            attn_num_entries = program_config.block_w * program_config.block_h;
        } else {
            attn_num_entries = program_config.block_w;
            if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
                // For some reason, if we have hw_dims_causal_mask version, single buffering is up to ~20% faster
                // Then double buffering CB3.
                attn_num_entries *= 2;
            }
        }
    }

    // define core ranges (full sharded grid)
    std::uint32_t num_cores_c = program_config.compute_with_storage_grid_size.x;
    std::uint32_t num_cores_r = program_config.compute_with_storage_grid_size.y;
    const CoreRangeSet all_device_cores{CoreRange({0, 0}, {num_cores_c - 1, num_cores_r - 1})};

    // ---- Resource names (program-scope; local to avoid unity-build symbol clashes) ----
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};

    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const TensorParamName MASK{"mask"};

    const DFBSpecName IN0{"in0"};
    const DFBSpecName MAX_SCALER{"max_scaler"};
    const DFBSpecName SUM_SCALER{"sum_scaler"};
    const DFBSpecName FUSED_SCALE{"fused_scale"};
    const DFBSpecName FUSED_ATTN{"fused_attn"};
    const DFBSpecName SCALE_MASK{"scale_mask"};
    const DFBSpecName EXPS{"exps"};
    const DFBSpecName RECIP_SUM_EXPS{"recip_sum_exps"};
    const DFBSpecName OUT0{"out0"};
    const DFBSpecName MAX{"max"};
    const DFBSpecName X{"x"};

    // ---- DataflowBuffers (mirrors legacy CB allocation; borrowed-memory DFBs for the sharded I/O) ----
    Group<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = in0_tile_size,
        .num_entries = program_config.block_w * program_config.block_h,
        .data_format_metadata = in0_cb_data_format,
        .borrowed_from = SRC});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = MAX_SCALER,
        .entry_size = max_scaler_tile_size,
        .num_entries = 1,
        .data_format_metadata = max_scaler_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SUM_SCALER,
        .entry_size = sum_scaler_tile_size,
        .num_entries = 1,
        .data_format_metadata = sum_scaler_cb_data_format});
    if (has_mask) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCALE_MASK,
            .entry_size = im_tile_size,
            .num_entries = program_config.block_w,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = FUSED_SCALE,
            .entry_size = fused_attention_scale_tile_size,
            .num_entries = 1,
            .data_format_metadata = fused_attention_scale_cb_data_format});
        DataflowBufferSpec attn_spec{
            .unique_id = FUSED_ATTN,
            .entry_size = mask_tile_size,
            .num_entries = attn_num_entries,
            .data_format_metadata = mask_cb_data_format};
        if (mask_sharded_resident) {
            attn_spec.borrowed_from = MASK;  // default sharded reader leaves c_3 resident (compute self-loop)
        }
        dfbs.push_back(attn_spec);
    }
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT0,
        .entry_size = out0_tile_size,
        .num_entries = program_config.block_w * program_config.block_h,
        .data_format_metadata = out0_cb_data_format,
        .borrowed_from = DST});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = EXPS,
        .entry_size = im_tile_size,
        .num_entries = program_config.block_w,
        .data_format_metadata = im_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = RECIP_SUM_EXPS,
        .entry_size = im_tile_size,
        .num_entries = 1,
        .data_format_metadata = im_cb_data_format});
    if (attributes.numeric_stable) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = MAX, .entry_size = im_tile_size, .num_entries = 1, .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = X,
            .entry_size = im_tile_size,
            .num_entries = program_config.block_w,
            .data_format_metadata = im_cb_data_format});
    }

    // ---- Defines ----
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (has_mask) {
        reader_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        reader_defines["CAUSAL_MASK"] = "1";
        // SHARDED_CAUSAL_MASK tells the compute to treat c_3 as resident (skip wait_front) and the default
        // reader to skip the mask read. Keyed on mask_sharded_resident so it stays off for the hw-dims path,
        // where the reader streams c_3 and the compute must wait for it (1P+1C).
        if (mask_sharded_resident) {
            reader_defines["SHARDED_CAUSAL_MASK"] = "1";
        }
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (has_mask) {
        compute_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.numeric_stable) {
        compute_defines["NUMERIC_STABLE"] = "1";
    }
    compute_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    compute_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";

    // ---- Reader kernel ----
    std::string reader_source;
    if (use_row_major_kernel) {
        reader_source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm_rm_mask.cpp";
    } else if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
        reader_source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm.cpp";
    } else {
        reader_source =
            std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm_causal_mask_hw_dims.cpp";
    }

    // The reader produces the reduce scalers always; the fused-scale / attn-mask bindings and its named args
    // depend on the mask config (and c_3 is borrowed, not reader-produced, on the sharded-resident path).
    Group<DFBBinding> reader_bindings = {
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensor_bindings;
    std::vector<std::string> reader_rta_names;
    KernelSpec::CompileTimeArgs reader_cta;
    if (has_mask) {
        reader_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_rta_names.push_back("pre_scale");
        const bool reader_reads_mask = !mask_sharded_resident;  // c_3 is borrowed on the resident path
        if (reader_reads_mask) {
            reader_bindings.push_back(DFBBinding{
                .dfb_spec_name = FUSED_ATTN,
                .accessor_name = "fused_attn",
                .endpoint_type = DFBEndpointType::PRODUCER});
            reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = MASK, .accessor_name = "mask"});
            reader_cta.insert({"block_w", program_config.block_w});
            reader_rta_names.push_back("mask_start_tile_id");
            if (use_row_major_kernel) {
                reader_cta.insert(
                    {"mask_float32", static_cast<std::uint32_t>(mask_cb_data_format == tt::DataFormat::Float32)});
            } else if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                reader_cta.insert({"block_ht", program_config.block_h});
                reader_rta_names.push_back("mask_num_tiles");
            } else if (attributes.is_causal_mask) {
                reader_cta.insert({"fused_head", program_config.block_h / mask_Ht});
                reader_cta.insert({"mask_block_ht", mask_Ht});
            }
        }
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_source,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    // ---- Compute kernel ----
    Group<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::CONSUMER},
    };
    auto add_self_loop = [&](const DFBSpecName& name, const char* accessor) {
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
    };
    if (has_mask) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::CONSUMER});
        add_self_loop(SCALE_MASK, "scale_mask");
        if (mask_sharded_resident) {
            add_self_loop(FUSED_ATTN, "fused_attn");  // borrowed-resident mask: compute is the only toucher
        } else {
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = FUSED_ATTN,
                .accessor_name = "fused_attn",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
    }
    if (attributes.numeric_stable) {
        add_self_loop(MAX, "max");
        add_self_loop(X, "x");
    }

    // Compute hardware config (Style A) + fp32 unpack modes for every Float32 DFB consumed.
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        auto& gen1 = std::get<ComputeGen1Config>(compute_hw);
        auto add_unpack = [&](const DFBSpecName& name, tt::DataFormat fmt) {
            if (fmt == tt::DataFormat::Float32) {
                gen1.unpack_modes.insert({name, tt::tt_metal::UnpackMode::UnpackToSrc});
            }
        };
        add_unpack(IN0, in0_cb_data_format);
        // out0 is a self-loop here (no writer; compute is CONSUMER as well as PRODUCER), so it needs an
        // unpack entry under fp32 — unlike the interleaved factory where out0 is producer-only.
        add_unpack(OUT0, out0_cb_data_format);
        add_unpack(MAX_SCALER, max_scaler_cb_data_format);
        add_unpack(SUM_SCALER, sum_scaler_cb_data_format);
        add_unpack(EXPS, im_cb_data_format);
        add_unpack(RECIP_SUM_EXPS, im_cb_data_format);
        if (has_mask) {
            add_unpack(FUSED_SCALE, fused_attention_scale_cb_data_format);
            add_unpack(FUSED_ATTN, mask_cb_data_format);
            add_unpack(SCALE_MASK, im_cb_data_format);
        }
        if (attributes.numeric_stable) {
            add_unpack(MAX, im_cb_data_format);
            add_unpack(X, im_cb_data_format);
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_sharded.cpp",
        .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = compute_bindings,
        .compile_time_args =
            {{"block_h", program_config.block_h},
             {"block_w", program_config.block_w},
             {"subblock_w", program_config.subblock_w},
             {"num_subblocks_w", num_subblocks_w},
             {"causal_mask", static_cast<std::uint32_t>(attributes.is_causal_mask)},
             {"sharded_causal_mask", static_cast<std::uint32_t>(mask_sharded_resident)},
             {"numeric_stable", static_cast<std::uint32_t>(attributes.numeric_stable)}},
        .hw_config = compute_hw,
    };

    // ---- Assemble spec ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = SRC, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = DST, .spec = output_tensor.tensor_spec()}};
    if (has_mask) {
        tensor_parameters.push_back(TensorParameter{.unique_id = MASK, .spec = tensor_args.mask->tensor_spec()});
    }

    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {READER, COMPUTE}, .target_nodes = all_device_cores});

    ProgramSpec spec{
        .name = "softmax_attention_optimized_sharded",
        .kernels = {reader, compute},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    // ---- Run args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};

    const std::uint32_t scale_u = std::bit_cast<std::uint32_t>(attributes.scale.value_or(1.0f));
    std::uint32_t mask_start_tile_id = 0;

    std::uint32_t num_tiles_in_attn_mask = 0;
    std::uint32_t num_tiles_of_attn_mask_needed_per_core = 0;
    if (has_mask && attributes.is_scale_causal_mask_hw_dims_softmax) {
        num_tiles_in_attn_mask = tensor_args.mask->padded_shape()[-1] * tensor_args.mask->padded_shape()[-2] / tile_hw;
        num_tiles_of_attn_mask_needed_per_core = program_config.block_h * program_config.block_w;
    }
    std::uint32_t num_cores_per_batch_index = 0;

    auto emit_reader_args = [&](const CoreCoord& core) {
        if (has_mask) {
            AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"pre_scale", scale_u}});
            if (!mask_sharded_resident) {
                AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"mask_start_tile_id", mask_start_tile_id}});
                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    AddRuntimeArgsForNode(
                        reader_ra.runtime_arg_values, core, {{"mask_num_tiles", num_tiles_in_attn_mask}});
                }
            }
        }

        num_cores_per_batch_index++;
        if (attributes.is_scale_causal_mask_hw_dims_softmax) {
            mask_start_tile_id = (mask_start_tile_id + num_tiles_of_attn_mask_needed_per_core) % num_tiles_in_attn_mask;
        } else if (num_cores_per_batch_index == num_cores_per_batch) {
            num_cores_per_batch_index = 0;
            if (has_mask) {
                if (attributes.is_causal_mask) {
                    mask_start_tile_id += tensor_args.mask->padded_shape()[-1] * tensor_args.mask->padded_shape()[-2] /
                                          tile_width / tile_height;
                } else {
                    mask_start_tile_id += use_row_major_kernel ? tensor_args.mask->padded_shape()[-2]
                                                               : tensor_args.mask->padded_shape()[-1] / tile_width;
                }
            }
        }
    };

    if (shard_orient == tt::tt_metal::ShardOrientation::COL_MAJOR) {
        for (std::uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            for (std::uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
                emit_reader_args(CoreCoord{core_idx_x, core_idx_y});
            }
        }
    } else {
        for (std::uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (std::uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                emit_reader_args(CoreCoord{core_idx_x, core_idx_y});
            }
        }
    }

    if (!reader_ra.runtime_arg_values.empty()) {
        run_args.kernel_run_args = {std::move(reader_ra)};
    }
    run_args.tensor_args.emplace(SRC, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(DST, output_tensor.mesh_tensor());
    if (has_mask) {
        run_args.tensor_args.emplace(MASK, tensor_args.mask->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
