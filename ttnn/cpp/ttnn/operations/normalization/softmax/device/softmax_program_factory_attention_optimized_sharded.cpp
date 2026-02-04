// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory_attention_optimized_sharded.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <utility>

namespace ttnn::prim {
SoftmaxShardedProgramFactoryAttentionOptimized::cached_program_t SoftmaxShardedProgramFactoryAttentionOptimized::create(
    const SoftmaxParams& attributes, const SoftmaxInputs& tensor_args, Tensor& output_tensor) {
    tt::tt_metal::Program program{};
    auto* device = tensor_args.input_tensor.device();

    // Guard against non-sharded inputs when using a sharded program config
    TT_FATAL(
        tensor_args.input_tensor.is_sharded() && tensor_args.input_tensor.shard_spec().has_value(),
        "Input tensor must be sharded when using SoftmaxShardedMultiCoreProgramConfig");

    // convert data format
    tt::DataFormat in0_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    tt::DataFormat out0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat mask_cb_data_format = tensor_args.mask.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(tensor_args.mask->dtype())
                                             : tt::DataFormat::Float16_b;
    tt::DataFormat scale_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;

    log_debug(tt::LogOp, "in0_cb_data_format: {}", in0_cb_data_format);
    log_debug(tt::LogOp, "out0_cb_data_format: {}", out0_cb_data_format);
    log_debug(tt::LogOp, "mask_cb_data_format: {}", mask_cb_data_format);
    log_debug(tt::LogOp, "im_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "scale_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "scalar_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shard_orient = tensor_args.input_tensor.shard_spec().value().orientation;
    const auto& shape = tensor_args.input_tensor.padded_shape();
    const uint32_t tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_hw = tensor_args.input_tensor.tensor_spec().tile().get_tile_hw();
    uint32_t num_cores_per_batch =
        (shape[1] * shape[2] * shape[3]) / (tensor_args.input_tensor.shard_spec().value().shape[0] *
                                            tensor_args.input_tensor.shard_spec().value().shape[1]);

    uint32_t mask_H = shape[2];
    if (tensor_args.mask.has_value()) {
        mask_H = tensor_args.mask->padded_shape()[2];
    }
    uint32_t mask_Ht = mask_H / tile_height;
    // block

    if (!std::holds_alternative<SoftmaxShardedMultiCoreProgramConfig>(attributes.program_config)) {
        TT_THROW("Invalid softmax sharded program config for given tensor and sharding shape");
    }
    SoftmaxShardedMultiCoreProgramConfig program_config =
        std::get<SoftmaxShardedMultiCoreProgramConfig>(attributes.program_config);
    uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;

    // single tile sizes
    uint32_t im_tile_size = tt::tile_size(im_cb_data_format);
    uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);
    uint32_t out0_tile_size = tt::tile_size(out0_cb_data_format);
    uint32_t mask_tile_size = tt::tile_size(mask_cb_data_format);
    uint32_t scale_tile_size = tt::tile_size(scale_cb_data_format);
    uint32_t scalar_tile_size = tt::tile_size(scalar_cb_data_format);
    // in out buffer
    auto* src0_buffer = tensor_args.input_tensor.buffer();
    auto* out0_buffer = output_tensor.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_CB_size = program_config.block_w * program_config.block_h * in0_tile_size;
    // scaler for reduce coming from reader
    uint32_t in1_CB_size = 1 * scalar_tile_size;
    // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in2_CB_size = 1 * scale_tile_size;
    // attention mask
    uint32_t in3_CB_size;
    if (attributes.is_causal_mask) {
        if (tensor_args.mask.value().is_sharded()) {
            in3_CB_size = program_config.block_w * program_config.block_h * mask_tile_size;
        } else {
            in3_CB_size = program_config.block_w * mask_tile_size;
            if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
                // For some reason, if we have hw_dims_causal_mask version, single buffering is up to ~20% faster
                // Then double buffering CB3.
                in3_CB_size *= 2;
            }
        }
    } else {
        in3_CB_size = program_config.block_w * mask_tile_size;
    }
    // cb_exps - keeps exps in tt::CBIndex in L1 to avoid recomputing
    uint32_t im0_CB_size = program_config.block_w * im_tile_size;
    // 1/sum(exp(x))
    uint32_t im1_CB_size = 1 * im_tile_size;
    // attn mask im
    uint32_t im2_CB_size = program_config.block_w * im_tile_size;
    // output buffer size
    uint32_t out_CB_size = program_config.block_w * program_config.block_h * out0_tile_size;
    // numeric_stable cb max
    uint32_t max_CB_size = 1 * im_tile_size;
    uint32_t x_CB_size = program_config.block_w * im_tile_size;

    // define core ranges
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = program_config.compute_with_storage_grid_size.x;
    uint32_t num_cores_r = program_config.compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_c * num_cores_r;
    CoreRange all_device_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    // reader compile arg
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)program_config.block_w};
    tt::tt_metal::TensorAccessorArgs(tensor_args.mask ? tensor_args.mask->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    std::map<std::string, std::string> softmax_defines;
    // hw_dims_only_causal_mask does not support RM Layout atm
    bool use_row_major_kernel =
        (tensor_args.mask.has_value() and tensor_args.mask->layout() == tt::tt_metal::Layout::ROW_MAJOR);
    if (use_row_major_kernel) {
        auto mask_stick_size = tensor_args.mask->padded_shape()[3] * tensor_args.mask->element_size();
        reader_compile_time_args.push_back(mask_stick_size);
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }
    if (attributes.is_causal_mask) {
        if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
            reader_compile_time_args.push_back((std::uint32_t)program_config.block_h / mask_Ht);  // fused head
        } else {
            reader_compile_time_args.push_back((std::uint32_t)program_config.block_h);
        }
    }
    reader_compile_time_args.push_back(
        (std::uint32_t)(mask_cb_data_format == tt::DataFormat::Float32));  // mask float32
    reader_compile_time_args.push_back((std::uint32_t)mask_Ht);

    if (tensor_args.mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        softmax_defines["CAUSAL_MASK"] = "1";
        if (tensor_args.mask.value().is_sharded()) {
            softmax_defines["SHARDED_CAUSAL_MASK"] = "1";
        }
    }
    std::string reader_kernel_path;
    if (use_row_major_kernel) {
        reader_kernel_path =
            std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm_rm_mask.cpp";
    } else if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
        reader_kernel_path = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm.cpp";
    } else {
        reader_kernel_path =
            std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_sharded_sm_causal_mask_hw_dims.cpp";
    }
    auto reader_kernels_id = CreateKernel(
        program,
        reader_kernel_path,
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, softmax_defines));
    // compute kernel compile time args
    std::vector<uint32_t> compute_compile_time_args = {
        program_config.block_h,
        program_config.block_w,
        program_config.subblock_w,
        num_subblocks_w,
    };
    if (attributes.numeric_stable) {
        softmax_defines["NUMERIC_STABLE"] = "1";
    }
    softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    softmax_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    CreateKernel(
        program,
        std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_sharded.cpp",
        all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = softmax_defines});

    // Create circular buffers
    // in0 sharded
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;
    auto c_in0_config = CircularBufferConfig(in0_CB_size, {{tt::CBIndex::c_0, in0_cb_data_format}})
                            .set_page_size(tt::CBIndex::c_0, in0_tile_size)
                            .set_globally_allocated_address(*src0_buffer);
    auto cb_in0_id = CreateCircularBuffer(program, all_device_cores, c_in0_config);
    // in1 scalar
    auto c_in1_config = CircularBufferConfig(in1_CB_size, {{tt::CBIndex::c_1, scalar_cb_data_format}})
                            .set_page_size(tt::CBIndex::c_1, scalar_tile_size);
    CreateCircularBuffer(program, all_device_cores, c_in1_config);
    // in2 in3 attn scale mask
    std::optional<CBHandle> cb_intermed2_id;
    std::optional<CBHandle> cb_in2_id;
    std::optional<CBHandle> cb_in3_id;
    if (tensor_args.mask.has_value()) {
        // im2
        auto c_intermed2_config = CircularBufferConfig(im2_CB_size, {{tt::CBIndex::c_8, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_8, im_tile_size);
        cb_intermed2_id = CreateCircularBuffer(program, all_device_cores, c_intermed2_config);
        // in2 scale
        auto c_in2_config = CircularBufferConfig(in2_CB_size, {{tt::CBIndex::c_2, scale_cb_data_format}})
                                .set_page_size(tt::CBIndex::c_2, scale_tile_size);
        cb_in2_id = CreateCircularBuffer(program, all_device_cores, c_in2_config);
        // in3 attn mask
        if (tensor_args.mask->is_sharded()) {
            auto* mask_buffer = tensor_args.mask->buffer();
            auto c_in3_config = CircularBufferConfig(in3_CB_size, {{tt::CBIndex::c_3, mask_cb_data_format}})
                                    .set_page_size(tt::CBIndex::c_3, mask_tile_size)
                                    .set_globally_allocated_address(*mask_buffer);
            cb_in3_id = CreateCircularBuffer(program, all_device_cores, c_in3_config);
        } else {
            auto c_in3_config = CircularBufferConfig(in3_CB_size, {{tt::CBIndex::c_3, mask_cb_data_format}})
                                    .set_page_size(tt::CBIndex::c_3, mask_tile_size);
            cb_in3_id = CreateCircularBuffer(program, all_device_cores, c_in3_config);
        }
    }
    // out
    auto c_out0_config = CircularBufferConfig(out_CB_size, {{tt::CBIndex::c_11, out0_cb_data_format}})
                             .set_page_size(tt::CBIndex::c_11, out0_tile_size)
                             .set_globally_allocated_address(*out0_buffer);
    auto cb_out0_id = CreateCircularBuffer(program, all_device_cores, c_out0_config);
    // im0 for exp(x)
    auto c_intermed0_config = CircularBufferConfig(im0_CB_size, {{tt::CBIndex::c_6, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_6, im_tile_size);
    CreateCircularBuffer(program, all_device_cores, c_intermed0_config);
    // im1 for 1/sum(exp(x))
    auto c_intermed1_config = CircularBufferConfig(im1_CB_size, {{tt::CBIndex::c_7, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_7, im_tile_size);
    CreateCircularBuffer(program, all_device_cores, c_intermed1_config);
    if (attributes.numeric_stable) {
        // cb_max
        auto c_intermed3_config = CircularBufferConfig(max_CB_size, {{tt::CBIndex::c_9, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_9, im_tile_size);
        CreateCircularBuffer(program, all_device_cores, c_intermed3_config);
        // cb_x
        auto c_intermed4_config = CircularBufferConfig(x_CB_size, {{tt::CBIndex::c_10, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_10, im_tile_size);
        CreateCircularBuffer(program, all_device_cores, c_intermed4_config);
    }

    // Runtime Args
    uint32_t mask_addr = tensor_args.mask.has_value() ? tensor_args.mask->buffer()->address() : 0;
    union {
        float f;
        uint32_t u;
    } s{};
    s.f = attributes.scale.value_or(1.0f);  // scale for fused scale-mask-softmax
    uint32_t mask_start_tile_id = 0;

    uint32_t num_tiles_in_attn_mask = 0;
    uint32_t num_tiles_of_attn_mask_needed_per_core = 0;
    if (attributes.is_scale_causal_mask_hw_dims_softmax) {
        num_tiles_in_attn_mask =
            tensor_args.mask.value().padded_shape()[-1] * tensor_args.mask.value().padded_shape()[-2] / tile_hw;
        num_tiles_of_attn_mask_needed_per_core = program_config.block_h * program_config.block_w;
    }
    uint32_t num_cores_per_batch_index = 0;

    if (shard_orient == tt::tt_metal::ShardOrientation::COL_MAJOR) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
                CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

                // reader args
                std::vector<uint32_t> reader_args;
                reader_args.push_back(0x3f803f80);
                reader_args.push_back(s.u);
                reader_args.push_back(mask_addr);
                reader_args.push_back(mask_start_tile_id);
                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

                num_cores_per_batch_index++;

                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    uint32_t mask_tile_id_end =
                        (mask_start_tile_id + num_tiles_of_attn_mask_needed_per_core) % num_tiles_in_attn_mask;
                    mask_start_tile_id = mask_tile_id_end;
                } else {
                    if (num_cores_per_batch_index == num_cores_per_batch) {
                        num_cores_per_batch_index = 0;
                        if (tensor_args.mask.has_value()) {
                            if (attributes.is_causal_mask) {
                                mask_start_tile_id += tensor_args.mask->padded_shape()[-1] *
                                                      tensor_args.mask->padded_shape()[-2] / tile_width / tile_height;
                            } else {
                                mask_start_tile_id += use_row_major_kernel
                                                          ? tensor_args.mask->padded_shape()[-2]
                                                          : tensor_args.mask->padded_shape()[-1] / tile_width;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

                // reader args
                std::vector<uint32_t> reader_args;
                reader_args.push_back(0x3f803f80);
                reader_args.push_back(s.u);
                reader_args.push_back(mask_addr);
                reader_args.push_back(mask_start_tile_id);
                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

                num_cores_per_batch_index++;

                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    uint32_t mask_tile_id_end =
                        (mask_start_tile_id + num_tiles_of_attn_mask_needed_per_core) % num_tiles_in_attn_mask;
                    mask_start_tile_id = mask_tile_id_end;
                } else {
                    if (num_cores_per_batch_index == num_cores_per_batch) {
                        num_cores_per_batch_index = 0;
                        if (tensor_args.mask.has_value()) {
                            if (attributes.is_causal_mask) {
                                mask_start_tile_id += tensor_args.mask->padded_shape()[-1] *
                                                      tensor_args.mask->padded_shape()[-2] / tile_width / tile_height;
                            } else {
                                mask_start_tile_id += use_row_major_kernel
                                                          ? tensor_args.mask->padded_shape()[-2]
                                                          : tensor_args.mask->padded_shape()[-1] / tile_width;
                            }
                        }
                    }
                }
            }
        }
    }
    return {
        std::move(program),
        {reader_kernels_id,
         cb_in0_id,
         cb_out0_id,
         cb_in3_id,
         num_cores,
         program_config.compute_with_storage_grid_size}};
}

void SoftmaxShardedProgramFactoryAttentionOptimized::override_runtime_arguments(
    cached_program_t& cached_program,
    const SoftmaxParams& /*attributes*/,
    const SoftmaxInputs& tensor_args,
    Tensor& output_tensor) {
    auto* in0_buffer = tensor_args.input_tensor.buffer();
    const auto& mask_tensor = tensor_args.mask;
    auto* out_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_in0_id, *in0_buffer);
    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_out0_id, *out_buffer);
    if (mask_tensor.has_value() && mask_tensor->is_sharded()) {
        UpdateDynamicCircularBufferAddress(
            cached_program.program, cached_program.shared_variables.cb_in3_id.value(), *mask_tensor->buffer());
    }

    if (mask_tensor.has_value()) {
        for (uint32_t i = 0; i < cached_program.shared_variables.num_cores; ++i) {
            CoreCoord core = {
                i % cached_program.shared_variables.grid_size.x, i / cached_program.shared_variables.grid_size.x};
            auto& runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernels_id, core);
            runtime_args[2] = mask_tensor->buffer()->address();
        }
    }
}
}  // namespace ttnn::prim
