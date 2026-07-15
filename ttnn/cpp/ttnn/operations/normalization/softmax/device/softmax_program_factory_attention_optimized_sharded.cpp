// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <map>
#include <utility>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor
SoftmaxDeviceOperation::SoftmaxShardedProgramFactoryAttentionOptimized::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    auto* device = tensor_args.input_tensor.device();

    // Guard against non-sharded inputs when using a sharded program config
    TT_FATAL(
        tensor_args.input_tensor.is_sharded() && tensor_args.input_tensor.shard_spec().has_value(),
        "Input tensor must be sharded when using SoftmaxShardedMultiCoreProgramConfig");

    // convert data format
    tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    tt::DataFormat out0_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat mask_cb_data_format = tensor_args.mask.has_value()
                                             ? datatype_to_dataformat_converter(tensor_args.mask->dtype())
                                             : tt::DataFormat::Float16_b;
    tt::DataFormat fused_attention_scale_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat max_scaler_cb_data_format =
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat sum_scaler_cb_data_format =
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
    uint32_t fused_attention_scale_tile_size = tt::tile_size(fused_attention_scale_cb_data_format);
    uint32_t max_scaler_tile_size = tt::tile_size(max_scaler_cb_data_format);
    uint32_t sum_scaler_tile_size = tt::tile_size(sum_scaler_cb_data_format);
    // in out buffer
    auto* src0_buffer = tensor_args.input_tensor.buffer();
    auto* out0_buffer = output_tensor.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_CB_size = program_config.block_w * program_config.block_h * in0_tile_size;
    // scaler for reduce coming from reader
    uint32_t max_scaler_CB_size = 1 * max_scaler_tile_size;
    uint32_t sum_scaler_CB_size = 1 * sum_scaler_tile_size;
    // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in2_CB_size = 1 * fused_attention_scale_tile_size;
    // attention mask
    uint32_t in3_CB_size = 0;
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
    const CoreRangeSet all_device_cores{CoreRange(
        {static_cast<std::size_t>(start_core_x), static_cast<std::size_t>(start_core_y)},
        {static_cast<std::size_t>(start_core_x) + num_cores_c - 1,
         static_cast<std::size_t>(start_core_y) + num_cores_r - 1})};
    // reader compile arg
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(program_config.block_w)};
    TensorAccessorArgs(tensor_args.mask ? tensor_args.mask->buffer() : nullptr).append_to(reader_compile_time_args);
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
            reader_compile_time_args.push_back(static_cast<uint32_t>(program_config.block_h) / mask_Ht);  // fused head
        } else {
            reader_compile_time_args.push_back(static_cast<uint32_t>(program_config.block_h));
        }
    }
    reader_compile_time_args.push_back(
        static_cast<uint32_t>(mask_cb_data_format == tt::DataFormat::Float32));  // mask float32
    reader_compile_time_args.push_back(static_cast<uint32_t>(mask_Ht));

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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = KernelDescriptor::Defines(softmax_defines.begin(), softmax_defines.end());
    reader_desc.config = ReaderConfigDescriptor{};

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

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.defines = KernelDescriptor::Defines(softmax_defines.begin(), softmax_defines.end());
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Create circular buffers
    ProgramDescriptor desc;
    // in0 sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_cb_data_format,
            .page_size = in0_tile_size,
        }}},
        .buffer = src0_buffer,
    });
    // in1 max scaler
    desc.cbs.push_back(CBDescriptor{
        .total_size = max_scaler_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = max_scaler_cb_data_format,
            .page_size = max_scaler_tile_size,
        }}},
    });
    // sum scaler
    desc.cbs.push_back(CBDescriptor{
        .total_size = sum_scaler_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_13),
            .data_format = sum_scaler_cb_data_format,
            .page_size = sum_scaler_tile_size,
        }}},
    });
    // in2 in3 attn scale mask
    if (tensor_args.mask.has_value()) {
        // im2
        desc.cbs.push_back(CBDescriptor{
            .total_size = im2_CB_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
        // in2 scale
        desc.cbs.push_back(CBDescriptor{
            .total_size = in2_CB_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = fused_attention_scale_cb_data_format,
                .page_size = fused_attention_scale_tile_size,
            }}},
        });
        // in3 attn mask
        if (tensor_args.mask->is_sharded()) {
            auto* mask_buffer = tensor_args.mask->buffer();
            desc.cbs.push_back(CBDescriptor{
                .total_size = in3_CB_size,
                .core_ranges = all_device_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                    .data_format = mask_cb_data_format,
                    .page_size = mask_tile_size,
                }}},
                .buffer = mask_buffer,
            });
        } else {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in3_CB_size,
                .core_ranges = all_device_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                    .data_format = mask_cb_data_format,
                    .page_size = mask_tile_size,
                }}},
            });
        }
    }
    // out
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_11),
            .data_format = out0_cb_data_format,
            .page_size = out0_tile_size,
        }}},
        .buffer = out0_buffer,
    });
    // im0 for exp(x)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = im_cb_data_format,
            .page_size = im_tile_size,
        }}},
    });
    // im1 for 1/sum(exp(x))
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_CB_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = im_cb_data_format,
            .page_size = im_tile_size,
        }}},
    });
    if (attributes.numeric_stable) {
        // cb_max
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_CB_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
        // cb_x
        desc.cbs.push_back(CBDescriptor{
            .total_size = x_CB_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
    }

    // Runtime Args
    Buffer* mask_buffer = tensor_args.mask.has_value() ? tensor_args.mask->buffer() : nullptr;
    uint32_t scale_u = std::bit_cast<uint32_t>(attributes.scale.value_or(1.0f));  // scale for fused scale-mask-softmax
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
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
                CoreCoord core = {
                    static_cast<std::size_t>(start_core_x) + core_idx_x,
                    static_cast<std::size_t>(start_core_y) + core_idx_y};

                // reader args
                std::vector<std::variant<uint32_t, Buffer*>> reader_args;
                reader_args.push_back(scale_u);
                reader_args.push_back(mask_buffer);
                reader_args.push_back(mask_start_tile_id);
                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                reader_desc.emplace_runtime_args(core, reader_args);

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
        for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {
                    static_cast<std::size_t>(start_core_x) + core_idx_x,
                    static_cast<std::size_t>(start_core_y) + core_idx_y};

                // reader args
                std::vector<std::variant<uint32_t, Buffer*>> reader_args;
                reader_args.push_back(scale_u);
                reader_args.push_back(mask_buffer);
                reader_args.push_back(mask_start_tile_id);
                if (attributes.is_scale_causal_mask_hw_dims_softmax) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                reader_desc.emplace_runtime_args(core, reader_args);

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

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
