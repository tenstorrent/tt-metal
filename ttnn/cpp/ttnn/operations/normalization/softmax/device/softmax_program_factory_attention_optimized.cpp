// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>
#include <map>
#include <utility>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SoftmaxDeviceOperation::SoftmaxProgramFactoryAttentionOptimized::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryAttentionOptimized selected");

    // Constants
    const auto& shape = tensor_args.input_tensor.padded_shape();
    const uint32_t W = shape[-1], H = (tensor_args.input_tensor.physical_volume() / (shape[0] * shape[-1])),
                   NC = shape[0];
    const uint32_t tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    const auto& shape_unpadded = tensor_args.input_tensor.logical_shape();
    auto* const src0_buffer = tensor_args.input_tensor.buffer();
    auto* const out0_buffer = output_tensor.buffer();

    bool mask_padded_data = false;
    uint32_t num_datum_padded = 0;
    uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    uint32_t mask_H = H;
    if (tensor_args.mask.has_value()) {
        mask_H = tensor_args.mask.value().padded_shape()[2];
    }
    const uint32_t mask_Ht = mask_H / tile_height;

    auto* device = tensor_args.input_tensor.device();

    const tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    const tt::DataFormat max_scaler_cb_data_format =
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t max_scaler_tile_size = tt::tile_size(max_scaler_cb_data_format);
    const uint32_t fused_attention_scale_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    const tt::DataFormat out0_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t out0_tile_size = tt::tile_size(out0_cb_data_format);

    const tt::DataFormat mask_cb_data_format = tensor_args.mask.has_value()
                                                   ? datatype_to_dataformat_converter(tensor_args.mask.value().dtype())
                                                   : tt::DataFormat::Float16_b;
    const uint32_t mask_tile_size = tt::tile_size(mask_cb_data_format);

    const tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t im_tile_size = tt::tile_size(im_cb_data_format);

    const tt::DataFormat sum_scaler_cb_data_format =
        im_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t sum_scaler_tile_size = tt::tile_size(sum_scaler_cb_data_format);

    uint32_t block_size =
        fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(Wt, 4) : tt::tt_metal::find_max_divisor(Wt, 8);

    // calc_numeric_stable() in softmax.cpp uses indexed access over Wt tiles of its input CB and a
    // WaitUpfrontNoPop reduce, so whichever CB it consumes must be sized to Wt:
    //   - no mask, no padding: it consumes cb_in0 directly.
    //   - fused scale-mask path or mask-padded path: it consumes cb_x (cb_in0 is streamed/popped per block).
    const bool small_kernel_numeric_stable_uses_cb_in0_at_wt =
        attributes.numeric_stable && !tensor_args.mask.has_value() && !mask_padded_data;
    const bool small_kernel_numeric_stable_uses_cb_x_at_wt =
        attributes.numeric_stable && (tensor_args.mask.has_value() || mask_padded_data);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t =
        small_kernel_numeric_stable_uses_cb_in0_at_wt ? tt::div_up(Wt, block_size) * block_size : block_size * 2;
    uint32_t out0_t = block_size * 2;
    uint32_t im1_t = 1;  // 1/sum(exp(x))
    uint32_t in2_t = 1;  // scaler for reduce coming from reader
    uint32_t in3_t = 1;  // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t =
        tt::div_up(Wt, block_size) * block_size;  // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    uint32_t in5_t = 1;
    // numeric_stable cb max
    uint32_t im2_t = 1;
    uint32_t im4_t = tt::div_up(Wt, block_size) * block_size;

    // cb_exps - keeps exps in tt::CBIndex in L1 to avoid recomputing
    uint32_t im0_t = block_size * tt::div_up(Wt, block_size);
    TT_FATAL(im0_t == Wt, "im0_t: {} == Wt: {}, (Non user error)", im0_t, Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t = block_size * (tt::div_up(Wt, block_size) + 1);

    uint32_t cb_length = in0_t;
    bool use_large_kernel = false;
    // Noisy CB estimator, if the cbs used take up 90% of L1 switch to large kernel implementation
    constexpr uint32_t single_tile_cb_count = 5;  // approximate
    uint32_t cb_size_sum_bytes = (in0_t * in0_tile_size) + (im0_t * im_tile_size) + (out0_t * out0_tile_size) +
                                 (single_tile_cb_count * im_tile_size);
    if (small_kernel_numeric_stable_uses_cb_x_at_wt) {
        // cb_x (c_10) is only allocated for the small kernel when calc_numeric_stable consumes it.
        // In the no-mask numeric_stable path the kernel aliases cb_x to cb_exps, so no extra CB is needed.
        cb_size_sum_bytes += im4_t * im_tile_size;
    }
    if (tensor_args.mask.has_value()) {
        cb_size_sum_bytes +=
            (im3_t * im_tile_size) + (in3_t * fused_attention_scale_tile_size) + (in4_t * mask_tile_size);
    }

    // Program specific checks
    if ((tensor_args.input_tensor.device()->l1_size_per_core() * 0.9) < cb_size_sum_bytes) {
        use_large_kernel = true;
        uint32_t large_kernel_cb_size = (80 / block_size) * block_size;
        cb_length = large_kernel_cb_size;
        in0_t = large_kernel_cb_size;
        im4_t = large_kernel_cb_size;
        im0_t = large_kernel_cb_size;
        im3_t = large_kernel_cb_size;
    }
    if (!use_large_kernel) {
        TT_FATAL(
            im3_t == Wt + block_size, "im3_t {} == Width in tiles {} + num_dest_regs to use {}", im3_t, Wt, block_size);
    }
    TT_FATAL(Wt % block_size == 0, "Wt: {} must be divisible by one of the numbers in the range from 8 to 1.", Wt);
    TT_FATAL((block_size != -1), "Wt: {} must be divisible by one of the numbers in the range from 8 to 1.", Wt);
    TT_FATAL(
        im0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        im0_t);
    TT_FATAL(
        out0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        out0_t);
    TT_FATAL(
        in4_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        in4_t);

    // Work split
    const uint32_t num_tile_rows = NC * Ht;
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    // Data movement kernels
    std::vector<uint32_t> reader_compile_time_args = {};
    TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);
    if (tensor_args.mask.has_value()) {
        TensorAccessorArgs(tensor_args.mask.value().buffer()).append_to(reader_compile_time_args);
    }
    if (attributes.is_causal_mask) {
        uint32_t num_tiles_causal_mask = tensor_args.mask.value().padded_shape()[-1] *
                                         tensor_args.mask.value().padded_shape()[-2] / tile_width / tile_height;
        reader_compile_time_args.push_back(num_tiles_causal_mask);
    }

    std::vector<uint32_t> writer_compile_time_args = {num_datum_padded, tile_height * tile_width};
    TensorAccessorArgs(out0_buffer).append_to(writer_compile_time_args);
    std::map<std::string, std::string> softmax_defines;
    if (tensor_args.mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        softmax_defines["CAUSAL_MASK"] = "1";
    }
    if (attributes.numeric_stable) {
        softmax_defines["NUMERIC_STABLE"] = "1";
    }
    std::string reader_kernel_path =
        use_large_kernel
            ? std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_interleaved_sm_large_tensor.cpp"
            : std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_interleaved_sm.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = KernelDescriptor::Defines(softmax_defines.begin(), softmax_defines.end());
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht

    softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    softmax_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    std::string softmax_kernel_path =
        use_large_kernel ? std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_large_tensor.cpp"
                         : std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax.cpp";

    KernelDescriptor softmax_desc;
    softmax_desc.kernel_source = softmax_kernel_path;
    softmax_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    softmax_desc.core_ranges = all_device_cores;
    softmax_desc.compile_time_args = {};
    softmax_desc.defines = KernelDescriptor::Defines(softmax_defines.begin(), softmax_defines.end());
    softmax_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Circular buffers
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * in0_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_cb_data_format,
            .page_size = in0_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * out0_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_11),
            .data_format = out0_cb_data_format,
            .page_size = out0_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * im_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = im_cb_data_format,
            .page_size = im_tile_size,
        }}},
    });
    if (use_large_kernel) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = im1_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_12),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = im1_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_15),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = im1_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * max_scaler_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = max_scaler_cb_data_format,
            .page_size = max_scaler_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * sum_scaler_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_13),
            .data_format = sum_scaler_cb_data_format,
            .page_size = sum_scaler_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * im_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = im_cb_data_format,
            .page_size = im_tile_size,
        }}},
    });
    if (tensor_args.mask.has_value()) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = im3_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = in3_t * fused_attention_scale_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = fused_attention_scale_tile_size,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = in4_t * mask_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = mask_cb_data_format,
                .page_size = mask_tile_size,
            }}},
        });
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = in5_t * mask_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
            .data_format = mask_cb_data_format,
            .page_size = mask_tile_size,
        }}},
    });
    if (attributes.numeric_stable) {
        // cb_max
        desc.cbs.push_back(CBDescriptor{
            .total_size = im2_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
    }
    // cb_x: only needed when calc_numeric_stable consumes a separate post-mask buffer (mask or
    // mask-padded path), or when the streaming large kernel is selected. In the no-mask
    // numeric_stable path softmax.cpp aliases cb_x to cb_exps, so we skip the allocation.
    if (small_kernel_numeric_stable_uses_cb_x_at_wt || use_large_kernel) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = im4_t * im_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = im_cb_data_format,
                .page_size = im_tile_size,
            }}},
        });
    }

    uint32_t mask_addr = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer()->address() : 0;

    uint32_t curr_row = 0;
    uint32_t scale_value =
        std::bit_cast<uint32_t>(attributes.scale.value_or(1.0f));  // scale for fused scale-mask-softmax
    for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        if (i >= num_cores) {
            if (attributes.is_causal_mask) {
                reader_desc.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u});
            } else {
                reader_desc.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u});
            }

            softmax_desc.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u, 0u, 0u});
            writer_desc.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u, 0u});
            continue;
        }
        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;
        uint32_t curr_ht = curr_row % Ht;
        uint32_t mask_curr_ht = curr_ht % mask_Ht;            // the start offset for causal mask
        uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt;  // causal mask batch offset
        uint32_t mask_id = attributes.is_causal_mask
                               ? ((mask_curr_ht * Wt) + mask_offset)
                               : (curr_row / Ht * Wt);  // causal mask start offset + causal mask batch offset

        // NOTE: do not pass Buffer* here. scale_value and mask_addr depend on
        // operation_attributes (scale + optional mask tensor); BufferBinding's
        // fast cache-hit path skips create_descriptor() and would leave them stale.
        if (attributes.is_causal_mask) {
            reader_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    src0_buffer->address(),
                    block_size,
                    scale_value,
                    num_tile_rows_per_core,
                    tile_offset,
                    Wt,
                    Ht,
                    mask_addr,
                    curr_ht,
                    mask_id,
                    in0_t,
                    mask_curr_ht,
                    mask_offset});
        } else {
            reader_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    src0_buffer->address(),
                    block_size,
                    scale_value,
                    num_tile_rows_per_core,
                    tile_offset,
                    Wt,
                    Ht,
                    mask_addr,
                    curr_ht,
                    mask_id,
                    in0_t});
        }

        softmax_desc.emplace_runtime_args(
            core,
            {num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, static_cast<uint32_t>(mask_padded_data), cb_length});

        // NOTE: do not pass Buffer* here. mask_padded_data and num_datum_padded
        // depend on attribute state; BufferBinding fast cache-hit path skips
        // create_descriptor() and would leave them stale.
        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                out0_buffer->address(),
                num_tile_rows_per_core * Wt,
                tile_offset,
                block_size,
                static_cast<uint32_t>(mask_padded_data),
                num_datum_padded});

        curr_row += num_tile_rows_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(softmax_desc));

    return desc;
}

}  // namespace ttnn::prim
