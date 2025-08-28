// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <utility>

namespace ttnn::operations::normalization::softmax::program {
// General-purpose softmax with arbitrary dimension support
SoftmaxProgramFactoryGeneral::cached_program_t SoftmaxProgramFactoryGeneral::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};

    return {std::move(program), {}};
}

void SoftmaxProgramFactoryGeneral::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Implementation details...
}

//
// Optimized for transformer attention patterns
//
// Interleaved memory
SoftmaxProgramFactoryAttentionOptimized::cached_program_t SoftmaxProgramFactoryAttentionOptimized::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    // Constants
    const auto& shape = tensor_args.input_tensor.padded_shape();
    const uint32_t W = shape[-1], H = (tensor_args.input_tensor.physical_volume() / (shape[0] * shape[-1])),
                   NC = shape[0];
    const uint32_t HW = H * W;
    const uint32_t tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_hw = tensor_args.input_tensor.tensor_spec().tile().get_tile_hw();
    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    const auto& shape_unpadded = tensor_args.input_tensor.logical_shape();

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
    uint32_t mask_Ht = mask_H / tile_height;

    // This should allocate input_tensor DRAM buffer on the device
    auto* device = tensor_args.input_tensor.device();

    tt::DataFormat in0_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(in0_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat out0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(out0_cb_data_format);

    tt::DataFormat mask_cb_data_format =
        tensor_args.mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(tensor_args.mask.value().dtype())
                                     : tt::DataFormat::Float16_b;
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_cb_data_format);

    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_cb_data_format);

    log_debug(tt::LogOp, "in0_cb_data_format: {}", in0_cb_data_format);
    log_debug(tt::LogOp, "out0_cb_data_format: {}", out0_cb_data_format);
    log_debug(tt::LogOp, "mask_cb_data_format: {}", mask_cb_data_format);
    log_debug(tt::LogOp, "im_cb_data_format: {}", im_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "num_datum_padded: {}", num_datum_padded);

    auto src0_buffer = tensor_args.input_tensor.buffer();
    auto out0_buffer = output_tensor.buffer();

    uint32_t num_tiles = tensor_args.input_tensor.physical_volume() / tile_hw;

    uint32_t block_size =
        fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(Wt, 4) : tt::tt_metal::find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t = attributes.numeric_stable ? tt::div_up(Wt, block_size) * block_size : block_size * 2;
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
    uint32_t cb_size_sum_bytes = (in0_t * in0_tile_size) + (im0_t * im_tile_size);
    if (tensor_args.mask.has_value()) {
        cb_size_sum_bytes += (im3_t * im_tile_size) + (im4_t * im_tile_size);
    }
    if ((tensor_args.input_tensor.device()->l1_size_per_core() * 0.9) < cb_size_sum_bytes) {
        use_large_kernel = true;
        cb_length = 80;
        in0_t = 80;
        im4_t = 80;
        im0_t = 80;
        im3_t = 80;
        TT_FATAL(!attributes.inplace, "Tensor is too large to run softmax inplace, please use standard softmax");
    }
    // TODO: Not sure why this fatal is here but not needed for use_large_kernel
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

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);
    if (tensor_args.mask.has_value()) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.mask.value().buffer()).append_to(reader_compile_time_args);
    }
    if (attributes.is_causal_mask) {
        uint32_t num_tiles_causal_mask = tensor_args.mask.value().padded_shape()[-1] *
                                         tensor_args.mask.value().padded_shape()[-2] / tile_width / tile_height;
        reader_compile_time_args.push_back(num_tiles_causal_mask);
    }

    std::vector<uint32_t> writer_compile_time_args = {num_datum_padded};
    tt::tt_metal::TensorAccessorArgs(out0_buffer).append_to(writer_compile_time_args);
    std::map<std::string, std::string> softmax_defines, writer_defines;
    if (tensor_args.mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        softmax_defines["CAUSAL_MASK"] = "1";
    }
    if (attributes.numeric_stable) {
        softmax_defines["NUMERIC_STABLE"] = "1";
    }
    std::string reader_kernel_path = use_large_kernel
                                         ? "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/"
                                           "reader_unary_interleaved_sm_large_tensor.cpp"
                                         : "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/"
                                           "reader_unary_interleaved_sm.cpp";
    auto reader_kernels_id = CreateKernel(
        program,
        reader_kernel_path,
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, softmax_defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked_sm.cpp",
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht

    softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    std::string softmax_kernel_path =
        use_large_kernel
            ? "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax_large_tensor.cpp"
            : "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax.cpp";
    auto softmax_kernels_id = CreateKernel(
        program,
        softmax_kernel_path,
        all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = {},
            .defines = softmax_defines});

    // Create circular buffers
    // see softmax.cpp for which buffers are needed
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;
    auto c_in0_config = CircularBufferConfig(in0_t * in0_tile_size, {{tt::CBIndex::c_0, in0_cb_data_format}})
                            .set_page_size(tt::CBIndex::c_0, in0_tile_size);
    auto cb_in0_id = CreateCircularBuffer(program, all_device_cores, c_in0_config);
    auto c_out0_config = CircularBufferConfig(out0_t * out0_tile_size, {{tt::CBIndex::c_11, out0_cb_data_format}})
                             .set_page_size(tt::CBIndex::c_11, out0_tile_size);
    auto cb_out0_id = CreateCircularBuffer(program, all_device_cores, c_out0_config);
    auto c_intermed1_config = CircularBufferConfig(im1_t * im_tile_size, {{tt::CBIndex::c_7, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_7, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, all_device_cores, c_intermed1_config);
    if (use_large_kernel) {
        auto c_intermedsum_config = CircularBufferConfig(im1_t * im_tile_size, {{tt::CBIndex::c_12, im_cb_data_format}})
                                        .set_page_size(tt::CBIndex::c_12, im_tile_size);
        auto cb_intermedsum_id = CreateCircularBuffer(program, all_device_cores, c_intermedsum_config);
        auto c_intermedmax_config = CircularBufferConfig(im1_t * im_tile_size, {{tt::CBIndex::c_15, im_cb_data_format}})
                                        .set_page_size(tt::CBIndex::c_15, im_tile_size);
        auto cb_intermedmax_id = CreateCircularBuffer(program, all_device_cores, c_intermedmax_config);
        auto c_recip_config = CircularBufferConfig(im1_t * im_tile_size, {{tt::CBIndex::c_16, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_16, im_tile_size);
        auto cb_recip_id = CreateCircularBuffer(program, all_device_cores, c_recip_config);
    }
    auto c_in2_config = CircularBufferConfig(in2_t * scalar_tile_size, {{tt::CBIndex::c_2, scalar_cb_data_format}})
                            .set_page_size(tt::CBIndex::c_2, scalar_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, all_device_cores, c_in2_config);
    auto c_intermed0_config = CircularBufferConfig(im0_t * im_tile_size, {{tt::CBIndex::c_6, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_6, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, all_device_cores, c_intermed0_config);
    std::optional<CBHandle> cb_intermed3_id;
    std::optional<CBHandle> cb_in3_id;
    std::optional<CBHandle> cb_in4_id;
    std::optional<CBHandle> cb_in5_id;
    if (tensor_args.mask.has_value()) {
        CircularBufferConfig c_intermed3_config =
            CircularBufferConfig(im3_t * im_tile_size, {{tt::CBIndex::c_9, im_cb_data_format}})
                .set_page_size(tt::CBIndex::c_9, im_tile_size);
        cb_intermed3_id = CreateCircularBuffer(program, all_device_cores, c_intermed3_config);
        CircularBufferConfig c_in3_config =
            CircularBufferConfig(in3_t * scalar_tile_size, {{tt::CBIndex::c_3, scalar_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, scalar_tile_size);
        cb_in3_id = CreateCircularBuffer(program, all_device_cores, c_in3_config);
        CircularBufferConfig c_in4_config =
            CircularBufferConfig(in4_t * mask_tile_size, {{tt::CBIndex::c_4, mask_cb_data_format}})
                .set_page_size(tt::CBIndex::c_4, mask_tile_size);
        cb_in4_id = CreateCircularBuffer(program, all_device_cores, c_in4_config);
    }
    CircularBufferConfig c_in5_config =
        CircularBufferConfig(in5_t * mask_tile_size, {{tt::CBIndex::c_5, mask_cb_data_format}})
            .set_page_size(tt::CBIndex::c_5, mask_tile_size);
    cb_in5_id = CreateCircularBuffer(program, all_device_cores, c_in5_config);
    std::optional<CBHandle> cb_intermed2_id;
    std::optional<CBHandle> cb_intermed4_id;
    if (attributes.numeric_stable) {
        // cb_max
        auto c_intermed2_config = CircularBufferConfig(im2_t * im_tile_size, {{tt::CBIndex::c_8, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_8, im_tile_size);
        cb_intermed2_id = CreateCircularBuffer(program, all_device_cores, c_intermed2_config);
    }
    // cb_x
    if (attributes.numeric_stable || use_large_kernel) {
        auto c_x_config = CircularBufferConfig(im4_t * im_tile_size, {{tt::CBIndex::c_10, im_cb_data_format}})
                              .set_page_size(tt::CBIndex::c_10, im_tile_size);
        cb_intermed4_id = CreateCircularBuffer(program, all_device_cores, c_x_config);
    }

    uint32_t src_addr = src0_buffer->address();
    uint32_t mask_addr = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer()->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t curr_row = 0;
    union {
        float f;
        uint32_t u;
    } s{};
    s.f = attributes.scale.value_or(1.0f);  // scale for fused scale-mask-softmax
    for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        if (i >= num_cores) {
            if (attributes.is_causal_mask) {
                SetRuntimeArgs(program, reader_kernels_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3f803f80, 0, 0});
            } else {
                SetRuntimeArgs(program, reader_kernels_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3f803f80});
            }

            SetRuntimeArgs(program, softmax_kernels_id, core, {0, 0, 0, 0, 0, 0});
            SetRuntimeArgs(program, writer_kernels_id, core, {0, 0, 0, 0, 0, 0});
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
                               ? (mask_curr_ht * Wt + mask_offset)
                               : (curr_row / Ht * Wt);  // causal mask start offset + causal mask batch offset

        if (attributes.is_causal_mask) {
            SetRuntimeArgs(
                program,
                reader_kernels_id,
                core,
                {src_addr,
                 block_size,
                 s.u,
                 num_tile_rows_per_core,
                 tile_offset,
                 Wt,
                 Ht,
                 mask_addr,
                 curr_ht,
                 mask_id,
                 0x3f803f80,
                 in0_t,
                 mask_curr_ht,
                 mask_offset});  // [10]=1.0f is scaler
        } else {
            SetRuntimeArgs(
                program,
                reader_kernels_id,
                core,
                {src_addr,
                 block_size,
                 s.u,
                 num_tile_rows_per_core,
                 tile_offset,
                 Wt,
                 Ht,
                 mask_addr,
                 curr_ht,
                 mask_id,
                 0x3f803f80,
                 in0_t});  // [10]=1.0f is scaler
        }

        SetRuntimeArgs(
            program,
            softmax_kernels_id,
            core,
            {num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, mask_padded_data, cb_length});

        SetRuntimeArgs(
            program,
            writer_kernels_id,
            core,
            {out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size, mask_padded_data, num_datum_padded});

        curr_row += num_tile_rows_per_core;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, softmax_kernels_id, grid_size,
                                 fp32_dest_acc_en,  scalar_tile_size,  in0_tile_size,      im_tile_size,
                                 out0_tile_size,    mask_tile_size,    cb_in0_id,          cb_out0_id,
                                 cb_intermed1_id,   cb_in2_id,         cb_intermed0_id,    cb_intermed3_id,
                                 cb_in3_id,         cb_in4_id,         cb_intermed2_id,    cb_intermed4_id}};
}

void SoftmaxProgramFactoryAttentionOptimized::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto src_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto mask_buffer_address = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer()->address() : 0;
    auto dst_buffer_address = output_tensor.buffer()->address();

    const auto shape = tensor_args.input_tensor.padded_shape();
    const uint32_t W = shape[-1], H = (tensor_args.input_tensor.physical_volume() / (shape[0] * shape[-1])),
                   NC = shape[0];
    const uint32_t HW = H * W;
    const uint32_t tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_hw = tensor_args.input_tensor.tensor_spec().tile().get_tile_hw();
    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;

    bool mask_padded_data = false;
    uint32_t num_datum_padded = 0;
    const auto& shape_unpadded = tensor_args.input_tensor.logical_shape();
    uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    int32_t num_tiles = tensor_args.input_tensor.physical_volume() / tile_hw;
    uint32_t block_size = cached_program.shared_variables.fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(Wt, 4)
                                                                           : tt::tt_metal::find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t = attributes.numeric_stable ? tt::div_up(Wt, block_size) * block_size : block_size * 2;
    uint32_t out0_t = block_size * 2;
    uint32_t im1_t = 1;  // 1/sum(exp(x))
    uint32_t in2_t = 1;  // scaler for reduce coming from reader
    uint32_t in3_t = 1;  // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t =
        tt::div_up(Wt, block_size) * block_size;  // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    uint32_t im2_t = 1;
    uint32_t im4_t = tt::div_up(Wt, block_size) * block_size;

    // cb_exps - keeps exps in tt::CBIndex in L1 to avoid recomputing
    uint32_t im0_t = block_size * tt::div_up(Wt, block_size);
    TT_FATAL(im0_t == Wt, "Intermediate buffer size (im0_t={}) must match width (Wt={})", im0_t, Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t = block_size * (tt::div_up(Wt, block_size) + 1);
    TT_FATAL(im3_t == Wt + block_size, "im3_t {} == Wt {}+ block_size {}", im3_t, Wt, block_size);

    TT_FATAL(Wt % block_size == 0, "Wt {} must be divisible by block size {}", Wt, block_size);
    TT_FATAL((block_size != -1), "Wt {} must be divisible by one of the numbers in the range from 8 to 1.", Wt);

    uint32_t NCHt = NC * Ht;
    uint32_t num_tile_rows = NC * Ht;
    auto all_device_cores = CoreRange(
        {0, 0}, {cached_program.shared_variables.grid_size.x - 1, cached_program.shared_variables.grid_size.y - 1});
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(cached_program.shared_variables.grid_size, num_tile_rows, true);

    UpdateCircularBufferTotalSize(
        cached_program.program,
        cached_program.shared_variables.cb_in0_id,
        in0_t * cached_program.shared_variables.in0_tile_size);
    UpdateCircularBufferTotalSize(
        cached_program.program,
        cached_program.shared_variables.cb_out0_id,
        out0_t * cached_program.shared_variables.out0_tile_size);
    UpdateCircularBufferTotalSize(
        cached_program.program,
        cached_program.shared_variables.cb_intermed1_id,
        im1_t * cached_program.shared_variables.im_tile_size);
    UpdateCircularBufferTotalSize(
        cached_program.program,
        cached_program.shared_variables.cb_in2_id,
        in2_t * cached_program.shared_variables.scalar_tile_size);
    UpdateCircularBufferTotalSize(
        cached_program.program,
        cached_program.shared_variables.cb_intermed0_id,
        im0_t * cached_program.shared_variables.im_tile_size);

    if (tensor_args.mask.has_value()) {
        UpdateCircularBufferTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_intermed3_id.value(),
            im3_t * cached_program.shared_variables.im_tile_size);
        UpdateCircularBufferTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_in3_id.value(),
            in3_t * cached_program.shared_variables.scalar_tile_size);
        UpdateCircularBufferTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_in4_id.value(),
            in4_t * cached_program.shared_variables.mask_tile_size);
    }
    if (attributes.numeric_stable) {
        UpdateCircularBufferTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_intermed2_id.value(),
            im2_t * cached_program.shared_variables.im_tile_size);
        UpdateCircularBufferTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_intermed4_id.value(),
            im4_t * cached_program.shared_variables.im_tile_size);
    }

    uint32_t curr_row = 0;
    union {
        float f;
        uint32_t u;
    } s{};
    s.f = attributes.scale.value_or(1.0f);  // scale for fused scale-mask-softmax

    auto& cached_reader_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernels_id);
    auto& cached_softmax_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.softmax_kernels_id);
    auto& cached_writer_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernels_id);

    for (uint32_t i = 0; i < cached_program.shared_variables.grid_size.x * cached_program.shared_variables.grid_size.y;
         ++i) {
        CoreCoord core = {
            i % cached_program.shared_variables.grid_size.x, i / cached_program.shared_variables.grid_size.x};
        uint32_t num_tile_rows_per_core = 0;

        auto& reader_kernel_args = cached_reader_args.at(core.x).at(core.y);
        auto& softmax_kernel_args = cached_softmax_args.at(core.x).at(core.y);
        auto& writer_kernel_args = cached_writer_args.at(core.x).at(core.y);

        if (i >= num_cores) {
            reader_kernel_args[3] = 0;
            softmax_kernel_args[0] = 0;
            writer_kernel_args[1] = 0;
            continue;
        }

        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;
        uint32_t curr_ht = curr_row % Ht;
        uint32_t mask_curr_ht = curr_ht % Wt;            // the start offset for causal mask
        uint32_t mask_offset = curr_row / Ht * Wt * Wt;  // causal mask batch offset
        uint32_t mask_id = attributes.is_causal_mask
                               ? (mask_curr_ht * Wt + mask_offset)
                               : (curr_row / Ht * Wt);  // causal mask start offset + causal mask batch offset

        reader_kernel_args[0] = src_buffer_address;
        reader_kernel_args[1] = block_size;
        reader_kernel_args[2] = s.u;
        reader_kernel_args[3] = num_tile_rows_per_core;
        reader_kernel_args[4] = tile_offset;
        reader_kernel_args[5] = Wt;
        reader_kernel_args[6] = Ht;
        reader_kernel_args[7] = mask_buffer_address;
        reader_kernel_args[8] = curr_ht;
        reader_kernel_args[9] = mask_id;
        // reader_kernel_args[10] = 0x3f803f80; // Hardcoded value doesn't need to be updated

        if (attributes.is_causal_mask) {
            reader_kernel_args[11] = mask_curr_ht;
            reader_kernel_args[12] = mask_offset;
        }

        softmax_kernel_args[0] = num_tile_rows_per_core;
        softmax_kernel_args[1] = Ht;
        softmax_kernel_args[2] = Wt;
        softmax_kernel_args[3] = block_size;
        softmax_kernel_args[4] = curr_ht;
        softmax_kernel_args[5] = mask_padded_data;
        softmax_kernel_args[6] = cached_program.shared_variables.fp32_dest_acc_en ? 1 : 0;

        writer_kernel_args[0] = dst_buffer_address;
        writer_kernel_args[1] = num_tile_rows_per_core * Wt;
        writer_kernel_args[2] = tile_offset;
        writer_kernel_args[3] = block_size;
        writer_kernel_args[4] = mask_padded_data;
        writer_kernel_args[5] = num_datum_padded;

        curr_row += num_tile_rows_per_core;
    }
}

SoftmaxShardedProgramFactoryAttentionOptimized::cached_program_t SoftmaxShardedProgramFactoryAttentionOptimized::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    auto* device = tensor_args.input_tensor.device();

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
    uint32_t M = shape[2] * shape[0];
    uint32_t K = shape[3] * shape[1];
    uint32_t Mt = M / tile_width;
    uint32_t Kt = K / tile_width;
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
    uint32_t block_w = program_config.block_w * tile_width;
    uint32_t block_h = program_config.block_h * tile_width;
    uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;

    // single tile sizes
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_cb_data_format);
    uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(in0_cb_data_format);
    uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(out0_cb_data_format);
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_cb_data_format);
    uint32_t scale_tile_size = tt::tt_metal::detail::TileSize(scale_cb_data_format);
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_cb_data_format);
    // in out buffer
    auto src0_buffer = tensor_args.input_tensor.buffer();
    auto out0_buffer = output_tensor.buffer();
    // num tiles
    uint32_t num_tiles = tensor_args.input_tensor.physical_volume() / tile_hw;

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
            "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/"
            "reader_unary_sharded_sm_rm_mask.cpp";
    } else if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/reader_unary_sharded_sm.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/"
            "readed_unary_sharded_sm_causal_mask_hw_dims.cpp";
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
    auto softmax_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax_sharded.cpp",
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
    auto cb_in1_id = CreateCircularBuffer(program, all_device_cores, c_in1_config);
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
            auto mask_buffer = tensor_args.mask->buffer();
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
    auto cb_intermed0_id = CreateCircularBuffer(program, all_device_cores, c_intermed0_config);
    // im1 for 1/sum(exp(x))
    auto c_intermed1_config = CircularBufferConfig(im1_CB_size, {{tt::CBIndex::c_7, im_cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_7, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, all_device_cores, c_intermed1_config);
    if (attributes.numeric_stable) {
        // cb_max
        auto c_intermed3_config = CircularBufferConfig(max_CB_size, {{tt::CBIndex::c_9, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_9, im_tile_size);
        auto cb_intermed3_id = CreateCircularBuffer(program, all_device_cores, c_intermed3_config);
        // cb_x
        auto c_intermed4_config = CircularBufferConfig(x_CB_size, {{tt::CBIndex::c_10, im_cb_data_format}})
                                      .set_page_size(tt::CBIndex::c_10, im_tile_size);
        auto cb_intermed4_id = CreateCircularBuffer(program, all_device_cores, c_intermed4_config);
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
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto in0_buffer = tensor_args.input_tensor.buffer();
    auto& mask_tensor = tensor_args.mask;
    auto out_buffer = output_tensor.buffer();

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
}  // namespace ttnn::operations::normalization::softmax::program
