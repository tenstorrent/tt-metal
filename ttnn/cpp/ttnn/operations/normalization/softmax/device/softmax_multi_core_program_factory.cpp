// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/logger.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using namespace tt::constants;
namespace ttnn::operations::normalization {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory::cached_program_t softmax_multi_core(
    const Tensor &input_tensor,
    const Tensor &output_tensor,
    const std::optional<const Tensor> mask,
    std::optional<float> scale,
    bool causal_mask,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config
) {

    const auto shape = input_tensor.get_legacy_shape();
    uint32_t W = shape[-1], H = (input_tensor.volume() / (shape[0] * shape[-1])), NC = shape[0];
    uint32_t HW = H*W;

    bool mask_padded_data = false;
    uint32_t num_datum_padded = 0;
    const auto shape_unpadded = input_tensor.get_shape();
    uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t mask_H = H;
    if (mask.has_value()) {
        mask_H = mask.value().get_legacy_shape()[2];
    }
    uint32_t mask_Ht = mask_H/TILE_HEIGHT;

    Program program = CreateProgram();

    // This should allocate input_tensor DRAM buffer on the device
    Device *device = input_tensor.device();

    tt::DataFormat in0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(in0_cb_data_format);

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    std::cout << "softmax_multi_core" << std::endl;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == tt::ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = in0_cb_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config.value());
    std::cout << "softmax_multi_core done" << std::endl;

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat out0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(out0_cb_data_format);

    tt::DataFormat mask_cb_data_format = mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(mask.value().get_dtype()) : tt::DataFormat::Float16_b;
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_cb_data_format);

    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_cb_data_format);

    tt::log_debug("in0_cb_data_format: {}", in0_cb_data_format);
    tt::log_debug("out0_cb_data_format: {}", out0_cb_data_format);
    tt::log_debug("mask_cb_data_format: {}", mask_cb_data_format);
    tt::log_debug("im_cb_data_format: {}", im_cb_data_format);
    tt::log_debug("math_fidelity: {}", math_fidelity);
    tt::log_debug("math_approx_mode: {}", math_approx_mode);
    tt::log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    auto src0_buffer = input_tensor.buffer();
    auto out0_buffer = output_tensor.buffer();

    uint32_t num_tiles = input_tensor.volume()/TILE_HW;

    uint32_t block_size = fp32_dest_acc_en ? find_max_divisor(Wt, 4) : find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size*2;
    uint32_t out0_t = block_size*2;
    uint32_t im1_t  = 1; // 1/sum(exp(x))
    uint32_t in2_t  = 1; // scaler for reduce coming from reader
    uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t  = tt::div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    uint32_t in5_t = 1;

    // cb_exps - keeps exps in tt::CB in L1 to avoid recomputing
    uint32_t im0_t  = block_size*tt::div_up(Wt, block_size);
    TT_ASSERT(im0_t == Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t  = block_size*(tt::div_up(Wt, block_size)+1);
    TT_ASSERT(im3_t == Wt+block_size);

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in4_t % block_size == 0);
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out0_is_dram = out0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        src0_is_dram
    };
    if (mask.has_value()) {
        bool mask_is_dram = mask.value().buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        reader_compile_time_args.push_back(mask_is_dram);
    }
    if (causal_mask) {
        uint32_t num_tiles_causal_mask = mask.value().get_legacy_shape()[-1] * mask.value().get_legacy_shape()[-2] / TILE_WIDTH / TILE_HEIGHT;
        reader_compile_time_args.push_back(num_tiles_causal_mask);
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      out0_is_dram};
    std::map<string, string> softmax_defines;
    if (mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (causal_mask) {
        softmax_defines["CAUSAL_MASK"] = "1";
    }
    auto reader_kernels_id = CreateKernel(
        program, "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/reader_unary_interleaved_sm.cpp", all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            reader_compile_time_args,
            softmax_defines
    ));

    auto writer_kernels_id = CreateKernel(
        program, "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp", all_device_cores,
        tt::tt_metal::WriterDataMovementConfig(
            writer_compile_time_args,
            softmax_defines
    ));

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht

    softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    auto softmax_kernels_id = CreateKernel(
        program, "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax.cpp", all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode,
            .compile_args = {},
            .defines = softmax_defines
    });

    // Create circular buffers
    // see softmax.cpp for which buffers are needed

    auto c_in0_config = CircularBufferConfig(in0_t * in0_tile_size, {{tt::CB::c_in0, in0_cb_data_format}}).set_page_size(tt::CB::c_in0, in0_tile_size);
    auto cb_in0_id = CreateCircularBuffer( program, all_device_cores, c_in0_config);
    auto c_out0_config = CircularBufferConfig(out0_t * out0_tile_size, {{tt::CB::c_out0, out0_cb_data_format}}).set_page_size(tt::CB::c_out0, out0_tile_size);
    auto cb_out0_id = CreateCircularBuffer( program, all_device_cores, c_out0_config );
    auto c_intermed1_config = CircularBufferConfig(im1_t * im_tile_size, {{tt::CB::c_intermed1, im_cb_data_format}}).set_page_size(tt::CB::c_intermed1, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer( program, all_device_cores, c_intermed1_config );
    auto c_in2_config = CircularBufferConfig(in2_t * scalar_tile_size, {{tt::CB::c_in2, scalar_cb_data_format}}).set_page_size(tt::CB::c_in2, scalar_tile_size);
    auto cb_in2_id = CreateCircularBuffer( program, all_device_cores, c_in2_config );
    auto c_intermed0_config = CircularBufferConfig(im0_t * im_tile_size, {{tt::CB::c_intermed0, im_cb_data_format}}).set_page_size(tt::CB::c_intermed0, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer( program, all_device_cores, c_intermed0_config );
    std::optional<CBHandle> cb_intermed3_id;
    std::optional<CBHandle> cb_in3_id;
    std::optional<CBHandle> cb_in4_id;
    std::optional<CBHandle> cb_in5_id;
    if (mask.has_value()) {
        CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t * im_tile_size, {{tt::CB::c_intermed3, im_cb_data_format}}).set_page_size(tt::CB::c_intermed3, im_tile_size);
        cb_intermed3_id = CreateCircularBuffer( program, all_device_cores, c_intermed3_config );
        CircularBufferConfig c_in3_config = CircularBufferConfig(in3_t * scalar_tile_size, {{tt::CB::c_in3, scalar_cb_data_format}}).set_page_size(tt::CB::c_in3, scalar_tile_size);
        cb_in3_id = CreateCircularBuffer( program, all_device_cores, c_in3_config );
        CircularBufferConfig c_in4_config = CircularBufferConfig(in4_t * mask_tile_size, {{tt::CB::c_in4, mask_cb_data_format}}).set_page_size(tt::CB::c_in4, mask_tile_size);
        cb_in4_id = CreateCircularBuffer( program, all_device_cores, c_in4_config);
    }
    CircularBufferConfig c_in5_config = CircularBufferConfig(in5_t * mask_tile_size, {{tt::CB::c_in5, mask_cb_data_format}}).set_page_size(tt::CB::c_in5, mask_tile_size);
    cb_in5_id = CreateCircularBuffer( program, all_device_cores, c_in5_config);

    uint32_t src_addr = src0_buffer->address();
    uint32_t mask_addr = mask.has_value() ? mask.value().buffer()->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t curr_row = 0;
    union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
    for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        if (i >= num_cores) {
            SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
            SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0, 0 });
            SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0 });
            continue;
        }
        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;
        uint32_t curr_ht = curr_row % Ht;
        uint32_t mask_curr_ht = curr_ht % mask_Ht;   // the start offset for causal mask
        uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt; // causal mask batch offset
        uint32_t mask_id = causal_mask ? (mask_curr_ht * Wt + mask_offset) : (curr_row / Ht * Wt); // causal mask start offset + causal mask batch offset

        if (causal_mask) {
            SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }); // [10]=1.0f is scaler
        } else {
            SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
        }

        SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, mask_padded_data });

        SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size, mask_padded_data, num_datum_padded, 0xFF00FF00});

        curr_row += num_tile_rows_per_core;
    }


    return {
        std::move(program),
        SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory::shared_variables_t{
            .reader_kernels_id = reader_kernels_id,
            .writer_kernels_id = writer_kernels_id,
            .softmax_kernels_id = softmax_kernels_id,
            .grid_size = grid_size,
            .scalar_tile_size = scalar_tile_size,
            .in0_tile_size = in0_tile_size,
            .im_tile_size = im_tile_size,
            .out0_tile_size = out0_tile_size,
            .mask_tile_size = mask_tile_size,
            .cb_in0_id = cb_in0_id,
            .cb_out0_id = cb_out0_id,
            .cb_in2_id = cb_in2_id,
            .cb_intermed0_id = cb_intermed0_id,
            .cb_intermed1_id = cb_intermed1_id,
            .cb_intermed3_id = cb_intermed3_id,
            .cb_in3_id = cb_in3_id,
            .cb_in4_id = cb_in4_id,
        }
    };
}

SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory::cached_program_t SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory::create(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    return softmax_multi_core(
        tensor_args.input_tensor,
        tensor_return_value,
        tensor_args.mask,
        attributes.scale,
        attributes.is_causal_mask,
        attributes.compute_kernel_config
    );
}

void SoftmaxDeviceOperation::SoftmaxMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto& softmax_kernels_id = cached_program.shared_variables.softmax_kernels_id;
    auto& grid_size = cached_program.shared_variables.grid_size;
    auto& scalar_tile_size = cached_program.shared_variables.scalar_tile_size;
    auto& in0_tile_size = cached_program.shared_variables.in0_tile_size;
    auto& im_tile_size = cached_program.shared_variables.im_tile_size;
    auto& out0_tile_size = cached_program.shared_variables.out0_tile_size;
    auto& mask_tile_size = cached_program.shared_variables.mask_tile_size;
    auto& cb_in0_id = cached_program.shared_variables.cb_in0_id;
    auto& cb_out0_id = cached_program.shared_variables.cb_out0_id;
    auto& cb_intermed1_id = cached_program.shared_variables.cb_intermed1_id;
    auto& cb_in2_id = cached_program.shared_variables.cb_in2_id;
    auto& cb_intermed0_id = cached_program.shared_variables.cb_intermed0_id;
    auto& cb_intermed3_id = cached_program.shared_variables.cb_intermed3_id;
    auto& cb_in3_id = cached_program.shared_variables.cb_in3_id;
    auto& cb_in4_id = cached_program.shared_variables.cb_in4_id;

    auto& program = cached_program.program;
    auto& scale = attributes.scale;
    auto causal_mask = attributes.is_causal_mask;
    auto& input_tensor = tensor_args.input_tensor;
    auto& mask_tensor = tensor_args.mask;

    auto src_buffer_address = input_tensor.buffer()->address();
    auto mask_buffer_address = mask_tensor.has_value() ? mask_tensor.value().buffer()->address() : 0;
    auto dst_buffer_address = tensor_return_value.buffer()->address();

    const auto shape = input_tensor.get_legacy_shape();
    uint32_t W = shape[-1], H = (input_tensor.volume() / (shape[0] * shape[-1])), NC = shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    bool mask_padded_data = false;
    uint32_t num_datum_padded = 0;
    const auto shape_unpadded = input_tensor.get_shape();
    uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    int32_t num_tiles = input_tensor.volume()/TILE_HW;
    uint32_t block_size = find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size*2;
    uint32_t out0_t = block_size*2;
    uint32_t im1_t  = 1; // 1/sum(exp(x))
    uint32_t in2_t  = 1; // scaler for reduce coming from reader
    uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t  = tt::div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled

    // cb_exps - keeps exps in tt::CB in L1 to avoid recomputing
    uint32_t im0_t  = block_size*tt::div_up(Wt, block_size);
    TT_ASSERT(im0_t == Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t  = block_size*(tt::div_up(Wt, block_size)+1);
    TT_ASSERT(im3_t == Wt+block_size);

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in4_t % block_size == 0);
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

    uint32_t NCHt = NC*Ht;
    uint32_t num_tile_rows = NC * Ht;
    auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    UpdateCircularBufferTotalSize(program, cb_in0_id, in0_t * in0_tile_size);
    UpdateCircularBufferTotalSize(program, cb_out0_id, out0_t * out0_tile_size);
    UpdateCircularBufferTotalSize(program, cb_intermed1_id, im1_t * im_tile_size);
    UpdateCircularBufferTotalSize(program, cb_in2_id, in2_t * scalar_tile_size);
    UpdateCircularBufferTotalSize(program, cb_intermed0_id, im0_t * im_tile_size);

    if (mask_tensor.has_value()) {
        UpdateCircularBufferTotalSize(program, cb_intermed3_id.value(), im3_t * im_tile_size);
        UpdateCircularBufferTotalSize(program, cb_in3_id.value(), in3_t * scalar_tile_size);
        UpdateCircularBufferTotalSize(program, cb_in4_id.value(), in4_t * mask_tile_size);
    }

    uint32_t curr_row = 0;
    union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
    for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        if (i >= num_cores) {
            SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
            SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0, 0 });
            SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0});
            continue;
        }

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;
        uint32_t curr_ht = curr_row % Ht;
        uint32_t mask_curr_ht = curr_ht % Wt;   // the start offset for causal mask
        uint32_t mask_offset = curr_row / Ht * Wt * Wt; // causal mask batch offset
        uint32_t mask_id = causal_mask ? (mask_curr_ht * Wt + mask_offset) : (curr_row / Ht * Wt); // causal mask start offset + causal mask batch offset

        if (causal_mask) {
            SetRuntimeArgs(program, reader_kernels_id, core, { src_buffer_address, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_buffer_address, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }); // [10]=1.0f is scaler
        } else {
            SetRuntimeArgs(program, reader_kernels_id, core, { src_buffer_address, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_buffer_address, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
        }

        SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, mask_padded_data });

        SetRuntimeArgs(program, writer_kernels_id, core, { dst_buffer_address, num_tile_rows_per_core * Wt, tile_offset, block_size, mask_padded_data, num_datum_padded, 0xFF00FF00});

        curr_row += num_tile_rows_per_core;
    }
}

}  // namespace ttnn::operations::normalization
