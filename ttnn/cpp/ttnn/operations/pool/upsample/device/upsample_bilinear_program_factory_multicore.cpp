// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"  // for reduce_op_utils

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"

using namespace tt::constants;

namespace ttnn::operations::pool::upsample::program {
using namespace tt;
using sliding_window::SlidingWindowConfig;

std::pair<Tensor, SlidingWindowConfig> HaloTensorCreation(
    const Tensor& input, uint32_t scale_factor_h, uint32_t scale_factor_w) {
    int batch_size = input.padded_shape()[0];
    int input_height = input.padded_shape()[1];
    int input_width = input.padded_shape()[2];
    int num_cores_nhw = input.shard_spec().value().num_cores();
    int num_cores_c = 1;

    ttnn::Tensor input_tensor = input;  // tensor to return
    SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_height, input_width},
        .window_hw = {2, 2},        // kernel size
        .stride_hw = {1, 1},        // stride
        .padding = {{1, 1, 1, 1}},  // padding (all sides)
        .dilation_hw = {1, 1},      // dilation
        .scale_h = scale_factor_h,  // upsampling scale factor height
        .scale_w = scale_factor_w,  // upsampling scale factor width
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .core_range_set = input_tensor.memory_config().shard_spec().value().grid,
        .snap_to_tile = false,
        .is_bilinear = true};

    const auto& input_shape = input.logical_shape();
    Shape new_shape({1, 1, input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]});
    input_tensor = ttnn::reshape(input_tensor, new_shape);

    auto halo_output = ttnn::halo(input_tensor, sliding_window_config, 0, false, false, input_tensor.memory_config());

    return {halo_output, sliding_window_config};
}

UpsampleBilinearProgramFactory::cached_program_t UpsampleBilinearProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    auto& output = output_tensor;
    const auto& scale_factor_h = operation_attributes.scale_factor_h;
    const auto& scale_factor_w = operation_attributes.scale_factor_w;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    Program program = tt::tt_metal::CreateProgram();

    auto input_shape = input.padded_shape();
    auto output_shape = output.padded_shape();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(input.padded_shape()[-1] % 32 == 0, "input channels should be divisible by 32");
    // NOTE: input is assumed to have channels last format: {N, H, W, C}, {N, 1, H * W, C}, {1, 1, N * H * W, C}
    // NOTE: Bfp8_b/TILE is not yet supported
    uint32_t input_stick_nbytes = input.padded_shape()[-1] * input.element_size();
    uint32_t output_stick_nbytes = output.padded_shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    // uint32_t batch_size = input.padded_shape()[0];
    uint32_t in_h = input.padded_shape()[1];
    uint32_t in_w = input.padded_shape()[2];
    uint32_t out_w = output.padded_shape()[2];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_nhw = ncores;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    uint32_t input_block_size_bytes = input_stick_nbytes;
    input_block_size_bytes =
        std::min(input_block_size_bytes, MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size());

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    // Validate sharding layout support
    if (input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(false, "Only HEIGHT_SHARDED layout is currently supported");
    }

    // Removed TT_FATAL checks for perfect divisibility - now supports uneven work distribution

    // creating halo input tensor
    auto [halo_in, sliding_window_config] = HaloTensorCreation(input, scale_factor_h, scale_factor_w);
    auto halo_shard_shape = halo_in.shard_spec().value().shape;

    // Generate op trace metadata for calculating min input offsets
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata_bilinear(sliding_window_config);

    // CBs
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    uint32_t next_cb_index = CBIndex::c_0;
    uint32_t buffering_factor = 2;  // only apply to intermediate buffers

    // input data is in a sharded CB
    uint32_t in_cb_pagesize = input_stick_nbytes;
    uint32_t in_cb_npages = halo_shard_shape[0];
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / constants::TILE_WIDTH);

    auto [halo_cb_id, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, in_cb_pagesize, in_cb_npages, input_cb_data_format, halo_in.buffer());

    // first intermediate CB
    uint32_t in1_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    uint32_t tilize_reduce_cb_0 = next_cb_index++;
    tt::tt_metal::create_cb(
        tilize_reduce_cb_0,
        program,
        all_cores,
        in1_cb_pagesize,
        4 * buffering_factor,
        input_cb_data_format);  // since 4 pixels per page are needed for intermediate tensor.

    // second intermediate CB
    uint32_t tilize_reduce_cb_1 = next_cb_index++;
    tt::tt_metal::create_cb(
        tilize_reduce_cb_1,
        program,
        all_cores,
        in_cb_pagesize,
        4 * buffering_factor,
        input_cb_data_format);  // since 4 pixels per page are needed for intermediate tensor.

    // scalar intermediate CBs
    uint32_t in_scalar_cb_pagesize = tile_size(input_cb_data_format);
    uint32_t in_scalar_cb_npages = 1 * buffering_factor;

    uint32_t in_scalar_cb_id1 = next_cb_index++;
    tt::tt_metal::create_cb(
        in_scalar_cb_id1, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, input_cb_data_format);

    uint32_t in_scalar_cb_id2 = next_cb_index++;
    tt::tt_metal::create_cb(
        in_scalar_cb_id2, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, input_cb_data_format);

    // output sharded CB with upsampled data
    uint32_t out_cb_pagesize = tt::constants::TILE_WIDTH * output.element_size();
    uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    auto [out_cb_id, out_cb] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, out_cb_pagesize, out_cb_npages, output_cb_data_format, output.buffer());

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", halo_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}", ncores);

    // Kernels
    // computation needed for the bilinear kernel. Passing them as an argument.
    // Convert to fixed-point Q16.16 format on host for better compile-time optimization in kernel
    // NOTE: These constants must match those in fixed_point_arithmetic.h to ensure consistency
    constexpr int32_t FIXED_POINT_SHIFT = 16;
    constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

    float scale_h_inv = 1.0f / (float)scale_factor_h;
    float scale_w_inv = 1.0f / (float)scale_factor_w;
    float y_index = ((float)(0.5f) * (float)scale_h_inv) + 0.5f;
    float x_index_compute = ((float)(0.5f) * (float)scale_w_inv) + 0.5f;  // Now +0.5f due to left padding

    // Convert to fixed-point Q16.16 format for kernel
    int32_t scale_h_inv_fixed = (int32_t)(scale_h_inv * FIXED_ONE);
    int32_t scale_w_inv_fixed = (int32_t)(scale_w_inv * FIXED_ONE);
    int32_t y_index_fixed = (int32_t)(y_index * FIXED_ONE);
    int32_t x_index_compute_fixed = (int32_t)(x_index_compute * FIXED_ONE);

    // Cast to uint32_t for passing as compile-time args
    uint32_t scale_h_inv_u32 = static_cast<uint32_t>(scale_h_inv_fixed);
    uint32_t scale_w_inv_u32 = static_cast<uint32_t>(scale_w_inv_fixed);
    uint32_t y_index_u32 = static_cast<uint32_t>(y_index_fixed);
    uint32_t x_index_compute_u32 = static_cast<uint32_t>(x_index_compute_fixed);

    uint32_t num_input_width_blocks =
        std::ceil((float)(input_shape[3]) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH));

    std::vector<uint32_t> reader_compile_time_args = {
        input_stick_nbytes,      // [0] stick_nbytes
        scale_factor_h,          // [1] scale_h (removed in_image_rows_per_core)
        scale_factor_w,          // [2] scale_w
        in_w,                    // [3] in_w
        out_w,                   // [4] out_w
        in_h,                    // [5] in_h
        halo_cb_id,              // [6] halo_cb_id
        tilize_reduce_cb_0,      // [7] tilize_reduce_cb_0
        in_scalar_cb_id1,        // [8] in_scalar_cb_id
        scale_h_inv_u32,         // [9] scale_h_inv_comp
        scale_w_inv_u32,         // [10] scale_w_inv_comp
        y_index_u32,             // [11] y_starting_coordinate_u32
        x_index_compute_u32,     // [12] x_starting_coordinate_u32
        1,                       // [13] is_reader
        num_input_width_blocks,  // [14] blocks
        input_block_size_bytes,  // [15] input_block_size_bytes
    };

    std::vector<uint32_t> writer_compile_time_args = {
        // Former runtime args (now compile-time)
        input_stick_nbytes,      // [0] stick_nbytes
        scale_factor_h,          // [1] scale_h (removed in_image_rows_per_core)
        scale_factor_w,          // [2] scale_w
        in_w,                    // [3] in_w
        out_w,                   // [4] out_w
        in_h,                    // [5] in_h
        halo_cb_id,              // [6] halo_cb_id
        tilize_reduce_cb_1,      // [7] tilize_reduce_cb_1
        in_scalar_cb_id2,        // [8] in_scalar_cb_id
        scale_h_inv_u32,         // [9] scale_h_inv_comp
        scale_w_inv_u32,         // [10] scale_w_inv_comp
        y_index_u32,             // [11] y_starting_coordinate_u32
        x_index_compute_u32,     // [12] x_starting_coordinate_u32
        0,                       // [13] is_reader (0 for writer)
        num_input_width_blocks,  // [14] blocks
        input_block_size_bytes,  // [15] input_block_size_bytes
    };

    std::string writer_kernel_fname, reader_kernel_fname, compute_kernel_fname;

    reader_kernel_fname = std::string(
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp");
    writer_kernel_fname = std::string(
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp");
    compute_kernel_fname = std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp");

    std::vector<uint32_t> compute_compile_time_args = {
        tilize_reduce_cb_0,
        tilize_reduce_cb_1,
        in_scalar_cb_id1,
        in_scalar_cb_id2,
        out_cb_id,
        in_ntiles_c,
        1 * in_ntiles_c,
        4,  // Number of input rows required for tilize reduction is 4 as we are processing single output row at a time.
        (uint32_t)std::ceil((float)output_shape[3] / constants::TILE_WIDTH),
        num_input_width_blocks,
        input_block_size_bytes,
    };

    auto reader_kernel = CreateKernel(
        program, reader_kernel_fname, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel = CreateKernel(
        program, writer_kernel_fname, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported. #12787 issue raised");
    auto reduce_op = tt::tt_metal::ReduceOpMath::SUM;
    auto reduce_dim = tt::tt_metal::ReduceOpDim::H;
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_compile_time_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};

    auto compute_kernel = CreateKernel(program, compute_kernel_fname, all_cores, compute_config);

    // runtime args - start_output_idx, min_input_offset, and work count per core
    uint32_t reader_nargs = 3;
    std::vector<uint32_t> reader_rt_args(reader_nargs);
    uint32_t compute_nargs = 1;
    std::vector<uint32_t> compute_rt_args(compute_nargs);

    // Calculate work distribution based on output sticks
    uint32_t batch_size = input.padded_shape()[0];
    uint32_t total_output_sticks = batch_size * output.padded_shape()[1] * output.padded_shape()[2];
    uint32_t max_out_sticks_per_core = div_up(total_output_sticks, ncores_nhw);

    log_debug(
        LogOp, "total_output_sticks: {}, max_out_sticks_per_core: {}", total_output_sticks, max_out_sticks_per_core);

    uint32_t start_output_idx = 0;

    if (input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        // Get logical cores in the correct order based on shard orientation

        auto logical_cores = corerange_to_cores(
            shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);

        uint32_t total_sticks_processed = 0;

        for (uint32_t i = 0; i < logical_cores.size(); ++i) {
            const auto& core_coord = logical_cores[i];

            // Calculate actual output sticks for this core
            uint32_t out_sticks_this_core =
                std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);

            // Calculate the output range for this core
            uint32_t output_index_start = start_output_idx;
            uint32_t output_index_end =
                std::min(output_index_start + out_sticks_this_core, (uint32_t)op_trace_metadata.size()) - 1;

            // Find the minimum input index for this core's output range
            if (out_sticks_this_core == 0) {
                // No work for this core
                reader_rt_args[0] = start_output_idx;
                reader_rt_args[1] = 0;
                reader_rt_args[2] = 0;  // No work
                SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
                SetRuntimeArgs(program, writer_kernel, core_coord, reader_rt_args);

                compute_rt_args[0] = 0;  // No work
                SetRuntimeArgs(program, compute_kernel, core_coord, compute_rt_args);

                continue;
            }
            auto [min_trace_idx, max_trace_idx] =
                sliding_window::find_minmax_trace_indices(op_trace_metadata, output_index_start, output_index_end);
            uint32_t min_input_offset = op_trace_metadata[min_trace_idx];

            // Set runtime arguments for reader/writer
            reader_rt_args[0] = start_output_idx;
            reader_rt_args[1] = min_input_offset;
            reader_rt_args[2] = out_sticks_this_core;  // NEW: actual work for this core
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
            SetRuntimeArgs(program, writer_kernel, core_coord, reader_rt_args);

            // Set runtime arguments for compute kernel
            compute_rt_args[0] = out_sticks_this_core;  // Work count for this core
            SetRuntimeArgs(program, compute_kernel, core_coord, compute_rt_args);

            // Next core starts where this core ends
            start_output_idx += out_sticks_this_core;

            total_sticks_processed += out_sticks_this_core;
        }
    } else {
        TT_FATAL(false, "Unsupported memory layout");
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .cb_src0 = cb_src0,
            .out_cb = out_cb,
        }};
}

void UpsampleBilinearProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& out_cb = cached_program.shared_variables.out_cb;

    const auto& input_tensor = tensor_args.input_tensor;

    auto [halo_in, _] =
        HaloTensorCreation(input_tensor, operation_attributes.scale_factor_h, operation_attributes.scale_factor_w);
    auto* src_buffer = halo_in.buffer();
    auto* dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
}

}  // namespace ttnn::operations::pool::upsample::program
