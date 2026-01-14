// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::pool::upsample::program {

using FixedPoint = int32_t;
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

static FixedPoint float_to_fixed(float value) { return static_cast<FixedPoint>(value * FIXED_ONE); }

UpsampleBilinearProgramFactory::cached_program_t UpsampleBilinearProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output_tensor) {
    const ttnn::Tensor& input = tensor_args.input_tensor;
    const ttnn::Tensor& output = output_tensor;
    const uint32_t scale_factor_h = operation_attributes.scale_factor_h;
    const uint32_t scale_factor_w = operation_attributes.scale_factor_w;
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Use the sliding window config passed from upsample.cpp (contains original input dimensions)
    TT_FATAL(
        operation_attributes.sliding_window_config.has_value(),
        "Bilinear upsample requires sliding_window_config to be provided");
    const sliding_window::SlidingWindowConfig sliding_window_config =
        operation_attributes.sliding_window_config.value();

    // Extract original (pre-halo) dimensions from sliding_window_config
    // These are the TRUE dimensions, not the haloed tensor dimensions
    const uint32_t in_batch_size = sliding_window_config.batch_size;
    const uint32_t in_h = sliding_window_config.input_hw.first;
    const uint32_t in_w = sliding_window_config.input_hw.second;
    const uint32_t in_channels = sliding_window_config.channels;

    // Output dimensions
    const tt::tt_metal::Shape& output_shape = output.padded_shape();
    const uint32_t out_w = output_shape[2];

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(in_channels % 32 == 0, "input channels should be divisible by 32");
    // NOTE: input is assumed to have channels last format: {N, H, W, C}, {N, 1, H * W, C}, {1, 1, N * H * W, C}
    // NOTE: Bfp8_b/TILE is not yet supported
    const uint32_t input_stick_nbytes = in_channels * input.element_size();
    const uint32_t output_stick_nbytes = output_shape[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    const std::tuple<MathFidelity, bool, bool, bool, bool> compute_config_tuple =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    const MathFidelity math_fidelity = std::get<0>(compute_config_tuple);
    const bool math_approx_mode = std::get<1>(compute_config_tuple);
    const bool fp32_dest_acc_en = std::get<2>(compute_config_tuple);

    const tt::tt_metal::ShardSpec shard_spec = input.shard_spec().value();
    const tt::tt_metal::CoreRangeSet all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();
    const uint32_t ncores_nhw = ncores;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    uint32_t input_block_size_bytes = input_stick_nbytes;
    input_block_size_bytes =
        std::min(input_block_size_bytes, MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size());

    const tt::tt_metal::ShardSpec out_shard_spec = output.shard_spec().value();

    const ttnn::Tensor& halo_in = input;
    const std::array<uint32_t, 2> halo_shard_shape = halo_in.shard_spec().value().shape;

    const std::vector<uint32_t> op_trace_metadata =
        sliding_window::generate_op_trace_metadata_bilinear(sliding_window_config);

    uint32_t next_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t buffering_factor = 2;

    // input data is in a sharded CB
    const uint32_t in_cb_pagesize = input_stick_nbytes;
    const uint32_t in_cb_npages = halo_shard_shape[0];
    const uint32_t in_ntiles_c = tt::div_up(in_channels, tt::constants::TILE_WIDTH);

    const std::tuple<uint32_t, tt::tt_metal::CBHandle> cb_tuple_src = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, in_cb_pagesize, in_cb_npages, input_cb_data_format, halo_in.buffer());
    const uint32_t halo_cb_id = std::get<0>(cb_tuple_src);
    const tt::tt_metal::CBHandle cb_src0 = std::get<1>(cb_tuple_src);

    // first intermediate CB
    const uint32_t in1_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    const uint32_t tilize_reduce_cb_0 = next_cb_index++;
    tt::tt_metal::create_cb(
        tilize_reduce_cb_0,
        program,
        all_cores,
        in1_cb_pagesize,
        4 * buffering_factor,
        input_cb_data_format);  // since 4 pixels per page are needed for intermediate tensor.

    // second intermediate CB
    const uint32_t tilize_reduce_cb_1 = next_cb_index++;
    tt::tt_metal::create_cb(
        tilize_reduce_cb_1,
        program,
        all_cores,
        in_cb_pagesize,
        4 * buffering_factor,
        input_cb_data_format);  // since 4 pixels per page are needed for intermediate tensor.

    // scalar intermediate CBs
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(input_cb_data_format);
    const uint32_t in_scalar_cb_npages = 1 * buffering_factor;

    const uint32_t in_scalar_cb_id1 = next_cb_index++;
    tt::tt_metal::create_cb(
        in_scalar_cb_id1, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, input_cb_data_format);

    const uint32_t in_scalar_cb_id2 = next_cb_index++;
    tt::tt_metal::create_cb(
        in_scalar_cb_id2, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, input_cb_data_format);

    // output sharded CB with upsampled data
    const uint32_t out_cb_pagesize = tt::constants::TILE_WIDTH * output.element_size();
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    const std::tuple<uint32_t, tt::tt_metal::CBHandle> cb_tuple_out = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, out_cb_pagesize, out_cb_npages, output_cb_data_format, output.buffer());
    const uint32_t out_cb_id = std::get<0>(cb_tuple_out);
    const tt::tt_metal::CBHandle out_cb = std::get<1>(cb_tuple_out);

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", halo_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(tt::LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(tt::LogOp, "ncores: {}", ncores);

    const float scale_h_inv = 1.0f / static_cast<float>(scale_factor_h);
    const float scale_w_inv = 1.0f / static_cast<float>(scale_factor_w);

    const float y_index = (0.5f * scale_h_inv) + 0.5f;
    const float x_index_compute = (0.5f * scale_w_inv) + 0.5f;

    const FixedPoint scale_h_inv_fixed = float_to_fixed(scale_h_inv);
    const FixedPoint scale_w_inv_fixed = float_to_fixed(scale_w_inv);
    const FixedPoint y_index_fixed = float_to_fixed(y_index);
    const FixedPoint x_index_compute_fixed = float_to_fixed(x_index_compute);

    const uint32_t num_input_width_blocks =
        std::ceil(static_cast<float>(in_channels) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH));

    std::vector<uint32_t> reader_compile_time_args = {
        input_stick_nbytes,      // [0] stick_nbytes
        scale_factor_h,          // [1] scale_h (removed in_image_rows_per_core)
        scale_factor_w,          // [2] scale_w
        in_w,                    // [3] in_w (from sliding_window_config, not haloed tensor)
        out_w,                   // [4] out_w
        in_h,                    // [5] in_h (from sliding_window_config, not haloed tensor)
        halo_cb_id,              // [6] halo_cb_id
        tilize_reduce_cb_0,      // [7] tilize_reduce_cb_0
        in_scalar_cb_id1,        // [8] in_scalar_cb_id
        scale_h_inv_fixed,       // [9] scale_h_inv_comp
        scale_w_inv_fixed,       // [10] scale_w_inv_comp
        y_index_fixed,           // [11] y_starting_coordinate_u32
        x_index_compute_fixed,   // [12] x_starting_coordinate_u32
        1,                       // [13] is_reader
        num_input_width_blocks,  // [14] blocks
        input_block_size_bytes,  // [15] input_block_size_bytes
    };

    std::vector<uint32_t> writer_compile_time_args = {
        input_stick_nbytes,      // [0] stick_nbytes
        scale_factor_h,          // [1] scale_h (removed in_image_rows_per_core)
        scale_factor_w,          // [2] scale_w
        in_w,                    // [3] in_w (from sliding_window_config, not haloed tensor)
        out_w,                   // [4] out_w
        in_h,                    // [5] in_h (from sliding_window_config, not haloed tensor)
        halo_cb_id,              // [6] halo_cb_id
        tilize_reduce_cb_1,      // [7] tilize_reduce_cb_1
        in_scalar_cb_id2,        // [8] in_scalar_cb_id
        scale_h_inv_fixed,       // [9] scale_h_inv_comp
        scale_w_inv_fixed,       // [10] scale_w_inv_comp
        y_index_fixed,           // [11] y_starting_coordinate_u32
        x_index_compute_fixed,   // [12] x_starting_coordinate_u32
        0,                       // [13] is_reader (0 for writer)
        num_input_width_blocks,  // [14] blocks
        input_block_size_bytes,  // [15] input_block_size_bytes
    };

    const std::string reader_kernel_fname = std::string(
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp");
    const std::string writer_kernel_fname = std::string(
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp");
    const std::string compute_kernel_fname =
        std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp");

    std::vector<uint32_t> compute_compile_time_args = {
        tilize_reduce_cb_0,
        tilize_reduce_cb_1,
        in_scalar_cb_id1,
        in_scalar_cb_id2,
        out_cb_id,
        in_ntiles_c,
        1 * in_ntiles_c,
        4,
        tt::div_up(in_channels, tt::constants::TILE_WIDTH),
        num_input_width_blocks,
        input_block_size_bytes,
    };

    const tt::tt_metal::KernelHandle reader_kernel = tt::tt_metal::CreateKernel(
        program, reader_kernel_fname, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    const tt::tt_metal::KernelHandle writer_kernel = tt::tt_metal::CreateKernel(
        program, writer_kernel_fname, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported for upsample bilinear");

    const tt::tt_metal::ReduceOpMath reduce_op = tt::tt_metal::ReduceOpMath::SUM;
    const tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::H;

    const tt::tt_metal::ComputeConfig compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_compile_time_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};

    const tt::tt_metal::KernelHandle compute_kernel =
        tt::tt_metal::CreateKernel(program, compute_kernel_fname, all_cores, compute_config);

    constexpr uint32_t reader_nargs = 3;
    std::vector<uint32_t> reader_rt_args(reader_nargs);
    constexpr uint32_t compute_nargs = 1;
    std::vector<uint32_t> compute_rt_args(compute_nargs);

    // Calculate work distribution based on output sticks

    const uint32_t total_output_sticks = in_batch_size * output.logical_shape()[1] * output.logical_shape()[2];
    const uint32_t max_out_sticks_per_core = tt::div_up(total_output_sticks, ncores_nhw);

    log_debug(
        tt::LogOp,
        "total_output_sticks: {}, max_out_sticks_per_core: {}",
        total_output_sticks,
        max_out_sticks_per_core);

    uint32_t start_output_idx = 0;

    const std::vector<tt::tt_metal::CoreCoord> logical_cores = tt::tt_metal::corerange_to_cores(
        shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    uint32_t total_sticks_processed = 0;

    for (const auto& core_coord : logical_cores) {
        // Calculate actual output sticks for this core
        uint32_t out_sticks_this_core = std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);

        if (out_sticks_this_core == 0) {
            // No work for this core
            reader_rt_args[0] = start_output_idx;
            reader_rt_args[1] = 0;
            reader_rt_args[2] = 0;  // No work
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core_coord, reader_rt_args);

            compute_rt_args[0] = 0;  // No work
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core_coord, compute_rt_args);
            continue;
        }

        // Calculate the output range for this core (only after confirming there's work)
        uint32_t output_index_start = start_output_idx;
        const uint32_t output_index_end =
            std::min(output_index_start + out_sticks_this_core, static_cast<uint32_t>(op_trace_metadata.size())) - 1;

        // Find the minimum input index for this core's output range
        const std::pair<uint32_t, uint32_t> minmax_indices =
            sliding_window::find_minmax_trace_indices(op_trace_metadata, output_index_start, output_index_end);
        const uint32_t min_trace_idx = minmax_indices.first;
        const uint32_t min_input_offset = op_trace_metadata[min_trace_idx];

        // Set runtime arguments for reader/writer
        reader_rt_args[0] = start_output_idx;
        reader_rt_args[1] = min_input_offset;
        reader_rt_args[2] = out_sticks_this_core;
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core_coord, reader_rt_args);

        // Set runtime arguments for compute kernel
        compute_rt_args[0] = out_sticks_this_core;  // Work count for this core
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core_coord, compute_rt_args);

        // Next core starts where this core ends
        start_output_idx += out_sticks_this_core;

        total_sticks_processed += out_sticks_this_core;
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
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::Program& program = cached_program.program;
    tt::tt_metal::CBHandle& cb_src0 = cached_program.shared_variables.cb_src0;
    tt::tt_metal::CBHandle& out_cb = cached_program.shared_variables.out_cb;

    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_src0, *input_tensor.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, out_cb, *output_tensor.buffer());
}

}  // namespace ttnn::operations::pool::upsample::program
