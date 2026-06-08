// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <cmath>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

using FixedPoint = int32_t;
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

static FixedPoint float_to_fixed(float value) { return static_cast<FixedPoint>(value * FIXED_ONE); }

ProgramDescriptor UpsampleBilinearProgramFactory::create_descriptor(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const ttnn::Tensor& input = input_tensor;
    const ttnn::Tensor& output = output_tensor;
    // This factory only supports integer scale factors
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Bilinear upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_factor_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Use the sliding window config passed from upsample.cpp (contains original input dimensions)
    TT_FATAL(
        operation_attributes.sliding_window_config.has_value(),
        "Bilinear upsample requires sliding_window_config to be provided");
    const ttnn::operations::sliding_window::SlidingWindowConfig sliding_window_config =
        operation_attributes.sliding_window_config.value();

    // Extract original (pre-halo) dimensions from sliding_window_config
    // These are the TRUE dimensions, not the haloed tensor dimensions
    const uint32_t in_batch_size = sliding_window_config.batch_size;
    const uint32_t in_h = sliding_window_config.input_hw.first;
    const uint32_t in_w = sliding_window_config.input_hw.second;
    const uint32_t in_channels = sliding_window_config.channels;

    // Output dimensions
    const Shape& output_shape = output.padded_shape();
    const uint32_t out_w = output_shape[2];

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

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

    const ShardSpec shard_spec = input.shard_spec().value();
    const CoreRangeSet all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();
    const uint32_t ncores_nhw = ncores;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    uint32_t input_block_size_bytes = input_stick_nbytes;
    input_block_size_bytes =
        std::min(input_block_size_bytes, MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size());

    const ttnn::Tensor& halo_in = input;
    const std::array<uint32_t, 2> halo_shard_shape = halo_in.shard_spec().value().shape;

    const std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata_bilinear(sliding_window_config);

    ProgramDescriptor desc;

    uint32_t next_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t buffering_factor = 2;

    // input data is in a sharded CB
    const uint32_t in_cb_pagesize = input_stick_nbytes;
    const uint32_t in_cb_npages = halo_shard_shape[0];
    const uint32_t in_ntiles_c = tt::div_up(in_channels, tt::constants::TILE_WIDTH);

    const uint32_t halo_cb_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(halo_cb_id),
            .data_format = input_cb_data_format,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = halo_in.buffer(),
    });

    // first intermediate CB (4 pixels per page are needed for intermediate tensor)
    const uint32_t in1_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    const uint32_t tilize_reduce_cb_0 = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_cb_pagesize * 4 * buffering_factor,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tilize_reduce_cb_0),
            .data_format = input_cb_data_format,
            .page_size = in1_cb_pagesize,
        }}},
    });

    // second intermediate CB
    const uint32_t tilize_reduce_cb_1 = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * 4 * buffering_factor,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tilize_reduce_cb_1),
            .data_format = input_cb_data_format,
            .page_size = in_cb_pagesize,
        }}},
    });

    // scalar intermediate CBs
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(input_cb_data_format);
    const uint32_t in_scalar_cb_npages = 1 * buffering_factor;

    const uint32_t in_scalar_cb_id1 = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_scalar_cb_pagesize * in_scalar_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_scalar_cb_id1),
            .data_format = input_cb_data_format,
            .page_size = in_scalar_cb_pagesize,
        }}},
    });

    const uint32_t in_scalar_cb_id2 = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_scalar_cb_pagesize * in_scalar_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_scalar_cb_id2),
            .data_format = input_cb_data_format,
            .page_size = in_scalar_cb_pagesize,
        }}},
    });

    // output sharded CB with upsampled data
    const uint32_t out_cb_pagesize = tt::constants::TILE_WIDTH * output.element_size();
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    const uint32_t out_cb_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_pagesize * out_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_id),
            .data_format = output_cb_data_format,
            .page_size = out_cb_pagesize,
        }}},
        .buffer = output.buffer(),
    });

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

    const uint32_t num_input_width_blocks = static_cast<uint32_t>(
        std::ceil(static_cast<float>(in_channels) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH)));

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        input_stick_nbytes,                            // [0] stick_nbytes
        scale_factor_h,                                // [1] scale_h
        scale_factor_w,                                // [2] scale_w
        in_w,                                          // [3] in_w (from sliding_window_config)
        out_w,                                         // [4] out_w
        in_h,                                          // [5] in_h (from sliding_window_config)
        halo_cb_id,                                    // [6] halo_cb_id
        tilize_reduce_cb_0,                            // [7] tilize_reduce_cb_0
        in_scalar_cb_id1,                              // [8] in_scalar_cb_id
        static_cast<uint32_t>(scale_h_inv_fixed),      // [9] scale_h_inv_comp
        static_cast<uint32_t>(scale_w_inv_fixed),      // [10] scale_w_inv_comp
        static_cast<uint32_t>(y_index_fixed),          // [11] y_starting_coordinate_u32
        static_cast<uint32_t>(x_index_compute_fixed),  // [12] x_starting_coordinate_u32
        1,                                             // [13] is_reader
        num_input_width_blocks,                        // [14] blocks
        input_block_size_bytes,                        // [15] input_block_size_bytes
    };

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        input_stick_nbytes,
        scale_factor_h,
        scale_factor_w,
        in_w,
        out_w,
        in_h,
        halo_cb_id,
        tilize_reduce_cb_1,
        in_scalar_cb_id2,
        static_cast<uint32_t>(scale_h_inv_fixed),
        static_cast<uint32_t>(scale_w_inv_fixed),
        static_cast<uint32_t>(y_index_fixed),
        static_cast<uint32_t>(x_index_compute_fixed),
        0,  // is_reader (0 for writer)
        num_input_width_blocks,
        input_block_size_bytes,
    };

    const std::string reader_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp";
    // Yes, the writer also runs reader_bilinear_multi_core_sharded.cpp — the kernel branches on the is_reader CT arg.
    const std::string writer_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp";
    const std::string compute_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp";

    KernelDescriptor::CompileTimeArgs compute_compile_time_args = {
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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_fname;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_fname;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported for upsample bilinear");

    constexpr ReduceOpMath reduce_op = ReduceOpMath::SUM;
    constexpr ReduceOpDim reduce_dim = ReduceOpDim::H;

    const std::map<std::string, std::string> reduce_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    KernelDescriptor::Defines compute_defines(reduce_defines_map.begin(), reduce_defines_map.end());

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel_fname;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Calculate work distribution based on output sticks
    const uint32_t total_output_sticks = in_batch_size * output.logical_shape()[1] * output.logical_shape()[2];
    const uint32_t max_out_sticks_per_core = tt::div_up(total_output_sticks, ncores_nhw);

    log_debug(
        tt::LogOp,
        "total_output_sticks: {}, max_out_sticks_per_core: {}",
        total_output_sticks,
        max_out_sticks_per_core);

    const std::vector<CoreCoord> logical_cores = corerange_to_cores(
        shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t start_output_idx = 0;
    uint32_t total_sticks_processed = 0;

    for (const auto& core_coord : logical_cores) {
        // Calculate actual output sticks for this core
        const uint32_t out_sticks_this_core =
            std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);

        if (out_sticks_this_core == 0) {
            // No work for this core
            const KernelDescriptor::CoreRuntimeArgs no_work_args = {start_output_idx, 0, 0};
            reader_desc.runtime_args.emplace_back(core_coord, no_work_args);
            writer_desc.runtime_args.emplace_back(core_coord, no_work_args);
            compute_desc.runtime_args.emplace_back(core_coord, KernelDescriptor::CoreRuntimeArgs{0});
            continue;
        }

        // Calculate the output range for this core (only after confirming there's work)
        const uint32_t output_index_start = start_output_idx;
        const uint32_t output_index_end =
            std::min(output_index_start + out_sticks_this_core, static_cast<uint32_t>(op_trace_metadata.size())) - 1;

        // Find the minimum input index for this core's output range
        const std::pair<uint32_t, uint32_t> minmax_indices =
            ttnn::operations::sliding_window::find_minmax_trace_indices(
                op_trace_metadata, output_index_start, output_index_end);
        const uint32_t min_trace_idx = minmax_indices.first;
        const uint32_t min_input_offset = op_trace_metadata[min_trace_idx];

        const KernelDescriptor::CoreRuntimeArgs rw_args = {start_output_idx, min_input_offset, out_sticks_this_core};
        reader_desc.runtime_args.emplace_back(core_coord, rw_args);
        writer_desc.runtime_args.emplace_back(core_coord, rw_args);
        compute_desc.runtime_args.emplace_back(core_coord, KernelDescriptor::CoreRuntimeArgs{out_sticks_this_core});

        // Next core starts where this core ends
        start_output_idx += out_sticks_this_core;
        total_sticks_processed += out_sticks_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
