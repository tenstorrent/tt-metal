// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace metal2 = tt::tt_metal::experimental;

using FixedPoint = int32_t;
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

static FixedPoint float_to_fixed(float value) { return static_cast<FixedPoint>(value * FIXED_ONE); }

ttnn::device_operation::ProgramArtifacts UpsampleBilinearProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const ttnn::Tensor& input = input_tensor;
    const ttnn::Tensor& output = output_tensor;
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Program-scope resource names (typed handles → generated dfb:: / tensor:: tokens)
    const metal2::KernelSpecName READER{"reader"};
    const metal2::KernelSpecName WRITER{"writer"};
    const metal2::KernelSpecName COMPUTE{"compute"};
    const metal2::DFBSpecName HALO{"halo"};                        // borrowed input (legacy c_0)
    const metal2::DFBSpecName TILIZE_REDUCE_0{"tilize_reduce_0"};  // legacy c_1
    const metal2::DFBSpecName TILIZE_REDUCE_1{"tilize_reduce_1"};  // legacy c_2
    const metal2::DFBSpecName IN_SCALAR_0{"in_scalar_0"};          // legacy c_3
    const metal2::DFBSpecName IN_SCALAR_1{"in_scalar_1"};          // legacy c_4
    const metal2::DFBSpecName OUT{"out"};                          // borrowed output (legacy c_5)
    const metal2::TensorParamName INPUT{"input"};
    const metal2::TensorParamName OUTPUT{"output"};

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

    constexpr uint32_t buffering_factor = 2;

    // input data is in a sharded DFB (borrowed from the input tensor)
    const uint32_t in_cb_pagesize = input_stick_nbytes;
    const uint32_t in_cb_npages = halo_shard_shape[0];
    const uint32_t in_ntiles_c = tt::div_up(in_channels, tt::constants::TILE_WIDTH);

    metal2::DataflowBufferSpec halo_dfb{
        .unique_id = HALO,
        .entry_size = in_cb_pagesize,
        .num_entries = in_cb_npages,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT,
    };

    // first intermediate DFB (4 pixels per page are needed for intermediate tensor)
    const uint32_t in1_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    metal2::DataflowBufferSpec tilize_reduce_dfb_0{
        .unique_id = TILIZE_REDUCE_0,
        .entry_size = in1_cb_pagesize,
        .num_entries = 4 * buffering_factor,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };

    // second intermediate DFB
    metal2::DataflowBufferSpec tilize_reduce_dfb_1{
        .unique_id = TILIZE_REDUCE_1,
        .entry_size = in_cb_pagesize,
        .num_entries = 4 * buffering_factor,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };

    // scalar intermediate DFBs
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(input_cb_data_format);
    const uint32_t in_scalar_cb_npages = 1 * buffering_factor;

    metal2::DataflowBufferSpec in_scalar_dfb_0{
        .unique_id = IN_SCALAR_0,
        .entry_size = in_scalar_cb_pagesize,
        .num_entries = in_scalar_cb_npages,
        .data_format_metadata = input_cb_data_format,
    };
    metal2::DataflowBufferSpec in_scalar_dfb_1{
        .unique_id = IN_SCALAR_1,
        .entry_size = in_scalar_cb_pagesize,
        .num_entries = in_scalar_cb_npages,
        .data_format_metadata = input_cb_data_format,
    };

    // output sharded DFB with upsampled data (borrowed from the output tensor)
    const uint32_t out_cb_pagesize = tt::constants::TILE_WIDTH * output.element_size();
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    metal2::DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = out_cb_pagesize,
        .num_entries = out_cb_npages,
        .data_format_metadata = output_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 1, .num_faces = 2},
        .borrowed_from = OUTPUT,
    };

    log_debug(tt::LogOp, "input_dfb: {}, npages: {}, pagesize: {}", HALO, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "output_dfb: {}, npages: {}, pagesize: {}", OUT, out_cb_npages, out_cb_pagesize);
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

    // Named compile-time args common to reader and writer (CB indices moved to DFB bindings).
    // `is_reader` is set per-instance below.
    auto make_dm_cta = [&](uint32_t is_reader) {
        return metal2::KernelSpec::CompileTimeArgs{
            {"stick_nbytes", input_stick_nbytes},
            {"scale_h", scale_factor_h},
            {"scale_w", scale_factor_w},
            {"in_w", in_w},
            {"out_w", out_w},
            {"in_h", in_h},
            {"scale_h_inv_fixed_u32", static_cast<uint32_t>(scale_h_inv_fixed)},
            {"scale_w_inv_fixed_u32", static_cast<uint32_t>(scale_w_inv_fixed)},
            {"y_starting_coordinate_fixed_u32", static_cast<uint32_t>(y_index_fixed)},
            {"x_starting_coordinate_fixed_u32", static_cast<uint32_t>(x_index_compute_fixed)},
            {"is_reader", is_reader},
            {"blocks", num_input_width_blocks},
            {"input_block_size_bytes", input_block_size_bytes},
        };
    };

    constexpr const char* dm_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp";
    constexpr const char* compute_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp";

    // Reader instance: produces the c_1/c_3 buffers behind accessors `tilize_reduce` / `in_scalar`, and
    // is the (cosmetic, Gen1) PRODUCER endpoint of the borrowed halo buffer that both DM instances raw-read.
    metal2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{dm_kernel_fname},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = HALO, .accessor_name = "halo", .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = TILIZE_REDUCE_0,
                 .accessor_name = "tilize_reduce",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = IN_SCALAR_0,
                 .accessor_name = "in_scalar",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .compile_time_args = make_dm_cta(/*is_reader=*/1),
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_idx", "min_input_offset", "out_sticks_this_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    // Writer instance: same source, branches on is_reader=0; produces c_2/c_4 behind the same accessors,
    // and is the CONSUMER endpoint of the borrowed halo buffer.
    metal2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{dm_kernel_fname},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = HALO, .accessor_name = "halo", .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = TILIZE_REDUCE_1,
                 .accessor_name = "tilize_reduce",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = IN_SCALAR_1,
                 .accessor_name = "in_scalar",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .compile_time_args = make_dm_cta(/*is_reader=*/0),
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_idx", "min_input_offset", "out_sticks_this_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };

    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported for upsample bilinear");

    constexpr ReduceOpMath reduce_op = ReduceOpMath::SUM;
    constexpr ReduceOpDim reduce_dim = ReduceOpDim::H;
    const std::map<std::string, std::string> reduce_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);

    // Compute hardware config. Style A (op resolves a TTNN ComputeKernelConfig): translate the resolved
    // config. The legacy factory dropped dst_full_sync_en (never set on its ComputeConfigDescriptor), so it
    // used the descriptor default (false) → double_buffer_dest = true; reproduce that faithfully. fp32 dest
    // is asserted off above, so enable_32_bit_dest is false and no unpack_modes entry is required.
    metal2::ComputeHardwareConfig compute_hw =
        ttnn::to_compute_hardware_config(input.device()->arch(), compute_kernel_config);
    if (auto* gen1 = std::get_if<metal2::ComputeGen1Config>(&compute_hw)) {
        gen1->double_buffer_dest = true;
    }

    // Compute kernel consumes the reader/writer intermediate buffers (c_1..c_4) and self-loops the borrowed
    // output buffer (c_5): it is the sole toucher (pack_untilize_dest + a raw fifo_wr_ptr advance), so it is
    // bound as both PRODUCER and CONSUMER of OUT.
    metal2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{compute_kernel_fname},
        .compiler_options =
            {.defines = metal2::KernelSpec::CompilerOptions::Defines(reduce_defines_map),
             .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = TILIZE_REDUCE_0,
                 .accessor_name = "tilize_reduce_0",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = TILIZE_REDUCE_1,
                 .accessor_name = "tilize_reduce_1",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = IN_SCALAR_0,
                 .accessor_name = "in_scalar_1",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = IN_SCALAR_1,
                 .accessor_name = "in_scalar_2",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = metal2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {
                {"in_ntiles_c", in_ntiles_c},
                {"in_ntiles_hwc", 1 * in_ntiles_c},
                {"window_size_hw", 4},
                {"out_ntiles_c", tt::div_up(in_channels, tt::constants::TILE_WIDTH)},
                {"blocks", num_input_width_blocks},
                {"input_block_size_bytes", input_block_size_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"nsticks_per_core"}},
        .hw_config = compute_hw,
    };

    metal2::ProgramSpec spec{
        .name = "upsample_bilinear_multicore",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {halo_dfb, tilize_reduce_dfb_0, tilize_reduce_dfb_1, in_scalar_dfb_0, in_scalar_dfb_1, out_dfb},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output_mesh.tensor_spec()},
            },
        .work_units = {metal2::WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_cores,
        }},
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

    // Per-node runtime args. Legacy node-first loop preserved; AddRuntimeArgsForNode transposes into the
    // name-first ProgramRunArgs table.
    metal2::ProgramRunArgs run_args;
    metal2::KernelRunArgs reader_ra{.kernel = READER};
    metal2::KernelRunArgs writer_ra{.kernel = WRITER};
    metal2::KernelRunArgs compute_ra{.kernel = COMPUTE};

    uint32_t start_output_idx = 0;
    uint32_t total_sticks_processed = 0;

    for (const auto& core_coord : logical_cores) {
        // Calculate actual output sticks for this core
        const uint32_t out_sticks_this_core =
            std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);

        if (out_sticks_this_core == 0) {
            // No work for this core
            metal2::AddRuntimeArgsForNode(
                reader_ra.runtime_arg_values,
                core_coord,
                {{"start_output_idx", start_output_idx}, {"min_input_offset", 0}, {"out_sticks_this_core", 0}});
            metal2::AddRuntimeArgsForNode(
                writer_ra.runtime_arg_values,
                core_coord,
                {{"start_output_idx", start_output_idx}, {"min_input_offset", 0}, {"out_sticks_this_core", 0}});
            metal2::AddRuntimeArgsForNode(compute_ra.runtime_arg_values, core_coord, {{"nsticks_per_core", 0}});
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

        metal2::AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core_coord,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"out_sticks_this_core", out_sticks_this_core}});
        metal2::AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core_coord,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"out_sticks_this_core", out_sticks_this_core}});
        metal2::AddRuntimeArgsForNode(
            compute_ra.runtime_arg_values, core_coord, {{"nsticks_per_core", out_sticks_this_core}});

        // Next core starts where this core ends
        start_output_idx += out_sticks_this_core;
        total_sticks_processed += out_sticks_this_core;
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra), std::move(compute_ra)};
    run_args.tensor_args = {
        {INPUT, metal2::TensorArgument{input_mesh}},
        {OUTPUT, metal2::TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
