// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::prim {

namespace {

constexpr std::int32_t FIXED_POINT_SHIFT = 16;
constexpr std::int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

std::int32_t float_to_fixed(float value) { return static_cast<std::int32_t>(value * FIXED_ONE); }

}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleBilinearProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const tt::tt_metal::experimental::KernelSpecName reader_kernel{"reader"};
    const tt::tt_metal::experimental::KernelSpecName writer_kernel{"writer"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel{"compute"};
    const tt::tt_metal::experimental::DFBSpecName halo_dfb_name{"halo"};
    const tt::tt_metal::experimental::DFBSpecName tilize_reduce0_dfb_name{"tilize_reduce0"};
    const tt::tt_metal::experimental::DFBSpecName tilize_reduce1_dfb_name{"tilize_reduce1"};
    const tt::tt_metal::experimental::DFBSpecName scalar0_dfb_name{"scalar0"};
    const tt::tt_metal::experimental::DFBSpecName scalar1_dfb_name{"scalar1"};
    const tt::tt_metal::experimental::DFBSpecName output_dfb_name{"output"};
    const tt::tt_metal::experimental::TensorParamName input_param{"input"};
    const tt::tt_metal::experimental::TensorParamName output_param{"output"};

    const Tensor& input = input_tensor;
    Tensor& output = output_tensor;
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Bilinear upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const std::uint32_t scale_factor_h = static_cast<std::uint32_t>(operation_attributes.scale_factor_h);
    const std::uint32_t scale_factor_w = static_cast<std::uint32_t>(operation_attributes.scale_factor_w);

    TT_FATAL(
        operation_attributes.sliding_window_config.has_value(),
        "Bilinear upsample requires sliding_window_config to be provided");
    // The input tensor contains haloed shards; the sliding-window configuration retains
    // the original pre-halo N/H/W/C dimensions needed for coordinate reconstruction.
    const auto sliding_window_config = operation_attributes.sliding_window_config.value();
    const std::uint32_t in_batch_size = sliding_window_config.batch_size;
    const std::uint32_t in_h = sliding_window_config.input_hw.first;
    const std::uint32_t in_w = sliding_window_config.input_hw.second;
    const std::uint32_t in_channels = sliding_window_config.channels;

    // Bilinear coordinate reconstruction assumes the channels-last [N, H, W, C]
    // interpretation carried by the sliding-window metadata.

    const Shape& output_shape = output.padded_shape();
    const std::uint32_t out_w = output_shape[2];
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Bfp8_b row-major sticks are supported; Bfp8_b TILE input is not supported by
    // this haloed-stick bilinear path.

    TT_FATAL(in_channels % 32 == 0, "input channels should be divisible by 32");
    const std::uint32_t input_stick_nbytes = in_channels * input.element_size();
    const std::uint32_t output_stick_nbytes = output_shape[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    const auto compute_config_args =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    const bool fp32_dest_acc_en = std::get<2>(compute_config_args);
    TT_FATAL(!fp32_dest_acc_en, "fp32_dest_acc_en as true not supported for upsample bilinear");

    const tt::tt_metal::ShardSpec shard_spec = input.shard_spec().value();
    const tt::tt_metal::CoreRangeSet all_cores = shard_spec.grid;
    const std::uint32_t ncores = shard_spec.num_cores();
    constexpr std::uint32_t MAX_TILES_PER_REDUCTION = 8;
    const std::uint32_t input_block_size_bytes =
        std::min(input_stick_nbytes, MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size());

    const std::array<std::uint32_t, 2> halo_shard_shape = input.shard_spec().value().shape;
    const std::vector<std::uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata_bilinear(sliding_window_config);

    constexpr std::uint32_t BUFFERING_FACTOR = 2;
    const std::uint32_t in_ntiles_c = tt::div_up(in_channels, tt::constants::TILE_WIDTH);
    const std::uint32_t halo_page_size = input_stick_nbytes;
    const std::uint32_t tilize_reduce0_page_size =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    const std::uint32_t scalar_page_size = tt::tile_size(input_cb_data_format);
    const std::uint32_t output_page_size = tt::constants::TILE_WIDTH * output.element_size();

    tt::tt_metal::experimental::DataflowBufferSpec halo_dfb_spec{
        .unique_id = halo_dfb_name,
        .entry_size = halo_page_size,
        .num_entries = halo_shard_shape[0],
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = input_param,
    };
    tt::tt_metal::experimental::DataflowBufferSpec tilize_reduce0_dfb_spec{
        .unique_id = tilize_reduce0_dfb_name,
        .entry_size = tilize_reduce0_page_size,
        .num_entries = 4 * BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = tt::tt_metal::FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };
    tt::tt_metal::experimental::DataflowBufferSpec tilize_reduce1_dfb_spec{
        .unique_id = tilize_reduce1_dfb_name,
        .entry_size = tilize_reduce0_page_size,
        .num_entries = 4 * BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = tt::tt_metal::FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };
    tt::tt_metal::experimental::DataflowBufferSpec scalar0_dfb_spec{
        .unique_id = scalar0_dfb_name,
        .entry_size = scalar_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
    };
    tt::tt_metal::experimental::DataflowBufferSpec scalar1_dfb_spec{
        .unique_id = scalar1_dfb_name,
        .entry_size = scalar_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_cb_data_format,
    };
    tt::tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb_name,
        .entry_size = output_page_size,
        .num_entries = output.shard_spec().value().shape[0] * in_ntiles_c,
        .data_format_metadata = output_cb_data_format,
        .unpack_face_geometry_metadata = tt::tt_metal::FaceGeometry{.face_r_dim = 1, .num_faces = 2},
        .borrowed_from = output_param,
    };

    log_debug(tt::LogOp, "halo_dfb: {}, npages: {}, pagesize: {}", halo_dfb_name, halo_shard_shape[0], halo_page_size);
    log_debug(
        tt::LogOp,
        "output_dfb: {}, npages: {}, pagesize: {}",
        output_dfb_name,
        output.shard_spec().value().shape[0] * in_ntiles_c,
        output_page_size);
    log_debug(tt::LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(tt::LogOp, "ncores: {}", ncores);

    const float scale_h_inv = 1.0f / static_cast<float>(scale_factor_h);
    const float scale_w_inv = 1.0f / static_cast<float>(scale_factor_w);
    const std::int32_t scale_h_inv_fixed = float_to_fixed(scale_h_inv);
    const std::int32_t scale_w_inv_fixed = float_to_fixed(scale_w_inv);
    const std::int32_t y_index_fixed = float_to_fixed((0.5f * scale_h_inv) + 0.5f);
    const std::int32_t x_index_fixed = float_to_fixed((0.5f * scale_w_inv) + 0.5f);
    const std::uint32_t num_input_width_blocks = static_cast<std::uint32_t>(
        std::ceil(static_cast<float>(in_channels) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH)));

    const tt::tt_metal::experimental::KernelSpec::CompileTimeArgs common_dataflow_cta{
        {"stick_nbytes", input_stick_nbytes},
        {"scale_h", scale_factor_h},
        {"scale_w", scale_factor_w},
        {"in_w", in_w},
        {"out_w", out_w},
        {"in_h", in_h},
        {"scale_h_inv_fixed_u32", static_cast<std::uint32_t>(scale_h_inv_fixed)},
        {"scale_w_inv_fixed_u32", static_cast<std::uint32_t>(scale_w_inv_fixed)},
        {"y_starting_coordinate_fixed_u32", static_cast<std::uint32_t>(y_index_fixed)},
        {"x_starting_coordinate_fixed_u32", static_cast<std::uint32_t>(x_index_fixed)},
        {"blocks", num_input_width_blocks},
        {"input_block_size_bytes", input_block_size_bytes},
    };
    const auto make_dataflow_cta = [&](std::uint32_t is_reader) {
        auto cta = common_dataflow_cta;
        cta.insert({"is_reader", is_reader});
        return cta;
    };

    constexpr const char* DATAFLOW_KERNEL =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp";
    tt::tt_metal::experimental::KernelSpec reader{
        .unique_id = reader_kernel,
        .source = DATAFLOW_KERNEL,
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ProducerOf(halo_dfb_name, "halo"),
                tt::tt_metal::experimental::ProducerOf(tilize_reduce0_dfb_name, "tilize_reduce"),
                tt::tt_metal::experimental::ProducerOf(scalar0_dfb_name, "scalar"),
            },
        .compile_time_args = make_dataflow_cta(/*is_reader=*/1),
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_idx", "min_input_offset", "output_shard_height"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };
    tt::tt_metal::experimental::KernelSpec writer{
        .unique_id = writer_kernel,
        .source = DATAFLOW_KERNEL,
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ConsumerOf(halo_dfb_name, "halo"),
                tt::tt_metal::experimental::ProducerOf(tilize_reduce1_dfb_name, "tilize_reduce"),
                tt::tt_metal::experimental::ProducerOf(scalar1_dfb_name, "scalar"),
            },
        .compile_time_args = make_dataflow_cta(/*is_reader=*/0),
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_idx", "min_input_offset", "output_shard_height"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };
    // Reader and writer intentionally share one source; is_reader selects the two roles at
    // compile time while each role keeps its architecture-preferred NoC configuration.

    auto compute_hw_config =
        ttnn::to_compute_hardware_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel,
        .source = "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ConsumerOf(tilize_reduce0_dfb_name, "tilize_reduce0"),
                tt::tt_metal::experimental::ConsumerOf(tilize_reduce1_dfb_name, "tilize_reduce1"),
                tt::tt_metal::experimental::ConsumerOf(scalar0_dfb_name, "scalar0"),
                tt::tt_metal::experimental::ConsumerOf(scalar1_dfb_name, "scalar1"),
                tt::tt_metal::experimental::ProducerOf(output_dfb_name, "output"),
                tt::tt_metal::experimental::ConsumerOf(output_dfb_name, "output"),
            },
        .compile_time_args =
            {
                {"in_ntiles_c", in_ntiles_c},
                {"blocks", num_input_width_blocks},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"nsticks_per_core_by_nblocks"}},
        .hw_config = std::move(compute_hw_config),
    };

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "upsample_bilinear_multicore",
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {halo_dfb_spec,
             tilize_reduce0_dfb_spec,
             tilize_reduce1_dfb_spec,
             scalar0_dfb_spec,
             scalar1_dfb_spec,
             output_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = input_param, .spec = input_mesh.tensor_spec()},
                {.unique_id = output_param, .spec = output_mesh.tensor_spec()},
            },
        .work_units = {tt::tt_metal::experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {reader_kernel, writer_kernel, compute_kernel},
            .target_nodes = all_cores,
        }},
    };

    tt::tt_metal::experimental::KernelRunArgs reader_run{.kernel = reader_kernel};
    tt::tt_metal::experimental::KernelRunArgs writer_run{.kernel = writer_kernel};
    tt::tt_metal::experimental::KernelRunArgs compute_run{.kernel = compute_kernel};
    const std::uint32_t total_output_sticks = in_batch_size * output.logical_shape()[1] * output.logical_shape()[2];
    const std::uint32_t max_out_sticks_per_core = tt::div_up(total_output_sticks, ncores);
    const std::vector<tt::tt_metal::CoreCoord> logical_cores = tt::tt_metal::corerange_to_cores(
        shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    std::uint32_t start_output_idx = 0;
    std::uint32_t total_sticks_processed = 0;
    for (const tt::tt_metal::CoreCoord& core : logical_cores) {
        const std::uint32_t out_sticks_this_core =
            std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);
        std::uint32_t min_input_offset = 0;
        if (out_sticks_this_core > 0) {
            // Derive the smallest input trace entry touched by this core's output interval.
            // Empty tail cores keep offset zero and execute no device work.
            const std::uint32_t output_index_end =
                std::min(
                    start_output_idx + out_sticks_this_core, static_cast<std::uint32_t>(op_trace_metadata.size())) -
                1;
            const auto [min_trace_idx, max_trace_idx] = ttnn::operations::sliding_window::find_minmax_trace_indices(
                op_trace_metadata, start_output_idx, output_index_end);
            (void)max_trace_idx;
            min_input_offset = op_trace_metadata[min_trace_idx];
        }

        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"output_shard_height", out_sticks_this_core}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"output_shard_height", out_sticks_this_core}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            compute_run.runtime_arg_values, core, {{"nsticks_per_core_by_nblocks", out_sticks_this_core}});

        start_output_idx += out_sticks_this_core;
        total_sticks_processed += out_sticks_this_core;
    }

    tt::tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)},
        .tensor_args =
            {
                {input_param, tt::tt_metal::experimental::TensorArgument{input_mesh}},
                {output_param, tt::tt_metal::experimental::TensorArgument{output_mesh}},
            },
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
