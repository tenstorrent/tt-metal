// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_device_operation.hpp"

#include <cmath>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <ttnn/operations/pool/pool_utils.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace metal2 = tt::tt_metal::experimental;

constexpr std::uint32_t BUFFERING_FACTOR = 2;

ttnn::device_operation::ProgramArtifacts UpsampleNearestFloatProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input, Tensor& output_tensor) {
    const metal2::KernelSpecName READER{"reader"};
    const metal2::KernelSpecName WRITER{"writer"};
    const metal2::DFBSpecName OUT{"out"};
    const metal2::TensorParamName INPUT{"input"};
    const metal2::TensorParamName OUTPUT{"output"};

    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output_tensor.mesh_tensor();

    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = output_tensor.device();

    const auto& input_shape = input.logical_shape();
    const std::uint32_t input_height = input_shape[1];
    const std::uint32_t input_width = input_shape[2];

    // Output dimensions (from logical shape)
    const std::uint32_t output_height =
        static_cast<std::uint32_t>(std::floor(input_height * operation_attributes.scale_factor_h));
    const std::uint32_t output_width =
        static_cast<std::uint32_t>(std::floor(input_width * operation_attributes.scale_factor_w));

    // Calculate reciprocal scale factors for kernel (fixed-point Q16.16)
    // src = floor(dst / scale) = floor(dst * reciprocal_scale)
    // We need to round UP the reciprocal to ensure boundary values are handled correctly.
    // For example, with scale=3, dst=3: we need floor(3/3)=1, which requires 3*(1/3) >= 1.0
    // Rounding up the reciprocal ensures this property is maintained.
    constexpr std::int32_t FIXED_ONE = 1 << 16;
    const float reciprocal_scale_h = 1.0f / operation_attributes.scale_factor_h;
    const float reciprocal_scale_w = 1.0f / operation_attributes.scale_factor_w;
    const std::int32_t reciprocal_scale_h_fixed = static_cast<std::int32_t>(std::ceil(reciprocal_scale_h * FIXED_ONE));
    const std::int32_t reciprocal_scale_w_fixed = static_cast<std::int32_t>(std::ceil(reciprocal_scale_w * FIXED_ONE));

    // Work distribution - Total work units = N * H_out * W_out (one output stick per work unit)
    const std::uint32_t total_pages_in_output = output_tensor.buffer()->num_pages();

    const Shape& output_shape = output_tensor.padded_shape();

    const std::uint32_t num_pages_across_width =
        total_pages_in_output / (output_shape[0] * output_shape[1] * output_shape[2]);

    const std::uint32_t aligned_input_page_size = input.buffer()->aligned_page_size();
    const std::uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();

    const std::uint32_t input_page_size = input.buffer()->page_size();
    const std::uint32_t output_page_size = output_tensor.buffer()->page_size();

    TT_FATAL(
        input_page_size == output_page_size,
        "Input and output page sizes must match for nearest upsample, got input_page_size={} output_page_size={}",
        input_page_size,
        output_page_size);

    const CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            split_work_to_cores(compute_grid_size, total_pages_in_output);

    const std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, std::nullopt, true);

    // Calculate stick sizes (aligned based on buffer type for efficient reads)
    const std::uint32_t num_cb_pages = BUFFERING_FACTOR;
    const std::uint32_t output_cb_page_size = aligned_output_page_size;

    metal2::DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_cb_page_size,
        .num_entries = num_cb_pages * BUFFERING_FACTOR,
        .data_format_metadata = output_cb_data_format,
    };

    metal2::KernelSpec reader{
        .unique_id = READER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_upsample_nearest_float.cpp"},
        .dfb_bindings = {metal2::DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {metal2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"aligned_input_page_size", aligned_input_page_size},
                {"input_height", input_height},
                {"input_width", input_width},
                {"output_height", output_height},
                {"output_width", output_width},
                {"num_pages_across_width", num_pages_across_width},
                {"reciprocal_scale_h_fixed", static_cast<std::uint32_t>(reciprocal_scale_h_fixed)},
                {"reciprocal_scale_w_fixed", static_cast<std::uint32_t>(reciprocal_scale_w_fixed)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    metal2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_nearest_float.cpp"},
        .dfb_bindings = {metal2::DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = metal2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {metal2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"aligned_stick_nbytes", aligned_output_page_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    metal2::ProgramSpec spec{
        .name = "upsample_nearest_float",
        .kernels = {reader, writer},
        .dataflow_buffers = {out_dfb},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output_mesh.tensor_spec()},
            },
        .work_units = {metal2::WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    // Set runtime arguments for each core
    metal2::ProgramRunArgs run_args;
    metal2::KernelRunArgs reader_run_args{.kernel = READER};
    metal2::KernelRunArgs writer_run_args{.kernel = WRITER};

    std::uint32_t sticks_processed = 0;
    for (std::uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const std::uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        metal2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks", num_sticks}, {"start_stick_id", sticks_processed}});
        metal2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks", num_sticks}, {"start_stick_id", sticks_processed}});

        sticks_processed += num_sticks;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, metal2::TensorArgument{input_mesh}},
        {OUTPUT, metal2::TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
