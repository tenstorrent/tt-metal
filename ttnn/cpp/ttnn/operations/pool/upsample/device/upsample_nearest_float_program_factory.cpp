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

using namespace tt::tt_metal::experimental;

constexpr uint32_t BUFFERING_FACTOR = 2;

namespace {

const KernelSpecName NEARFLOAT_READER{"upsample_nearfloat_reader"};
const KernelSpecName NEARFLOAT_WRITER{"upsample_nearfloat_writer"};
const DFBSpecName NEARFLOAT_OUT{"upsample_nearfloat_out"};
const TensorParamName NEARFLOAT_INPUT{"upsample_nearfloat_input"};
const TensorParamName NEARFLOAT_OUTPUT{"upsample_nearfloat_output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleNearestFloatProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input, Tensor& output_tensor) {
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output_tensor.mesh_tensor();

    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = output_tensor.device();

    const auto& input_shape = input.logical_shape();
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];

    // Output dimensions (from logical shape)
    const uint32_t output_height =
        static_cast<uint32_t>(std::floor(input_height * operation_attributes.scale_factor_h));
    const uint32_t output_width = static_cast<uint32_t>(std::floor(input_width * operation_attributes.scale_factor_w));

    // Calculate reciprocal scale factors for kernel (fixed-point Q16.16)
    // src = floor(dst / scale) = floor(dst * reciprocal_scale)
    // We need to round UP the reciprocal to ensure boundary values are handled correctly.
    // For example, with scale=3, dst=3: we need floor(3/3)=1, which requires 3*(1/3) >= 1.0
    // Rounding up the reciprocal ensures this property is maintained.
    constexpr int32_t FIXED_ONE = 1 << 16;
    const float reciprocal_scale_h = 1.0f / operation_attributes.scale_factor_h;
    const float reciprocal_scale_w = 1.0f / operation_attributes.scale_factor_w;
    const int32_t reciprocal_scale_h_fixed = static_cast<int32_t>(std::ceil(reciprocal_scale_h * FIXED_ONE));
    const int32_t reciprocal_scale_w_fixed = static_cast<int32_t>(std::ceil(reciprocal_scale_w * FIXED_ONE));

    // Work distribution - Total work units = N * H_out * W_out (one output stick per work unit)
    const uint32_t total_pages_in_output = output_tensor.buffer()->num_pages();

    const tt::tt_metal::Shape& output_shape = output_tensor.padded_shape();

    const uint32_t num_pages_across_width =
        total_pages_in_output / (output_shape[0] * output_shape[1] * output_shape[2]);

    const uint32_t aligned_input_page_size = input.buffer()->aligned_page_size();
    const uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();

    const uint32_t input_page_size = input.buffer()->page_size();
    const uint32_t output_page_size = output_tensor.buffer()->page_size();

    TT_FATAL(
        input_page_size == output_page_size,
        "Input and output page sizes must match for nearest upsample, got input_page_size={} output_page_size={}",
        input_page_size,
        output_page_size);

    const tt::tt_metal::CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_pages_in_output);

    const std::vector<tt::tt_metal::CoreCoord> logical_cores =
        tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, true);

    // Calculate stick sizes (aligned based on buffer type for efficient reads)
    const uint32_t num_cb_pages = BUFFERING_FACTOR;
    const uint32_t output_cb_page_size = aligned_output_page_size;

    DataflowBufferSpec out_dfb{
        .unique_id = NEARFLOAT_OUT,
        .entry_size = output_cb_page_size,
        .num_entries = num_cb_pages * BUFFERING_FACTOR,
        .data_format_metadata = output_cb_data_format,
    };

    KernelSpec reader{
        .unique_id = NEARFLOAT_READER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_upsample_nearest_float.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = NEARFLOAT_OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = NEARFLOAT_INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"aligned_input_page_size", aligned_input_page_size},
                {"input_height", input_height},
                {"input_width", input_width},
                {"output_height", output_height},
                {"output_width", output_width},
                {"num_pages_across_width", num_pages_across_width},
                {"reciprocal_scale_h_fixed", static_cast<uint32_t>(reciprocal_scale_h_fixed)},
                {"reciprocal_scale_w_fixed", static_cast<uint32_t>(reciprocal_scale_w_fixed)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = NEARFLOAT_WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_nearest_float.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = NEARFLOAT_OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = NEARFLOAT_OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"aligned_stick_nbytes", aligned_output_page_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_stick_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ProgramSpec spec{
        .name = "upsample_nearest_float",
        .kernels = {reader, writer},
        .dataflow_buffers = {out_dfb},
        .tensor_parameters =
            {
                {.unique_id = NEARFLOAT_INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = NEARFLOAT_OUTPUT, .spec = output_mesh.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {NEARFLOAT_READER, NEARFLOAT_WRITER},
            .target_nodes = all_cores,
        }},
    };

    // Set runtime arguments for each core
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = NEARFLOAT_READER};
    KernelRunArgs writer_run_args{.kernel = NEARFLOAT_WRITER};

    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks", num_sticks}, {"start_stick_id", sticks_processed}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks", num_sticks}, {"start_stick_id", sticks_processed}});

        sticks_processed += num_sticks;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {NEARFLOAT_INPUT, TensorArgument{input_mesh}},
        {NEARFLOAT_OUTPUT, TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
