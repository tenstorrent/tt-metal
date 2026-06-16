// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeCNProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: input stream (reader produces, writer consumes)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_cn_interleaved_start_id.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_cn_interleaved_start_id.cpp";

    const auto& input_tensor = tensor_args.input;
    auto input_shape = input_tensor.padded_shape();
    const bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t page_shape[2] = {TILE_WIDTH, TILE_HEIGHT};
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        page_shape[0] = 1;
        page_shape[1] = input_shape[-1];
    }
    const uint32_t page_size = page_shape[0] * page_shape[1];
    const uint32_t stick_size =
        (row_major) ? page_shape[1] * input_tensor.element_size() : tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    const uint32_t num_tensor_pages = input_tensor.physical_volume() / page_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // ------------------------------------------------------------------------
    // DataflowBufferSpec (legacy CB c_0, normal/non-borrowed).
    // ------------------------------------------------------------------------
    const uint32_t num_input_pages = 2;
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = stick_size,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters. Both carried RuntimeTensorShape on the legacy accessors,
    // which maps to dynamic_tensor_shape = true.
    // ------------------------------------------------------------------------
    TensorParameter input_param{
        .unique_id = INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };
    TensorParameter output_param{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };

    // CTA values, preserved from the legacy compile-time args.
    const uint32_t reader_page_size = src0_buffer->aligned_page_size();
    const uint32_t writer_page_size = dst_buffer->aligned_page_size();

    KernelSpec::CompilerOptions reader_compiler_options;
    KernelSpec::CompilerOptions writer_compiler_options;
    if (row_major) {
        reader_compiler_options.defines = {{"CN_RM", "1"}};
        writer_compiler_options.defines = {{"CN_RM", "1"}};
    }

    // ------------------------------------------------------------------------
    // Reader: walks the CN index space, streaming pages into cb_in0 (c_0).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .compiler_options = reader_compiler_options,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"page_size", reader_page_size}, {"read_size", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"N", "C", "HtWt", "batch_step", "channel_step", "num_pages", "start_id", "hw", "n"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ------------------------------------------------------------------------
    // Writer: consumes cb_in0 (c_0) and writes the output pages (Case-1).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .compiler_options = writer_compiler_options,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args = {{"page_size", writer_page_size}, {"write_size", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args: the CN walk state, preserved exactly from legacy.
    // Kernels are placed on the full grid; idle cores carry num_pages = 0.
    // ------------------------------------------------------------------------
    const uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    const uint32_t Wt = W / page_shape[1];
    const uint32_t Ht = H / page_shape[0];
    const uint32_t HtWt = Ht * Wt;
    const uint32_t CHtWt = C * HtWt;
    const uint32_t NCHtWt = num_tensor_pages;
    const uint32_t batch_step = CHtWt - HtWt;
    const uint32_t channel_step = NCHtWt - HtWt;

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, num_pages_read = 0; i < num_cores_total; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        }

        const uint32_t hw = num_pages_read % HtWt;
        const uint32_t curr_c = num_pages_read / HtWt;
        const uint32_t n = curr_c % N;
        const uint32_t start_tile = num_pages_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"N", N},
              {"C", C},
              {"HtWt", HtWt},
              {"batch_step", batch_step},
              {"channel_step", channel_step},
              {"num_pages", num_pages_per_core},
              {"start_id", start_tile},
              {"hw", hw},
              {"n", n}}});
        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages", num_pages_per_core}, {"start_id", num_pages_read}}});

        num_pages_read += num_pages_per_core;
    }

    WorkUnitSpec wu{
        .name = "transpose_cn",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = total_cores,
    };

    ProgramSpec spec{
        .name = "transpose_cn",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
