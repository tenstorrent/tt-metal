// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TransposeHCRMProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: stick stream (reader produces, writer consumes)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_start_id_rm.cpp";

    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    const uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    const uint32_t NCH = N * C * H;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, NCH);

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_sticks = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                          : num_sticks_per_core_group_2;

    const uint32_t aligned_page = std::max(src0_buffer->aligned_page_size(), dst_buffer->aligned_page_size());
    const uint32_t stick_size = std::max(W * input_tensor.element_size(), aligned_page);

    // ------------------------------------------------------------------------
    // DataflowBufferSpec (legacy CB c_0, normal/non-borrowed).
    // ------------------------------------------------------------------------
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = stick_size,
        .num_entries = num_sticks,
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

    // ------------------------------------------------------------------------
    // Reader: streams transposed sticks into cb_in0 (c_0). W_size_bytes == stick_size.
    // (The legacy unused aligned_page_size CTA slot is dropped under named args.)
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"N", N}, {"H", H}, {"C", C}, {"W_size_bytes", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core_read", "num_read_per_barrier", "start_id", "curr_c", "curr_h", "curr_n"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ------------------------------------------------------------------------
    // Writer: consumes cb_in0 (c_0) and writes the transposed sticks (Case-1).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args = {{"W_size_bytes", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core_read", "num_read_per_barrier", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args: the HC stick walk, preserved exactly from legacy.
    // Kernels are placed on the full grid; idle cores carry num_sticks = 0.
    // ------------------------------------------------------------------------
    const uint32_t W_bytes = W * input_tensor.element_size();
    const uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core_read", num_sticks_per_core_read},
              {"num_read_per_barrier", num_read_per_barrier},
              {"start_id", curr_sticks_read},
              {"curr_c", curr_c},
              {"curr_h", curr_h},
              {"curr_n", curr_n}}});
        writer_run.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core_read", num_sticks_per_core_read},
              {"num_read_per_barrier", num_read_per_barrier},
              {"start_id", curr_sticks_write}}});

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }

    WorkUnitSpec wu{
        .name = "transpose_hc_rm",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = total_cores,
    };

    ProgramSpec spec{
        .name = "transpose_hc_rm",
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

}  // namespace ttnn::prim::qsr
