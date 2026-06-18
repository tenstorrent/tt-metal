// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeCNProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    auto input_shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t page_shape[2] = {TILE_WIDTH, TILE_HEIGHT};
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        page_shape[0] = 1;
        page_shape[1] = input_shape[-1];
    }
    uint32_t page_size = page_shape[0] * page_shape[1];
    uint32_t stick_size = (row_major) ? page_shape[1] * input_tensor.element_size() : tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    uint32_t num_tensor_pages = input_tensor.physical_volume() / page_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_input_pages = 2;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_cn";

    // src0 DFB (legacy CB c_0): one page is read from src by the reader and consumed by the writer.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = stick_size,
            .num_entries = num_input_pages,
            .data_format_metadata = cb_data_format,
        },
    };

    // src0_buffer / dst_buffer addresses are no longer plumbed through a CTA + RTA; the legacy
    // TensorAccessorArgs(...).append_to(cta, common_rta) host plumbing and the kernel-side
    // TensorAccessorArgs<3>() / buffer-address RTA reads collapse to a TensorBinding end-to-end.
    // NOTE: the legacy factory built these accessors with ArgConfig::RuntimeTensorShape (tensor
    // shape passed as a runtime arg so one cached program serves varying shapes). The Metal 2.0
    // TensorParameter binding bakes the shape into the accessor's compile-time args by default;
    // the dynamic_tensor_shape relaxation is a known open item (shape baked into accessor CT args).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "reader_unary_transpose_cn_interleaved_start_id.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"},
            },
        .compile_time_args = {{"page_size", src0_buffer->aligned_page_size()}, {"read_size", stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id", "hw", "n"},
                .common_runtime_arg_names = {"N", "C", "HtWt", "batch_step", "channel_step"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "writer_unary_transpose_cn_interleaved_start_id.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"},
            },
        .compile_time_args = {{"page_size", dst_buffer->aligned_page_size()}, {"write_size", stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    if (row_major) {
        reader.compiler_options.defines = {{"CN_RM", "1"}};
        writer.compiler_options.defines = {{"CN_RM", "1"}};
    }

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // The src0 DFB's producer (reader) and consumer (writer) must share the same WorkUnitSpec —
    // every node hosting the DFB must host both endpoints. The legacy factory launches both kernels
    // on the full grid (total_cores); no-op cores receive num_pages = 0.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_cn",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = total_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / page_shape[1];
    uint32_t Ht = H / page_shape[0];
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_pages;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    // N/C/HtWt/batch_step/channel_step are uniform across cores → common runtime args.
    reader_run.common_runtime_arg_values = {
        {"N", N}, {"C", C}, {"HtWt", HtWt}, {"batch_step", batch_step}, {"channel_step", channel_step}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    for (uint32_t i = 0, num_pages_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        }

        uint32_t hw = num_pages_read % HtWt;
        uint32_t curr_c = num_pages_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_pages_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        reader_run.runtime_arg_values.push_back(
            {core, {{"num_pages", num_pages_per_core}, {"start_id", start_tile}, {"hw", hw}, {"n", n}}});
        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", num_pages_per_core}, {"start_id", num_pages_read}}});

        num_pages_read += num_pages_per_core;
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
