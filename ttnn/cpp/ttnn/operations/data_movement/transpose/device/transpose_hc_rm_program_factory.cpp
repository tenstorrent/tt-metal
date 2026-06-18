// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Compute per-core runtime args (reader+writer) for HC RM transpose and append them to the
// supplied KernelRunArgs. The traversal logic that advances (curr_c, curr_h, curr_n) was
// previously shared between `create` and `override_runtime_arguments`; now it has a single home.
// Only the dispatch channel changes (named RTAs); the buffer-address RTA (legacy slot 0) is
// replaced by the input/output TensorBindings, so it is dropped here.
void emit_runtime_args_hc_rm(
    m2::KernelRunArgs& reader_run,
    m2::KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    reader_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
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

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"num_sticks_per_core_read", num_sticks_per_core_read},
              {"num_read_per_barrier", num_read_per_barrier},
              {"start_id", curr_sticks_read},
              {"curr_c", curr_c},
              {"curr_h", curr_h},
              {"curr_n", curr_n}}});

        writer_run.runtime_arg_values.push_back(
            {core,
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
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCRMProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t NCH = N * C * H;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, NCH);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto num_sticks = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                : num_sticks_per_core_group_2;

    Buffer* src0_buffer = input_tensor.buffer();
    uint32_t aligned_page = std::max(src0_buffer->aligned_page_size(), dst_buffer->aligned_page_size());
    auto stick_size = std::max(W * input_tensor.element_size(), aligned_page);

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_hc_rm";

    // src0 DFB (legacy CB c_0): the reader produces sticks into it, the writer consumes them.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = stick_size,
            .num_entries = num_sticks,
            .data_format_metadata = cb_data_format,
        },
    };

    // The legacy factory built the input/output accessors with TensorAccessorArgs(RuntimeTensorShape)
    // and plumbed the buffer addresses through RTA slot 0; both collapse to the TensorBindings below.
    // The dynamic_tensor_shape relaxation is a known open item. The legacy
    // CTA `src0_buffer->aligned_page_size()` was consumed only by the TensorAccessorArgs plumbing
    // (never read by the kernel) and disappears with it.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp"},
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
        .compile_time_args = {{"N", N}, {"H", H}, {"C", C}, {"W_size_bytes", stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks_per_core_read", "num_read_per_barrier", "start_id", "curr_c", "curr_h", "curr_n"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Legacy writer carried the src0 CB index as CTA slot 0; that magic index is replaced by the
    // dfb::src0 binding. The `dst_buffer->aligned_page_size()` CTA was consumed only by the
    // TensorAccessorArgs plumbing and disappears with the TensorBinding.
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "writer_unary_transpose_hc_interleaved_start_id_rm.cpp"},
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
        .compile_time_args = {{"W_size_bytes", stick_size}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_sticks_per_core_read", "num_read_per_barrier", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // The src0 DFB's producer (reader) and consumer (writer) must share the same WorkUnitSpec —
    // every node hosting the DFB must host both endpoints. The legacy factory launches both kernels
    // on the full grid (total_cores); no-op cores receive num_sticks_per_core_read = 0.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_hc_rm",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = total_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    emit_runtime_args_hc_rm(
        reader_run,
        writer_run,
        input_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2);

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
