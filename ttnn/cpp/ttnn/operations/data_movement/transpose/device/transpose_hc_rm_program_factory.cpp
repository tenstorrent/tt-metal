// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

const DFBSpecName HCRM_SRC0_DFB{"hcrm_src0"};
const TensorParamName HCRM_INPUT{"hcrm_input"};
const TensorParamName HCRM_OUTPUT{"hcrm_output"};
const KernelSpecName HCRM_READER{"hcrm_reader"};
const KernelSpecName HCRM_WRITER{"hcrm_writer"};

// Per-core runtime args (reader+writer) for HC RM transpose. The traversal logic that
// advances (curr_c, curr_h, curr_n) matches the legacy emitter; only the sink changed
// (name-first ProgramRunArgs tables instead of node-first KernelDescriptor rows).
void emit_runtime_args_hc_rm(
    KernelRunArgs& reader_run,
    KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    uint32_t num_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t curr_sticks_read = 0, curr_sticks_write = 0;
    for (const auto& core : cores) {
        uint32_t num_sticks_per_core;

        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_sticks_per_core_read", num_sticks_per_core_read},
             {"num_read_per_barrier", num_read_per_barrier},
             {"start_id", curr_sticks_read},
             {"curr_c", curr_c},
             {"curr_h", curr_h},
             {"curr_n", curr_n}});

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_sticks_per_core_read", num_sticks_per_core_read},
             {"num_read_per_barrier", num_read_per_barrier},
             {"start_id", curr_sticks_write}});

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

ttnn::device_operation::ProgramArtifacts TransposeHCRMProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, N * C * H);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto num_sticks = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                : num_sticks_per_core_group_2;

    Buffer* src0_buffer = input_tensor.buffer();
    uint32_t aligned_page = std::max(src0_buffer->aligned_page_size(), dst_buffer->aligned_page_size());
    auto stick_size = std::max(W * input_tensor.element_size(), aligned_page);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "transpose_hc_rm";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = HCRM_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = HCRM_OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = HCRM_SRC0_DFB,
        .entry_size = stick_size,
        .num_entries = num_sticks,
        .data_format_metadata = cb_data_format,
    });

    KernelSpec reader{
        .unique_id = HCRM_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = HCRM_SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = HCRM_INPUT, .accessor_name = "src"}},
        .compile_time_args = {{"N", N}, {"H", H}, {"C", C}, {"W_size_bytes", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core_read", "num_read_per_barrier", "start_id", "curr_c", "curr_h", "curr_n"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = HCRM_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_hc_interleaved_start_id_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = HCRM_SRC0_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = HCRM_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"W_size_bytes", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core_read", "num_read_per_barrier", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels = {reader, writer};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {HCRM_READER, HCRM_WRITER}, .target_nodes = all_cores}};

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = HCRM_READER};
    KernelRunArgs writer_run{.kernel = HCRM_WRITER};

    emit_runtime_args_hc_rm(
        reader_run,
        writer_run,
        input_tensor,
        all_cores,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2);

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.emplace(HCRM_INPUT, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(HCRM_OUTPUT, TensorArgument{output_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
