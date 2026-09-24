// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Compute per-core runtime args (reader+writer) for HC RM transpose and add them to the
// supplied KernelRunArgs. The traversal logic that advances (curr_c, curr_h, curr_n) was
// previously shared between `create` and `override_runtime_arguments`; now it has a single home.
void emit_runtime_args_hc_rm(
    ProgramRunArgs::KernelRunArgs& reader_run_args,
    ProgramRunArgs::KernelRunArgs& writer_run_args,
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
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core_read", num_sticks_per_core_read},
             {"num_read_per_barrier", num_read_per_barrier},
             {"start_id", curr_sticks_read},
             {"curr_c", curr_c},
             {"curr_h", curr_h},
             {"curr_n", curr_n}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
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
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t NCH = N * C * H;

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "dfb_data_format: {}", dfb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, NCH);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto num_sticks = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                : num_sticks_per_core_group_2;

    Buffer* src0_buffer = input_tensor.buffer();
    uint32_t aligned_page = std::max(src0_buffer->aligned_page_size(), dst_buffer->aligned_page_size());
    auto stick_size = std::max(W * input_tensor.element_size(), aligned_page);

    ProgramSpec spec{.name = "transpose_hc_rm"};

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = stick_size,
        .num_entries = num_sticks,
        .data_format_metadata = dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    // The legacy factory also emitted `N` and an `aligned_page_size` on each kernel. Neither kernel
    // ever read them (each one's TensorAccessorArgs boundary sat past those slots), so they carried
    // no behavior and are not re-emitted here.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args = {{"H", H}, {"C", C}, {"W_size_bytes", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core_read", "num_read_per_barrier", "start_id", "curr_c", "curr_h", "curr_n"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "writer_unary_transpose_hc_interleaved_start_id_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "out0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"W_size_bytes", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core_read", "num_read_per_barrier", "start_id"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};

    emit_runtime_args_hc_rm(
        reader_run_args,
        writer_run_args,
        input_tensor,
        all_cores,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2);

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
