// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/reshape_new/device/reshape_new_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts ReshapeNewPageCopyProgramFactory::create_program_artifacts(
    const ReshapeNewParams& operation_attributes, const ReshapeNewInputs& tensor_args, Tensor& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    const auto& input = tensor_args.input;
    const auto params = ttnn::operations::data_movement::reshape_new::make_page_copy_params(
        input.tensor_spec(), output.tensor_spec(), input.mesh_buffer().alignment(), output.mesh_buffer().alignment());
    const auto split = split_reshape_units(output.device(), operation_attributes.sub_core_grid, params.num_units);

    const DFBSpecName STAGE{"stage"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};

    ProgramSpec spec;
    spec.name = "reshape_new_page_copy";
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = STAGE,
        .entry_size = params.entry_bytes,
        .num_entries = params.num_entries,
    });
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = SRC, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = DST, .spec = output.tensor_spec()});

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"page_bytes", params.page_bytes},
        {"chunk_bytes", params.chunk_bytes},
        {"chunks_per_page", params.chunks_per_page},
        {"entry_bytes", params.entry_bytes},
        {"units_per_batch", params.units_per_batch},
    };
    const Group<std::string> runtime_arg_names = {"unit_begin", "unit_end"};

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_new/device/kernels/page_copy_reader.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGE, .accessor_name = "stage", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}},
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_new/device/kernels/page_copy_writer.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGE, .accessor_name = "stage", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });
    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = split.all_cores});

    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    for (size_t i = 0; i < split.cores.size(); ++i) {
        const uint32_t unit_begin = split.unit_begin[i];
        const uint32_t unit_end = split.unit_begin[i + 1];
        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, split.cores[i], {{"unit_begin", unit_begin}, {"unit_end", unit_end}});
        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, split.cores[i], {{"unit_begin", unit_begin}, {"unit_end", unit_end}});
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.tensor_args.emplace(SRC, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(DST, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
