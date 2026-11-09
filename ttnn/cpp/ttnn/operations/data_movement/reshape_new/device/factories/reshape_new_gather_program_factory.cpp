// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/reshape_new/device/reshape_new_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts ReshapeNewGatherProgramFactory::create_program_artifacts(
    const ReshapeNewParams& operation_attributes, const ReshapeNewInputs& tensor_args, Tensor& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    namespace plan = ttnn::operations::data_movement::reshape_new;
    const auto& input = tensor_args.input;
    const auto& sub_core_grid = operation_attributes.sub_core_grid;
    const auto grid_size = output.device()->compute_with_storage_grid_size();
    const uint32_t max_cores =
        sub_core_grid.has_value() ? static_cast<uint32_t>(sub_core_grid->num_cores()) : grid_size.x * grid_size.y;
    const auto params =
        plan::make_gather_params(input.tensor_spec(), output.tensor_spec(), input.mesh_buffer().alignment(), max_cores);
    const auto split = split_reshape_units(output.device(), sub_core_grid, params.num_units);

    const bool fill_padding = operation_attributes.fill_padding;
    // Two units being assembled (one is written out while the next one fills), plus the padding pattern.
    const uint32_t scratch_bytes = (fill_padding ? 3 : 2) * params.unit_slot_bytes;

    const DFBSpecName STAGE0{"stage0"};
    const DFBSpecName STAGE1{"stage1"};
    const ScratchpadSpecName ASSEMBLY{"assembly"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};

    ProgramSpec spec;
    spec.name = "reshape_new_gather";
    for (const auto& stage : {STAGE0, STAGE1}) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = stage,
            .entry_size = params.stage_entry_bytes,
            .num_entries = params.num_stage_entries,
        });
    }
    spec.scratchpads.push_back(ScratchpadSpec{.unique_id = ASSEMBLY, .size_per_node = scratch_bytes});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = SRC, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = DST, .spec = output.tensor_spec()});

    const Group<std::string> runtime_arg_names = {"unit_begin", "unit_end"};
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_new/device/kernels/gather_reader.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = STAGE0, .accessor_name = "stage0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = STAGE1, .accessor_name = "stage1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}},
        .compile_time_args =
            {
                {"tiled", static_cast<uint32_t>(params.tiled)},
                {"element_size", params.element_size},
                {"page_bytes", params.page_bytes},
                {"unit_bytes", params.unit_bytes},
                {"chunks_per_page", params.chunks_per_page},
                {"pages_per_unit", params.pages_per_unit},
                {"page_slot_bytes", params.page_slot_bytes},
                {"in_width", params.in.width},
                {"in_height", params.in.height},
                {"in_padded_height", params.in.padded_height},
                {"in_page_width", params.in.page_width},
                {"in_pages_per_row", params.in.pages_per_row},
                {"out_width", params.out.width},
                {"out_height", params.out.height},
                {"out_padded_height", params.out.padded_height},
                {"out_page_width", params.out.page_width},
                {"out_pages_per_row", params.out.pages_per_row},
                {"out_num_pages", params.out.num_pages},
                {"tile_height", params.tile_height},
                {"tile_width", params.tile_width},
                {"face_height", params.face_height},
                {"face_width", params.face_width},
                {"source_alignment", params.source_alignment},
                {"records_per_entry", params.records_per_entry},
                {"header_bytes", params.header_bytes},
                {"stage_entry_bytes", params.stage_entry_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_new/device/kernels/gather_writer.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = STAGE0, .accessor_name = "stage0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = STAGE1, .accessor_name = "stage1", .endpoint_type = DFBEndpointType::CONSUMER}},
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = ASSEMBLY, .accessor_name = "assembly"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .compile_time_args =
            {
                {"unit_slot_bytes", params.unit_slot_bytes},
                {"page_slot_bytes", params.page_slot_bytes},
                {"fill_padding", static_cast<uint32_t>(fill_padding)},
                {"pad_bits", operation_attributes.pad_bits},
            },
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
