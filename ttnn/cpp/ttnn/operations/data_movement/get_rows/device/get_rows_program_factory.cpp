// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "get_rows_program_factory.hpp"

#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

ttnn::device_operation::ProgramArtifacts GetRowsProgramFactory::create_program_artifacts(
    const GetRowsParams& operation_attributes, const GetRowsInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    auto& output = tensor_return_value;

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& indices_mesh_tensor = indices.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    MeshDevice* device = input.device();

    // Row-major pages: one row of the table per page, and the whole [N] index vector as one page.
    const uint32_t element_size = input.element_size();
    const uint32_t row_size = input.padded_shape()[-1] * element_size;
    const uint32_t output_page_size = output.padded_shape()[-1] * output.element_size();
    const uint32_t indices_element_size = indices.element_size();

    // The reader stages every row in an allocator-aligned entry; the writer writes only the raw row
    // bytes. One staged entry is covered by the reader while the writer drains the other, so the
    // budget is two entries. A chunked path for wider rows is future work (see #15942).
    const auto alignment = input.mesh_buffer().alignment();
    const uint32_t rounded_row_size = tt::align(row_size, alignment);
    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for the output staging
    TT_FATAL(
        rounded_row_size * 2 <= max_l1_budget_bytes,
        "get_rows: row size {} B (rounded to {} B) exceeds the interleaved staging budget; rows wider than {} B "
        "need the chunked path, which is not implemented yet (see #15942)",
        row_size,
        rounded_row_size,
        max_l1_budget_bytes / 2);

    // Index blocks: the reader pulls the index vector in alignment-sized blocks (one NoC read per
    // block) and works through the block's indices from a private scratchpad.
    const auto indices_alignment = indices.mesh_buffer().alignment();
    const uint32_t indices_page_size = indices.padded_shape()[-1] * indices_element_size;
    const uint32_t block_bytes = tt::align(indices_element_size, indices_alignment);
    const uint32_t indices_per_block = block_bytes / indices_element_size;
    const uint32_t num_rows_total = indices.padded_shape()[-1];
    TT_FATAL(indices_page_size == num_rows_total * indices_element_size, "The index vector must be one row-major page");
    const uint32_t num_index_blocks = (num_rows_total + indices_per_block - 1) / indices_per_block;
    const uint32_t last_block_bytes = indices_page_size - (num_index_blocks - 1) * block_bytes;

    // One output row per index; cores take contiguous slices of the index order.
    const uint32_t problem_size = num_rows_total;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& [_, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    uint32_t g1_numcores = core_group_1.num_cores();

    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const DFBSpecName OUTPUT{"output"};
    const ScratchpadSpecName INDEX_SCRATCH_SP{"index_scratch_sp"};

    const TensorParamName INDICES_PARAM{"indices"};
    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName OUTPUT_PARAM{"output"};

    ProgramSpec spec;
    spec.name = "get_rows";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT,
        .entry_size = rounded_row_size,
        .num_entries = 2,  // double buffered
        .data_format_metadata = data_format,
    });

    spec.scratchpads.push_back(ScratchpadSpec{
        .unique_id = INDEX_SCRATCH_SP,
        .size_per_node = block_bytes,
    });

    // -----------------------------------------------------------------------
    // Tensor parameters
    // -----------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INDICES_PARAM, .spec = indices_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_PARAM, .spec = output_mesh_tensor.tensor_spec()});

    // -----------------------------------------------------------------------
    // Reader: decodes its slice of the index vector and stages the selected rows
    // -----------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/get_rows/device/kernels/dataflow/get_rows_reader.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                KernelSpec::ScratchpadBinding{.scratchpad_spec_name = INDEX_SCRATCH_SP, .accessor_name = "idx"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INDICES_PARAM, .accessor_name = "indices"},
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
            },
        .compile_time_args =
            {
                {"block_bytes", block_bytes},
                {"indices_per_block", indices_per_block},
                {"row_bytes", rounded_row_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_row", "num_rows", "num_index_blocks", "last_block_bytes"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // -----------------------------------------------------------------------
    // Writer: drains the staged rows into the interleaved output, preserving order
    // -----------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"stick_size", "num_sticks", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    uint32_t row_offset = 0;

    auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_major=*/false);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t local_num_rows = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"start_row", row_offset},
             {"num_rows", local_num_rows},
             {"num_index_blocks", num_index_blocks},
             {"last_block_bytes", last_block_bytes}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"stick_size", output_page_size}, {"num_sticks", local_num_rows}, {"start_id", row_offset}});

        row_offset += local_num_rows;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INDICES_PARAM, indices_mesh_tensor);
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT_PARAM, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
