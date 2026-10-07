// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_program_factory.hpp"

#include "scatter_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts ScatterProgramFactory::create_program_artifacts(
    const ScatterParams& args, const ScatterInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& src_shape{src_tensor.logical_shape()};
    const auto& output_shape{output_tensor.logical_shape()};

    const uint32_t input_stick_size = input_shape[-1];
    const uint32_t index_stick_size = index_shape[-1];
    const uint32_t source_stick_size = src_shape[-1];
    const uint32_t output_stick_size = output_shape[-1];

    // input dtype byte sizes
    const uint32_t input_datum_size = input_tensor.element_size();
    const uint32_t index_datum_size = index_tensor.element_size();
    const uint32_t source_datum_size = src_tensor.element_size();
    const uint32_t output_datum_size = output_tensor.element_size();

    // output row byte size (the writer walks each output stick in chunks of this many bytes)
    const uint32_t output_stick_size_bytes = output_stick_size * output_datum_size;

    // maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
    // BH available L1 mem size of nearly 1.5 MB...
    // ... minimized by the amount of memory reserved by a model...
    // ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
    // tensors)
    // ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
    // ... minimized by ~10% to account for reserved memory
    const uint32_t input_and_output_max_chunk_size = calculate_optimal_chunk_size(input_tensor);
    const uint32_t index_and_source_max_chunk_size = calculate_optimal_chunk_size(index_tensor);
    const uint32_t input_and_output_chunk_size = std::min(input_stick_size, input_and_output_max_chunk_size);
    const uint32_t index_chunk_size = std::min(index_stick_size, index_and_source_max_chunk_size);
    const uint32_t source_chunk_size = std::min(source_stick_size, index_and_source_max_chunk_size);
    const uint32_t input_and_output_chunk_size_bytes = input_and_output_chunk_size * input_datum_size;
    const uint32_t index_chunk_size_bytes = index_chunk_size * index_datum_size;
    const uint32_t source_chunk_size_bytes = source_chunk_size * source_datum_size;

    // pad pages to 32
    const uint32_t input_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);
    const uint32_t index_page_size_bytes = ceil32(index_chunk_size_bytes);
    const uint32_t source_page_size_bytes = ceil32(source_chunk_size_bytes);
    const uint32_t output_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);

    constexpr const char* reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/reader_scatter.cpp";
    constexpr const char* writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/writer_scatter.cpp";

    auto* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t work_units = input_tensor.logical_volume() / input_stick_size;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            args.sub_core_grid.has_value()
                ? tt::tt_metal::split_work_to_cores(*args.sub_core_grid, work_units)
                : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units);

    const auto farthest_x_y =
        args.sub_core_grid.has_value() ? args.sub_core_grid->bounding_box().end_coord : compute_with_storage_grid_size;
    const uint32_t all_cores_in_bounding_box = (farthest_x_y.x + 1) * (farthest_x_y.y + 1);

    // Metal 2.0 resource names. Declared local (not at namespace scope) so the sibling factory in
    // the same unity-build translation unit can reuse the same identifiers without collision.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName INDEX_DFB{"index"};
    const DFBSpecName SRC_DFB{"source"};
    const DFBSpecName DST_DFB{"output"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName INDEX_TENSOR{"index"};
    const TensorParamName SRC_TENSOR{"source"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // Each scatter DFB holds exactly one chunk page (num_entries = 1; entry_size is the 32-aligned
    // chunk byte size). The data format is set even though these are data-movement-only DFBs,
    // because the kernels select their C++ element type at compile time via get_dataformat(dfb::name).
    auto make_dfb = [](const DFBSpecName& name, DataType dtype, uint32_t page_size_bytes) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = page_size_bytes,
            .num_entries = 1,
            .data_format_metadata = datatype_to_dataformat_converter(dtype),
        };
    };

    Group<DataflowBufferSpec> dataflow_buffers{
        make_dfb(INPUT_DFB, input_tensor.dtype(), input_page_size_bytes),
        make_dfb(INDEX_DFB, index_tensor.dtype(), index_page_size_bytes),
        make_dfb(SRC_DFB, src_tensor.dtype(), source_page_size_bytes),
        make_dfb(DST_DFB, output_tensor.dtype(), output_page_size_bytes),
    };

    // The reader alone fills and drains INPUT/INDEX/SRC (self-loop: bound PRODUCER + CONSUMER); it
    // produces DST, which the writer consumes.
    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SRC_DFB, .accessor_name = "source", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = SRC_DFB, .accessor_name = "source", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = DST_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = INDEX_TENSOR, .accessor_name = "index"},
                TensorBinding{.tensor_parameter_name = SRC_TENSOR, .accessor_name = "source"},
            },
        .compile_time_args =
            {
                {"input_stick_size", input_stick_size},
                {"index_stick_size", index_stick_size},
                {"source_stick_size", source_stick_size},
                {"input_rank", static_cast<uint32_t>(input_shape.rank())},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"start_stick_id",
                     "sticks_for_core",
                     "input_and_output_chunk_size",
                     "index_chunk_size",
                     "source_chunk_size",
                     "scatter_reduction_type"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
        // Per-dimension shape extents (input dims then index dims). N = rank-1 per tensor; the
        // count varies with rank across instantiations, so these are delivered as runtime varargs.
        .advanced_options =
            {.num_runtime_varargs = static_cast<uint32_t>((input_shape.rank() - 1) + (index_shape.rank() - 1))},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DST_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"output_stick_size_bytes", output_stick_size_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_stick_id", "sticks_for_core", "input_and_output_chunk_size"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Per-dimension shape extents are the same on every node; build once and bind per node.
    std::vector<uint32_t> shape_varargs;
    shape_varargs.reserve((input_shape.rank() - 1) + (index_shape.rank() - 1));
    for (const auto* it = input_shape.cbegin(); it != input_shape.cend() - 1; ++it) {
        shape_varargs.push_back(static_cast<uint32_t>(*it));
    }
    for (const auto* it = index_shape.cbegin(); it != index_shape.cend() - 1; ++it) {
        shape_varargs.push_back(static_cast<uint32_t>(*it));
    }

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    AdvancedKernelRunArgs reader_run_advanced;

    uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < all_cores_in_bounding_box; ++i) {
        const CoreCoord core{i / (farthest_x_y.y + 1), i % (farthest_x_y.y + 1)};
        uint32_t sticks_per_core;
        if (core_group_1.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_2;
        } else {
            continue;
        }

        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {{"start_stick_id", stick_offset},
             {"sticks_for_core", sticks_per_core},
             {"input_and_output_chunk_size", input_and_output_chunk_size},
             {"index_chunk_size", index_chunk_size},
             {"source_chunk_size", source_chunk_size},
             {"scatter_reduction_type", static_cast<uint32_t>(args.opt_reduction)}});
        reader_run_advanced.runtime_varargs.emplace(core, shape_varargs);

        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {{"start_stick_id", stick_offset},
             {"sticks_for_core", sticks_per_core},
             {"input_and_output_chunk_size", input_and_output_chunk_size}});

        stick_offset += sticks_per_core;
    }

    ProgramSpec spec;
    spec.name = "scatter";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = INDEX_TENSOR, .spec = index_tensor.tensor_spec()},
        TensorParameter{.unique_id = SRC_TENSOR, .spec = src_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "scatter",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    }};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = std::move(reader_node_args),
            .advanced_options = std::move(reader_run_advanced),
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = std::move(writer_node_args),
        },
    };
    run_args.tensor_args.emplace(INPUT_TENSOR, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(INDEX_TENSOR, index_tensor.mesh_tensor());
    run_args.tensor_args.emplace(SRC_TENSOR, src_tensor.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT_TENSOR, output_tensor.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
