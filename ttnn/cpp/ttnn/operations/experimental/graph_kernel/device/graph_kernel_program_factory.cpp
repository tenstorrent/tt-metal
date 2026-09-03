// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_kernel_program_factory.hpp"

#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

// Basis program: a reader/writer pair copies inputs[0] into the output, page by page,
// with the pages split across the compute grid. Every input tensor is bound into the
// reader kernel (as tensor::in0, tensor::in1, ...) so the graph described by `text`
// can later be lowered onto them without changing the host-side wiring.
ttnn::device_operation::ProgramArtifacts GraphKernelProgramFactory::create_program_artifacts(
    const GraphKernelParams& operation_attributes, const GraphKernelInputs& tensor_args, Tensor& output) {
    const auto& inputs = tensor_args.inputs;
    const Tensor& src = inputs.front();
    auto* device = src.device();

    log_debug(tt::LogOp, "graph_kernel: {} input(s), text = \"{}\"", inputs.size(), operation_attributes.text);

    // ---- Page geometry (taken from inputs[0]; the output has an identical spec) ----
    const auto* src_buffer = src.buffer();
    const uint32_t page_size = static_cast<uint32_t>(src_buffer->page_size());
    const uint32_t num_pages = static_cast<uint32_t>(src_buffer->num_pages());
    const uint32_t aligned_page_size = tt::align(page_size, static_cast<uint32_t>(src_buffer->alignment()));
    const tt::DataFormat data_format = datatype_to_dataformat_converter(src.dtype());

    // ---- Work split ----
    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_g1, pages_per_core_g2] =
        split_work_to_cores(grid, num_pages);

    // ---- Resource names ----
    const DFBSpecName PAGES{"pages"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName OUTPUT{"out"};

    std::vector<TensorParamName> input_params;
    input_params.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        input_params.emplace_back("in" + std::to_string(i));
    }

    ProgramSpec spec;
    spec.name = "graph_kernel";

    // ---- Dataflow buffer: reader produces pages, writer consumes them ----
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = PAGES,
        .entry_size = aligned_page_size,
        .num_entries = 2,
        .data_format_metadata = data_format,
    });

    // ---- Tensor parameters: every input + the output ----
    for (size_t i = 0; i < inputs.size(); ++i) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = input_params[i], .spec = inputs[i].mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.mesh_tensor().tensor_spec()});

    // ---- Reader: binds all inputs, streams inputs[0] into PAGES ----
    Group<TensorBinding> reader_tensor_bindings;
    for (size_t i = 0; i < inputs.size(); ++i) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = input_params[i], .accessor_name = "in" + std::to_string(i)});
    }
    const Group<std::string> rta_names{"num_pages", "start_id"};

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/graph_kernel/device/kernels/reader_graph_kernel.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = PAGES, .accessor_name = "pages", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"page_size", page_size}, {"num_inputs", static_cast<uint32_t>(inputs.size())}},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer: drains PAGES into the output ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/graph_kernel/device/kernels/writer_graph_kernel.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = PAGES, .accessor_name = "pages", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
        .compile_time_args = {{"page_size", page_size}},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    // ---- Runtime args: contiguous page ranges per core ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    const uint32_t num_cores_g1 = core_group_1.num_cores();
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    uint32_t start_id = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t pages_this_core = i < num_cores_g1 ? pages_per_core_g1 : pages_per_core_g2;
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values, cores[i], {{"num_pages", pages_this_core}, {"start_id", start_id}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values, cores[i], {{"num_pages", pages_this_core}, {"start_id", start_id}});
        start_id += pages_this_core;
    }
    run_args.kernel_run_args.push_back(std::move(reader_ra));
    run_args.kernel_run_args.push_back(std::move(writer_ra));

    for (size_t i = 0; i < inputs.size(); ++i) {
        run_args.tensor_args.emplace(input_params[i], TensorArgument{inputs[i].mesh_tensor()});
    }
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
