// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "nlp_concat_heads_decode_program_factory.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeProgramFactory::create_program_artifacts(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;

    // Program-scope resource names (function-local: this op's two factories share a
    // translation unit under unity builds, so no anonymous-namespace constants).
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // cores to read and write to output
    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    // NoC coordinate tables for the input shard grid, x block then y block; these ride the
    // kernels' runtime varargs (the kernel indexes them with a data-driven cursor).
    std::vector<uint32_t> noc_coords;
    noc_coords.reserve(in_num_cores_x + in_num_cores_y);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of
    // a tile respectively)
    KernelSpec::CompileTimeArgs reader_compile_time_args{
        {"element_size", element_size},
        {"subtile_line_bytes", sub_tile_line_bytes},
        {"head_size", head_size},
        {"batch", batch},
        {"head_size_num_tiles", head_tiles},
        {"phases_to_read", 1},  // read the first phase
        {"num_x", in_num_cores_x},
        {"num_y", in_num_cores_y},
    };

    KernelSpec::CompileTimeArgs writer_compile_time_args = reader_compile_time_args;
    writer_compile_time_args["phases_to_read"] = 2;  // read the second phase

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_decode.cpp",
        // Both instances of the kernel only raw-write the output shard in place (no FIFO
        // traffic), so the output is bound as a plain tensor (LocalTensorAccessor kernel-side),
        // not as a DFB.
        .tensor_bindings =
            {TensorBinding{
                 .tensor_parameter_name = INPUT,
                 .accessor_name = "input",
             },
             TensorBinding{
                 .tensor_parameter_name = OUTPUT,
                 .accessor_name = "q_out",
             }},
        .compile_time_args = std::move(reader_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = in_num_cores_x + in_num_cores_y},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_decode.cpp",
        .tensor_bindings =
            {TensorBinding{
                 .tensor_parameter_name = INPUT,
                 .accessor_name = "input",
             },
             TensorBinding{
                 .tensor_parameter_name = OUTPUT,
                 .accessor_name = "q_out",
             }},
        .compile_time_args = std::move(writer_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = in_num_cores_x + in_num_cores_y},
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0; i < num_cores; ++i) {
        // Each output core i corresponds to head index i. Within the input shard, that head lives in
        // head-tile (i / 32) at row (i % 32). The two cases below pick the row's byte offset within
        // a single 32x32 tile (face 0 for rows < 16, face 2 for rows >= 16); add the head-tile skip
        // to land in the right tile when padded_heads > 32.
        uint32_t head_tile_idx = i / 32;
        uint32_t head_in_tile = i % 32;
        uint32_t in_tile_offset_by_batch =
            (head_in_tile < 16 ? head_in_tile * sub_tile_line_bytes
                               : (head_in_tile - 16) * sub_tile_line_bytes + 512 * element_size) +
            head_tile_idx * head_size;

        const auto& core = cores[i];
        // Reader and writer instances receive identical per-core values; only the phase CTA differs.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}});
        reader_run_args.advanced_options.runtime_varargs[core] = noc_coords;
        writer_run_args.advanced_options.runtime_varargs[core] = noc_coords;
    }

    ProgramSpec spec{
        .name = "nlp_concat_heads_decode",
        .kernels = {std::move(reader), std::move(writer)},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
             // OUTPUT is written in place through the kernels' "q_out" LocalTensorAccessor binding.
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = q_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
