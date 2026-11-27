// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_tiled_row_shift_program_factory.hpp"

#include <functional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::MeshTensor;

// Inputs beyond this would need more varargs than a concat of a few pieces calls for.
constexpr uint32_t kMaxInputs = 16;

std::vector<tt::tt_metal::TensorParamName> row_shift_input_names(uint32_t num_input_tensors) {
    std::vector<tt::tt_metal::TensorParamName> names;
    names.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        names.emplace_back(tt::tt_metal::TensorParamName{"input_" + std::to_string(i)});
    }
    return names;
}

std::vector<CoreCoord> full_grid_cores(const MeshTensor& output) {
    const CoreCoord grid = output.device().compute_with_storage_grid_size();
    std::vector<CoreCoord> cores;
    cores.reserve(grid.x * grid.y);
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            cores.emplace_back(x, y);
        }
    }
    return cores;
}

// Everything in a cache entry that follows the row counts: each core's range of output tiles, the
// output's tile rows per block, each input's logical rows and tile rows, and the tensor bindings.
tt::tt_metal::ProgramRunArgs row_shift_run_args(
    const std::vector<std::reference_wrapper<const MeshTensor>>& inputs, const MeshTensor& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName OUTPUT{"output"};
    const auto INPUTS = row_shift_input_names(inputs.size());

    const uint32_t num_output_tiles = output.physical_volume() / TILE_HW;
    const uint32_t out_tile_rows = output.padded_shape()[-2] / TILE_HEIGHT;
    experimental::AdvancedKernelRunArgs::Varargs row_varargs;
    row_varargs.reserve(2 * inputs.size());
    for (const auto& input : inputs) {
        row_varargs.push_back(input.get().logical_shape()[-2]);
    }
    for (const auto& input : inputs) {
        row_varargs.push_back(input.get().padded_shape()[-2] / TILE_HEIGHT);
    }

    // Every core of the grid, with an empty range where there are fewer tiles than cores, so that the
    // core set does not follow the tile count.
    const auto cores = full_grid_cores(output);
    const uint32_t base = num_output_tiles / cores.size();
    const uint32_t extra = num_output_tiles % cores.size();
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0, start = 0; i < cores.size(); ++i) {
        const uint32_t count = base + (i < extra ? 1 : 0);
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            cores[i],
            {{"start_tile", start}, {"num_tiles", count}, {"out_tile_rows", out_tile_rows}});
        reader_run_args.advanced_options.runtime_varargs.emplace(cores[i], row_varargs);
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, cores[i], {{"num_pages", count}, {"start_id", start}});
        start += count;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        run_args.tensor_args.emplace(INPUTS[i], inputs[i].get());
    }
    run_args.tensor_args.emplace(OUTPUT, output);
    return run_args;
}

std::vector<std::reference_wrapper<const MeshTensor>> row_shift_inputs(const ConcatInputs& tensor_args) {
    std::vector<std::reference_wrapper<const MeshTensor>> inputs;
    inputs.reserve(tensor_args.input_tensors.size());
    for (const auto& input_tensor : tensor_args.input_tensors) {
        inputs.emplace_back(input_tensor.mesh_tensor());
    }
    return inputs;
}

}  // namespace

bool can_use_tiled_row_shift_concat(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    using namespace tt::tt_metal;
    if (input_tensors.size() < 2 || input_tensors.size() > kMaxInputs || groups != 1 || sub_core_grids.has_value() ||
        output_mem_config.is_sharded()) {
        return false;
    }
    const auto& first = input_tensors.front();
    const uint32_t rank = first.logical_shape().rank();
    if (rank < 2 || normalized_dim != rank - 2) {
        return false;
    }
    const DataType dtype = first.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32) {
        return false;
    }
    for (const auto& input : input_tensors) {
        if (!is_device_tensor(input) || input.layout() != Layout::TILE || input.is_sharded() ||
            input.dtype() != dtype || input.tensor_spec().tile().get_height() != tt::constants::TILE_HEIGHT ||
            input.tensor_spec().tile().get_width() != tt::constants::TILE_WIDTH ||
            input.padded_shape()[-1] != first.padded_shape()[-1]) {
            return false;
        }
    }
    return true;
}

ttnn::device_operation::ProgramArtifacts ConcatTiledRowShiftProgramFactory::create_program_artifacts(
    const ConcatParams& /*operation_attributes*/, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& output = tensor_return_value.mesh_tensor();
    const auto inputs = row_shift_inputs(tensor_args);
    const uint32_t num_input_tensors = inputs.size();

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName OUT_DFB{"out"};
    const ScratchpadSpecName SRC_SCRATCH{"src"};
    const TensorParamName OUTPUT{"output"};
    const auto INPUTS = row_shift_input_names(num_input_tensors);

    const tt::DataFormat data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t tile_bytes = tt::tile_size(data_format);
    const uint32_t face_row_bytes = (TILE_WIDTH / 2) * output.element_size();

    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = tile_bytes,
        .num_entries = 2,
        .data_format_metadata = data_format,
    };
    // Two source tiles the reader copies rows out of.
    ScratchpadSpec src_scratch{
        .unique_id = SRC_SCRATCH,
        .size_per_node = 2 * tile_bytes,
    };

    Group<TensorBinding> reader_tensor_bindings;
    std::vector<std::string> input_accessor_names;
    reader_tensor_bindings.reserve(num_input_tensors);
    input_accessor_names.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        input_accessor_names.push_back("in" + std::to_string(i));
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUTS[i],
            .accessor_name = input_accessor_names.back(),
        });
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_tiled_row_shift.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = SRC_SCRATCH, .accessor_name = "src"},
            },
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args =
            {
                {"width_tiles", output.padded_shape()[-1] / TILE_WIDTH},
                {"face_row_bytes", face_row_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_tile", "num_tiles", "out_tile_rows"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options =
            {
                // in_rows[N], then in_tile_rows[N].
                .num_runtime_varargs = 2 * num_input_tensors,
                .tensor_binding_sequences =
                    {
                        experimental::KernelAdvancedOptions::TensorBindingSequence{
                            .sequence_name = "inputs",
                            .members = input_accessor_names,
                        },
                    },
            },
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<KernelSpec> kernels;
    kernels.reserve(2);
    kernels.push_back(std::move(reader_spec));
    kernels.push_back(std::move(writer_spec));

    // The program serves every row count, so its tensors may differ from these in shape; interleaved tiles
    // keep their accessors at any shape.
    TensorSpecRelaxations relaxations;
    relaxations.dynamic_tensor_shape = true;
    Group<TensorParameter> tensor_parameters;
    tensor_parameters.reserve(num_input_tensors + 1);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        tensor_parameters.push_back(TensorParameter{
            .unique_id = INPUTS[i],
            .spec = inputs[i].get().tensor_spec(),
            .relaxations = relaxations,
        });
    }
    tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = output.tensor_spec(),
        .relaxations = relaxations,
    });

    const CoreCoord grid = output.device().compute_with_storage_grid_size();
    ProgramSpec spec{
        .name = "concat_tiled_row_shift",
        .kernels = std::move(kernels),
        .dataflow_buffers = {std::move(out_dfb)},
        .scratchpads = {std::move(src_scratch)},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
                },
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = row_shift_run_args(inputs, output),
    };
}

tt::tt_metal::ProgramRunArgs ConcatTiledRowShiftProgramFactory::override_runtime_arguments(
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return row_shift_run_args(row_shift_inputs(tensor_args), tensor_return_value.mesh_tensor());
}

}  // namespace ttnn::prim
