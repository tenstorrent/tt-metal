// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::MeshTensor;

constexpr bool kRmOrientation = false;

// How the output pages are split over the cores. With the full compute grid and at least as many
// pages as cores, the core set is the whole grid for any page count, and only the counts change.
struct ConcatSplit {
    uint32_t num_output_pages = 0;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t g1_num_cores = 0;
    uint32_t pages_per_core_g1 = 0;
    uint32_t pages_per_core_g2 = 0;
};

ConcatSplit split_concat_work(const MeshTensor& output, const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    ConcatSplit split;
    split.num_output_pages =
        rm_layout ? output.physical_volume() / output.padded_shape()[-1] : output.physical_volume() / TILE_HW;
    if (sub_core_grids.has_value() && !output.is_sharded()) {
        // Use sub_core_grids for interleaved output
        uint32_t ncores = sub_core_grids->num_cores();
        TT_FATAL(ncores != 0, "number of cores cannot be 0");

        // Find the maximum number of cores that evenly divides num_output_pages
        for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
            if (split.num_output_pages % core_id == 0) {
                ncores = core_id;
                break;
            }
            ncores--;
        }
        TT_FATAL(
            (split.num_output_pages % ncores == 0),
            "{} num of pages are not split uniformly across {} num of cores",
            split.num_output_pages,
            ncores);

        split.cores = corerange_to_cores(sub_core_grids.value(), ncores, kRmOrientation);
        split.all_cores =
            num_cores_to_corerangeset_in_subcoregrids(split.cores[0], ncores, sub_core_grids.value(), kRmOrientation);
        if (ncores == 1) {
            split.all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(split.cores[0]));
        }
        split.g1_num_cores = ncores;
        split.pages_per_core_g1 = split.num_output_pages / ncores;
        split.pages_per_core_g2 = 0;
    } else {
        // Use full compute grid
        const CoreCoord grid = output.device().compute_with_storage_grid_size();
        auto [num_cores, all_cores, core_group_1, core_group_2, pages_g1, pages_g2] =
            split_work_to_cores(grid, split.num_output_pages, kRmOrientation);
        split.all_cores = all_cores;
        split.cores = grid_to_cores(num_cores, grid.x, grid.y, kRmOrientation);
        split.g1_num_cores = core_group_1.num_cores();
        split.pages_per_core_g1 = pages_g1;
        split.pages_per_core_g2 = pages_g2;
    }
    return split;
}

// Pages each input contributes to one output block (the dims from `dim` inward), and the output's.
struct ConcatBlocks {
    std::vector<uint32_t> num_pages_per_block;
    std::vector<uint32_t> page_size_per_tensor;
    uint32_t num_output_pages_per_block = 0;
};

ConcatBlocks concat_blocks(
    const std::vector<std::reference_wrapper<const MeshTensor>>& inputs, const MeshTensor& output, uint32_t dim) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    const uint32_t num_input_tensors = inputs.size();
    const uint32_t num_dims = output.padded_shape().rank();
    ConcatBlocks blocks;
    blocks.num_pages_per_block.resize(num_input_tensors);
    blocks.page_size_per_tensor.resize(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.padded_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && dim < num_dims - 1) {
            num_accum_pages /= output.padded_shape()[-1];
        }
    } else {
        if (dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            blocks.page_size_per_tensor[i] = inputs[i].get().mesh_buffer().page_size();
            if (dim == num_dims - 1) {
                blocks.num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = inputs[i].get().padded_shape()[dim];
                blocks.num_pages_per_block[i] = num_accum_pages * dim_pages;
                blocks.num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            blocks.num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            blocks.page_size_per_tensor[i] = inputs[i].get().mesh_buffer().page_size();
            uint32_t dim_pages = inputs[i].get().padded_shape()[dim] / scale_factor;
            blocks.num_pages_per_block[i] = num_accum_pages * dim_pages;
            blocks.num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }
    return blocks;
}

std::vector<std::reference_wrapper<const MeshTensor>> concat_inputs(const ConcatInputs& tensor_args) {
    std::vector<std::reference_wrapper<const MeshTensor>> inputs;
    inputs.reserve(tensor_args.input_tensors.size());
    for (const auto& input_tensor : tensor_args.input_tensors) {
        inputs.emplace_back(input_tensor.mesh_tensor());
    }
    return inputs;
}

std::vector<tt::tt_metal::TensorParamName> concat_input_names(uint32_t num_input_tensors) {
    // One TensorParameter per input. The count is an op attribute (up to 47 inputs), so these are
    // built rather than declared.
    std::vector<tt::tt_metal::TensorParamName> names;
    names.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        names.emplace_back(tt::tt_metal::TensorParamName{"input_" + std::to_string(i)});
    }
    return names;
}

// The per-core runtime args and the tensor bindings: everything in a cache entry that follows the
// page counts, so a hit at another page count can rewrite it.
tt::tt_metal::ProgramRunArgs concat_run_args(
    const ConcatSplit& split,
    const ConcatBlocks& blocks,
    const std::vector<std::reference_wrapper<const MeshTensor>>& inputs,
    const MeshTensor& output) {
    using namespace tt::tt_metal;
    using tt::tt_metal::experimental::AdvancedKernelRunArgs;
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    const uint32_t num_input_tensors = inputs.size();
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName OUTPUT{"output"};
    const auto INPUTS = concat_input_names(num_input_tensors);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);

    for (uint32_t i = 0, num_pages_written = 0; i < split.cores.size(); ++i) {
        const CoreCoord& core = split.cores[i];
        const uint32_t num_pages_per_core =
            (i < split.g1_num_cores) ? split.pages_per_core_g1 : split.pages_per_core_g2;
        const uint32_t block_id = num_pages_written / blocks.num_output_pages_per_block;
        uint32_t id_within_block = num_pages_written % blocks.num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            page_id_per_tensor[j] = block_id * blocks.num_pages_per_block[j];
            if (id_within_block == 0) {
                continue;
            }
            if (id_within_block >= blocks.num_pages_per_block[j]) {
                page_id_per_tensor[j] += blocks.num_pages_per_block[j];
                id_within_block -= blocks.num_pages_per_block[j];
                curr_tensor = j + 1;
            } else {
                page_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{rm_layout ? "num_pages" : "num_tiles", num_pages_per_core},
             {"start_tensor", curr_tensor},
             {"start_tensor_id", curr_tensor_id}});

        AdvancedKernelRunArgs::Varargs reader_varargs;
        reader_varargs.reserve(2 * num_input_tensors);
        reader_varargs.insert(
            reader_varargs.end(), blocks.num_pages_per_block.cbegin(), blocks.num_pages_per_block.cend());
        reader_varargs.insert(reader_varargs.end(), page_id_per_tensor.cbegin(), page_id_per_tensor.cend());
        reader_run_args.advanced_options.runtime_varargs.emplace(core, std::move(reader_varargs));

        if (rm_layout) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"stick_size", static_cast<uint32_t>(output.mesh_buffer().page_size())},
                 {"num_sticks", num_pages_per_core},
                 {"start_id", num_pages_written}});
        } else {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_pages", num_pages_per_core}, {"start_id", num_pages_written}});
        }
        num_pages_written += num_pages_per_core;
    }

    ProgramRunArgs run_args;
    // push_back rather than brace-init: an initializer_list would copy the per-node arg tables.
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        run_args.tensor_args.emplace(INPUTS[i], inputs[i].get());
    }
    run_args.tensor_args.emplace(OUTPUT, output);
    return run_args;
}

}  // namespace

bool concat_program_is_row_free(const ConcatParams& operation_attributes, const ConcatInputs& tensor_args) {
    using namespace tt::tt_metal;
    const auto& inputs = tensor_args.input_tensors;
    const auto& first = inputs.front();
    const uint32_t rank = first.padded_shape().rank();
    if (first.layout() != Layout::TILE || rank < 2 || operation_attributes.dim != rank - 2 ||
        operation_attributes.sub_core_grids.has_value() || operation_attributes.output_mem_config.is_sharded()) {
        return false;
    }
    uint64_t output_tiles = 0;
    for (const auto& input : inputs) {
        // Rows of whole tiles, so the inputs' tiles add up to the output's.
        if (input.is_sharded() || input.layout() != Layout::TILE ||
            input.logical_shape()[-2] != input.padded_shape()[-2]) {
            return false;
        }
        output_tiles += input.physical_volume() / tt::constants::TILE_HW;
    }
    // Every core of the grid gets pages, so the core set does not follow the page count.
    const CoreCoord grid = first.device()->compute_with_storage_grid_size();
    return output_tiles >= static_cast<uint64_t>(grid.x) * grid.y;
}

ttnn::device_operation::ProgramArtifacts ConcatProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using tt::tt_metal::experimental::AdvancedKernelRunArgs;
    using tt::tt_metal::experimental::KernelAdvancedOptions;

    const uint32_t dim = operation_attributes.dim;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    // Work against the Metalium device-tensor type throughout; the TTNN wrappers only carry us in.
    const auto& output = tensor_return_value.mesh_tensor();
    const uint32_t num_input_tensors = tensor_args.input_tensors.size();
    std::vector<std::reference_wrapper<const MeshTensor>> inputs;
    inputs.reserve(num_input_tensors);
    for (const auto& input_tensor : tensor_args.input_tensors) {
        inputs.emplace_back(input_tensor.mesh_tensor());
    }

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName SRC0_DFB{"src0"};
    const TensorParamName OUTPUT{"output"};
    // One TensorParameter per input. The count is an op attribute (up to 47 inputs), so these are
    // built rather than declared.
    std::vector<TensorParamName> INPUTS;
    INPUTS.reserve(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        INPUTS.emplace_back(TensorParamName{"input_" + std::to_string(i)});
    }

    const auto& device = output.device();

    const tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    uint32_t single_page_size;
    const uint32_t common_align_len = std::max(
        inputs[0].get().mesh_buffer().get_reference_buffer()->alignment(),
        output.mesh_buffer().get_reference_buffer()->alignment());
    if (rm_layout) {
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
        single_page_size = tt::tile_size(dfb_data_format);
    }

    const ConcatSplit split = split_concat_work(output, sub_core_grids);
    const CoreRangeSet& all_cores = split.all_cores;

    // Depth=2 is a prefetch optimization; fall back to depth=1 when it would overflow L1.
    const uint32_t l1_budget =
        (device.l1_size_per_core() / 2) - device.allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_capacity =
        device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(HalMemType::L1);
    TT_FATAL(
        single_page_size <= l1_capacity,
        "ttnn.concat: required DFB entry size ({} B) exceeds per-core L1 capacity ({} B); "
        "op cannot fit on this device.",
        single_page_size,
        l1_capacity);
    uint32_t num_input_pages = 2;
    if (num_input_pages * single_page_size > l1_budget) {
        num_input_pages = 1;
    }

    // The staging buffer the reader fills a page at a time and the writer drains: one locked FIFO
    // producer and one locked FIFO consumer, so a plain 1:1 DFB.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0_DFB,
        .entry_size = single_page_size,
        .num_entries = num_input_pages,
        .data_format_metadata = dfb_data_format,
    };

    const uint32_t num_dims = output.padded_shape().rank();
    const ConcatBlocks blocks = concat_blocks(inputs, output, dim);
    const std::vector<uint32_t>& page_size_per_tensor = blocks.page_size_per_tensor;

    // --- Reader kernel ---

    KernelSpec::CompilerOptions::Defines reader_defines;
    if (rm_layout && dim == num_dims - 1) {
        reader_defines.emplace("WIDTH_CONCAT", "1");
    }

    // One TensorBinding per input, plus a TensorBindingSequence over them. The reader picks which
    // input a page comes from with a value it advances at run time, so it needs the bindings
    // positionally as well as by name; the sequence is what makes that legal, and it carries its own
    // length, which is why no tensor-count argument is passed.
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

    KernelAdvancedOptions reader_advanced_options{
        // num_pages_per_block[N] followed by page_id_per_tensor[N] — two N-element blocks the reader
        // reads in a counted loop, N being an op attribute rather than a source literal.
        .num_runtime_varargs = 2 * num_input_tensors,
        .tensor_binding_sequences =
            {
                KernelAdvancedOptions::TensorBindingSequence{
                    .sequence_name = "inputs",
                    .members = input_accessor_names,
                },
            },
    };
    if (rm_layout) {
        // The RM reader selects a per-tensor page size with a value it advances at run time, so
        // these cannot be named compile-time arguments. The TILE reader needs none of them: it
        // takes its transfer size from the DFB's tile size.
        reader_advanced_options.compile_time_varargs = page_size_per_tensor;
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = rm_layout ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                              "reader_concat_stick_layout_interleaved_start_id.cpp"
                            : "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                              "reader_concat_interleaved_start_id.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0_DFB,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {
                // Slot 0's name follows each reader's own local: the tiled reader counts tiles, the
                // row-major one counts pages.
                .runtime_arg_names = {rm_layout ? "num_pages" : "num_tiles", "start_tensor", "start_tensor_id"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = std::move(reader_advanced_options),
    };

    // --- Writer kernel ---
    // Both writers are Metal 2.0 forks of shared donor kernels that live beside their originals in
    // other ops' directories. Their binding names and argument sets are the forks' interface, not
    // this op's choice.
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = rm_layout ? "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp"
                            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                              "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0_DFB,
                    .accessor_name = rm_layout ? "out0" : "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = rm_layout ? Group<std::string>{"stick_size", "num_sticks", "start_id"}
                                               : Group<std::string>{"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<KernelSpec> kernels;
    kernels.reserve(2);
    kernels.push_back(std::move(reader_spec));
    kernels.push_back(std::move(writer_spec));

    // A row-free program serves every row count (see concat_program_is_row_free), so its tensors
    // may differ from these in shape. Interleaved tiles keep their accessors at any shape.
    TensorSpecRelaxations relaxations;
    relaxations.dynamic_tensor_shape = concat_program_is_row_free(operation_attributes, tensor_args);

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

    ProgramSpec spec{
        .name = "concat",
        .kernels = std::move(kernels),
        .dataflow_buffers = {std::move(src0_dfb)},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    // --- Per-core runtime args ---
    ProgramRunArgs run_args = concat_run_args(split, blocks, inputs, output);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::ProgramRunArgs ConcatProgramFactory::override_runtime_arguments(
    const ConcatParams& operation_attributes,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // A hit can serve another row count when concat_program_is_row_free holds: the core set is
    // then the whole grid at every count, and the counts and start pages are runtime args.
    // Otherwise the hash holds the full shapes, and only the tensors change.
    const auto& output = tensor_return_value.mesh_tensor();
    const auto inputs = concat_inputs(tensor_args);
    if (!concat_program_is_row_free(operation_attributes, tensor_args)) {
        tt::tt_metal::ProgramRunArgs run_args;
        const auto names = concat_input_names(inputs.size());
        for (uint32_t i = 0; i < inputs.size(); ++i) {
            run_args.tensor_args.emplace(names[i], inputs[i].get());
        }
        run_args.tensor_args.emplace(tt::tt_metal::TensorParamName{"output"}, output);
        return run_args;
    }
    return concat_run_args(
        split_concat_work(output, operation_attributes.sub_core_grids),
        concat_blocks(inputs, output, operation_attributes.dim),
        inputs,
        output);
}

}  // namespace ttnn::prim
