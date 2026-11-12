// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [Quasar bring-up: concat interleaved metal2]
//
// Metal-2.0 twin of concat_program_factory.cpp. Every line of shape/work-split arithmetic below is
// carried over from that file unchanged and deliberately so: the only intended behavioural delta is
// the program-construction API. If the two drift, the legacy file is the reference.

#include "ttnn/operations/data_movement/concat/device/concat_metal2_program_factory.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// One DFB, one reader, one writer: the reader fills a page, the writer drains it to the output
// tensor. Nothing else touches it, so this is an ordinary 1-producer / 1-consumer DataflowBuffer
// and NOT a ScratchpadSpec -- see the note at the DataflowBufferSpec below.
const DFBSpecName CONCAT_DFB{"stage"};
const TensorParamName CONCAT_OUTPUT_TENSOR{"output"};
const KernelSpecName CONCAT_READER{"reader"};
const KernelSpecName CONCAT_WRITER{"writer"};

std::string input_param_name(uint32_t i) { return "input" + std::to_string(i); }
std::string input_accessor_name(uint32_t i) { return "src" + std::to_string(i); }

}  // namespace

ttnn::device_operation::ProgramArtifacts ConcatMetalV2ProgramFactory::create_program_artifacts(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = output.device();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    constexpr bool rm_orientation = false;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    const uint32_t common_align_len = std::max(input_tensors[0].buffer()->alignment(), output.buffer()->alignment());
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
        num_output_pages = output.physical_volume() / TILE_HW;
        single_page_size = tt::tile_size(cb_data_format);
    }

    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores;
    uint32_t num_tiles_per_core_group_1;
    uint32_t num_tiles_per_core_group_2;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
    std::vector<CoreCoord> cores_list;

    if (sub_core_grids.has_value() && !output.is_sharded()) {
        uint32_t ncores = sub_core_grids->num_cores();
        TT_FATAL(ncores != 0, "number of cores cannot be 0");

        for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
            if (num_output_pages % core_id == 0) {
                ncores = core_id;
                break;
            }
            ncores--;
        }
        TT_FATAL(
            (num_output_pages % ncores == 0),
            "{} num of pages are not split uniformly across {} num of cores",
            num_output_pages,
            ncores);

        cores_list = corerange_to_cores(sub_core_grids.value(), ncores, rm_orientation);
        all_cores =
            num_cores_to_corerangeset_in_subcoregrids(cores_list[0], ncores, sub_core_grids.value(), rm_orientation);
        if (ncores == 1) {
            all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores_list[0]));
        }
        num_cores = ncores;
        num_tiles_per_core_group_1 = num_output_pages / ncores;
        num_tiles_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    } else {
        const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_tiles_per_core_group_1_result,
             num_tiles_per_core_group_2_result] =
                split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_tiles_per_core_group_1 = num_tiles_per_core_group_1_result;
        num_tiles_per_core_group_2 = num_tiles_per_core_group_2_result;
    }

    const uint32_t num_input_tensors = input_tensors.size();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Depth=2 is a prefetch optimization; fall back to depth=1 when it would overflow L1.
    const uint32_t l1_budget =
        (device->l1_size_per_core() / 2) - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_capacity =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
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

    const uint32_t num_dims = output.padded_shape().rank();

    std::vector<uint32_t> num_pages_per_block(num_input_tensors);
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);
    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

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

    uint32_t num_output_pages_per_block = 0;
    const bool width_concat = rm_layout && dim == num_dims - 1;

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            page_size_per_tensor[i] = buffer->page_size();
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = input_tensors[i].padded_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages;
                num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            page_size_per_tensor[i] = buffer->page_size();
            uint32_t dim_pages = input_tensors[i].padded_shape()[dim] / scale_factor;
            num_pages_per_block[i] = num_accum_pages * dim_pages;
            num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }

    // Correctness guard, not present in the legacy factory (which stages into a raw CB and would
    // simply run off the end of the page). The width-concat reader lays every input's sub-stick
    // down inside ONE DFB entry, so the entry has to be wide enough to hold all of them. Abort
    // rather than let the last tensor's bytes land in the next entry: a silent wrong answer here
    // is far more expensive than a failed segment.
    if (width_concat) {
        uint64_t assembled = 0;
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            assembled += page_size_per_tensor[i];
        }
        TT_FATAL(
            assembled <= single_page_size,
            "ttnn.concat (Quasar, row-major width concat): the assembled output stick is {} B but the "
            "staging entry is only {} B. Refusing to run -- this would overwrite the next entry.",
            assembled,
            single_page_size);
    }

    // ---------------------------------------------------------------------
    // ProgramSpec
    // ---------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "concat_interleaved_metal2";

    spec.tensor_parameters = {};
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TensorParamName{input_param_name(i)}, .spec = input_tensors[i].tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = CONCAT_OUTPUT_TENSOR, .spec = output.tensor_spec()});

    // DFB, NOT a scratchpad: this buffer carries data BETWEEN two kernels (reader -> writer), which
    // is precisely the DataflowBuffer contract. Each node's instance has exactly one producer
    // (the reader) and exactly one consumer (the writer), satisfying the Gen2 topology rule in
    // program_spec.cpp:3205-3310. A ScratchpadSpec would be wrong here: it carries no
    // producer/consumer credit, so the writer would have no way to know a page is ready.
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = CONCAT_DFB,
        .entry_size = single_page_size,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format,
    }};

    // ---------------------------------------------------------------------
    // Reader
    // ---------------------------------------------------------------------
    KernelSpec reader{
        .unique_id = CONCAT_READER,
        .source = rm_layout
                      ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                        "reader_concat_stick_layout_interleaved_start_id_metal2.cpp"
                      : "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                        "reader_concat_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {ProducerOf(CONCAT_DFB, "in")},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    {
        std::vector<std::string> sequence_members;
        sequence_members.reserve(num_input_tensors);
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            reader.tensor_bindings.push_back(TensorBinding{
                .tensor_parameter_name = TensorParamName{input_param_name(i)},
                .accessor_name = input_accessor_name(i)});
            sequence_members.push_back(input_accessor_name(i));
        }
        // The variadic input count lives in the binding sequence; the kernel recovers it as
        // std::tuple_size_v<decltype(tensor::inputs)>, so no count argument is dispatched.
        reader.advanced_options.tensor_binding_sequences = {
            KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = sequence_members}};
    }
    reader.runtime_arg_schema = {
        .runtime_arg_names = rm_layout ? Group<std::string>{"num_pages", "start_tensor", "start_tensor_id"}
                                       : Group<std::string>{"num_tiles", "start_tensor", "start_tensor_id"}};
    // Per-tensor uint32 arrays: 2N for the tiled reader, 3N for the row-major reader (which also
    // needs each input's stick size, since width concat mixes differently-sized sticks).
    const uint32_t varargs_per_tensor = rm_layout ? 3u : 2u;
    reader.advanced_options.num_runtime_varargs = varargs_per_tensor * num_input_tensors;
    if (width_concat) {
        reader.compiler_options.defines["WIDTH_CONCAT"] = "1";
    }

    // ---------------------------------------------------------------------
    // Writer -- the shared Metal-2.0 unary writers, bound as the DFB's single consumer
    // ---------------------------------------------------------------------
    KernelSpec writer{
        .unique_id = CONCAT_WRITER,
        .source = rm_layout ? "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp"
                            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                              "writer_unary_interleaved_start_id_metal2.cpp",
        // Accessor names are the writers' own published interface: dfb::out0 for the row-major
        // fork, dfb::out for the tiled one. Both name the same DFBSpec.
        .dfb_bindings = {ConsumerOf(CONCAT_DFB, rm_layout ? "out0" : "out")},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = CONCAT_OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema =
            {.runtime_arg_names = rm_layout ? Group<std::string>{"stick_size", "num_sticks", "start_id"}
                                            : Group<std::string>{"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{
        .name = "concat_interleaved_wu",
        .kernels = {CONCAT_READER, CONCAT_WRITER},
        .target_nodes = all_cores,
    }};

    // ---------------------------------------------------------------------
    // ProgramRunArgs
    // ---------------------------------------------------------------------
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = CONCAT_READER};
    KernelRunArgs writer_run{.kernel = CONCAT_WRITER};
    KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
    KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;

    const auto cores = (sub_core_grids.has_value() && !output.is_sharded())
                           ? cores_list
                           : grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    const uint32_t g1_num_cores = core_group_1.num_cores();
    const uint32_t dst_page_size = dst_buffer->page_size();

    for (uint32_t i = 0, num_pages_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_pages_per_core =
            (i < g1_num_cores) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
        const uint32_t block_id = num_pages_written / num_output_pages_per_block;
        uint32_t id_within_block = num_pages_written % num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            page_id_per_tensor[j] = block_id * num_pages_per_block[j];
            if (id_within_block == 0) {
                continue;
            }
            if (id_within_block >= num_pages_per_block[j]) {
                page_id_per_tensor[j] += num_pages_per_block[j];
                id_within_block -= num_pages_per_block[j];
                curr_tensor = j + 1;
            } else {
                page_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {rm_layout ? "num_pages" : "num_tiles", num_pages_per_core},
                {"start_tensor", curr_tensor},
                {"start_tensor_id", curr_tensor_id},
            });

        AdvancedKernelRunArgs::Varargs reader_varargs(varargs_per_tensor * num_input_tensors, 0u);
        for (uint32_t j = 0; j < num_input_tensors; ++j) {
            reader_varargs[j] = num_pages_per_block[j];
            reader_varargs[num_input_tensors + j] = page_id_per_tensor[j];
            if (rm_layout) {
                reader_varargs[2 * num_input_tensors + j] = page_size_per_tensor[j];
            }
        }
        // NodeCoord is CoreCoord; the vararg block is per-core because page_id_per_tensor is.
        reader_run.advanced_options.runtime_varargs.emplace(core, std::move(reader_varargs));

        if (rm_layout) {
            AddRuntimeArgsForNode(
                writer_rtas,
                core,
                {
                    {"stick_size", dst_page_size},
                    {"num_sticks", num_pages_per_core},
                    {"start_id", num_pages_written},
                });
        } else {
            AddRuntimeArgsForNode(
                writer_rtas,
                core,
                {
                    {"num_pages", num_pages_per_core},
                    {"start_id", num_pages_written},
                });
        }
        num_pages_written += num_pages_per_core;
    }

    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    // tensor_args is a Table, not a Group: key it, don't push_back. The framework refreshes these
    // bindings on every program-cache hit (ProgramSpecFactoryConcept), which is why the reader no
    // longer dispatches source base addresses as runtime args at all.
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        run_args.tensor_args.emplace(
            TensorParamName{input_param_name(i)}, ProgramRunArgs::TensorArgument{input_tensors[i].mesh_tensor()});
    }
    run_args.tensor_args.emplace(CONCAT_OUTPUT_TENSOR, ProgramRunArgs::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
