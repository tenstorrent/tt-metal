// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "clone_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::data_movement::clone {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts CloneProgramFactory::create_program_artifacts(
    const CloneOperation::operation_attributes_t& operation_attributes,
    const CloneOperation::tensor_args_t& tensor_args,
    CloneOperation::tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto input_data_format = datatype_to_dataformat_converter(input.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    bool convert_dtype = input_data_format != output_data_format;
    bool tilized = output.layout() == Layout::TILE;
    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tilized ? tt::tile_size(data_format) : tensor.logical_shape()[-1] * tensor.element_size();
    };
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.logical_shape()[-1];

    auto output_memory_layout = output.memory_config().memory_layout();
    bool is_sharded = output_memory_layout != TensorMemoryLayout::INTERLEAVED;

    uint32_t num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1;
    uint32_t num_units_per_core_group_2;
    uint32_t num_cores_x;
    uint32_t num_cores_y;

    if (is_sharded) {
        auto shard_spec = output.buffer()->shard_spec();
        all_cores = shard_spec.grid();
        num_cores = all_cores.num_cores();

        auto shard_shape = shard_spec.shape();
        uint32_t shard_height = shard_shape[0];
        uint32_t shard_width = shard_shape[1];

        // For row-major sharded, the unit (stick) size must be the shard width, not the
        // full tensor width. Using tensor width causes OOB reads past the shard boundary.
        if (!tilized) {
            input_unit_size = shard_width * input.element_size();
            output_unit_size = shard_width * output.element_size();
        }

        if (tilized) {
            num_units_per_core_group_1 = (shard_height * shard_width) / TILE_HW;
        } else {
            num_units_per_core_group_1 = shard_height;
        }

        num_units_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();

        auto grid_size = all_cores.bounding_box();
        num_cores_x = grid_size.end_coord.x + 1;
        num_cores_y = grid_size.end_coord.y + 1;
    } else {
        auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_units_per_core_group_1_result,
             num_units_per_core_group_2_result] = split_work_to_cores(compute_with_storage_grid_size, num_units);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_units_per_core_group_1 = num_units_per_core_group_1_result;
        num_units_per_core_group_2 = num_units_per_core_group_2_result;
    }

    auto alignment = input.buffer()->alignment();
    uint32_t aligned_input_unit_size = tt::align(input_unit_size, alignment);
    uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);

    // ---------------------------------------------------------------------
    // Program-scope resource names (typed handles → generated dfb:: / tensor:: tokens)
    // ---------------------------------------------------------------------
    const DFBSpecName SRC{"src"};
    const DFBSpecName DST{"dst"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // The writer consumes DST when converting dtype (compute produces DST), otherwise it
    // consumes SRC directly (reader → writer, no compute). The writer kernel names its
    // endpoint `dfb::dst` in both cases; only which DataflowBufferSpec it binds changes.
    const DFBSpecName writer_dfb = convert_dtype ? DST : SRC;

    // ---------------------------------------------------------------------
    // DataflowBufferSpecs (replaces the legacy source / dest CBs; c_4 / c_20)
    // ---------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "clone";

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC,
        .entry_size = aligned_input_unit_size,
        .num_entries = 2,
        .data_format_metadata = input_data_format,
    });
    if (convert_dtype) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DST,
            .entry_size = aligned_output_unit_size,
            .num_entries = 2,
            .data_format_metadata = output_data_format,
        });
    }

    // ---------------------------------------------------------------------
    // Tensor parameters (typed bindings replace the buffer-address RTA slot 0)
    // ---------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // ---------------------------------------------------------------------
    // Kernel sources (per config branch)
    // ---------------------------------------------------------------------
    const char* read_kernel_path;
    const char* write_kernel_path;
    if (is_sharded) {
        read_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp";
        write_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp";
    } else {
        read_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                                   : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp";
        write_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp";
    }

    // Reader / writer runtime-arg schema (buffer addresses now ride the TensorBinding;
    // the interleaved paths carry a per-node start_id, the sharded paths do not).
    Group<std::string> rta_names;
    if (tilized) {
        rta_names = is_sharded ? Group<std::string>{"num_tiles"} : Group<std::string>{"num_tiles", "start_id"};
    } else {
        rta_names = is_sharded ? Group<std::string>{"stick_size", "num_sticks"}
                               : Group<std::string>{"stick_size", "num_sticks", "start_id"};
    }

    // ---------------------------------------------------------------------
    // Reader / writer KernelSpecs
    // ---------------------------------------------------------------------
    KernelSpec reader{
        .unique_id = READER,
        .source = read_kernel_path,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = write_kernel_path,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = writer_dfb, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };
    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    // ---------------------------------------------------------------------
    // Compute KernelSpecs for dtype conversion (per core group — preserved multiplicity)
    // ---------------------------------------------------------------------
    if (convert_dtype) {
        auto compute_hw =
            ttnn::to_compute_hardware_config(input.device()->arch(), operation_attributes.compute_kernel_config);
        // Metal 2.0 requires an explicit unpack_modes entry when a compute kernel consumes a
        // Float32 DFB with a 32-bit dest register. Legacy ComputeConfigDescriptor left
        // unpack_to_dest_mode default (== UnpackToSrc); mirror that value faithfully.
        if (auto* gen1 = std::get_if<ComputeGen1Config>(&compute_hw)) {
            if (input_data_format == tt::DataFormat::Float32 && gen1->enable_32_bit_dest) {
                gen1->unpack_modes = ComputeUnpackModes{{SRC, UnpackMode::UnpackToSrc}};
            }
        }

        auto make_compute = [&](const KernelSpecName& unique_id, uint32_t num_tiles) {
            return KernelSpec{
                .unique_id = unique_id,
                .source = "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp",
                .dfb_bindings =
                    {DFBBinding{
                         .dfb_spec_name = SRC, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = DST, .accessor_name = "dst", .endpoint_type = DFBEndpointType::PRODUCER}},
                .compile_time_args = {{"num_tiles", num_tiles}},
                .hw_config = compute_hw,
            };
        };

        if (!core_group_1.ranges().empty()) {
            spec.kernels.push_back(make_compute(COMPUTE_G1, num_units_per_core_group_1));
        }
        if (!core_group_2.ranges().empty()) {
            spec.kernels.push_back(make_compute(COMPUTE_G2, num_units_per_core_group_2));
        }
    }

    // ---------------------------------------------------------------------
    // Work units (placement). Reader/writer share each compute group's work unit so a
    // group node hosts reader + writer + its compute instance together; the two compute
    // groups cover disjoint nodes. Without conversion, one work unit over all_cores.
    // ---------------------------------------------------------------------
    if (convert_dtype) {
        if (!core_group_1.ranges().empty()) {
            spec.work_units.push_back(
                WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
        }
        if (!core_group_2.ranges().empty()) {
            spec.work_units.push_back(
                WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
        }
    } else {
        spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});
    }

    // ---------------------------------------------------------------------
    // Runtime args (per node). Legacy node-first loop preserved; AddRuntimeArgsForNode
    // transposes into the name-first ProgramRunArgs table. Compute kernels carry only a
    // CTA (num_tiles), so they need no KernelRunArgs entry.
    // ---------------------------------------------------------------------
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    uint32_t start_id = 0;
    uint32_t num_cores_group_1 = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t num_units_per_core = i < num_cores_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (is_sharded) {
            if (tilized) {
                AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"num_tiles", num_units_per_core}});
                AddRuntimeArgsForNode(writer_ra.runtime_arg_values, core, {{"num_tiles", num_units_per_core}});
            } else {
                AddRuntimeArgsForNode(
                    reader_ra.runtime_arg_values,
                    core,
                    {{"stick_size", input_unit_size}, {"num_sticks", num_units_per_core}});
                AddRuntimeArgsForNode(
                    writer_ra.runtime_arg_values,
                    core,
                    {{"stick_size", output_unit_size}, {"num_sticks", num_units_per_core}});
            }
        } else {
            if (tilized) {
                AddRuntimeArgsForNode(
                    reader_ra.runtime_arg_values, core, {{"num_tiles", num_units_per_core}, {"start_id", start_id}});
                AddRuntimeArgsForNode(
                    writer_ra.runtime_arg_values, core, {{"num_tiles", num_units_per_core}, {"start_id", start_id}});
            } else {
                AddRuntimeArgsForNode(
                    reader_ra.runtime_arg_values,
                    core,
                    {{"stick_size", input_unit_size}, {"num_sticks", num_units_per_core}, {"start_id", start_id}});
                AddRuntimeArgsForNode(
                    writer_ra.runtime_arg_values,
                    core,
                    {{"stick_size", output_unit_size}, {"num_sticks", num_units_per_core}, {"start_id", start_id}});
            }
            start_id += num_units_per_core;
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_ra));
    run_args.kernel_run_args.push_back(std::move(writer_ra));

    run_args.tensor_args.emplace(INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::data_movement::clone
