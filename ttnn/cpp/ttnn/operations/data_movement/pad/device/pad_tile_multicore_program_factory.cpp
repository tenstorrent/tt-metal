// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_multicore_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::get_num_pages;

namespace {
// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName TILE_MC_READER{"reader"};
const KernelSpecName TILE_MC_WRITER{"writer"};
const DFBSpecName TILE_MC_IN0{"in0"};
const DFBSpecName TILE_MC_PAD{"pad"};
const TensorParamName TILE_MC_INPUT{"input"};
const TensorParamName TILE_MC_OUTPUT{"output"};
}  // namespace

static inline int advance_tensor_index(std::vector<uint32_t>& idx, const ttnn::Shape& dims, uint32_t ndims) {
    // increment least-significant dim first
    for (int32_t d = ndims - 1; d >= 0; d--) {
        uint32_t v = idx[d] + 1;
        if (v < dims[d]) {
            idx[d] = v;
            return 1;
        }
        idx[d] = 0;  // wrap and carry
    }
    return 0;  // overflowed most-significant dim
}

ttnn::device_operation::ProgramArtifacts PadTileMulticoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const auto& a_shape = a.logical_shape();
    uint32_t num_pages = get_num_pages(output);

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), num_pages)
                                   : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    auto cores_in_order = corerange_to_cores(all_cores, num_cores, true);

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t page_size = output.buffer()->page_size();
    uint32_t multi_buffering_size = 2;

    DataflowBufferSpec in0_dfb{
        .unique_id = TILE_MC_IN0,
        .entry_size = page_size,
        .num_entries = multi_buffering_size,
        .data_format_metadata = dfb_data_format,
    };

    // Pad buffer: the writer fills one entry with the pad value and NoC-writes it out for every
    // page outside the input region. Nothing drains it, so the writer is its only toucher and
    // binds both endpoints (self-loop).
    DataflowBufferSpec pad_dfb{
        .unique_id = TILE_MC_PAD,
        .entry_size = page_size,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    switch (a.dtype()) {
        case DataType::INT32:
        case DataType::UINT32: packed_pad_value = pad_value; break;
        case DataType::BFLOAT16:
            packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});
            break;
        case DataType::UINT16:
            packed_pad_value = ttnn::operations::data_movement::pack_two_uint16_into_uint32(
                {ttnn::operations::data_movement::float_to_uint16(pad_value),
                 ttnn::operations::data_movement::float_to_uint16(pad_value)});
            break;
        case DataType::FLOAT32: packed_pad_value = std::bit_cast<uint32_t>(pad_value); break;
        default:
            packed_pad_value = 0;
            TT_ASSERT(
                false,
                "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                "FLOAT32");
    }

    // The four num_dims-long RTA blocks below (input/output page shapes and per-dim ids) are
    // reached by index in `for (d < num_dims)` loops, so they travel as runtime varargs rather
    // than named arguments.
    const uint32_t num_dims = static_cast<uint32_t>(output_padded_shape.rank());
    const uint32_t num_varargs = 4 * num_dims;

    KernelSpec reader{
        .unique_id = TILE_MC_READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_MC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_MC_INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args =
            {
                {"page_size", page_size},
                {"num_dims", num_dims},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = num_varargs},
    };

    KernelSpec writer{
        .unique_id = TILE_MC_WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_tiled.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_MC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = TILE_MC_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TILE_MC_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_MC_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .compile_time_args =
            {
                {"page_size", page_size},
                {"num_dims", num_dims},
                {"pad_value", packed_pad_value},
                {"element_size", static_cast<uint32_t>(output.element_size())},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = num_varargs},
    };

    KernelRunArgs reader_run_args{.kernel = TILE_MC_READER};
    KernelRunArgs writer_run_args{.kernel = TILE_MC_WRITER};

    /*
    As an example, lets say we want to pad a [2, 1, 32, 32] tensor to [2, 3, 64, 64]
    The input tensor exists as [2, 2, 1, 1] if we reduce by tile (page) size, and the output as [2, 3, 2, 2]
    we increment through these shapes, and will write a total of 2 * 3 * 2 * 2 = 24 tiles, so we will utilize 24 cores
    for each core, we calculate if we are within the "input region" of the output. this does a check of
    if any element in the incremented input_id_per_dim is less than the output_id_per_dim, if so, we are outside
    the input region, and we will write a padding tile, and we will not increment the input_id_per_dim for that tile.
    if we are within the input region, we will write the tile from input to the output, and increment the
    input_id_per_dim. This works because we increment the least-significant dim first, and the input region correctly
    matches the output after the output wraps around. In this example:
    Core 0: input_id_per_dim: [0,0,0,0] ; output_id_per_dim: [0,0,0,0], we copy the tile and increment both input and
    output dims, next ->
    Core 1: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,0,1], the last output dim is
    greater than input, so we write the pad tile, and increment only output dim, next ->
    Core 2: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,1,0], the second last output dim is greater than
    input, so we write the pad tile, and increment only output dim, next ->
    Core 3: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,1,1], the last 2 output dims is greater than input,
    so we write the pad tile, and increment only output dim, next ->
    Core 4: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,1,0,0], we copy the tile and increment, next ->
    Core 5: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [0,1,0,1], Some output dims are greater
    than input, so we write the pad tile, and increment only output dim. next ->
    Core 6: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [0,1,1,0],
    Core 7, 8, 9, 10, 11, we write pad tiles, incrementing only output dim each time, next ->
    Core 12: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [1,0,0,0], we copy the tile and increment, next ->
    Core 13: input_id_per_dim: [1,1,0,0] ; output_id_per_dim: [1,0,0,1], Core 13, 14, 15, we write pad tiles,
    incrementing only output dim each time, next -> Core 16: input_id_per_dim: [1,1,0,0] ; output_id_per_dim: [1,1,0,0],
    we copy the tile and increment, next ->
    From now on, input_id_per_dim wraps around it's most significant dim, resulting in [0,0,0,0].
    This means for every output_id_per_dim, an element will always be greater than
    input_id_per_dim, so every core after core 16 will only write pad tiles, which is correct as we have filled all of
    the input region, and will always be outside of it from now on.

    As you can see, the input_id_per_dim only increments when we are within the input region of the output,
    and the output_id_per_dim increments every time, this means that when the output wraps around, the input
    will be correctly positioned for the next set of output tiles.
    */

    std::vector<uint32_t> input_id_per_dim, output_id_per_dim;  // input and output id_per_dims
    // initialize id_per_dims to vectors of length num_dims filled with 0
    input_id_per_dim.resize(a_shape.rank(), 0);
    output_id_per_dim.resize(output_padded_shape.rank(), 0);
    // instantiate the input and output tensor padded shapes
    auto input_page_shape = a.padded_shape();
    auto output_page_shape = output_padded_shape;
    input_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    input_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    bool within_input_region;
    uint32_t input_page_offset = 0;
    uint32_t output_page_offset = 0;

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = cores_in_order[i];

        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            num_pages_per_core = 0;  // no-op
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_pages_to_write", num_pages_per_core}, {"start_offset", input_page_offset}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages_to_write", num_pages_per_core}, {"start_offset", output_page_offset}});

        // Every core should get the same input and output tile shapes, and then where the core
        // should start writing in the output tensor.
        AdvancedKernelRunArgs::Varargs varargs;
        varargs.reserve(num_varargs);
        for (auto v : input_page_shape) {
            varargs.push_back(v);
        }
        for (auto v : output_page_shape) {
            varargs.push_back(v);
        }
        for (uint32_t v : input_id_per_dim) {
            varargs.push_back(v);
        }
        for (uint32_t v : output_id_per_dim) {
            varargs.push_back(v);
        }
        reader_run_args.advanced_options.runtime_varargs[core] = varargs;
        writer_run_args.advanced_options.runtime_varargs[core] = std::move(varargs);

        // We now need to increment the input and output id_per_dims by the number of pages this core is processing
        // Similarly to in the kernel, we only increment the input id_per_dim if we are within the input region
        for (uint32_t p = 0; p < num_pages_per_core; p++) {
            within_input_region = true;
            for (uint32_t d = 0; d < input_id_per_dim.size(); d++) {
                if (input_id_per_dim[d] < output_id_per_dim[d]) {
                    within_input_region = false;
                    break;
                }
            }
            if (within_input_region) {
                advance_tensor_index(input_id_per_dim, input_page_shape, input_id_per_dim.size());
                input_page_offset++;
            }
            advance_tensor_index(output_id_per_dim, output_page_shape, output_id_per_dim.size());
            output_page_offset++;
        }
        // The input and output id_per_dim should now be set correctly for the next core
    }

    ProgramSpec spec{
        .name = "pad_tile_multicore",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in0_dfb), std::move(pad_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = TILE_MC_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = TILE_MC_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {TILE_MC_READER, TILE_MC_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {TILE_MC_INPUT, TensorArgument{input_mesh_tensor}},
        {TILE_MC_OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
