// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace ttnn::prim {

namespace {

// Per-core runtime args for the reader + writer kernels. The work split uses
// two parallel partitions (unpadded vs padded tile counts) so each core needs a
// (start, end) pair from both. Writer also tracks the padded range; reader only
// the unpadded count.
void emit_runtime_args_hc_tiled_interleaved(
    ProgramRunArgs::KernelRunArgs& reader_run_args,
    ProgramRunArgs::KernelRunArgs& writer_run_args,
    const CoreRangeSet& active_cores,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2,
    const CoreRangeSet& padded_core_group_1,
    uint32_t padded_num_tiles_per_core_group_1,
    const CoreRangeSet& padded_core_group_2,
    uint32_t padded_num_tiles_per_core_group_2) {
    auto cores = corerange_to_cores(active_cores, std::nullopt);

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core;
        uint32_t padded_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        if (padded_core_group_1.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_1;
        } else if (padded_core_group_2.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_2;
        } else {
            padded_tiles_per_core = 0;
        }

        uint32_t end_idx = start_idx + num_tiles_per_core;
        uint32_t padded_end_idx = padded_start_idx + padded_tiles_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"start_id", start_idx}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_tile_idx", start_idx},
             {"end_tile_idx", end_idx},
             {"start_padding_tile_idx", padded_start_idx},
             {"end_padding_tile_idx", padded_end_idx}});

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledInterleavedProgramFactory::create_program_artifacts(
    const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const DFBSpecName PAD{"pad"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    // pad_value is always defined at API level; padding is decided purely by shape
    const float pad_value = operation_attributes.pad_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    auto tile = input_tensor.tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    uint32_t C = input_tensor.logical_shape()[1];
    bool needs_padding = (C % tile_shape[1] != 0);

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] / tile_shape[0]);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    CoreRangeSet active_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;

    ProgramSpec spec{.name = "transpose_hc_tiled_interleaved"};

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dfb_data_format,
    });

    auto max_padding_write = face_shape[0] * face_shape[1];
    if (needs_padding) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD,
            .entry_size = max_padding_write * input_tensor.element_size(),
            .num_entries = 1,
            .data_format_metadata = dfb_data_format,
        });
    }

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    uint32_t element_size = input_tensor.element_size();
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];

    if (C % tile_shape[1] != 0) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = max_padding_write / num_packed_values;
        switch (input_tensor.dtype()) {
            case DataType::INT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    // The reader is the shared Metal 2.0 fork that the permute op also binds; its binding
    // vocabulary (dfb::cb_in0 / dfb::cb_pad / tensor::input, and the NEEDS_PADDING define that
    // gates the padding buffer) is fixed by that kernel, so these accessor names conform to it.
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (needs_padding) {
        reader_defines.insert({"NEEDS_PADDING", "1"});
        writer_defines.insert({"NEEDS_PADDING", "1"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "cb_in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_writes", num_writes},
             {"padding_val_packed", padding_val_packed},
             {"swap_hw", 0u},
             {"H", 1u},
             {"W", 1u},
             {"accumulated_outer_dims", 1u},
             {"tile_height", 1u},
             {"tile_width", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "out0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"element_size", element_size},
             {"C", C},
             {"H", H},
             {"W", W},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_idx", "end_tile_idx", "start_padding_tile_idx", "end_padding_tile_idx"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    };

    if (needs_padding) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = PAD,
            .accessor_name = "cb_pad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = PAD,
            .accessor_name = "pad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = active_cores,
    });

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};

    emit_runtime_args_hc_tiled_interleaved(
        reader_run_args,
        writer_run_args,
        active_cores,
        core_group_1,
        num_tiles_per_core_group_1,
        core_group_2,
        num_tiles_per_core_group_2,
        padded_core_group_1,
        padded_num_tiles_per_core_group_1,
        padded_core_group_2,
        padded_num_tiles_per_core_group_2);

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
