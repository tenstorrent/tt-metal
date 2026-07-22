// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/math.hpp"

#include <bit>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// DFB / kernel / tensor names for the HC tiled interleaved factory's ProgramSpec.
const DFBSpecName SRC_CB{"src_cb"};
const DFBSpecName PAD_CB{"pad_cb"};
const KernelSpecName HCTI_READER{"hcti_reader"};
const KernelSpecName HCTI_WRITER{"hcti_writer"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledInterleavedProgramFactory::create_program_artifacts(
    const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
    const auto& input_tensor = tensor_args.input;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();
    // pad_value is always defined at API level; padding is decided purely by shape
    const float pad_value = operation_attributes.pad_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    auto tile = input_tensor.tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    uint32_t C = input_tensor.logical_shape()[1];
    bool needs_padding = (C % tile_shape[1] != 0);

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto max_padding_write = face_shape[0] * face_shape[1];

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

    // -------------------------------------------------------------------------
    // ProgramSpec
    // -------------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "transpose_hc_tiled_interleaved";

    // SRC_CB (legacy c_0): reader produces tiles, writer consumes them.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC_CB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });
    // PAD_CB (legacy c_1): only present when the output needs channel-tile padding.
    // Reader fills it once (PRODUCER); writer drains it (CONSUMER).
    if (needs_padding) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD_CB,
            .entry_size = max_padding_write * element_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

    // Tensor parameters. RuntimeTensorShape in the legacy factory → mirror via
    // dynamic_tensor_shape (faithful mirror of an existing relaxation).
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    });

    // NEEDS_PADDING define gates the conditional PAD_CB binding on both kernels.
    KernelSpec::CompilerOptions::Defines pad_defines;
    if (needs_padding) {
        pad_defines.emplace("NEEDS_PADDING", "1");
    }

    // Reader DFB bindings (SRC_CB producer; PAD_CB producer when padding).
    Group<DFBBinding> reader_dfbs = {
        DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    if (needs_padding) {
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = PAD_CB, .accessor_name = "padding", .endpoint_type = DFBEndpointType::PRODUCER});
    }

    KernelSpec reader{
        .unique_id = HCTI_READER,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"),
        .compiler_options = {.defines = pad_defines},
        .dfb_bindings = std::move(reader_dfbs),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
            },
        // The legacy kernel reads these as named CTAs already (get_named_compile_time_arg_val).
        .compile_time_args =
            {
                {"num_writes", num_writes},
                {"padding_val_packed", padding_val_packed},
                {"swap_hw", 0u},
                {"H", 1u},
                {"W", 1u},
                {"accumulated_outer_dims", 1u},
                {"tile_height", 1u},
                {"tile_width", 1u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
    };

    // Writer DFB bindings (SRC_CB consumer; PAD_CB consumer when padding).
    Group<DFBBinding> writer_dfbs = {
        DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    if (needs_padding) {
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = PAD_CB, .accessor_name = "padding", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    KernelSpec writer{
        .unique_id = HCTI_WRITER,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"),
        .compiler_options = {.defines = std::move(pad_defines)},
        .dfb_bindings = std::move(writer_dfbs),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"element_size", element_size},
                {"C", C},
                {"H", H},
                {"W", W},
                {"tile_height", tile_shape[0]},
                {"tile_width", tile_shape[1]},
                {"face_height", face_shape[0]},
                {"face_width", face_shape[1]},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_idx", "end_tile_idx", "start_padding_tile_idx", "end_padding_tile_idx"}},
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "hcti_wu",
        .kernels = {HCTI_READER, HCTI_WRITER},
        .target_nodes = total_cores,
    });

    // -------------------------------------------------------------------------
    // ProgramRunArgs (per-core; identical work split to the legacy factory)
    // -------------------------------------------------------------------------
    ProgramRunArgs run_args;

    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] /
                                                           tile_shape[0]);  // only last row of Ct should have padding

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

    KernelRunArgs reader_run{.kernel = HCTI_READER};
    KernelRunArgs writer_run{.kernel = HCTI_WRITER};

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    // Need to set runtime args for all cores, not just the ones doing work.
    for (const auto& core : total_cores) {
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

        KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"num_tiles", num_tiles_per_core},
                {"start_id", start_idx},
            });
        AddRuntimeArgsForNode(
            writer_rtas,
            core,
            {
                {"start_tile_idx", start_idx},
                {"end_tile_idx", end_idx},
                {"start_padding_tile_idx", padded_start_idx},
                {"end_padding_tile_idx", padded_end_idx},
            });

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    run_args.tensor_args.emplace(INPUT, input_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
