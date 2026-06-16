// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/math.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

#include <bit>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeHCTiledInterleavedProgramFactory::create_program_artifacts(
    const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};          // legacy c_0: tile stream (reader produces, writer consumes)
    const DFBSpecName CB_PADDING{"cb_padding"};  // legacy c_1: pad-tile scratchpad (reader produces, writer consumes)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp";

    const auto& input_tensor = tensor_args.input;
    // pad_value is always defined at API level; padding is decided purely by shape.
    const float pad_value = operation_attributes.pad_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    auto tile = input_tensor.tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    const uint32_t C = input_tensor.logical_shape()[1];
    const bool needs_padding = (C % tile_shape[1] != 0);

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t element_size = input_tensor.element_size();
    const uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];

    log_debug(tt::LogOp, "transpose_hc_tiled_interleaved");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "needs_padding: {}", needs_padding);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // ------------------------------------------------------------------------
    // Pad-value packing, preserved exactly from the legacy factory.
    // ------------------------------------------------------------------------
    const auto max_padding_write = face_shape[0] * face_shape[1];
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (needs_padding) {
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

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs. cb_in0 (c_0) is the main tile FIFO. cb_padding (c_1) is the pad-tile
    // scratchpad: legacy only allocated it when needs_padding, but both kernels construct the
    // handle unconditionally, so under the Metal 2.0 binding model it is always declared (a single
    // face-sized entry, never touched when needs_padding is false).
    // ------------------------------------------------------------------------
    const uint32_t num_input_tiles = 2;
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec cb_padding_spec{
        .unique_id = CB_PADDING,
        .entry_size = max_padding_write * element_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters. Both carried RuntimeTensorShape on the legacy accessors,
    // which maps to dynamic_tensor_shape = true.
    // ------------------------------------------------------------------------
    TensorParameter input_param{
        .unique_id = INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };
    TensorParameter output_param{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };

    // ------------------------------------------------------------------------
    // Reader: streams input tiles linearly into cb_in0, then optionally fills one pad tile into
    // cb_padding. The legacy H/W/swap_hw/etc. named CTAs are dummies (swap_hw=0) for this caller
    // but preserved verbatim since the shared kernel reads them.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CB_PADDING,
                 .accessor_name = "cb_padding",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args =
            {{"num_writes", num_writes},
             {"padding_val_packed", padding_val_packed},
             {"needs_padding", static_cast<uint32_t>(needs_padding)},
             {"swap_hw", 0u},
             {"H", 1u},
             {"W", 1u},
             {"accumulated_outer_dims", 1u},
             {"tile_height", 1u},
             {"tile_width", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ------------------------------------------------------------------------
    // Writer: consumes cb_in0 (legacy cb_id_out0 == c_0) and writes transposed sub-tile lines,
    // then consumes cb_padding to write the channel-padding region (Case-1 page access).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = CB_IN0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = CB_PADDING,
                 .accessor_name = "cb_padding",
                 .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args =
            {{"element_size", element_size},
             {"C", C},
             {"H", H},
             {"W", W},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]},
             {"needs_padding", static_cast<uint32_t>(needs_padding)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_idx", "end_tile_idx", "start_padding_tile_idx", "end_padding_tile_idx"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args. Two parallel work splits (unpadded vs padded tile counts) over the
    // full grid; idle cores carry zero counts. Walk preserved exactly from the legacy factory.
    // ------------------------------------------------------------------------
    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles =
        num_output_tiles / (output_tensor.padded_shape()[2] / tile_shape[0]);  // only last row of Ct has padding

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

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
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

        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back({node, {{"num_tiles", num_tiles_per_core}, {"start_id", start_idx}}});
        writer_run.runtime_arg_values.push_back(
            {node,
             {{"start_tile_idx", start_idx},
              {"end_tile_idx", end_idx},
              {"start_padding_tile_idx", padded_start_idx},
              {"end_padding_tile_idx", padded_end_idx}}});

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }

    WorkUnitSpec wu{
        .name = "transpose_hc_tiled_interleaved",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = total_cores,
    };

    ProgramSpec spec{
        .name = "transpose_hc_tiled_interleaved",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec, cb_padding_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
