// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_single_core_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingSingleCoreFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};
    const NodeCoord node = core.start_coord;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    uint32_t num_tiles = output.physical_volume() / TILE_HW;

    auto true_input_shape = a.padded_shape();
    auto true_output_shape = output.padded_shape();

    auto input_w = true_input_shape.rank() >= 4 ? true_input_shape[-4] : 1;
    auto input_z = true_input_shape.rank() >= 3 ? true_input_shape[-3] : 1;
    auto input_y = true_input_shape.rank() >= 2 ? true_input_shape[-2] : 1;
    auto input_x = true_input_shape[-1];

    auto output_w = true_output_shape.rank() >= 4 ? true_output_shape[-4] : 1;
    auto output_z = true_output_shape.rank() >= 3 ? true_output_shape[-3] : 1;
    auto output_y = true_output_shape.rank() >= 2 ? true_output_shape[-2] : 1;
    auto output_x = true_output_shape[-1];

    uint32_t unpadded_row_size_bytes = input_x * a.element_size();
    uint32_t padded_row_size_bytes = output_x * a.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = output_x / TILE_WIDTH;
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }

    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    uint32_t block_row_leftover_size = unpadded_row_size_bytes - (num_blocks_w_input * block_row_size);

    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (output_y - input_y) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (output_z - input_z) * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks =
        (output_w - input_w) * output_z * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = input_y - (input_y / TILE_HEIGHT * TILE_HEIGHT);

    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    uint32_t tile_row_size_bytes = a.element_size() * TILE_HEIGHT;

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16).
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                "reader_unary_pad_dims_split_rows.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args = {{"bytes_per_tile_row", tile_row_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "padded_W_diff_blocks",
                  "num_unpadded_Z",
                  "padded_Z_diff_blocks",
                  "num_unpadded_Y",
                  "padded_Y_diff_blocks",
                  "num_leftover_Y",
                  "pad_value",
                  "num_blocks_w_input",
                  "num_blocks_w_diff",
                  "block_row_size",
                  "block_row_leftover_size"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles / num_tiles_per_block}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_modes,
            },
    };

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    reader_run.runtime_arg_values.push_back(
        {node,
         {{"num_unpadded_W", input_w},
          {"padded_W_diff_blocks", padded_W_diff_blocks},
          {"num_unpadded_Z", input_z},
          {"padded_Z_diff_blocks", padded_Z_diff_blocks},
          {"num_unpadded_Y", input_y},
          {"padded_Y_diff_blocks", padded_Y_diff_blocks},
          {"num_leftover_Y", num_leftover_Y},
          {"pad_value", packed_pad_value},
          {"num_blocks_w_input", num_blocks_w_input},
          {"num_blocks_w_diff", num_blocks_w_diff},
          {"block_row_size", block_row_size},
          {"block_row_leftover_size", block_row_leftover_size}}});

    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    writer_run.runtime_arg_values.push_back({node, {{"num_pages", num_tiles}, {"start_id", 0u}}});

    WorkUnitSpec wu{
        .name = "tilize_with_val_padding_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = core_ranges,
    };

    ProgramSpec spec{
        .name = "tilize_with_val_padding_single_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
