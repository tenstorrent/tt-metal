// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_single_core_program_factory.hpp"

#include <cmath>
#include <filesystem>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingSingleCoreFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    int32_t num_tiles = output.physical_volume() / TILE_HW;

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

    uint32_t unpadded_row_size_bytes = input_x * a.element_size();  // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output_x * a.element_size();   // Assuming bfloat16 dataformat

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = output_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
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

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - (num_blocks_w_input * block_row_size);

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (output_y - input_y) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (output_z - input_z) * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks =
        (output_w - input_w) * output_z * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = input_y - (input_y / TILE_HEIGHT * TILE_HEIGHT);

    const uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
    const uint32_t num_output_tiles = num_tiles_per_block;

    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    uint32_t tile_row_size_bytes = a.element_size() * TILE_HEIGHT;

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers (legacy CB 0 / c_16) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
            "reader_unary_pad_dims_split_rows.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
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
                  "num_unpadded_X",
                  "padded_X_size",
                  "pad_value",
                  "num_blocks_w_input",
                  "num_blocks_w_output",
                  "num_blocks_w_diff",
                  "block_row_size",
                  "block_row_leftover_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // ---- Writer (Metal 2.0 fork of writer_unary_interleaved_start_id) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // ---- Compute (Metal 2.0 fork of tilize) ----
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_llk_acc) {
        std::visit(
            [&](auto& c) { c.unpack_to_dest_mode.emplace(IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32); },
            compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/compute/"
            "tilize_metal2.cpp"),
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(num_tiles / num_tiles_per_block)},
             {"per_core_block_tile_cnt", static_cast<uint32_t>(num_tiles_per_block)}},
        .hw_config = compute_hw,
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_ranges}};

    // ---- Per-core runtime args (single core) ----
    KernelRunArgs reader_args{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core.start_coord,
            {{"num_unpadded_W", static_cast<uint32_t>(input_w)},
             {"padded_W_diff_blocks", padded_W_diff_blocks},
             {"num_unpadded_Z", static_cast<uint32_t>(input_z)},
             {"padded_Z_diff_blocks", padded_Z_diff_blocks},
             {"num_unpadded_Y", static_cast<uint32_t>(input_y)},
             {"padded_Y_diff_blocks", padded_Y_diff_blocks},
             {"num_leftover_Y", num_leftover_Y},
             {"num_unpadded_X", static_cast<uint32_t>(input_x)},
             {"padded_X_size", padded_row_size_bytes},
             {"pad_value", packed_pad_value},
             {"num_blocks_w_input", num_blocks_w_input},
             {"num_blocks_w_output", num_blocks_w_output},
             {"num_blocks_w_diff", num_blocks_w_diff},
             {"block_row_size", block_row_size},
             {"block_row_leftover_size", block_row_leftover_size}})};

    KernelRunArgs writer_args{
        .kernel = WRITER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core.start_coord, {{"num_pages", static_cast<uint32_t>(num_tiles)}, {"start_id", 0u}})};

    ProgramSpec spec{
        .name = "tilize_with_val_padding_single_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
