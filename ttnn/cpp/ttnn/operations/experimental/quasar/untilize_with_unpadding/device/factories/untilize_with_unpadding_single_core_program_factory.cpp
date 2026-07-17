// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_single_core_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreCoord core_0 = corerange_to_cores(core).at(0);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    log_debug(tt::LogOp, "untilize_with_unpadding_single_core");
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "output_cb_data_format: {}", output_cb_data_format);

    int32_t num_tiles = a.physical_volume() / TILE_HW;

    auto input_w = input_shape.rank() >= 4 ? input_shape[-4] : 1;
    auto input_z = input_shape.rank() >= 3 ? input_shape[-3] : 1;
    auto input_y = input_shape.rank() >= 2 ? input_shape[-2] : 1;
    auto input_x = input_shape[-1];

    auto output_w = output_shape.rank() >= 4 ? output_shape[-4] : 1;
    auto output_z = output_shape.rank() >= 3 ? output_shape[-3] : 1;
    auto output_y = output_shape.rank() >= 2 ? output_shape[-2] : 1;
    auto output_x = output_shape[-1];

    uint32_t padded_stick_size = input_x * output.element_size();  // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_x * output.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = input_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (output.element_size() * TILE_HEIGHT * 2 + output.element_size());
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
    uint32_t block_row_size = block_width * output.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - (num_blocks_w_output * block_row_size);

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (input_y - output_y) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (input_z - output_z) * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (input_w - output_w) * input_z * input_y / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_y - (output_y / TILE_HEIGHT * TILE_HEIGHT);

    const bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 or
                               input_cb_data_format == tt::DataFormat::UInt32 or
                               input_cb_data_format == tt::DataFormat::Int32;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    uint32_t num_input_tiles = num_tiles_per_block;
    uint32_t num_output_tiles = num_tiles_per_block;

    // ---- DataflowBuffers (legacy c_0 / c_16 CBs) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader kernel ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // ---- Writer kernel ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_unpad_dims_split_rows.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"float32_dtype", static_cast<uint32_t>(float32_dtype)}},
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
                  "num_blocks_w_input",
                  "num_blocks_w_output",
                  "num_blocks_w_diff",
                  "block_row_size",
                  "block_row_leftover_size"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // ---- Compute kernel ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (float32_dtype) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_dest_acc_en) {
        std::visit(
            [&](auto& c) { c.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/compute/"
            "untilize_metal2.cpp"),
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(num_tiles / num_tiles_per_block)},
             {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = compute_hw,
    };

    // ---- ProgramSpec ----
    ProgramSpec spec{
        .name = "untilize_with_unpadding_single_core",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core}},
    };

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                core_0, {{"num_pages", static_cast<uint32_t>(num_tiles)}, {"start_id", 0u}})},
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                core_0,
                {{"num_unpadded_W", static_cast<uint32_t>(output_w)},
                 {"padded_W_diff_blocks", padded_W_diff_blocks},
                 {"num_unpadded_Z", static_cast<uint32_t>(output_z)},
                 {"padded_Z_diff_blocks", padded_Z_diff_blocks},
                 {"num_unpadded_Y", static_cast<uint32_t>(output_y)},
                 {"padded_Y_diff_blocks", padded_Y_diff_blocks},
                 {"num_leftover_Y", num_leftover_Y},
                 {"num_unpadded_X", static_cast<uint32_t>(output_x)},
                 {"padded_X_size", padded_stick_size},
                 {"num_blocks_w_input", num_blocks_w_input},
                 {"num_blocks_w_output", num_blocks_w_output},
                 {"num_blocks_w_diff", num_blocks_w_diff},
                 {"block_row_size", block_row_size},
                 {"block_row_leftover_size", block_row_leftover_size}})},
    };
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
