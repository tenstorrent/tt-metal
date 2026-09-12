// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_single_core_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName SC_READER{"reader"};
const KernelSpecName SC_WRITER{"writer"};
const KernelSpecName SC_COMPUTE{"compute"};
const DFBSpecName SC_IN{"in"};
const DFBSpecName SC_OUT{"out"};
const TensorParamName SC_INPUT{"input"};
const TensorParamName SC_OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;

    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    log_debug(tt::LogOp, "untilize_with_unpadding_single_core");
    log_debug(tt::LogOp, "input_dfb_data_format: {}", input_dfb_data_format);
    log_debug(tt::LogOp, "output_dfb_data_format: {}", output_dfb_data_format);

    int32_t num_tiles = a.physical_volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

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
    // Memory usage is 2 buffers of width W, plus buffer of size alignment + (W * datum size)
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

    uint32_t num_input_tiles = num_tiles_per_block;
    DataflowBufferSpec in_dfb{
        .unique_id = SC_IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_dfb_data_format,
    };

    uint32_t num_output_tiles = num_tiles_per_block;
    DataflowBufferSpec out_dfb{
        .unique_id = SC_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_dfb_data_format,
    };

    // Tilized reader
    KernelSpec reader{
        .unique_id = SC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SC_IN,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = SC_INPUT,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // Untilized writer
    KernelSpec writer{
        .unique_id = SC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_unpad_dims_split_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SC_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = SC_OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args =
            {{"float32_dtype",
              (std::uint32_t)((
                  input_dfb_data_format == tt::DataFormat::Float32 or input_dfb_data_format == tt::DataFormat::UInt32 or
                  input_dfb_data_format == tt::DataFormat::Int32))},
             {"unpadded_stick_size", unpadded_stick_size}},
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

    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_dfb_data_format == tt::DataFormat::Int32 || input_dfb_data_format == tt::DataFormat::UInt32 ||
        input_dfb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_modes = {{SC_IN, UnpackMode::UnpackToDest}};
    }

    KernelSpec compute{
        .unique_id = SC_COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp",
        .compiler_options =
            {
                .defines = std::move(compute_kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SC_IN,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SC_OUT,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", uint32_t(num_tiles / num_tiles_per_block)},
             {"per_core_block_tile_cnt", uint32_t(num_tiles_per_block)}},
        .hw_config = compute_hw_config,
    };

    ProgramSpec spec{
        .name = "untilize_with_unpadding_single_core",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = SC_INPUT, .spec = input_mesh_tensor.tensor_spec()},
             TensorParameter{.unique_id = SC_OUTPUT, .spec = output_mesh_tensor.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {SC_READER, SC_WRITER, SC_COMPUTE},
            .target_nodes = core,
        }},
    };

    NodeCoord core_0 = corerange_to_cores(core).at(0);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = SC_READER,
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(core_0, {{"num_pages", uint32_t(num_tiles)}, {"start_id", 0u}}),
        },
        KernelRunArgs{
            .kernel = SC_WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                core_0,
                {{"num_unpadded_W", output_w},
                 {"padded_W_diff_blocks", padded_W_diff_blocks},
                 {"num_unpadded_Z", output_z},
                 {"padded_Z_diff_blocks", padded_Z_diff_blocks},
                 {"num_unpadded_Y", output_y},
                 {"padded_Y_diff_blocks", padded_Y_diff_blocks},
                 {"num_leftover_Y", num_leftover_Y},
                 {"num_unpadded_X", output_x},
                 {"padded_X_size", padded_stick_size},
                 {"num_blocks_w_input", num_blocks_w_input},
                 {"num_blocks_w_output", num_blocks_w_output},
                 {"num_blocks_w_diff", num_blocks_w_diff},
                 {"block_row_size", block_row_size},
                 {"block_row_leftover_size", block_row_leftover_size}}),
        },
    };
    run_args.tensor_args = {
        {SC_INPUT, input_mesh_tensor},
        {SC_OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
