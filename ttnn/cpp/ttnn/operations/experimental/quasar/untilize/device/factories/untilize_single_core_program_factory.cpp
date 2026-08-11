// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>
#include "untilize_single_core_program_factory.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {
ttnn::device_operation::ProgramArtifacts UntilizeSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    CoreRange core({0, 0}, {0, 0});
    CoreRangeSet core_ranges{core};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_volume = tile_height * tile_width;

    uint32_t num_tiles = a.physical_volume() / tile_volume;
    uint32_t num_blocks_across_height = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    uint32_t num_columns_of_blocks = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        uint32_t output_shard_width;
        if (output.shard_spec().has_value()) {
            output_shard_width = output.shard_spec().value().shape[1];
        } else {
            output_shard_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        num_columns_of_blocks = a.padded_shape()[-1] / output_shard_width;
    }
    uint32_t num_tiles_per_column_row = a.padded_shape()[-1] / num_columns_of_blocks / tile_width;

    // Determine how much L1 space we can use for input and output CBs,
    // ensuring that we don't intrude into other L1 storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

    // Determine how many tiles each block will store.
    // Currently we require that the number of tiles in a row is divisible by the number of blocks in a row, or
    // equivalently the number of tiles in a row is divisible by the number of tiles in a block.
    uint32_t num_tiles_per_block = num_tiles_per_column_row;
    if (num_tiles_per_block > max_tiles_per_cb) {
        for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
            if (num_tiles_per_column_row % i == 0) {
                num_tiles_per_block = i;
                break;
            }
        }
    }

    uint32_t num_blocks_per_column_row = num_tiles_per_column_row / num_tiles_per_block;
    uint32_t output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    // (legacy computed output_stick_size / num_total_sticks only to feed the writer's
    // output_stick_size CTA, which the kernel never read; dropped with that dead CTA.)

    uint32_t input_cb_num_tiles = num_tiles_per_block;
    uint32_t output_cb_num_tiles = num_tiles_per_block;
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers (legacy CB c_0 / c_16) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader (Metal 2.0 fork of reader_unary_start_id) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/dataflow/"
                                        "reader_unary_start_id_metal2.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/dataflow/"
                                        "writer_unary_stick_layout_split_rows_single_core.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_blocks_across_height", num_blocks_across_height},
             {"num_output_columns_of_blocks", num_columns_of_blocks},
             {"num_blocks_per_output_column_row", num_blocks_per_column_row},
             {"num_tiles_per_output_block", num_tiles_per_block},
             {"output_single_block_width_size", output_single_block_width_size}},
        .hw_config =
            ttnn::create_writer_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Compute (Metal 2.0 fork of untilize) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_dest_acc_en) {
        std::visit([&](auto& c) { c.unpack_modes.emplace(IN, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"per_core_block_cnt", num_blocks}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = compute_hw,
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_ranges}};

    // ---- Per-core runtime args (single core) ----
    uint32_t start_page_id = 0;
    KernelRunArgs reader_args{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core.start_coord, {{"num_tiles", num_tiles}, {"start_page_id", start_page_id}})};

    ProgramSpec spec{
        .name = "untilize_single_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim::qsr
