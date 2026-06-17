// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "untilize_single_core_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts UntilizeSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

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

    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

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
    uint32_t num_total_sticks = a.physical_volume() / a.padded_shape()[-1] * num_columns_of_blocks;
    uint32_t output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks;
    (void)output_stick_size;  // legacy CTA slot kept for parity; not read by the kernel.

    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;

    // ------------------------------------------------------------------------
    // DataflowBuffers (single-tile-page input/output streams).
    // ------------------------------------------------------------------------
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_block,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_block,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp"},
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                        "writer_unary_stick_layout_split_rows_single_core.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_blocks_across_height", num_blocks_across_height},
             {"num_output_columns_of_blocks", num_columns_of_blocks},
             {"num_blocks_per_output_column_row", num_blocks_per_column_row},
             {"num_tiles_per_output_block", num_tiles_per_block},
             {"output_single_block_width_size", output_single_block_width_size}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_dest_acc_en) {
        unpack_to_dest_modes.insert({IN_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_compute_metal2.cpp"},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"per_core_block_cnt", num_blocks}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_modes,
            },
    };

    // Only the reader carries runtime args; the writer's per-node values are all compile-time
    // (its sole legacy RTA, the output address, is now supplied via the output TensorArgument).
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    const NodeCoord node = CoreCoord{0, 0};
    reader_run.runtime_arg_values.push_back({node, {{"num_tiles", num_tiles}, {"start_page_id", 0u}}});

    WorkUnitSpec wu{
        .name = "untilize_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
    };

    ProgramSpec spec{
        .name = "untilize_single_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
