// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

#include <vector>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& output) {
    const auto& a = tensor_args.input;
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

    tt::tt_metal::IDevice* device = a.device();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t tensor_height = a.physical_volume() / tensor_width;

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    bool input_is_sharded = a.is_sharded();
    std::vector<CoreCoord> ordered_cores_with_data;

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_compute_cores,
         compute_core_range,
         full_compute_core_range,
         cliff_compute_core_range,
         num_rows_per_full_core,
         num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);

    uint32_t num_input_blocks_across_width = 1;
    uint32_t num_tiles_per_input_block = num_tiles_per_row;
    uint32_t num_input_blocks_per_full_core = num_rows_per_full_core;
    uint32_t num_input_blocks_per_cliff_core = num_rows_per_cliff_core;
    uint32_t input_shard_height = 0;
    uint32_t input_shard_width = 0;
    if (input_is_sharded) {
        ShardSpec input_shard_spec = a.shard_spec().value();
        input_shard_height = input_shard_spec.shape[0];
        input_shard_width = input_shard_spec.shape[1];
        num_compute_cores = input_shard_spec.grid.num_cores();

        num_input_blocks_across_width = tt::div_up(tensor_width, input_shard_width);
        num_tiles_per_input_block = input_shard_width / tile_width;
        num_input_blocks_per_full_core = input_shard_height / tile_height;
        num_input_blocks_per_cliff_core = 0;

        ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(a);
        compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));
        full_compute_core_range = compute_core_range;
        cliff_compute_core_range = CoreRangeSet();
    }

    bool has_uneven_sharding = false;
    if (input_is_sharded) {
        uint32_t height_remainder = tensor_height % input_shard_height;
        uint32_t width_remainder = tensor_width % input_shard_width;
        has_uneven_sharding = (height_remainder != 0) || (width_remainder != 0);
    }

    const bool input_is_dram_sharded = input_is_sharded && src0_buffer->buffer_type() == BufferType::DRAM;
    bool use_block_reader = input_is_sharded && (has_uneven_sharding || input_is_dram_sharded);

    // Input DFB
    uint32_t input_cb_num_tiles;
    if (input_is_sharded && !use_block_reader) {
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        input_cb_num_tiles =
            (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    }
    const bool input_dfb_borrowed = input_is_sharded && !use_block_reader;

    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    if (input_dfb_borrowed) {
        in_dfb_spec.borrowed_from = INPUT_TENSOR;
    }

    // Output DFB
    uint32_t output_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader (one of three variants, chosen exactly as the legacy factory did).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    if (use_block_reader) {
        reader_spec.source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_sharded_blocks.cpp"};
        reader_spec.tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}};
        reader_spec.compile_time_args = {{"tiles_per_block", num_tiles_per_input_block}};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"start_shard_id", "num_blocks"}};
    } else if (input_is_sharded) {
        // Even sharding with pack_untilize: the DFB is borrowed onto the sharded input buffer.
        reader_spec.source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}};
    } else {
        // Interleaved input
        reader_spec.source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp"};
        reader_spec.tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}};
    }

    // ------------------------------------------------------------------------
    // Writer
    // ------------------------------------------------------------------------
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width = tensor_width;
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(tensor_width, output_page_width);
    }

    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                        "writer_unary_stick_layout_split_rows_multi_core.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_output_blocks_across_width", output_num_blocks_across_width},
             {"output_element_size", output_element_size},
             {"num_cols_per_input_block", num_cols_per_input_block},
             {"num_cols_per_output_block", num_cols_per_output_block}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_blocks_to_process",
                  "height_wise_input_block_start_index",
                  "num_unpadded_cols_per_input_block",
                  "width_wise_output_block_start_index",
                  "num_cols_already_processed_in_first_output_block"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ------------------------------------------------------------------------
    // Compute (single kernel; full/cliff cores share CTAs and differ only in the runtime block
    // count, so we place one KernelSpec over the whole compute grid).
    // ------------------------------------------------------------------------
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
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
                                        "untilize_variable_num_blocks_metal2.cpp"},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_modes,
            },
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args (reproduces the legacy full-then-cliff walk exactly).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    uint32_t tile_start_index = 0;
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = input_is_sharded
                                            ? ordered_cores_with_data
                                            : corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);

    auto push_reader_args = [&](const CoreCoord& core,
                                uint32_t shard_or_start_index,
                                uint32_t num_blocks,
                                uint32_t num_tiles_to_read,
                                uint32_t start_page_id) {
        const NodeCoord node = core;
        if (use_block_reader) {
            reader_run.runtime_arg_values.push_back(
                {node, {{"start_shard_id", shard_or_start_index}, {"num_blocks", num_blocks}}});
        } else if (input_is_sharded) {
            reader_run.runtime_arg_values.push_back({node, {{"num_tiles_per_core", num_tiles_to_read}}});
        } else {
            reader_run.runtime_arg_values.push_back(
                {node, {{"num_tiles", num_tiles_to_read}, {"start_page_id", start_page_id}}});
        }
    };

    for (uint32_t i = 0; i < full_cores.size(); ++i) {
        CoreCoord core = full_cores[i];
        uint32_t height_wise_input_block_start_index =
            (i / num_input_blocks_across_width) * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = i % num_input_blocks_across_width;

        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        if (input_is_sharded) {
            bool is_last_input_shard_in_row = width_wise_input_block_index == num_input_blocks_across_width - 1;
            if (is_last_input_shard_in_row) {
                uint32_t input_shard_width_local = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, input_shard_width_local) - tensor_width);
            }
        }

        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t input_shard_height_local = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, input_shard_height_local);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process =
                    num_input_blocks_per_full_core -
                    (tt::round_up(tensor_height, input_shard_height_local) - tensor_height) / tile_height;
            }
        }

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        push_reader_args(core, i, num_input_blocks_to_process, num_tiles_to_read, tile_start_index);

        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        const NodeCoord node = core;
        writer_run.runtime_arg_values.push_back(
            {node,
             {{"num_input_blocks_to_process", num_input_blocks_to_process},
              {"height_wise_input_block_start_index", height_wise_input_block_start_index},
              {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
              {"width_wise_output_block_start_index", width_wise_output_block_start_index},
              {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block}}});

        compute_run.runtime_arg_values.push_back({node, {{"per_core_block_cnt", num_input_blocks_to_process}}});

        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    std::vector<CoreCoord> cliff_cores = corerange_to_cores(cliff_compute_core_range, std::nullopt, is_row_major);
    if (!cliff_cores.empty()) {
        CoreCoord cliff_core = cliff_cores[0];
        uint32_t height_wise_input_block_start_index = full_cores.size() * num_input_blocks_per_full_core;
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        uint32_t num_input_blocks_to_process = num_input_blocks_per_cliff_core;

        uint32_t input_block_global_col_index = 0;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        const NodeCoord node = cliff_core;
        writer_run.runtime_arg_values.push_back(
            {node,
             {{"num_input_blocks_to_process", num_input_blocks_to_process},
              {"height_wise_input_block_start_index", height_wise_input_block_start_index},
              {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
              {"width_wise_output_block_start_index", width_wise_output_block_start_index},
              {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block}}});

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        push_reader_args(cliff_core, 0, num_input_blocks_to_process, num_tiles_to_read, tile_start_index);

        compute_run.runtime_arg_values.push_back({node, {{"per_core_block_cnt", num_input_blocks_to_process}}});
    }

    WorkUnitSpec wu{
        .name = "untilize_multi_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = compute_core_range,
    };

    ProgramSpec spec{
        .name = "untilize_multi_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
