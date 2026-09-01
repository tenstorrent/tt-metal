// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& output) {
    const auto& a = tensor_args.input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const DFBSpecName SRC0{"src0"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    constexpr const char* READER_BLOCK_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_sharded_blocks.cpp";
    constexpr const char* READER_SHARDED_SRC =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp";
    constexpr const char* READER_INTERLEAVED_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id_metal2.cpp";
    constexpr const char* WRITER_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp";
    constexpr const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
        "untilize_variable_num_blocks_metal2.cpp";

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    IDevice* device = a.device();
    Buffer* src0_buffer = a.buffer();
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

    // Block reader: unbacked double-buffer DFB, reads from L1 shard block-by-block.
    // Even sharding uses zero-copy borrowed-memory DFB (fast production path).
    bool use_block_reader = input_is_sharded && (has_uneven_sharding || input_is_dram_sharded);

    // Input DFB
    uint32_t input_dfb_num_tiles;
    if (input_is_sharded && !use_block_reader) {
        // Even sharding with pack_untilize: DFB is backed by the sharded buffer (zero-copy)
        input_dfb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        // Block reader (sharded) or interleaved: double-buffer
        input_dfb_num_tiles =
            (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    }
    const bool input_dfb_borrowed = input_is_sharded && !use_block_reader;

    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = input_single_tile_size,
        .num_entries = input_dfb_num_tiles,
        .data_format_metadata = input_data_format,
    };
    if (input_dfb_borrowed) {
        src0_dfb.borrowed_from = INPUT;
    }

    // Output DFB
    uint32_t output_dfb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_dfb_num_tiles,
        .data_format_metadata = output_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Reader kernel — source and bindings selected by the input layout.
    KernelSpec reader_spec{
        .unique_id = READER,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    if (use_block_reader) {
        reader_spec.source = std::filesystem::path{READER_BLOCK_SRC};
        reader_spec.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}};
        reader_spec.compile_time_args = {{"tiles_per_block", num_tiles_per_input_block}};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"start_shard_id", "num_blocks"}};
    } else if (input_is_sharded) {
        // Even sharding: DFB borrowed from the input buffer; reader only pushes the readiness handshake.
        reader_spec.source = std::filesystem::path{READER_SHARDED_SRC};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}};
    } else {
        reader_spec.source = std::filesystem::path{READER_INTERLEAVED_SRC};
        reader_spec.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}};
    }

    // Writer kernel.
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
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
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
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Compute kernel(s) — full + cliff (cliff only for interleaved input).
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }
    auto make_compute = [&](const KernelSpecName& id) {
        ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});
        }
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{COMPUTE_SRC},
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
            .hw_config = std::move(compute_cfg),
        };
    };

    const bool full_present = !full_compute_core_range.ranges().empty();
    const bool cliff_present = !cliff_compute_core_range.ranges().empty();

    ProgramSpec spec{
        .name = "untilize_multi_core",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {src0_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
    };
    if (full_present) {
        spec.kernels.push_back(make_compute(COMPUTE_FULL));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = full_compute_core_range});
    }
    if (cliff_present) {
        spec.kernels.push_back(make_compute(COMPUTE_CLIFF));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = cliff_compute_core_range});
    }

    // Run-time args.
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_full_run{.kernel = COMPUTE_FULL};
    KernelRunArgs compute_cliff_run{.kernel = COMPUTE_CLIFF};

    uint32_t tile_start_index = 0;

    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = input_is_sharded
                                            ? ordered_cores_with_data
                                            : corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);

    auto set_reader_args = [&](const CoreCoord& core, uint32_t block_index, uint32_t num_blocks, uint32_t num_tiles) {
        if (use_block_reader) {
            AddRuntimeArgsForNode(
                reader_run.runtime_arg_values, core, {{"start_shard_id", block_index}, {"num_blocks", num_blocks}});
        } else if (input_is_sharded) {
            AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles}});
        } else {
            AddRuntimeArgsForNode(
                reader_run.runtime_arg_values, core, {{"num_tiles", num_tiles}, {"start_id", tile_start_index}});
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
                uint32_t shard_w = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, shard_w) - tensor_width);
            }
        }

        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t shard_h = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, shard_h);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process = num_input_blocks_per_full_core -
                                              (tt::round_up(tensor_height, shard_h) - tensor_height) / tile_height;
            }
        }

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        set_reader_args(core, i, num_input_blocks_to_process, num_tiles_to_read);

        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_input_blocks_to_process", num_input_blocks_to_process},
             {"height_wise_input_block_start_index", height_wise_input_block_start_index},
             {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
             {"width_wise_output_block_start_index", width_wise_output_block_start_index},
             {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block}});

        if (full_present) {
            AddRuntimeArgsForNode(
                compute_full_run.runtime_arg_values, core, {{"per_core_block_cnt", num_input_blocks_to_process}});
        }

        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    // Cliff core (interleaved input only).
    std::vector<CoreCoord> cliff_cores = corerange_to_cores(cliff_compute_core_range, std::nullopt, is_row_major);
    if (!cliff_cores.empty()) {
        CoreCoord cliff_core = cliff_cores[0];
        uint32_t height_wise_input_block_start_index = full_cores.size() * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = 0;
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        uint32_t num_input_blocks_to_process = num_input_blocks_per_cliff_core;

        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            cliff_core,
            {{"num_input_blocks_to_process", num_input_blocks_to_process},
             {"height_wise_input_block_start_index", height_wise_input_block_start_index},
             {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
             {"width_wise_output_block_start_index", width_wise_output_block_start_index},
             {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block}});

        // Cliff core only exists for interleaved input.
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        set_reader_args(cliff_core, 0u, num_input_blocks_to_process, num_tiles_to_read);

        if (cliff_present) {
            AddRuntimeArgsForNode(
                compute_cliff_run.runtime_arg_values,
                cliff_core,
                {{"per_core_block_cnt", num_input_blocks_to_process}});
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    if (full_present) {
        run_args.kernel_run_args.push_back(compute_full_run);
    }
    if (cliff_present) {
        run_args.kernel_run_args.push_back(compute_cliff_run);
    }
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
