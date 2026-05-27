// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md alongside the untilize op directory.

#include "untilize_multi_core_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace untilize_multi_core_factory {

constexpr const char* PROGRAM_ID = "untilize_multi_core";
constexpr const char* READER = "reader";
constexpr const char* WRITER = "writer";
constexpr const char* COMPUTE = "compute";
constexpr const char* SRC_DFB = "src_dfb";
constexpr const char* OUT_DFB = "out_dfb";
constexpr const char* INPUT = "input";
constexpr const char* OUTPUT = "output";
constexpr const char* MAIN_WU = "main";

NodeRangeSet to_node_range_set(const CoreRangeSet& crs) {
    std::vector<NodeRange> ranges;
    ranges.reserve(crs.ranges().size());
    for (const auto& cr : crs.ranges()) {
        ranges.emplace_back(NodeCoord{cr.start_coord.x, cr.start_coord.y}, NodeCoord{cr.end_coord.x, cr.end_coord.y});
    }
    return NodeRangeSet(std::move(ranges));
}

NodeCoord to_node_coord(const CoreCoord& c) { return NodeCoord{c.x, c.y}; }

}  // namespace untilize_multi_core_factory

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreProgramFactory::create_program_spec(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& output) {
    using namespace untilize_multi_core_factory;
    const auto& a = tensor_args.input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t tensor_height = a.physical_volume() / tensor_width;
    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    bool input_is_sharded = a.is_sharded();
    std::vector<CoreCoord> ordered_cores_with_data;

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = a.device()->compute_with_storage_grid_size();
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
    bool use_block_reader = input_is_sharded && has_uneven_sharding;

    NodeRangeSet compute_nodes = to_node_range_set(compute_core_range);

    // ---- DFB specs ----
    // SRC DFB: backed (borrowed_from INPUT) only in the even-sharded zero-copy path.
    uint32_t input_cb_num_tiles;
    if (input_is_sharded && !use_block_reader) {
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        input_cb_num_tiles =
            (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    }
    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    if (input_is_sharded && !use_block_reader) {
        src_dfb.borrowed_from = INPUT;
    }

    uint32_t output_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Reader: 3 dispatch paths share one KernelSpec slot ----
    KernelSpec reader;
    reader.unique_id = READER;
    reader.dfb_bindings = {{
        .dfb_spec_name = SRC_DFB,
        .local_accessor_name = "input",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
    }};
    reader.config_spec = DataMovementConfiguration{
        .gen1_data_movement_config =
            DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
    };
    if (use_block_reader) {
        reader.source = KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_metal2.cpp"};
        reader.tensor_bindings = {{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.compile_time_arg_bindings = {{"tiles_per_block", num_tiles_per_input_block}};
        reader.runtime_arguments_schema = {.named_runtime_args = {"num_blocks"}};
    } else if (input_is_sharded) {
        // sharded backed-CB path: same as Phase 1 sharded reader (PRODUCER signal only)
        reader.source = KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"};
        // No tensor binding needed — the DFB itself is borrowed_from INPUT.
        // Reader kernel reads `args::num_tiles_per_core` from RTAs.
        // Match the binding name expected by reader_unary_sharded_metal2.cpp (dfb::shard).
        reader.dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "shard",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }};
        reader.runtime_arguments_schema = {.named_runtime_args = {"num_tiles_per_core"}};
    } else {
        reader.source = KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id_metal2.cpp"};
        reader.tensor_bindings = {{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.runtime_arguments_schema = {.named_runtime_args = {"num_tiles", "start_page_id"}};
    }

    // ---- Writer ----
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

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                             "writer_unary_stick_layout_split_rows_multi_core_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "out",
        }},
        .compile_time_arg_bindings =
            {
                {"tile_height", tile_height},
                {"num_tiles_per_input_block", num_tiles_per_input_block},
                {"num_output_blocks_across_width", output_num_blocks_across_width},
                {"output_element_size", output_element_size},
                {"num_cols_per_input_block", num_cols_per_input_block},
                {"num_cols_per_output_block", num_cols_per_output_block},
            },
        .runtime_arguments_schema =
            {.named_runtime_args =
                 {
                     "num_input_blocks_to_process",
                     "height_wise_input_block_start_index",
                     "num_unpadded_cols_per_input_block",
                     "width_wise_output_block_start_index",
                     "num_cols_already_processed_in_first_output_block",
                 }},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    std::vector<ComputeConfiguration::UnpackToDestModeEntry> unpack_to_dest_mode;
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode.emplace_back(SRC_DFB, UnpackToDestMode::UnpackToDestFp32);
    }

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace_back("DST_ACCUM_MODE", "1");
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            KernelSpec::SourceFilePath{
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp"},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {
                {.dfb_spec_name = SRC_DFB,
                 .local_accessor_name = "src",
                 .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "dst",
                 .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER},
            },
        .compile_time_arg_bindings = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
        .runtime_arguments_schema = {.named_runtime_args = {"per_core_block_cnt"}},
        .config_spec =
            ComputeConfiguration{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            },
    };

    WorkUnitSpec main_wu{
        .unique_id = MAIN_WU,
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = compute_nodes,
    };

    ProgramSpec spec{
        .program_id = PROGRAM_ID,
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(src_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = a.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {std::move(main_wu)},
    };

    // ---- Run params ----
    using KernelRunParams = ProgramRunParams::KernelRunParams;
    KernelRunParams reader_rp{.kernel_spec_name = READER};
    KernelRunParams writer_rp{.kernel_spec_name = WRITER};
    KernelRunParams compute_rp{.kernel_spec_name = COMPUTE};

    uint32_t tile_start_index = 0;
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = input_is_sharded
                                            ? ordered_cores_with_data
                                            : corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);
    for (uint32_t i = 0; i < full_cores.size(); ++i) {
        CoreCoord core = full_cores[i];
        uint32_t height_wise_input_block_start_index =
            (i / num_input_blocks_across_width) * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = i % num_input_blocks_across_width;

        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        if (input_is_sharded) {
            bool is_last_input_shard_in_row = width_wise_input_block_index == num_input_blocks_across_width - 1;
            if (is_last_input_shard_in_row) {
                uint32_t input_shard_width_ = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, input_shard_width_) - tensor_width);
            }
        }

        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t input_shard_height_ = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, input_shard_height_);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process =
                    num_input_blocks_per_full_core -
                    (tt::round_up(tensor_height, input_shard_height_) - tensor_height) / tile_height;
            }
        }

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        if (use_block_reader) {
            reader_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"num_blocks", num_input_blocks_to_process}},
            });
        } else if (input_is_sharded) {
            reader_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"num_tiles_per_core", num_tiles_to_read}},
            });
        } else {
            reader_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args =
                    {
                        {"num_tiles", num_tiles_to_read},
                        {"start_page_id", tile_start_index},
                    },
            });
        }

        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args =
                {
                    {"num_input_blocks_to_process", num_input_blocks_to_process},
                    {"height_wise_input_block_start_index", height_wise_input_block_start_index},
                    {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
                    {"width_wise_output_block_start_index", width_wise_output_block_start_index},
                    {"num_cols_already_processed_in_first_output_block",
                     num_cols_already_processed_in_first_output_block},
                },
        });

        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"per_core_block_cnt", num_input_blocks_to_process}},
        });

        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    // Cliff core(s) — interleaved input only.
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

        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(cliff_core),
            .args =
                {
                    {"num_input_blocks_to_process", num_input_blocks_to_process},
                    {"height_wise_input_block_start_index", height_wise_input_block_start_index},
                    {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
                    {"width_wise_output_block_start_index", width_wise_output_block_start_index},
                    {"num_cols_already_processed_in_first_output_block",
                     num_cols_already_processed_in_first_output_block},
                },
        });

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        // Cliff path is interleaved-only.
        reader_rp.named_runtime_args.push_back({
            .node = to_node_coord(cliff_core),
            .args =
                {
                    {"num_tiles", num_tiles_to_read},
                    {"start_page_id", tile_start_index},
                },
        });
        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(cliff_core),
            .args = {{"per_core_block_cnt", num_input_blocks_to_process}},
        });
    }

    ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_rp));
    run_params.kernel_run_params.push_back(std::move(writer_rp));
    run_params.kernel_run_params.push_back(std::move(compute_rp));
    run_params.tensor_args = {
        {.tensor_parameter_name = INPUT, .tensor = std::cref(a.mesh_tensor())},
        {.tensor_parameter_name = OUTPUT, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
