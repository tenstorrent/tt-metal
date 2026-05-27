// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md alongside the untilize op directory.

#include "untilize_single_core_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace untilize_single_core_factory {

constexpr const char* PROGRAM_ID = "untilize_single_core";
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

}  // namespace untilize_single_core_factory

ttnn::device_operation::ProgramArtifacts UntilizeSingleCoreProgramFactory::create_program_spec(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    using namespace untilize_single_core_factory;
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    CoreRange core({0, 0}, {0, 0});
    CoreRangeSet core_ranges{core};
    NodeRangeSet target_nodes = to_node_range_set(core_ranges);

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
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

    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_block,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_block,
        .data_format_metadata = output_cb_data_format,
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                             "reader_unary_start_id_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "input",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles", "start_page_id"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                             "writer_unary_stick_layout_split_rows_single_core_metal2.cpp"},
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
                {"num_blocks_across_height", num_blocks_across_height},
                {"num_output_columns_of_blocks", num_columns_of_blocks},
                {"num_blocks_per_output_column_row", num_blocks_per_column_row},
                {"num_tiles_per_output_block", num_tiles_per_block},
                {"output_single_block_width_size", output_single_block_width_size},
            },
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
        .compile_time_arg_bindings = {{"per_core_block_tile_cnt", num_tiles_per_block}},
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
        .target_nodes = target_nodes,
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

    using KernelRunParams = ProgramRunParams::KernelRunParams;
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;

    KernelRunParams reader_rp{.kernel_spec_name = READER};
    reader_rp.named_runtime_args.push_back({
        .node = to_node_coord(core.start_coord),
        .args =
            {
                {"num_tiles", num_tiles},
                {"start_page_id", std::uint32_t{0}},
            },
    });

    KernelRunParams writer_rp{.kernel_spec_name = WRITER};
    writer_rp.named_runtime_args.push_back({
        .node = to_node_coord(core.start_coord),
        .args = {},  // writer has only CTAs; no RTAs except the buffer (provided via tensor_args).
    });

    KernelRunParams compute_rp{.kernel_spec_name = COMPUTE};
    compute_rp.named_runtime_args.push_back({
        .node = to_node_coord(core.start_coord),
        .args = {{"per_core_block_cnt", num_blocks}},
    });

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
