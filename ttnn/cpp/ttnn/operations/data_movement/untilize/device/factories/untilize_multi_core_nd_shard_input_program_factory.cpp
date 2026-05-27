// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md alongside the untilize op directory.

#include "untilize_multi_core_nd_shard_input_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace untilize_nd_shard_input_factory {

constexpr const char* PROGRAM_ID = "untilize_nd_shard_input";
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

}  // namespace untilize_nd_shard_input_factory

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreNDShardInputProgramFactory::create_program_spec(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    using namespace untilize_nd_shard_input_factory;
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t output_tensor_width = output.padded_shape()[-1];
    uint32_t output_tensor_height = output.physical_volume() / output_tensor_width;
    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_input_row = tensor_width / tile_width;

    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t input_shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t input_shard_width = nd_shard_spec.shard_shape[-1];

    const auto distribution_spec = a.buffer()->buffer_distribution_spec().value();
    uint32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& groups = distribution_spec.core_groups();
    const auto& ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(a);
    uint32_t num_compute_cores = ordered_cores_with_data.size();
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));
    NodeRangeSet compute_nodes = to_node_range_set(compute_core_range);

    uint32_t num_tiles_per_input_block = input_shard_width / tile_width;
    uint32_t num_blocks_per_shard_plane = input_shard_height / tile_height;
    const auto& shard_shape = nd_shard_spec.shard_shape;
    size_t num_planes_per_shard = 1;
    if (shard_shape.rank() > 2) {
        for (int i = 0; i < static_cast<int>(shard_shape.rank()) - 2; ++i) {
            num_planes_per_shard *= shard_shape[i];
        }
    }
    uint32_t num_blocks_per_shard = num_planes_per_shard * num_blocks_per_shard_plane;
    uint32_t num_input_blocks_per_full_core = groups.num_shards_per_core_in_group_1 * num_blocks_per_shard;

    uint32_t input_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    uint32_t output_cb_num_tiles = input_cb_num_tiles;

    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                                             "reader_unary_nd_sharded_blocks_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "input",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .compile_time_arg_bindings =
            {
                {"num_tiles_per_input_block", num_tiles_per_input_block},
                {"num_shards", num_shards},
                {"num_cores", num_compute_cores},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"start_shard_id"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    // Writer has TWO tensor bindings — output (writes go here) and input (reads
    // ND-shard distribution info via accessor_src.shard_pages()).
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width = output_tensor_width;
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(output_tensor_width, output_page_width);
    }
    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                                             "writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings =
            {
                {.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
                {.tensor_parameter_name = INPUT, .accessor_name = "input"},
            },
        .compile_time_arg_bindings =
            {
                {"tile_height", tile_height},
                {"num_tiles_per_input_block", num_tiles_per_input_block},
                {"num_output_blocks_across_width", output_num_blocks_across_width},
                {"output_element_size", output_element_size},
                {"num_cols_per_input_block", num_cols_per_input_block},
                {"num_cols_per_output_block", num_cols_per_output_block},
                {"num_shards", num_shards},
                {"num_cores", num_compute_cores},
                {"num_tiles_per_row", num_tiles_per_input_row},
                {"tile_width", tile_width},
                {"output_tensor_width", output_tensor_width},
                {"output_tensor_height", output_tensor_height},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"start_shard_id"}},
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

    using KernelRunParams = ProgramRunParams::KernelRunParams;
    KernelRunParams reader_rp{.kernel_spec_name = READER};
    KernelRunParams writer_rp{.kernel_spec_name = WRITER};
    KernelRunParams compute_rp{.kernel_spec_name = COMPUTE};

    const auto& mapped_cores = page_mapping.all_cores;
    uint32_t start_shard_id = 0;
    for (const auto& core : ordered_cores_with_data) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_input_blocks_to_process = 0;

        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const auto& host_page_indices = page_mapping.core_host_page_indices[core_idx];

            uint32_t page_offset = 0;
            const uint32_t total_pages = host_page_indices.size();
            while (page_offset < total_pages) {
                if (host_page_indices[page_offset] != UncompressedBufferPageMapping::PADDING) {
                    num_input_blocks_to_process++;
                } else if (page_offset == 0) {
                    break;
                }
                page_offset += num_tiles_per_input_block;
            }
        }

        reader_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"start_shard_id", start_shard_id}},
        });
        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"start_shard_id", start_shard_id}},
        });
        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"per_core_block_cnt", num_input_blocks_to_process}},
        });
        start_shard_id++;
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
