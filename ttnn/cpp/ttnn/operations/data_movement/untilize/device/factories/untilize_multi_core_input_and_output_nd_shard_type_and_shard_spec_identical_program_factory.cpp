// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md alongside the untilize op directory.

#include "untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace untilize_nd_shard_spec_identical_factory {

constexpr const char* PROGRAM_ID = "untilize_nd_shard_spec_identical";
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

}  // namespace untilize_nd_shard_spec_identical_factory

ttnn::device_operation::ProgramArtifacts
UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::create_program_spec(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    using namespace untilize_nd_shard_spec_identical_factory;
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t shard_width = nd_shard_spec.shard_shape[-1];
    const CoreRangeSet& grid = nd_shard_spec.grid;
    ShardOrientation orientation = nd_shard_spec.orientation;
    NodeRangeSet all_nodes = to_node_range_set(grid);

    uint32_t shard_vol = nd_shard_spec.shard_shape.volume();
    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_shard = (shard_height / tile_height) * (shard_vol / (shard_height * shard_width));
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_shard;

    const auto& distribution_spec = a.buffer()->buffer_distribution_spec().value();
    uint32_t total_shards = distribution_spec.num_shards();
    uint32_t num_cores = grid.num_cores();
    const auto& groups = distribution_spec.core_groups();
    uint32_t num_shards_per_core = groups.num_shards_per_core_in_group_1;
    log_debug(
        tt::LogOp,
        "ND sharding: total_shards={}, cores={}, base_shards_per_core={}",
        total_shards,
        num_cores,
        num_shards_per_core);

    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard * num_shards_per_core,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard * num_shards_per_core,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT,
    };

    KernelSpec reader{
        .unique_id = READER,
        .source =
            KernelSpec::SourceFilePath{
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "shard",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        // Validator requires a kernel to claim each TensorParameter (program_spec.cpp:421).
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles_per_core"}},
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
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                                             "writer_unary_sharded_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "output",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_units"}},
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
        .target_nodes = all_nodes,
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

    auto cores = corerange_to_cores(grid, std::nullopt, orientation == ShardOrientation::ROW_MAJOR);
    auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& mapped_cores = page_mapping.all_cores;
    for (const auto& core : cores) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_blocks_to_process = 0;
        uint32_t num_tiles_to_process = 0;
        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const size_t num_shards_on_core = distribution_spec.num_shards_per_core(core_idx);
            num_blocks_to_process = num_blocks_per_shard * num_shards_on_core;
            num_tiles_to_process = num_tiles_per_block * num_blocks_to_process;
        }
        reader_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"num_tiles_per_core", num_tiles_to_process}},
        });
        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"num_units", num_tiles_to_process}},
        });
        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"per_core_block_cnt", num_blocks_to_process}},
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
