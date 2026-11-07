// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const DFBSpecName SRC0{"src0"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    constexpr const char* READER_SRC =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp";
    constexpr const char* WRITER_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp";
    constexpr const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
        "untilize_variable_num_blocks_metal2.cpp";

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t shard_width = nd_shard_spec.shard_shape[-1];
    CoreRangeSet grid = nd_shard_spec.grid;
    ShardOrientation orientation = nd_shard_spec.orientation;

    uint32_t shard_vol = nd_shard_spec.shard_shape.volume();

    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_shard = (shard_height / tile_height) * (shard_vol / (shard_height * shard_width));
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_shard;

    const auto& distribution_spec =
        a.mesh_buffer().device_local_config().sharding_args.buffer_distribution_spec().value();

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

    // Sharded input/output DFBs — borrowed-memory, backed by the input/output tensor buffers
    // (zero-copy). The framework refreshes their backing L1 address from the tensor arguments.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard * num_shards_per_core,
        .data_format_metadata = input_data_format,
        .borrowed_from = INPUT,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard * num_shards_per_core,
        .data_format_metadata = output_data_format,
        .borrowed_from = OUTPUT,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{READER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (operations::data_movement::needs_fp32_unpack_to_dest(fp32_dest_acc_en, a.dtype())) {
        compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{COMPUTE_SRC},
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC0,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_block}},
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
        .hw_config = std::move(compute_cfg),
    };

    ProgramSpec spec{
        .name = "untilize_multi_core_identical_nd_shard",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = grid,
        }},
    };

    // Run-time args
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};
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

        // Sharded readers/writers consume only the (per-launch) tile count; no buffer address is
        // needed because the borrowed DFB itself carries the tensor binding.
        AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_to_process}});
        AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"per_core_block_cnt", num_blocks_to_process}});
        AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"num_units", num_tiles_to_process}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
