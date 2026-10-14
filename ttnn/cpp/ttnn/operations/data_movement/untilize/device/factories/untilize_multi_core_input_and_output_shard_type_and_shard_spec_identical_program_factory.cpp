// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory::create_program_artifacts(
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
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp";

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    ShardSpec shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];

    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_core = shard_height / tile_height;
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_core;

    // Sharded input/output DFBs — borrowed-memory, backed by the input/output tensor buffers
    // (zero-copy). The framework refreshes their backing L1 address from the tensor arguments.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = input_data_format,
        .borrowed_from = INPUT,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = output_data_format,
        .borrowed_from = OUTPUT,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{READER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{COMPUTE_SRC},
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", num_blocks_per_core}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = std::move(compute_cfg),
    };

    ProgramSpec spec{
        .name = "untilize_multi_core_identical_shard",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = shard_spec.grid,
        }},
    };

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    auto cores =
        corerange_to_cores(shard_spec.grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    for (const auto& core : cores) {
        uint32_t num_tiles_to_read = num_tiles_per_block * num_blocks_per_core;
        uint32_t num_tiles_to_write = num_tiles_per_block * num_blocks_per_core;
        AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_to_read}});
        AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"num_units", num_tiles_to_write}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
