// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::cached_program_t
UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    TT_FATAL(output.is_allocated(), "Output buffer should be allocated on device!");
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

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

    auto distribution_spec = BufferDistributionSpec::from_shard_spec(
        a.padded_shape(),
        nd_shard_spec.shard_shape,
        tile_shape,
        nd_shard_spec.grid,
        nd_shard_spec.orientation,
        nd_shard_spec.shard_distribution_strategy);

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

    // Input CB
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        grid,
        input_single_tile_size,
        num_tiles_per_shard * num_shards_per_core,
        input_cb_data_format,
        src0_buffer);

    // Output CB
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        grid,
        output_single_tile_size,
        num_tiles_per_shard * num_shards_per_core,
        output_cb_data_format,
        dst_buffer);

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};

    // Reader kernel
    KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};

    // Writer kernel
    KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Compute compile-time args
    std::vector<uint32_t> compute_compile_time_args = {num_tiles_per_block, src0_cb_index, output_cb_index};

    // Compute kernel
    std::map<std::string, std::string> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel;
    if (!use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");

        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "untilize_variable_num_blocks.cpp");

        unpack_to_dest_mode[src0_cb_index] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");

        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "pack_untilize_variable_num_blocks.cpp");
    }
    KernelHandle untilize_kernel_id = CreateKernel(
        program,
        compute_kernel,
        grid,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_kernel_defines});

    // Run-time args
    auto cores = corerange_to_cores(grid, std::nullopt, orientation == ShardOrientation::ROW_MAJOR);
    auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& mapped_cores = page_mapping.all_cores;
    for (auto core : cores) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_blocks_to_process = 0;
        uint32_t num_tiles_to_process = 0;
        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const size_t num_shards_on_core = distribution_spec.num_shards_per_core(core_idx);
            num_blocks_to_process = num_blocks_per_shard * num_shards_on_core;
            num_tiles_to_process = num_tiles_per_block * num_blocks_to_process;
        }

        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {num_tiles_to_process};

        // Writer run-time args
        std::vector<uint32_t> writer_run_time_args = {num_tiles_to_process};

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_blocks_to_process};
        tt::tt_metal::SetRuntimeArgs(program, untilize_kernel_id, core, compute_run_time_args);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);
    }

    return cached_program_t{std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output}};
}

void UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::override_runtime_arguments(
    UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& cb_output = cached_program.shared_variables.cb_output;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
}
}  // namespace ttnn::prim
