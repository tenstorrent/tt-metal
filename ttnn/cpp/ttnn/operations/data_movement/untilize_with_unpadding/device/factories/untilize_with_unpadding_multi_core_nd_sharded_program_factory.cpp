// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "untilize_with_unpadding_multi_core_nd_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName ND_READER{"nd_reader"};
const KernelSpecName ND_WRITER{"nd_writer"};
const KernelSpecName ND_COMPUTE{"nd_compute"};
const DFBSpecName ND_IN{"nd_in"};
const DFBSpecName ND_OUT{"nd_out"};
const TensorParamName ND_INPUT{"nd_input"};
const TensorParamName ND_OUTPUT{"nd_output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t tensor_width = input.padded_shape()[-1];
    uint32_t output_tensor_width = output.padded_shape()[-1];
    uint32_t output_tensor_height = output.padded_shape()[-2];

    const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_input_row = tensor_width / tile_width;
    uint32_t num_tiles_per_output_row = tt::div_up(output_tensor_width, tile_width);

    const auto& nd_shard_spec = input.nd_shard_spec().value();
    uint32_t input_shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t input_shard_width = nd_shard_spec.shard_shape[-1];

    const auto distribution_spec = input.buffer()->buffer_distribution_spec().value();

    uint32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& groups = distribution_spec.core_groups();
    const auto& ordered_cores_with_data = distribution_spec.cores_with_data();
    uint32_t num_compute_cores = ordered_cores_with_data.size();
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    uint32_t num_tiles_per_input_block = input_shard_width / tile_width;
    uint32_t num_blocks_per_shard_plane =
        input_shard_height /
        tile_height;  // Note: a "shard plane" here refers to a 2D plane the size of the last 2 dimensions of the shard.
                      // For example, a shard of shape [b, c, h, w] has b * c planes each of shape [h, w].
    const auto& shard_shape = nd_shard_spec.shard_shape;
    size_t num_planes_per_shard = 1;
    if (shard_shape.rank() > 2) {
        for (int i = 0; i < static_cast<int>(shard_shape.rank()) - 2; ++i) {
            num_planes_per_shard *= shard_shape[i];
        }
    }
    uint32_t num_blocks_per_shard = num_planes_per_shard * num_blocks_per_shard_plane;
    uint32_t num_input_blocks_per_full_core = groups.num_shards_per_core_in_group_1 * num_blocks_per_shard;

    // Input buffer
    uint32_t input_dfb_num_entries;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        input_dfb_num_entries = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        input_dfb_num_entries = num_tiles_per_input_block * 2;
    }
    DataflowBufferSpec in_dfb{
        .unique_id = ND_IN,
        .entry_size = input_single_tile_size,
        .num_entries = input_dfb_num_entries,
        .data_format_metadata = input_dfb_data_format,
    };

    // Output buffer
    uint32_t output_dfb_num_entries;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_dfb_num_entries = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_dfb_num_entries = num_tiles_per_input_block * 2;
    }
    DataflowBufferSpec out_dfb{
        .unique_id = ND_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_dfb_num_entries,
        .data_format_metadata = output_dfb_data_format,
    };

    // Reader kernel (sharded input)
    KernelSpec reader{
        .unique_id = ND_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_nd_sharded_blocks_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = ND_IN,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = ND_INPUT,
            .accessor_name = "src",
        }},
        .compile_time_args =
            {{"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    // Writer compile-time args
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width =
        output_tensor_width;  // In height-sharded and interleaved cases, the output page is the entire tensor row
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
    uint32_t output_stick_size = num_cols_per_output_block * output_element_size;
    uint32_t tensor_rank = input.padded_shape().rank();

    // Writer kernel. The output and input tensors are both bound: the input accessor is used only
    // for the shard geometry the walk below needs (shard_pages), never to read input data.
    KernelSpec writer{
        .unique_id = ND_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = ND_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings =
            {TensorBinding{
                 .tensor_parameter_name = ND_OUTPUT,
                 .accessor_name = "dst",
             },
             TensorBinding{
                 .tensor_parameter_name = ND_INPUT,
                 .accessor_name = "src",
             }},
        .compile_time_args =
            {{"output_stick_size", output_stick_size},
             {"tile_height", tile_height},
             {"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_output_blocks_across_width", output_num_blocks_across_width},
             {"output_element_size", output_element_size},
             {"num_cols_per_input_block", num_cols_per_input_block},
             {"num_cols_per_output_block", num_cols_per_output_block},
             {"input_single_tile_size", input_single_tile_size},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores},
             {"num_tiles_per_input_row", num_tiles_per_input_row},
             {"num_tiles_per_output_row", num_tiles_per_output_row},
             {"tile_width", tile_width},
             {"output_tensor_width", output_tensor_width},
             {"output_tensor_height", output_tensor_height},
             {"tensor_rank", tensor_rank}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
        // The two shape vectors are a tensor_rank-bounded stream the kernel walks by index, so they
        // ride as common runtime varargs (broadcast to every node) rather than as named args.
        .advanced_options = {.num_common_runtime_varargs = 2 * tensor_rank},
    };

    // Due to tensor squeezing from ND to 4D when the input tensor has rank > 4, the vararg payload
    // will have at most 8 entries.
    AdvancedKernelRunArgs::Varargs writer_common_varargs;
    for (const auto dim : output.padded_shape()) {
        writer_common_varargs.push_back(dim);
    }
    for (const auto dim : input.padded_shape()) {
        writer_common_varargs.push_back(dim);
    }

    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_modes = {{ND_IN, UnpackMode::UnpackToDest}};
    }

    // Compute kernel
    // Note: This condition is always true for sharded input
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 || input.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }
    bool has_compute = !compute_core_range.ranges().empty();
    KernelSpec compute{
        .unique_id = ND_COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "untilize_variable_num_blocks_metal2.cpp",
        .compiler_options =
            {
                .defines = std::move(compute_kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = ND_IN,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = ND_OUT,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
        .hw_config = compute_hw_config,
    };

    // Run-time args
    // Logic for ND sharding makes as few assumptions about page locations as possible. Padded pages will be handled
    // in the writer kernel.
    const auto& mapped_cores = page_mapping.all_cores;

    // Use page_mapping to count non-padding blocks per core
    // page_mapping.core_host_page_indices[core_id] contains host page indices for all device pages on that core,
    // with UncompressedBufferPageMapping::PADDING indicating padding pages
    uint32_t start_shard_id = 0;
    KernelRunArgs reader_run_args{.kernel = ND_READER};
    KernelRunArgs writer_run_args{.kernel = ND_WRITER};
    KernelRunArgs compute_run_args{.kernel = ND_COMPUTE};
    writer_run_args.advanced_options.common_runtime_varargs = std::move(writer_common_varargs);
    for (auto core : ordered_cores_with_data) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_input_blocks_to_process = 0;

        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const auto& host_page_indices = page_mapping.core_host_page_indices[core_idx];

            // Iterate through device pages in blocks of num_tiles_per_input_block.
            uint32_t page_offset = 0;
            const uint32_t total_pages = host_page_indices.size();

            while (page_offset < total_pages) {
                if (host_page_indices[page_offset] != UncompressedBufferPageMapping::PADDING) {
                    num_input_blocks_to_process++;
                } else if (page_offset == 0) {  // First page is PADDING means this core has no shards, no need to
                                                // iterate further. This should never happen, as we are iterating over
                                                // only cores with data.
                    break;
                }
                // Advance by num_tiles_per_input_block
                page_offset += num_tiles_per_input_block;
            }
        }
        // Reader run-time args
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_shard_id", start_shard_id}});

        // Writer run-time args
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"start_shard_id", start_shard_id}});
        start_shard_id++;

        // Compute run-time args
        if (has_compute) {
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"per_core_block_cnt", num_input_blocks_to_process}});
        }
    }

    Group<KernelSpec> kernels = {std::move(reader), std::move(writer)};
    Group<KernelSpecName> work_unit_kernels = {ND_READER, ND_WRITER};
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    if (has_compute) {
        kernels.push_back(std::move(compute));
        work_unit_kernels.push_back(ND_COMPUTE);
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_nd_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = ND_INPUT, .spec = input_mesh_tensor.tensor_spec()},
             TensorParameter{.unique_id = ND_OUTPUT, .spec = output_mesh_tensor.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = std::move(work_unit_kernels),
            .target_nodes = compute_core_range,
        }},
    };

    run_args.tensor_args = {
        {ND_INPUT, input_mesh_tensor},
        {ND_OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}
}  // namespace ttnn::prim
