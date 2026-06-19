// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_nd_sharded_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include "ttnn/operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

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
    const auto& shard_shape = nd_shard_spec.shard_shape;
    size_t num_planes_per_shard = 1;
    if (shard_shape.rank() > 2) {
        for (int i = 0; i < static_cast<int>(shard_shape.rank()) - 2; ++i) {
            num_planes_per_shard *= shard_shape[i];
        }
    }
    uint32_t num_blocks_per_shard = num_planes_per_shard * num_blocks_per_shard_plane;
    uint32_t num_input_blocks_per_full_core = groups.num_shards_per_core_in_group_1 * num_blocks_per_shard;

    // Input/Output CBs (regular FIFOs; ND-sharded input is staged through c_0, not borrowed).
    uint32_t input_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    uint32_t output_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;

    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader (sharded ND input): bound to INPUT for accessor_src.shard_pages().
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                                        "reader_unary_nd_sharded_blocks_metal2.cpp"},
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ------------------------------------------------------------------------
    // Writer: bound to OUTPUT (write target) and INPUT (accessor_src.shard_pages()).
    // The per-block output/input tensor shapes are passed as common runtime varargs (read
    // positionally via get_common_vararg), matching the legacy get_common_arg_val loop.
    // Legacy dead CTAs output_stick_size (slot 1) and input_single_tile_size (slot 8) are dropped.
    // ------------------------------------------------------------------------
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

    std::vector<uint32_t> writer_common_runtime_args;
    for (const auto dim : output.padded_shape()) {
        writer_common_runtime_args.push_back(dim);
    }
    for (const auto dim : input.padded_shape()) {
        writer_common_runtime_args.push_back(dim);
    }

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings =
            {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
             {.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_output_blocks_across_width", output_num_blocks_across_width},
             {"output_element_size", output_element_size},
             {"num_cols_per_input_block", num_cols_per_input_block},
             {"num_cols_per_output_block", num_cols_per_output_block},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores},
             {"num_tiles_per_input_row", num_tiles_per_input_row},
             {"num_tiles_per_output_row", num_tiles_per_output_row},
             {"tile_width", tile_width},
             {"output_tensor_width", output_tensor_width},
             {"output_tensor_height", output_tensor_height},
             {"tensor_rank", static_cast<uint32_t>(input.padded_shape().rank())}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
        .advanced_options =
            KernelAdvancedOptions{
                .num_common_runtime_varargs = static_cast<uint32_t>(writer_common_runtime_args.size())},
    };

    // ------------------------------------------------------------------------
    // Compute: untilize with a per-core variable block count (idle cores get 0 blocks).
    // ------------------------------------------------------------------------
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 || input.dtype() == DataType::FLOAT32) {
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
        .compiler_options = {.defines = compute_defines},
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
    // Per-core runtime args. start_shard_id is the core's index in ordered_cores_with_data; the
    // compute block count per core is derived from the page mapping (non-padding blocks).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};
    reader_run.runtime_arg_values.reserve(ordered_cores_with_data.size());
    writer_run.runtime_arg_values.reserve(ordered_cores_with_data.size());
    compute_run.runtime_arg_values.reserve(ordered_cores_with_data.size());

    const auto& mapped_cores = page_mapping.all_cores;
    uint32_t start_shard_id = 0;
    for (auto core : ordered_cores_with_data) {
        const NodeCoord node = core;
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

        reader_run.runtime_arg_values.push_back({node, {{"start_shard_id", start_shard_id}}});
        writer_run.runtime_arg_values.push_back({node, {{"start_shard_id", start_shard_id}}});
        compute_run.runtime_arg_values.push_back({node, {{"per_core_block_cnt", num_input_blocks_to_process}}});
        start_shard_id++;
    }
    writer_run.common_runtime_arg_values = {};
    writer_run.advanced_options.common_runtime_varargs = std::move(writer_common_runtime_args);

    WorkUnitSpec wu{
        .name = "untilize_with_unpadding_multi_core_nd_sharded",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = compute_core_range,
    };

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_nd_sharded",
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
