// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_retile_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreShardedRetileProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    const Tile& input_tile = a.tensor_spec().tile();
    const Tile& output_tile = operation_attributes.tile;

    const uint32_t in_tile_width = input_tile.get_width();
    const uint32_t in_tile_height = input_tile.get_height();
    const uint32_t out_tile_width = output_tile.get_width();
    const uint32_t out_tile_height = output_tile.get_height();

    TT_FATAL(
        in_tile_width == TILE_WIDTH && out_tile_width == TILE_WIDTH,
        "Retile requires tile width {}, got input {} and output {}",
        TILE_WIDTH,
        in_tile_width,
        out_tile_width);
    const bool shrink = in_tile_height >= out_tile_height;
    TT_FATAL(
        shrink ? (in_tile_height % out_tile_height) == 0 : (out_tile_height % in_tile_height) == 0,
        "Retile requires one tile height to divide the other exactly; got {} -> {}",
        in_tile_height,
        out_tile_height);

    TT_FATAL(a.is_sharded(), "Sharded retile program factory requires a sharded input");

    const auto& shard_spec = a.shard_spec().value();
    const uint32_t shard_height = shard_spec.shape[0];
    const uint32_t shard_width = shard_spec.shape[1];
    const CoreRangeSet& all_cores = shard_spec.grid;

    TT_FATAL(
        shard_width % in_tile_width == 0,
        "Sharded retile requires shard width {} divisible by tile width {}",
        shard_width,
        in_tile_width);
    TT_FATAL(
        shard_height % in_tile_height == 0,
        "Sharded retile requires shard height {} divisible by input tile height {}",
        shard_height,
        in_tile_height);
    TT_FATAL(
        shard_height % out_tile_height == 0,
        "Sharded retile requires shard height {} divisible by output tile height {}",
        shard_height,
        out_tile_height);

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_single_tile_size = input_tile.get_tile_size(input_data_format);
    const uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    const uint32_t mid_page_size = input_single_tile_size;
    // The intermediate stays in the input data format (conversion happens on the final pack), so the
    // consumer view sizes an output tile in the input format, not the output format.
    const uint32_t out_tile_size_input_fmt = output_tile.get_tile_size(input_data_format);

    const bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                              output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    TT_FATAL(a.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const bool output_is_interleaved = output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    // A retile leaves element dimensions unchanged, so each core's shard maps to whole tile-rows on
    // both sides; only the tiling of those elements changes. Work is per-core and independent.
    const uint32_t tiles_per_block = shard_width / in_tile_width;
    const uint32_t num_input_tile_rows = shard_height / in_tile_height;
    const uint32_t num_output_tile_rows = shard_height / out_tile_height;
    const uint32_t num_tiles_per_shard_in = num_input_tile_rows * tiles_per_block;
    const uint32_t num_tiles_per_shard_out = num_output_tile_rows * tiles_per_block;

    const uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);
    // One output block occupies `ratio` input tile-rows of RM in the grow case, one otherwise.
    const uint32_t mid_pages_per_out_block = (shrink ? 1u : ratio) * tiles_per_block;

    auto* device = a.device();

    // ---- Metal 2.0 spec resource names (function-local) ----
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName MID_DFB{"mid"};            // input tile geometry (untilize producer)
    const DFBSpecName MID_VIEW_DFB{"mid_view"};  // output tile geometry (tilize consumer), aliases MID
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // Input DFB (tiled, input tile shape) — borrowed from the input shard buffer for zero-copy read.
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard_in,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = input_tile,
        .borrowed_from = INPUT,
    };

    // MID and MID_VIEW are two views over one shared intermediate L1 region (avoids an L1 copy
    // between untilize and tilize). They are separate DFBs because face geometry is fixed per-DFB at
    // program-creation time: MID carries the input tile shape for pack_untilize to write into,
    // MID_VIEW the output tile shape so llk_unpack_tilize reads the correct number of RM rows. The
    // two share one allocation via advanced_options.alias_with, which requires equal total size.
    const uint32_t mid_total_size = 2 * mid_pages_per_out_block * mid_page_size;
    DataflowBufferSpec mid_dfb{
        .unique_id = MID_DFB,
        .entry_size = mid_page_size,
        .num_entries = 2 * mid_pages_per_out_block,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = input_tile,
        .advanced_options = {.alias_with = {MID_VIEW_DFB}},
    };
    DataflowBufferSpec mid_view_dfb{
        .unique_id = MID_VIEW_DFB,
        .entry_size = out_tile_size_input_fmt,
        .num_entries = mid_total_size / out_tile_size_input_fmt,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = output_tile,
        .advanced_options = {.alias_with = {MID_DFB}},
    };

    // Output DFB (tiled, output tile shape):
    //   Sharded output  → borrowed from the output shard buffer (zero-copy write); full shard size.
    //   Interleaved output → local DFB sized to a couple of output tile-rows; writer drains it via
    //     TensorAccessor as the compute kernel produces rows.
    DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_is_interleaved ? (2u * tiles_per_block) : num_tiles_per_shard_out,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };
    if (!output_is_interleaved) {
        output_dfb.borrowed_from = OUTPUT;
    }

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Reader: sharded unary — the input DFB is borrowed memory, so the reader only handshakes.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer: sharded in-place (handshake only) or interleaved scatter (TensorAccessor).
    KernelSpec writer;
    if (output_is_interleaved) {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUTPUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    } else {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUTPUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    }

    // Compute: retile. MID / MID_VIEW are self-loops (compute is the only toucher; MID_VIEW's read
    // cursor is hand-driven, it has no FIFO producer).
    ComputeGen1Config compute_cfg;
    compute_cfg.enable_32_bit_dest = fp32_llk_acc;
    if (fp32_llk_acc) {
        compute_cfg.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest);
        compute_cfg.unpack_modes.emplace(MID_DFB, UnpackMode::UnpackToDest);
        compute_cfg.unpack_modes.emplace(MID_VIEW_DFB, UnpackMode::UnpackToDest);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/retile.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = MID_DFB,
                 .accessor_name = "mid",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = MID_DFB,
                 .accessor_name = "mid",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = MID_VIEW_DFB,
                 .accessor_name = "mid_view",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = MID_VIEW_DFB,
                 .accessor_name = "mid_view",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"tiles_per_block", tiles_per_block},
             {"in_tile_height", in_tile_height},
             {"out_tile_height", out_tile_height},
             {"out_tile_size", out_tile_size_input_fmt},
             {"mid_page_size", mid_page_size}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_input_blocks", "num_real_input_rows", "num_real_output_rows"}},
        .hw_config = ComputeHardwareConfig{compute_cfg},
    };

    ProgramSpec spec{
        .name = "tilize_multi_core_sharded_retile",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {input_dfb, mid_dfb, mid_view_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "sharded_retile",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_cores,
        }},
    };

    // ---- Run args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    KernelRunArgs compute_ra{.kernel = COMPUTE};

    if (output_is_interleaved) {
        // HEIGHT_SHARDED with ROW_MAJOR orientation: each core's shard maps to a contiguous tile
        // range in the output, so start_id = i * num_tiles_per_shard_out.
        const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
        uint32_t tile_start_id = 0;
        for (const auto& core : cores) {
            AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_per_shard_in}});
            AddRuntimeArgsForNode(
                writer_ra.runtime_arg_values,
                core,
                {{"num_pages", num_tiles_per_shard_out}, {"start_id", tile_start_id}});
            tile_start_id += num_tiles_per_shard_out;
        }
    } else {
        for (const auto& core : corerange_to_cores(all_cores)) {
            AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_per_shard_in}});
            AddRuntimeArgsForNode(writer_ra.runtime_arg_values, core, {{"num_units", num_tiles_per_shard_out}});
        }
    }

    // All shards are the same size, so every core does identical work; num_input_blocks is in input
    // tile-rows and all rows are real (no grow-case height padding within a shard). retile.cpp uses
    // num_real_output_rows only as an output-row cap; in the sharded case every output row is real, so
    // num_output_tile_rows is the correct (non-limiting) value.
    for (const auto& core : corerange_to_cores(all_cores)) {
        AddRuntimeArgsForNode(
            compute_ra.runtime_arg_values,
            core,
            {{"num_input_blocks", num_input_tile_rows},
             {"num_real_input_rows", num_input_tile_rows},
             {"num_real_output_rows", num_output_tile_rows}});
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra), std::move(compute_ra)};
    run_args.tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs TilizeMultiCoreShardedRetileProgramFactory::override_runtime_arguments(
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The borrowed input DFB and the output binding (borrowed shard buffer, or interleaved
    // TensorAccessor) both refresh their backing address from the tensor args on a cache hit.
    // (This replaces the legacy slot-0 patch + borrowed-address rebuild.)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramRunArgs params;
    params.tensor_args = {{INPUT, tensor_args.input_tensor.mesh_tensor()}, {OUTPUT, tensor_return_value.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
