// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_retile_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <algorithm>

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

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreRetileProgramFactory::create_program_artifacts(
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
    TT_FATAL(
        !a.is_sharded() && output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Retile program factory currently supports interleaved input/output only");

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t input_single_tile_size = input_tile.get_tile_size(input_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    const uint32_t mid_page_size = input_single_tile_size;
    // The intermediate stays in the input data format (conversion happens on the final pack), so
    // the consumer view sizes an output tile in the input format, not the output format.
    const uint32_t out_tile_size_input_fmt = output_tile.get_tile_size(input_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    TT_FATAL(a.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const auto& padded_shape = a.padded_shape();
    const uint32_t tensor_width = padded_shape[-1];
    const uint32_t tensor_height = std::max(output.physical_volume(), a.physical_volume()) / tensor_width;

    TT_FATAL(tensor_width % in_tile_width == 0, "Tensor width must be divisible by input tile width");
    TT_FATAL(tensor_height % in_tile_height == 0, "Tensor height must be divisible by input tile height");
    TT_FATAL(tensor_height % out_tile_height == 0, "Tensor height must be divisible by output tile height");

    const uint32_t tiles_per_block = tensor_width / in_tile_width;
    const uint32_t num_input_tile_rows = tensor_height / in_tile_height;
    const uint32_t num_output_tile_rows = tensor_height / out_tile_height;

    // In the grow case the padded output can be taller than the real input, so some trailing input
    // tile-rows don't exist in DRAM. Only these real rows are read; the compute kernel zero-fills
    // the rest rather than reading invalid input.
    const uint32_t num_real_input_tile_rows = a.physical_volume() / tensor_width / in_tile_height;
    // Shrink-case dual of the reader clamp: bounds the writer so surplus rows don't OOB past `output`.
    const uint32_t num_real_output_tile_rows = output.physical_volume() / tensor_width / out_tile_height;

    const uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    // Split by whole tile-rows of the taller tile so each core's work maps to whole tile-rows on
    // both sides.
    const uint32_t num_split_units = shrink ? num_input_tile_rows : num_output_tile_rows;

    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    const CoreRangeSet available_grid = operation_attributes.sub_core_grids.value_or(default_grid);
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_split_units);

    // Double-buffer when a core processes more than one block so reader/compute/writer overlap.
    const uint32_t dfb_num_pages_per_block = tiles_per_block;
    const uint32_t dfb_factor = (nblocks_per_core > 1 || nblocks_per_core_cliff > 1) ? 2 : 1;
    const uint32_t src_dfb_tiles = dfb_num_pages_per_block * dfb_factor;
    const uint32_t out_dfb_tiles = dfb_num_pages_per_block * dfb_factor;

    // One output block occupies `ratio` input tile-rows of RM in the grow case, one otherwise.
    const uint32_t mid_pages_per_out_block = (shrink ? 1u : ratio) * tiles_per_block;

    // ---- Metal 2.0 spec resource names (function-local) ----
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName MID_DFB{"mid"};            // input tile geometry (untilize producer)
    const DFBSpecName MID_VIEW_DFB{"mid_view"};  // output tile geometry (tilize consumer), aliases MID
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // Input DFB (tiled, input tile shape) — double-buffered for reader/compute overlap.
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = src_dfb_tiles,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = input_tile,
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

    // Output DFB (tiled, output tile shape) — double-buffered for compute/writer overlap.
    DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_dfb_tiles,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Reader: interleaved tiled pages.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer: interleaved tiled pages.
    KernelSpec writer{
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

    // Compute: retile. MID / MID_VIEW are self-loops (compute is the only toucher; MID_VIEW's read
    // cursor is hand-driven, it has no FIFO producer). Same CTAs on the full and cliff instances;
    // only the per-node RTAs differ.
    auto make_compute = [&](const KernelSpecName& id) {
        ComputeGen1Config compute_cfg;
        compute_cfg.enable_32_bit_dest = fp32_llk_acc;
        if (fp32_llk_acc) {
            compute_cfg.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest);
            compute_cfg.unpack_modes.emplace(MID_DFB, UnpackMode::UnpackToDest);
            compute_cfg.unpack_modes.emplace(MID_VIEW_DFB, UnpackMode::UnpackToDest);
        }
        return KernelSpec{
            .unique_id = id,
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
    };

    const bool has_full = !core_range.ranges().empty();
    const bool has_cliff = !core_range_cliff.empty();

    Group<KernelSpec> kernels;
    kernels.push_back(reader);
    kernels.push_back(writer);
    if (has_full) {
        kernels.push_back(make_compute(COMPUTE_FULL));
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF));
    }

    Group<WorkUnitSpec> work_units;
    if (has_full) {
        work_units.push_back(WorkUnitSpec{
            .name = "retile_full",
            .kernels = {READER, WRITER, COMPUTE_FULL},
            .target_nodes = core_range,
        });
    }
    if (has_cliff) {
        work_units.push_back(WorkUnitSpec{
            .name = "retile_cliff",
            .kernels = {READER, WRITER, COMPUTE_CLIFF},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramSpec spec{
        .name = "tilize_multi_core_retile",
        .kernels = std::move(kernels),
        .dataflow_buffers = {input_dfb, mid_dfb, mid_view_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    // ---- Run args (per-node, in the legacy core order) ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    KernelRunArgs compute_full_ra{.kernel = COMPUTE_FULL};
    KernelRunArgs compute_cliff_ra{.kernel = COMPUTE_CLIFF};

    const uint32_t ncores_full = ncores - (has_cliff ? 1 : 0);
    uint32_t input_tile_start_id = 0;
    uint32_t output_tile_start_id = 0;
    const auto& cores = corerange_to_cores(all_cores);

    auto derive_per_core_counts = [&](uint32_t input_rows, uint32_t output_rows) {
        const uint32_t core_input_row_start = input_tile_start_id / tiles_per_block;
        const uint32_t real_input_rows = core_input_row_start >= num_real_input_tile_rows
                                             ? 0u
                                             : std::min(num_real_input_tile_rows - core_input_row_start, input_rows);
        const uint32_t core_output_row_start = output_tile_start_id / tiles_per_block;
        const uint32_t real_output_rows =
            core_output_row_start >= num_real_output_tile_rows
                ? 0u
                : std::min(num_real_output_tile_rows - core_output_row_start, output_rows);
        // Hang-free invariant: both clamps zero together (both derive from same H via ceil(H/tile_h)).
        TT_FATAL(
            (real_input_rows == 0) == (real_output_rows == 0),
            "tilize retile clamps out of sync: real_in={}, real_out={} (would hang reader)",
            real_input_rows,
            real_output_rows);
        return std::make_pair(real_input_rows, real_output_rows);
    };

    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t input_rows = shrink ? nblocks_per_core : nblocks_per_core * ratio;
        const uint32_t output_rows = shrink ? nblocks_per_core * ratio : nblocks_per_core;
        const uint32_t num_input_blocks = input_rows;
        auto [real_rows, real_output_rows] = derive_per_core_counts(input_rows, output_rows);
        const uint32_t num_input_tiles = real_rows * tiles_per_block;
        const uint32_t num_output_tiles = real_output_rows * tiles_per_block;

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values, core, {{"num_tiles", num_input_tiles}, {"start_id", input_tile_start_id}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values, core, {{"num_pages", num_output_tiles}, {"start_id", output_tile_start_id}});
        AddRuntimeArgsForNode(
            compute_full_ra.runtime_arg_values,
            core,
            {{"num_input_blocks", num_input_blocks},
             {"num_real_input_rows", real_rows},
             {"num_real_output_rows", real_output_rows}});

        input_tile_start_id += input_rows * tiles_per_block;
        output_tile_start_id += output_rows * tiles_per_block;
    }

    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        const uint32_t input_rows = shrink ? nblocks_per_core_cliff : nblocks_per_core_cliff * ratio;
        const uint32_t output_rows = shrink ? nblocks_per_core_cliff * ratio : nblocks_per_core_cliff;
        const uint32_t num_input_blocks = input_rows;
        auto [real_rows, real_output_rows] = derive_per_core_counts(input_rows, output_rows);
        const uint32_t num_input_tiles = real_rows * tiles_per_block;
        const uint32_t num_output_tiles = real_output_rows * tiles_per_block;

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values, core, {{"num_tiles", num_input_tiles}, {"start_id", input_tile_start_id}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values, core, {{"num_pages", num_output_tiles}, {"start_id", output_tile_start_id}});
        AddRuntimeArgsForNode(
            compute_cliff_ra.runtime_arg_values,
            core,
            {{"num_input_blocks", num_input_blocks},
             {"num_real_input_rows", real_rows},
             {"num_real_output_rows", real_output_rows}});
    }

    run_args.kernel_run_args.push_back(std::move(reader_ra));
    run_args.kernel_run_args.push_back(std::move(writer_ra));
    if (has_full) {
        run_args.kernel_run_args.push_back(std::move(compute_full_ra));
    }
    if (has_cliff) {
        run_args.kernel_run_args.push_back(std::move(compute_cliff_ra));
    }
    run_args.tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs TilizeMultiCoreRetileProgramFactory::override_runtime_arguments(
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every shape-derived arg is baked; only the input/output buffer addresses move on a cache hit
    // (this replaces the legacy patch_tilize_kernel_slot0 slot-0 re-point).
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramRunArgs params;
    params.tensor_args = {{INPUT, tensor_args.input_tensor.mesh_tensor()}, {OUTPUT, tensor_return_value.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
