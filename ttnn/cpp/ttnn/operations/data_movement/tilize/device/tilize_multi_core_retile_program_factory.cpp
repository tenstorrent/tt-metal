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
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

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

    TT_FATAL(a.is_allocated(), "Input buffer should be allocated on device!");
    TT_FATAL(output.is_allocated(), "Output buffer should be allocated on device!");

    // Each outer slice (the product of the dims above the last two) is padded to its own tile height on each
    // side: the input to a multiple of in_tile_height, the output to a multiple of out_tile_height. So the work
    // is split per slice, never over one flattened matrix, whose rows line up with the output's only when there
    // is a single slice or the height is a multiple of both tile heights.
    const auto& input_padded_shape = a.padded_shape();
    const auto& output_padded_shape = output.padded_shape();
    const uint32_t tensor_width = input_padded_shape[-1];
    TT_FATAL(
        output_padded_shape[-1] == tensor_width,
        "Retile keeps the padded width; got input {} and output {}",
        tensor_width,
        output_padded_shape[-1]);
    TT_FATAL(tensor_width % in_tile_width == 0, "Tensor width must be divisible by input tile width");
    const uint32_t input_slice_height = input_padded_shape.rank() >= 2 ? input_padded_shape[-2] : 1;
    const uint32_t output_slice_height = output_padded_shape.rank() >= 2 ? output_padded_shape[-2] : 1;
    TT_FATAL(input_slice_height % in_tile_height == 0, "Tensor height must be divisible by input tile height");
    TT_FATAL(output_slice_height % out_tile_height == 0, "Tensor height must be divisible by output tile height");
    const uint32_t num_slices = a.physical_volume() / (input_slice_height * tensor_width);
    TT_FATAL(
        output.physical_volume() / (output_slice_height * tensor_width) == num_slices,
        "Retile input and output must have the same number of outer slices");

    const uint32_t tiles_per_block = tensor_width / in_tile_width;
    // Tile-rows of one slice on each side.
    const uint32_t slice_in_rows = input_slice_height / in_tile_height;
    const uint32_t slice_out_rows = output_slice_height / out_tile_height;

    const uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    // A unit is one tile-row of the taller tile: an input tile-row when shrinking, an output tile-row when
    // growing, so each unit maps to whole tile-rows on both sides. Within a slice, unit p reads
    // min(ratio, slice_in_rows - p * ratio) input tile-rows when growing (the rest is height padding the compute
    // kernel zero-fills) and writes min(ratio, slice_out_rows - p * ratio) output tile-rows when shrinking (the
    // rest is the input's own height padding, dropped); either count stops at 0. When growing, an input padded
    // past the output's height gets units that read its trailing rows and write nothing. The compute kernel
    // derives the same counts from the unit's position in its slice.
    TT_FATAL(
        !shrink || slice_out_rows <= slice_in_rows * ratio,
        "Retile output must not need more rows than the input holds; got {} input and {} output tile-rows per slice",
        slice_in_rows,
        slice_out_rows);
    const uint32_t units_per_slice =
        shrink ? slice_in_rows : std::max(slice_out_rows, tt::div_up(slice_in_rows, ratio));
    const uint32_t num_split_units = num_slices * units_per_slice;

    // Tile-rows read (input) and written (output) by the units before global unit `unit`.
    auto input_rows_before = [&](uint32_t unit) {
        const uint32_t pos = unit % units_per_slice;
        return (unit / units_per_slice) * slice_in_rows + (shrink ? pos : std::min(slice_in_rows, pos * ratio));
    };
    auto output_rows_before = [&](uint32_t unit) {
        const uint32_t pos = unit % units_per_slice;
        return (unit / units_per_slice) * slice_out_rows + (shrink ? std::min(slice_out_rows, pos * ratio) : std::min(slice_out_rows, pos));
    };

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
    // MID_VIEW's num_entries divides the shared allocation by the output-shaped tile size; this is
    // exact for every dtype the op admits (tile size scales with tile height), but a block-float
    // input would break it (fixed exponent header). Block-float inputs are rejected upstream
    // (validate_on_program_cache_miss dtype whitelist), so this cannot fire today; the guard makes
    // the assumption explicit for anyone widening the dtype list later.
    TT_FATAL(
        mid_total_size % out_tile_size_input_fmt == 0,
        "retile intermediate ({} B) must hold a whole number of {}-byte output-shaped tiles",
        mid_total_size,
        out_tile_size_input_fmt);
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
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
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
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // Compute: retile. MID / MID_VIEW are self-loops (compute is the only toucher; MID_VIEW's read
    // cursor is hand-driven, it has no FIFO producer). Same CTAs on the full and cliff instances;
    // only the per-node RTAs differ.
    auto make_compute = [&](const KernelSpecName& id) {
        ComputeHardwareConfig compute_cfg;
        compute_cfg.enable_32_bit_dest = fp32_llk_acc;
        if (fp32_llk_acc) {
            compute_cfg.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest);
            compute_cfg.unpack_modes.emplace(MID_DFB, UnpackMode::UnpackToDest);
            compute_cfg.unpack_modes.emplace(MID_VIEW_DFB, UnpackMode::UnpackToDest);
        }
        // Quasar gets only the common fields set above; WH/BH use compute_cfg as is.
        ComputeHardwareConfig compute_hw = compute_cfg;
        if (device->arch() == tt::ARCH::QUASAR) {
            ComputeHardwareConfig compute_cfg_gen2;
            compute_cfg_gen2.enable_32_bit_dest = compute_cfg.enable_32_bit_dest;
            compute_cfg_gen2.unpack_modes = compute_cfg.unpack_modes;  // TODO(#52269): copied from WH/BH
            compute_hw = compute_cfg_gen2;
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
                {.runtime_arg_names =
                     {"num_units", "first_unit_in_slice", "units_per_slice", "slice_in_rows", "slice_out_rows"}},
            .hw_config = std::move(compute_hw),
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
    // Walk the grid in the order split_blocks_for_tilize used (it picks the full/cliff cores by
    // walking available_grid), so cores[0..ncores_full) are the full cores and cores[ncores_full] is
    // the cliff core it chose. Iterating all_cores (a sorted set) instead would permute per-core args
    // relative to the split whenever sub_core_grids spans more than one range and a cliff exists.
    const auto& cores = corerange_to_cores(available_grid);

    // A core's units are contiguous across slice boundaries, and so are the tile-rows they read and write,
    // because slices are stored one after another on both sides.
    uint32_t first_unit = 0;
    auto add_core_args = [&](const CoreCoord& core, uint32_t num_units, KernelRunArgs& compute_ra) {
        const uint32_t end_unit = first_unit + num_units;
        const uint32_t input_row_start = input_rows_before(first_unit);
        const uint32_t output_row_start = output_rows_before(first_unit);
        const uint32_t num_input_tiles = (input_rows_before(end_unit) - input_row_start) * tiles_per_block;
        const uint32_t num_output_tiles = (output_rows_before(end_unit) - output_row_start) * tiles_per_block;

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_tiles", num_input_tiles}, {"start_id", input_row_start * tiles_per_block}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_pages", num_output_tiles}, {"start_id", output_row_start * tiles_per_block}});
        AddRuntimeArgsForNode(
            compute_ra.runtime_arg_values,
            core,
            {{"num_units", num_units},
             {"first_unit_in_slice", first_unit % units_per_slice},
             {"units_per_slice", units_per_slice},
             {"slice_in_rows", slice_in_rows},
             {"slice_out_rows", slice_out_rows}});
        first_unit = end_unit;
    };

    for (uint32_t i = 0; i < ncores_full; ++i) {
        add_core_args(cores[i], nblocks_per_core, compute_full_ra);
    }
    if (has_cliff) {
        add_core_args(cores[ncores_full], nblocks_per_core_cliff, compute_cliff_ra);
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

ProgramRunArgs TilizeMultiCoreRetileProgramFactory::override_runtime_arguments(
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
