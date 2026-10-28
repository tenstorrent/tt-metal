// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_program_factory.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Pages in flight per core per read/write round: covers a DRAM round trip at the K3 copy sizes, and holds a core's
// whole share of the 54 KiB tail in one round.
constexpr uint32_t kBatchPages = 8;
// Workers sit on shifted diagonals, one per row and column on each, so their routes spread over the grid's links
// instead of crowding a few rows or columns; three diagonals (30 cores on Blackhole) measured fastest.
constexpr uint32_t kDiagonals = 3;

// Reads go out on NOC_0 and writes on NOC_1 so the two directions do not share links; dynamic NoC mode lets the one
// kernel drive both.
DataMovementHardwareConfig copier_hw_config() {
    auto hw = ttnn::create_reader_datamovement_config();
    hw.config_1xx->noc = NOC::NOC_0;
    hw.config_1xx->noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
    return hw;
}

// Cores on `diagonals` shifted diagonals: core (x = (i + k) % width, y = i) for row i and diagonal k.
CoreRangeSet diagonal_cores(CoreCoord grid, uint32_t diagonals) {
    std::set<CoreRange> cores;
    for (uint32_t k = 0; k < diagonals; ++k) {
        for (uint32_t i = 0; i < grid.y; ++i) {
            const CoreCoord core{(i + k) % grid.x, i};
            cores.insert(CoreRange(core, core));
        }
    }
    return CoreRangeSet(cores);
}

// Pages of one dim-0 entry: tiles for TILE layout, rows for ROW_MAJOR.
uint32_t pages_per_entry(const Tensor& single_entry) {
    const auto& shape = single_entry.padded_shape();
    if (single_entry.layout() == Layout::TILE) {
        const auto& tile = single_entry.tensor_spec().tile();
        return shape.volume() / (tile.get_height() * tile.get_width());
    }
    return shape.volume() / shape[-1];
}

}  // namespace

ttnn::device_operation::ProgramArtifacts IndexedEntryCopyProgramFactory::create_program_artifacts(
    const IndexedEntryCopyParams& params, const IndexedEntryCopyInputs& inputs, Tensor& /*output*/) {
    const auto& source = inputs.source;
    const auto& destination = inputs.destination;
    const auto& index = inputs.index;

    // Exactly one side may hold several entries along dim 0; with one entry on both, entry 0 copies the whole tensor.
    const uint32_t source_entries = source.logical_shape()[0];
    const bool indexed_is_source = source_entries > 1;
    const uint32_t num_entries = std::max(source_entries, static_cast<uint32_t>(destination.logical_shape()[0]));
    const uint32_t entry_pages = pages_per_entry(indexed_is_source ? destination : source);
    // index[0] < index_limit  <=>  index[0] * index_scale + index_offset < num_entries, with no 32-bit wrap.
    const uint32_t index_limit =
        params.index_offset < num_entries ? 1 + (num_entries - params.index_offset - 1) / params.index_scale : 0;

    // Validation equates page sizes and both sides are DRAM, so their aligned page sizes match.
    const uint32_t page_size = source.buffer()->page_size();
    const uint32_t page_stride = source.buffer()->aligned_page_size();

    const auto grid = source.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_1, pages_per_core_2] =
        split_work_to_cores(diagonal_cores(grid, kDiagonals), entry_pages);

    const TensorParamName SOURCE{"source"};
    const TensorParamName DESTINATION{"destination"};
    const TensorParamName INDEX{"index"};
    const ScratchpadSpecName STAGING{"staging"};
    const KernelSpecName COPIER{"copier"};

    KernelSpec copier{
        .unique_id = COPIER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/indexed_entry_copy/device/kernels/indexed_entry_copy.cpp",
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = STAGING, .accessor_name = "staging"}},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SOURCE, .accessor_name = "source"},
                TensorBinding{.tensor_parameter_name = DESTINATION, .accessor_name = "destination"},
                TensorBinding{.tensor_parameter_name = INDEX, .accessor_name = "index"},
            },
        .compile_time_args =
            {
                {"page_size", page_size},
                {"page_stride", page_stride},
                {"batch_pages", kBatchPages},
                {"pages_per_entry", entry_pages},
                {"index_limit", index_limit},
                {"index_scale", params.index_scale},
                {"index_offset", params.index_offset},
                {"indexed_is_source", static_cast<uint32_t>(indexed_is_source)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"first_page", "num_pages"}},
        .hw_config = copier_hw_config(),
    };

    // Page ranges depend only on shapes, so they are invariant across cache hits; only tensor addresses are
    // rebound, and the entry is resolved on device from `index` on every dispatch.
    KernelRunArgs copier_run_args{.kernel = COPIER};
    uint32_t first_page = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        const uint32_t num_pages = core_group_1.contains(core) ? pages_per_core_1 : pages_per_core_2;
        AddRuntimeArgsForNode(
            copier_run_args.runtime_arg_values, core, {{"first_page", first_page}, {"num_pages", num_pages}});
        first_page += num_pages;
    }

    ProgramSpec spec{
        .name = "indexed_entry_copy",
        .kernels = {std::move(copier)},
        // The index word is read into the staging area first, then the area is reused for the entry's pages.
        .scratchpads = {ScratchpadSpec{.unique_id = STAGING, .size_per_node = kBatchPages * page_stride}},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SOURCE, .spec = source.tensor_spec()},
                TensorParameter{.unique_id = DESTINATION, .spec = destination.tensor_spec()},
                TensorParameter{.unique_id = INDEX, .spec = index.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {COPIER}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(copier_run_args)};
    run_args.tensor_args = {
        {SOURCE, source.mesh_tensor()},
        {DESTINATION, destination.mesh_tensor()},
        {INDEX, index.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
