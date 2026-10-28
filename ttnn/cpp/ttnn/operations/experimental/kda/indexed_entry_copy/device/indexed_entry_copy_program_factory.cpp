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

// Pages staged in L1 per read/write round; bounded so wide row-major pages stay within a modest footprint.
constexpr uint32_t kMaxBatchPages = 8;
constexpr uint32_t kStagingBudgetBytes = 64 * 1024;
// Workers sit on this many shifted diagonals of the worker grid, one per row and per column on each diagonal.
// NoC reads route along x then y and writes along y then x, so a worker that shares no row or column with
// another keeps its DRAM traffic off the other workers' links. Measured on a Blackhole P150 (11x10 grid) for
// the Kimi-K3 copies: 3 diagonals (30 cores) copy the 54 KiB convolution tail in 1.8 us against 2.5 us on the
// full grid, and the 1.5 MiB recurrent carry at the same ~12.4 us; more cores only add start-up cost.
constexpr uint32_t kDiagonals = 3;

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
    const uint32_t batch_pages = std::clamp(kStagingBudgetBytes / page_stride, 1u, kMaxBatchPages);

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
                {"batch_pages", batch_pages},
                {"pages_per_entry", entry_pages},
                {"index_limit", index_limit},
                {"index_scale", params.index_scale},
                {"index_offset", params.index_offset},
                {"indexed_is_source", static_cast<uint32_t>(indexed_is_source)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"first_page", "num_pages"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
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
        .scratchpads = {ScratchpadSpec{.unique_id = STAGING, .size_per_node = batch_pages * page_stride}},
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
