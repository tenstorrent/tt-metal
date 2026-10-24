// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_program_factory.hpp"

#include <algorithm>

#include "indexed_entry_copy_device_operation.hpp"

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

}  // namespace

ttnn::device_operation::ProgramArtifacts IndexedEntryCopyProgramFactory::create_program_artifacts(
    const IndexedEntryCopyParams& params, const IndexedEntryCopyInputs& inputs, Tensor& /*output*/) {
    const auto& source = inputs.source;
    const auto& destination = inputs.destination;
    const auto& index = inputs.index;

    // Exactly one side may hold several entries along dim 0; with one entry on both, either reading copies
    // the whole tensor at entry 0.
    const uint32_t source_entries = source.padded_shape()[0];
    const uint32_t destination_entries = destination.padded_shape()[0];
    const bool indexed_is_source = source_entries > 1;
    const uint32_t num_entries = std::max(source_entries, destination_entries);
    const uint32_t pages_per_entry = indexed_entry_copy_num_pages(indexed_is_source ? destination : source);

    const uint32_t page_size = source.buffer()->page_size();
    const uint32_t page_stride =
        std::max(source.buffer()->aligned_page_size(), destination.buffer()->aligned_page_size());
    const uint32_t batch_pages = std::clamp(kStagingBudgetBytes / page_stride, 1u, kMaxBatchPages);
    const uint32_t index_page_size = index.buffer()->aligned_page_size();

    const auto grid = source.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_1, pages_per_core_2] =
        split_work_to_cores(grid, pages_per_entry);

    const TensorParamName SOURCE{"source"};
    const TensorParamName DESTINATION{"destination"};
    const TensorParamName INDEX{"index"};
    const DFBSpecName STAGING{"staging"};
    const DFBSpecName INDEX_STAGING{"index_staging"};
    const KernelSpecName COPIER{"copier"};

    // Both buffers are raw L1 scratch for the single kernel that touches them (no FIFO handshake), so each
    // binds the kernel as producer and consumer under one accessor name.
    DataflowBufferSpec staging{
        .unique_id = STAGING,
        .entry_size = page_stride,
        .num_entries = batch_pages,
        .data_format_metadata = datatype_to_dataformat_converter(source.dtype()),
    };
    DataflowBufferSpec index_staging{
        .unique_id = INDEX_STAGING,
        .entry_size = index_page_size,
        .num_entries = 1,
        .data_format_metadata = datatype_to_dataformat_converter(index.dtype()),
    };

    KernelSpec copier{
        .unique_id = COPIER,
        .source = "ttnn/cpp/ttnn/operations/experimental/indexed_entry_copy/device/kernels/indexed_entry_copy.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = INDEX_STAGING,
                    .accessor_name = "index_staging",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INDEX_STAGING,
                    .accessor_name = "index_staging",
                    .endpoint_type = DFBEndpointType::CONSUMER},
            },
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
                {"pages_per_entry", pages_per_entry},
                {"num_entries", num_entries},
                {"index_scale", params.index_scale},
                {"index_offset", params.index_offset},
                {"index_page_size", index_page_size},
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
        uint32_t num_pages = 0;
        if (core_group_1.contains(core)) {
            num_pages = pages_per_core_1;
        } else if (core_group_2.contains(core)) {
            num_pages = pages_per_core_2;
        }
        AddRuntimeArgsForNode(
            copier_run_args.runtime_arg_values, core, {{"first_page", first_page}, {"num_pages", num_pages}});
        first_page += num_pages;
    }

    ProgramSpec spec{
        .name = "indexed_entry_copy",
        .kernels = {std::move(copier)},
        .dataflow_buffers = {std::move(staging), std::move(index_staging)},
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
