// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Pages per reader push / writer pop: covers a DRAM round trip at the K3 copy sizes, and holds a core's whole share of
// the 54 KiB tail in one batch.
constexpr uint32_t kBatchPages = 8;
// Staging holds two batches so the reader fills one while the writer drains the other.
constexpr uint32_t kStagingBatches = 2;
// Workers fill the first full-width rows of the grid as one core range. Dispatch sends a program's binary, config
// and launch message once per range, so scattered single-core ranges add microseconds per launch under trace; full
// rows also measured faster than taller blocks of as many cores.
constexpr uint32_t kWorkerRows = 2;

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
        split_work_to_cores(CoreCoord{grid.x, std::min<std::size_t>(kWorkerRows, grid.y)}, entry_pages);

    const TensorParamName SOURCE{"source"};
    const TensorParamName DESTINATION{"destination"};
    const TensorParamName INDEX{"index"};
    const DFBSpecName STAGING{"staging"};
    const ScratchpadSpecName READER_INDEX_WORD{"reader_index_word"};
    const ScratchpadSpecName WRITER_INDEX_WORD{"writer_index_word"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"page_size", page_size},
        {"page_stride", page_stride},
        {"batch_pages", kBatchPages},
        {"pages_per_entry", entry_pages},
        {"index_limit", index_limit},
        {"index_scale", params.index_scale},
        {"index_offset", params.index_offset},
        {"indexed_is_source", static_cast<uint32_t>(indexed_is_source)},
    };
    const KernelSpec::RuntimeArgSchema runtime_arg_schema{.runtime_arg_names = {"first_page", "num_pages"}};

    // Each kernel resolves the entry from `index` into its own word.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/indexed_entry_copy/device/kernels/"
            "indexed_entry_copy_reader.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::PRODUCER}},
        .scratchpad_bindings = {ScratchpadBinding{
            .scratchpad_spec_name = READER_INDEX_WORD, .accessor_name = "index_word"}},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SOURCE, .accessor_name = "source"},
                TensorBinding{.tensor_parameter_name = INDEX, .accessor_name = "index"},
            },
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/indexed_entry_copy/device/kernels/"
            "indexed_entry_copy_writer.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::CONSUMER}},
        .scratchpad_bindings = {ScratchpadBinding{
            .scratchpad_spec_name = WRITER_INDEX_WORD, .accessor_name = "index_word"}},
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = DESTINATION, .accessor_name = "destination"},
                TensorBinding{.tensor_parameter_name = INDEX, .accessor_name = "index"},
            },
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Page ranges depend only on shapes, so they are invariant across cache hits; only tensor addresses are
    // rebound, and the entry is resolved on device from `index` on every dispatch.
    KernelRunArgs reader_run_args{.kernel = READER};
    uint32_t first_page = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        const uint32_t num_pages = core_group_1.contains(core) ? pages_per_core_1 : pages_per_core_2;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"first_page", first_page}, {"num_pages", num_pages}});
        first_page += num_pages;
    }
    KernelRunArgs writer_run_args{.kernel = WRITER, .runtime_arg_values = reader_run_args.runtime_arg_values};

    const uint32_t index_word_size = index.buffer()->aligned_page_size();
    ProgramSpec spec{
        .name = "indexed_entry_copy",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {DataflowBufferSpec{
            .unique_id = STAGING, .entry_size = page_stride, .num_entries = kStagingBatches * kBatchPages}},
        .scratchpads =
            {
                ScratchpadSpec{.unique_id = READER_INDEX_WORD, .size_per_node = index_word_size},
                ScratchpadSpec{.unique_id = WRITER_INDEX_WORD, .size_per_node = index_word_size},
            },
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SOURCE, .spec = source.tensor_spec()},
                TensorParameter{.unique_id = DESTINATION, .spec = destination.tensor_spec()},
                TensorParameter{.unique_id = INDEX, .spec = index.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {SOURCE, source.mesh_tensor()},
        {DESTINATION, destination.mesh_tensor()},
        {INDEX, index.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
