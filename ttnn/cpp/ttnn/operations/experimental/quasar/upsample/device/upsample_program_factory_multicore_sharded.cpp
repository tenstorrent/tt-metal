// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar (Metal 2.0, DFB implicit sync) nearest-neighbour upsample over a row-major height- or block-sharded
// activation [N, H, W, C].
//
// Design (differs from ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.cpp):
//   - no host-built stick-interval config tensor: the producer kernel derives each output stick's source stick
//     arithmetically and addresses it through a TensorAccessor over the sharded INPUT tensor binding (one page =
//     one shard-wide row, for block sharding the row is split into ncols pages);
//   - a plain L1 staging ring DFB (one entry = one stick) sits between a reader and a writer kernel, both SPMD
//     with the same `num_threads` DM threads (STRIDED: thread t owns ring entries / sticks t, t+T, ...), both
//     fully implicit-sync: the reader fills entries with noc.async_read<NocOptions::TXN_ID> from the sharded
//     input, the writer drains them with noc.async_write<NocOptions::TXN_ID> into the local output shard
//     (TensorAccessor over the OUTPUT tensor). The DM0 ISR posts / acks the credits, finish() handles the
//     tails. (A DFB borrowed from the output tensor cannot be used as the implicit-sync target: the ISR never
//     posts for it, see tests/tt_metal/.../test_borrowed_memory_dataflow_buffer.cpp which opts out too.)
// Only full shards are supported: every core in the (reduced) grid holds exactly shard_shape[0] input sticks.

#include "ttnn/operations/experimental/quasar/upsample/device/upsample_device_operation.hpp"

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace metal2 = tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// DM threads per kernel (reader and writer run the same count so ring entry i is handled by thread i % T on
// both sides). Quasar has six user DM harts per node, so up to three each. Override with
// TT_METAL_QSR_UPSAMPLE_THREADS for experiments.
uint32_t num_dm_threads() {
    if (const char* env = std::getenv("TT_METAL_QSR_UPSAMPLE_THREADS")) {
        const int v = std::atoi(env);
        TT_FATAL(v >= 1 && v <= 3, "TT_METAL_QSR_UPSAMPLE_THREADS must be in [1, 3], got {}", v);
        return static_cast<uint32_t>(v);
    }
    return 2;
}

// Staging ring depth in entries per thread. The implicit-sync rules need num_entries to be a multiple of
// (transaction ids x threads), i.e. an even count per thread; 8 keeps a few sticks in flight per thread.
constexpr uint32_t kStageEntriesPerThread = 8;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleMultiCoreShardedProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const metal2::KernelSpecName READER{"upsample_reader"};
    const metal2::KernelSpecName WRITER{"upsample_writer"};
    const metal2::DFBSpecName STAGE_DFB{"upsample_stage"};
    const metal2::TensorParamName INPUT{"upsample_input"};
    const metal2::TensorParamName OUTPUT{"upsample_output"};
    constexpr const char* READER_KERNEL =
        "ttnn/cpp/ttnn/operations/experimental/quasar/upsample/device/kernels/dataflow/"
        "reader_upsample_sharded_implicit.cpp";
    constexpr const char* WRITER_KERNEL =
        "ttnn/cpp/ttnn/operations/experimental/quasar/upsample/device/kernels/dataflow/"
        "writer_upsample_sharded_implicit.cpp";

    const auto& input = input_tensor;
    auto& output = output_tensor;
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Sharded upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only row-major layout is supported in nearest upsample");
    TT_FATAL(input.logical_shape()[-1] == output.logical_shape()[-1], "Expected input and output channels to match");
    TT_FATAL(input.dtype() == output.dtype(), "Expected input and output dtypes to match");

    const uint32_t batch = input.padded_shape()[0];
    const uint32_t in_h = input.padded_shape()[1];
    const uint32_t in_w = input.padded_shape()[2];
    const uint32_t out_h = in_h * scale_h;
    const uint32_t out_w = in_w * scale_w;

    const auto in_shard_spec = input.shard_spec().value();
    const auto out_shard_spec = output.shard_spec().value();
    const bool block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    TT_FATAL(
        block_sharded || input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Sharded upsample supports HEIGHT_SHARDED or BLOCK_SHARDED inputs");
    TT_FATAL(
        in_shard_spec.grid == out_shard_spec.grid && in_shard_spec.orientation == out_shard_spec.orientation,
        "Input and output shard grids must match");

    // One page of the sharded input = one shard-wide row. Height sharded: the full row (1 page per row);
    // block sharded: the row is split over the grid's columns (ncols pages per row, page = this core's column).
    const uint32_t in_nsticks_per_core = in_shard_spec.shape[0];
    const uint32_t out_nsticks_per_core = out_shard_spec.shape[0];
    TT_FATAL(
        out_nsticks_per_core == in_nsticks_per_core * scale_h * scale_w,
        "Output shard height {} must be input shard height {} * scale_h * scale_w",
        out_nsticks_per_core,
        in_nsticks_per_core);
    const uint32_t stick_nbytes = out_shard_spec.shape[1] * output.element_size();
    TT_FATAL(
        in_shard_spec.shape[1] * input.element_size() == stick_nbytes,
        "Input and output shard widths must match (block sharding keeps the channel split)");

    const uint32_t total_in_rows = batch * in_h * in_w;
    TT_FATAL(
        total_in_rows % in_nsticks_per_core == 0,
        "Sharded upsample on Quasar needs full input shards: {} rows is not a multiple of the shard height {}",
        total_in_rows,
        in_nsticks_per_core);
    const uint32_t nhw_cores = total_in_rows / in_nsticks_per_core;

    // ---- Per-core placement: which output rows / which column each core owns ----
    const auto& range = *in_shard_spec.grid.ranges().begin();
    TT_FATAL(in_shard_spec.grid.ranges().size() == 1, "Sharded upsample expects a single rectangular shard grid");
    const bool row_major = in_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> core_row_start;
    std::vector<uint32_t> core_col;
    uint32_t pages_per_row = 1;
    if (block_sharded) {
        // ROW_MAJOR: channels split along x, rows along y (COL_MAJOR: the other way round).
        const uint32_t ncols =
            row_major ? range.end_coord.x - range.start_coord.x + 1 : range.end_coord.y - range.start_coord.y + 1;
        const uint32_t nrows =
            row_major ? range.end_coord.y - range.start_coord.y + 1 : range.end_coord.x - range.start_coord.x + 1;
        TT_FATAL(nhw_cores <= nrows, "Input needs {} row blocks but the shard grid has {}", nhw_cores, nrows);
        pages_per_row = ncols;
        for (uint32_t r = 0; r < nhw_cores; ++r) {
            for (uint32_t c = 0; c < ncols; ++c) {
                cores.push_back(
                    row_major ? CoreCoord(range.start_coord.x + c, range.start_coord.y + r)
                              : CoreCoord(range.start_coord.x + r, range.start_coord.y + c));
                core_row_start.push_back(r * out_nsticks_per_core);
                core_col.push_back(c);
            }
        }
    } else {
        const auto all = corerange_to_cores(in_shard_spec.grid, std::nullopt, row_major);
        TT_FATAL(nhw_cores <= all.size(), "Input needs {} cores but the shard grid has {}", nhw_cores, all.size());
        for (uint32_t k = 0; k < nhw_cores; ++k) {
            cores.push_back(all[k]);
            core_row_start.push_back(k * out_nsticks_per_core);
            core_col.push_back(0);
        }
    }
    std::set<CoreRange> core_ranges;
    for (const auto& core : cores) {
        core_ranges.insert(CoreRange(core, core));
    }
    const CoreRangeSet cores_with_work(core_ranges);

    // ---- Staging ring DFB: one entry per in-flight stick, plain L1 (not borrowed) ----
    const uint32_t num_threads = num_dm_threads();
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const uint32_t entry_size = tt::round_up(stick_nbytes, output.buffer()->alignment());
    TT_FATAL(
        out_nsticks_per_core % num_threads == 0,
        "Output shard height {} must be a multiple of the DM thread count {}",
        out_nsticks_per_core,
        num_threads);
    const uint32_t stage_entries = std::min(out_nsticks_per_core, kStageEntriesPerThread * num_threads);
    metal2::DataflowBufferSpec stage_dfb{
        .unique_id = STAGE_DFB,
        .entry_size = entry_size,
        .num_entries = stage_entries,
        .data_format_metadata = out_df,
    };

    log_debug(
        tt::LogOp,
        "quasar upsample: {} cores, {} DM threads per kernel, ring {} entries, {} in / {} out sticks per core of {} B, "
        "pages_per_row {}",
        cores.size(),
        num_threads,
        stage_entries,
        in_nsticks_per_core,
        out_nsticks_per_core,
        stick_nbytes,
        pages_per_row);

    // ---- Kernels ----
    metal2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{READER_KERNEL},
        .num_threads = num_threads,
        .dfb_bindings = {metal2::ProducerOf(STAGE_DFB, "stage")},
        .tensor_bindings = {metal2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"scale_h", scale_h},
             {"scale_w", scale_w},
             {"in_h", in_h},
             {"in_w", in_w},
             {"out_h", out_h},
             {"out_w", out_w},
             {"pages_per_row", pages_per_row},
             {"out_nsticks_per_core", out_nsticks_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"out_row_start", "col"}},
        // implicit sync ON (default): the TXN_ID reads post credits through the DM0 ISR
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    metal2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_KERNEL},
        .num_threads = num_threads,
        .dfb_bindings = {metal2::StridedConsumerOf(STAGE_DFB, "stage")},
        .tensor_bindings = {metal2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"pages_per_row", pages_per_row}, {"out_nsticks_per_core", out_nsticks_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"out_row_start", "col"}},
        // implicit sync ON (default): the TXN_ID writes ack the entries through the DM0 ISR
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    metal2::ProgramSpec spec{
        .name = "upsample_multicore_sharded_qsr",
        .kernels = {reader, writer},
        .dataflow_buffers = {stage_dfb},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output_mesh.tensor_spec()},
            },
        .work_units = {metal2::WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = cores_with_work,
        }},
    };

    // ---- Per-core runtime args ----
    metal2::KernelRunArgs reader_run{.kernel = READER};
    metal2::KernelRunArgs writer_run{.kernel = WRITER};
    for (size_t k = 0; k < cores.size(); ++k) {
        for (auto* run : {&reader_run, &writer_run}) {
            run->runtime_arg_values["out_row_start"][cores[k]] = core_row_start[k];
            run->runtime_arg_values["col"][cores[k]] = core_col[k];
        }
    }
    metal2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT, metal2::TensorArgument{input_mesh}},
        {OUTPUT, metal2::TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
