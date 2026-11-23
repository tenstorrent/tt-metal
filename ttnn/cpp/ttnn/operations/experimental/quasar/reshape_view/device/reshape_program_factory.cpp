// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_program_factory.hpp"

#include <algorithm>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/scratchpad_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

const KernelSpecName MOVER{"reshape"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
const ScratchpadSpecName STAGE{"stage"};

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/device/kernels/dataflow/";

// Bytes of one stage half of one hart: a window of the output is gathered into it and written out from
// it. A multiple of 64 so that two harts writing pieces of one wide row-major page never share a DRAM
// (64 B) or L1 (16 B) memory word.
constexpr uint32_t WINDOW_BYTES = 8192;
// Cap on row-major pages per window: with tiny pages the window is bounded by its NOC writes, not its bytes.
constexpr uint32_t MAX_PAGES_PER_WINDOW = 64;
constexpr uint32_t CACHE_LINE_BYTES = 64;

// One DM kernel of `num_threads` harts on every node that draws work. The `num_windows` windows of the
// output are dealt to the nodes in contiguous ranges; hart t of a node takes windows t, t + T, ... of its
// range. Each hart owns `hart_stage_bytes` of the stage, which hold its two halves of `window_bytes`.
ttnn::device_operation::ProgramArtifacts build(
    const char* program_name,
    const char* kernel_file,
    const ReshapeViewParams& operation_attributes,
    const Tensor& input,
    const Tensor& output,
    uint32_t num_windows,
    uint32_t hart_stage_bytes,
    KernelSpec::CompileTimeArgs compile_time_args) {
    const uint32_t T = operation_attributes.num_threads;

    const auto grid = input.device()->compute_with_storage_grid_size();
    const CoreRangeSet candidate_nodes = operation_attributes.sub_core_grid.has_value()
                                             ? operation_attributes.sub_core_grid.value()
                                             : CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto [num_nodes, all_nodes, group_1, group_2, windows_per_node_1, windows_per_node_2] =
        split_work_to_cores(candidate_nodes, num_windows);

    ProgramSpec spec;
    spec.name = program_name;
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
    };
    spec.scratchpads = {ScratchpadSpec{.unique_id = STAGE, .size_per_node = T * hart_stage_bytes}};

    spec.kernels = {KernelSpec{
        .unique_id = MOVER,
        .source = std::string(KERNEL_DIR) + kernel_file,
        .num_threads = T,
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = STAGE, .accessor_name = "stage"}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = std::move(compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"window_start", "num_windows"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    }};
    spec.work_units = {WorkUnitSpec{.name = "wu", .kernels = {MOVER}, .target_nodes = all_nodes}};

    ProgramRunArgs run_args;
    KernelRunArgs kernel_run_args{.kernel = MOVER};
    uint32_t window_start = 0;
    for (const auto& node : corerange_to_cores(all_nodes, std::nullopt)) {
        uint32_t windows = 0;
        if (group_1.contains(node)) {
            windows = windows_per_node_1;
        } else if (group_2.contains(node)) {
            windows = windows_per_node_2;
        } else {
            TT_THROW("reshape: node ({}, {}) is not in either work group", node.x, node.y);
        }
        AddRuntimeArgsForNode(
            kernel_run_args.runtime_arg_values, node, {{"window_start", window_start}, {"num_windows", windows}});
        window_start += windows;
    }
    TT_FATAL(window_start == num_windows, "reshape: dealt {} of {} windows", window_start, num_windows);
    run_args.kernel_run_args = {std::move(kernel_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts ReshapeViewRowMajorProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;

    // A page is one row of the last dimension; only its logical bytes carry data.
    const uint32_t element_bytes = input.element_size();
    const uint32_t src_page_bytes = input.logical_shape()[-1] * element_bytes;
    const uint32_t dst_page_bytes = output.logical_shape()[-1] * element_bytes;
    const uint32_t num_dst_pages = output.logical_volume() / output.logical_shape()[-1];

    // Narrow pages: a window is as many whole destination pages as fit. Wide pages: a page is moved in
    // WINDOW_BYTES pieces, which also spreads one page over harts and nodes.
    uint32_t windows_per_page = 1;
    uint32_t pages_per_window = 1;
    uint32_t window_bytes = dst_page_bytes;
    uint32_t num_windows = num_dst_pages;
    if (dst_page_bytes > WINDOW_BYTES) {
        windows_per_page = tt::div_up(dst_page_bytes, WINDOW_BYTES);
        window_bytes = WINDOW_BYTES;
        num_windows = num_dst_pages * windows_per_page;
    } else {
        pages_per_window = std::min(WINDOW_BYTES / dst_page_bytes, MAX_PAGES_PER_WINDOW);
        window_bytes = pages_per_window * dst_page_bytes;
        num_windows = tt::div_up(num_dst_pages, pages_per_window);
    }

    return build(
        "quasar_reshape_row_major",
        "reshape_row_major.cpp",
        operation_attributes,
        input,
        output,
        num_windows,
        2 * window_bytes,
        {{"src_page_bytes", src_page_bytes},
         {"dst_page_bytes", dst_page_bytes},
         {"window_bytes", window_bytes},
         {"windows_per_page", windows_per_page},
         {"pages_per_window", pages_per_window},
         {"num_dst_pages", num_dst_pages}});
}

ttnn::device_operation::ProgramArtifacts ReshapeViewTiledProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;
    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;

    const auto& in_shape = input.logical_shape();
    const auto& out_shape = output.logical_shape();
    const uint32_t element_bytes = input.element_size();
    const uint32_t tile_bytes = TILE_HW * element_bytes;

    const uint32_t in_tiles_h = tt::div_up(in_shape[-2], TILE_HEIGHT);
    const uint32_t in_tiles_w = tt::div_up(in_shape[-1], TILE_WIDTH);
    const uint32_t out_tiles_h = tt::div_up(out_shape[-2], TILE_HEIGHT);
    const uint32_t out_tiles_w = tt::div_up(out_shape[-1], TILE_WIDTH);
    const uint32_t num_out_tiles = out_shape[0] * out_tiles_h * out_tiles_w;

    const uint32_t tiles_per_window = WINDOW_BYTES / tile_bytes;
    const uint32_t num_windows = tt::div_up(num_out_tiles, tiles_per_window);
    // The harts fill tile padding with stores, which reach L1 through a cache of 64-byte lines. One guard
    // line per hart keeps a line one hart writes back clear of the bytes another hart's NOC reads land in.
    const uint32_t hart_stage_bytes = 2 * tiles_per_window * tile_bytes + CACHE_LINE_BYTES;

    return build(
        "quasar_reshape_tiled",
        "reshape_tiled.cpp",
        operation_attributes,
        input,
        output,
        num_windows,
        hart_stage_bytes,
        {{"element_bytes", element_bytes},
         {"hart_stage_bytes", hart_stage_bytes},
         {"pad_word", operation_attributes.pad_word},
         {"in_h", in_shape[-2]},
         {"in_w", in_shape[-1]},
         {"out_h", out_shape[-2]},
         {"out_w", out_shape[-1]},
         {"in_tiles_h", in_tiles_h},
         {"in_tiles_w", in_tiles_w},
         {"out_tiles_h", out_tiles_h},
         {"out_tiles_w", out_tiles_w},
         {"num_out_tiles", num_out_tiles},
         {"tiles_per_window", tiles_per_window}});
}

}  // namespace ttnn::prim::qsr
