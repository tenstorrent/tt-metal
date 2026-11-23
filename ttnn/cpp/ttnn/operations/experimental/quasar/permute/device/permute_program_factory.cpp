// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar-native permute: one data-movement kernel on all six free DM harts of every cluster, no compute kernel.
//
// The tensor is cut into blocks (permute_plan.hpp). Every cluster takes a contiguous range of blocks, and its
// DM thread t takes blocks t, t + T, ... of that range. A thread moves a block on its own: NoC reads into its
// private L1 window, an element reorder in L1 when pages change content, NoC writes of the finished output
// pages. Threads share nothing (no dataflow buffer, no semaphore): each owns two windows of one scratchpad.

#include "permute_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts PermuteProgramFactory::create_program_artifacts(
    const PermuteParams& operation_attributes, const PermuteInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const uint32_t threads = operation_attributes.num_threads;

    const auto grid = input.device()->compute_with_storage_grid_size();
    // Enough blocks for every thread of every cluster to hold two.
    const uint32_t target_blocks = static_cast<uint32_t>(grid.x * grid.y) * threads * 2;
    const permute_plan::BlockPlan plan = permute_block_plan(operation_attributes, input, target_blocks);

    const auto [num_cores, all_cores, core_group_1, core_group_2, blocks_per_core_1, blocks_per_core_2] =
        split_work_to_cores(grid, plan.num_blocks);

    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const ScratchpadSpecName WINDOWS{"windows"};
    const KernelSpecName MOVER{"mover"};

    KernelSpec mover{
        .unique_id = MOVER,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/permute/device/kernels/dataflow/permute_blocks.cpp",
        .num_threads = threads,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = WINDOWS, .accessor_name = "windows"}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"elem_size", plan.kernel_elem_size}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_block", "num_blocks"},
             .common_runtime_array_args = {{.name = "plan", .size = permute_plan::kPlanWords}}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelRunArgs mover_run_args{.kernel = MOVER};
    mover_run_args.common_runtime_array_arg_values["plan"] = plan.words;
    uint32_t first_block = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt)) {
        const uint32_t blocks = core_group_1.contains(core) ? blocks_per_core_1 : blocks_per_core_2;
        AddRuntimeArgsForNode(
            mover_run_args.runtime_arg_values, core, {{"first_block", first_block}, {"num_blocks", blocks}});
        first_block += blocks;
    }
    TT_FATAL(
        first_block == plan.num_blocks,
        "Quasar permute: the work split covers {} of {} blocks",
        first_block,
        plan.num_blocks);

    ProgramSpec spec;
    spec.name = "quasar_permute";
    spec.kernels = {std::move(mover)};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = WINDOWS,
        .size_per_node = threads * (plan.in_window_bytes + plan.out_window_bytes),
    }};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {MOVER}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(mover_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
