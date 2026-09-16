// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_topology_device_operation.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
namespace ttnn::experimental::prim {
ttnn::device_operation::ProgramArtifacts ChronologyFactory::create_program_artifacts(
    const ChronologyParams& a, const ChronologyInputs& in, std::vector<Tensor>& outputs) {
    using namespace tt::tt_metal::experimental;
    const auto& start = in.start.mesh_tensor();
    const auto& rank = in.rank.mesh_tensor();
    const auto& out = outputs[0].mesh_tensor();
    const KernelSpecName kernel{"derive"};
    const ScratchpadSpecName scratch{"scratch"};
    const TensorParamName sn{"start"}, rn{"rank"}, on{"output"};
    KernelSpec reader{
        .unique_id = kernel,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/chronological_topology/device/kernels/derive.cpp",
        .scratchpad_bindings = {{scratch, "scratch"}},
        .tensor_bindings = {{sn, "start"}, {rn, "rank"}, {on, "output"}},
        .compile_time_args =
            {{"P", a.sp_size}, {"C", a.local_rows}, {"BH", a.batch_heads}, {"K", a.key_dim}, {"V", a.value_dim}},
        .hw_config = ttnn::create_reader_datamovement_config(start.device().arch()),
    };
    ProgramSpec spec{
        .name = "kda_chronological_topology",
        .kernels = {std::move(reader)},
        .scratchpads = {{.unique_id = scratch, .size_per_node = 32}},
        .tensor_parameters =
            {{.unique_id = sn, .spec = start.tensor_spec()},
             {.unique_id = rn, .spec = rank.tensor_spec()},
             {.unique_id = on, .spec = out.tensor_spec()}},
        .work_units =
            {{.name = "main", .kernels = {kernel}, .target_nodes = CoreRangeSet({CoreRange({0, 0}, {0, 0})})}},
    };
    ProgramRunArgs run;
    run.tensor_args = {{sn, start}, {rn, rank}, {on, out}};
    return {.spec = std::move(spec), .run_params = std::move(run)};
}
}  // namespace ttnn::experimental::prim
