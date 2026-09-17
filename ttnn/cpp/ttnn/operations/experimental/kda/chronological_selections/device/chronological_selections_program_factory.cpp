// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_selections_device_operation.hpp"
#include "kernels/chronology.hpp"
#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
namespace ttnn::experimental::prim {
ttnn::device_operation::MeshWorkloadArtifacts ChronologicalSelectionsFactory::create_mesh_workload_artifacts(
    const ChronologicalSelectionsParams& a,
    const ChronologicalSelectionsInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal::experimental;
    const auto& actual_start = in.actual_start.mesh_tensor();
    const auto& actual_end = (in.actual_end ? *in.actual_end : in.actual_start).mesh_tensor();
    const auto& out = outputs[0].mesh_tensor();
    const KernelSpecName kernel{"derive"};
    const ScratchpadSpecName scratch{"scratch"};
    const TensorParamName sn{"actual_start"}, en{"actual_end"}, on{"output"};
    KernelSpec reader{
        .unique_id = kernel,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/derive.cpp",
        .scratchpad_bindings = {{scratch, "scratch"}},
        .tensor_bindings = {{sn, "actual_start"}, {en, "actual_end"}, {on, "output"}},
        .compile_time_args =
            {{"has_actual_end", uint32_t(in.actual_end.has_value())},
             {"BH", a.batch_heads},
             {"K", a.key_dim},
             {"V", a.value_dim}},
        .hw_config = ttnn::create_reader_datamovement_config(actual_start.device().arch()),
    };
    ProgramSpec spec{
        .name = "kda_chronological_selections",
        .kernels = {std::move(reader)},
        .scratchpads =
            {{.unique_id = scratch, .size_per_node = kda_chronology::selection::record_width * sizeof(uint32_t)}},
        .tensor_parameters =
            {{.unique_id = sn, .spec = actual_start.tensor_spec()},
             {.unique_id = en, .spec = actual_end.tensor_spec()},
             {.unique_id = on, .spec = out.tensor_spec()}},
        .work_units =
            {{.name = "main", .kernels = {kernel}, .target_nodes = CoreRangeSet({CoreRange({0, 0}, {0, 0})})}},
    };
    ProgramRunArgs run;
    run.tensor_args = {{sn, actual_start}, {en, actual_end}, {on, out}};
    return kda_factory_detail::chronology_workload(
        {.spec = std::move(spec), .run_params = std::move(run)},
        tensor_coords,
        actual_start.device(),
        a.sequence_parallel_axis,
        a.local_rows,
        kernel);
}
}  // namespace ttnn::experimental::prim
