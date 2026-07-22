// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// TODO(Task 3, Phase A): implement the DRAM-staged fused program for one device — fabric mux v2 gather of the
// in0 K-shards into a per-device DRAM gather buffer with a full-gather barrier (diagnostic) / per-transport
// readiness semaphores (streaming), then a replicated regime_a compute engine reading the gather buffer.
ttnn::device_operation::CachedProgram<AllGatherRegimeAMatmulAsyncProgramFactory::shared_variables_t>
AllGatherRegimeAMatmulAsyncProgramFactory::create_at(
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const AllGatherRegimeAMatmulAsyncInputs& /*tensor_args*/,
    std::vector<Tensor>& /*output_tensors*/) {
    TT_THROW("all_gather_regime_a_matmul_async: Phase A streaming program factory is under construction (Task 3)");
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const AllGatherRegimeAMatmulAsyncInputs& /*tensor_args*/,
    std::vector<Tensor>& /*output_tensors*/) {}

}  // namespace ttnn::experimental::prim
