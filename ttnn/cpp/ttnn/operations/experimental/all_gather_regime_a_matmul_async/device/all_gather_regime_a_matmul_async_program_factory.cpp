// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// TODO(Task 3, Phase A): implement the DRAM-staged fused program — fabric mux v2 gather of in0 shards into
// a persistent DRAM gather buffer with per-transport-chunk readiness semaphores, then regime_a compute
// reading in0 progressively (compile-gated full-gather diagnostic vs streaming). Blueprint in progress.
AllGatherRegimeAMatmulAsyncProgramFactory::cached_program_t AllGatherRegimeAMatmulAsyncProgramFactory::create(
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const AllGatherRegimeAMatmulAsyncInputs& /*tensor_args*/,
    std::vector<Tensor>& /*output_tensors*/) {
    TT_THROW("all_gather_regime_a_matmul_async: Phase A streaming program factory is under construction (Task 3)");
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const AllGatherRegimeAMatmulAsyncInputs& /*tensor_args*/,
    std::vector<Tensor>& /*output_tensors*/) {}

}  // namespace ttnn::experimental::prim
