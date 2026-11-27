// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/workload/specification/compute_modes.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::all_to_all_matmul {

// What the caller chose and the entry resolved, as scalars. All structural: the default key hashes them with the
// tensor specs. The plan solved from them (and from the mesh topology) is memoized in
// device/all_to_all_matmul_planner.hpp and never an attribute; the per-mesh global semaphores' addresses are runtime
// args, never attributes.
struct AllToAllMatmulParams {
    uint32_t cluster_axis = 1;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring;
    uint32_t num_links = 2;
    tt::tt_metal::MemoryConfig output_memory_config;
    // From the caller's ttnn.ComputeConfigDescriptor (default: HiFi4, fp32 DEST, no approximation).
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool math_approx_mode = false;
    // q-stem mode (MLA early-Q): 0 = off (the input is (B, Hl, S, K)). Else the input is the unsplit q_b_proj output
    // (1, 1, S, Hl * (K + qk_rope_head_dim)) and the op also returns rope_seq (1, H, S/G, qk_rope_head_dim): the
    // rope columns of every head moved by the same exchange, no matmul, placed by rope_memory_config.
    uint32_t qk_rope_head_dim = 0;
    tt::tt_metal::MemoryConfig rope_memory_config;
};

struct AllToAllMatmulInputs {
    // (B, Hl, S, K), bf16 TILE DRAM interleaved; q-stem mode: (1, 1, S, Hl * (K + R)), DRAM or L1 interleaved
    const Tensor& input;
    const Tensor& weight;  // (1, H, K, N), replicated, TILE interleaved
    // Landing scratch (height-sharded L1 over the compute cores, scratch_spec in the planner header). Always set
    // when launched from ttnn::all_to_all_matmul: the caller's scratch_tensor, or one the entry allocated.
    const std::optional<Tensor>& scratch;
};

// The one way the support contract refuses (EXCLUSIONS is empty): raised in Python as
// ttnn.operations._op_contract.UnsupportedAxisValue.
struct UnsupportedAxisValue : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::all_to_all_matmul
