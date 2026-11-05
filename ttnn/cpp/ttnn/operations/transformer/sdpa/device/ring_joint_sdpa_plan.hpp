// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side planning for the ring-joint SDPA program factory: the joint-input geometry and the K split. Pure functions
// of the operation's attributes and inputs.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim::ring_joint_plan {

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Resolved joint Q/K/V params + sequence lengths for both replicated and sharded-joint paths.
// Centralizes the optional-tensor / logical_l branching so descriptor construction and the
// runtime planners share one definition of L (full) and L_local (per-device).
struct RingJointInputParams {
    bool has_joint_tensors = false;
    bool joint_is_sharded = false;
    const Tensor* joint_q = nullptr;
    const Tensor* joint_k = nullptr;
    const Tensor* joint_v = nullptr;
    const Tensor* gathered_joint_k = nullptr;
    const Tensor* gathered_joint_v = nullptr;
    uint32_t L = 0;          // full (padded) joint sequence length across the ring
    uint32_t L_local = 0;    // per-device joint length (padded L/P when sharded, else L)
    uint32_t logical_l = 0;  // true (unpadded) joint token count; <= L on the sharded path
};

RingJointInputParams resolve_ring_joint_input_params(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// The kernels run the chunked-prefill path. Cross is non-causal on chunked-shaped tensors, so kernels and the work
// planner use the non-chunked path.
bool kernel_chunked(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// The fused gather's K/V inputs and gathered outputs: K, plus V unless V is K's latent alias.
std::pair<std::vector<Tensor>, std::vector<Tensor>> fused_gather_kv_tensors(const RingJointSDPAInputs& tensor_args);

// The K split (kernels/ring_joint_ksplit.hpp) and the values it is derived from.
struct KSplitShape {
    uint32_t count = 1;           // bands; 1 is no split
    uint32_t rows_per_split = 0;  // grid rows per band
    CoreCoord grid;
    uint32_t num_q_chunks = 0;  // per head, local plus joint
    uint32_t max_q_per_core = 0;
    bool use_streaming_compute = false;
    // The streaming QK^T @ V output subblock, and the row group the writer drains cb_out in.
    uint32_t out_subblock_h = 0;
    uint32_t out_subblock_w = 0;
    uint32_t writer_out_row_group_h = 0;
};

// The K split and the subblocks it depends on. A chunked call with k_splits > 1 gets exactly that many bands or a
// TT_FATAL; non-chunked calls ignore it, so one config serves every chunk.
KSplitShape derive_ksplit(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// Why this call cannot run the K split or segmented accumulation (both need one Q chunk per core and whole output row
// groups, static_assert in ring_joint_sdpa.cpp), or "". `shape` is derive_ksplit's.
std::string single_q_chunk_blocker(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, const KSplitShape& shape);

}  // namespace ttnn::prim::ring_joint_plan
