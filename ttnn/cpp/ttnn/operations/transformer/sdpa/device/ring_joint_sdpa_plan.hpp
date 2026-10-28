// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side planning shared by the ring-joint SDPA device operation and its program factory: the joint-input
// geometry, the K split, progressive slice readiness and the expanded-K floaters' eligibility. Pure functions of the
// operation's attributes and inputs, so the device operation (output specs) and the factory (program) decide alike.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/transformer/sdpa/device/ring_joint_floater_layout.hpp"
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

// The K split and the subblocks it depends on. A chunked call that asks for k_splits > 1 gets exactly that many
// bands, or a TT_FATAL naming why it cannot (calls that are not chunked prefill ignore it, so one config serves every
// chunk).
KSplitShape derive_ksplit(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// Why this call cannot run the K split or segmented accumulation (both need one Q chunk per core and whole output row
// groups, static_assert in ring_joint_sdpa.cpp), or "". `shape` is derive_ksplit's.
std::string single_q_chunk_blocker(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, const KSplitShape& shape);

// Progressive readiness (RingAttentionSliceReadiness): the gather sends every full slice, its own and the ones it
// relays, in parts that each hold a share of every K-split band, with a marker after each, so a remote ring step
// starts on the first part instead of waiting for the whole slice. Limited to the plain chunked dense path, where the
// gather's slice is the valid prefix of local K chunks the reader iterates. Reads the gather's buffers and the fabric
// payload size (its bank-owned schedule).
bool slice_readiness_enabled(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// The plain chunked dense gather slice_readiness_enabled builds on, on any topology with more than two devices: what
// the K split's bands and the floaters' pieces are cut from.
bool plain_gathered_slices(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

// ring_joint::floater_compute_to_transfer of this call: its QK^T / PV and K expansion per arriving K/V token at the
// matmuls' fidelity, over that token's transfer on the ring. Needs expanded K and a program config.
double floater_compute_to_transfer(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, const KSplitShape& ksplit);

// Expanded-K floaters (kernels/ring_joint_floaters.hpp): the schedule of a shape on the 11x10 grid of the 8-chip ring
// with 2 gather links, or nullopt. The schedule is SDPAProgramConfig::floater_schedule if set; otherwise the layout's
// (ring_joint::kFloaterLayoutSchedules), where the call's compute per arriving token is large enough against its
// transfer for the floaters to pay off (floater_compute_to_transfer >= ring_joint::kFloaterMinComputeToTransfer). On
// those shapes the K split's 24 home blocks of 2x2 cores fill a 10x10 sub-grid, leaving the 25th block, the last grid
// column and the gather column below its workers idle: 20 cores, five floater groups of 4 that take over pieces of the
// homes' bands. The gather's workers are the top 2 x num_links cores of column grid.x (column-major from (grid.x, 0)).
// The floaters cut their pieces from the plain gathered slices (plain_gathered_slices), checked last, so other shapes
// never query the gather's buffers or the fabric.
//
// refuse_unusable_schedule: TT_FATAL on a floater_schedule the call cannot use, or a malformed one. Set only after the
// op's validation (the program factory); output-spec inference runs before it and leaves errors to the validator.
std::optional<ring_joint::FloaterSchedule> floater_eligibility(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    const KSplitShape& ksplit,
    bool refuse_unusable_schedule);

// Whether a validated call's program runs the floaters (floater_eligibility on its K split).
bool runs_floaters(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args);

}  // namespace ttnn::prim::ring_joint_plan
