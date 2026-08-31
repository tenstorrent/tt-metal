// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "kernels/dataflow/chunked_prefill_utils.hpp"
#include "kernels/sliding_window_geometry.hpp"
#include "sliding_halo_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <cmath>
#include <string>
#include <deque>
#include <limits>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/common_values.hpp>

using namespace tt::tt_metal;
using ttnn::Tensor;

namespace {

namespace ag_rt = ttnn::ring_attention_all_gather_async_detail;
namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Host-side summary of which ring-loop iterations do useful SDPA work. Bits are indexed by ring_iter,
// not ring_id; kernels still advance their sync/ring-id sequence on every iter before checking the mask.
struct RingWorkMasks {
    uint32_t active_ring_iter_mask = 0;
    uint32_t single_valid_kv_chunk_mask = 0;
};

struct RingWorkPlan {
    RingWorkMasks masks;
    uint32_t last_active_ring_iter = 0;
};

struct KVPadQMapping {
    uint32_t q_pre_wrap_start_tile = 0;
    uint32_t q_pre_wrap_tile_count = 0;
    uint32_t q_post_wrap_start_tile = 0;
    uint32_t q_valid_tile_count = 0;
};

struct TileSegment {
    uint32_t start_tile = 0;
    uint32_t tile_count = 0;
};

struct RingJointRuntimePlan {
    uint32_t logical_nt = 0;
    KVPadQMapping kv_pad_q_mapping;
    RingWorkPlan ring_work_plan;
    bool kernel_chunked = false;
    bool kernel_is_causal = false;
};

struct RingJointRuntimeDerivation {
    uint32_t logical_nt = 0;
    uint32_t ring_size = 0;
    uint32_t q_local_padded_Nt = 0;
    uint32_t kv_local_padded_Nt = 0;
    uint32_t q_chunk_group_tile_count = 0;
    uint32_t num_local_k_chunks = 0;
    uint32_t k_chunk_tile_count = 0;
    // For sharded joint: per-iteration count (L_local / k_chunk_size). For replicated: full L count.
    uint32_t num_joint_k_chunks = 0;
    uint32_t joint_seq_len = 0;
    // Sharded-joint tail boundary (mirrors the kernel's kv_chunk_is_beyond_logical_l skip): a joint
    // shard whose global start tile (ring_id * joint_local_padded_Nt) is at/after logical_lt is pure
    // padding and must be treated as no work, exactly as the spatial path treats a beyond-logical_n shard.
    uint32_t logical_lt = 0;             // true (unpadded) joint length in tiles
    uint32_t joint_local_padded_Nt = 0;  // Lt_local: per-device joint tile count (sharded path)
    bool joint_is_sharded = false;
    bool kernel_chunked = false;
    bool kv_pad_rotation_enabled = false;
    bool kernel_is_causal = false;
};

struct RingJointRuntimeArgLayout {
    uint32_t reader_kv_cache_batch_idx = 0;
    uint32_t reader_logical_nt = 0;
    uint32_t reader_active_ring_iter_mask = 0;
    uint32_t writer_logical_nt = 0;
    uint32_t writer_active_ring_iter_mask = 0;
    uint32_t writer_single_valid_kv_chunk_mask = 0;
    uint32_t compute_logical_nt = 0;
    uint32_t compute_q_pre_wrap_start_tile = 0;
    uint32_t compute_q_pre_wrap_tile_count = 0;
    uint32_t compute_q_post_wrap_start_tile = 0;
    uint32_t compute_q_valid_tile_count = 0;
    uint32_t compute_active_ring_iter_mask = 0;
    CoreCoord grid_size = {0, 0};
};

struct RingWritePlan {
    uint32_t transport_rank = 0;
    uint32_t tensor_rank = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
};

uint32_t tensor_rank_from_transport_rank(const ttnn::prim::RingJointSDPAParams& args, uint32_t transport_rank) {
    const auto& ag = args.all_gather_operation_attributes;
    return ag.full_mesh ? ttnn::ccl::snake_ring::row_major_index(
                              transport_rank, ag.mesh_rows, ag.mesh_cols, ag.snake_orientation)
                        : transport_rank;
}

ttnn::operations::ccl::common::MeshRingPlan mesh_ring_plan_from_attributes(
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncParams& ag) {
    return {
        .cluster_axis = ag.cluster_axis,
        .full_mesh = ag.full_mesh,
        .orientation = ag.snake_orientation,
        .mesh_rows = ag.mesh_rows,
        .mesh_cols = ag.mesh_cols,
        .ring_size = ag.ring_size,
        .num_links = ag.num_links,
        .topology = ag.topology,
        .route_plan_hash = ag.route_plan_hash,
    };
}

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

// Resolved joint Q/K/V params + sequence lengths for both replicated and sharded-joint paths.
// Centralizes the optional-tensor / logical_l branching so descriptor construction and the
// runtime planners share one definition of L (full) and L_local (per-device).
RingJointInputParams resolve_ring_joint_input_params(
    const ttnn::prim::RingJointSDPAParams& args, const ttnn::prim::RingJointSDPAInputs& tensor_args) {
    RingJointInputParams joint_input_params;
    joint_input_params.has_joint_tensors = tensor_args.joint_q.has_value();
    joint_input_params.joint_is_sharded = tensor_args.joint_is_sharded();

    if (joint_input_params.has_joint_tensors) {
        joint_input_params.joint_q = &tensor_args.joint_q.value();
        joint_input_params.joint_k = &tensor_args.joint_k.value();
        joint_input_params.joint_v =
            tensor_args.joint_v.has_value() ? &tensor_args.joint_v.value() : joint_input_params.joint_k;
    }

    if (joint_input_params.joint_is_sharded) {
        joint_input_params.gathered_joint_k = &tensor_args.gathered_joint_k.value();
        joint_input_params.gathered_joint_v = &tensor_args.gathered_joint_v.value();
        // Per-shard physical length is the (padded, tile-aligned) joint Q shard on this device.
        // The padded total L is that shard times the ring; logical_l carries the true token count,
        // which may be smaller than L when the joint prompt does not fill the last shard. This
        // mirrors how the spatial path keeps logical_n separate from the padded gathered length.
        joint_input_params.L_local = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L =
            joint_input_params.L_local * static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
        joint_input_params.logical_l = static_cast<uint32_t>(args.logical_l);
    } else if (joint_input_params.has_joint_tensors) {
        joint_input_params.L = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L_local = joint_input_params.L;
        joint_input_params.logical_l = joint_input_params.L;
    }

    return joint_input_params;
}

constexpr uint32_t kReaderKernelIndex = 0;
constexpr uint32_t kWriterKernelIndex = 1;
constexpr uint32_t kComputeKernelIndex = 2;

// Dense all-gather appends reader-forward, writer-forward, reader-backward, writer-backward.
// Compact sliding appends only its predecessor reader/writer pair at indices 3 and 4.
constexpr uint32_t kAllGatherReaderForwardKernelIndex = 3;
constexpr uint32_t kAllGatherWriterForwardKernelIndex = 4;
constexpr uint32_t kAllGatherReaderBackwardKernelIndex = 5;
constexpr uint32_t kAllGatherWriterBackwardKernelIndex = 6;
constexpr uint32_t kNeighborHaloReaderKernelIndex = 3;
constexpr uint32_t kNeighborHaloWriterKernelIndex = 4;

// Runtime-arg offsets used by cache-hit patching. Descriptor construction appends the same slots through
// CheckedRuntimeArgList, so future layout edits fail on program creation instead of corrupting cache hits.
// Q, K, V, gathered K, gathered V, and attention sink (nullable).
constexpr uint32_t kReaderBaseBufferArgCount = 6;
constexpr uint32_t kReaderJointBufferArgCount = 3;
constexpr uint32_t kReaderQWorkArgCount = 3;
constexpr uint32_t kRingJointChainConfigArgCount = ring_joint::kChainConfigRuntimeArgCount;
constexpr uint32_t kRingJointChainCompileArgCount = ring_joint::kChainCompileArgCount;
constexpr uint32_t kRingJointChainSemaphoreCompileArgCount = ring_joint::kChainSemaphoreCompileArgCount;
constexpr uint32_t kRingJointChainMcastEnabledCompileArgOffset = ring_joint::kChainMcastEnabledCompileArgOffset;
constexpr uint32_t kReaderBatchChainExtraArgCount = 1;
constexpr uint32_t kReaderGQAChainExtraArgCount = 1;
constexpr uint32_t kWriterBaseArgCount = 5;
constexpr uint32_t kComputeRingSequencerArgCount = 6;

struct CheckedRuntimeArgList {
    KernelDescriptor::RTArgList args;
    uint32_t size = 0;

    template <typename T>
    void push_back(const T& value) {
        args.push_back(value);
        size++;
    }

    void append(const std::vector<uint32_t>& values) {
        args.append(values);
        size += values.size();
    }

    template <typename T>
    void push_checked(uint32_t expected_index, const T& value, const char* name) {
        TT_FATAL(
            size == expected_index,
            "RingJoint runtime arg {} expected index {}, got {} before append",
            name,
            expected_index,
            size);
        push_back(value);
    }
};

// Match the kernel's local-K to global-sequence tile mapping so the host can prune empty ring iters.
uint32_t kv_global_tile_for_host_ring_plan(
    bool is_chunked,
    uint32_t ring_id,
    uint32_t local_tile_start,
    uint32_t q_chunk_group_tile_count,
    uint32_t q_local_padded_tile_count,
    uint32_t kv_local_padded_tile_count) {
    if (is_chunked) {
        return chunked_kv_global_tile_for_local(
            ring_id, local_tile_start, q_chunk_group_tile_count, q_local_padded_tile_count);
    }
    return ring_id * kv_local_padded_tile_count + local_tile_start;
}

// Build the per-device ring-loop masks passed to reader/compute/writer. This mirrors the kernel
// ring-id order, marks ring_iter entries that have non-padded spatial or joint KV work, and applies
// the same causal unbalanced skip rule used by compute.
RingWorkPlan build_ring_work_plan(
    const ttnn::prim::RingJointSDPAParams& args,
    const RingWritePlan& ring_write_plan,
    const RingJointRuntimeDerivation& derivation,
    bool is_balanced) {
    RingWorkPlan plan;
    RingIdSequencer seq(
        ring_write_plan.transport_rank,
        derivation.ring_size,
        ring_write_plan.backward_writes_expected,
        ring_write_plan.forward_writes_expected);
    // RingIdSequencer accepts a sync callback for kernel semaphore waits. Host planning only needs the
    // same ring-id sequence, so use a no-op callback.
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < derivation.ring_size; ++ring_iter) {
        const uint32_t ring_id = tensor_rank_from_transport_rank(args, seq.get_next_ring_id(noop_sync));
        // Sharded joint: each ring iteration delivers one L/P shard immediately, so process
        // joint K/V on every ring iteration (no need to wait for the full gather to complete).
        // Replicated joint: process joint when ring_id == ring_size-1
        const bool has_joint_work = derivation.num_joint_k_chunks > 0 && derivation.joint_seq_len != 0;
        // Whether this ring iteration is a candidate to consume joint K/V at all (sharded: every iter;
        // replicated: when ring_id == ring_size-1, matching the kernel's do_joint_kv condition).
        const bool joint_iter_selected =
            has_joint_work && (derivation.joint_is_sharded || ring_id == derivation.ring_size - 1);
        // Count only the joint K chunks that carry REAL tokens, mirroring the kernel's
        // kv_chunk_is_beyond_logical_l skip: a joint chunk whose global start tile
        // (ring_id * joint_local_padded_Nt + k * k_chunk_tile_count) is at/after logical_lt is pure
        // padding. On the replicated path there is no per-shard tail (logical_lt == Lt), so all chunks
        // count.
        uint32_t valid_joint_kv_chunks = 0;
        if (joint_iter_selected) {
            if (derivation.joint_is_sharded) {
                for (uint32_t k = 0; k < derivation.num_joint_k_chunks; ++k) {
                    const uint32_t joint_global_start_tile =
                        ring_id * derivation.joint_local_padded_Nt + k * derivation.k_chunk_tile_count;
                    if (joint_global_start_tile < derivation.logical_lt) {
                        valid_joint_kv_chunks++;
                    }
                }
            } else {
                valid_joint_kv_chunks = derivation.num_joint_k_chunks;
            }
        }
        const bool joint_contributes = valid_joint_kv_chunks > 0;
        uint32_t valid_spatial_kv_chunks = 0;
        for (uint32_t k_chunk = 0; k_chunk < derivation.num_local_k_chunks; ++k_chunk) {
            const uint32_t local_tile_start = k_chunk * derivation.k_chunk_tile_count;
            if (local_tile_start >= derivation.kv_local_padded_Nt) {
                continue;
            }
            if (kv_global_tile_for_host_ring_plan(
                    derivation.kernel_chunked,
                    ring_id,
                    local_tile_start,
                    derivation.q_chunk_group_tile_count,
                    derivation.q_local_padded_Nt,
                    derivation.kv_local_padded_Nt) < derivation.logical_nt) {
                valid_spatial_kv_chunks++;
            }
        }
        const uint32_t valid_kv_chunks = valid_spatial_kv_chunks + valid_joint_kv_chunks;
        // Non-pad chunked prefill historically keeps every spatial ring iter active; KV-pad rotation
        // tightens this to valid chunks so empty pad slabs can be skipped.
        const bool has_kv_work =
            (derivation.kernel_chunked && !derivation.kv_pad_rotation_enabled) || valid_spatial_kv_chunks > 0;
        const bool ring_iter_does_work =
            (has_kv_work || joint_contributes) &&
            !(derivation.kernel_is_causal && ring_write_plan.tensor_rank < ring_id && !is_balanced);
        if (ring_iter_does_work) {
            plan.masks.active_ring_iter_mask |= (1u << ring_iter);
            plan.last_active_ring_iter = ring_iter;
        }
        if (valid_kv_chunks <= 1) {
            plan.masks.single_valid_kv_chunk_mask |= (1u << ring_iter);
        }
    }

    return plan;
}

KVPadQMapping build_kv_pad_q_mapping(
    uint32_t kv_actual_tile_count,
    uint32_t logical_tile_count,
    uint32_t ring_size,
    uint32_t q_local_padded_tile_count,
    uint32_t device_index) {
    // The current Q range is [kv_actual_tile_count, logical_tile_count). In KV-pad rotation it is packed into
    // this device's fixed Q tile slab, but it may straddle one global chunk-group boundary.
    // Store the first-group segment followed by the optional next-group segment; padded rows stay invalid.
    const uint32_t q_chunk_group_tile_count = ring_size * q_local_padded_tile_count;
    const uint32_t first_group = kv_actual_tile_count / q_chunk_group_tile_count;
    const uint32_t last_group = (logical_tile_count - 1) / q_chunk_group_tile_count;
    TT_FATAL(
        last_group <= first_group + 1,
        "KV-pad-aware rotation expects the current valid Q to fit in one fixed global chunk. "
        "Got kv_actual_tile_count={}, new_actual_tile_count={}, q_chunk_group_tile_count={}",
        kv_actual_tile_count,
        logical_tile_count - kv_actual_tile_count,
        q_chunk_group_tile_count);

    const auto intersect_device_group = [&](uint32_t group) -> TileSegment {
        const uint32_t block_start_tile = group * q_chunk_group_tile_count + device_index * q_local_padded_tile_count;
        const uint32_t block_end_tile = block_start_tile + q_local_padded_tile_count;
        const uint32_t start_tile = std::max(kv_actual_tile_count, block_start_tile);
        const uint32_t end_tile = std::min(logical_tile_count, block_end_tile);
        if (end_tile <= start_tile) {
            return {};
        }
        return TileSegment{start_tile, end_tile - start_tile};
    };

    const TileSegment first_segment = intersect_device_group(first_group);
    const TileSegment second_segment =
        last_group == first_group ? TileSegment{} : intersect_device_group(first_group + 1);

    KVPadQMapping mapping;
    mapping.q_pre_wrap_start_tile = first_segment.start_tile;
    mapping.q_pre_wrap_tile_count = first_segment.tile_count;
    mapping.q_post_wrap_start_tile = second_segment.start_tile;
    mapping.q_valid_tile_count = first_segment.tile_count + second_segment.tile_count;
    TT_FATAL(
        mapping.q_valid_tile_count <= q_local_padded_tile_count,
        "KV-pad-aware rotation mapped more valid Q tiles to this device than its local Q slab can hold. "
        "Got q_valid_tile_count={}, q_local_padded_tile_count={}, device_index={}",
        mapping.q_valid_tile_count,
        q_local_padded_tile_count,
        device_index);
    return mapping;
}

RingWritePlan build_ring_write_plan(
    const ttnn::prim::RingJointSDPAParams& args,
    const ttnn::prim::RingJointSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& coord) {
    RingWritePlan plan;
    const auto& ag = args.all_gather_operation_attributes;
    if (ag.full_mesh) {
        const auto position = ttnn::operations::ccl::common::get_mesh_ring_position(
            tensor_args.input_q, coord, mesh_ring_plan_from_attributes(ag));
        plan.transport_rank = position.transport_rank;
        plan.tensor_rank = position.tensor_rank;
        plan.forward_coord = position.forward_coord;
        plan.backward_coord = position.backward_coord;
    } else {
        plan.transport_rank =
            ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.input_q, coord, ag.cluster_axis);
        plan.tensor_rank = plan.transport_rank;
        plan.forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_q, coord, 1, ag.topology, ag.cluster_axis);
        plan.backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_q, coord, -1, ag.topology, ag.cluster_axis);
    }

    // Chunked sliding consumes the local slab followed by its cyclic
    // predecessor. Keep that dependency on direction 1 for every device,
    // independent of the dense ring's parity-based split.
    if (args.has_sliding_window() && tensor_args.is_chunked() && !args.is_cross) {
        plan.forward_writes_expected = 1;
        plan.backward_writes_expected = 0;
        return plan;
    }

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ttnn::ccl::get_forward_backward_configuration(
        args.all_gather_operation_attributes.ring_size,
        plan.transport_rank,
        args.all_gather_operation_attributes.topology);
    (void)dynamic_alternate;
    if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring && plan.transport_rank % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }

    if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Linear) {
        plan.forward_writes_expected = num_targets_backward;
        plan.backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(
            args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring,
            "Topology must be Linear or Ring");
        plan.forward_writes_expected = num_targets_forward;
        plan.backward_writes_expected = num_targets_backward;
    }

    return plan;
}

RingJointRuntimeDerivation build_runtime_derivation(
    const ttnn::prim::RingJointSDPAParams& args, const ttnn::prim::RingJointSDPAInputs& tensor_args) {
    const auto& q_shape = tensor_args.input_q.logical_shape();
    const uint32_t k_chunk_size = args.get_k_chunk_size();
    const uint32_t q_local_padded_N = q_shape[2];
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const RingJointInputParams joint_input_params = resolve_ring_joint_input_params(args, tensor_args);

    RingJointRuntimeDerivation derivation;
    derivation.logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    derivation.ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    derivation.q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    derivation.kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    derivation.q_chunk_group_tile_count = derivation.q_local_padded_Nt * derivation.ring_size;
    derivation.num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);
    derivation.k_chunk_tile_count = k_chunk_size / tt::constants::TILE_HEIGHT;
    // Sharded joint: each ring iteration delivers one L/P shard, so num_joint_k_chunks counts
    // chunks within a single shard (ceil(L_local / k_chunk_size)). Replicated: full L.
    derivation.joint_is_sharded = joint_input_params.joint_is_sharded;
    derivation.num_joint_k_chunks = tt::div_up(joint_input_params.L_local, k_chunk_size);
    derivation.joint_seq_len = joint_input_params.L;
    derivation.joint_local_padded_Nt = tt::div_up(joint_input_params.L_local, tt::constants::TILE_HEIGHT);
    derivation.logical_lt = tt::div_up(joint_input_params.logical_l, tt::constants::TILE_HEIGHT);
    // Cross is non-causal on chunked-shaped tensors, so kernels and the work planner use the
    // non-chunked path.
    derivation.kernel_chunked = tensor_args.is_chunked() && !args.is_cross;
    // The metadata path derives kv_actual_isl on-device for chunked prefill.
    derivation.kv_pad_rotation_enabled =
        args.has_kv_pad_rotation() || (tensor_args.has_metadata() && tensor_args.is_chunked());
    derivation.kernel_is_causal = args.is_causal && !derivation.kernel_chunked;

    TT_FATAL(
        derivation.ring_size <= std::numeric_limits<uint32_t>::digits,
        "Ring-joint host ring-work masks support up to {} ring iterations. Got ring_size={}",
        std::numeric_limits<uint32_t>::digits,
        derivation.ring_size);

    return derivation;
}

RingJointRuntimePlan build_runtime_plan(
    const ttnn::prim::RingJointSDPAParams& args,
    const ttnn::prim::RingJointSDPAInputs& tensor_args,
    const RingWritePlan& ring_write_plan) {
    const RingJointRuntimeDerivation derivation = build_runtime_derivation(args, tensor_args);

    RingJointRuntimePlan plan;
    plan.logical_nt = derivation.logical_nt;
    const uint32_t kv_actual_tile_count =
        args.kv_actual_isl.has_value() ? args.kv_actual_isl.value() / tt::constants::TILE_HEIGHT : 0;
    if (args.kv_actual_isl.has_value()) {
        plan.kv_pad_q_mapping = build_kv_pad_q_mapping(
            kv_actual_tile_count,
            derivation.logical_nt,
            derivation.ring_size,
            derivation.q_local_padded_Nt,
            ring_write_plan.tensor_rank);
    }

    if (args.has_sliding_window()) {
        // Sliding folds its local and predecessor ranges into one synthetic ring iteration.
        plan.ring_work_plan.masks.active_ring_iter_mask = 1;
    } else {
        plan.ring_work_plan = build_ring_work_plan(args, ring_write_plan, derivation, args.is_balanced);
    }
    plan.kernel_chunked = derivation.kernel_chunked;
    plan.kernel_is_causal = derivation.kernel_is_causal;
    return plan;
}

RingJointRuntimeArgLayout get_runtime_arg_layout(
    const ttnn::prim::RingJointSDPAParams& args, const ttnn::prim::RingJointSDPAInputs& tensor_args) {
    const auto& k_shape = tensor_args.gathered_k.logical_shape();
    const RingJointInputParams joint_input_params = resolve_ring_joint_input_params(args, tensor_args);
    const uint32_t NH = tensor_args.input_q.logical_shape()[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = tensor_args.v_num_heads();
    const bool v_shares_k_buffer = tensor_args.has_latent_v();
    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);

    RingJointRuntimeArgLayout layout;
    layout.grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                       : tensor_args.input_q.device()->compute_with_storage_grid_size();

    const uint32_t joint_buffer_args = (joint_input_params.L != 0) ? kReaderJointBufferArgCount : 0;
    // 2 extra buffer slots for gathered_joint_k/v when the sharded-joint path is active
    const uint32_t gathered_joint_buffer_args = joint_input_params.joint_is_sharded ? 2 : 0;
    const bool enable_kv_chains = !args.has_sliding_window();
    const bool use_head_chain = ring_joint::uses_v_head_chain(enable_kv_chains, gqa_grouped_kv, v_shares_k_buffer);
    const uint32_t head_chain_args = use_head_chain ? kRingJointChainConfigArgCount : 0;
    const uint32_t batch_chain_args =
        enable_kv_chains && k_uses_batch_chain ? (kRingJointChainConfigArgCount + kReaderBatchChainExtraArgCount) : 0;
    const uint32_t gqa_chain_args =
        enable_kv_chains && gqa_grouped_kv ? (kRingJointChainConfigArgCount + kReaderGQAChainExtraArgCount) : 0;
    layout.reader_kv_cache_batch_idx = kReaderBaseBufferArgCount + joint_buffer_args + gathered_joint_buffer_args + 2;
    layout.reader_logical_nt = kReaderBaseBufferArgCount + joint_buffer_args + gathered_joint_buffer_args +
                               kReaderQWorkArgCount + head_chain_args + batch_chain_args + gqa_chain_args;
    layout.reader_active_ring_iter_mask = layout.reader_logical_nt + 1;
    layout.writer_logical_nt = kWriterBaseArgCount;
    layout.writer_active_ring_iter_mask = layout.writer_logical_nt + 1;
    layout.writer_single_valid_kv_chunk_mask = layout.writer_active_ring_iter_mask + 1;
    layout.compute_logical_nt = kComputeRingSequencerArgCount;
    layout.compute_q_pre_wrap_start_tile = layout.compute_logical_nt + 1;
    layout.compute_q_pre_wrap_tile_count = layout.compute_q_pre_wrap_start_tile + 1;
    layout.compute_q_post_wrap_start_tile = layout.compute_q_pre_wrap_tile_count + 1;
    layout.compute_q_valid_tile_count = layout.compute_q_post_wrap_start_tile + 1;
    layout.compute_active_ring_iter_mask = layout.compute_q_valid_tile_count + 1;
    return layout;
}

void write_runtime_arg(RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(
        index < args.size(), "Missing RingJoint runtime arg {} at index {}; args.size()={}", name, index, args.size());
    args[index] = value;
}

// Tile-rows of the latent KV the fused all-gather must move for this chunk: the first
// ceil(logical_n / chunk_global) block-cyclic slabs (a contiguous per-device page prefix), so an
// oversized (growing) KV cache only moves kv_actual-sized data. Returns nullopt when KV-pad rotation
// is off (gather the full input). Shared by the descriptor-create path (so the first / cache-miss
// dispatch is bounded) and the cache-hit override path.
std::optional<uint32_t> compute_gather_valid_Ht(
    const ttnn::prim::RingJointSDPAParams& args, const ttnn::prim::RingJointSDPAInputs& tensor_args) {
    if (!args.has_kv_pad_rotation() && !(tensor_args.has_metadata() && tensor_args.is_chunked())) {
        return std::nullopt;
    }
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t n_local_q = tensor_args.input_q.padded_shape()[2];  // per-device Q slab (chunk_local)
    const uint32_t chunk_global = n_local_q * ring_size;
    const uint32_t valid_slabs = (static_cast<uint32_t>(args.logical_n) + chunk_global - 1) / chunk_global;
    return valid_slabs * (n_local_q / tt::constants::TILE_HEIGHT);
}

void apply_ring_joint_scalar_runtime_args(
    Program& program,
    const ttnn::prim::RingJointSDPAParams& args,
    const ttnn::prim::RingJointSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& mesh_dispatch_coordinate) {
    const bool patch_indexed_kv_cache = args.has_indexed_kv_cache();
    const bool patch_kv_pad_rotation = args.has_kv_pad_rotation();
    if (!patch_indexed_kv_cache && !patch_kv_pad_rotation) {
        return;
    }

    // Indexed KV-cache hits also need the current device index and logical sequence geometry for
    // the compact sliding halo. Build these plans for either kind of runtime patch; using default
    // plans here would incorrectly select device 0's halo tail when no KV-pad rotation is present.
    const RingWritePlan ring_write_plan = build_ring_write_plan(args, tensor_args, mesh_dispatch_coordinate);
    const RingJointRuntimePlan runtime_plan = build_runtime_plan(args, tensor_args, ring_write_plan);
    const RingWorkMasks& ring_work_masks = runtime_plan.ring_work_plan.masks;
    const RingJointRuntimeArgLayout layout = get_runtime_arg_layout(args, tensor_args);
    const uint32_t num_cores = layout.grid_size.x * layout.grid_size.y;
    const uint32_t kv_cache_batch_idx = args.kv_cache_batch_idx.value_or(0);

    // Gather inputs (K, plus V when it isn't the latent-V alias of K). Shared by the indexed-slot
    // and valid-pages patches below.
    const Tensor& input_k = tensor_args.input_k;
    const uint32_t num_ag_inputs = tensor_args.has_latent_v() ? 1u : (tensor_args.input_v.has_value() ? 2u : 1u);
    const std::array<const Tensor*, 2> ag_inputs = {
        &input_k, tensor_args.input_v.has_value() ? &tensor_args.input_v.value() : &input_k};
    const bool uses_neighbor_halo = args.has_sliding_window();

    // Re-patch the fused all-gather readers to gather the single cache slot `kv_cache_batch_idx`.
    // input_batch_base is uniform across all gather cores/links, so patch every core that runs the
    // reader. Mirrors the helper's create-time arithmetic so miss and hit paths agree.
    if (patch_indexed_kv_cache) {
        const auto patch_reader_batch_base = [&](uint32_t kernel_id,
                                                 uint32_t header_count,
                                                 uint32_t descriptor_field_count,
                                                 uint32_t batch_base_offset) {
            auto& grid_args = GetRuntimeArgs(program, kernel_id);  // [x][y] per-core args
            for (auto& col_args : grid_args) {
                for (auto& core_args : col_args) {
                    for (uint32_t in = 0; in < num_ag_inputs; ++in) {
                        const auto& shape = ag_inputs[in]->padded_shape();
                        const uint32_t num_heads = shape[1];
                        const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
                        const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
                        const uint32_t input_batch_base =
                            ag_rt::input_batch_base_pages(kv_cache_batch_idx, num_heads, Ht, Wt);
                        const uint32_t idx = header_count + in * descriptor_field_count + batch_base_offset;
                        if (core_args.size() > idx) {  // skip cores that don't run this kernel
                            write_runtime_arg(core_args, idx, input_batch_base, "all_gather_reader.input_batch_base");
                        }
                    }
                }
            }
        };
        if (uses_neighbor_halo) {
            patch_reader_batch_base(
                kNeighborHaloReaderKernelIndex,
                ag_rt::kNeighborReaderRuntimeArgHeaderCount,
                ag_rt::kNeighborReaderTensorDescriptorFieldCount,
                ag_rt::kNeighborReaderInputBatchBaseFieldOffset);
        } else {
            patch_reader_batch_base(
                kAllGatherReaderForwardKernelIndex,
                ag_rt::kReaderRuntimeArgHeaderCount,
                ag_rt::kTensorDescriptorFieldCount,
                ag_rt::kInputBatchBaseFieldOffset);
            patch_reader_batch_base(
                kAllGatherReaderBackwardKernelIndex,
                ag_rt::kReaderRuntimeArgHeaderCount,
                ag_rt::kTensorDescriptorFieldCount,
                ag_rt::kInputBatchBaseFieldOffset);
        }
    }

    // Bound the fused all-gather to the logical_n-valid slab prefix so an oversized (growing) KV
    // cache only moves kv_actual-sized data instead of the whole physical buffer. The cache is
    // block-cyclic / slab-major per device, so the valid tokens are the first
    // ceil(logical_n / chunk_global) slabs == a contiguous page prefix. valid_pages is uniform
    // across cores/links/devices, so producer/consumer page counts and the ring slice protocol stay
    // matched (the AG kernels clamp input_tile_id_end to it). Patch readers AND writers — both key
    // their loops off input_tile_id_end — at their respective header offsets (3 vs 5).
    if (patch_kv_pad_rotation && !uses_neighbor_halo) {
        const uint32_t gather_valid_Ht = compute_gather_valid_Ht(args, tensor_args).value();
        const auto patch_valid_pages = [&](uint32_t kernel_id,
                                           uint32_t header_count,
                                           uint32_t descriptor_field_count,
                                           uint32_t valid_pages_offset) {
            auto& grid_args = GetRuntimeArgs(program, kernel_id);  // [x][y] per-core args
            for (auto& col_args : grid_args) {
                for (auto& core_args : col_args) {
                    for (uint32_t in = 0; in < num_ag_inputs; ++in) {
                        const auto& shape = ag_inputs[in]->padded_shape();
                        const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
                        const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
                        const uint32_t valid_Ht = std::min(gather_valid_Ht, Ht);
                        const uint32_t valid_pages = valid_Ht * Wt;
                        const uint32_t idx = header_count + in * descriptor_field_count + valid_pages_offset;
                        if (core_args.size() > idx) {  // skip cores that don't run this kernel
                            write_runtime_arg(core_args, idx, valid_pages, "all_gather.valid_pages");
                        }
                    }
                }
            }
        };
        patch_valid_pages(
            kAllGatherReaderForwardKernelIndex,
            ag_rt::kReaderRuntimeArgHeaderCount,
            ag_rt::kTensorDescriptorFieldCount,
            ag_rt::kValidPagesFieldOffset);
        patch_valid_pages(
            kAllGatherWriterForwardKernelIndex,
            ag_rt::kWriterRuntimeArgHeaderCount,
            ag_rt::kTensorDescriptorFieldCount,
            ag_rt::kValidPagesFieldOffset);
        patch_valid_pages(
            kAllGatherReaderBackwardKernelIndex,
            ag_rt::kReaderRuntimeArgHeaderCount,
            ag_rt::kTensorDescriptorFieldCount,
            ag_rt::kValidPagesFieldOffset);
        patch_valid_pages(
            kAllGatherWriterBackwardKernelIndex,
            ag_rt::kWriterRuntimeArgHeaderCount,
            ag_rt::kTensorDescriptorFieldCount,
            ag_rt::kValidPagesFieldOffset);
    }

    if (args.has_sliding_window() && (patch_indexed_kv_cache || patch_kv_pad_rotation)) {
        const uint32_t runtime_ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
        const auto runtime_chunked_sliding_layout = ring_joint::build_chunked_sliding_halo_layout(
            tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT,
            args.get_k_chunk_size() / tt::constants::TILE_HEIGHT,
            args.sliding_window_size.value(),
            tt::constants::TILE_HEIGHT,
            runtime_ring_size,
            runtime_plan.logical_nt);
        TT_FATAL(runtime_chunked_sliding_layout.uses_neighbor_halo(), "Sliding attention requires a neighbor halo");
        // logical_n/kv_actual_isl are runtime-patched and excluded from the program hash. For
        // compact chunked sliding they also choose which cache-group tail the one-hop gather reads.
        // Relocate every per-link reader/writer slice from the descriptor's previous group to the current group.
        const uint32_t runtime_tail_start_Ht =
            runtime_chunked_sliding_layout.send_tail_start_tile(ring_write_plan.transport_rank);
        auto& reader_grid_args = GetRuntimeArgs(program, kNeighborHaloReaderKernelIndex);
        auto& writer_grid_args = GetRuntimeArgs(program, kNeighborHaloWriterKernelIndex);
        TT_FATAL(reader_grid_args.size() == writer_grid_args.size(), "Directional gather runtime grids disagree");
        for (uint32_t x = 0; x < reader_grid_args.size(); ++x) {
            TT_FATAL(
                reader_grid_args[x].size() == writer_grid_args[x].size(),
                "Directional gather runtime grid columns disagree");
            for (uint32_t y = 0; y < reader_grid_args[x].size(); ++y) {
                auto& reader_args = reader_grid_args[x][y];
                auto& writer_args = writer_grid_args[x][y];
                if (reader_args.size() == 0 && writer_args.size() == 0) {
                    continue;
                }
                TT_FATAL(
                    reader_args.size() != 0 && writer_args.size() != 0, "Directional gather worker pair is incomplete");
                for (uint32_t in = 0; in < num_ag_inputs; ++in) {
                    const uint32_t reader_base = ag_rt::kNeighborReaderRuntimeArgHeaderCount +
                                                 in * ag_rt::kNeighborReaderTensorDescriptorFieldCount;
                    const uint32_t writer_base = ag_rt::kNeighborWriterRuntimeArgHeaderCount +
                                                 in * ag_rt::kNeighborWriterTensorDescriptorFieldCount;
                    const uint32_t writer_origin_idx = writer_base + ag_rt::kNeighborWriterInputOriginPageFieldOffset;
                    TT_FATAL(
                        writer_args.size() > writer_origin_idx, "Directional gather writer descriptor is incomplete");
                    const uint32_t input_Wt = ag_inputs[in]->padded_shape()[3] / tt::constants::TILE_WIDTH;
                    const uint32_t runtime_origin_page = runtime_tail_start_Ht * input_Wt;
                    const uint32_t previous_origin_page = writer_args[writer_origin_idx];
                    if (previous_origin_page != runtime_origin_page) {
                        const int64_t page_delta = static_cast<int64_t>(runtime_origin_page) - previous_origin_page;
                        const auto relocate_pages = [&](auto& runtime_args, uint32_t start_idx, uint32_t end_idx) {
                            const int64_t relocated_start = static_cast<int64_t>(runtime_args[start_idx]) + page_delta;
                            const int64_t relocated_end = static_cast<int64_t>(runtime_args[end_idx]) + page_delta;
                            TT_FATAL(
                                relocated_start >= 0 && relocated_end >= relocated_start,
                                "Invalid cached neighbor-halo relocation");
                            runtime_args[start_idx] = static_cast<uint32_t>(relocated_start);
                            runtime_args[end_idx] = static_cast<uint32_t>(relocated_end);
                        };
                        relocate_pages(
                            reader_args,
                            reader_base + ag_rt::kNeighborReaderInputTileStartFieldOffset,
                            reader_base + ag_rt::kNeighborReaderInputTileEndFieldOffset);
                        relocate_pages(
                            writer_args,
                            writer_base + ag_rt::kNeighborWriterInputTileStartFieldOffset,
                            writer_base + ag_rt::kNeighborWriterInputTileEndFieldOffset);
                    }
                    writer_args[writer_origin_idx] = runtime_origin_page;
                }
            }
        }
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i % layout.grid_size.x, i / layout.grid_size.x};

        // Patch EVERY core, exactly as the create-time build sets these scalars on all cores
        // unconditionally. A core with no Q chunks (global_q_start == global_q_end) is NOT dead: in the
        // GQA / shared-K row-wide multicast path it runs padded handshake iterations (loop_q_count =
        // *_max_q_per_core) so the injector's mcast rectangle never targets a silent worker. Every such
        // iteration is gated by active_ring_iter_mask (ring_joint_reader.cpp), so a stale mask makes the
        // padded receiver skip a ring iter the injector still multicasts to — the injector then blocks
        // forever waiting for that receiver's ready signal. Previously these cores were skipped on the
        // assumption their scalars were dead; that held only while every dispatch shared the create-time
        // logical_n. When logical_n grows across dispatches that reuse one cached program (chunked-prefill
        // accumulation), the create-miss mask is stale for later hits, deadlocking the mcast handshake
        // (RingJointSDPA hang, all-gather eth reads left undrained).
        auto& compute_args = GetRuntimeArgs(program, kComputeKernelIndex, core);

        auto& reader_args = GetRuntimeArgs(program, kReaderKernelIndex, core);
        if (patch_indexed_kv_cache) {
            write_runtime_arg(
                reader_args, layout.reader_kv_cache_batch_idx, kv_cache_batch_idx, "reader.kv_cache_batch_idx");
        }
        if (!patch_kv_pad_rotation) {
            continue;
        }

        write_runtime_arg(reader_args, layout.reader_logical_nt, runtime_plan.logical_nt, "reader.logical_nt");
        write_runtime_arg(
            reader_args,
            layout.reader_active_ring_iter_mask,
            ring_work_masks.active_ring_iter_mask,
            "reader.active_ring_iter_mask");

        auto& writer_args = GetRuntimeArgs(program, kWriterKernelIndex, core);
        write_runtime_arg(writer_args, layout.writer_logical_nt, runtime_plan.logical_nt, "writer.logical_nt");
        write_runtime_arg(
            writer_args,
            layout.writer_active_ring_iter_mask,
            ring_work_masks.active_ring_iter_mask,
            "writer.active_ring_iter_mask");
        write_runtime_arg(
            writer_args,
            layout.writer_single_valid_kv_chunk_mask,
            ring_work_masks.single_valid_kv_chunk_mask,
            "writer.single_valid_kv_chunk_mask");

        write_runtime_arg(compute_args, layout.compute_logical_nt, runtime_plan.logical_nt, "compute.logical_nt");
        write_runtime_arg(
            compute_args,
            layout.compute_q_pre_wrap_start_tile,
            runtime_plan.kv_pad_q_mapping.q_pre_wrap_start_tile,
            "compute.q_pre_wrap_start_tile");
        write_runtime_arg(
            compute_args,
            layout.compute_q_pre_wrap_tile_count,
            runtime_plan.kv_pad_q_mapping.q_pre_wrap_tile_count,
            "compute.q_pre_wrap_tile_count");
        write_runtime_arg(
            compute_args,
            layout.compute_q_post_wrap_start_tile,
            runtime_plan.kv_pad_q_mapping.q_post_wrap_start_tile,
            "compute.q_post_wrap_start_tile");
        write_runtime_arg(
            compute_args,
            layout.compute_q_valid_tile_count,
            runtime_plan.kv_pad_q_mapping.q_valid_tile_count,
            "compute.q_valid_tile_count");
        write_runtime_arg(
            compute_args,
            layout.compute_active_ring_iter_mask,
            ring_work_masks.active_ring_iter_mask,
            "compute.active_ring_iter_mask");
    }
}

}  // namespace

namespace ttnn::prim {

namespace {

// Per-coord ProgramDescriptor build. Pulled into an anonymous-namespace helper so
// create_workload_descriptor() can loop coords and reuse this body verbatim. The
// op-specific name suffix avoids Unity-build collisions with the sibling ring
// sdpa factories that share the same helper signature.
// Descriptor construction must keep host/runtime argument layouts together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
tt::tt_metal::ProgramDescriptor build_ring_joint_sdpa_program_descriptor(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "build_ring_joint_sdpa_program_descriptor requires mesh_dispatch_coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    /*
    The QKV inputs are fractured on the sequence dimension across ring_size.
    The sequence length comes in padded such that it is divisible by `TILE_HEIGHT * ring_size`.
    Therefore each device has `padded_N / ring_size` local tokens.

    Naming:
        - padded_N: the global, padded sequence length
        - kv_local_padded_N: local shard of padded sequence length for K/V (== padded_N / ring_size)
        - q_local_padded_N: local Q seq length. For chunked prefill < kv_local_padded_N; otherwise equal.
        - logical_n: the logical global sequence length. logical_n <= padded_N.
        - L: the full (global) joint sequence length
        - L_local: per-device joint length. Equals L/ring_size on the sharded path, or L on the replicated path.

    input_tensor_q: B x NH  x q_local_padded_N  x DH
    input_tensor_k: B x NHK x kv_local_padded_N x DH
    input_tensor_v: B x NH  x kv_local_padded_N x DH

    gathered_input_tensor_k: B x NHK x padded_N x DH
    gathered_input_tensor_v: B x NH  x padded_N x DH

    Replicated joint path (logical_l == 0 or joint seq == L):
        joint_tensor_q/k/v: B x NH x L x DH  (full joint on every device)
        joint_output_tensor: B x NH x L x DH

    Sharded joint path (logical_l > 0 and joint seq == L / ring_size):
        joint_tensor_q/k/v: B x NH x L_local x DH  (one shard per device)
        gathered_joint_k/v: B x NH x L x DH         (scratch buffer; filled by fused all-gather)
        joint_output_tensor: B x NH x L_local x DH

    output_tensor: B x NH x q_local_padded_N x DH

    The algorithm is roughly described below.
    - for each ring iteration:
        - read a Q chunk from input_tensor_q
        - for each KV chunk in kv_local_padded_N:
            - on the first ring iteration, read from local input_tensor_k and input_tensor_v
            - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
            - Replicated joint: when ring_id == ring_size-1, also read from joint_tensor_k/v (full L).
            - Sharded joint: on every ring iteration, read one L_local shard from gathered_joint_k/v
              (or from the local joint_tensor_k/v when ring_id == this device's ring_index).
            - if the KV chunk is from the non-joint input and contains the global token index (logical_n - 1),
    generate a mask
            - else if the KV chunk is from non-joint input and contains the local token index (kv_local_padded_N - 1),
    generate an attention mask
            - else if the KV chunk is from the joint input and contains the local token index (L_local - 1),
    generate a mask
            - compute attention
        - write the output Q chunk
        - if this is not the first ring iteration, do the LSE update.
    */

    log_debug(tt::LogOp, "RingJointSDPA create_descriptor");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const bool v_shares_k_buffer = tensor_args.has_latent_v();
    const auto& input_tensor_v = tensor_args.input_v.has_value() ? tensor_args.input_v.value() : input_tensor_k;

    const RingJointInputParams joint_input_params = resolve_ring_joint_input_params(args, tensor_args);
    const Tensor* joint_tensor_q = joint_input_params.joint_q;
    const Tensor* joint_tensor_k = joint_input_params.joint_k;
    const Tensor* joint_tensor_v = joint_input_params.joint_v;
    const Tensor* gathered_joint_tensor_k = joint_input_params.gathered_joint_k;
    const Tensor* gathered_joint_tensor_v = joint_input_params.gathered_joint_v;
    const bool joint_is_sharded = joint_input_params.joint_is_sharded;

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v =
        tensor_args.gathered_v.has_value() ? tensor_args.gathered_v.value() : gathered_input_tensor_k;
    const auto& attention_sink = tensor_args.attention_sink;
    const bool use_attention_sink = attention_sink.has_value();

    auto& output_tensor = output_tensors[RING_JOINT_SDPA_OUTPUT_IDX];
    auto& joint_output_tensor = output_tensors[RING_JOINT_SDPA_JOINT_OUTPUT_IDX];
    auto& stats_output_tensor = output_tensors[RING_JOINT_SDPA_STATS_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::ProgramDescriptor desc;

    auto* mesh_device = input_tensor_q.device();
    const RingWritePlan ring_write_plan = build_ring_write_plan(args, tensor_args, coord);
    const uint32_t transport_rank = ring_write_plan.transport_rank;
    const uint32_t tensor_rank = ring_write_plan.tensor_rank;
    const uint32_t forward_writes_expected = ring_write_plan.forward_writes_expected;
    const uint32_t backward_writes_expected = ring_write_plan.backward_writes_expected;
    const auto& forward_coord = ring_write_plan.forward_coord;
    const auto& backward_coord = ring_write_plan.backward_coord;

    log_debug(tt::LogOp, "transport rank: {}, tensor rank: {}", transport_rank, tensor_rank);
    log_debug(tt::LogOp, "is_causal: {}", args.is_causal);
    log_debug(tt::LogOp, "is_balanced: {}", args.is_balanced);

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::optional<ttnn::prim::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = ttnn::prim::RingSDPAFusedOpSignaler();

    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        args.all_gather_operation_attributes.ring_size,
        transport_rank,
        forward_writes_expected,
        backward_writes_expected);

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();

    log_debug(tt::LogOp, "q_shape: {}", q_shape);
    log_debug(tt::LogOp, "k_shape (gathered): {}", k_shape);
    if (tensor_args.gathered_v.has_value()) {
        log_debug(tt::LogOp, "v_shape (gathered): {}", tensor_args.gathered_v->logical_shape());
    } else {
        log_debug(
            tt::LogOp,
            "v_shape (latent): [B={}, NHV=1, N=0, DH={}]",
            q_shape[0],
            tensor_args.v_head_dim(args.latent_v_head_dim));
    }

    // q_local_padded_N (Q rows per device) can be shorter than kv_local_padded_N for chunked prefill.
    // Metadata uses an on-device cache-slot value, but needs the same single-slot program structure.
    const bool slot_from_metadata = tensor_args.has_metadata();
    const bool indexed_kv_cache = args.has_indexed_kv_cache() || slot_from_metadata;
    // Latent-V mode: V tensors are omitted; the reader reuses K's buffer and
    // reads only the first vDHt head-dim tiles.
    const uint32_t B = q_shape[0];
    const uint32_t NH = q_shape[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = tensor_args.v_num_heads();
    const uint32_t DH = q_shape[3];
    const uint32_t q_local_padded_N = q_shape[2];
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t gathered_padded_N = k_shape[2];
    const uint32_t global_padded_N = kv_local_padded_N * ring_size;
    const uint32_t sliding_window_size = args.sliding_window_size.value_or(0);
    const bool has_sliding_window = sliding_window_size > 0;
    const bool enable_kv_chains = !has_sliding_window;
    // The supported sliding specialization always uses a compact neighbor-halo buffer.
    const uint32_t padded_N = has_sliding_window ? global_padded_N : gathered_padded_N;
    const uint32_t kv_cache_batch_idx = args.kv_cache_batch_idx.value_or(0);
    // L / L_local resolved once in resolve_ring_joint_input_params (full vs per-device joint seq).
    const uint32_t L = joint_input_params.L;
    const uint32_t L_local = joint_input_params.L_local;
    // True (unpadded) joint token count. Equals L on the replicated/aligned path; smaller than the
    // padded L when the sharded joint prompt leaves pad rows on the global tail.
    const uint32_t logical_l = joint_input_params.logical_l;
    const uint32_t vDH = tensor_args.v_head_dim(args.latent_v_head_dim);
    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    const bool use_head_chain = ring_joint::uses_v_head_chain(enable_kv_chains, gqa_grouped_kv, v_shares_k_buffer);
    // The store-and-forward chains are scheduled per head, not per (batch, head).
    // Until their batch-aware scheduling is restored, multi-batch requests read K/V
    // independently on each core. This preserves the established B>1 functional path.
    const bool build_kv_chains = enable_kv_chains && B == 1;

    const uint32_t q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Nt = padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t gathered_padded_Nt = gathered_padded_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t Lt_local = tt::div_up(L_local, tt::constants::TILE_HEIGHT);
    // True joint length in tiles (number of tiles holding any real joint token). Drives the
    // per-ring-iteration joint tail mask, mirroring logical_nt for the spatial path.
    const uint32_t logical_lt = tt::div_up(logical_l, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const bool kv_pad_from_metadata = tensor_args.has_metadata() && tensor_args.is_chunked();
    const bool kv_pad_rotation_enabled = args.has_kv_pad_rotation() || kv_pad_from_metadata;
    const RingJointRuntimePlan runtime_plan = build_runtime_plan(args, tensor_args, ring_write_plan);
    const RingJointRuntimeArgLayout runtime_arg_layout = get_runtime_arg_layout(args, tensor_args);
    const uint32_t logical_nt = runtime_plan.logical_nt;
    const KVPadQMapping& kv_pad_q_mapping = runtime_plan.kv_pad_q_mapping;

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Chunked-prefill balanced layout: each device holds one per-chunk K region per chunk.
    // The region is q_local_padded_Nt tiles (Q is exactly one such region per call). The
    // group size below is that Q-sized region across all devices.
    // diagonal-tile CB slot is shared with is_causal — needed whenever either is on.
    const uint32_t q_chunk_group_tile_count = q_local_padded_Nt * ring_size;
    // kernel_chunked drives the chunked-prefill math in the kernels and the host ring-work planner.
    // Cross runs the non-causal full-prefill path on chunked-shaped tensors, so it is excluded; the
    // kernel-level is_causal flag carries the legacy local-frame causal-stamp semantics (chunked
    // prefill supersedes it via absolute-coords stamps). Both are derived once in build_runtime_plan.
    const bool kernel_chunked = runtime_plan.kernel_chunked;
    const bool kernel_is_causal = runtime_plan.kernel_is_causal;
    ring_joint::ChunkedSlidingHaloLayout chunked_sliding_halo_layout;
    if (has_sliding_window) {
        chunked_sliding_halo_layout = ring_joint::build_chunked_sliding_halo_layout(
            q_local_padded_Nt, Sk_chunk_t, sliding_window_size, tt::constants::TILE_HEIGHT, ring_size, logical_nt);
        TT_FATAL(
            kernel_chunked && chunked_sliding_halo_layout.uses_neighbor_halo(),
            "Sliding K/V requires neighbor-halo geometry; gathered rows={}, global rows={}",
            gathered_padded_N,
            global_padded_N);
        TT_FATAL(
            gathered_padded_N < global_padded_N,
            "Sliding K/V requires a compact halo buffer; gathered rows={}, global rows={}",
            gathered_padded_N,
            global_padded_N);
        TT_FATAL(
            gathered_padded_Nt >= chunked_sliding_halo_layout.halo_tile_rows,
            "Compact sliding K/V buffer has {} tile rows but requires at least {}",
            gathered_padded_Nt,
            chunked_sliding_halo_layout.halo_tile_rows);
    }
    const bool diag_tile_enabled = (args.is_causal || kernel_chunked) && !has_sliding_window;

    // Lightweight mask: needed when any K/joint dimension has padding, or when causal/chunked
    // masking is active.
    const bool local_n_has_padding = (kv_local_padded_Nt % Sk_chunk_t) != 0;
    const uint32_t global_n_partial_col = args.logical_n % tt::constants::TILE_HEIGHT;
    const uint32_t compile_time_logical_n = kv_pad_rotation_enabled ? 0 : static_cast<uint32_t>(args.logical_n);
    const uint32_t compile_time_logical_nt = kv_pad_rotation_enabled ? 0 : logical_nt;
    const uint32_t compile_time_global_n_partial_col = kv_pad_rotation_enabled ? 0 : global_n_partial_col;

    const bool global_n_has_padding = (compile_time_logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    // Joint masking mirrors spatial's TWO independent enable flags (local_n AND global_n), not just
    // global_n. The first term is the local_n analogue: when the K-chunk is wider than the per-device
    // joint shard (Lt_local % Sk_chunk_t != 0) every chunk carries fully-padded trailing tiles that
    // must be generated/masked (e.g. wadada Lt_local=2, Sk_chunk_t=16 -> 14 pad tiles per shard).
    // Omitting it — keying only off the logical tail — was the wadada PCC regression. The second term
    // is the global_n analogue: real joint tokens do not fill the last real shard's chunk, covering
    // fully-padded trailing tiles and a sub-tile partial column (logical_l=63, L_local=32: 63 % 32 != 0).
    const bool joint_has_padding =
        L > 0 && (((Lt_local % Sk_chunk_t) != 0) || ((logical_l % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0));
    const bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled || has_sliding_window;

    // Partial tile support when the joint padding boundary falls inside a tile. Uses the true
    // logical length (logical_l % TILE_HEIGHT), mirroring global_n_partial_col = logical_n % TILE.
    const uint32_t joint_l_partial_col = logical_l % tt::constants::TILE_HEIGHT;
    const uint32_t partial_mask_tiles =
        (compile_time_global_n_partial_col != 0 ? 1 : 0) + (joint_l_partial_col != 0 ? 1 : 0);
    const uint32_t edge_mask_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : (diag_tile_enabled ? 1 : 0);
    // Single CB holds neginf, either the causal diagonal or sliding edge palette, and partial masks.
    const uint32_t total_lightweight_mask_tiles = 1 + edge_mask_tiles + partial_mask_tiles;

    const uint32_t num_local_q_chunks = tt::div_up(q_local_padded_N, q_chunk_size);
    // Q chunking uses L_local (per-device shard on sharded path, full L on replicated).
    const uint32_t num_joint_q_chunks = tt::div_up(L_local, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);
    // Sharded joint: per-iteration K chunk count for one shard (L_local). Replicated: full L.
    // Kernels process this many joint K chunks on every ring iteration (sharded) or just the last (replicated).
    const uint32_t num_joint_k_chunks = tt::div_up(L_local, k_chunk_size);

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "NHK: {}", NHK);
    log_debug(tt::LogOp, "NHV: {}", NHV);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);
    log_debug(tt::LogOp, "vDH: {}", vDH);

    // Log padded dimensions
    log_debug(tt::LogOp, "q_local_padded_N: {}", q_local_padded_N);
    log_debug(tt::LogOp, "kv_local_padded_N: {}", kv_local_padded_N);
    log_debug(tt::LogOp, "padded_N: {}", padded_N);
    log_debug(tt::LogOp, "L: {}", L);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "vDHt: {}", vDHt);
    log_debug(tt::LogOp, "q_local_padded_Nt: {}", q_local_padded_Nt);
    log_debug(tt::LogOp, "kv_local_padded_Nt: {}", kv_local_padded_Nt);
    log_debug(tt::LogOp, "padded_Nt: {}", padded_Nt);
    log_debug(tt::LogOp, "Lt: {}", Lt);

    // Log chunking parameters
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "num_local_q_chunks: {}", num_local_q_chunks);
    log_debug(tt::LogOp, "num_joint_q_chunks: {}", num_joint_q_chunks);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "num_q_chunks: {}", num_q_chunks);
    log_debug(tt::LogOp, "num_local_k_chunks: {}", num_local_k_chunks);
    log_debug(tt::LogOp, "num_joint_k_chunks: {}", num_joint_k_chunks);

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : mesh_device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet core_grid_set(core_grid);
    uint32_t num_cores = grid_size.x * grid_size.y;

    // Init fused op signaler — descriptor-pattern equivalent of
    // RingSDPAFusedOpSignaler::init_fused_op. The signaler stores the receiver-cores
    // noc list and two signal semaphore IDs (one for forward, one for backward).
    // Semaphore IDs match insertion order into desc.semaphores.
    {
        sdpa_fused_op_signaler->fused_op_signaler_mode = ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc.clear();
        const auto cores = tt::tt_metal::corerange_to_cores(core_grid_set, std::nullopt, /*row_wise=*/true);
        for (const auto& core : cores) {
            sdpa_fused_op_signaler->fused_op_receiver_cores_noc.push_back(
                mesh_device->worker_core_from_logical_core(core));
        }
        const uint32_t fused_sem0_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = fused_sem0_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid_set,
            .initial_value = 0,
        });
        const uint32_t fused_sem1_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = fused_sem1_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid_set,
            .initial_value = 0,
        });
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.clear();
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.push_back(fused_sem0_id);
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.push_back(fused_sem1_id);
        sdpa_fused_op_signaler->num_fused_op_cores_to_signal =
            sdpa_fused_op_signaler->fused_op_receiver_cores_noc.size();
        sdpa_fused_op_signaler->initialized_fused_op = true;
    }

    // Single host-derived split-forwarding decision, shared with the all-gather (passed to the
    // helper below), so producer and consumer cannot disagree. Latent-V stays on the established
    // unsplit protocol (its cache-replay consumption would deadlock waiting for a second half);
    // sliding-window consumes shards via get_next_ring_id_and_consume_one_signal, which has no
    // split second-half wait.
    sdpa_fused_op_signaler->split_forwarding_enabled =
        (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring) &&
        (args.all_gather_operation_attributes.ring_size % 2 == 0) &&
        (args.all_gather_operation_attributes.ring_size > 2) && !tensor_args.has_latent_v() &&
        !args.has_sliding_window();

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(
        tt::LogOp, "mesh_device->compute_with_storage_grid_size(): {}", mesh_device->compute_with_storage_grid_size());
    log_debug(tt::LogOp, "grid_size: {}", grid_size);

    TT_FATAL(
        num_cores <= mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y);

    /**
     * This parallelization scheme is efficient because it divides the global work,
     * the total number of Q chunks across all batches and heads, evenly across the cores.
     *
     */
    const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
    const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    const uint32_t q_buffer_factor = (max_q_per_core > 1) ? 2 : 1;

    log_debug(tt::LogOp, "max_q_per_core: {}", max_q_per_core);

    // In-place latent-V optimization: when the Q chunk is a single tile (Sq_chunk_t==1) the
    // second matmul (softmax@V) is data-movement bound, so instead of materializing V from K^T
    // (an L1->L1 transfer) we read the first vDHt rows of K^T directly. V is never produced and the
    // phase-2 matmul consumes one V column tile per issue (out_subblock_w=1). The kernels derive the
    // same predicate from their compile-time args via the shared kt_inplace_v_enabled() helper.
    const bool kt_inplace_v = kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    // Latent-V keeps the K CB triple-buffered. With V rematerialized, the 3rd slot let the reader
    // build the next V while compute consumed the current one; with in-place latent V (kt_inplace_v)
    // there is no V entry, but the 3rd K^T slot still buys prefetch slack that hides the reader's
    // NoC latency tail — measured ~+3pt math util on the dv512 q32 shape vs double-buffering.
    uint32_t k_tiles = Sk_chunk_t * DHt * (v_shares_k_buffer ? 3 : 2);
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;  // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "mask_tiles: {}", mask_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    TT_FATAL(
        Sq_chunk_t % qk_out_subblock_h == 0,
        "Sq_chunk_t ({}) must be divisible by qk_out_subblock_h ({})",
        Sq_chunk_t,
        qk_out_subblock_h);
    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Ring-joint streaming supports single-Q-subblock shapes; only fp32 dest acc stays on the legacy path.
    const bool use_streaming_compute = !fp32_dest_acc_en;
    TT_FATAL(
        !kv_pad_rotation_enabled || use_streaming_compute,
        "kv_actual_isl requires the ring-joint streaming compute path; the compute_common.hpp path selected by "
        "fp32_dest_acc_en=true is not supported.");
    // Sharded joint with a padded tail (logical_l < padded L) needs the reader to skip joint K chunks
    // beyond the real tail. That skip is mirrored only in the streaming compute path (sdpa_ring_v2);
    // the legacy fp32 path (sdpa_ring/sdpa_inner_loop) would leave compute waiting on K/V chunks the
    // reader never pushed. Require streaming rather than risk a deadlock.
    TT_FATAL(
        !(joint_is_sharded && logical_l < L) || use_streaming_compute,
        "Sharded joint with a padded joint tail (logical_l {} < padded L {}) requires the streaming compute "
        "path (set fp32_dest_acc_en=false)",
        logical_l,
        L);
    TT_FATAL(
        use_streaming_compute || !v_shares_k_buffer,
        "Latent-V ring attention is implemented only for streaming compute (fp32_dest_acc_en must be false)");
    log_debug(
        tt::LogOp,
        "use_streaming_compute: {} (is_causal={}, Sq_chunk_t={}, Sk_chunk_t={}, sbh={}, sbw={})",
        use_streaming_compute,
        args.is_causal,
        Sq_chunk_t,
        Sk_chunk_t,
        qk_out_subblock_h,
        qk_out_subblock_w);

    // In-place latent-V reads non-contiguous K^T rows as V columns, so the phase-2 matmul must
    // emit exactly one output column tile per issue (max_subblock_w=1).
    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(
        Sq_chunk_t,
        vDHt,
        dst_size,
        /*max_subblock_h=*/use_streaming_compute ? 2 : UINT32_MAX,
        /*max_subblock_w=*/kt_inplace_v ? 1u : UINT32_MAX);
    // Streaming compute may widen the QKT@V row group beyond the host matmul subblock
    // height for odd Q chunks. The writer must drain cb_out with the same row-group
    // cadence that compute pushes, otherwise deferred-save rows can be popped and
    // reused before the matching grouped write has safely landed.
    const uint32_t writer_out_row_group_h =
        use_streaming_compute
            ? ttnn::transformer::sdpa::streaming_qktv_h(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t)
            : out_out_subblock_h;

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;

    // Streaming: shrink cb_out to a 2-slot ping-pong (see sdpa_subblock_utils.hpp), unless either:
    //  - Phase-2's save_to_staging branch can fire (packs at offset qktv_h*vDHt, overruns the
    //    2*qktv_h*vDHt buffer on the 2nd Q chunk): gated by ring_size==1 || max_q_per_core==1.
    //  - the full Sq_chunk_t*vDHt output doesn't fit in 2 row groups: Phase-2 reserves it in a
    //    single reserve_back, which then blocks forever (deadlock seen at q_chunk=256 causal).
    // Otherwise keep the full-size cb_out (the default path).
    const bool streaming_shrink_fits = Sq_chunk_t <= 2 * writer_out_row_group_h;
    const bool streaming_shrink_safe = use_streaming_compute &&
                                       (args.all_gather_operation_attributes.ring_size == 1 || max_q_per_core == 1) &&
                                       streaming_shrink_fits;
    if (streaming_shrink_safe) {
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming cb_out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }
    log_debug(tt::LogOp, "streaming_shrink_safe={} out0_t={}", streaming_shrink_safe, out0_t);

    // log all values
    log_debug(tt::LogOp, "dst_size: {}", dst_size);
    log_debug(tt::LogOp, "qk_in0_block_w: {}", qk_in0_block_w);
    log_debug(tt::LogOp, "qk_out_subblock_w: {}", qk_out_subblock_w);
    log_debug(tt::LogOp, "qk_out_subblock_h: {}", qk_out_subblock_h);
    log_debug(tt::LogOp, "qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_debug(tt::LogOp, "qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_debug(tt::LogOp, "qk_num_blocks: {}", qk_num_blocks);
    log_debug(tt::LogOp, "out_in0_block_w: {}", out_in0_block_w);
    log_debug(tt::LogOp, "out_out_subblock_w: {}", out_out_subblock_w);
    log_debug(tt::LogOp, "out_out_subblock_h: {}", out_out_subblock_h);
    log_debug(tt::LogOp, "out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_debug(tt::LogOp, "out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_debug(tt::LogOp, "out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    // DHT_GRANULARITY is used in the kernel with both DHt and vDHt as the cols parameter,
    // so the granularity must evenly divide both to avoid dropping tiles.
    uint32_t dht_granularity = std::min({DHt, vDHt, dst_size});
    while (dht_granularity > 1 && (DHt % dht_granularity != 0 || vDHt % dht_granularity != 0)) {
        dht_granularity--;
    }
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    // Log these
    log_debug(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "reduce_granularity: {}", reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const float scale_value = scale.value_or(1.0f);
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);

    // log scale
    log_debug(tt::LogOp, "scale: {}", scale_value);

    // Enable per-head zigzag for load balancing in balanced causal mode
    // Requires even num_q_chunks for symmetric light/heavy work distribution
    // Chunked prefill rides its own absolute-coords path, not the legacy local-frame causal stamp,
    // so the zigzag asymmetry doesn't apply — gate on kernel_is_causal, not args.is_causal.
    const bool enable_zigzag_balancing = args.is_balanced && kernel_is_causal && (num_q_chunks % 2 == 0);

    // The masks let kernels skip ring iterations that contain only padded KV, while preserving the
    // per-iteration sync order described in RingWorkPlan.
    const RingWorkPlan& ring_work_plan = runtime_plan.ring_work_plan;
    const uint32_t active_ring_iter_mask = ring_work_plan.masks.active_ring_iter_mask;
    const uint32_t last_active_ring_iter = ring_work_plan.last_active_ring_iter;
    const uint32_t single_valid_kv_chunk_mask = ring_work_plan.masks.single_valid_kv_chunk_mask;
    const uint32_t compile_time_active_ring_iter_mask = kv_pad_rotation_enabled ? 0 : active_ring_iter_mask;
    const uint32_t compile_time_last_active_ring_iter = kv_pad_rotation_enabled ? 0 : last_active_ring_iter;
    const uint32_t compile_time_single_valid_kv_chunk_mask = kv_pad_rotation_enabled ? 0 : single_valid_kv_chunk_mask;
    const KVPadQMapping compile_time_kv_pad_q_mapping = kv_pad_rotation_enabled ? KVPadQMapping{} : kv_pad_q_mapping;
    const auto& ag_attributes = args.all_gather_operation_attributes;
    const RingAttentionRankMapping rank_mapping{
        .full_mesh = ag_attributes.full_mesh,
        .orientation = ag_attributes.snake_orientation,
        .mesh_rows = ag_attributes.mesh_rows,
        .mesh_cols = ag_attributes.mesh_cols};

    const uint32_t q_heads_per_kv = NH / NHK;

    // Cores actually issuing Q reads. When the flat q-chunk distribution is smaller
    // than the grid the trailing cores get zero work; zigzag distributes pairs, so
    // the unit count is total_pairs = all_heads_num_q_chunks / 2.
    const uint32_t num_active_cores = enable_zigzag_balancing ? std::min(num_cores, all_heads_num_q_chunks / 2)
                                                              : std::min(num_cores, all_heads_num_q_chunks);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        padded_Nt,
        compile_time_logical_n,
        compile_time_logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.all_gather_operation_attributes.ring_size,
        qk_out_subblock_h,
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        // Reader slot 24: chunked_enabled. Writer/compute use their corresponding slot for use_streaming_compute.
        static_cast<uint32_t>(kernel_chunked),
        num_active_cores,
        q_chunk_group_tile_count,
        static_cast<uint32_t>(indexed_kv_cache),
        static_cast<uint32_t>(kv_pad_rotation_enabled),
        compile_time_active_ring_iter_mask,
        NHV,
        static_cast<uint32_t>(v_shares_k_buffer),
        static_cast<uint32_t>(use_attention_sink),
        sliding_window_size,
        gathered_padded_Nt,
        static_cast<uint32_t>(slot_from_metadata),
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slots 37-38, sharded-joint path: Lt_local (Q-axis tile count = L_local/TILE_HEIGHT) and flag.
        Lt_local,
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 39: true (unpadded) joint length in tiles (twins spatial logical_nt). The reader uses it
        // to skip joint K chunks that lie entirely beyond the real joint tail (padding).
        // Slots 40-43: transport-to-tensor rank mapping. Tensor accessors start at slot 44.
        logical_lt,
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
    };

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    if (L != 0) {
        TensorAccessorArgs(joint_tensor_q->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(joint_tensor_k->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(joint_tensor_v->buffer()).append_to(reader_compile_time_args);
    }
    // Sharded-joint path: the gathered joint K/V accessors follow the local joint accessors, matching
    // get_post_tensor_args_offset<has_joint_inputs, has_gathered_joint_k, ...>() in the reader kernel.
    // They must precede the attention-sink and metadata accessors below, which the kernel places at
    // post_joint_tensor_args_offset / post_tensor_args_offset respectively.
    if (joint_is_sharded) {
        TensorAccessorArgs(gathered_joint_tensor_k->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(gathered_joint_tensor_v->buffer()).append_to(reader_compile_time_args);
    }
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    // Metadata accessors follow the tensor accessors (metadata path only) and precede the chain semaphore
    // compile args; the reader kernel gates their offsets on slot_from_metadata / kv_pad_from_metadata.
    // sem_args_offset below is computed after this append, so the chain/CB compile-arg indices stay correct.
    // slot_id and kv_actual_isl are SEPARATELY allocated single-page DRAM tensors that can land in different
    // DRAM banks, so each needs its OWN accessor -- a shared accessor's dspec (bank for page 0) is baked
    // from one buffer and reads the wrong bank for the other (kv read silently returned 0, breaking the
    // rotation derivation). The writer already appends kv_actual_isl's own accessor for the same reason.
    if (slot_from_metadata) {
        TensorAccessorArgs(tensor_args.slot_id->buffer()).append_to(reader_compile_time_args);
        if (kv_pad_from_metadata) {
            TensorAccessorArgs(tensor_args.kv_actual_isl->buffer()).append_to(reader_compile_time_args);
        }
    }

    /**
     * Create semaphores used for L1-L1 store-and-forward of KV between cores.
     * ChainSemaphores groups the three semaphore IDs for a single chain (sender,
     * receiver, valid) and pushes them as SemaphoreDescriptor entries on the
     * descriptor. The IDs are sequential indices into desc.semaphores.
     */
    struct ChainSemaphores {
        uint32_t sender_id;
        uint32_t receiver_id;
        uint32_t valid_id;

        static ChainSemaphores create(ProgramDescriptor& desc, const CoreRangeSet& cores) {
            ChainSemaphores out;
            out.sender_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.sender_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = INVALID,
            });
            out.receiver_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.receiver_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = INVALID,
            });
            out.valid_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.valid_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = VALID,
            });
            return out;
        }

        void append_to_compile_args(std::vector<uint32_t>& args) const {
            const size_t start_size = args.size();
            args.push_back(sender_id);
            args.push_back(receiver_id);
            args.push_back(valid_id);
            TT_FATAL(
                args.size() == start_size + kRingJointChainSemaphoreCompileArgCount,
                "RingJoint ChainSemaphores expected to append {} compile-time args, appended {}",
                kRingJointChainSemaphoreCompileArgCount,
                args.size() - start_size);
        }
    };

    std::optional<ChainSemaphores> head_sems;
    std::optional<ChainSemaphores> batch_sems;
    std::optional<ChainSemaphores> gqa_sems;
    if (use_head_chain) {
        head_sems = ChainSemaphores::create(desc, core_grid_set);  // head chain (MHA or separate-V V)
    }
    if (enable_kv_chains) {
        if (k_uses_batch_chain) {
            batch_sems = ChainSemaphores::create(desc, core_grid_set);  // shared-K chain
        }
        if (gqa_grouped_kv) {
            gqa_sems = ChainSemaphores::create(desc, core_grid_set);  // grouped K/V chain
        }
    }

    // Append semaphore ids to reader compile-time args (must match reader kernel expectations)
    const auto sem_args_offset = reader_compile_time_args.size();
    if (use_head_chain) {
        head_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // head_mcast_enabled placeholder (patched after chain construction)
    }
    if (enable_kv_chains) {
        if (k_uses_batch_chain) {
            batch_sems->append_to_compile_args(reader_compile_time_args);
            reader_compile_time_args.push_back(0);  // shared-K mcast placeholder
        }
        if (gqa_grouped_kv) {
            gqa_sems->append_to_compile_args(reader_compile_time_args);
            reader_compile_time_args.push_back(0);  // GQA mcast placeholder
        }
    }

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        padded_Nt,
        compile_time_logical_n,
        compile_time_logical_nt,
        Lt_local,  // slot 12: per-device joint tile count (Lt_local == Lt on replicated path)
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        packed_identity_scalar,
        scale_packed,
        args.all_gather_operation_attributes.ring_size,
        compile_time_global_n_partial_col,
        joint_l_partial_col,
        static_cast<std::uint32_t>(use_streaming_compute),
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        static_cast<std::uint32_t>(writer_out_row_group_h),
        static_cast<uint32_t>(kernel_chunked),
        q_chunk_group_tile_count,
        compile_time_active_ring_iter_mask,
        compile_time_last_active_ring_iter,
        compile_time_single_valid_kv_chunk_mask,
        sliding_window_size,
        // Slot 35: trace-safe KV-pad derivation -- the writer recomputes logical_nt + masks from
        // metadata[1] on-device (it's dataflow).
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slot 36: sharded-joint flag. When true, one shard per ring iteration; do_joint_kv fires every iter.
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 37: true (unpadded) joint length in tiles (twins spatial logical_nt). Combined with
        // joint_l_partial_col it drives the joint mask-generation gate.
        logical_lt,
        // Slots 38-41: transport-to-tensor rank mapping. Output accessors start at slot 42.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(stats_output_tensor.buffer()).append_to(writer_compile_time_args);
    if (kv_pad_from_metadata) {
        TensorAccessorArgs(tensor_args.kv_actual_isl->buffer()).append_to(writer_compile_time_args);
    }

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        padded_Nt,
        compile_time_logical_n,
        compile_time_logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.all_gather_operation_attributes.ring_size,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        scale_packed,
        static_cast<std::uint32_t>(use_streaming_compute),
        compile_time_global_n_partial_col,
        joint_l_partial_col,
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        static_cast<uint32_t>(kernel_chunked),
        q_chunk_group_tile_count,
        static_cast<uint32_t>(kv_pad_rotation_enabled),
        compile_time_kv_pad_q_mapping.q_pre_wrap_start_tile,
        compile_time_kv_pad_q_mapping.q_pre_wrap_tile_count,
        compile_time_kv_pad_q_mapping.q_post_wrap_start_tile,
        compile_time_kv_pad_q_mapping.q_valid_tile_count,
        compile_time_active_ring_iter_mask,
        compile_time_last_active_ring_iter,
        static_cast<uint32_t>(v_shares_k_buffer),
        static_cast<uint32_t>(use_attention_sink),
        sliding_window_size,
        // Slot 51: trace-safe KV-pad derivation. When set, compute reads logical_nt / q-mapping /
        // active_ring_iter_mask from cb_kv_pad_derived (produced by the reader) instead of its runtime
        // args, so a captured trace replays across chunks.
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slot 52: sharded-joint flag. When true, one shard per ring iteration; do_joint_kv fires every iter.
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 53: true (unpadded) joint length in tiles (twins spatial logical_nt). Drives the
        // per-ring-iteration joint tail mask and the joint out-of-bounds K-chunk skip.
        logical_lt,
        // Slots 54-57: transport-to-tensor rank mapping. CB block starts at 58.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols};

    std::map<std::string, std::string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    // NOTE: CreateKernel calls are deferred until after chain construction so that
    // the mcast_enabled compile-time arg can be determined first.

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());

    // Lightweight mask: both causal and non-causal paths use Float16_b
    // to support L1-accumulation and avoid Bfp4_b precision loss.
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.
    tt::DataFormat stats_df = im_df;
    // Use fp32 precision for cb_sum_A/B when fp32 accumulation is enabled so
    // the running softmax denominator doesn't lose precision with K-iter rounding.
    tt::DataFormat sum_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    // Use fp32 precision for cb_qk_im when fp32 accumulation is enabled so operations on QK retain precision.
    tt::DataFormat qk_im_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);
    uint32_t sum_tile_size = tt::tile_size(sum_df);
    uint32_t qk_im_tile_size = tt::tile_size(qk_im_df);

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "mask_data_format: {}", mask_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);
    log_debug(tt::LogOp, "sum_data_format: {}", sum_df);
    log_debug(tt::LogOp, "qk_im_data_format: {}", qk_im_df);

    uint32_t next_cb_index = 0;
    const auto allocate_cb = [&](uint32_t page_size_bytes, uint32_t num_pages, tt::DataFormat data_format) -> uint32_t {
        const uint32_t cb_index = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size_bytes * num_pages,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = data_format,
                .page_size = page_size_bytes,
            }}},
        });
        return cb_index;
    };
    const auto allocate_tile_cb = [&](uint32_t num_tiles, uint32_t tile_size, tt::DataFormat data_format) -> uint32_t {
        return allocate_cb(tile_size, num_tiles, data_format);
    };

    const uint32_t cb_q_in = allocate_tile_cb(q_tiles, q_tile_size, q_df);
    const uint32_t cb_k_in = allocate_tile_cb(k_tiles, k_tile_size, k_df);
    const uint32_t cb_v_in = v_shares_k_buffer ? cb_k_in : allocate_tile_cb(v_tiles, v_tile_size, v_df);

    // Lightweight mask CB: holds neginf + optional causal diagonal + optional partial tiles.
    // Used for both causal (ring_iter 0) and padding (ring_iter > 0) masking.
    constexpr uint32_t inactive_cb = std::numeric_limits<uint32_t>::max();
    const uint32_t cb_mask_in =
        needs_lightweight_mask ? allocate_tile_cb(total_lightweight_mask_tiles, mask_tile_size, mask_df) : inactive_cb;

    // Streaming normalization broadcasts the per-head sink scalar directly.
    // Alias a valid CB while disabled so compile-time tile-size queries remain valid;
    // sink producer/consumer code is removed by if constexpr in that specialization.
    const uint32_t cb_attention_sink = [&]() {
        if (!use_attention_sink) {
            return cb_q_in;
        }
        const tt::DataFormat sink_df = tt::tt_metal::datatype_to_dataformat_converter(attention_sink.value().dtype());
        return allocate_tile_cb(1, tt::tile_size(sink_df), sink_df);
    }();

    const uint32_t cb_scale_in = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);
    const uint32_t cb_identity_scale_in = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);
    const uint32_t cb_col_identity = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);

    const uint32_t cb_qk_im = allocate_tile_cb(qk_tiles, qk_im_tile_size, qk_im_df);
    const uint32_t cb_out_im_A = allocate_tile_cb(out_im_tiles, im_tile_size, im_df);
    const uint32_t cb_out_im_B = allocate_tile_cb(out_im_tiles, im_tile_size, im_df);
    const uint32_t cb_max_A = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    const uint32_t cb_max_B = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    const uint32_t cb_sum_A = allocate_tile_cb(statistics_tiles, sum_tile_size, sum_df);
    const uint32_t cb_sum_B = allocate_tile_cb(statistics_tiles, sum_tile_size, sum_df);
    const uint32_t cb_exp_max_diff = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);

    const uint32_t cb_out = allocate_tile_cb(out0_t, out_tile_size, out_df);

    // Sliding folds every local/halo K/V range into one final pass per Q, so it never saves
    // or restores accumulators through DRAM. Keep valid, format-compatible CB indices in the
    // compile-time ABI without reserving separate L1 storage for those unreachable paths.
    const bool needs_dram_accumulator_staging = !has_sliding_window;
    const uint32_t cb_stats_in =
        needs_dram_accumulator_staging ? allocate_tile_cb(statistics_tiles, im_tile_size, im_df) : cb_max_A;
    const uint32_t cb_prev_out =
        needs_dram_accumulator_staging ? allocate_tile_cb(out_im_tiles, out_tile_size, out_df) : cb_out;
    const uint32_t cb_stats_out =
        needs_dram_accumulator_staging ? allocate_tile_cb(statistics_tiles, im_tile_size, im_df) : cb_max_B;

    // Streaming compute v2: 1-tile recip scratch CB for normalize_row_streaming.
    // cb_scale_in is live in ring joint, so streaming uses a dedicated scratch CB.
    const uint32_t cb_recip_scratch = use_streaming_compute ? allocate_tile_cb(1, im_tile_size, im_df) : inactive_cb;

    // Deferred norm: sum save/restore CBs for multi Q-chunk DRAM round-trip.
    // cb_sum_out = compute pushes sum for writer to save to DRAM.
    // cb_sum_in = writer pushes restored sum from DRAM for compute to read.
    const uint32_t cb_sum_out =
        use_streaming_compute
            ? (needs_dram_accumulator_staging ? allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df)
                                              : cb_sum_A)
            : inactive_cb;
    const uint32_t cb_sum_in =
        use_streaming_compute
            ? (needs_dram_accumulator_staging ? allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df)
                                              : cb_sum_B)
            : inactive_cb;

    // Signal CB: compute signals writer when last K-chunk starts.
    // 1 page suffices: writer pops during SALAD before compute pushes the next Q's signal.
    constexpr uint32_t signal_page_size = 16;
    const uint32_t cb_signal =
        use_streaming_compute ? allocate_cb(signal_page_size, 1, tt::DataFormat::UInt16) : inactive_cb;
    // Reader-to-compute mailbox for the metadata-derived logical geometry.
    const uint32_t cb_kv_pad_derived = allocate_cb(64, 1, tt::DataFormat::UInt32);

    const std::vector<uint32_t> cb_compile_time_args = {
        cb_q_in,     cb_k_in,     cb_v_in,         cb_mask_in,       cb_scale_in,     cb_identity_scale_in,
        cb_stats_in, cb_prev_out, cb_col_identity, cb_recip_scratch, cb_sum_out,      cb_sum_in,
        cb_signal,   cb_out,      cb_stats_out,    cb_qk_im,         cb_out_im_A,     cb_out_im_B,
        cb_max_A,    cb_max_B,    cb_sum_A,        cb_sum_B,         cb_exp_max_diff, cb_kv_pad_derived};
    const std::vector<uint32_t> reader_cb_compile_time_args = {
        cb_q_in, cb_k_in, cb_v_in, cb_attention_sink, cb_kv_pad_derived};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), reader_cb_compile_time_args.begin(), reader_cb_compile_time_args.end());
    writer_compile_time_args.insert(
        writer_compile_time_args.end(), cb_compile_time_args.begin(), cb_compile_time_args.end());
    auto compute_cb_compile_time_args = cb_compile_time_args;
    compute_cb_compile_time_args.push_back(cb_attention_sink);
    compute_compile_time_args.insert(
        compute_compile_time_args.end(), compute_cb_compile_time_args.begin(), compute_cb_compile_time_args.end());

    auto* const q_buf = input_tensor_q.buffer();
    auto* const k_buf = input_tensor_k.buffer();
    auto* const v_buf = input_tensor_v.buffer();
    auto* const gathered_k_buf = gathered_input_tensor_k.buffer();
    auto* const gathered_v_buf = gathered_input_tensor_v.buffer();
    auto* const attention_sink_buf = attention_sink.has_value() ? attention_sink->buffer() : nullptr;
    auto* const out_buf = output_tensor.buffer();
    auto* const joint_out_buf = joint_output_tensor.buffer();
    auto* const stats_buf = stats_output_tensor.buffer();

    /**
     * Build chain selection for store-and-forward across cores per head.
     */
    struct CoreHeadWork {
        uint32_t head = 0;
        uint32_t q_chunk_count = 0;
    };

    struct CoreWork {
        CoreCoord physical_core;
        uint32_t global_q_start = 0;
        uint32_t global_q_count = 0;
        std::vector<CoreHeadWork> head_work;
    };

    struct HeadSegmentRef {
        uint32_t core_idx = 0;
        uint32_t head_work_index = 0;
    };

    // Unified chain configuration for head-level and shared-K chains.
    struct ChainConfig {
        // Core participation flags
        bool participates = false;
        bool is_injector = false;
        bool is_sink = false;

        // Chain scope: head distinguishes head-level from shared-K chains.
        uint32_t head = 0;  // 0 for shared-K chains

        // Linear chain topology
        CoreCoord prev_physical = CoreCoord{0, 0};
        CoreCoord next_physical = CoreCoord{0, 0};
        uint32_t next_core_q_chunks = 0;

        // Multicast configuration (1D for V, 2D for K)
        CoreCoord mcast_start = CoreCoord{0, 0};        // Rectangle start (physical)
        CoreCoord mcast_end = CoreCoord{0, 0};          // Rectangle end (physical)
        CoreCoord injector_physical = CoreCoord{0, 0};  // Injector's coords (for receiver sem addr in mcast)
        uint32_t mcast_num_dests = 0;                   // Receivers count (excludes self)

        // Append runtime args in canonical order
        void append_to_args(std::vector<uint32_t>& args) const {
            const size_t start_size = args.size();
            args.push_back(static_cast<uint32_t>(participates));
            args.push_back(static_cast<uint32_t>(is_injector));
            args.push_back(static_cast<uint32_t>(is_sink));
            args.push_back(head);
            args.push_back(static_cast<uint32_t>(prev_physical.x));
            args.push_back(static_cast<uint32_t>(prev_physical.y));
            args.push_back(static_cast<uint32_t>(next_physical.x));
            args.push_back(static_cast<uint32_t>(next_physical.y));
            args.push_back(next_core_q_chunks);
            args.push_back(static_cast<uint32_t>(mcast_start.x));
            args.push_back(static_cast<uint32_t>(mcast_start.y));
            args.push_back(static_cast<uint32_t>(mcast_end.x));
            args.push_back(static_cast<uint32_t>(mcast_end.y));
            args.push_back(static_cast<uint32_t>(injector_physical.x));
            args.push_back(static_cast<uint32_t>(injector_physical.y));
            args.push_back(mcast_num_dests);
            TT_FATAL(
                args.size() == start_size + kRingJointChainConfigArgCount,
                "RingJoint ChainConfig expected to append {} runtime args, appended {}",
                kRingJointChainConfigArgCount,
                args.size() - start_size);
        }
    };

    std::vector<CoreWork> core_work(num_cores);
    std::vector<ChainConfig> head_chain_configs(use_head_chain ? num_cores : 0);  // MHA K/V or separate-V V
    std::vector<ChainConfig> batch_chain_configs(
        enable_kv_chains && k_uses_batch_chain ? num_cores : 0);  // Shared K for separate-V/latent modes
    std::vector<ChainConfig> gqa_chain_configs(
        enable_kv_chains && gqa_grouped_kv ? num_cores : 0);  // Grouped K/V for GQA
    // Sliding attention reads K/V independently on every core and does not build chains.
    std::vector<std::vector<HeadSegmentRef>> head_segments(use_head_chain ? NH : 0);

    // Evenly distribute flat global q chunks across cores
    const uint32_t total_q_chunks = B * NH * num_q_chunks;

    uint32_t base_chunks_per_core = 0;
    uint32_t extra_chunks_per_core = 0;
    uint32_t cores_doing_extra_work = 0;
    if (enable_zigzag_balancing) {
        log_debug(tt::LogOp, "Enabling zigzag balancing with even num_q_chunks: {}", num_q_chunks);
        const uint32_t total_pairs = total_q_chunks / 2;
        cores_doing_extra_work = total_pairs % num_cores;
        base_chunks_per_core = (num_cores == 0) ? 0 : (total_pairs / num_cores) * 2;
        extra_chunks_per_core = (num_cores == 0) ? 0 : 2;
    } else {
        cores_doing_extra_work = total_q_chunks % num_cores;
        base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
        extra_chunks_per_core = (num_cores == 0) ? 0 : 1;
    }

    uint32_t next_global_chunk = 0;

    auto decode_flat_chunk = [&](uint32_t flat_chunk_index) {
        const uint32_t head_span = num_q_chunks;
        const uint32_t head_index = head_span == 0 ? 0 : (flat_chunk_index / head_span);
        const uint32_t q_chunk = head_span == 0 ? 0 : (flat_chunk_index % head_span);
        const uint32_t head = (NH == 0) ? 0 : (head_index % NH);
        return std::pair<uint32_t, uint32_t>{head, q_chunk};
    };

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t chunk_count = base_chunks_per_core + ((i < cores_doing_extra_work) ? extra_chunks_per_core : 0);
        if (next_global_chunk >= total_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > total_q_chunks - next_global_chunk) {
            chunk_count = total_q_chunks - next_global_chunk;
        }

        auto& work = core_work.at(i);
        work.physical_core = device->worker_core_from_logical_core(core);
        work.global_q_start = next_global_chunk;
        work.global_q_count = chunk_count;

        uint32_t remaining = chunk_count;
        uint32_t flat_chunk = next_global_chunk;
        while (remaining > 0) {
            auto [head_idx, q_chunk_idx] = decode_flat_chunk(flat_chunk);
            uint32_t chunk_capacity_in_head = num_q_chunks - q_chunk_idx;
            uint32_t chunk_take = std::min(remaining, chunk_capacity_in_head);

            if (enable_kv_chains) {
                work.head_work.push_back(CoreHeadWork{
                    .head = head_idx,
                    .q_chunk_count = chunk_take,
                });
                if (use_head_chain) {
                    TT_FATAL(
                        head_idx < head_segments.size(),
                        "Head-chain segment index {} is outside {} configured query heads",
                        head_idx,
                        head_segments.size());
                    head_segments[head_idx].push_back(HeadSegmentRef{
                        .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
                }
            }

            remaining -= chunk_take;
            flat_chunk += chunk_take;
        }

        next_global_chunk += chunk_count;
    }

    // Helper: build a linear chain from sorted (core_idx, q_chunk_count) pairs.
    // - chain_segs[i].second = q iterations the i-th core will process in this chain scope
    // - injector = first core with head_work.size() == 1 (single head segment = no straddling)
    // - no wrap-around: wrapping would inflate q_iter_local and cause deadlock
    // - injector reselection for mcast is done separately in the mcast eligibility pass
    using ChainSegment = std::pair<uint32_t, uint32_t>;  // (core_idx, q_chunk_count)
    auto build_linear_chain = [](const std::vector<ChainSegment>& chain_segs,
                                 uint32_t head,
                                 std::vector<ChainConfig>& chain_configs,
                                 const std::vector<CoreWork>& core_work,
                                 bool require_single_head_injector = true) -> bool {
        if (chain_segs.size() < 2) {
            return false;
        }
        std::optional<size_t> injector_pos;
        for (size_t idx = 0; idx + 1 < chain_segs.size(); ++idx) {
            if (core_work[chain_segs[idx].first].global_q_count == 0) {
                continue;
            }
            if (!require_single_head_injector || core_work[chain_segs[idx].first].head_work.size() == 1) {
                injector_pos = idx;
                break;
            }
        }
        if (!injector_pos.has_value()) {
            return false;
        }
        const size_t start = *injector_pos;
        for (size_t idx = start; idx < chain_segs.size(); ++idx) {
            uint32_t ci = chain_segs[idx].first;
            auto& cfg = chain_configs[ci];
            cfg.participates = true;
            cfg.head = head;
            cfg.is_injector = (idx == start);
            cfg.is_sink = (idx == chain_segs.size() - 1);
            if (idx > start) {
                cfg.prev_physical = core_work[chain_segs[idx - 1].first].physical_core;
            }
            if (idx + 1 < chain_segs.size()) {
                cfg.next_physical = core_work[chain_segs[idx + 1].first].physical_core;
                cfg.next_core_q_chunks = chain_segs[idx + 1].second;
            }
        }
        return true;
    };

    struct RowWideChainMcastSelection {
        uint32_t row = 0;
        uint32_t injector_idx = 0;
        uint32_t max_q = 0;
        uint32_t num_receivers = 0;
        CoreCoord injector_physical = CoreCoord{0, 0};
        CoreCoord phys_start = CoreCoord{0, 0};
        CoreCoord phys_end = CoreCoord{0, 0};
    };

    // Pick a row-wide multicast injector among max-work cores. The recent-column FIFO spreads injectors across
    // physical X columns for NoC diversity; correctness still requires a max-work injector so padded loops never
    // read K/V beyond the real q-iteration span.
    auto select_row_wide_chain_mcast =
        [&](uint32_t row, std::deque<uint32_t>& recent_cols) -> std::optional<RowWideChainMcastSelection> {
        if (recent_cols.size() >= grid_size.x) {
            recent_cols.pop_front();
        }

        uint32_t row_max_q = 0;
        for (uint32_t col = 0; col < grid_size.x; ++col) {
            const uint32_t ci = row * grid_size.x + col;
            row_max_q = std::max(row_max_q, core_work[ci].global_q_count);
        }
        if (row_max_q == 0) {
            return std::nullopt;
        }

        uint32_t injector_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t col = 0; col < grid_size.x; ++col) {
            const uint32_t ci = row * grid_size.x + col;
            if (core_work[ci].global_q_count != row_max_q) {
                continue;
            }
            const uint32_t phys_x = core_work[ci].physical_core.x;
            const bool excluded = std::find(recent_cols.begin(), recent_cols.end(), phys_x) != recent_cols.end();
            if (!excluded) {
                injector_idx = ci;
                break;
            }
            if (injector_idx == std::numeric_limits<uint32_t>::max()) {
                injector_idx = ci;
            }
        }

        TT_FATAL(
            injector_idx != std::numeric_limits<uint32_t>::max(),
            "RingJoint row mcast failed to find a max-work injector for row {}",
            row);
        const CoreCoord injector_physical = core_work[injector_idx].physical_core;
        recent_cols.push_back(injector_physical.x);

        return RowWideChainMcastSelection{
            .row = row,
            .injector_idx = injector_idx,
            .max_q = row_max_q,
            .num_receivers = grid_size.x - 1,
            .injector_physical = injector_physical,
            .phys_start = device->worker_core_from_logical_core(CoreCoord{0, row}),
            .phys_end = device->worker_core_from_logical_core(CoreCoord{grid_size.x - 1, row}),
        };
    };

    auto configure_row_wide_chain_mcast = [&](const RowWideChainMcastSelection& selection,
                                              uint32_t head,
                                              std::vector<ChainConfig>& chain_configs,
                                              std::vector<uint32_t>& chain_max_q) {
        const auto configure_core = [&](uint32_t ci) {
            auto& cfg = chain_configs[ci];
            cfg.participates = true;
            cfg.head = head;
            cfg.prev_physical = CoreCoord{0, 0};
            cfg.next_physical = CoreCoord{0, 0};
            cfg.mcast_start = selection.phys_start;
            cfg.mcast_end = selection.phys_end;
            cfg.injector_physical = selection.injector_physical;
            cfg.is_injector = (ci == selection.injector_idx);
            cfg.is_sink = !cfg.is_injector;
            if (cfg.is_injector) {
                cfg.mcast_num_dests = selection.num_receivers;
                cfg.next_core_q_chunks = selection.max_q;
            } else {
                cfg.mcast_num_dests = 0;
                cfg.next_core_q_chunks = 0;
            }
            chain_max_q[ci] = selection.max_q;
        };
        for (uint32_t col = 0; col < grid_size.x; ++col) {
            configure_core(selection.row * grid_size.x + col);
        }
    };

    uint32_t gqa_grouped_chains = 0;
    uint32_t gqa_grouped_participant_cores = 0;
    uint32_t gqa_local_fallback_cores = 0;
    bool gqa_mcast_enabled = false;
    std::string gqa_mcast_fallback_reason;
    std::vector<uint32_t> gqa_chain_max_q(gqa_chain_configs.size(), 0);  // per-core loop-padding count
    if (gqa_grouped_kv && build_kv_chains) {
        std::vector<std::vector<ChainSegment>> kv_group_segments(NHK);

        for (uint32_t ci = 0; ci < num_cores; ++ci) {
            if (core_work[ci].global_q_count == 0) {
                continue;
            }

            bool has_single_group = false;
            bool spans_multiple_groups = false;
            uint32_t single_group_id = 0;
            uint32_t q_chunk_count = 0;
            for (const auto& hw : core_work[ci].head_work) {
                const uint32_t kv_head = hw.head / q_heads_per_kv;
                if (!has_single_group) {
                    has_single_group = true;
                    single_group_id = kv_head;
                } else if (single_group_id != kv_head) {
                    spans_multiple_groups = true;
                    break;
                }
                q_chunk_count += hw.q_chunk_count;
            }

            if (has_single_group && !spans_multiple_groups) {
                kv_group_segments[single_group_id].emplace_back(ci, q_chunk_count);
                gqa_grouped_participant_cores++;
            } else {
                gqa_local_fallback_cores++;
            }
        }

        for (uint32_t kv_head = 0; kv_head < static_cast<uint32_t>(kv_group_segments.size()); ++kv_head) {
            const auto& chain_segs = kv_group_segments[kv_head];
            if (chain_segs.size() < 2) {
                continue;
            }
            if (build_linear_chain(chain_segs, kv_head, gqa_chain_configs, core_work, false)) {
                gqa_grouped_chains++;
            }
        }

        // Production Minimax3 GQA has one local K/V head per chip (B=1, NHK=NHV=1). In that case every
        // active Q-head core consumes the same K and V chunks, so use row-wide multicast instead
        // of the store-and-forward grouped chain. Idle cores in an active row participate in padded iterations.
        if (NHK != 1) {
            gqa_mcast_fallback_reason = "NHK != 1 (multi-KV-head GQA mcast not supported)";
        } else if (num_cores < 2) {
            gqa_mcast_fallback_reason = "num_cores < 2";
        } else if (grid_size.x < 2) {
            gqa_mcast_fallback_reason = "grid_size.x < 2 (singleton rows)";
        } else {
            std::deque<uint32_t> recent_cols;  // FIFO of <= grid.x-1 most-recent claimed phys_x
            uint32_t gqa_mcast_rows = 0;

            for (uint32_t row = 0; row < grid_size.y; ++row) {
                const auto selection = select_row_wide_chain_mcast(row, recent_cols);
                if (!selection.has_value()) {
                    continue;
                }
                configure_row_wide_chain_mcast(*selection, 0, gqa_chain_configs, gqa_chain_max_q);
                gqa_mcast_rows++;
                log_debug(
                    tt::LogOp,
                    "GQA K/V mcast row {}: injector core {} phys=({},{}) max_q={}, rect ({},{})-({},{})",
                    selection->row,
                    selection->injector_idx,
                    selection->injector_physical.x,
                    selection->injector_physical.y,
                    selection->max_q,
                    selection->phys_start.x,
                    selection->phys_start.y,
                    selection->phys_end.x,
                    selection->phys_end.y);
            }

            gqa_mcast_enabled = gqa_mcast_rows > 0;
            if (!gqa_mcast_enabled) {
                gqa_mcast_fallback_reason = "no active groups";
            }
        }
    }

    // Build the shared-K chain for separate-V/latent cases.
    // K is shared across all heads, so all active cores form one chain.
    // Sorted by physical position for a stable unicast ordering (overwritten by mcast pass if eligible).
    if (k_uses_batch_chain && build_kv_chains) {
        std::vector<uint32_t> core_indices;
        for (uint32_t i = 0; i < num_cores; ++i) {
            if (core_work[i].global_q_count == 0) {
                continue;
            }
            core_indices.push_back(i);
        }

        std::sort(core_indices.begin(), core_indices.end(), [&](uint32_t a, uint32_t b) {
            const auto& pa = core_work[a].physical_core;
            const auto& pb = core_work[b].physical_core;
            return (pa.y < pb.y) || (pa.y == pb.y && pa.x < pb.x);
        });

        std::vector<ChainSegment> chain_segs;
        chain_segs.reserve(core_indices.size());
        for (uint32_t ci : core_indices) {
            chain_segs.emplace_back(ci, core_work[ci].global_q_count);
        }
        if (build_linear_chain(chain_segs, 0, batch_chain_configs, core_work)) {
            log_debug(tt::LogOp, "K unicast chain: {} cores", chain_segs.size());
        }
    }

    // K multicast pass: one mcast chain per logical row. Shared-K keeps the all-or-nothing policy:
    // every row must contain work, otherwise the previously built linear chain remains active.
    bool k_mcast_enabled = false;
    std::string k_mcast_fallback_reason;
    std::vector<uint32_t> k_chain_max_q(batch_chain_configs.size(), 0);  // per-core loop-padding count

    if (!k_uses_batch_chain || !enable_kv_chains) {
        // No non-GQA shared-K chain to multicast.
    } else if (num_cores < 2) {
        k_mcast_fallback_reason = "num_cores < 2";
    } else if (grid_size.x < 2) {
        // Each chain would be a singleton (1 core, no sinks) — mcast is degenerate.
        k_mcast_fallback_reason = "grid_size.x < 2 (singleton chains)";
    } else {
        std::vector<RowWideChainMcastSelection> row_mcast_selections;
        row_mcast_selections.reserve(grid_size.y);
        std::deque<uint32_t> recent_cols;  // FIFO of <= grid.x-1 most-recent claimed phys_x

        bool all_chains_picked = true;
        for (uint32_t row = 0; row < grid_size.y; ++row) {
            const std::optional<RowWideChainMcastSelection> selection = select_row_wide_chain_mcast(row, recent_cols);
            if (!selection.has_value()) {
                k_mcast_fallback_reason = fmt::format("row {} has no work", row);
                all_chains_picked = false;
                break;
            }
            row_mcast_selections.push_back(*selection);
        }

        if (all_chains_picked) {
            k_mcast_enabled = true;

            for (const auto& selection : row_mcast_selections) {
                configure_row_wide_chain_mcast(selection, 0, batch_chain_configs, k_chain_max_q);

                log_debug(
                    tt::LogOp,
                    "K mcast row {}: injector core {} phys=({},{}) max_q={}, rect ({},{})-({},{})",
                    selection.row,
                    selection.injector_idx,
                    selection.injector_physical.x,
                    selection.injector_physical.y,
                    selection.max_q,
                    selection.phys_start.x,
                    selection.phys_start.y,
                    selection.phys_end.x,
                    selection.phys_end.y);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Rotated per-ring-iteration Q distribution ("rotated q split").
    //
    // With row-wide K mcast, each grid row is an independent lockstep pipe that pays
    // ring_size * max(chunks per core in the row) K-stream slots. The flat split pins the
    // total_q_chunks % num_cores remainder ("float") chunks to fixed cores, so their rows pay the
    // +1 slot on EVERY ring iteration: grid time = ring_size * ceil(U/C) slots vs the ideal
    // U * ring_size / C (e.g. 40 vs 34.9 slots on a 110-core grid -> 87% occupancy ceiling).
    // The (m, l, O) accumulators already round-trip through DRAM between ring iterations addressed
    // by chunk identity, so a float chunk may change owner core between ring iterations: rotate the
    // floats across rows so each row runs the +1-slot mode only ~ring_size*F/C of the time.
    // Cross-core ordering (donor's accumulator save must land in DRAM before the receiver's restore
    // read) uses a small ring of handoff semaphores indexed by (ring_iter - 1) % kRotHandoffSemDepth
    // (derived in chunked_prefill_utils.hpp): the donor writer increments the receiver's semaphore
    // after a write barrier on the save TRID; the receiver waits before issuing the float's restore
    // reads. Floats sit LAST in every owner's per-iteration list, so the deferred save flushes at
    // the start of iteration t+1 while the restore is needed near the end of iteration t+1 -- about
    // one ring iteration of slack.
    // ---------------------------------------------------------------------------
    // The same numbers the flat split above already derived (the !args.is_balanced term below pins
    // us to its non-zigzag branch); aliased rather than recomputed so the two cannot drift.
    const uint32_t rot_base_chunks = base_chunks_per_core;
    const uint32_t rot_float_chunks = cores_doing_extra_work;
    // Grid rows hosting floats on any one iteration. Rotation only buys something while at least
    // one row is float-free: at rot_rows_needed == grid_size.y the per-iteration row offset is
    // always 0 mod grid_size.y, ownership never actually moves, and every row pays the +1 mcast slot
    // every iteration exactly as the flat split does -- all of the cost, none of the win.
    const uint32_t rot_rows_needed = grid_size.x ? tt::div_up(rot_float_chunks, grid_size.x) : 0;
    // Do the floats fill their hosting rows exactly? This decides whether the rotation PAYS for
    // separate-V. Empirical rule from four measured points (table above), not a derivation: even fill
    // won (+4.0% at 100 cores, +8.9% at 80), partial row lost (-11.4% at 110, -3.4% at 60).
    // Mechanism, plausible but unverified: a partially-occupied hosting row still pays base+1 mcast
    // slots while only some of its cores use the extra slot, so rotating spreads that waste over
    // every row instead of confining it to one. Latent-V does NOT consult this -- it has no V-chain
    // amortization to lose and measured a win at every base, partial-row core counts included.
    // Extrapolation caveat: the four points are all from THIS box's reachable core counts (110/100/
    // 80/60). Other hardware will hit even-fill combinations that were never measured -- e.g. a
    // 4-device QuietBox at 100 cores gives kimi50k separate-V 20 floats, which is even and so
    // engages on prediction, not measurement. A wrong prediction costs PERF only: accuracy is
    // verified independently of this rule and does not depend on it.
    const bool rot_floats_fill_rows_evenly = grid_size.x != 0 && (rot_float_chunks % grid_size.x) == 0;
    const uint32_t full_ring_iter_mask = ring_size >= 32 ? 0xFFFFFFFFu : ((1u << ring_size) - 1);
    // v_shares_k_buffer is required: separate-V modes stream V through the per-head
    // store-and-forward chain, whose per-core forwarding counts are built from the flat split and
    // would desync under rotated per-iteration ownership. Latent/shared-buffer V never touches the
    // V chain (V is materialized from, or read in place of, the mcast K^T).
    // Debug kill-switch for A/B measurements: RING_MLA_DISABLE_ROTATED_Q_SPLIT=1 forces the static
    // flat split on an otherwise identical build. The value is parsed rather than merely tested for
    // presence, so "=0" leaves the feature ON as the name implies. Latched once per process, so
    // toggling the variable between dispatches (mock.patch.dict and friends) has no effect after the
    // first ring-joint invocation -- A/B it across two separate runs.
    // Separate-V rotation is implemented and correct, and its perf is REGIME-DEPENDENT, so it is
    // opt-in: RING_MLA_ROTATE_SEPARATE_V=1. kimi50k q32/k640 separate-V, in-process profiler, 2 runs
    // per cell:
    //    100 cores (grid 10x10, base 2, floats 80): rotated 7.595/7.607 vs static 7.907/7.914 ms
    //                                               -> rotation WINS by 4.0%
    //    110 cores (grid 11x10, base 2, floats 60): rotated 8.473/8.465 vs static 7.602/7.600 ms
    //                                               -> rotation LOSES by 11.4%
    // So it is NOT inherently slower -- and note 100 cores is the STOCK p150 split, while the
    // 110-core box these numbers were taken on is the re-flashed outlier. The mechanism behind the
    // penalty is that under rotation the float chunks land in heads with no base chunks, so their V
    // is read from DRAM instead of riding the per-head store-and-forward chain; what is not
    // explained by float count alone is why 80 floats at 100 cores beats 60 floats at 110 (the
    // partial float row at 110 -- 60 floats over 11-wide rows -- is the obvious suspect and is
    // unverified). Two data points do not locate the boundary, which is exactly why this stays
    // opt-in rather than being enabled on a guessed heuristic.
    // Latent-V never has the penalty at all: V is packed into K and rides the K mcast already.
    // The principled fix is head-ALIGNED float packing (group floats so each hosting row's floats
    // share one head, then row-mcast V on the float slot, reusing the head chain's semaphores which
    // are idle there) -- a separate change, and it would likely make this always-on-able.
    // Force separate-V rotation on even where the even-fill rule below predicts a loss. For
    // experimentation only -- it is how the four data points in the table above were taken.
    static const bool force_rotate_separate_v = []() {
        const char* value = std::getenv("RING_MLA_ROTATE_SEPARATE_V");
        return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
    }();
    static const bool rotated_q_split_disabled = []() {
        const char* value = std::getenv("RING_MLA_DISABLE_ROTATED_Q_SPLIT");
        return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
    }();
    // Generality, measured 2026-08-30 on bh-lb-33 ring-8 by sweeping RING_MLA_SDPA_GRID_OVERRIDE
    // (num_cores is what sets base/floats/rows_needed, so it is the only knob that moves this
    // schedule without new hardware). PCC passed on every one of:
    //   base 1 (q128, 70 cores, floats 50, rows_needed 8)   base 5 (q32, 90,  floats 30, rn 4)
    //   base 2 (q64,  90 cores, floats 60, rn 7)            base 6 (q32, 70,  floats 60, rn 9)
    //   base 3 (q64,  70 cores, floats 30, rn 5)            base 9 (q32, 50,  floats 30, rn 6)
    //   base 4 (q32, 100 cores, floats 80, rn 8)
    // Perf across that whole range, rotation ON vs OFF (same build, kill switch for OFF): base 1
    // 15.587 -> 9.091 ms (1.71x), base 2 11.351 -> 8.678 (1.31x), base 4 9.687 -> 8.676 (1.12x),
    // base 5 63.32 -> 68.36%, base 6 69.63 -> 69.62% (NEUTRAL), base 9 68.49 -> 71.18%. Never a
    // regression anywhere it engages. The win tracks how many rows are FLOAT-FREE
    // (grid_size.y - rot_rows_needed), not base -- base 6 above is break-even precisely because
    // rows_needed is 9 of 10 rows, leaving almost nothing to rebalance.
    // plus three DECLINE cases, all PCC-passing on the static fallback: floats == 0 (60 cores, q32
    // and q64), and rows_needed == grid_size.y (11x2 grid, q64: base 10, floats 20, rows_needed 2
    // of 2 rows). That covers rows_needed 1..9 and floats 10..80.
    //
    // GALAXY (sp=8, tp=4, 110 SDPA cores) is not reachable from this box, but needs no separate
    // validation for kimi_k3 q32: 24 heads x 20 chunks over 110 cores is base 4 / floats 40 /
    // rows_needed 4 -- bit-identical to the default local config exercised on every run here. tp=4
    // is 4 independent rings and does not enter this schedule. Note galaxy kimi50k q32 DECLINES
    // (16 heads -> 320 units, floats 100, rows_needed 10 == grid_size.y) and kimi50k q128 declines
    // on base 0; both are correct, the first because rotation there cannot move ownership at all.
    //
    // RING_SIZE != 8 is not constructible on this hardware: RING_MLA_RING_SIZE_OVERRIDE opens the
    // sub-mesh, but a 4- or 2-device sub-mesh of this 8-chip box dies in MeshDeviceImpl::create
    // with a fabric router / ethernet handshake timeout, before any SDPA kernel runs. It is covered
    // wherever these tests run on a natively 4-device host, since the accuracy test is not
    // ring-gated (only the perf check is).
    //
    // Every term below is load-bearing, and for each the FIRST thing that breaks is known. Two are
    // cost guards rather than correctness guards; that is called out. One caveat on minimality:
    // k_mcast_enabled is *nearly* implied by the rest -- given the other terms its only residual
    // content is grid_size.x >= 2 -- so it is kept for robustness and self-documentation, not
    // because it is independent.
    //
    //   k_mcast_enabled        purpose: the lockstep unit is the ROW. Under mcast every row member
    //                          loops the row max, so one float makes all grid_size.x cores of its
    //                          row pay a +1 K slot EVERY iteration -- the imbalance this fixes is
    //                          largely a row-mcast artifact. Removing it: at grid_size.x == 1 the
    //                          chain is unicast with a single injector, so injector_col is unset for
    //                          other rows and the TT_FATAL below fires.
    //   build_kv_chains        carries B == 1. Removing it is a SILENT wrong answer, not a hang:
    //                          k_mcast_enabled's own guard tests enable_kv_chains, so at B > 1 the
    //                          row mcast is live and the injector multicasts K for ITS nb to peers
    //                          whose rotated slot decodes a different nb.
    //   v_shares_k_buffer      separate-V turns on the per-head V chain, whose per-(head, core)
    //                          forwarding counts come from the STATIC split; a rotated float can
    //                          belong to a head that core has no segment for -> V relay mismatch ->
    //                          hang. Also subsumes !use_attention_sink, use_streaming_compute
    //                          (!fp32_dest_acc_en) and !gqa_grouped_kv, which separate-V would have
    //                          to state explicitly.
    //   !args.is_balanced      two independent breaks. Zigzag sets extra_chunks_per_core = 2, so
    //                          cores_doing_extra_work counts PAIRS while this schedule appends ONE
    //                          float per owner -- chunk ids [base*C+F, base*C+2F) are then never
    //                          scheduled (silent garbage). And balanced_skip_q's parity decision
    //                          desyncs injector from receivers at the float slot -> hang.
    //   (!kv_pad_rotation_enabled was here and has been REMOVED -- kv-pad now rotates. See the
    //    rot_active_ordinal note in the predicate for why the device-derived mask stopped mattering.)
    //   (active_ring_iter_mask == full_ring_iter_mask was here and has been REMOVED -- partial masks
    //    are handled by ordinal indexing. See the predicate. It still feeds the log line below, which
    //    reports whether the mask was full, since that is useful context when reading a decline.)
    //   (rot_float_chunks != 0 and rot_rows_needed < grid_size.y were COST guards here and have
    //    been REMOVED -- see the predicate for why both are provable no-ops rather than guards.)
    //   rot_base_chunks >= 1   definitional. At base 0 the first failure is actually the compute
    //                          kernel's static_assert(ROTATED_Q_SPLIT >= 2), ahead of the reader's
    //                          rot_my_count - 1 underflow.
    const bool use_rotated_q_split =
        !rotated_q_split_disabled &&
        // Path scope: latent-V (V packed into K) on the streaming compute path, with K streamed
        // through the row-wide mcast batch chain. Only the streaming compute kernel carries the
        // ROTATED_Q_SPLIT hook; on any other path compute would desync from the reader and writer,
        // so the compute kernel static_asserts this too. These two terms also subsume
        // !has_sliding_window, enable_kv_chains, k_uses_batch_chain and B == 1, and v_shares_k_buffer
        // subsumes !use_attention_sink (the op rejects that pair outright).
        k_mcast_enabled && build_kv_chains &&
        // V transport. Latent-V packs V into K's prefix so it rides the K mcast and needs nothing
        // more. Separate-V puts V on the per-head store-and-forward chain, which works here only
        // because the chain is rebuilt from the ROTATED base ranges below, and because it must now
        // state explicitly the three things v_shares_k_buffer was supplying implicitly:
        // use_streaming_compute (latent-V is TT_FATALed into it; separate-V with fp32_dest_acc_en
        // would otherwise trip the compute kernel's static_assert), !use_attention_sink (the op
        // rejects sink+latent-V outright, but permits sink+separate-V, and sink under rotation is
        // unvalidated), and the head-boundary condition: floats are the TAIL flat ids, so keeping
        // base*num_cores a whole multiple of num_q_chunks puts the base/float boundary on a head
        // boundary. Then no float ever belongs to a head that has base chunks, i.e. never to a head
        // whose chain exists, so every float falls through to the local DRAM V read that
        // `nq != chain_head` already triggers -- and no core can hit should_receive() at its float
        // slot against an upstream that will not forward. Without it a "mixed head" holding both
        // base and float chunks deadlocks in the V relay.
        (v_shares_k_buffer ||
         ((rot_floats_fill_rows_evenly || force_rotate_separate_v) && use_head_chain && use_streaming_compute &&
          !use_attention_sink && num_q_chunks != 0 && (rot_base_chunks * num_cores) % num_q_chunks == 0)) &&
        // !kv_pad_rotation_enabled is MEASURED load-bearing. Removing it does make the rotation
        // engage on the kv-pad path (kv_actual_isl ring MLA, chunk_size_local 256 on a 3x2 grid:
        // base 5, floats 2, active_iters 5 of 8) -- and the device HANGS ("potential hang detected,
        // unrecoverable", cores 15-3/15-2). Rebuilding the rotation cycle over the ordinal sequence
        // of active iterations did NOT fix it, and the reason is structural: under kv-pad the
        // KERNELS derive their own active mask on device (ring_joint_reader.cpp / ring_joint_writer.cpp
        // overwrite active_ring_iter_mask from the kv-pad metadata), so the host's notion of which
        // iterations run is not what the kernels obey. A skipped iteration then means the donor
        // never reaches flush_deferred_save() and its Semaphore(...).up(), while the next receiver
        // blocks in handoff_sem.wait_min() -- a post that never happens.
        //
        // ATTEMPT 2 got much closer and is the right shape: stop trying to match the device mask and
        // make the handoff MASK-INDEPENDENT instead, via three writer changes -- (a) flush a
        // migrating float EAGERLY in its own iteration so its accumulators are in DRAM regardless of
        // whether that core runs the next iteration, (b) have the donor therefore signal slot
        // (ring_iter + 1), the receiver's iteration, and (c) emit the ownership token (a pure signal,
        // no data) even on a SKIPPED iteration so the chain advances through iterations the host
        // cannot predict. That WORKS for kv-pad: rotation engages on the metadata path and
        // test_ring_mla_metadata_matches_scalar_rotation passes BIT-EXACT (3 passed).
        // It was still reverted, because it breaks rot_base_chunks == 2: the latent-V sweep went
        // 4 failed / 6 passed (every q64 id) and determinism 4 failed. The eager flush removes the
        // cross-iteration pending save that the deferred-save / TRID / prefetch machinery at base 2
        // depends on -- the same interaction behind the two bugs fixed earlier in this branch
        // (flush_before_prefetch's positional proxy, and the cross-ring prefetch ordering).
        // So kv-pad is NOT architecturally blocked. The remaining work is narrow and named: make the
        // eager flush coexist with the deferred/TRID scheme at base 2. Note the default q32 set does
        // NOT catch this -- the base-2 shapes are the q64 ids behind RING_MLA_K_SWEEP.
        // test_ring_mla_chunked_kv_actual_isl_rotated_q_split_accuracy is the reachable case to
        // re-attempt with -- the default kv-pad configs have total_q_chunks == 4, too small to engage
        // at any grid.
        //
        // Schedules this rotation cannot model. Balanced/zigzag SKIPS whole Q chunks per device, so
        // the equal-cost-per-chunk counting the rotation is built on stops holding -- a causal mask
        // wants enable_zigzag_balancing instead, which is the right tool for that job. The kv-pad
        // variants do their own per-iteration index remapping, which would have to COMPOSE with
        // rotated ownership rather than merely coexist. A joint K (L != 0) would likewise add
        // per-iteration chunks this schedule does not count; unreachable today, since joint tensors
        // are a video-gen shape and latent-V an MLA one, but validation does not forbid the pair.
        !args.is_balanced &&
        // Partial active masks are handled, not excluded. The rotation schedule is indexed by the
        // ORDINAL of an iteration within the active subsequence (rot_active_ordinal, in
        // chunked_prefill_utils.hpp) rather than by absolute ring_iter, so a skipped iteration is
        // harmless: consecutive ordinals are consecutive EXECUTED iterations, which is the only thing
        // the donor-signal / receiver-wait pair actually requires. For a full mask ordinal ==
        // ring_iter identically, so chunked prefill is bit-identical to before.
        //
        // This is also what unblocked kv-pad, and the reason two earlier attempts failed. Both tried
        // to make the HOST agree with the device about which iterations run -- impossible under
        // kv_pad_from_metadata, where the kernels overwrite active_ring_iter_mask from trace metadata
        // the host never sees. The host does not need to agree: every per-ordinal entry it emits is
        // already a valid partition of all Q chunks, and the KERNELS choose which entries get used.
        // All three read the same device-derived mask before their ring loop (reader :506, writer
        // :569, compute via cb_kv_pad_derived) and all three `continue` past inactive iterations on
        // it, so their ordinals cannot diverge.
        //
        // No guard is needed for the last active iteration either: is_last_active_ring_iter is
        // mask-aware, so that iteration takes the final-output branch and creates NO deferred save,
        // which means the host's float_dest at the last ordinal is simply never read -- no unpaired
        // donor signal, no stale semaphore.
        // Degenerate cases need NO guard: they are provably no-ops, and both were measured.
        // rot_float_chunks == 0 (work divides evenly): with no floats, every core's rotated range
        // i*base .. i*base+base-1 is IDENTICAL to its static range on every iteration, so the
        // rotation reduces to the static split rather than merely resembling it. There is nothing to
        // hand off, and iteration 0 never receives a float.
        // rot_rows_needed == grid_size.y (floats reach every row): no row is float-free, so there is
        // nowhere cheaper to move the +1 mcast slot to; the cycle still permutes WHICH row hosts
        // which float, which is harmless.
        // Both used to decline to the static path purely as a cost guard. Removing them widens the
        // predicate by two terms with no behavioural change. Verified on bh-lb-33 ring-8 after the
        // removal, kimi50k q32/k640 via RING_MLA_SDPA_GRID_OVERRIDE:
        //   grid 7x10 (70 cores | 280 chunks) -> base 4, floats 0, rows_needed 0: ACTIVE, PCC pass
        //   grid 11x2 (base 12, floats 16, rows_needed 2 of 2 rows): ACTIVE, PCC pass
        // and the whole latent-V suite unchanged on the default grid: sweep accuracy 10 passed
        // (includes every base-2 q64 id, the shapes that catch a broken handoff), determinism
        // 10 passed, perf q32/k640 68.04% and q64/k448 68.00% -- both inside their committed bands.
        // Zero-float perf is neutral rather than better, as expected: ON 15.525/15.572 ms vs OFF
        // 15.586/15.556 ms.
        // rot_base_chunks >= 1: every core must own at least one chunk every iteration, since the
        // reader decodes slot (rot_my_count - 1) on padded iterations and would underflow at 0.
        // base == 1 is now supported and is the largest win (measured 1.71x on kimi_k3 q128: 15.587
        // -> 9.091 ms at k224, 14.967 -> 8.746 ms at k256), because it is the worst static imbalance
        // -- half the rows would otherwise pay the +1 mcast slot on every ring iteration.
        //
        // Getting below the old >= 2 bound took three fixes, all in the writer/reader, and all of
        // them are no-ops at base >= 3 (see each site for why):
        //   - the reader pinned need_q_read and the writer pinned single_q_chunk to the FIXED slot
        //     count rather than the static flat count. Compute derives its accumulator mode from
        //     rot_max_slots, and at base == 1 a core whose static count was 1 pushed Q once for the
        //     whole op while compute expected it per iteration -- that was the base == 1 HANG
        //     (cores parked in chain_link.hpp receiver_sem.wait(VALID), the rest behind them);
        //   - the early-flush decision compares flat chunk ids instead of using the positional
        //     q_per_core == 2 proxy, which was evaluated on the current iteration's count while the
        //     pending save came from the previous one. That was the base == 2 wrong-numbers bug;
        //   - the cross-ring prefetch is postponed past this slot's own save when they are the same
        //     chunk, which only happens when a core owns exactly one chunk that iteration.
        rot_base_chunks >= 1;

    struct RotatedIterSched {
        std::vector<uint32_t> my_chunks;   // flat chunk ids; base chunks first, float (if any) last
        uint32_t row_slot_count = 0;       // row max chunk count this iteration = K mcast slots to run
        uint32_t float_migrated_in = 0;    // 1 if the last chunk was owned by another core last iteration
        uint32_t float_dest = kRotNoDest;  // packed physical core owning this float next iteration
    };
    std::vector<std::vector<RotatedIterSched>> rot_sched;  // [core][ring_iter]
    std::vector<uint32_t> rot_handoff_sem_ids;
    // Appends one iteration's chunk-id list padded to the fixed ROTATED_Q_SPLIT length, so every
    // ring iteration occupies the same number of runtime args and the kernels can index by stride.
    const auto append_rot_chunk_ids = [&](CheckedRuntimeArgList& args_out, const RotatedIterSched& sched) {
        for (uint32_t slot = 0; slot < rot_base_chunks + 1; ++slot) {
            args_out.push_back(slot < sched.my_chunks.size() ? sched.my_chunks[slot] : 0);
        }
    };
    if (use_rotated_q_split) {
        const uint32_t cores_per_row = grid_size.x;
        const uint32_t num_rows = grid_size.y;
        const uint32_t rows_needed = rot_rows_needed;
        // The block below indexes batch_chain_configs by core; k_mcast_enabled implies it is built
        // per core, but that implication is ~90 lines away, and getting it wrong is out-of-bounds
        // UB on an empty vector rather than a diagnosable failure.
        TT_FATAL(
            batch_chain_configs.size() == num_cores,
            "RingJoint rotated Q split expects one batch-chain config per core, got {} for {} cores",
            batch_chain_configs.size(),
            num_cores);
        // The row-wide mcast machinery assumes the injector never runs a padded iteration (it is
        // chosen among row-max cores; padded members freeze their K-CB write phase, and the mcast
        // lands at the injector's phase). Preserve that invariant per iteration: within a row
        // hosting floats, the injector's column takes the first float.
        constexpr uint32_t kNoInjectorCol = std::numeric_limits<uint32_t>::max();
        std::vector<uint32_t> injector_col(num_rows, kNoInjectorCol);
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t col = 0; col < cores_per_row; ++col) {
                if (batch_chain_configs[row * cores_per_row + col].is_injector) {
                    injector_col[row] = col;
                    break;
                }
            }
            // Defaulting to column 0 instead would put the first float on a non-injector core and
            // quietly violate the invariant above, as a wrong-but-running program.
            TT_FATAL(
                injector_col[row] != kNoInjectorCol,
                "RingJoint rotated Q split: row {} has no K mcast injector, but k_mcast_enabled "
                "configures one per row",
                row);
        }
        // Float f's owner at iteration t: rows rotate by rows_needed each iteration so every row
        // hosts floats (the +1 mcast slot) an equal ~ring_size*rows_needed/num_rows share of
        // iterations. (row, position) is unique per f within an iteration, so a core owns at most
        // one float at a time; position 0 maps to the row's injector column.
        auto float_owner = [&](uint32_t ring_iter, uint32_t float_idx) {
            const uint32_t row = ((ring_iter * rows_needed) + (float_idx / cores_per_row)) % num_rows;
            const uint32_t pos = float_idx % cores_per_row;
            const uint32_t inj = injector_col[row];
            const uint32_t col = pos == 0 ? inj : (pos <= inj ? pos - 1 : pos);
            return row * cores_per_row + col;
        };
        rot_sched.assign(num_cores, std::vector<RotatedIterSched>(ring_size));
        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
                auto& sched = rot_sched[core_idx][ring_iter];
                sched.my_chunks.reserve(rot_base_chunks + 1);
                for (uint32_t b = 0; b < rot_base_chunks; ++b) {
                    sched.my_chunks.push_back(core_idx * rot_base_chunks + b);
                }
            }
        }
        // owner_of[ring_iter][float_idx]: which core holds float `float_idx` on that iteration.
        // Materialized because the loop below needs each float's owner in the previous and next
        // iterations as well as this one, and re-evaluating float_owner four times per (iter, float)
        // is what made this schedule hard to check.
        std::vector<std::vector<uint32_t>> owner_of(ring_size, std::vector<uint32_t>(rot_float_chunks));
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            for (uint32_t float_idx = 0; float_idx < rot_float_chunks; ++float_idx) {
                owner_of[ring_iter][float_idx] = float_owner(ring_iter, float_idx);
            }
        }
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            for (uint32_t float_idx = 0; float_idx < rot_float_chunks; ++float_idx) {
                const uint32_t owner = owner_of[ring_iter][float_idx];
                auto& sched = rot_sched[owner][ring_iter];
                sched.my_chunks.push_back(rot_base_chunks * num_cores + float_idx);
                // (row, pos) is unique per float within an iteration, so a core holds at most one
                // float and these two fields are assigned at most once each.
                TT_ASSERT(sched.my_chunks.size() == rot_base_chunks + 1);
                sched.float_migrated_in = (ring_iter > 0 && owner_of[ring_iter - 1][float_idx] != owner) ? 1 : 0;
                if (ring_iter + 1 < ring_size) {
                    const uint32_t next_owner = owner_of[ring_iter + 1][float_idx];
                    if (next_owner != owner) {
                        const auto& dest_phys = core_work[next_owner].physical_core;
                        sched.float_dest = rot_pack_dest(dest_phys.x, dest_phys.y);
                    }
                }
            }
            // Every core in a row runs the row's max slot count, so padded members still relay the
            // mcast handshakes.
            for (uint32_t row = 0; row < num_rows; ++row) {
                uint32_t row_max = 0;
                for (uint32_t col = 0; col < cores_per_row; ++col) {
                    row_max = std::max(
                        row_max,
                        static_cast<uint32_t>(rot_sched[row * cores_per_row + col][ring_iter].my_chunks.size()));
                }
                for (uint32_t col = 0; col < cores_per_row; ++col) {
                    rot_sched[row * cores_per_row + col][ring_iter].row_slot_count = row_max;
                }
            }
        }
        // The mcast injector's forward gate (q_iter_local < next_core_q_chunks) must cover the
        // +1-slot iterations of every row, not just the injector's static flat-split count.
        for (auto& cfg : batch_chain_configs) {
            if (cfg.participates && cfg.is_injector) {
                cfg.next_core_q_chunks = rot_base_chunks + 1;
            }
        }
        // Handoff semaphores for the iterations that can RECEIVE a migrated float (1..ring_size-1;
        // iteration 0 starts fresh). Receiver resets its slot to 0 after the wait so a cached
        // program replays cleanly. Slots are REUSED across iterations as a ring of
        // kRotHandoffSemDepth (see rot_handoff_sem_count) rather than one per iteration: program
        // semaphores are a scarce resource (NUM_SEMAPHORES=16, shared with the chain and fused
        // all-gather sems), and one-per-iteration made this feature the largest single consumer
        // (7 of 16 at ring-8) and scaled with ring length. With the V head chain skipped for
        // latent-V, ring-8 now fits comfortably and the count no longer grows with ring_size; the
        // TT_FATAL below keeps that true rather than leaving it to this comment.
        for (uint32_t sem_slot = 0; sem_slot < rot_handoff_sem_count(ring_size); ++sem_slot) {
            const uint32_t sem_id = static_cast<uint32_t>(desc.semaphores.size());
            // The descriptor path has no budget check of its own: an id >= NUM_SEMAPHORES surfaces
            // later as a bare "bitset::set: __position (17) >= _Nb (16)" IndexError from whichever
            // helper next scans for a free id, naming neither SDPA nor this feature.
            // Mirrors tt::tt_metal::NUM_SEMAPHORES (tt_metal/impl/buffers/semaphore.hpp), which is
            // not reachable from a ttnn op through any public header.
            constexpr uint32_t kSemaphoresPerCore = 16;
            TT_FATAL(
                sem_id < kSemaphoresPerCore,
                "Ring MLA rotated Q split needs {} handoff semaphores, but the program has already "
                "allocated {} and a core supports {}. Free some (e.g. skip the V head chain) or set "
                "RING_MLA_DISABLE_ROTATED_Q_SPLIT=1.",
                rot_handoff_sem_count(ring_size),
                desc.semaphores.size(),
                kSemaphoresPerCore);
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = sem_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = core_grid_set,
                .initial_value = 0,
            });
            rot_handoff_sem_ids.push_back(sem_id);
        }
        // Define value = chunk-list length per iteration; each kernel adds its own header words
        // to get its runtime-arg stride (see chunked_prefill_utils.hpp).
        defines["ROTATED_Q_SPLIT"] = std::to_string(rot_base_chunks + 1);

        if (!v_shares_k_buffer) {
            // Separate-V: the V head chain's per-(head, core) forwarding counts come from
            // head_work/head_segments, which were derived from the STATIC flat split. Under rotation
            // each core instead owns the contiguous base range [i*base, (i+1)*base) on every
            // iteration, so rebuild both from that. Floats are deliberately NOT added: they are the
            // tail flat ids and, by the head-boundary term in the predicate, live only in heads with
            // no base chunks -- heads whose chain is therefore never built (segs.size() < 2), so the
            // reader's `nq != chain_head` fallback reads their V from DRAM.
            // The K chain was already built above from the static head_work, which is what its
            // single-head-injector rule needs; only the head chain, built below, sees this version.
            for (auto& work : core_work) {
                work.head_work.clear();
            }
            for (auto& segs : head_segments) {
                segs.clear();
            }
            for (uint32_t i = 0; i < num_cores; ++i) {
                auto& work = core_work.at(i);
                uint32_t flat_chunk = i * rot_base_chunks;
                uint32_t remaining = rot_base_chunks;
                while (remaining > 0) {
                    const auto [head_idx, q_chunk_idx] = decode_flat_chunk(flat_chunk);
                    const uint32_t take = std::min(remaining, num_q_chunks - q_chunk_idx);
                    work.head_work.push_back(CoreHeadWork{.head = head_idx, .q_chunk_count = take});
                    TT_FATAL(
                        head_idx < head_segments.size(),
                        "Rotated head-chain segment index {} is outside {} configured query heads",
                        head_idx,
                        head_segments.size());
                    head_segments[head_idx].push_back(HeadSegmentRef{
                        .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
                    remaining -= take;
                    flat_chunk += take;
                }
            }
        }
        // log_info, matching the decline branch below: one line per program compile (programs are
        // cached), and it is the only way a user can confirm the rotation is actually active for a
        // given shape and core count.
        log_info(
            tt::LogOp,
            "Rotated Q split ACTIVE: base={} floats={} rows_needed={} ring_size={} active_iters={} "
            "kv_pad_rotation={} (ideal slots {} vs flat {})",
            rot_base_chunks,
            rot_float_chunks,
            rows_needed,
            ring_size,
            std::popcount(active_ring_iter_mask),
            kv_pad_rotation_enabled,
            total_q_chunks * ring_size / num_cores,
            ring_size * (rot_base_chunks + 1));
    } else if (v_shares_k_buffer && kernel_chunked) {
        // Latent-V chunked prefill is the shape this rotation exists for, so when it declines there
        // say why at a level a user actually sees. Silently taking the +1-slot-every-iteration split
        // reads as an unexplained regression.
        log_info(
            tt::LogOp,
            "Ring MLA rotated Q split declined (base={} floats={} rows_needed={} of {} rows, ring_size={}, "
            "disabled={} balanced={} kv_pad_rotation={} k_mcast={} all_iters_active={}); using the static "
            "flat split.",
            rot_base_chunks,
            rot_float_chunks,
            rot_rows_needed,
            grid_size.y,
            ring_size,
            rotated_q_split_disabled,
            args.is_balanced,
            kv_pad_rotation_enabled,
            k_mcast_enabled,
            active_ring_iter_mask == full_ring_iter_mask);
    }

    // Head chains (MHA and separate-V shared-K) are built HERE, after the rotated-Q-split
    // decision, because their per-(head, core) forwarding counts must describe whichever Q split
    // actually ships. Moved down from above the K-chain pass; verified behaviour-neutral -- nothing
    // between the old and new positions reads head_chain_configs or mcast_chains, and the K chain's
    // own build_linear_chain call still sees the STATIC head_work it needs for its injector rule.
    // Build head chains for MHA and separate-V shared-K. GQA uses KV-head-grouped chains instead.
    if (use_head_chain && build_kv_chains) {
        for (uint32_t head_id = 0; head_id < static_cast<uint32_t>(head_segments.size()); ++head_id) {
            const auto& segs = head_segments[head_id];
            if (segs.size() < 2) {
                continue;
            }
            std::vector<ChainSegment> chain_segs;
            chain_segs.reserve(segs.size());
            for (const auto& seg : segs) {
                chain_segs.emplace_back(
                    seg.core_idx, core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count);
            }
            build_linear_chain(chain_segs, head_id, head_chain_configs, core_work);
        }
    }

    // Check query-head chain multicast eligibility and configure mcast for eligible chains.
    uint32_t mcast_chains = 0;
    if (use_head_chain && build_kv_chains) {
        struct McastCandidate {
            std::vector<uint32_t> core_indices;
            uint32_t ref_q_chunks;
        };
        std::vector<McastCandidate> candidates;
        candidates.reserve(head_segments.size());
        bool all_eligible = true;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            const auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;
            }

            // Gather chain participants with their per-head q_chunk_count
            std::vector<uint32_t> chain_core_indices;
            chain_core_indices.reserve(segments.size());
            std::vector<uint32_t> chain_q_counts;
            chain_q_counts.reserve(segments.size());
            for (const auto& seg : segments) {
                if (seg.core_idx < head_chain_configs.size() && head_chain_configs[seg.core_idx].participates &&
                    head_chain_configs[seg.core_idx].head == head_id) {
                    chain_core_indices.push_back(seg.core_idx);
                    chain_q_counts.push_back(core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count);
                }
            }

            if (chain_core_indices.size() < 2) {
                continue;
            }

            // Eligibility condition 1: All physical cores share the same Y coordinate
            const uint32_t ref_y = core_work[chain_core_indices[0]].physical_core.y;
            bool same_row = true;
            for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
                if (core_work[chain_core_indices[ci]].physical_core.y != ref_y) {
                    same_row = false;
                    break;
                }
            }

            if (!same_row) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - cores span multiple rows", head_id);
                break;
            }

            // Eligibility condition 2: no non-chain worker cores inside the mcast rectangle.
            uint32_t min_x = core_work[chain_core_indices[0]].physical_core.x;
            uint32_t max_x = min_x;
            for (const auto& ci : chain_core_indices) {
                uint32_t x = core_work[ci].physical_core.x;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }

            bool has_gap = false;
            for (uint32_t ci = 0; ci < num_cores; ++ci) {
                const auto& phys = core_work[ci].physical_core;
                if (phys.y == ref_y && phys.x >= min_x && phys.x <= max_x) {
                    bool in_chain = false;
                    for (const auto& chain_ci : chain_core_indices) {
                        if (chain_ci == ci) {
                            in_chain = true;
                            break;
                        }
                    }
                    if (!in_chain) {
                        has_gap = true;
                        break;
                    }
                }
            }

            if (has_gap) {
                all_eligible = false;
                log_debug(
                    tt::LogOp, "Head {}: mcast ineligible - non-chain worker core inside mcast rectangle", head_id);
                break;
            }

            // Eligibility condition 3: All chain cores must have the same q_chunk_count.
            const uint32_t ref_q_chunks = chain_q_counts[0];
            bool uniform_q_mcast = true;
            for (size_t ci = 1; ci < chain_q_counts.size(); ++ci) {
                if (chain_q_counts[ci] != ref_q_chunks) {
                    uniform_q_mcast = false;
                    break;
                }
            }

            if (!uniform_q_mcast) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - mixed q_chunk_counts", head_id);
                break;
            }

            candidates.push_back(McastCandidate{std::move(chain_core_indices), ref_q_chunks});
        }

        if (all_eligible && !candidates.empty()) {
            mcast_chains = candidates.size();
            for (uint32_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
                const auto& cand = candidates[cand_idx];
                const uint32_t chain_size = cand.core_indices.size();
                const uint32_t num_receivers = chain_size - 1;

                // Find current injector
                uint32_t injector_idx = cand.core_indices[0];
                for (const auto& ci : cand.core_indices) {
                    if (head_chain_configs[ci].is_injector) {
                        injector_idx = ci;
                        break;
                    }
                }

                // Reselect injector for diagonal placement: the n-th chain
                // picks the core at offset n within its chain, wrapping around.
                // This places injectors on the diagonal (0,0), (1,1), (2,2)...
                {
                    uint32_t target_offset = cand_idx % chain_size;
                    uint32_t best_idx = cand.core_indices[target_offset];
                    if (best_idx != injector_idx) {
                        // Clear old injector, set new one
                        head_chain_configs[injector_idx].is_injector = false;
                        head_chain_configs[injector_idx].is_sink = true;
                        head_chain_configs[best_idx].is_injector = true;
                        head_chain_configs[best_idx].is_sink = false;
                        injector_idx = best_idx;
                    }
                }

                uint32_t min_x = core_work[cand.core_indices[0]].physical_core.x;
                uint32_t max_x = min_x;
                for (size_t ci = 1; ci < cand.core_indices.size(); ++ci) {
                    uint32_t x = core_work[cand.core_indices[ci]].physical_core.x;
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                }
                const uint32_t injector_y = core_work[injector_idx].physical_core.y;
                const CoreCoord rect_start = CoreCoord{min_x, injector_y};
                const CoreCoord rect_end = CoreCoord{max_x, injector_y};
                const CoreCoord injector_phys = core_work[injector_idx].physical_core;

                auto& injector_chain = head_chain_configs[injector_idx];
                injector_chain.mcast_start = rect_start;
                injector_chain.mcast_end = rect_end;
                injector_chain.injector_physical = injector_phys;
                injector_chain.mcast_num_dests = num_receivers;
                injector_chain.next_core_q_chunks = cand.ref_q_chunks;

                for (const auto& ci : cand.core_indices) {
                    if (ci == injector_idx) {
                        continue;
                    }
                    auto& receiver_chain = head_chain_configs[ci];
                    receiver_chain.mcast_start = rect_start;
                    receiver_chain.mcast_end = rect_end;
                    receiver_chain.injector_physical = injector_phys;
                    receiver_chain.next_core_q_chunks = 0;
                    receiver_chain.is_sink = true;
                }

                log_debug(
                    tt::LogOp,
                    "Head: mcast enabled - {} receivers, injector core {} (phys_x={}), num_dests={} -> rect "
                    "({},{}) to ({},{})",
                    num_receivers,
                    injector_idx,
                    core_work[injector_idx].physical_core.x,
                    num_receivers,
                    rect_start.x,
                    rect_start.y,
                    rect_end.x,
                    rect_end.y);
            }
        }

        log_debug(
            tt::LogOp,
            "Multicast eligibility: {}/{} chains using mcast (all-or-nothing)",
            mcast_chains,
            static_cast<uint32_t>(candidates.size()));
    }

    // Update mcast compile-time args
    const bool head_mcast_enabled = (mcast_chains > 0);

    if (use_head_chain) {
        reader_compile_time_args[sem_args_offset + kRingJointChainMcastEnabledCompileArgOffset] =
            head_mcast_enabled ? 1 : 0;
    }
    if (enable_kv_chains) {
        const uint32_t head_chain_compile_args = use_head_chain ? kRingJointChainCompileArgCount : 0;
        if (k_uses_batch_chain) {
            const uint32_t batch_mcast_arg_index =
                sem_args_offset + head_chain_compile_args + kRingJointChainMcastEnabledCompileArgOffset;
            reader_compile_time_args[batch_mcast_arg_index] = k_mcast_enabled ? 1 : 0;
        }
        if (gqa_grouped_kv) {
            const uint32_t batch_chain_compile_args = k_uses_batch_chain ? kRingJointChainCompileArgCount : 0;
            const uint32_t gqa_mcast_arg_index = sem_args_offset + head_chain_compile_args + batch_chain_compile_args +
                                                 kRingJointChainMcastEnabledCompileArgOffset;
            reader_compile_time_args[gqa_mcast_arg_index] = gqa_mcast_enabled ? 1 : 0;
        }
    }

    if (gqa_grouped_kv) {
        log_debug(
            tt::LogOp,
            "K/V chain mode: grouped GQA {} (chains={}, participant_cores={}, local_fallback_cores={})",
            enable_kv_chains ? (gqa_mcast_enabled ? "mcast" : fmt::format("unicast, {}", gqa_mcast_fallback_reason))
                             : "independent per-core sliding reads",
            gqa_grouped_chains,
            gqa_grouped_participant_cores,
            gqa_local_fallback_cores);
    } else if (k_uses_batch_chain) {
        log_debug(
            tt::LogOp,
            "K chain mode: batch ({})",
            k_mcast_enabled ? "mcast" : fmt::format("unicast, {}", k_mcast_fallback_reason));
    } else {
        log_debug(tt::LogOp, "K chain mode: head (NHK != 1, {})", head_mcast_enabled ? "mcast" : "unicast");
    }

    // Convert std::map<string,string> defines to KernelDescriptor::Defines vector form.
    KernelDescriptor::Defines kernel_defines(defines.begin(), defines.end());

    // Build kernel descriptors locally so we can append per-core runtime args
    // before pushing them into desc.kernels at the end. KernelDescriptor creation
    // is deferred (just like the original CreateKernel calls were) until after chain
    // construction, since the mcast_enabled compile-time arg is patched above.
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = core_grid_set;
    reader_kernel.compile_time_args = reader_compile_time_args;
    reader_kernel.defines = kernel_defines;
    reader_kernel.config = ReaderConfigDescriptor{};
    if (slot_from_metadata) {
        reader_kernel.emplace_common_runtime_args(
            {tensor_args.slot_id->buffer()->address(),  // smuggled-rta-ok: metadata tensor addr (on-device)
             args.kv_cache_num_layers,
             args.kv_cache_layer_idx,
             tensor_args.kv_actual_isl->buffer()->address()});  // smuggled-rta-ok: metadata tensor addr (on-device)
    }

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = core_grid_set;
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.defines = kernel_defines;
    writer_kernel.config = WriterConfigDescriptor{};
    if (kv_pad_from_metadata) {
        writer_kernel.emplace_common_runtime_args(
            {tensor_args.kv_actual_isl->buffer()->address()});  // smuggled-rta-ok: metadata tensor addr (on-device)
    }

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = core_grid_set;
    compute_kernel.compile_time_args = compute_compile_time_args;
    compute_kernel.defines = kernel_defines;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // Prefer the computed even distribution above for chain construction
        const auto& work = core_work.at(i);
        uint32_t global_q_start = work.global_q_start;
        uint32_t global_q_end = work.global_q_start + work.global_q_count;

        // log the above
        log_debug(tt::LogOp, "core: {}", i);
        log_debug(tt::LogOp, "x={},y={}", core.x, core.y);
        log_debug(tt::LogOp, "global_q_start: {}", global_q_start);
        log_debug(tt::LogOp, "global_q_end: {}", global_q_end);

        CheckedRuntimeArgList reader_args;
        reader_args.push_back(q_buf);
        reader_args.push_back(k_buf);
        reader_args.push_back(v_buf);
        reader_args.push_back(gathered_k_buf);
        reader_args.push_back(gathered_v_buf);
        if (L != 0) {
            reader_args.push_back(joint_tensor_q->buffer());
            reader_args.push_back(joint_tensor_k->buffer());
            reader_args.push_back(joint_tensor_v->buffer());
        }
        reader_args.push_back(attention_sink_buf);
        // Read by the kernel right after attention_sink_addr and before global_q_start.
        if (joint_is_sharded) {
            reader_args.push_back(gathered_joint_tensor_k->buffer());
            reader_args.push_back(gathered_joint_tensor_v->buffer());
        }
        reader_args.push_back(global_q_start);
        reader_args.push_back(global_q_end);
        reader_args.push_checked(
            runtime_arg_layout.reader_kv_cache_batch_idx, kv_cache_batch_idx, "reader.kv_cache_batch_idx");
        if (use_head_chain) {
            const auto& head_chain = head_chain_configs.at(i);
            log_debug(
                tt::LogOp,
                "core logical=({},{})->phys=({},{}), q=[{},{}), head_chain={{part:{}, inj:{}, sink:{}, "
                "h:{}, next_cnt:{}}}",
                core.x,
                core.y,
                core_work.at(i).physical_core.x,
                core_work.at(i).physical_core.y,
                global_q_start,
                global_q_end,
                head_chain.participates,
                head_chain.is_injector,
                head_chain.is_sink,
                head_chain.head,
                head_chain.next_core_q_chunks);

            // Head chain: MHA uses it for K/V; separate-V shared-K uses it for V only.
            std::vector<uint32_t> head_chain_args;
            head_chain.append_to_args(head_chain_args);
            reader_args.append(head_chain_args);
        }

        if (enable_kv_chains) {
            if (k_uses_batch_chain) {
                const auto& batch_chain = batch_chain_configs.at(i);
                std::vector<uint32_t> batch_chain_args;
                batch_chain.append_to_args(batch_chain_args);
                reader_args.append(batch_chain_args);
                reader_args.push_back(k_chain_max_q[i]);
            }
            if (gqa_grouped_kv) {
                const auto& gqa_chain = gqa_chain_configs.at(i);
                std::vector<uint32_t> gqa_chain_args;
                gqa_chain.append_to_args(gqa_chain_args);
                reader_args.append(gqa_chain_args);
                reader_args.push_back(gqa_chain_max_q[i]);
            }
        }

        reader_args.push_checked(runtime_arg_layout.reader_logical_nt, logical_nt, "reader.logical_nt");
        reader_args.push_checked(
            runtime_arg_layout.reader_active_ring_iter_mask, active_ring_iter_mask, "reader.active_ring_iter_mask");

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        std::vector<uint32_t> reader_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_signaler_args);
        reader_args.append(reader_signaler_args);

        // Rotated Q split: per ring iteration [row_slot_count, my_count, chunk ids].
        // Header width must stay kRotReaderIterHeaderWords.
        if (use_rotated_q_split) {
            for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
                const auto& sched = rot_sched[i][ring_iter];
                reader_args.push_back(sched.row_slot_count);
                reader_args.push_back(static_cast<uint32_t>(sched.my_chunks.size()));
                append_rot_chunk_ids(reader_args, sched);
            }
        }

        reader_kernel.emplace_runtime_args(core, reader_args.args);

        // Writer args
        CheckedRuntimeArgList writer_args;
        writer_args.push_back(out_buf);
        writer_args.push_back(joint_out_buf);
        writer_args.push_back(stats_buf);
        writer_args.push_back(global_q_start);
        writer_args.push_back(global_q_end);
        writer_args.push_checked(runtime_arg_layout.writer_logical_nt, logical_nt, "writer.logical_nt");
        writer_args.push_checked(
            runtime_arg_layout.writer_active_ring_iter_mask, active_ring_iter_mask, "writer.active_ring_iter_mask");
        writer_args.push_checked(
            runtime_arg_layout.writer_single_valid_kv_chunk_mask,
            single_valid_kv_chunk_mask,
            "writer.single_valid_kv_chunk_mask");
        std::vector<uint32_t> writer_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(writer_signaler_args);
        // The writer's RingSDPAOpReceiver is constructed with wait_for_op_signal=false and so
        // consumes only the first few of these; ring_joint_writer.cpp steps over the remainder to
        // reach the rotated block appended below. Pin the count here so a signaler that grows a word
        // fails the build instead of silently shifting every rotated arg.
        TT_FATAL(
            !use_rotated_q_split || writer_signaler_args.size() == RingSDPAFusedOpSignaler::kRtArgCount,
            "RingJoint rotated Q split assumes {} fused-op writer args, got {}",
            RingSDPAFusedOpSignaler::kRtArgCount,
            writer_signaler_args.size());
        writer_args.append(writer_signaler_args);
        // Rotated Q split: the handoff semaphore ids, indexed by (ring_iter - 1) % count on both
        // sides, then per ring iteration [my_count, float_migrated_in, float_dest, chunk ids].
        // Header width must stay kRotWriterIterHeaderWords.
        if (use_rotated_q_split) {
            for (uint32_t sem_slot = 0; sem_slot < rot_handoff_sem_count(ring_size); ++sem_slot) {
                writer_args.push_back(rot_handoff_sem_ids[sem_slot]);
            }
            for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
                const auto& sched = rot_sched[i][ring_iter];
                writer_args.push_back(static_cast<uint32_t>(sched.my_chunks.size()));
                writer_args.push_back(sched.float_migrated_in);
                writer_args.push_back(sched.float_dest);
                append_rot_chunk_ids(writer_args, sched);
            }
        }
        writer_kernel.emplace_runtime_args(core, writer_args.args);

        // Compute args
        CheckedRuntimeArgList compute_args;
        compute_args.push_back(global_q_start);
        compute_args.push_back(global_q_end);
        compute_args.push_back(ring_size);
        compute_args.push_back(transport_rank);
        compute_args.push_back(forward_writes_expected);
        compute_args.push_back(backward_writes_expected);
        compute_args.push_checked(runtime_arg_layout.compute_logical_nt, logical_nt, "compute.logical_nt");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_start_tile,
            kv_pad_q_mapping.q_pre_wrap_start_tile,
            "compute.q_pre_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_tile_count,
            kv_pad_q_mapping.q_pre_wrap_tile_count,
            "compute.q_pre_wrap_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_post_wrap_start_tile,
            kv_pad_q_mapping.q_post_wrap_start_tile,
            "compute.q_post_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_valid_tile_count,
            kv_pad_q_mapping.q_valid_tile_count,
            "compute.q_valid_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_active_ring_iter_mask, active_ring_iter_mask, "compute.active_ring_iter_mask");
        // Rotated Q split: per ring iteration [my_count, chunk ids].
        // Header width must stay kRotComputeIterHeaderWords.
        if (use_rotated_q_split) {
            for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
                const auto& sched = rot_sched[i][ring_iter];
                compute_args.push_back(static_cast<uint32_t>(sched.my_chunks.size()));
                append_rot_chunk_ids(compute_args, sched);
            }
        }
        compute_kernel.emplace_runtime_args(core, compute_args.args);
    }

    // Push the SDPA kernels into desc before invoking the all-gather helper so
    // the helper appends its own kernels after these. Their indices in
    // desc.kernels will be 0/1/2 respectively (they are the first kernels appended).
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k};
    std::vector<Tensor> all_gather_output_tensors = {gathered_input_tensor_k};
    if (!v_shares_k_buffer) {
        all_gather_input_tensors.push_back(input_tensor_v);
        all_gather_output_tensors.push_back(gathered_input_tensor_v);
    }
    // Sharded-joint path: include joint K/V in the same fused gather so gathered_joint_k/v are
    // produced by the gather that actually runs on device. Same as spatial K/V: the AG omits each
    // device's own local slice from the gathered buffer; the SDPA reader fetches that slice from
    // the local joint tensor when ring_id == ring_index, and remote slices from the gathered buffer.
    // Sliding-window attention does not support joint tokens, so this never feeds the halo path below.
    if (joint_is_sharded) {
        TT_FATAL(
            tensor_args.gathered_joint_k.has_value() && tensor_args.gathered_joint_v.has_value(),
            "joint_is_sharded but gathered_joint_k/v not set in tensor_args");
        all_gather_input_tensors.push_back(*joint_tensor_k);
        all_gather_output_tensors.push_back(tensor_args.gathered_joint_k.value());
        all_gather_input_tensors.push_back(*joint_tensor_v);
        all_gather_output_tensors.push_back(tensor_args.gathered_joint_v.value());
    }
    if (has_sliding_window) {
        const bool linear_wrap_halo = args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Linear &&
                                      transport_rank + 1 == ring_size;
        std::optional<MeshCoordinate> halo_transport_coord = forward_coord;
        std::optional<MeshCoordinate> halo_destination_coord = forward_coord;
        if (linear_wrap_halo) {
            // Preserve the cyclic logical Q/K layout on a physical line: send device N-1's
            // predecessor tail backward over N-1 hops to device 0. The transport connection is
            // its immediate backward neighbor; the packet route names the actual endpoint.
            TT_FATAL(backward_coord.has_value(), "Linear sliding halo wrap requires a backward neighbor");
            TT_FATAL(
                args.all_gather_operation_attributes.cluster_axis.has_value(),
                "Linear sliding halo wrap requires cluster_axis");
            halo_transport_coord = backward_coord;
            halo_destination_coord = coord;
            halo_destination_coord->operator[](args.all_gather_operation_attributes.cluster_axis.value()) = 0;
        }
        TT_FATAL(
            halo_transport_coord.has_value() && halo_destination_coord.has_value(),
            "Sliding attention requires a next-device route");
        const RingAttentionNeighborHaloConfig neighbor_halo{
            .send_to_next_start_Ht = chunked_sliding_halo_layout.send_tail_start_tile(transport_rank),
            .send_to_next_count_Ht = chunked_sliding_halo_layout.halo_tile_rows,
            .send_backward = linear_wrap_halo,
            .unicast_hops = linear_wrap_halo ? ring_size - 1 : 1,
        };
        log_debug(
            tt::LogOp,
            "Chunked sliding K/V halo: device={}, predecessor={}, tail=[{}, {}), payload_rows={}",
            transport_rank,
            (transport_rank + ring_size - 1) % ring_size,
            neighbor_halo.send_to_next_start_Ht,
            neighbor_halo.send_to_next_start_Ht + neighbor_halo.send_to_next_count_Ht,
            neighbor_halo.send_to_next_count_Ht);
        ring_attention_neighbor_halo_exchange_helper(
            desc,
            all_gather_input_tensors,
            coord,
            halo_transport_coord.value(),
            halo_destination_coord.value(),
            all_gather_output_tensors,
            args.all_gather_operation_attributes.num_links,
            args.all_gather_operation_attributes.ring_size,
            transport_rank,
            args.all_gather_operation_attributes.topology,
            args.all_gather_operation_attributes.semaphore,
            args.all_gather_operation_attributes.sub_device_id,
            all_gather_fused_op_signaler.value(),
            args.ccl_core_grid_offset,
            args.all_gather_operation_attributes.core_allocation_strategy,
            args.kv_cache_batch_idx,
            compute_gather_valid_Ht(args, tensor_args),
            neighbor_halo);
    } else {
        // Append the all-gather portion to `desc`. Buffer addresses are auto-patched on cache hits; the
        // indexed-mode input_batch_base scalar is re-patched in apply_ring_joint_scalar_runtime_args.
        // Single-slot gather is engaged whenever the op is in indexed mode -- either a host
        // kv_cache_batch_idx (scalar path) or a metadata tensor (trace-safe path, where the slot is read
        // on-device from metadata[0]). On the metadata path the host slot is absent, so pass a valid
        // placeholder (0) to turn on single-slot structure; the AG reader recomputes the real offset.
        const bool ag_indexed = args.has_indexed_kv_cache() || tensor_args.has_metadata();
        const std::optional<uint32_t> gather_slice_idx =
            ag_indexed ? std::optional<uint32_t>(args.kv_cache_batch_idx.value_or(0)) : std::nullopt;
        ring_attention_all_gather_async_multi_core_with_workers_helper(
            desc,
            all_gather_input_tensors,
            coord,
            forward_coord,
            backward_coord,
            all_gather_output_tensors,
            args.all_gather_operation_attributes.dim,
            args.all_gather_operation_attributes.num_links,
            args.all_gather_operation_attributes.ring_size,
            transport_rank,
            args.all_gather_operation_attributes.topology,
            args.all_gather_operation_attributes.semaphore,
            args.all_gather_operation_attributes.sub_device_id,
            all_gather_fused_op_signaler,
            args.ccl_core_grid_offset,
            args.all_gather_operation_attributes.core_allocation_strategy,
            gather_slice_idx,
            // Bound the gather to the logical_n-valid prefix at create time so the first (cache-miss)
            // dispatch moves only kv_actual-sized data, not the whole oversized cache. Re-patched per
            // dispatch on cache hits in apply_ring_joint_scalar_runtime_args.
            compute_gather_valid_Ht(args, tensor_args),
            tensor_args.slot_id,
            tensor_args.kv_actual_isl,
            // chunk_local_tiles: per-device Q slab in tiles, for the reader's on-device gather-extent recompute.
            tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT,
            // (user, layer)-major KV-cache batch factor: the all-gather reader computes the gathered slot as
            // slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx. Defaults (1, 0) keep callers unaffected.
            args.kv_cache_num_layers,
            args.kv_cache_layer_idx,
            // Share the split-forwarding decision derived above so the all-gather only splits when this
            // consumer implements the second-half wait.
            sdpa_fused_op_signaler->split_forwarding_enabled,
            rank_mapping);
    }

    return desc;
}

}  // namespace

// Ring-joint SDPA returns a WorkloadDescriptor with one ProgramDescriptor per coord:
// transport rank / forward_coord / backward_coord (used by the all-gather portion) all
// depend on the mesh coordinate, so descriptors cannot be shared across coords. Returning
// a WorkloadDescriptor (rather than a per-coord ProgramDescriptor) keeps the framework on
// its no-rebuild cache-hit fast path; the dynamic scalar runtime args (indexed kv-cache /
// kv-pad rotation) are still re-applied every dispatch by override_runtime_arguments below.
tt::tt_metal::WorkloadDescriptor RingJointSDPAProgramFactory::create_workload_descriptor(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_ring_joint_sdpa_program_descriptor(args, tensor_args, output_tensors, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

RingJointSDPAMeshWorkloadFactory::cached_mesh_workload_t RingJointSDPAMeshWorkloadFactory::create_mesh_workload(
    const RingJointSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void RingJointSDPAMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const ttnn::MeshCoordinate coord = coordinate_range.start_coord();
        TT_FATAL(
            coord == coordinate_range.end_coord(),
            "Expected RingJointSDPA cached programs to cover a single coordinate, got range {} to {}",
            coord,
            coordinate_range.end_coord());
        apply_ring_joint_scalar_runtime_args(program, args, tensor_args, coord);
    }
}

}  // namespace ttnn::prim
