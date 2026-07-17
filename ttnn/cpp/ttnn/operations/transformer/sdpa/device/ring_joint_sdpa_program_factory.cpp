// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "kernels/dataflow/chunked_prefill_utils.hpp"
#include "kernels/sliding_window_geometry.hpp"
#include "sliding_halo_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_id_sequencer.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
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
    uint32_t num_joint_k_chunks = 0;
    uint32_t joint_seq_len = 0;
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
    uint32_t device_index = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;
};

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
    const RingWritePlan& ring_write_plan, const RingJointRuntimeDerivation& derivation, bool is_balanced) {
    RingWorkPlan plan;
    RingIdSequencer seq(
        ring_write_plan.device_index,
        derivation.ring_size,
        ring_write_plan.backward_writes_expected,
        ring_write_plan.forward_writes_expected);
    // RingIdSequencer accepts a sync callback for kernel semaphore waits. Host planning only needs the
    // same ring-id sequence, so use a no-op callback.
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < derivation.ring_size; ++ring_iter) {
        const uint32_t ring_id = seq.get_next_ring_id(noop_sync);
        const bool joint_contributes =
            ring_id == derivation.ring_size - 1 && derivation.num_joint_k_chunks > 0 && derivation.joint_seq_len != 0;
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
        const uint32_t valid_kv_chunks =
            valid_spatial_kv_chunks + (joint_contributes ? derivation.num_joint_k_chunks : 0);
        // Non-pad chunked prefill historically keeps every spatial ring iter active; KV-pad rotation
        // tightens this to valid chunks so empty pad slabs can be skipped.
        const bool has_kv_work =
            (derivation.kernel_chunked && !derivation.kv_pad_rotation_enabled) || valid_spatial_kv_chunks > 0;
        const bool ring_iter_does_work =
            (has_kv_work || joint_contributes) &&
            !(derivation.kernel_is_causal && ring_write_plan.device_index < ring_id && !is_balanced);
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
    plan.device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_q, coord, args.all_gather_operation_attributes.cluster_axis);

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
        plan.device_index,
        args.all_gather_operation_attributes.topology);
    (void)dynamic_alternate;
    if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring && plan.device_index % 2 == 0) {
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
    const bool has_joint_tensors = tensor_args.joint_q.has_value();
    const uint32_t k_chunk_size = args.get_k_chunk_size();
    const uint32_t q_local_padded_N = q_shape[2];
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const uint32_t L = has_joint_tensors ? tensor_args.joint_q->logical_shape()[2] : 0;

    RingJointRuntimeDerivation derivation;
    derivation.logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    derivation.ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    derivation.q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    derivation.kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    derivation.q_chunk_group_tile_count = derivation.q_local_padded_Nt * derivation.ring_size;
    derivation.num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);
    derivation.k_chunk_tile_count = k_chunk_size / tt::constants::TILE_HEIGHT;
    derivation.num_joint_k_chunks = tt::div_up(L, k_chunk_size);
    derivation.joint_seq_len = L;
    // Cross is non-causal on chunked-shaped tensors, so kernels and the work planner use the
    // non-chunked path.
    derivation.kernel_chunked = tensor_args.is_chunked() && !args.is_cross;
    derivation.kv_pad_rotation_enabled = args.has_kv_pad_rotation();
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
        derivation.kv_pad_rotation_enabled ? args.kv_actual_isl.value() / tt::constants::TILE_HEIGHT : 0;
    if (derivation.kv_pad_rotation_enabled) {
        plan.kv_pad_q_mapping = build_kv_pad_q_mapping(
            kv_actual_tile_count,
            derivation.logical_nt,
            derivation.ring_size,
            derivation.q_local_padded_Nt,
            ring_write_plan.device_index);
    }

    if (args.has_sliding_window()) {
        // Sliding folds its local and predecessor ranges into one synthetic ring iteration.
        plan.ring_work_plan.masks.active_ring_iter_mask = 1;
    } else {
        plan.ring_work_plan = build_ring_work_plan(ring_write_plan, derivation, args.is_balanced);
    }
    plan.kernel_chunked = derivation.kernel_chunked;
    plan.kernel_is_causal = derivation.kernel_is_causal;
    return plan;
}

RingJointRuntimeArgLayout get_runtime_arg_layout(
    const ttnn::prim::RingJointSDPAParams& args, const ttnn::prim::RingJointSDPAInputs& tensor_args) {
    const auto& k_shape = tensor_args.gathered_k.logical_shape();
    const bool has_joint_tensors = tensor_args.joint_q.has_value();
    const uint32_t L = has_joint_tensors ? tensor_args.joint_q->logical_shape()[2] : 0;
    const uint32_t NH = tensor_args.input_q.logical_shape()[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = tensor_args.v_num_heads();
    const bool v_shares_k_buffer = tensor_args.has_latent_v();
    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);

    RingJointRuntimeArgLayout layout;
    layout.grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                       : tensor_args.input_q.device()->compute_with_storage_grid_size();

    const uint32_t joint_buffer_args = (L != 0) ? kReaderJointBufferArgCount : 0;
    const bool enable_kv_chains = !args.has_sliding_window();
    const bool use_head_chain = enable_kv_chains && !gqa_grouped_kv;
    const uint32_t head_chain_args = use_head_chain ? kRingJointChainConfigArgCount : 0;
    const uint32_t batch_chain_args =
        enable_kv_chains && k_uses_batch_chain ? (kRingJointChainConfigArgCount + kReaderBatchChainExtraArgCount) : 0;
    const uint32_t gqa_chain_args =
        enable_kv_chains && gqa_grouped_kv ? (kRingJointChainConfigArgCount + kReaderGQAChainExtraArgCount) : 0;
    layout.reader_kv_cache_batch_idx = kReaderBaseBufferArgCount + joint_buffer_args + 2;
    layout.reader_logical_nt = kReaderBaseBufferArgCount + joint_buffer_args + kReaderQWorkArgCount + head_chain_args +
                               batch_chain_args + gqa_chain_args;
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
    if (!args.has_kv_pad_rotation()) {
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

    const RingWritePlan ring_write_plan =
        patch_kv_pad_rotation ? build_ring_write_plan(args, tensor_args, mesh_dispatch_coordinate) : RingWritePlan{};
    const RingJointRuntimePlan runtime_plan =
        patch_kv_pad_rotation ? build_runtime_plan(args, tensor_args, ring_write_plan) : RingJointRuntimePlan{};
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
                        const uint32_t input_batch_base = kv_cache_batch_idx * num_heads * Ht * Wt;
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

    if (args.has_sliding_window()) {
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
            runtime_chunked_sliding_layout.send_tail_start_tile(ring_write_plan.device_index);
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
        - L: the logical joint sequence length

    input_tensor_q: B x NH  x q_local_padded_N  x DH
    input_tensor_k: B x NHK x kv_local_padded_N x DH
    input_tensor_v: B x NH  x kv_local_padded_N x DH

    gathered_input_tensor_k: B x NHK x padded_N x DH
    gathered_input_tensor_v: B x NH  x padded_N x DH

    joint_tensor_q: B x NH x L x DH
    joint_tensor_k: B x NH x L x DH
    joint_tensor_v: B x NH x L x DH

    output_tensor: B x NH x q_local_padded_N x DH
    joint_output_tensor: B x NH x L x DH


    The algorithm is roughly described below.
    - for each ring iteration:
        - read a Q chunk from input_tensor_q
        - for each KV chunk in kv_local_padded_N:
            - on the first ring iteration, read from local input_tensor_k and input_tensor_v
            - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
            - on the last ring iteration, also read from joint_tensor_k and joint_tensor_v
            - if the KV chunk is from the non-joint input and contains the global token index (logical_n - 1), generate
    a mask
            - else if the KV chunk is from non-joint input and contains the local token index (kv_local_padded_N - 1),
    generate an attention mask
            - else if the KV chunk is from the joint input and contains the local token index (L - 1), generate a mask
            - compute attention
        - write the output Q chunk
        - if this is not the first ring iteration, do the LSE update.
    */

    log_debug(tt::LogOp, "RingJointSDPA create_descriptor");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const bool v_shares_k_buffer = tensor_args.has_latent_v();
    const auto& input_tensor_v = tensor_args.input_v.has_value() ? tensor_args.input_v.value() : input_tensor_k;

    const bool has_joint_tensors = tensor_args.joint_q.has_value();
    const Tensor* joint_tensor_q = has_joint_tensors ? &tensor_args.joint_q.value() : nullptr;
    const Tensor* joint_tensor_k = has_joint_tensors ? &tensor_args.joint_k.value() : nullptr;
    const Tensor* joint_tensor_v =
        has_joint_tensors ? (tensor_args.joint_v.has_value() ? &tensor_args.joint_v.value() : joint_tensor_k) : nullptr;

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
    const uint32_t device_index = ring_write_plan.device_index;
    const uint32_t forward_writes_expected = ring_write_plan.forward_writes_expected;
    const uint32_t backward_writes_expected = ring_write_plan.backward_writes_expected;

    std::optional<MeshCoordinate> forward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        1,
        args.all_gather_operation_attributes.topology,
        args.all_gather_operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        -1,
        args.all_gather_operation_attributes.topology,
        args.all_gather_operation_attributes.cluster_axis);

    log_debug(tt::LogOp, "device index: {}", device_index);
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
        device_index,
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
    const bool indexed_kv_cache = args.has_indexed_kv_cache();
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
    const uint32_t L = has_joint_tensors ? joint_tensor_q->logical_shape()[2] : 0;
    const uint32_t vDH = tensor_args.v_head_dim(args.latent_v_head_dim);
    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    const bool use_head_chain = enable_kv_chains && !gqa_grouped_kv;

    const uint32_t q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Nt = padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t gathered_padded_Nt = gathered_padded_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const bool kv_pad_rotation_enabled = args.has_kv_pad_rotation();
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
    const bool joint_has_padding = L > 0 && (L % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled || has_sliding_window;

    // Partial tile support when padding boundary falls inside a tile.
    const uint32_t joint_l_partial_col = L % tt::constants::TILE_HEIGHT;
    const uint32_t partial_mask_tiles =
        (compile_time_global_n_partial_col != 0 ? 1 : 0) + (joint_l_partial_col != 0 ? 1 : 0);
    const uint32_t edge_mask_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : (diag_tile_enabled ? 1 : 0);
    // Single CB holds neginf, either the causal diagonal or sliding edge palette, and partial masks.
    const uint32_t total_lightweight_mask_tiles = 1 + edge_mask_tiles + partial_mask_tiles;

    const uint32_t num_local_q_chunks = tt::div_up(q_local_padded_N, q_chunk_size);
    const uint32_t num_joint_q_chunks = tt::div_up(L, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);
    const uint32_t num_joint_k_chunks = tt::div_up(L, k_chunk_size);

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
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);

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
        Lt,
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
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(stats_output_tensor.buffer()).append_to(writer_compile_time_args);

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
    };

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

    const std::vector<uint32_t> cb_compile_time_args = {
        cb_q_in,     cb_k_in,     cb_v_in,         cb_mask_in,       cb_scale_in,    cb_identity_scale_in,
        cb_stats_in, cb_prev_out, cb_col_identity, cb_recip_scratch, cb_sum_out,     cb_sum_in,
        cb_signal,   cb_out,      cb_stats_out,    cb_qk_im,         cb_out_im_A,    cb_out_im_B,
        cb_max_A,    cb_max_B,    cb_sum_A,        cb_sum_B,         cb_exp_max_diff};
    const std::vector<uint32_t> reader_cb_compile_time_args = {cb_q_in, cb_k_in, cb_v_in, cb_attention_sink};
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

    // Build head chains for MHA and separate-V shared-K. GQA uses KV-head-grouped chains instead.
    if (use_head_chain) {
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
    if (use_head_chain) {
        struct McastCandidate {
            std::vector<uint32_t> core_indices;
            uint32_t ref_q_chunks;
        };
        std::vector<McastCandidate> candidates;
        bool all_eligible = true;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            const auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;
            }

            // Gather chain participants with their per-head q_chunk_count
            std::vector<uint32_t> chain_core_indices;
            std::vector<uint32_t> chain_q_counts;
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

    uint32_t gqa_grouped_chains = 0;
    uint32_t gqa_grouped_participant_cores = 0;
    uint32_t gqa_local_fallback_cores = 0;
    bool gqa_mcast_enabled = false;
    std::string gqa_mcast_fallback_reason;
    std::vector<uint32_t> gqa_chain_max_q(gqa_chain_configs.size(), 0);  // per-core loop-padding count
    if (gqa_grouped_kv && enable_kv_chains) {
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
    if (k_uses_batch_chain && enable_kv_chains) {
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

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = core_grid_set;
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.defines = kernel_defines;
    writer_kernel.config = WriterConfigDescriptor{};

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
        writer_args.append(writer_signaler_args);
        writer_kernel.emplace_runtime_args(core, writer_args.args);

        // Compute args
        CheckedRuntimeArgList compute_args;
        compute_args.push_back(global_q_start);
        compute_args.push_back(global_q_end);
        compute_args.push_back(ring_size);
        compute_args.push_back(device_index);
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
    if (has_sliding_window) {
        TT_FATAL(forward_coord.has_value(), "Sliding attention requires a cyclic next-device route");
        const RingAttentionNeighborHaloConfig neighbor_halo{
            .send_to_next_start_Ht = chunked_sliding_halo_layout.send_tail_start_tile(device_index),
            .send_to_next_count_Ht = chunked_sliding_halo_layout.halo_tile_rows,
        };
        log_debug(
            tt::LogOp,
            "Chunked sliding K/V halo: device={}, predecessor={}, tail=[{}, {}), payload_rows={}",
            device_index,
            (device_index + ring_size - 1) % ring_size,
            neighbor_halo.send_to_next_start_Ht,
            neighbor_halo.send_to_next_start_Ht + neighbor_halo.send_to_next_count_Ht,
            neighbor_halo.send_to_next_count_Ht);
        ring_attention_neighbor_halo_exchange_helper(
            desc,
            all_gather_input_tensors,
            coord,
            forward_coord.value(),
            all_gather_output_tensors,
            args.all_gather_operation_attributes.num_links,
            args.all_gather_operation_attributes.ring_size,
            device_index,
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
            device_index,
            args.all_gather_operation_attributes.topology,
            args.all_gather_operation_attributes.semaphore,
            args.all_gather_operation_attributes.sub_device_id,
            all_gather_fused_op_signaler,
            args.ccl_core_grid_offset,
            args.all_gather_operation_attributes.core_allocation_strategy,
            args.kv_cache_batch_idx,
            compute_gather_valid_Ht(args, tensor_args));
    }

    return desc;
}

}  // namespace

// Ring-joint SDPA returns a WorkloadDescriptor with one ProgramDescriptor per coord:
// device_index / forward_coord / backward_coord (used by the all-gather portion) all
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
