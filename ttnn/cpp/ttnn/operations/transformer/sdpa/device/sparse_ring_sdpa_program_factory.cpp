// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_program_factory.hpp"

#include <cstdlib>
#include <string>
#include "kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <cmath>
#include <deque>
#include <limits>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/common_values.hpp>

using namespace tt::tt_metal;
using ttnn::Tensor;

namespace {

namespace ag_rt = ttnn::ring_attention_all_gather_async_detail;
namespace ag_dyn = ttnn::experimental::prim::ring_attention_all_gather_async_dynamic;
namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Host-side summary of which ring-loop iterations do useful SDPA work. Bits are indexed by ring_iter,
// not ring_id; kernels still advance their sync/ring-id sequence on every iter before checking the mask.
struct RingWorkMasks {
    uint32_t active_ring_iter_mask = 0;
    uint32_t single_valid_kv_chunk_mask = 0;
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
};

struct RingWritePlan {
    uint32_t transport_rank = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
};

constexpr uint32_t kReaderKernelIndex = 0;
constexpr uint32_t kWriterKernelIndex = 1;
constexpr uint32_t kComputeKernelIndex = 2;
// Q blocks a core may hold in L1 at once (the kernels' per-block state is a 32-bit bitmask).
constexpr uint32_t kSolResidentMaxSlots = 32;

// The all-gather appends reader-forward, writer-forward, reader-backward, writer-backward after the three SDPA kernels.
constexpr uint32_t kFirstAllGatherKernelIndex = 3;
constexpr uint32_t kAllGatherReaderForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kReaderForwardKernelOffset;
constexpr uint32_t kAllGatherWriterForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kWriterForwardKernelOffset;
constexpr uint32_t kAllGatherReaderBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kReaderBackwardKernelOffset;
constexpr uint32_t kAllGatherWriterBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kWriterBackwardKernelOffset;

// Runtime-arg offsets. Descriptor construction appends the same slots through CheckedRuntimeArgList, so layout
// edits fail on program creation.
// Q, K, V, gathered K, gathered V, and attention sink (always null).
constexpr uint32_t kReaderBaseBufferArgCount = 6;
constexpr uint32_t kReaderQWorkArgCount = 3;
constexpr uint32_t kRingJointChainConfigArgCount = ring_joint::kChainConfigRuntimeArgCount;
constexpr uint32_t kRingJointChainCompileArgCount = ring_joint::kChainCompileArgCount;
constexpr uint32_t kRingJointChainSemaphoreCompileArgCount = ring_joint::kChainSemaphoreCompileArgCount;
constexpr uint32_t kRingJointChainMcastEnabledCompileArgOffset = ring_joint::kChainMcastEnabledCompileArgOffset;
constexpr uint32_t kReaderBatchChainExtraArgCount = 1;
constexpr uint32_t kReaderGQAChainExtraArgCount = 1;
constexpr uint32_t kWriterBaseArgCount = 3;  // output address, Q range
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
            "SparseRingSDPA runtime arg {} expected index {}, got {} before append",
            name,
            expected_index,
            size);
        push_back(value);
    }
};

// Build the per-device ring-loop masks passed to reader/compute/writer. This mirrors the kernel ring-id order and
// marks the ring_iter entries whose shard has K/V inside logical_n.
RingWorkMasks build_ring_work_masks(
    const ttnn::prim::SparseRingSDPAParams& args,
    const ttnn::prim::SparseRingSDPAInputs& tensor_args,
    const RingWritePlan& ring_write_plan) {
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    TT_FATAL(
        ring_size <= std::numeric_limits<uint32_t>::digits,
        "SparseRingSDPA host ring-work masks support up to {} ring iterations. Got ring_size={}",
        std::numeric_limits<uint32_t>::digits,
        ring_size);
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    const uint32_t kv_local_padded_Nt = tensor_args.local_kv_seq_len() / tt::constants::TILE_HEIGHT;
    const uint32_t k_chunk_tile_count = args.get_k_chunk_size() / tt::constants::TILE_HEIGHT;
    const uint32_t num_local_k_chunks = tt::div_up(tensor_args.local_kv_seq_len(), args.get_k_chunk_size());

    RingWorkMasks masks;
    RingIdSequencer seq(
        ring_write_plan.transport_rank,
        ring_size,
        ring_write_plan.backward_writes_expected,
        ring_write_plan.forward_writes_expected);
    // RingIdSequencer accepts a sync callback for kernel semaphore waits. Host planning only needs the
    // same ring-id sequence, so use a no-op callback.
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id = seq.get_next_ring_id(noop_sync);
        uint32_t valid_kv_chunks = 0;
        for (uint32_t k_chunk = 0; k_chunk < num_local_k_chunks; ++k_chunk) {
            const uint32_t local_tile_start = k_chunk * k_chunk_tile_count;
            if (local_tile_start < kv_local_padded_Nt && ring_id * kv_local_padded_Nt + local_tile_start < logical_nt) {
                valid_kv_chunks++;
            }
        }
        if (valid_kv_chunks > 0) {
            masks.active_ring_iter_mask |= (1u << ring_iter);
        }
        if (valid_kv_chunks <= 1) {
            masks.single_valid_kv_chunk_mask |= (1u << ring_iter);
        }
    }

    return masks;
}

RingWritePlan build_ring_write_plan(
    const ttnn::prim::SparseRingSDPAParams& args,
    const ttnn::prim::SparseRingSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& coord) {
    RingWritePlan plan;
    const auto& ag = args.all_gather_operation_attributes;
    plan.transport_rank =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.input_q, coord, ag.cluster_axis);
    plan.forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, 1, ag.topology, ag.cluster_axis);
    plan.backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, -1, ag.topology, ag.cluster_axis);

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ag.ring_size, plan.transport_rank, ag.topology);
    (void)dynamic_alternate;
    if (ag.topology == ttnn::ccl::Topology::Ring && plan.transport_rank % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }

    if (ag.topology == ttnn::ccl::Topology::Linear) {
        plan.forward_writes_expected = num_targets_backward;
        plan.backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(ag.topology == ttnn::ccl::Topology::Ring, "Topology must be Linear or Ring");
        plan.forward_writes_expected = num_targets_forward;
        plan.backward_writes_expected = num_targets_backward;
    }

    return plan;
}

RingJointRuntimeArgLayout get_runtime_arg_layout(bool use_head_chain, bool k_uses_batch_chain, bool gqa_grouped_kv) {
    const uint32_t head_chain_args = use_head_chain ? kRingJointChainConfigArgCount : 0;
    const uint32_t batch_chain_args =
        k_uses_batch_chain ? (kRingJointChainConfigArgCount + kReaderBatchChainExtraArgCount) : 0;
    const uint32_t gqa_chain_args = gqa_grouped_kv ? (kRingJointChainConfigArgCount + kReaderGQAChainExtraArgCount) : 0;

    RingJointRuntimeArgLayout layout;
    layout.reader_kv_cache_batch_idx = kReaderBaseBufferArgCount + 2;
    layout.reader_logical_nt =
        kReaderBaseBufferArgCount + kReaderQWorkArgCount + head_chain_args + batch_chain_args + gqa_chain_args;
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

}  // namespace

namespace ttnn::prim {

namespace {

// Per-coord ProgramDescriptor build, looped over coords by create_workload_descriptor().
//
// The kernels keep the dense ring op's argument layout: positional compile-time and runtime args for features this
// op never uses (joint tokens, KV cache, causal, sliding window, latent V) are still emitted, as constants, so every
// index the kernels read stays where it was.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
tt::tt_metal::ProgramDescriptor build_sparse_ring_sdpa_program_descriptor(
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "build_sparse_ring_sdpa_program_descriptor requires mesh_dispatch_coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    /*
    Q, K, V are sequence-sharded across the ring (local_N = padded_N / ring_size rows per device).

        input_tensor_q: B x NH  x local_N x DH
        input_tensor_k: B x NHK x local_N x DH
        input_tensor_v: B x NHV x local_N x DH
        gathered_k / gathered_v: B x NHK / NHV x padded_N x DH   (fused all-gather)
        block_mask: 1 x NH x nb_pad x n_blocks_pad (uint32);  pooled_k / pooled_v: 1 x NH x n_blocks_pad x DH
        output_tensor: B x NH x local_N x DH

    Every core holds its 64-row Q blocks in L1 for the whole op (one accumulator set each). Each ring iteration,
    every K/V piece (k_chunk_size keys) that any Q block of the core's head group keeps is delivered once to the
    group and applied to every resident Q block that keeps one of its 64-token blocks. On the first active ring
    iteration the pooled entries (one per dropped block, weighted by its token count through a generated column
    bias) are applied too. After the last ring iteration each Q block is normalized and written out.
    */

    log_debug(tt::LogOp, "SparseRingSDPA create_descriptor");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;
    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    auto& output_tensor = output_tensors[SPARSE_RING_SDPA_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::ProgramDescriptor desc;

    auto* mesh_device = input_tensor_q.device();
    const RingWritePlan ring_write_plan = build_ring_write_plan(args, tensor_args, coord);
    const uint32_t transport_rank = ring_write_plan.transport_rank;
    const uint32_t forward_writes_expected = ring_write_plan.forward_writes_expected;
    const uint32_t backward_writes_expected = ring_write_plan.backward_writes_expected;
    const auto& forward_coord = ring_write_plan.forward_coord;
    const auto& backward_coord = ring_write_plan.backward_coord;

    log_debug(tt::LogOp, "transport rank: {}", transport_rank);

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
    log_debug(tt::LogOp, "v_shape (gathered): {}", gathered_input_tensor_v.logical_shape());

    const uint32_t B = q_shape[0];
    const uint32_t NH = q_shape[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = input_tensor_v.logical_shape()[1];
    const uint32_t DH = q_shape[3];
    const uint32_t vDH = input_tensor_v.logical_shape()[3];
    const uint32_t q_local_padded_N = q_shape[2];
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t gathered_padded_N = k_shape[2];

    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(/*v_shares_k_buffer=*/false, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    const bool use_head_chain =
        ring_joint::uses_v_head_chain(/*enable_kv_chains=*/true, gqa_grouped_kv, /*v_shares_k_buffer=*/false);
    // The store-and-forward chains are scheduled per head, not per (batch, head), so multi-batch requests read K/V
    // independently on each core.
    const bool build_kv_chains = B == 1;

    const uint32_t q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t gathered_padded_Nt = gathered_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const RingWorkMasks ring_work_masks = build_ring_work_masks(args, tensor_args, ring_write_plan);
    const RingJointRuntimeArgLayout runtime_arg_layout =
        get_runtime_arg_layout(use_head_chain, k_uses_batch_chain, gqa_grouped_kv);
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    // Chunked-prefill Q mapping words of the shared runtime-arg layout; unused here.
    constexpr uint32_t kUnusedQMappingWord = 0;

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Block geometry. Blocks are 64 tokens (2 tiles) inside each shard; the pooled tensors hold ring_size x
    // sol_nb_pad blocks; a pooled "chunk" is Sk_chunk_t pooled rows.
    constexpr uint32_t kSolBlockTiles = 2;
    const uint32_t sol_nkc = tt::div_up(kv_local_padded_Nt, Sk_chunk_t);
    const uint32_t sol_n_blocks_pad = tensor_args.pooled_k.logical_shape()[2];
    const uint32_t sol_nb_pad = sol_n_blocks_pad / ring_size;
    const uint32_t sol_nb = tt::div_up(kv_local_padded_Nt, kSolBlockTiles);
    const uint32_t sol_nqc_pad = tensor_args.block_mask.logical_shape()[2];
    const uint32_t sol_pooled_tiles = sol_n_blocks_pad / tt::constants::TILE_HEIGHT;
    const uint32_t sol_pooled_chunks = tt::div_up(sol_pooled_tiles, Sk_chunk_t);
    const uint32_t sol_pooled_last_tiles = sol_pooled_tiles - (sol_pooled_chunks - 1) * Sk_chunk_t;
    // The reader reads whole mask rows.
    const uint32_t sol_mask_row_bytes = tensor_args.block_mask.logical_shape()[3] * sizeof(uint32_t);
    TT_FATAL(sol_nkc <= 64, "SparseRingSDPA: at most 64 K pieces per shard");
    TT_FATAL(sol_nb <= sol_nb_pad, "SparseRingSDPA: {} blocks per shard exceed the pooled row {}", sol_nb, sol_nb_pad);
    TT_FATAL(Sk_chunk_t % kSolBlockTiles == 0, "SparseRingSDPA: k chunk must hold whole blocks");

    // Global chunk width the kernels divide by the ring size (one Q slab per device).
    const uint32_t q_chunk_group_tile_count = q_local_padded_Nt * ring_size;

    const uint32_t compile_time_logical_n = static_cast<uint32_t>(args.logical_n);
    const uint32_t compile_time_global_n_partial_col = compile_time_logical_n % tt::constants::TILE_HEIGHT;

    const uint32_t num_local_q_chunks = tt::div_up(q_local_padded_N, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);

    log_debug(tt::LogOp, "B: {}, NH: {}, NHK: {}, NHV: {}, DH: {}, vDH: {}", B, NH, NHK, NHV, DH, vDH);
    log_debug(tt::LogOp, "q_local_padded_N: {}, kv_local_padded_N: {}", q_local_padded_N, kv_local_padded_N);
    log_debug(
        tt::LogOp,
        "Sq_chunk_t: {}, Sk_chunk_t: {}, num_local_q_chunks: {}, num_local_k_chunks: {}",
        Sq_chunk_t,
        Sk_chunk_t,
        num_local_q_chunks,
        num_local_k_chunks);

    MeshDevice* device = input_tensor_q.device();

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

    // Init fused op signaler — descriptor-pattern equivalent of RingSDPAFusedOpSignaler::init_fused_op. The signaler
    // stores the receiver-cores noc list and two signal semaphore IDs (one for forward, one for backward).
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

    // Single host-derived split-forwarding decision, shared with the all-gather (passed to the helper below), so
    // producer and consumer cannot disagree.
    sdpa_fused_op_signaler->split_forwarding_enabled =
        (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring) &&
        (args.all_gather_operation_attributes.ring_size % 2 == 0) &&
        (args.all_gather_operation_attributes.ring_size > 2);

    log_debug(tt::LogOp, "num_cores: {}, grid_size: {}", num_cores, grid_size);

    TT_FATAL(
        num_cores <= mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y);

    // The (batch, head, Q block) units are divided evenly across the cores, in flat order.
    const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
    const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    log_debug(tt::LogOp, "max_q_per_core: {}", max_q_per_core);

    // Every Q block the core owns stays fronted in cb_q_in for the whole op (read once).
    TT_FATAL(
        max_q_per_core <= kSolResidentMaxSlots,
        "SparseRingSDPA holds at most {} Q blocks per core, this split needs {}",
        kSolResidentMaxSlots,
        max_q_per_core);
    // The plan page's 16-word header carries the shard's kept-block bits (one word per 32 blocks) plus the union
    // piece bits (2 words) and the pooled flag.
    TT_FATAL(
        tt::div_up(sol_nb, 32u) + 3 <= 16,
        "SparseRingSDPA: {} blocks per shard exceed the plan header (at most 416)",
        sol_nb);
    uint32_t q_tiles = Sq_chunk_t * DHt * max_q_per_core;
    // Group delivery: one producer per head group multicasts each piece (see sol_res_mcast in the reader) over a
    // 3-deep K/V CB, so the producer can send the next piece while every member computes on the previous ones.
    // Without it (B > 1, or no head chain) the K/V CBs are double-buffered.
    const bool sol_res_mcast = build_kv_chains && use_head_chain;
    const uint32_t kv_cb_pages = sol_res_mcast ? 3u : 2u;
    uint32_t k_tiles = Sk_chunk_t * DHt * kv_cb_pages;
    uint32_t v_tiles = Sk_chunk_t * vDHt * kv_cb_pages;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    log_debug(
        tt::LogOp,
        "q_tiles: {}, k_tiles: {}, v_tiles: {}, qk_tiles: {}, out0_t: {}, statistics_tiles: {}",
        q_tiles,
        k_tiles,
        v_tiles,
        qk_tiles,
        out0_t,
        statistics_tiles);

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

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    // Validated: fp32_dest_acc_en is off, so compute takes the streaming path.
    const bool use_streaming_compute = !fp32_dest_acc_en;
    TT_FATAL(use_streaming_compute, "SparseRingSDPA requires streaming compute (fp32_dest_acc_en=false)");

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(
        Sq_chunk_t,
        vDHt,
        dst_size,
        /*max_subblock_h=*/2,
        /*max_subblock_w=*/UINT32_MAX);
    // Streaming compute may widen the QKT@V row group beyond the host matmul subblock height for odd Q chunks. The
    // writer must drain cb_out with the same row-group cadence that compute pushes.
    const uint32_t writer_out_row_group_h =
        ttnn::transformer::sdpa::streaming_qktv_h(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;

    // Shrink cb_out to a 2-slot ping-pong (see sdpa_subblock_utils.hpp) when the save-to-staging branch cannot fire
    // (ring_size == 1 or one Q block per core) and the full output fits in 2 row groups.
    const bool streaming_shrink_fits = Sq_chunk_t <= 2 * writer_out_row_group_h;
    const bool streaming_shrink_safe = (ring_size == 1 || max_q_per_core == 1) && streaming_shrink_fits;
    if (streaming_shrink_safe) {
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming cb_out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }
    log_debug(tt::LogOp, "streaming_shrink_safe={} out0_t={}", streaming_shrink_safe, out0_t);
    log_debug(
        tt::LogOp,
        "qk subblock {}x{} ({}x{} subblocks), out subblock {}x{} ({}x{} subblocks)",
        qk_out_subblock_h,
        qk_out_subblock_w,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_out_subblock_h,
        out_out_subblock_w,
        out_in0_num_subblocks,
        out_in1_num_subblocks);

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    // DHT_GRANULARITY is used in the kernel with both DHt and vDHt as the cols parameter,
    // so the granularity must evenly divide both to avoid dropping tiles.
    uint32_t dht_granularity = std::min({DHt, vDHt, dst_size});
    while (dht_granularity > 1 && (DHt % dht_granularity != 0 || vDHt % dht_granularity != 0)) {
        dht_granularity--;
    }
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    log_debug(
        tt::LogOp,
        "granularity: stats {}, sub_exp {}, dht {}, reduce {}",
        stats_granularity,
        sub_exp_granularity,
        dht_granularity,
        reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const float scale_value = scale.value_or(1.0f);
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);

    log_debug(tt::LogOp, "scale: {}", scale_value);

    const uint32_t active_ring_iter_mask = ring_work_masks.active_ring_iter_mask;
    const uint32_t single_valid_kv_chunk_mask = ring_work_masks.single_valid_kv_chunk_mask;
    const auto& ag_attributes = args.all_gather_operation_attributes;
    const RingAttentionRankMapping rank_mapping{
        .full_mesh = ag_attributes.full_mesh,
        .orientation = ag_attributes.snake_orientation,
        .mesh_rows = ag_attributes.mesh_rows,
        .mesh_cols = ag_attributes.mesh_cols};

    const uint32_t q_heads_per_kv = NH / NHK;

    // Cores actually issuing Q reads. When the flat Q block distribution is smaller than the grid the trailing cores
    // get zero work.
    const uint32_t num_active_cores = std::min(num_cores, all_heads_num_q_chunks);

    // Positional reader args. Slots for features this op does not use are constants (see the function comment).
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
        0,  // Lt (no joint tokens)
        0,  // L
        num_local_q_chunks,
        0,  // num_joint_q_chunks
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_q_chunks,
        ring_size,
        qk_out_subblock_h,
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        0,  // chunked prefill
        num_active_cores,
        q_chunk_group_tile_count,
        0,  // indexed KV cache
        0,  // KV-pad rotation
        NHV,
        0,  // v_shares_k_buffer
        0,  // attention sink
        0,  // sliding_window_size
        gathered_padded_Nt,
        0,  // slot from metadata
        0,  // KV pad from metadata
        0,  // Lt_local
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // circular KV slab count
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    // Accessors: Q, K, V, gathered K, gathered V, attention sink (null), then block mask, pooled K, pooled V.
    tt::tt_metal::Buffer* const attention_sink_buf = nullptr;
    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(attention_sink_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.block_mask.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.pooled_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.pooled_v.buffer()).append_to(reader_compile_time_args);

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
                "ChainSemaphores expected to append {} compile-time args, appended {}",
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
    if (k_uses_batch_chain) {
        batch_sems = ChainSemaphores::create(desc, core_grid_set);  // shared-K chain
    }
    if (gqa_grouped_kv) {
        gqa_sems = ChainSemaphores::create(desc, core_grid_set);  // grouped K/V chain
    }

    // Append semaphore ids to reader compile-time args (must match reader kernel expectations)
    const auto sem_args_offset = reader_compile_time_args.size();
    if (use_head_chain) {
        head_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // head_mcast_enabled: never (group delivery replaces it)
    }
    if (k_uses_batch_chain) {
        batch_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // shared-K mcast placeholder (patched after chain construction)
    }
    if (gqa_grouped_kv) {
        gqa_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // GQA mcast placeholder (patched after chain construction)
    }

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        compile_time_logical_n,
        0,  // Lt_local
        0,  // L
        num_local_q_chunks,
        0,  // num_joint_q_chunks
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_q_chunks,
        packed_identity_scalar,
        scale_packed,
        ring_size,
        compile_time_global_n_partial_col,
        0,  // joint_l_partial_col
        static_cast<std::uint32_t>(use_streaming_compute),
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        static_cast<std::uint32_t>(writer_out_row_group_h),
        0,  // chunked prefill
        q_chunk_group_tile_count,
        0,  // sliding_window_size
        0,  // KV pad from metadata
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        NH,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        compile_time_logical_n,
        logical_nt,
        0,  // Lt
        0,  // L
        num_local_q_chunks,
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_q_chunks,
        ring_size,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        scale_packed,
        static_cast<std::uint32_t>(use_streaming_compute),
        compile_time_global_n_partial_col,
        0,  // joint_l_partial_col
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        0,  // chunked prefill
        q_chunk_group_tile_count,
        0,  // KV-pad rotation
        0,  // v_shares_k_buffer
        0,  // attention sink
        0,  // sliding_window_size
        0,  // KV pad from metadata
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // circular KV slab count
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    std::map<std::string, std::string> defines;
    defines["GROUPED_KV_SOURCE_COUNT"] = "1";
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    if (args.program_config.has_value() && args.program_config->matmul_math_fidelity.has_value()) {
        defines["SDPA_MATMUL_FIDELITY"] =
            std::to_string(static_cast<uint32_t>(*args.program_config->matmul_math_fidelity));
    }
    // MATH_FIDELITY is not defined on the unpack TRISC, and all three must agree on how P.V is set up.
    if (math_fidelity == MathFidelity::LoFi) {
        defines["SDPA_COMPUTE_LOFI"] = "1";
    }

    // NOTE: KernelDescriptors are built after chain construction.

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());

    // The pooled-column bias tiles are Float16_b: the step adds them into the scores by L1 accumulation.
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    // Intermediates stay in bf16 to save L1 (fp32 dest accumulation is off).
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;
    tt::DataFormat qk_im_df = tt::DataFormat::Float16_b;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);
    uint32_t qk_im_tile_size = tt::tile_size(qk_im_df);

    log_debug(tt::LogOp, "data formats: q {}, k {}, v {}, out {}, scalar {}", q_df, k_df, v_df, out_df, scalar_df);

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
    const uint32_t cb_v_in = allocate_tile_cb(v_tiles, v_tile_size, v_df);

    // The dense op's padding-mask and scale CBs: the resident step masks padding by narrowing its K width and applies
    // the scale inside its exp, so neither is allocated. Their slots in the shared CB argument block stay.
    constexpr uint32_t inactive_cb = std::numeric_limits<uint32_t>::max();
    const uint32_t cb_mask_in = inactive_cb;

    // No attention sink: alias a valid CB so compile-time tile-size queries remain valid.
    const uint32_t cb_attention_sink = cb_q_in;

    const uint32_t cb_scale_in = inactive_cb;
    const uint32_t cb_identity_scale_in = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);
    const uint32_t cb_col_identity = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);

    // One step covers one piece for a group of resident Q blocks, each with its own kept blocks; member g's scores are
    // stored compactly (row stride = its kept width), and members are grouped until the scores fill
    // kSolResidentGroup x Sq_chunk_t rows of Sk_chunk_t tiles (at least qk_tiles). The new-max / alpha scratches
    // hold one tile row per resident Q block. The compute kernel derives the same capacity (sdpa_ring_resident,
    // kGroupCap) and ASSERTs the page count.
    constexpr uint32_t kSolResidentGroup = 4;
    const uint32_t qk_im_cb_tiles =
        std::max(qk_tiles, std::min(max_q_per_core, kSolResidentGroup) * Sq_chunk_t * Sk_chunk_t);
    const uint32_t res_stats_tiles = statistics_tiles * max_q_per_core;
    const uint32_t cb_qk_im = allocate_tile_cb(qk_im_cb_tiles, qk_im_tile_size, qk_im_df);
    const uint32_t cb_max_A = allocate_tile_cb(res_stats_tiles, stats_tile_size, stats_df);
    const uint32_t cb_exp_max_diff = allocate_tile_cb(res_stats_tiles, stats_tile_size, stats_df);

    const uint32_t cb_out = allocate_tile_cb(out0_t, out_tile_size, out_df);

    // The staging CBs (max, out, sum) hold one accumulator set per resident Q block, addressed by block slot and
    // updated in place: exactly max_q_per_core entries, filled once on the first active ring iteration and neither
    // pushed nor popped until the final normalization, so the unpacker's read index and the packer's write index
    // name the same tile (the buffer is exactly full, the write pointer wrapped to its base).
    const uint32_t fifo_entries = max_q_per_core;
    const uint32_t cb_stats_in = allocate_tile_cb(statistics_tiles * fifo_entries, im_tile_size, im_df);
    const uint32_t cb_prev_out = allocate_tile_cb(out_im_tiles * fifo_entries, out_tile_size, out_df);

    // 1-tile recip scratch CB for normalize_row_streaming.
    const uint32_t cb_recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);

    const uint32_t cb_sum_in = allocate_tile_cb(statistics_tiles * fifo_entries, stats_tile_size, stats_df);

    // Slots of the dense op's CB argument block no kernel of this op reads: the second accumulator set, the
    // save / restore and signal CBs, and the metadata mailbox. Not allocated; the slots stay so the indices hold.
    const uint32_t cb_out_im_A = inactive_cb;
    const uint32_t cb_out_im_B = inactive_cb;
    const uint32_t cb_max_B = inactive_cb;
    const uint32_t cb_sum_A = inactive_cb;
    const uint32_t cb_sum_B = inactive_cb;
    const uint32_t cb_stats_out = inactive_cb;
    const uint32_t cb_sum_out = inactive_cb;
    const uint32_t cb_signal = inactive_cb;
    const uint32_t cb_kv_pad_derived = inactive_cb;
    // Reader->compute plan words per ring iteration, and the pooled-chunk column bias tiles.
    // Plan page: 16 summary words, then the Q block's full mask row (scratch the reader scans in place). The summary
    // is [kept-block words (sol_res_block_words)][union piece bits lo, hi][do_pooled].
    // Single page: a second page pushed the static CB region past the L1 buffers the full pipeline's CCL manager
    // pins (clash by 1792 B at the 10 s shape).
    const uint32_t cb_sol_plan = allocate_cb(64 + sol_mask_row_bytes, 1, tt::DataFormat::UInt32);
    // Every Q block's plan record goes in one page per ring iteration, after the header (16 words) and the shard
    // slice the reader scans (sol_nb_pad words), so the reader does not wait for compute to pop a page per block at
    // each iteration boundary. The records fit in the scratch the whole-row reads already need; a short row (one
    // shard: the slice is the whole row) leaves no room for them, and then each block gets its own page. No
    // phase-alignment dummies: every chain / group member pushes the same sequence, so the pointers stay in step.
    const uint32_t sol_res_rec_words = tt::div_up(sol_nb, 32u) + 3;
    const bool sol_res_plan_fits =
        (16 + sol_nb_pad + max_q_per_core * sol_res_rec_words) * sizeof(uint32_t) <= 64 + sol_mask_row_bytes;
    const uint32_t sol_res_plan_rec_base = sol_res_plan_fits ? 16 + sol_nb_pad : 0;
    // One pooled chunk of bias tiles in flight; 2*Sk_chunk_t pushed the static CB region past the L1 buffers at the
    // 10 s / 15 s shapes.
    const uint32_t cb_sol_bias = allocate_tile_cb(Sk_chunk_t, mask_tile_size, mask_df);
    {
        uint64_t cb_bytes = 0;
        for (const auto& cb : desc.cbs) {
            cb_bytes += cb.total_size;
        }
        log_debug(
            tt::LogOp,
            "SparseRingSDPA CBs: {} buffers, {} bytes (q {} t, k {} t, v {} t, qk {} t, out_im {} t, stats {} t; "
            "im tile {} B)",
            desc.cbs.size(),
            cb_bytes,
            q_tiles,
            k_tiles,
            v_tiles,
            qk_tiles,
            out_im_tiles,
            statistics_tiles,
            im_tile_size);
    }

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
    auto* const out_buf = output_tensor.buffer();

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
                "ChainConfig expected to append {} runtime args, appended {}",
                kRingJointChainConfigArgCount,
                args.size() - start_size);
        }
    };

    std::vector<CoreWork> core_work(num_cores);
    std::vector<ChainConfig> head_chain_configs(use_head_chain ? num_cores : 0);       // MHA K/V or separate-V V
    std::vector<ChainConfig> batch_chain_configs(k_uses_batch_chain ? num_cores : 0);  // Shared K for separate-V
    std::vector<ChainConfig> gqa_chain_configs(gqa_grouped_kv ? num_cores : 0);        // Grouped K/V for GQA
    std::vector<std::vector<HeadSegmentRef>> head_segments(use_head_chain ? NH : 0);

    // Evenly distribute flat global Q blocks across cores
    const uint32_t total_q_chunks = B * NH * num_q_chunks;
    const uint32_t cores_doing_extra_work = total_q_chunks % num_cores;
    const uint32_t base_chunks_per_core = total_q_chunks / num_cores;

    uint32_t next_global_chunk = 0;

    auto decode_flat_chunk = [&](uint32_t flat_chunk_index) {
        const uint32_t head_span = num_q_chunks;
        const uint32_t head_index = head_span == 0 ? 0 : (flat_chunk_index / head_span);
        const uint32_t q_chunk = head_span == 0 ? 0 : (flat_chunk_index % head_span);
        const uint32_t head = (NH == 0) ? 0 : (head_index % NH);
        return std::pair<uint32_t, uint32_t>{head, q_chunk};
    };

    // Describe a core's flat Q range for K/V chain construction.
    auto append_head_work = [&](uint32_t core_idx, uint32_t flat_chunk, uint32_t remaining) {
        auto& work = core_work.at(core_idx);
        while (remaining > 0) {
            const auto [head_idx, q_chunk_idx] = decode_flat_chunk(flat_chunk);
            const uint32_t take = std::min(remaining, num_q_chunks - q_chunk_idx);
            work.head_work.push_back(CoreHeadWork{.head = head_idx, .q_chunk_count = take});
            if (use_head_chain) {
                TT_FATAL(
                    head_idx < head_segments.size(),
                    "Head-chain segment index {} is outside {} configured query heads",
                    head_idx,
                    head_segments.size());
                head_segments[head_idx].push_back(HeadSegmentRef{
                    .core_idx = core_idx, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
            }
            remaining -= take;
            flat_chunk += take;
        }
    };

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t chunk_count = base_chunks_per_core + ((i < cores_doing_extra_work) ? 1 : 0);
        if (next_global_chunk >= total_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > total_q_chunks - next_global_chunk) {
            chunk_count = total_q_chunks - next_global_chunk;
        }

        auto& work = core_work.at(i);
        work.physical_core = device->worker_core_from_logical_core(core);
        work.global_q_start = next_global_chunk;
        work.global_q_count = chunk_count;

        append_head_work(i, next_global_chunk, chunk_count);

        next_global_chunk += chunk_count;
    }

    // Helper: build a linear chain from sorted (core_idx, q_chunk_count) pairs.
    // - chain_segs[i].second = q iterations the i-th core will process in this chain scope
    // - injector = first core with head_work.size() == 1 (single head segment = no straddling)
    // - no wrap-around: wrapping would inflate q_iter_local and cause deadlock
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
            "Row mcast failed to find a max-work injector for row {}",
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

        // With one local K/V head per chip every active Q-head core consumes the same K and V chunks, so use
        // row-wide multicast instead of the store-and-forward grouped chain. Idle cores in an active row participate
        // in padded iterations.
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
            }

            gqa_mcast_enabled = gqa_mcast_rows > 0;
            if (!gqa_mcast_enabled) {
                gqa_mcast_fallback_reason = "no active groups";
            }
        }
    }

    // Build the shared-K chain for separate-V cases.
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

    if (!k_uses_batch_chain) {
        // No shared-K chain to multicast.
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
            }
        }
    }

    // The kernels read rotated_max_slots (the dense op's rotated Q split) as their LAST compile-time arg. Resident Q
    // blocks never change owner, so it is 0.
    constexpr uint32_t rotated_max_slots_ct = 0;
    for (auto* ct_args : {&reader_compile_time_args, &writer_compile_time_args, &compute_compile_time_args}) {
        ct_args->push_back(rotated_max_slots_ct);
    }
    const std::array<size_t, 3> ct_arg_sizes_with_rotated_last = {
        reader_compile_time_args.size(), writer_compile_time_args.size(), compute_compile_time_args.size()};

    // Build static MHA/shared-K V head chains. GQA uses KV-head-grouped chains instead.
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
            // A core whose resident Q blocks span two heads streams the second head's pieces into the same K/V CBs
            // between this head's pieces, so its CB write pointer drifts from the chain's. The chain forward writes to
            // the sender's own CB address on the next core (ChainLink::forward), which needs every member's pointer in
            // lockstep; a drifted sink receives the forward over the entry its compute is reading. Such cores read
            // their own K/V for both heads instead: drop them from the chain's tail (the injector rule already skips
            // them at the head).
            while (!chain_segs.empty() && core_work[chain_segs.back().first].head_work.size() > 1) {
                chain_segs.pop_back();
            }
            build_linear_chain(chain_segs, head_id, head_chain_configs, core_work);
        }
    }

    // The head chains are never multicast (group delivery below replaces that). Patch the shared-K and GQA
    // multicast flags.
    {
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
            gqa_mcast_enabled ? "mcast" : fmt::format("unicast, {}", gqa_mcast_fallback_reason),
            gqa_grouped_chains,
            gqa_grouped_participant_cores,
            gqa_local_fallback_cores);
    } else if (k_uses_batch_chain) {
        log_debug(
            tt::LogOp,
            "K chain mode: batch ({})",
            k_mcast_enabled ? "mcast" : fmt::format("unicast, {}", k_mcast_fallback_reason));
    } else {
        log_debug(tt::LogOp, "K chain mode: head");
    }

    // Convert std::map<string,string> defines to KernelDescriptor::Defines vector form.
    KernelDescriptor::Defines kernel_defines(defines.begin(), defines.end());

    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_ring_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = core_grid_set;
    // The kernels read rotated_max_slots as their final compile-time arg, so nothing may be
    // appended after it (in-place patching of earlier indices is fine and does not change size).
    TT_FATAL(
        reader_compile_time_args.size() == ct_arg_sizes_with_rotated_last[0] &&
            writer_compile_time_args.size() == ct_arg_sizes_with_rotated_last[1] &&
            compute_compile_time_args.size() == ct_arg_sizes_with_rotated_last[2],
        "rotated_max_slots must stay the last compile-time arg; args were appended after it "
        "(reader {}->{}, writer {}->{}, compute {}->{})",
        ct_arg_sizes_with_rotated_last[0],
        reader_compile_time_args.size(),
        ct_arg_sizes_with_rotated_last[1],
        writer_compile_time_args.size(),
        ct_arg_sizes_with_rotated_last[2],
        compute_compile_time_args.size());

    reader_kernel.compile_time_args = reader_compile_time_args;
    reader_kernel.defines = kernel_defines;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_ring_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = core_grid_set;
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.defines = kernel_defines;
    writer_kernel.config = WriterConfigDescriptor{};

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_ring_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = core_grid_set;
    compute_kernel.compile_time_args = compute_compile_time_args;
    // Named compile-time args shared by the three kernels.
    const KernelDescriptor::NamedCompileTimeArgs named_args = {
        // The kernels share the dense op's chain and ring helpers, which take a KV split factor; no split here.
        {"kv_stripe_split", 1u},
        // Resident Q blocks: the slot count every core sizes its state arrays by, and the words of per-shard
        // kept-block bits the reader puts in every plan page (block b: word b/32, bit b%32).
        {"sol_resident_slots", max_q_per_core},
        {"sol_res_block_words", tt::div_up(sol_nb, 32u)},
        // Group delivery by multicast (see sol_res_mcast above).
        {"sol_res_mcast", sol_res_mcast ? 1u : 0u},
        // The one-page plan's record base word (0 = a page per block); see sol_res_plan_rec_base above.
        {"sol_res_plan_rec_base", sol_res_plan_rec_base},
        {"sol_nb", sol_nb},
        {"sol_nb_pad", sol_nb_pad},
        {"sol_n_blocks_pad", sol_n_blocks_pad},
        {"sol_nqc_pad", sol_nqc_pad},
        {"sol_pooled_chunks", sol_pooled_chunks},
        {"sol_pooled_last_tiles", sol_pooled_last_tiles},
        {"sol_mask_row_bytes", sol_mask_row_bytes},
        // Logical length in tokens, for the on-device pooled-column bias.
        {"sol_logical_n", static_cast<uint32_t>(args.logical_n)},
        {"cb_sol_plan", cb_sol_plan},
        {"cb_sol_bias", cb_sol_bias},
    };
    for (auto* kernel : {&reader_kernel, &writer_kernel, &compute_kernel}) {
        kernel->named_compile_time_args = named_args;
    }
    reader_kernel.named_compile_time_args.emplace_back(
        "k_row_Wt", input_tensor_k.padded_shape()[3] / tt::constants::TILE_WIDTH);
    reader_kernel.named_compile_time_args.emplace_back(
        "v_row_Wt", input_tensor_v.padded_shape()[3] / tt::constants::TILE_WIDTH);
    compute_kernel.defines = kernel_defines;
    // Per-phase device-profiler zones inside the step. They only emit under TT_METAL_DEVICE_PROFILER=1
    // (PROFILE_KERNEL); this define flips the template gate that is otherwise false. Measurement knob, not hashed:
    // set it for the whole process. =1: one zone per step, so the per-core profiler buffer holds ~10 steps and the
    // gaps between them are measurable. =2: plus the two phase zones per Q subblock / row group. =3: plus the
    // per-key-subblock zones (~6 KB of TRISC code).
    if (std::getenv("TT_SOL_STEP_PROFILE") && std::string(std::getenv("TT_SOL_STEP_PROFILE")) != "0" &&
        std::string(std::getenv("TT_SOL_STEP_PROFILE")) != "") {
        const std::string level = std::getenv("TT_SOL_STEP_PROFILE");
        compute_kernel.defines.emplace_back("SOL_STEP_PROFILE", "1");
        // The reader's and writer's wait zones (SOL_ZONE in sparse_ring_reader.cpp / sparse_ring_writer.cpp).
        reader_kernel.defines.emplace_back("SOL_STEP_PROFILE", "1");
        writer_kernel.defines.emplace_back("SOL_STEP_PROFILE", "1");
        if (level == "2" || level == "3") {
            compute_kernel.defines.emplace_back("SOL_STEP_PROFILE_PHASES", "1");
        }
        if (level == "3") {
            compute_kernel.defines.emplace_back("SOL_STEP_PROFILE_FINE", "1");
        }
    }
    // The compute kernel builds at O2: at the default level the 5 s program sits at the 70,656 B kernel config buffer,
    // and O2 cuts the three TRISC binaries by ~30 %. Profiled runs at O2 showed the same per-piece compute as the
    // default-level timings: the step is bound by Tensix instruction throughput, not by RISC issue. The kernel's build
    // hash carries the level (ComputeKernel::config_hash), so the JIT cache cannot serve the other binary for it.
    compute_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O2;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // On-device bias generator table (see the reader arg layout below). bf16 round-to-nearest-even of ln(len) / scale;
    // -30000 for len 0.
    constexpr uint32_t kSolLenBiasTableWords = 33;  // 66 bf16 entries, len 0..65 (65 unused)
    std::array<uint32_t, kSolLenBiasTableWords> sol_len_bias_table{};
    {
        const auto f2bf = [](float f) -> uint32_t {
            uint32_t u = std::bit_cast<uint32_t>(f);
            const uint32_t lsb = (u >> 16) & 1u;
            u += 0x7FFFu + lsb;
            return u >> 16;
        };
        constexpr float kSolBiasNeg = -30000.0f;  // sol_prepass.NEG
        for (uint32_t len = 0; len <= 65; ++len) {
            const float v = (len == 0 || len > 64) ? kSolBiasNeg : std::log(static_cast<float>(len)) / scale_value;
            sol_len_bias_table[len / 2] |= f2bf(v) << (16 * (len % 2));
        }
    }

    // Group delivery: per core [role, producer x, producer y, receivers, rectangles, (x0, y0, x1, y1, cores)...].
    // A group is a head chain's members (chains hold single-head cores only); the producer reads every piece from
    // DRAM. The receivers are covered by rectangles of consecutive NoC x on one row each, so a multicast never lands
    // on a core outside the group. The producer is the first member in core order (not the chain's injector): it
    // starts its row, so it does not split a rectangle, and a group of consecutive cores spanning two rows needs two
    // rectangles.
    constexpr uint32_t kSolGroupMaxRects = 8;
    std::vector<std::vector<uint32_t>> sol_group_args(num_cores, std::vector<uint32_t>{0u, 0u, 0u, 0u, 0u});
    if (sol_res_mcast) {
        for (uint32_t h = 0; h < static_cast<uint32_t>(head_segments.size()); ++h) {
            std::vector<uint32_t> members;
            for (const auto& seg : head_segments[h]) {
                const auto& cfg = head_chain_configs.at(seg.core_idx);
                if (!cfg.participates || cfg.head != h ||
                    std::find(members.begin(), members.end(), seg.core_idx) != members.end()) {
                    continue;
                }
                TT_FATAL(
                    core_work[seg.core_idx].head_work.size() == 1,
                    "SparseRingSDPA group delivery: core {} of head {} spans two heads",
                    seg.core_idx,
                    h);
                members.push_back(seg.core_idx);
            }
            if (members.size() < 2) {
                continue;
            }
            const uint32_t producer = *std::min_element(members.begin(), members.end());
            const CoreCoord prod_phys = core_work[producer].physical_core;
            std::map<uint32_t, std::vector<uint32_t>> receiver_xs_by_row;
            for (const uint32_t ci : members) {
                if (ci != producer) {
                    receiver_xs_by_row[core_work[ci].physical_core.y].push_back(core_work[ci].physical_core.x);
                }
            }
            std::vector<std::array<uint32_t, 5>> rects;
            for (auto& [y, xs] : receiver_xs_by_row) {
                std::sort(xs.begin(), xs.end());
                size_t run_start = 0;
                for (size_t k = 1; k <= xs.size(); ++k) {
                    if (k == xs.size() || xs[k] != xs[k - 1] + 1) {
                        rects.push_back({xs[run_start], y, xs[k - 1], y, static_cast<uint32_t>(k - run_start)});
                        run_start = k;
                    }
                }
            }
            TT_FATAL(
                rects.size() <= kSolGroupMaxRects,
                "SparseRingSDPA group delivery: head {} needs {} multicast rectangles (at most {})",
                h,
                rects.size(),
                kSolGroupMaxRects);
            const uint32_t n_receivers = static_cast<uint32_t>(members.size() - 1);
            for (const uint32_t ci : members) {
                auto& a = sol_group_args[ci];
                a = {
                    ci == producer ? 1u : 2u,
                    static_cast<uint32_t>(prod_phys.x),
                    static_cast<uint32_t>(prod_phys.y),
                    n_receivers,
                    0u};
                if (ci == producer) {
                    a[4] = static_cast<uint32_t>(rects.size());
                    for (const auto& r : rects) {
                        a.insert(a.end(), r.begin(), r.end());
                    }
                }
            }
            log_debug(
                tt::LogOp,
                "SparseRingSDPA group: head {} producer core {} -> {} receivers in {} rectangles",
                h,
                producer,
                n_receivers,
                rects.size());
        }
    }

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        const auto& work = core_work.at(i);
        const uint32_t global_q_start = work.global_q_start;
        const uint32_t global_q_end = global_q_start + work.global_q_count;

        log_debug(tt::LogOp, "core {} ({},{}): q [{}, {})", i, core.x, core.y, global_q_start, global_q_end);

        CheckedRuntimeArgList reader_args;
        reader_args.push_back(q_buf);
        reader_args.push_back(k_buf);
        reader_args.push_back(v_buf);
        reader_args.push_back(gathered_k_buf);
        reader_args.push_back(gathered_v_buf);
        reader_args.push_back(attention_sink_buf);
        reader_args.push_back(global_q_start);
        reader_args.push_back(global_q_end);
        reader_args.push_checked(runtime_arg_layout.reader_kv_cache_batch_idx, 0u, "reader.kv_cache_batch_idx");
        if (use_head_chain) {
            // Head chain: MHA uses it for K/V; separate-V shared-K uses it for V only.
            std::vector<uint32_t> head_chain_args;
            head_chain_configs.at(i).append_to_args(head_chain_args);
            reader_args.append(head_chain_args);
        }
        if (k_uses_batch_chain) {
            std::vector<uint32_t> batch_chain_args;
            batch_chain_configs.at(i).append_to_args(batch_chain_args);
            reader_args.append(batch_chain_args);
            reader_args.push_back(k_chain_max_q[i]);
        }
        if (gqa_grouped_kv) {
            std::vector<uint32_t> gqa_chain_args;
            gqa_chain_configs.at(i).append_to_args(gqa_chain_args);
            reader_args.append(gqa_chain_args);
            reader_args.push_back(gqa_chain_max_q[i]);
        }

        reader_args.push_checked(runtime_arg_layout.reader_logical_nt, logical_nt, "reader.logical_nt");
        reader_args.push_checked(
            runtime_arg_layout.reader_active_ring_iter_mask, active_ring_iter_mask, "reader.active_ring_iter_mask");
        reader_args.push_back(kUnusedQMappingWord);  // q_pre_wrap_start_tile
        reader_args.push_back(kUnusedQMappingWord);  // q_pre_wrap_tile_count
        reader_args.push_back(kUnusedQMappingWord);  // q_post_wrap_start_tile
        reader_args.push_back(kUnusedQMappingWord);  // q_valid_tile_count

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        std::vector<uint32_t> reader_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_signaler_args);
        reader_args.append(reader_signaler_args);

        // Block-sparse args, read by the kernel at rotated_args_base. Word offsets from there:
        //   +0 mask, +1 pooled K, +2 pooled V, +3 bias (0: generated), +4 n_members, +5 my_start, then the
        //   kSolLenBiasTableWords packed bf16 table, then members (start, count)..., then the group delivery words.
        // Buffer entries are registered as bindings and re-patched on cache hits by the descriptor adapter.
        reader_args.push_back(tensor_args.block_mask.buffer());
        reader_args.push_back(tensor_args.pooled_k.buffer());
        reader_args.push_back(tensor_args.pooled_v.buffer());
        reader_args.push_back(0u);
        // Linear head chains: the chain's members as (Q block start within the head, count) plus this core's own
        // start. At stream position j every member is on Q block start_m + j, so each core derives the same
        // chain-wide kept set and the chain skips a piece nobody needs in lockstep. Non-participants (head-boundary
        // cores before the injector) get no members and skip on their own mask.
        std::vector<std::pair<uint32_t, uint32_t>> sol_members;
        uint32_t sol_my_start = 0;
        if (use_head_chain && head_chain_configs[i].participates) {
            const uint32_t h = head_chain_configs[i].head;
            for (const auto& seg : head_segments[h]) {
                const auto& cfg = head_chain_configs[seg.core_idx];
                if (!cfg.participates || cfg.head != h) {
                    continue;
                }
                const auto& w = core_work[seg.core_idx];
                uint32_t flat = w.global_q_start;
                for (uint32_t s = 0; s < seg.head_work_index; ++s) {
                    flat += w.head_work[s].q_chunk_count;
                }
                const uint32_t start_in_head = flat % num_q_chunks;
                const uint32_t count = w.head_work[seg.head_work_index].q_chunk_count;
                if (seg.core_idx == i) {
                    sol_my_start = start_in_head;
                }
                sol_members.emplace_back(start_in_head, count);
            }
            TT_FATAL(
                sol_members.size() <= 64, "SparseRingSDPA: head chain of {} members is too long", sol_members.size());
        }
        reader_args.push_back(static_cast<uint32_t>(sol_members.size()));
        reader_args.push_back(sol_my_start);
        // Column-bias value per block token count, bf16, two per word: entry[len] for len in [0, 64] (entry[0] = the
        // "exact / padding" large negative). The generator weights each pooled column by its token count through
        // exp((QK + ln(len)/scale - max) * scale) = len * exp(...).
        for (uint32_t w = 0; w < kSolLenBiasTableWords; ++w) {
            reader_args.push_back(sol_len_bias_table[w]);
        }
        for (const auto& [start, count] : sol_members) {
            reader_args.push_back(start);
            reader_args.push_back(count);
        }
        if (sol_res_mcast) {
            for (const uint32_t w : sol_group_args[i]) {
                reader_args.push_back(w);
            }
        }
        reader_kernel.emplace_runtime_args(core, reader_args.args);

        // Writer args
        CheckedRuntimeArgList writer_args;
        writer_args.push_back(out_buf);
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
        compute_args.push_back(transport_rank);
        compute_args.push_back(forward_writes_expected);
        compute_args.push_back(backward_writes_expected);
        compute_args.push_checked(runtime_arg_layout.compute_logical_nt, logical_nt, "compute.logical_nt");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_start_tile, kUnusedQMappingWord, "compute.q_pre_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_tile_count, kUnusedQMappingWord, "compute.q_pre_wrap_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_post_wrap_start_tile, kUnusedQMappingWord, "compute.q_post_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_valid_tile_count, kUnusedQMappingWord, "compute.q_valid_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_active_ring_iter_mask, active_ring_iter_mask, "compute.active_ring_iter_mask");
        compute_kernel.emplace_runtime_args(core, compute_args.args);
    }

    // Push the SDPA kernels into desc before invoking the all-gather helper so the helper appends its own kernels
    // after these, at kFirstAllGatherKernelIndex.
    static_assert(kReaderKernelIndex == 0 && kWriterKernelIndex == 1 && kComputeKernelIndex == 2);
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k, input_tensor_v};
    std::vector<Tensor> all_gather_output_tensors = {gathered_input_tensor_k, gathered_input_tensor_v};
    // Append the all-gather portion to `desc`. Buffer addresses are auto-patched on cache hits.
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
        /*input_batch_slice_idx=*/std::nullopt,
        /*gather_valid_Ht=*/std::nullopt,
        /*slot_id=*/std::nullopt,
        /*kv_actual_isl=*/std::nullopt,
        /*chunk_local_tiles=*/tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT,
        /*kv_cache_num_layers=*/1,
        /*kv_cache_layer_idx=*/0,
        // Share the split-forwarding decision derived above so the all-gather only splits when this
        // consumer implements the second-half wait.
        sdpa_fused_op_signaler->split_forwarding_enabled,
        /*partial_readiness_enabled=*/false,
        rank_mapping);

    return desc;
}

}  // namespace

// One ProgramDescriptor per coord: transport rank / forward_coord / backward_coord (used by the all-gather portion)
// depend on the mesh coordinate, so descriptors cannot be shared across coords. Returning a WorkloadDescriptor keeps
// the framework on its no-rebuild cache-hit fast path.
tt::tt_metal::WorkloadDescriptor SparseRingSDPAProgramFactory::create_workload_descriptor(
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_sparse_ring_sdpa_program_descriptor(args, tensor_args, output_tensors, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

SparseRingSDPAMeshWorkloadFactory::cached_mesh_workload_t SparseRingSDPAMeshWorkloadFactory::create_mesh_workload(
    const SparseRingSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void SparseRingSDPAMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);

    // The fused all-gather's GlobalSemaphore addresses are excluded from the program hash and baked into the
    // all-gather reader/writer runtime args at creation, so re-apply them on every cache hit (the standalone
    // all-gather op does the same); otherwise a cached program keeps signalling the ping-pong semaphore it was
    // created with.
    const auto& ag_semaphores = args.all_gather_operation_attributes.semaphore;
    if (ag_semaphores.size() <= std::max(ag_dyn::kForwardSemaphoreIdx, ag_dyn::kBackwardSemaphoreIdx)) {
        return;
    }
    const auto forward_sem_addr = static_cast<uint32_t>(ag_semaphores.at(ag_dyn::kForwardSemaphoreIdx).address());
    const auto backward_sem_addr = static_cast<uint32_t>(ag_semaphores.at(ag_dyn::kBackwardSemaphoreIdx).address());

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        TT_FATAL(
            coordinate_range.start_coord() == coordinate_range.end_coord(),
            "Expected SparseRingSDPA cached programs to cover a single coordinate, got range {} to {}",
            coordinate_range.start_coord(),
            coordinate_range.end_coord());
        // The all-gather always appends its four kernels at kFirstAllGatherKernelIndex.
        const auto patch_semaphore_arg = [&program](uint32_t kernel_idx, uint32_t arg_idx, uint32_t address) {
            auto& args_by_core = tt::tt_metal::GetRuntimeArgs(program, kernel_idx);
            for (auto& args_by_y : args_by_core) {
                for (auto& core_args : args_by_y) {
                    if (arg_idx < core_args.size()) {
                        core_args[arg_idx] = address;
                    }
                }
            }
        };
        patch_semaphore_arg(kAllGatherReaderForwardKernelIndex, ag_dyn::kReaderSemaphoreArg, forward_sem_addr);
        patch_semaphore_arg(kAllGatherWriterForwardKernelIndex, ag_dyn::kWriterSemaphoreArg, forward_sem_addr);
        patch_semaphore_arg(kAllGatherReaderBackwardKernelIndex, ag_dyn::kReaderSemaphoreArg, backward_sem_addr);
        patch_semaphore_arg(kAllGatherWriterBackwardKernelIndex, ag_dyn::kWriterSemaphoreArg, backward_sem_addr);
    }
}

}  // namespace ttnn::prim
