// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_kv_pad_derivation.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/sliding_ring_sdpa_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// One completion bit per signaler, kept from the shared op's rotated Q split handoff. Each signaler
// adds its own power of two, so atomic addition is equivalent to OR: no carries, no lost signals,
// and no ordering assumption between signalers. The K-split reducer waits on one bit per sender band.
class RotatedQHandoff {
public:
    explicit RotatedQHandoff(uint32_t semaphore_id) : semaphore_(semaphore_id) {}

    bool ready(uint32_t ordinal) const { return (semaphore_.value() & bit(ordinal)) != 0; }

    void wait(uint32_t ordinal) const {
        WAYPOINT("RQHW");
        while (!ready(ordinal)) {
            invalidate_l1_cache();
        }
        WAYPOINT("RQHD");
    }

    // Caller must have the signaled state in place before signaling.
    void signal(const Noc& noc, uint32_t x, uint32_t y, uint32_t ordinal) { semaphore_.up(noc, x, y, bit(ordinal)); }

    // Only after every expected incoming signal has been observed. There are then
    // no outstanding signals to this core; program completion separates cached runs.
    void reset_after_run() { semaphore_.set(0); }

private:
    static uint32_t bit(uint32_t ordinal) {
        ASSERT(ordinal >= 1 && ordinal < 32);
        return 1u << (ordinal - 1);
    }

    Semaphore<> semaphore_;
};

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t stats_seq_start_tile;
    uint32_t stats_seq_end_tile;
};

// Compute DRAM address info (output slice + stats range) for one Q chunk.
// has_joint_q: when false, joint branch is statically dead and dropped by the compiler.
template <bool has_joint_q>
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t num_local_q_chunks,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt,
    const uint32_t Lt,
    const uint32_t q_local_padded_Nt) {
    QChunkInfo info;
    info.is_joint_q = false;
    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
    info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
    info.stats_seq_start_tile = q_chunk * Sq_chunk_t;
    info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
    info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, q_local_padded_Nt);
    info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, q_local_padded_Nt);
    return info;
}

// Padding boundary for write paths — needs ring_id to know how far the valid sequence extends.
// has_joint_q: when false, joint branch is statically dead.
template <bool has_joint_q>
inline uint32_t get_end_seq_tile(const QChunkInfo& qi, uint32_t ring_id, uint32_t Lt, uint32_t q_local_padded_Nt) {
    return q_local_padded_Nt * (ring_id + 1);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(11);

    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(16);
    constexpr uint32_t scale_val = get_compile_time_arg_val(17);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(19);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(20);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t is_causal = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t is_balanced = get_compile_time_arg_val(23) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(25);
    constexpr bool chunked_enabled = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(27);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Slot 29: trace-safe KV-pad derivation. When set, the writer reads kv_actual_isl from the
    // kv_actual_isl tensor[0] (common runtime arg 0 = its DRAM addr) and recomputes logical_nt + ring
    // masks on-device (it's a dataflow kernel, can NoC-read), so a captured trace replays across chunks.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(29) == 1;
    // Slot 30: sharded-joint flag (appended after upstream's kv_pad_from_metadata). When true, one L/P
    // shard arrives per ring iteration rather than only the last active iteration.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(30) == 1;
    // Slot 31: true (unpadded) joint length in tiles (twins spatial logical_nt). Drives the joint
    // mask-generation gate together with joint_l_partial_col.
    constexpr uint32_t logical_lt = get_compile_time_arg_val(31);
    // Slots 32-35: transport-to-tensor rank mapping.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(32) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(33));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);
    // Slots 36-37: logical-length transport. Being a dataflow kernel, the writer NoC-reads the live values
    // itself (as it already does for kv_actual_isl) and re-derives the masks and partial-column stamps.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(36) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(37) == 1;
    constexpr bool has_logical_length_tensor = has_logical_n_tensor || has_logical_l_tensor;
    // Diagonal-mask tile slot is shared by the kernel's is_causal path and the chunked-prefill
    // path. The program factory masks kernel_is_causal off when chunked is on, so only one of
    // the two paths drives the stamp per program — but they share the CB slot layout.
    constexpr bool diag_tile_enabled = ((is_causal == 1) || chunked_enabled) && !has_sliding_window;
    // The host builds this op only for sliding attention; the shared op's dense streaming and eager
    // output paths are not carried here.
    static_assert(has_sliding_window, "Gemma4 sliding ring SDPA requires a sliding window");

    // Joint-path compile-time gating. When zero, joint Q branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and the joint_out_generator.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;

    // Slots 28-37: sliding_window_size, kv_pad_from_metadata, joint_is_sharded, logical_lt, the rank-mapping
    // descriptor, and the two logical-length transport flags. Output accessors start at slot 38.
    constexpr uint32_t kFirstAccessorArgOffset = 38;
    constexpr auto out_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();
    // Metadata accessor (metadata path only) follows the output accessors and precedes the CB compile
    // args; gate the offset on kv_pad_from_metadata so the no-metadata program never names a non-accessor
    // compile arg (fall back to a valid unused accessor offset = out_args' slot).
    constexpr uint32_t meta_args_offset =
        kv_pad_from_metadata ? stats_args.next_compile_time_args_offset() : kFirstAccessorArgOffset;
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();
#ifdef KV_VALID_END
    constexpr bool has_kv_valid_end = kv_pad_from_metadata;
#else
    constexpr bool has_kv_valid_end = false;
#endif
    constexpr uint32_t kv_valid_end_args_offset =
        has_kv_valid_end ? meta_args.next_compile_time_args_offset() : meta_args_offset;
    constexpr auto kv_valid_end_args = TensorAccessorArgs<kv_valid_end_args_offset>();
    // logical_n / logical_l accessors follow the metadata accessor, appended as a pair so the offsets do
    // not depend on which one was supplied.
    constexpr uint32_t post_meta_args_offset = kv_pad_from_metadata ? kv_valid_end_args.next_compile_time_args_offset()
                                                                    : stats_args.next_compile_time_args_offset();
    constexpr uint32_t logical_n_args_offset =
        has_logical_length_tensor ? post_meta_args_offset : kFirstAccessorArgOffset;
    constexpr auto logical_n_args = TensorAccessorArgs<logical_n_args_offset>();
    constexpr uint32_t logical_l_args_offset =
        has_logical_length_tensor ? logical_n_args.next_compile_time_args_offset() : logical_n_args_offset;
    constexpr auto logical_l_args = TensorAccessorArgs<logical_l_args_offset>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    argidx++;  // single_valid_kv_chunk_mask: orders the dense save/restore only, which sliding never runs
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_arg_offset =
        has_logical_length_tensor ? logical_l_args.next_compile_time_args_offset() : post_meta_args_offset;

    // The factory keeps the shared op's rotated Q split slot as the final compile-time arg of every kernel.
    // Sliding never rotates Q, so it is always 0.
    constexpr uint32_t rotated_max_slots = get_ct_arg<kernel_compile_time_args.size() - 1>();
    static_assert(rotated_max_slots == 0, "Sliding attention has no rotated Q split");

    // K split (ring_joint_ksplit.hpp) runtime args follow the fused-op receiver block: [band], then for a core with
    // a unit the physical (x, y) of its reducer (a sender) or of its senders in band order (the reducer).
    constexpr uint32_t ksplit_count = get_named_compile_time_arg_val("ksplit_count");
    constexpr bool ksplit_enabled = ksplit_count > 1;
    static_assert(!ksplit_enabled || use_streaming_compute);
    static_assert(ksplit_count <= ring_joint::kKSplitMaxCount);
    const bool ksplit_active = ksplit_enabled && global_q_end - global_q_start == 1;
    // Segmented accumulation is dense-only; the host rejects it with a sliding window.
    static_assert(get_named_compile_time_arg_val("seg_accum") == 0, "Sliding attention has no segmented accumulation");
    const uint32_t ksplit_band = ksplit_enabled ? get_arg_val<uint32_t>(argidx) : 0;
    const uint32_t ksplit_peer_args = argidx + 1;
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);  // K-split reducer: sender max
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_signal = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    // A K-split sender leaves its state in accumulator half A. Those CBs hold one entry and sit at the same L1 offset
    // on every core, so the reducer reads a sender's state at its own CB bases. cb_sum_in exists only on the streaming
    // path; alias a live CB so the other modes still compile.
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 18);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 20);
    constexpr uint32_t ksplit_cb_sum_in = ksplit_enabled ? cb_sum_in : cb_out;

    Noc noc;

    // Common runtime args: kv_actual_isl slot first when present, then the logical-length pair.
    constexpr uint32_t logical_length_common_arg_base =
        kv_pad_from_metadata ? ring_joint::kWriterMetadataCommonArgCount : 0;
    // kv_valid_end is validated incompatible with the logical_n / logical_l tensors, so its arg directly
    // follows the kv_actual_isl slot.
    static_assert(!has_kv_valid_end || !has_logical_length_tensor);
    [[maybe_unused]] constexpr uint32_t kv_valid_end_common_arg = logical_length_common_arg_base;
    [[maybe_unused]] uint32_t global_n_partial_col_live = global_n_partial_col;
    [[maybe_unused]] uint32_t joint_l_partial_col_live = joint_l_partial_col;

    if constexpr (kv_pad_from_metadata) {
        CircularBuffer cb_meta_scratch(cb_out);
        uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(0), cb_meta_scratch.get_write_ptr());
        kv_actual_isl = trace_metadata::bounded_sliding_kv_actual_isl(
            kv_actual_isl, q_local_padded_Nt, ring_size, kv_local_padded_Nt, SLIDING_HALO_SLOT_COUNT);
        logical_nt = trace_metadata::logical_tile_rows_clamped_to_cache(
            kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
        if constexpr (has_kv_valid_end) {
            const uint32_t kv_valid_end = trace_metadata::read_metadata_scalar_u32(
                noc,
                kv_valid_end_args,
                get_common_arg_val<uint32_t>(kv_valid_end_common_arg),
                cb_meta_scratch.get_write_ptr());
            logical_nt =
                trace_metadata::logical_tile_rows_clamped_to_valid_end(logical_nt, kv_actual_isl, kv_valid_end);
        }
        // Joint trio stays defaulted: KV-pad rotation is validated incompatible with a sharded joint.
        const auto masks = ring_joint::build_ring_work_masks_device<full_mesh_rank_mapping>({
            .transport_rank = fused_op_receiver.seq.ring_index,
            .ring_size = ring_size,
            .mesh_rows = mesh_rows,
            .mesh_cols = mesh_cols,
            .snake_orientation = snake_orientation,
            .backward_writes_expected = fused_op_receiver.seq.expected[0],
            .forward_writes_expected = fused_op_receiver.seq.expected[1],
            .num_local_k_chunks = num_local_k_chunks,
            .k_chunk_tile_count = Sk_chunk_t,
            .kv_local_padded_Nt = kv_local_padded_Nt,
            .kernel_chunked = chunked_enabled,
            .q_chunk_group_tile_count = chunk_size_t,
            .kv_region_Nt = kv_region_Nt,
            .logical_nt = logical_nt,
            .num_joint_k_chunks = num_joint_k_chunks,
            .joint_seq_len = L,
            .kv_pad_rotation_enabled = true,
            .kernel_is_causal = is_causal != 0,
            .is_balanced = is_balanced != 0,
        });
        active_ring_iter_mask = masks.active_ring_iter_mask;
    }

    const auto out_writer = TensorAccessor(out_args, out_addr);

    constexpr bool output_has_no_padding = !has_joint_q && (q_local_padded_Nt % Sq_chunk_t == 0);
    using StaticOutputTileShape = StaticTensorTileShape<B, NH, q_local_padded_Nt, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;

    const auto output_tile_logical = OutputTileShape(B, NH, q_local_padded_Nt, vDHt);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);

    generate_bcast_unary_scalar(CircularBuffer(cb_scale_in), scale_val);
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    // Needed when any K/joint dimension has padding, or when causal/chunked masking is active.
    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    // Joint mask generation mirrors spatial's TWO independent flags (like local_n AND global_n).
    //   (local_n analogue) Lt % Sk_chunk_t != 0: the K-chunk is wider than the per-device joint shard
    //     (writer slot 9 Lt is already per-device Lt_local), so every fully-real shard carries
    //     fully-padded trailing tiles (e.g. wadada Lt=2, Sk=16 -> 14 pad tiles per shard).
    //   (global_n analogue) logical_lt % Sk_chunk_t != 0 || joint_l_partial_col != 0: real tokens do
    //     not fill the last real shard's chunk — fully-padded trailing tiles and/or a sub-tile column.
    constexpr bool joint_has_padding =
        L > 0 && ((Lt % Sk_chunk_t != 0) || (logical_lt % Sk_chunk_t != 0) || (joint_l_partial_col != 0));
    constexpr bool needs_lightweight_mask = (local_n_has_padding || global_n_has_padding || joint_has_padding) ||
                                            diag_tile_enabled || has_sliding_window || has_logical_length_tensor;
    // partial_tile_present keeps tile presence in lock-step with the factory's CB sizing and compute's tile
    // indices. A present tile needs a non-zero template column (fall back to 1) so
    // generate_lightweight_mask_tiles allocates the slot; the stamped live column may still be 0, which
    // compute ignores.
    constexpr uint32_t global_n_partial_col_layout =
        ring_joint::partial_tile_present(global_n_partial_col, has_logical_n_tensor)
            ? (global_n_partial_col > 0 ? global_n_partial_col : 1u)
            : 0u;
    constexpr uint32_t joint_l_partial_col_layout =
        ring_joint::partial_tile_present(joint_l_partial_col, has_logical_l_tensor)
            ? (joint_l_partial_col > 0 ? joint_l_partial_col : 1u)
            : 0u;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<
            global_n_partial_col_layout,
            joint_l_partial_col_layout,
            cb_mask_in,
            (is_causal == 1) || chunked_enabled,
            sliding_window_size>(noc, global_n_partial_col_live, joint_l_partial_col_live);
    }

    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);

    auto arrival_preview = fused_op_receiver.seq;
    const ring_joint::PackedKVSchedule packed_schedule = ring_joint::make_packed_kv_schedule<GROUPED_KV_SOURCE_COUNT>(
        ring_size,
        kv_local_padded_Nt,
        logical_nt,
        active_ring_iter_mask,
        kv_region_Nt,
        Sk_chunk_t,
        has_sliding_window,
        [&] {
            return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                arrival_preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
        });
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        const uint32_t ring_id = ring_index;

        // Sliding compute consumes every local/halo source for a Q in one pass. There is
        // therefore no intermediate state to restore or save: wait for the final result,
        // write it once, and advance directly to the next assigned Q.
        const bool single_q_chunk = (global_q_end - global_q_start == 1);
        for (uint32_t q_index = 0; q_index + global_q_start < global_q_end; ++q_index) {
            const auto decoded_q =
                decompose_global_q_index(global_q_start + q_index, num_q_chunks, NH, use_zigzag_balancing);
            const uint32_t nb = decoded_q.nb;
            const uint32_t nq = decoded_q.nq;
            const uint32_t q_chunk = decoded_q.q_chunk;
            const auto qi = get_q_chunk_info<has_joint_q>(
                q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
            const uint32_t end_seq_tile =
                get_end_seq_tile<has_joint_q>(qi, ring_id / kv_stripe_split, Lt, q_local_padded_Nt);

            if (!single_q_chunk) {
                CircularBuffer cb_sig(cb_signal);
                cb_sig.wait_front(1);
                cb_sig.pop_front(1);
            }

            // K-split bands hand their state to the reducer, which writes after the merge.
            if (ksplit_active) {
                continue;
            }
            const auto& gen = [&]() -> const auto& { return out_generator; }();
            write_block_row_grouped_trid<output_has_no_padding>(
                noc, gen, qi.out_slice, end_seq_tile, cb_out, tile_bytes, out_subblock_h, /*flush_trid=*/0);
        }
        noc.async_write_barrier();
    }
    // K-split reducers normalize after the ring loop and write their one Q chunk here.
    [[maybe_unused]] auto write_output_after_ring = [&]() {
        const auto decoded_q = decompose_global_q_index(global_q_start, num_q_chunks, NH, use_zigzag_balancing);
        const auto qi = get_q_chunk_info<has_joint_q>(
            decoded_q.q_chunk, decoded_q.nb, decoded_q.nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
        const uint32_t end_seq_tile = get_end_seq_tile<has_joint_q>(qi, ring_size - 1, Lt, q_local_padded_Nt);
        write_block_row_grouped_trid<output_has_no_padding>(
            noc, out_generator, qi.out_slice, end_seq_tile, cb_out, tile_bytes, out_subblock_h, /*flush_trid=*/0);
        noc.async_write_barrier();
    };
    if constexpr (ksplit_enabled) {
        RotatedQHandoff ready(get_named_compile_time_arg_val("ksplit_sem_id"));
        if (ksplit_active && ksplit_band + 1 < ksplit_count) {
            // Sender: compute signals once its state sits in half A, where it stays until the program ends.
            CircularBuffer staged(cb_signal);
            staged.wait_front(1);
            ready.signal(
                noc,
                get_arg_val<uint32_t>(ksplit_peer_args),
                get_arg_val<uint32_t>(ksplit_peer_args + 1),
                ksplit_band + 1);
            noc.async_atomic_barrier();
            staged.pop_front(1);
        } else if (ksplit_active) {
            // Reducer: land each sender's state in the restore CBs, one at a time (they hold one entry each).
            constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
            static_assert(
                !ksplit_enabled || get_tile_size(cb_out_im_A) == get_tile_size(cb_prev_out),
                "K split copies out tiles raw");
            UnicastEndpoint sender_l1;
            for (uint32_t sender = 0; sender + 1 < ksplit_count; ++sender) {
                const uint32_t x = get_arg_val<uint32_t>(ksplit_peer_args + 2 * sender);
                const uint32_t y = get_arg_val<uint32_t>(ksplit_peer_args + 2 * sender + 1);
                ready.wait(sender + 1);
                auto pull = [&](CircularBuffer& cb, uint32_t source_cb, uint32_t tiles) {
                    cb.reserve_back(tiles);
                    noc.async_read(
                        sender_l1,
                        CoreLocalMem<uint32_t>(cb.get_write_ptr()),
                        tiles * get_tile_size(source_cb),
                        {.noc_x = x, .noc_y = y, .addr = CircularBuffer(source_cb).get_read_ptr()},
                        {});
                };
                CircularBuffer in_max(cb_max_in);
                CircularBuffer in_sum(ksplit_cb_sum_in);
                CircularBuffer in_out(cb_prev_out);
                pull(in_max, cb_max_A, Sq_chunk_t);
                pull(in_sum, cb_sum_A, Sq_chunk_t);
                pull(in_out, cb_out_im_A, out_tiles);
                noc.async_read_barrier();
                in_max.push_back(Sq_chunk_t);
                in_sum.push_back(Sq_chunk_t);
                in_out.push_back(out_tiles);
            }
            ready.reset_after_run();

            write_output_after_ring();
        }
    }
}
