// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA writer (compute: ring_joint_sdpa.cpp, the precision recipes). It generates the reduce scaler and
// the column identity, follows the ring order (all-gather handshake, the active ring iterations derived from the live
// lengths), writes each Q chunk's normalized output on the iteration compute normalizes it, and services compute's
// state checkpoints for cores with several Q chunks (recipe_state_transfer.hpp: raw state bytes to the internal state
// tensor, common runtime arg 3, on the checkpoint request / ack CBs).

// Size-optimized: the dataflow kernels leave the kernel config buffer to the recipe compute (the ring's masked recipe
// builds sit within a few hundred bytes of it on Blackhole).
#pragma GCC optimize("Os")

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "metadata_scalar_read.hpp"
#include "fused_op_receiver.hpp"
#include "recipe_state_transfer.hpp"
#include "ring_utils.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
};

// The output slice of one Q chunk. has_joint_q: when false, the joint branch is statically dead.
template <bool has_joint_q>
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t num_local_q_chunks,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt) {
    QChunkInfo info;
    if constexpr (has_joint_q) {
        info.is_joint_q = q_chunk >= num_local_q_chunks;
        if (info.is_joint_q) {
            const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
            info.out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
            return info;
        }
    } else {
        info.is_joint_q = false;
    }
    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
    info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
    return info;
}

// Padding boundary for write paths — needs ring_id to know how far the valid sequence extends.
// has_joint_q: when false, joint branch is statically dead.
template <bool has_joint_q>
inline uint32_t get_end_seq_tile(const QChunkInfo& qi, uint32_t ring_id, uint32_t Lt, uint32_t q_local_padded_Nt) {
    if constexpr (has_joint_q) {
        return qi.is_joint_q ? Lt : q_local_padded_Nt * (ring_id + 1);
    } else {
        return q_local_padded_Nt * (ring_id + 1);
    }
}

// One raw state checkpoint request from compute: streamed (BF16 recipes with fused chunks) or synchronous.
template <bool stream_state, bool fp32_state, uint32_t Sq_chunk_t, uint32_t vDHt, typename Accessor>
FORCE_INLINE void transfer_state(Noc& noc, const Accessor& backing) {
    if constexpr (stream_state) {
        stream_recipe_state<Sq_chunk_t, recipe_ring_cb::checkpoint_request, recipe_ring_cb::checkpoint_ack, vDHt>(
            noc, backing);
    } else {
        transfer_recipe_state<fp32_state, Sq_chunk_t, recipe_ring_cb::checkpoint_request, recipe_ring_cb::checkpoint_ack, vDHt>(
            noc, backing);
    }
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    // Length-geometry slots (the _ct values): placeholders with dynamic_local_n, see ring_joint_derived_slots.hpp.
    constexpr uint32_t q_local_padded_Nt_ct = get_compile_time_arg_val(6);
    constexpr uint32_t kv_local_padded_Nt_ct = get_compile_time_arg_val(7);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks_ct = get_compile_time_arg_val(11);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_local_k_chunks_ct = get_compile_time_arg_val(13);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_chunks_ct = get_compile_time_arg_val(15);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr uint32_t is_causal = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t is_balanced = get_compile_time_arg_val(23) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t out_row_group_h = get_compile_time_arg_val(25);
    constexpr bool chunked_enabled = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t chunk_size_t_ct = get_compile_time_arg_val(27);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    constexpr bool dynamic_local_n = get_named_compile_time_arg_val("dynamic_local_n") == 1;
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt_ct % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt_ct = q_local_padded_Nt_ct / kv_stripe_split;
    static_assert(kv_region_Nt_ct * ring_size == chunk_size_t_ct, "chunk_size_t must span one KV region per ring rank");
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Slot 29: trace-safe KV-pad derivation. When set, the writer reads kv_actual_isl from the
    // kv_actual_isl tensor[0] (metadata common arg 0 = its DRAM addr) and recomputes logical_nt + ring
    // masks on-device (it's a dataflow kernel, can NoC-read), so a captured trace replays across chunks.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(29) == 1;
    // Slot 30: sharded-joint flag. When true, one L/P shard arrives per ring iteration rather than only the last
    // active iteration.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(30) == 1;
    // Slot 31: true (unpadded) joint length in tiles (twins spatial logical_nt).
    constexpr uint32_t logical_lt = get_compile_time_arg_val(31);
    // Slots 32-35: transport-to-tensor rank mapping.
    constexpr uint32_t full_mesh_rank_mapping =
        get_compile_time_arg_val(32);  // k{Axis,RowMajor,ColumnMajor}RankMapping
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(33));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);
    // Slots 36-37: logical-length transport. Being a dataflow kernel, the writer NoC-reads the live values
    // itself (as it already does for kv_actual_isl) and re-derives the active ring iterations.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(36) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(37) == 1;
    constexpr bool has_logical_length_tensor = has_logical_n_tensor || has_logical_l_tensor;
    // The recipe's state planes (the host gives every ring kernel the recipe's named arguments): FP32 maxima for the
    // FP32 recipes; streamed checkpoints for BF16 recipes with fused chunks.
    constexpr bool fp32_state = get_named_compile_time_arg_val("recipe_fp32") != 0;
    constexpr bool stream_state = get_named_compile_time_arg_val("recipe_ring_stream_state") != 0;

    // Joint-path compile-time gating. When zero, joint Q branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and the joint_out_generator.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;

    // Slots 28-37: sliding_window_size, kv_pad_from_metadata, joint_is_sharded, logical_lt, the rank-mapping
    // descriptor, and the two logical-length transport flags. Output accessors start at slot 38: the output, the joint
    // output, the stats tensor, the recipe state tensor.
    constexpr uint32_t kFirstAccessorArgOffset = 38;
    constexpr auto out_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    // Metadata accessor (metadata path only) follows the output accessors; gate the offset on kv_pad_from_metadata so
    // the no-metadata program never names a non-accessor compile arg (fall back to a valid unused accessor offset).
    constexpr uint32_t meta_args_offset =
        kv_pad_from_metadata ? state_args.next_compile_time_args_offset() : kFirstAccessorArgOffset;
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();
    constexpr bool has_kv_valid_end = get_named_compile_time_arg_val("kv_valid_end") == 1 && kv_pad_from_metadata;
    constexpr uint32_t kv_valid_end_args_offset =
        has_kv_valid_end ? meta_args.next_compile_time_args_offset() : meta_args_offset;
    constexpr auto kv_valid_end_args = TensorAccessorArgs<kv_valid_end_args_offset>();
    // logical_n / logical_l accessors follow the metadata accessor, appended as a pair so the offsets do
    // not depend on which one was supplied.
    constexpr uint32_t post_meta_args_offset = kv_pad_from_metadata ? kv_valid_end_args.next_compile_time_args_offset()
                                                                    : state_args.next_compile_time_args_offset();
    constexpr uint32_t logical_n_args_offset =
        has_logical_length_tensor ? post_meta_args_offset : kFirstAccessorArgOffset;
    constexpr auto logical_n_args = TensorAccessorArgs<logical_n_args_offset>();
    constexpr uint32_t logical_l_args_offset =
        has_logical_length_tensor ? logical_n_args.next_compile_time_args_offset() : logical_n_args_offset;
    constexpr auto logical_l_args = TensorAccessorArgs<logical_l_args_offset>();

    // Tensor addresses are the same on every core, so the factory passes them as the first common runtime args.
    constexpr uint32_t address_common_arg_count = ring_joint::kWriterAddressCommonArgCount;
    uint32_t common_argidx = 0;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(common_argidx++);
    const uint32_t joint_out_addr = get_common_arg_val<uint32_t>(common_argidx++);
    common_argidx++;  // the stats tensor (not written)
    const uint32_t state_addr = get_common_arg_val<uint32_t>(common_argidx++);
    ASSERT(common_argidx == address_common_arg_count);

    const ring_joint::LocalNGeometry local_n =
        dynamic_local_n
            ? ring_joint::derive_local_n_geometry(
                  get_common_arg_val<uint32_t>(ring_joint::dynamic_local_n_writer_common_arg(
                      kv_pad_from_metadata, has_logical_length_tensor, has_kv_valid_end)),
                  Sq_chunk_t,
                  Sk_chunk_t,
                  kv_stripe_split,
                  ring_size,
                  num_joint_q_chunks)
            : ring_joint::LocalNGeometry{
                  q_local_padded_Nt_ct,
                  kv_local_padded_Nt_ct,
                  num_local_q_chunks_ct,
                  num_local_k_chunks_ct,
                  num_q_chunks_ct,
                  kv_region_Nt_ct,
                  chunk_size_t_ct};
    const uint32_t q_local_padded_Nt = local_n.q_local_padded_Nt;
    const uint32_t kv_local_padded_Nt = local_n.kv_local_padded_Nt;
    const uint32_t num_local_q_chunks = local_n.num_local_q_chunks;
    const uint32_t num_local_k_chunks = local_n.num_local_k_chunks;
    const uint32_t num_q_chunks = local_n.num_q_chunks;
    const uint32_t kv_region_Nt = local_n.kv_region_Nt;
    const uint32_t chunk_size_t = local_n.q_chunk_group_tile_count;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    argidx++;  // single_valid_kv_chunk_mask (the deferred-normalization writer's)
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_out = recipe_ring_cb::out;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    Noc noc;

    // Common runtime args after the addresses: kv_actual_isl slot first when present, then the logical-length pair.
    constexpr uint32_t metadata_common_arg_base = address_common_arg_count;
    constexpr uint32_t logical_length_common_arg_base =
        metadata_common_arg_base + (kv_pad_from_metadata ? ring_joint::kWriterMetadataCommonArgCount : 0);
    // kv_valid_end is validated incompatible with the logical_n / logical_l tensors, so its arg directly
    // follows the kv_actual_isl slot.
    static_assert(!has_kv_valid_end || !has_logical_length_tensor);
    [[maybe_unused]] constexpr uint32_t kv_valid_end_common_arg = logical_length_common_arg_base;

    if constexpr (has_logical_length_tensor) {
        CircularBuffer cb_meta_scratch(cb_out);
        const uint32_t meta_l1 = cb_meta_scratch.get_write_ptr();
        uint32_t logical_lt_live = logical_lt;
        if constexpr (has_logical_n_tensor) {
            const uint32_t logical_n_live = trace_metadata::read_metadata_scalar_u32(
                noc, logical_n_args, get_common_arg_val<uint32_t>(logical_length_common_arg_base), meta_l1);
            ASSERT(logical_n_live >= 1);
            logical_nt = ring_joint::tiles_for(logical_n_live);
        }
        if constexpr (has_logical_l_tensor) {
            const uint32_t logical_l_live = trace_metadata::read_metadata_scalar_u32(
                noc, logical_l_args, get_common_arg_val<uint32_t>(logical_length_common_arg_base + 1), meta_l1);
            logical_lt_live = ring_joint::tiles_for(logical_l_live);
        }
        // Re-derive from the live lengths so the writer agrees with the reader about which iterations carry work.
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
            // Always false here: the logical-length-tensor path is validated incompatible with KV-pad rotation.
            .kv_pad_rotation_enabled = false,
            .kernel_is_causal = is_causal != 0,
            .is_balanced = is_balanced != 0,
            .joint_is_sharded = joint_is_sharded,
            // Writer slot 9 (Lt) is already the per-device joint tile count; matches the reader's Lt_local.
            .joint_local_padded_Nt = Lt,
            .logical_lt = logical_lt_live,
        });
        active_ring_iter_mask = masks.active_ring_iter_mask;
    }

    if constexpr (kv_pad_from_metadata) {
        CircularBuffer cb_meta_scratch(cb_out);
        uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(metadata_common_arg_base), cb_meta_scratch.get_write_ptr());
        if constexpr (has_sliding_window) {
            kv_actual_isl = trace_metadata::bounded_sliding_kv_actual_isl(
                kv_actual_isl,
                q_local_padded_Nt,
                ring_size,
                kv_local_padded_Nt,
                get_named_compile_time_arg_val("sliding_halo_slot_count"));
        } else {
            kv_actual_isl =
                trace_metadata::bounded_kv_actual_isl(kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
        }
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
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr);
    const auto state_backing = TensorAccessor(state_args, state_addr);

    constexpr bool output_has_no_padding =
        !dynamic_local_n && !has_joint_q && (q_local_padded_Nt_ct % Sq_chunk_t == 0);
    using StaticOutputTileShape = StaticTensorTileShape<B, NH, q_local_padded_Nt_ct, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;

    const auto output_tile_logical = OutputTileShape(B, NH, q_local_padded_Nt, vDHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, vDHt);
    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

    // The recipe's reduce scaler and column identity (as the dense recipe reader, reader_recipe.cpp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        recipe_ring_cb::identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(recipe_ring_cb::col_identity), 0x3f803f80);

    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    const uint32_t half_sequence = num_q_chunks / 2;

    // Balanced causal: the early half's Q chunks see no step on a later device's K and normalize on the last step on
    // this or an earlier device's (ring_joint_sdpa.cpp walks the same order).
    [[maybe_unused]] uint32_t early_last_iter = 0;
    if constexpr (is_causal && is_balanced && !has_sliding_window) {
        RingIdSequencer order = fused_op_receiver.seq;
        for (uint32_t it = 0; it < ring_size; ++it) {
            const uint32_t id = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                order.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
            if (((active_ring_iter_mask >> it) & 1u) != 0 && id <= ring_index) {
                early_last_iter = it;
            }
        }
    }

    // One Q chunk's normalized output, in the row groups compute packs it.
    const auto write_output = [&](uint32_t q_flat, uint32_t ring_id) {
        const auto decoded = decompose_global_q_index(q_flat, num_q_chunks, NH, use_zigzag_balancing);
        const auto qi =
            get_q_chunk_info<has_joint_q>(decoded.q_chunk, decoded.nb, decoded.nq, num_local_q_chunks, Sq_chunk_t, vDHt);
        const auto& gen = [&]() -> const auto& {
            if constexpr (has_joint_q) {
                if (qi.is_joint_q) {
                    return joint_out_generator;
                }
            }
            return out_generator;
        }();
        write_block_row_grouped_trid<output_has_no_padding>(
            noc,
            gen,
            qi.out_slice,
            get_end_seq_tile<has_joint_q>(qi, ring_id / kv_stripe_split, Lt, q_local_padded_Nt),
            cb_out,
            tile_bytes,
            out_row_group_h,
            /*flush_trid=*/0);
    };

    // Several Q chunks on a core checkpoint their state between ring iterations; one stays resident in L1.
    const bool staged = global_q_end - global_q_start > 1;
    bool seen_active_iter = false;
    const uint32_t sdpa_ring_iterations = ring_joint::ring_pass_count(ring_size, has_sliding_window);
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        // Sliding compute consumes all local/halo source ranges in one logical pass, so the writer sees exactly one
        // final output per Q chunk and no checkpoint.
        const uint32_t ring_id =
            has_sliding_window
                ? ring_index
                : ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                      fused_op_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so writer stays aligned with reader, compute, and all-gather.
        if (!has_sliding_window && ((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        const bool first_active_iter = !seen_active_iter;
        seen_active_iter = true;
        const bool is_last_ring_iter = has_sliding_window || is_last_active_ring_iter(active_ring_iter_mask, ring_iter);
        const bool skip_early = is_causal && is_balanced && !has_sliding_window && ring_index < ring_id;
        const bool early_last = is_causal && is_balanced && !has_sliding_window && ring_iter == early_last_iter;
        for (uint32_t q_flat = global_q_start; q_flat < global_q_end; ++q_flat) {
            const uint32_t q_chunk = decompose_global_q_index(q_flat, num_q_chunks, NH, use_zigzag_balancing).q_chunk;
            const bool early = q_chunk < half_sequence;
            if (skip_early && early) {
                continue;
            }
            if (staged && !first_active_iter) {
                transfer_state<stream_state, fp32_state, Sq_chunk_t, vDHt>(noc, state_backing);  // restore
            }
            if (is_last_ring_iter || (early && early_last)) {
                write_output(q_flat, ring_id);
            } else if (staged) {
                transfer_state<stream_state, fp32_state, Sq_chunk_t, vDHt>(noc, state_backing);  // save
            }
        }
        // Every checkpoint saved this iteration lands before the next restores it.
        noc.async_write_barrier();
    }
}
