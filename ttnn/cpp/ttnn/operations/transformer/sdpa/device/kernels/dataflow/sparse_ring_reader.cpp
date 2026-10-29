// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "chain_link.hpp"
#include "pooled_bias_gen.hpp"
#include "fused_op_receiver.hpp"
#include "ring_utils.hpp"

// Device-profiler zones around the reader's waits (when built with SPARSE_STEP_PROFILE, under
// TT_METAL_DEVICE_PROFILER=1), so a stall shows up as a long RD_* zone on the core that waits instead of as a gap in
// the compute zones.
#if defined(PROFILE_KERNEL) && defined(SPARSE_STEP_PROFILE)
#define SPARSE_ZONE(name) DeviceZoneScopedN(name)
#else
#define SPARSE_ZONE(name)
#endif
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Reader of the block-sparse ring SDPA: every core holds its Q blocks ("resident" chunks of
// block_size rows) in L1 for the whole op and streams each K/V piece of a ring iteration's shard once, for all of
// them. Per ring iteration it publishes one plan (each chunk's kept blocks of the shard and its head group's piece
// union), streams the pieces (own DRAM read, chain, or the head group's multicast), and on the first active iteration
// the pooled K/V entries with their generated column bias.
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    static_assert(Sq_chunk_t == pooled_bias_gen::kBlockTiles, "one Q block (q chunk) per mask row");
    static_assert(Sk_chunk_t % pooled_bias_gen::kBlockTiles == 0, "a K/V piece holds whole blocks");
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(8);
    // Slots 9-12 and 14 are the joint-token sizes (no joint tokens), 17-19 the subblock and causal flags.
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t ring_size = get_compile_time_arg_val(16);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(20) == 1;
    constexpr bool chunked_enabled = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t num_q_readers = get_compile_time_arg_val(22);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(23);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    // Slots 24, 25, 27-29 are KV-cache, KV-pad rotation, latent V, attention sink and sliding window (all off).
    constexpr uint32_t NHV = get_compile_time_arg_val(26);
    constexpr bool v_shares_k_buffer = false;
    constexpr bool enable_kv_chains = true;
    constexpr uint32_t gathered_padded_Nt = get_compile_time_arg_val(30);
    // Slots 31-35 are the trace metadata and joint-token fields (unused).
    constexpr bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    constexpr bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    constexpr bool use_head_chain = ring_joint::uses_v_head_chain(enable_kv_chains, gqa_grouped_kv, v_shares_k_buffer);
    constexpr uint32_t q_heads_per_v = NH / NHV;
    // Slots 36-39: transport-to-tensor rank mapping. Slots 40-42 (circular KV, logical-length tensors) are unused.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(36) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(37));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(38);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(39);

    // Tensor accessors from slot 43: Q, K, V, gathered K, gathered V, the (absent) attention sink's placeholder,
    // then the block mask, pooled K and pooled V.
    constexpr uint32_t kFirstAccessorArgOffset = 43;
    constexpr auto q_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gathered_k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gathered_v_args = TensorAccessorArgs<gathered_k_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<gathered_v_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();
    constexpr auto pooled_k_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto pooled_v_args = TensorAccessorArgs<pooled_k_args.next_compile_time_args_offset()>();
    constexpr uint32_t chains_base_offset = pooled_v_args.next_compile_time_args_offset();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_v_addr = get_arg_val<uint32_t>(argidx++);
    argidx++;  // attention sink address (none)
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    argidx++;  // KV cache batch index (no KV cache)
    const uint32_t q_per_core = global_q_end - global_q_start;

    ChainConfig head_cfg;
    if constexpr (use_head_chain) {
        head_cfg = ChainConfig::read_from_args(argidx);
    }

    // Batch chain runtime args (only present for non-GQA shared-K modes).
    ChainConfig batch_cfg;  // default zero-initialized
    if constexpr (enable_kv_chains && k_uses_batch_chain) {
        batch_cfg = ChainConfig::read_from_args(argidx);
        argidx++;  // max_q_per_core (dense loop only)
    }
    ChainConfig gqa_cfg;  // default zero-initialized
    if constexpr (enable_kv_chains && gqa_grouped_kv) {
        gqa_cfg = ChainConfig::read_from_args(argidx);
        argidx++;  // gqa_max_q_per_core (dense loop only)
    }

    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    argidx += 4;  // chunked-prefill Q mapping (not chunked)
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // The sparse runtime args follow the fused-op receiver block.
    const uint32_t sparse_args_base = argidx;
    // Compile-time semaphore ids and chain flags are appended after all TensorAccessorArgs().
    // ChainLink takes semaphore IDs directly (the new Semaphore<> wrapper resolves them to L1 addrs).
    constexpr uint32_t chain_sender_semaphore_arg_offset = ring_joint::kChainSenderSemaphoreCompileArgOffset;
    constexpr uint32_t chain_receiver_semaphore_arg_offset = ring_joint::kChainReceiverSemaphoreCompileArgOffset;
    constexpr uint32_t chain_valid_semaphore_arg_offset = ring_joint::kChainValidSemaphoreCompileArgOffset;
    constexpr uint32_t chain_mcast_enabled_arg_offset = ring_joint::kChainMcastEnabledCompileArgOffset;
    constexpr uint32_t chain_compile_arg_count = ring_joint::kChainCompileArgCount;

    uint32_t head_sender_semaphore_id = 0;
    uint32_t head_receiver_semaphore_id = 0;
    uint32_t head_valid_semaphore_id = 0;
    constexpr bool head_mcast_enabled = []() {
        if constexpr (use_head_chain) {
            return get_compile_time_arg_val(chains_base_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();
    if constexpr (use_head_chain) {
        head_sender_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_sender_semaphore_arg_offset);
        head_receiver_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_receiver_semaphore_arg_offset);
        head_valid_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_valid_semaphore_arg_offset);
    }
    constexpr uint32_t head_chain_arg_count = use_head_chain ? chain_compile_arg_count : 0;
    constexpr uint32_t batch_chain_arg_count = enable_kv_chains && k_uses_batch_chain ? chain_compile_arg_count : 0;
    constexpr uint32_t gqa_chain_arg_count = enable_kv_chains && gqa_grouped_kv ? chain_compile_arg_count : 0;
    constexpr uint32_t batch_chain_ct_offset = chains_base_offset + head_chain_arg_count;
    constexpr uint32_t gqa_chain_ct_offset = batch_chain_ct_offset + batch_chain_arg_count;

    // Batch chain semaphores (only present for non-GQA shared-K modes).
    // Initialize to 0; will be overwritten if k_uses_batch_chain. Non-participants never use them.
    uint32_t batch_sender_semaphore_id = 0;
    uint32_t batch_receiver_semaphore_id = 0;
    uint32_t batch_valid_semaphore_id = 0;

    // batch_mcast_enabled: read from compile-time args if present, else false (for template instantiation)
    constexpr bool batch_mcast_enabled = []() {
        if constexpr (enable_kv_chains && k_uses_batch_chain) {
            return get_compile_time_arg_val(batch_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();

    if constexpr (enable_kv_chains && k_uses_batch_chain) {
        batch_sender_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_sender_semaphore_arg_offset);
        batch_receiver_semaphore_id =
            get_compile_time_arg_val(batch_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        batch_valid_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    uint32_t gqa_sender_semaphore_id = 0;
    uint32_t gqa_receiver_semaphore_id = 0;
    uint32_t gqa_valid_semaphore_id = 0;
    constexpr bool gqa_mcast_enabled = []() {
        if constexpr (enable_kv_chains && gqa_grouped_kv) {
            return get_compile_time_arg_val(gqa_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();
    if constexpr (enable_kv_chains && gqa_grouped_kv) {
        gqa_sender_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_sender_semaphore_arg_offset);
        gqa_receiver_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        gqa_valid_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    constexpr uint32_t cb_arg_offset =
        chains_base_offset + head_chain_arg_count + batch_chain_arg_count + gqa_chain_arg_count;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);

    const uint32_t mask_addr = get_arg_val<uint32_t>(sparse_args_base + 0);
    [[maybe_unused]] const uint32_t pooled_k_addr = get_arg_val<uint32_t>(sparse_args_base + 1);
    [[maybe_unused]] const uint32_t pooled_v_addr = get_arg_val<uint32_t>(sparse_args_base + 2);
    // +3: bias tensor address slot (the bias is generated). +4: head chain member count, +5: this core's first Q
    // chunk in the chain. Then the packed bf16 bias-per-token-count table (pooled_bias_gen::kTableWords words), the
    // members (start, count) and the head group's delivery arguments.
    const uint32_t group_n_members = get_arg_val<uint32_t>(sparse_args_base + 4);
    const uint32_t pooled_bias_table_base = sparse_args_base + 6;
    const uint32_t group_members_base = sparse_args_base + 6 + pooled_bias_gen::kTableWords;
    constexpr uint32_t logical_n_tokens = get_named_compile_time_arg_val("sparse_logical_n");
    constexpr uint32_t mask_nb = get_named_compile_time_arg_val("mask_nb");
    constexpr uint32_t mask_nb_pad = get_named_compile_time_arg_val("mask_nb_pad");
    constexpr uint32_t mask_n_blocks_pad = get_named_compile_time_arg_val("mask_n_blocks_pad");
    constexpr uint32_t mask_nqb_pad = get_named_compile_time_arg_val("mask_nqb_pad");
    constexpr uint32_t pooled_chunks = get_named_compile_time_arg_val("pooled_chunks");
    constexpr uint32_t pooled_last_tiles = get_named_compile_time_arg_val("pooled_last_tiles");
    constexpr uint32_t mask_row_bytes = get_named_compile_time_arg_val("mask_row_bytes");
    constexpr uint32_t cb_sparse_plan = get_named_compile_time_arg_val("cb_sparse_plan");
    constexpr uint32_t cb_pooled_bias = get_named_compile_time_arg_val("cb_pooled_bias");
    // The pooled route (Sol-Attn's correction for the dropped blocks): pooled K/V entries and their column bias on
    // each pass's first active ring iteration. Off: kept blocks only, the pooled accessors are null and unused.
    constexpr bool pooled_route = get_named_compile_time_arg_val("pooled_route") == 1;
    // Words of per-shard kept-block bits in every plan record (block-form mask).
    constexpr uint32_t plan_block_words = get_named_compile_time_arg_val("plan_block_words");
    static_assert(plan_block_words * 32 >= mask_nb, "block words cover the shard");
    // Head group delivery: one producer per head group reads each piece from DRAM and multicasts it to every other
    // member (no forwarding). The head chain's membership and semaphores are reused: sender = the producer's credit
    // count (+1 per consumer per piece), receiver = each consumer's delivered count (set to the running piece count
    // by multicast), valid = the producer's local source word for that set. Off when the factory builds no head
    // chains; the pieces then go over the chain or are read by each core.
    constexpr bool group_mcast = get_named_compile_time_arg_val("group_mcast") == 1;
    // rec_base != 0: one plan page per ring iteration holding every chunk's record from that word on (0 = one page
    // per chunk, when the records do not fit one page).
    constexpr uint32_t plan_rec_base = get_named_compile_time_arg_val("plan_rec_base");
    // Passes: the core's Q blocks are split evenly into num_q_passes passes of at most resident_slots blocks (the
    // factory gives every working core at least num_q_passes blocks). The first pass runs the ring with the
    // all-gather's arrival waits; later passes replay the same ring order over the gathered K/V without waits.
    constexpr uint32_t resident_slots = get_named_compile_time_arg_val("resident_slots");
    constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
    static_assert(num_q_passes >= 1 && resident_slots >= 1 && resident_slots <= 32, "passes of 1..32 Q blocks");
    static_assert(GROUPED_KV_SOURCE_COUNT == 1, "no grouped K/V sources");
    // Float16_b 32x32 bias tiles (pooled_bias_gen.hpp); the CB only exists with the pooled route.
    constexpr uint32_t pooled_bias_tile_bytes = pooled_route ? get_tile_size(cb_pooled_bias) : 2048u;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t v_cb_entry_tiles = v_chunk_tiles;

    Noc noc;

    // Head chain (query-head level): MHA uses it for K/V; separate-V shared-K uses it for V only.
    ChainLink<head_mcast_enabled, true> head_chain(
        head_cfg.participates,
        head_cfg.is_injector,
        head_cfg.is_sink,
        head_sender_semaphore_id,
        head_receiver_semaphore_id,
        head_valid_semaphore_id,
        head_cfg.signal_target_x<head_mcast_enabled>(),
        head_cfg.signal_target_y<head_mcast_enabled>(),
        head_cfg.next_physical_x,
        head_cfg.next_physical_y,
        head_cfg.mcast_start_x,
        head_cfg.mcast_start_y,
        head_cfg.mcast_end_x,
        head_cfg.mcast_end_y,
        head_cfg.mcast_num_dests,
        v_chunk_tiles,
        v_tile_bytes,
        head_cfg.head,
        head_cfg.next_core_q_chunks);

    // Shared-K chain used by K in non-GQA shared-K modes.
    ChainLink<batch_mcast_enabled, false> batch_chain(
        batch_cfg.participates,
        batch_cfg.is_injector,
        batch_cfg.is_sink,
        batch_sender_semaphore_id,
        batch_receiver_semaphore_id,
        batch_valid_semaphore_id,
        batch_cfg.signal_target_x<batch_mcast_enabled>(),
        batch_cfg.signal_target_y<batch_mcast_enabled>(),
        batch_cfg.next_physical_x,
        batch_cfg.next_physical_y,
        batch_cfg.mcast_start_x,
        batch_cfg.mcast_start_y,
        batch_cfg.mcast_end_x,
        batch_cfg.mcast_end_y,
        batch_cfg.mcast_num_dests,
        k_chunk_tiles,
        k_tile_bytes,
        0,  // chain_head unused for the shared-K chain
        batch_cfg.next_core_q_chunks);

    // GQA grouped chain (head-level where head means KV head): used by both K and V in GQA_GROUPED_KV.
    ChainLink<gqa_mcast_enabled, true> gqa_chain(
        gqa_cfg.participates,
        gqa_cfg.is_injector,
        gqa_cfg.is_sink,
        gqa_sender_semaphore_id,
        gqa_receiver_semaphore_id,
        gqa_valid_semaphore_id,
        gqa_cfg.signal_target_x<gqa_mcast_enabled>(),
        gqa_cfg.signal_target_y<gqa_mcast_enabled>(),
        gqa_cfg.next_physical_x,
        gqa_cfg.next_physical_y,
        gqa_cfg.mcast_start_x,
        gqa_cfg.mcast_start_y,
        gqa_cfg.mcast_end_x,
        gqa_cfg.mcast_end_y,
        gqa_cfg.mcast_num_dests,
        v_chunk_tiles,
        v_tile_bytes,
        gqa_cfg.head,
        gqa_cfg.next_core_q_chunks);

    // Non-shared V uses the head chain, except GQA where K and V share the grouped KV-head chain.
    auto& v_chain = [&]() -> auto& {
        if constexpr (gqa_grouped_kv) {
            return gqa_chain;
        } else {
            return head_chain;
        }
    }();

    // K uses the grouped GQA chain, the shared-K chain, or the query-head chain.
    auto& k_chain = [&]() -> auto& {
        if constexpr (gqa_grouped_kv) {
            return gqa_chain;
        } else if constexpr (k_uses_batch_chain) {
            return batch_chain;
        } else {
            return head_chain;
        }
    }();

    constexpr uint32_t q_heads_per_k = NH / NHK;

    // Throttle Q DRAM reads so many readers don't saturate the NoC outstanding-read budget.
    constexpr uint32_t q_barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_q_readers>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);

    const auto input_q_tile_logical = TensorTileShape(B, NH, q_local_padded_Nt, DHt);
    // Packed KV: K/V rows can be wider than the head dims; V is the last vDHt columns of its row.
    constexpr uint32_t k_row_Wt = get_named_compile_time_arg_val("k_row_Wt");
    constexpr uint32_t v_row_Wt = get_named_compile_time_arg_val("v_row_Wt");
    constexpr uint32_t v_col_offset_t = v_row_Wt - vDHt;
    const auto input_k_tile_logical = TensorTileShape(B, NHK, kv_local_padded_Nt, k_row_Wt);
    const auto gathered_k_input_tile_logical = TensorTileShape(B, NHK, gathered_padded_Nt, k_row_Wt);

    const auto q_generator = PaddedAddrGenerator(q_reader, input_q_tile_logical);
    const auto local_k_generator = PaddedAddrGenerator(local_k_reader, input_k_tile_logical);
    const auto gathered_k_generator = PaddedAddrGenerator(gathered_k_reader, gathered_k_input_tile_logical);
    const auto local_v_reader = TensorAccessor(v_args, v_addr);
    const auto input_v_tile_logical = TensorTileShape(B, NHV, kv_local_padded_Nt, v_row_Wt);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr);
    const auto gathered_v_input_tile_logical = TensorTileShape(B, NHV, gathered_padded_Nt, v_row_Wt);
    const auto local_v_generator = PaddedAddrGenerator(local_v_reader, input_v_tile_logical);
    const auto gathered_v_generator = PaddedAddrGenerator(gathered_v_reader, gathered_v_input_tile_logical);
    // Sparse sources: mask [1, NH, nqc_pad, n_blocks_pad] uint32 row-major (one page per row); pooled K/V
    // [1, NH, n_blocks_pad, DH] tiles (pooled route only, their readers are built where they are used).
    const auto mask_reader = TensorAccessor(mask_args, mask_addr);
    // Pooled chunks are consumed on the first active ring iteration of every pass.
    bool pooled_pending = true;
    // Block b of shard s can be exact only if it is a FULL block (partial shard / logical_n tails are always
    // pooled, so a set mask bit on them is ignored here and in the bias generator alike).
    const auto block_is_full = [&](uint32_t s, uint32_t b) -> bool {
        return pooled_bias_gen::block_len(
                   pooled_bias_gen::Geometry{kv_local_padded_Nt * 32, logical_n_tokens, mask_nb, mask_nb_pad}, s, b) ==
               pooled_bias_gen::kBlockTokens;
    };

    // A pass's Q blocks are read once, on its first active ring iteration.
    bool q_pushed = false;
    // Pieces this core has pushed through the group stream since launch. The producer's delivered count and the
    // consumers' credits are running totals of it, never reset in a launch: every member streams the same pieces in
    // every pass, so the totals stay equal across passes.
    [[maybe_unused]] uint32_t res_group_pieces = 0;

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    // The ring order before any iteration, replayed by the passes after the first.
    const auto ring_seq_start = fused_op_receiver.seq;
    auto arrival_preview = fused_op_receiver.seq;
    const ring_joint::PackedKVSchedule packed_schedule = ring_joint::packed_kv_schedule(
        GROUPED_KV_SOURCE_COUNT,
        ring_size,
        kv_local_padded_Nt,
        logical_nt,
        active_ring_iter_mask,
        kv_region_Nt,
        Sk_chunk_t,
        /*has_sliding_window=*/false,
        [&] {
            return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                arrival_preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
        });
    const uint32_t source_group_size = packed_schedule.source_group_size;
    // Grouped K/V sources are statically off (GROUPED_KV_SOURCE_COUNT == 1, asserted above); the constexpr guard lets
    // the compiler drop the packed branches, as the dense reader does.
    constexpr bool packed_kv_enabled = GROUPED_KV_SOURCE_COUNT > 1;
    const bool packed_sources = packed_kv_enabled && packed_schedule.packed();
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        // This pass's blocks: the core's [q_pass, q_pass + 1) share of [global_q_start, global_q_end).
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        auto ring_seq_replay = ring_seq_start;
        q_pushed = false;
        pooled_pending = true;
        for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
            uint32_t packed_source_ids[GROUPED_KV_SOURCE_COUNT];
            ring_joint::PackedKVSourceReadiness packed_readiness;
            if (packed_sources) {
                // Planning identities must not wait for every member. Compute and
                // writer also derive this order without gather waits; this reader
                // alone gates access to the corresponding gathered source data.
                auto preview = fused_op_receiver.seq;
                for (uint32_t source = 0; source < source_group_size; ++source) {
                    packed_source_ids[source] =
                        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                            preview.get_next_ring_id([](uint32_t, uint32_t) {}),
                            mesh_rows,
                            mesh_cols,
                            snake_orientation);
                }
            }
            const auto wait_packed_source = [&](uint32_t source) {
                const uint32_t ready_id =
                    ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                        fused_op_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
                ASSERT(ready_id == packed_source_ids[source]);
            };
            const bool ring_iter_is_active = packed_sources || ((active_ring_iter_mask >> ring_iter) & 1u) != 0;
            const uint32_t ring_id = [&]() -> uint32_t {
                if (q_pass > 0) {
                    // Every shard arrived (and was read) in the first pass: replay the ring order without waits.
                    return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                        ring_seq_replay.get_next_ring_id([](uint32_t, uint32_t) {}),
                        mesh_rows,
                        mesh_cols,
                        snake_orientation);
                }
                SPARSE_ZONE("RD_SYNC");  // the wait for this iteration's all-gather arrival signal
                return packed_sources
                           ? packed_source_ids[0]
                           : ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                                 fused_op_receiver.get_next_ring_id_and_sync(),
                                 mesh_rows,
                                 mesh_cols,
                                 snake_orientation);
            }();
            // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
            // still advances above so reader stays aligned with compute, writer, and all-gather.
            if (!ring_iter_is_active) {
                continue;
            }

            const uint32_t do_pooled = (pooled_route && pooled_pending) ? 1u : 0u;
            // One pass of each K/V piece per ring iteration, applied by compute to every resident Q chunk
            // (sdpa_ring_resident). This reader:
            //   1. on the pass's first active iteration reads the pass's Q chunks into cb_q_in, once for the pass;
            //   2. per head the core's chunks fall in: the piece union of the head group, then one plan record per
            //      resident chunk [this shard's kept-block words, union piece bits lo, hi, do_pooled]; a piece is kept
            //      by a chunk when the chunk keeps any block of it, and compute steps only on those blocks;
            //   3. streams the shard's pieces once each, heads interleaved per k in head order, skipping a piece no
            //      member of the head group needs (compute mirrors both rules);
            //   4. on the pass's first active iteration the pooled K/V chunks once per head, then one bias row per
            //   resident
            //      chunk of that head -- K and V before the biases, so compute's first step on the entry never waits
            //      for V behind a bias row the reader could not push yet.
            constexpr uint32_t kResMax = 32;
            const uint32_t n_res = pass_q_end - pass_q_start;
            ASSERT(n_res <= resident_slots);
            uint32_t res_nb[kResMax];
            uint32_t res_nq[kResMax];
            uint32_t res_qc[kResMax];
            for (uint32_t i = 0; i < n_res; ++i) {
                const auto d = decompose_global_q_index(pass_q_start + i, num_q_chunks, NH, use_zigzag_balancing);
                res_nb[i] = d.nb;
                res_nq[i] = d.nq;
                res_qc[i] = d.q_chunk;
            }
            // 1. Q, once per pass. The reserve waits until compute has popped the previous pass's Q.
            if (!q_pushed) {
                SPARSE_ZONE("RD_RES_Q");
                for (uint32_t i = 0; i < n_res; ++i) {
                    const Slice q_slice(
                        res_nb[i], res_nq[i], res_qc[i] * Sq_chunk_t, res_qc[i] * Sq_chunk_t + Sq_chunk_t, 0, DHt);
                    read_block(
                        q_generator, q_slice, q_local_padded_Nt, cb_q_in, q_tile_bytes, false, q_barrier_threshold);
                }
                if constexpr (num_q_passes > 1) {
                    // Fill cb_q_in to its capacity (resident_slots blocks) so compute pops it whole at the end of the
                    // pass and every pass's Q starts at the buffer's base: compute addresses Q block i at i blocks past
                    // the read pointer, which must not wrap.
                    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
                    const uint32_t pad_tiles = (resident_slots - n_res) * q_chunk_tiles;
                    if (pad_tiles > 0) {
                        CircularBuffer cb_q(cb_q_in);
                        cb_q.reserve_back(pad_tiles);
                        cb_q.push_back(pad_tiles);
                    }
                }
                q_pushed = true;
            }
            // 2. Kept bits and plans. The plan page's scratch (past the 64-byte header) holds one chunk row at a time:
            //    the whole block-form row (n_blocks_pad words, every shard), which the bias generator needs too.
            CircularBuffer cb_plan(cb_sparse_plan);
            cb_plan.reserve_back(1);
            uint32_t scratch = cb_plan.get_write_ptr() + 64;
            const auto read_chunk_row = [&](uint32_t nq, uint32_t qc) {
                noc.async_read(
                    mask_reader,
                    CoreLocalMem<uint32_t>(scratch),
                    mask_row_bytes,
                    {.page_id = nq * mask_nqb_pad + qc},
                    {});
                noc.async_read_barrier();
            };
            // This shard's slice of a row (nb_pad words from block ring_id * nb_pad) into the scratch: all the bit
            // extraction needs, and 8x less than the whole row for every member row on the chain's critical path.
            const auto read_shard_slice = [&](uint32_t nq, uint32_t qc) {
                noc.async_read(
                    mask_reader,
                    CoreLocalMem<uint32_t>(scratch),
                    mask_nb_pad * sizeof(uint32_t),
                    {.page_id = nq * mask_nqb_pad + qc, .offset_bytes = ring_id * mask_nb_pad * sizeof(uint32_t)},
                    {});
                noc.async_read_barrier();
            };
            // This shard's kept FULL blocks of the slice in the scratch, as bits (block b: word b/32, bit b%32).
            // Block 0 forced: every (Q chunk, shard) computes at least one block, compute creates each chunk's state
            // on piece 0, and the pre-pass forces the same block (the bias row is patched to match below).
            constexpr uint32_t kBlockWords = plan_block_words;
            constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / pooled_bias_gen::kBlockTiles;
            uint32_t full_bits[kBlockWords];  // the shard's full blocks, once per ring iteration
            for (uint32_t w = 0; w < kBlockWords; ++w) {
                full_bits[w] = 0;
            }
            for (uint32_t b = 0; b < mask_nb; ++b) {
                if (block_is_full(ring_id, b)) {
                    full_bits[b >> 5] |= 1u << (b & 31);
                }
            }
            const auto row_block_bits = [&](uint32_t* words) {
                CoreLocalMem<volatile uint32_t> row(scratch);
                for (uint32_t w = 0; w < kBlockWords; ++w) {
                    words[w] = 0;
                }
                for (uint32_t b = 0; b < mask_nb; ++b) {
                    if (row[b] != 0) {
                        words[b >> 5] |= 1u << (b & 31);
                    }
                }
                for (uint32_t w = 0; w < kBlockWords; ++w) {
                    words[w] &= full_bits[w];
                }
                words[0] |= 1u;
            };
            // Piece k is kept when any of its blocks is: the streaming / chain-union granularity.
            const auto piece_bits = [&](const uint32_t* words, uint32_t& lo, uint32_t& hi) {
                lo = 0;
                hi = 0;
                for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                    bool kept = false;
                    for (uint32_t j = 0; j < kBlocksPerPiece && !kept; ++j) {
                        const uint32_t b = k * kBlocksPerPiece + j;
                        kept = b < mask_nb && ((words[b >> 5] >> (b & 31)) & 1u) != 0;
                    }
                    if (kept) {
                        if (k < 32) {
                            lo |= 1u << k;
                        } else {
                            hi |= 1u << (k - 32);
                        }
                    }
                }
            };
            const auto row_kept_bits = [&](uint32_t& lo, uint32_t& hi) {
                uint32_t words[kBlockWords];
                row_block_bits(words);
                piece_bits(words, lo, hi);
            };
            uint32_t res_lo[kResMax];
            uint32_t res_hi[kResMax];
            uint32_t seg_ulo[kResMax];
            uint32_t seg_uhi[kResMax];
            // One-page plan (plan_rec_base != 0): every chunk's record [kept-block words][union lo, hi]
            // [do_pooled] in the one plan page, from word rec_base (past the shard slice the scratch reads), pushed
            // once per iteration. The per-chunk pages made the reader wait for compute to pop each one, so the next
            // iteration's plan could not go out before compute had finished this iteration's pieces.
            constexpr uint32_t kPlanRec = kBlockWords + 3;
            [[maybe_unused]] const auto plan_rec = [&](uint32_t i) {
                return CoreLocalMem<volatile uint32_t>(
                    cb_plan.get_write_ptr() + (plan_rec_base + i * kPlanRec) * sizeof(uint32_t));
            };
            {
                SPARSE_ZONE("RD_RES_BITS");  // mask rows of this core's chunks and of every chain member's
                for (uint32_t i = 0; i < n_res; ++i) {
                    read_shard_slice(res_nq[i], res_qc[i]);
                    if constexpr (plan_rec_base != 0) {
                        // One read per chunk: the block words go straight into the chunk's record.
                        uint32_t words[kBlockWords];
                        row_block_bits(words);
                        auto rec = plan_rec(i);
                        for (uint32_t w = 0; w < kBlockWords; ++w) {
                            rec[w] = words[w];
                        }
                        piece_bits(words, res_lo[i], res_hi[i]);
                    } else {
                        row_kept_bits(res_lo[i], res_hi[i]);
                    }
                }
            }
            // Head segments and their chain-wide unions. The chain members cover the head this core participates
            // in; a segment of another head is a non-participant there (it reads its own K/V), so union = own.
            for (uint32_t i0 = 0; i0 < n_res;) {
                uint32_t i1 = i0;
                uint32_t ulo = 0;
                uint32_t uhi = 0;
                while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                    ulo |= res_lo[i1];
                    uhi |= res_hi[i1];
                    ++i1;
                }
                if (group_n_members > 0 && head_cfg.participates && res_nq[i0] == head_cfg.head) {
                    // Chain / group members stream every piece that holds a full block of this shard (block 0
                    // forced, as in row_block_bits): a superset of the members' union that needs no mask reads (the
                    // union read every member's row per ring iteration, ~200 us while compute idled, and at these
                    // densities it is nearly every piece anyway). The same on every member (it depends on the shard
                    // only) and inside every live piece, so compute agrees; it skips a piece it keeps no block of.
                    uint32_t words[kBlockWords];
                    for (uint32_t w = 0; w < kBlockWords; ++w) {
                        words[w] = full_bits[w];
                    }
                    words[0] |= 1u;
                    piece_bits(words, ulo, uhi);
                }
                for (uint32_t i = i0; i < i1; ++i) {
                    seg_ulo[i] = ulo;
                    seg_uhi[i] = uhi;
                }
                i0 = i1;
            }
            if constexpr (plan_rec_base != 0) {
                SPARSE_ZONE("RD_RES_PLAN");
                for (uint32_t i = 0; i < n_res; ++i) {
                    auto rec = plan_rec(i);
                    rec[kBlockWords] = seg_ulo[i];
                    rec[kBlockWords + 1] = seg_uhi[i];
                    rec[kBlockWords + 2] = do_pooled;
                }
                cb_plan.push_back(1);
            } else {
                SPARSE_ZONE("RD_RES_PLAN");  // long when compute is behind: the reserve waits for the previous plan pop
                for (uint32_t i = 0; i < n_res; ++i) {
                    if (i > 0) {
                        cb_plan.reserve_back(1);  // one page: waits for compute to pop the previous plan
                        scratch = cb_plan.get_write_ptr() + 64;
                    }
                    // The chunk's own block bits, from its slice again (the scratch held the members' since).
                    read_shard_slice(res_nq[i], res_qc[i]);
                    uint32_t words[kBlockWords];
                    row_block_bits(words);
                    CoreLocalMem<volatile uint32_t> plan_words(cb_plan.get_write_ptr());
                    for (uint32_t w = 0; w < kBlockWords; ++w) {
                        plan_words[w] = words[w];
                    }
                    plan_words[kBlockWords] = seg_ulo[i];
                    plan_words[kBlockWords + 1] = seg_uhi[i];
                    plan_words[kBlockWords + 2] = do_pooled;
                    cb_plan.push_back(1);
                }
            }
            // 3. The shard's pieces, once each; heads interleaved per k in head order.
            const bool source_is_local = ring_id == ring_index;
            const uint32_t source_start_tile = ring_id * kv_local_padded_Nt;
            const uint32_t source_valid_kv_tiles =
                logical_nt > source_start_tile ? std::min(logical_nt - source_start_tile, kv_local_padded_Nt) : 0;
            const uint32_t end_seq_tile =
                source_is_local ? source_valid_kv_tiles : source_start_tile + source_valid_kv_tiles;
            // Chain overlap. The chain protocol allows one outstanding transfer per LINK: a member signals
            // "ready", the upstream writes and flags valid. In MHA the K and V transfers of a piece go over the
            // same link (k_chain and v_chain are the same head_chain, one semaphore set), so the link's handshakes
            // form one sequence: K(p), V(p), K(p+1), ... Signalling each only when the previous forward was done
            // made the chain a bucket brigade, every hop idle while its neighbours moved: a profile measured
            // 25-28 us per piece against ~13 us for a core fetching from DRAM on its own. Now a member signals the
            // link's NEXT handshake before forwarding the current chunk, whenever that chunk's page is already
            // free: V(p)'s ready before forwarding K(p), K(p+1)'s ready before forwarding V(p). The upstream's
            // write then overlaps this core's forward, and the link never has more than one transfer outstanding
            // (the next signal always follows the wait for the previous data). An armed K page carries to the
            // next stream_piece call; nothing is armed after the iteration's last piece (the pooled fetches
            // reserve pages themselves). Fetchers (injectors, standalone two-head cores) signal
            // nothing and only gain the early reserve.
            bool k_armed = false;
            uint32_t k_armed_addr = 0;
            const auto stream_piece = [&](uint32_t nb, uint32_t nq, uint32_t k, bool has_next) {
                const uint32_t nk = nq / q_heads_per_k;
                const uint32_t nv = nq / q_heads_per_v;
                const uint32_t src_batch = nb;
                const uint32_t row0 = (source_is_local ? 0 : source_start_tile) + k * Sk_chunk_t;
                const Slice k_slice(src_batch, nk, row0, row0 + Sk_chunk_t, 0, DHt);
                // Chain scope keys: the query head for MHA chains, the KV head for grouped ones (as the dense loop).
                const uint32_t k_head = gqa_grouped_kv ? nk : nq;
                const bool k_rx = k_chain.should_receive(k_head);
                const bool v_rx = v_chain.should_receive(nv);
                CircularBuffer cb_k(cb_k_in);
                uint32_t k_addr;
                if (k_armed) {
                    // Armed pages are only created for chain-received pieces, and a chain member receives every
                    // piece of its stream (cores whose chunks span two heads are kept off the chains), so an armed
                    // page always meets a chain piece: its ready signal has already been sent.
                    ASSERT(k_rx);
                    k_addr = k_armed_addr;
                    k_armed = false;
                } else {
                    cb_k.reserve_back(k_chunk_tiles);
                    k_addr = cb_k.get_write_ptr();
                    if (k_rx) {
                        k_chain.signal_ready(noc);
                    }
                }
                if (k_rx) {
                    SPARSE_ZONE(
                        "RD_CHAIN_RX");  // member: the wait for the upstream write (the ready signal went early)
                    k_chain.wait_data();
                } else if (source_is_local) {
                    SPARSE_ZONE("RD_FETCH_K");
                    fetch_block(local_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                } else {
                    SPARSE_ZONE("RD_FETCH_K");
                    fetch_block(gathered_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                }
                // Push before the forward: the forward only reads the page, its NoC write is flushed inside
                // forward(), and the page is not reused before piece p+2's reserve.
                cb_k.push_back(k_chunk_tiles);
                // The link's next handshake is V(p)'s: arm it (reserve V's page, signal ready) before forwarding
                // K(p) when the page is free now; otherwise reserve it after the forward, as before.
                const Slice v_slice(src_batch, nv, row0, row0 + Sk_chunk_t, v_col_offset_t, v_col_offset_t + vDHt);
                CircularBuffer cb_v(cb_v_in);
                uint32_t v_addr = 0;
                bool v_reserved = false;
                if (v_rx && cb_v.pages_reservable_at_back(v_cb_entry_tiles)) {
                    cb_v.reserve_back(v_cb_entry_tiles);
                    v_addr = cb_v.get_write_ptr();
                    v_reserved = true;
                    v_chain.signal_ready(noc);
                }
                if (k_chain.should_forward(k_head, 0)) {
                    SPARSE_ZONE("RD_CHAIN_TX");  // the wait for the downstream ready signal, then the write
                    k_chain.forward(noc, k_addr, k_chunk_tiles, k_tile_bytes);
                }
                if (!v_reserved) {
                    cb_v.reserve_back(v_cb_entry_tiles);
                    v_addr = cb_v.get_write_ptr();
                    if (v_rx) {
                        v_chain.signal_ready(noc);
                    }
                }
                if (v_rx) {
                    v_chain.wait_data();
                } else if (source_is_local) {
                    fetch_block(local_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                } else {
                    fetch_block(gathered_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                }
                cb_v.push_back(v_cb_entry_tiles);
                // The link's next handshake is K(p+1)'s: arm it before forwarding V(p) when there is a next piece
                // and K's page is free now. The armed page is consumed by the next stream_piece call.
                if (k_rx && has_next && cb_k.pages_reservable_at_back(k_chunk_tiles)) {
                    cb_k.reserve_back(k_chunk_tiles);
                    k_armed_addr = cb_k.get_write_ptr();
                    k_armed = true;
                    k_chain.signal_ready(noc);
                }
                if (v_chain.should_forward(nv, 0)) {
                    v_chain.forward(noc, v_addr);
                }
            };
            uint32_t n_pieces = 0;  // pieces streamed this iteration (the first pass counts them)
            // Group delivery. Arguments (after the chain members): role (0 = not in a group: reads its own
            // K/V; 1 = producer; 2 = consumer), producer NoC x, y, receiver count, rectangle count, then per
            // rectangle (x_start, y_start, x_end, y_end, cores), producer only. Every member pushes the same page
            // sequence into cb_k_in / cb_v_in (same pieces, same pooled pages), so
            // the producer's write pointer for a piece is every consumer's page for it.
            //   consumer: reserve the piece's K and V pages, credit the producer (+1), wait until the delivered
            //     count reaches the piece, push. At most one credit is outstanding (the next piece's), and never
            //     past the iteration's last piece (the pooled pages after it are this core's own).
            //   producer: reserve its own pages, read K and V from DRAM, push them for its compute, wait for
            //     receivers x (piece + 1) credits, multicast K and V to every rectangle, then the new count.
            // The credits share one counter, so the sum says every consumer has freed its page only because no
            // consumer can credit two pieces ahead: when the producer waits for piece p, every consumer has
            // received at most p - 1 and so credited at most p, and the sum reaches receivers x p only if each one
            // has. With a deeper lookahead, a fast consumer's extra credits would stand in for a slow one's and the
            // multicast would overwrite a page the slow core's compute has not popped. The CB still holds
            // kv_cb_pages pieces: the lookahead bounds only the producer's lead over the slowest receipt.
            [[maybe_unused]] const uint32_t g_base = group_members_base + 2 * group_n_members;
            [[maybe_unused]] const uint32_t g_role = group_mcast ? get_arg_val<uint32_t>(g_base) : 0;
            [[maybe_unused]] uint32_t g_granted = 0;  // pieces credited this iteration
            [[maybe_unused]] uint32_t g_ahead = 0;    // pieces credited, not yet pushed
            const auto g_try_grant = [&](bool blocking) -> bool {
                if (g_granted >= n_pieces) {
                    return false;
                }
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                const uint32_t k_need = k_chunk_tiles * (g_ahead + 1);
                const uint32_t v_need = v_cb_entry_tiles * (g_ahead + 1);
                if (!blocking &&
                    (g_ahead > 0 || !cb_k.pages_reservable_at_back(k_need) || !cb_v.pages_reservable_at_back(v_need))) {
                    return false;
                }
                cb_k.reserve_back(k_need);
                cb_v.reserve_back(v_need);
                ++g_ahead;
                ++g_granted;
                Semaphore<>(head_sender_semaphore_id)
                    .up(noc, get_arg_val<uint32_t>(g_base + 1), get_arg_val<uint32_t>(g_base + 2), 1);
                return true;
            };
            const auto g_grant_ahead = [&]() {
                while (g_try_grant(false)) {
                }
            };
            const auto stream_piece_group = [&](uint32_t nb, uint32_t nq, uint32_t k) {
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                if (g_role == 2) {
                    if (g_ahead == 0) {
                        g_try_grant(true);
                    }
                    {
                        SPARSE_ZONE(
                            "RD_GROUP_RX");  // the wait for the producer's multicast (the credit went out earlier)
                        Semaphore<>(head_receiver_semaphore_id).wait_min(res_group_pieces + 1);
                    }
                    cb_k.push_back(k_chunk_tiles);
                    cb_v.push_back(v_cb_entry_tiles);
                    --g_ahead;
                    ++res_group_pieces;
                    g_grant_ahead();
                    return;
                }
                // Producer.
                const uint32_t nk = nq / q_heads_per_k;
                const uint32_t nv = nq / q_heads_per_v;
                const uint32_t src_batch = nb;
                const uint32_t row0 = (source_is_local ? 0 : source_start_tile) + k * Sk_chunk_t;
                const Slice k_slice(src_batch, nk, row0, row0 + Sk_chunk_t, 0, DHt);
                const Slice v_slice(src_batch, nv, row0, row0 + Sk_chunk_t, v_col_offset_t, v_col_offset_t + vDHt);
                cb_k.reserve_back(k_chunk_tiles);
                cb_v.reserve_back(v_cb_entry_tiles);
                const uint32_t k_addr = cb_k.get_write_ptr();
                const uint32_t v_addr = cb_v.get_write_ptr();
                {
                    SPARSE_ZONE("RD_FETCH_K");
                    if (source_is_local) {
                        fetch_block(local_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                        fetch_block(local_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                    } else {
                        fetch_block(gathered_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                        fetch_block(gathered_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                    }
                }
                // Push first: the multicast only reads the pages, and they are not reserved again before this
                // core's compute has popped them and the writes below are flushed.
                cb_k.push_back(k_chunk_tiles);
                cb_v.push_back(v_cb_entry_tiles);
                ++res_group_pieces;
                const uint32_t n_recv = get_arg_val<uint32_t>(g_base + 3);
                const uint32_t n_rects = get_arg_val<uint32_t>(g_base + 4);
                {
                    SPARSE_ZONE("RD_GROUP_TX");  // the wait for the slowest consumer's credit, then the multicasts
                    Semaphore<>(head_sender_semaphore_id).wait_min(n_recv * res_group_pieces);
                    Semaphore<> count_src(head_valid_semaphore_id);
                    count_src.set(res_group_pieces);
                    for (uint32_t r = 0; r < n_rects; ++r) {
                        const uint32_t a = g_base + 5 + 5 * r;
                        // NoC 1 runs the other way: its rectangle starts at the high corner.
                        const bool rev = noc.get_noc_id() == 1;
                        const uint32_t x0 = get_arg_val<uint32_t>(a + (rev ? 2 : 0));
                        const uint32_t y0 = get_arg_val<uint32_t>(a + (rev ? 3 : 1));
                        const uint32_t x1 = get_arg_val<uint32_t>(a + (rev ? 0 : 2));
                        const uint32_t y1 = get_arg_val<uint32_t>(a + (rev ? 1 : 3));
                        const uint32_t n_dests = get_arg_val<uint32_t>(a + 4);
                        // K and V linked, then the count: back to back, no flush in between (ChainLink::forward).
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(k_addr),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            n_dests,
                            {},
                            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = k_addr},
                            /*linked=*/true);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(v_addr),
                            MulticastEndpoint{},
                            v_cb_entry_tiles * v_tile_bytes,
                            n_dests,
                            {},
                            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = v_addr},
                            /*linked=*/true);
                        count_src.relay_multicast(
                            noc, Semaphore<>(head_receiver_semaphore_id), x0, y0, x1, y1, n_dests, /*linked=*/false);
                    }
                    noc.async_writes_flushed();
                }
            };
            {
                SPARSE_ZONE("RD_RES_PIECES");  // the whole piece stream of this iteration (fetch / chain / CB waits)
                // Two passes over the same (piece, head group) order: a count first, so each streamed piece knows
                // whether another follows in this iteration (the lookahead arms the next page), then the stream.
                uint32_t j = 0;
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    if constexpr (group_mcast) {
                        if (pass == 1 && g_role == 2) {
                            g_grant_ahead();  // the first pieces' credits, before the first wait
                        }
                    }
                    for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                        if (!chunked_kv_chunk_is_live<
                                false,  // no KV-pad rotation
                                chunked_enabled,
                                kv_local_padded_Nt,
                                chunk_size_t,
                                kv_region_Nt>(ring_id, k, Sk_chunk_t, logical_nt, logical_nt)) {
                            continue;
                        }
                        for (uint32_t i0 = 0; i0 < n_res;) {
                            uint32_t i1 = i0;
                            while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                                ++i1;
                            }
                            const bool union_kept =
                                k < 32 ? ((seg_ulo[i0] >> k) & 1u) != 0 : ((seg_uhi[i0] >> (k - 32)) & 1u) != 0;
                            if (union_kept) {
                                if (pass == 0) {
                                    ++n_pieces;
                                } else {
                                    ++j;
                                    if constexpr (group_mcast) {
                                        // A group serves one head; a member's chunks never span two (factory).
                                        if (g_role != 0) {
                                            stream_piece_group(res_nb[i0], res_nq[i0], k);
                                        } else {
                                            stream_piece(res_nb[i0], res_nq[i0], k, j < n_pieces);
                                        }
                                    } else {
                                        stream_piece(res_nb[i0], res_nq[i0], k, j < n_pieces);
                                    }
                                }
                            }
                            i0 = i1;
                        }
                    }
                }
            }
            // 4. Pooled chunks, once per head, then one bias row per resident chunk of the head (pooled route only).
            if constexpr (pooled_route) {
                if (do_pooled) {
                    const auto pooled_k_generator = PaddedAddrGenerator(
                        TensorAccessor(pooled_k_args, pooled_k_addr),
                        TensorTileShape(1, NH, mask_n_blocks_pad / 32, DHt));
                    const auto pooled_v_generator = PaddedAddrGenerator(
                        TensorAccessor(pooled_v_args, pooled_v_addr),
                        TensorTileShape(1, NH, mask_n_blocks_pad / 32, vDHt));
                    SPARSE_ZONE("RD_RES_POOLED");
                    if constexpr (plan_rec_base != 0) {
                        // The row reads below fill the whole scratch, records included: take the page back first,
                        // which waits until compute has popped (read) this iteration's plan. Short iterations
                        // can push every piece before compute starts. The next iteration's reserve returns it.
                        cb_plan.reserve_back(1);
                        scratch = cb_plan.get_write_ptr() + 64;
                    }
                    constexpr uint32_t pooled_rows_t = mask_n_blocks_pad / 32;
                    const pooled_bias_gen::Geometry geo{
                        kv_local_padded_Nt * 32, logical_n_tokens, mask_nb, mask_nb_pad};
                    for (uint32_t i0 = 0; i0 < n_res;) {
                        uint32_t i1 = i0;
                        while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                            ++i1;
                        }
                        const uint32_t nq = res_nq[i0];
                        for (uint32_t pc = 0; pc < pooled_chunks; ++pc) {
                            const uint32_t tiles = pc + 1 == pooled_chunks ? pooled_last_tiles : Sk_chunk_t;
                            const uint32_t row0 = pc * Sk_chunk_t;
                            CircularBuffer cb_k(cb_k_in);
                            cb_k.reserve_back(k_chunk_tiles);
                            pooled_k_generator.issue_reads(
                                Slice(0, nq, row0, row0 + tiles, 0, DHt),
                                pooled_rows_t,
                                cb_k_in,
                                cb_k.get_write_ptr(),
                                k_tile_bytes,
                                k_tile_bytes * Sk_chunk_t,
                                0);
                            noc.async_read_barrier();
                            noc.write_zeros_l1_barrier();
                            cb_k.push_back(k_chunk_tiles);
                            CircularBuffer cb_v(cb_v_in);
                            cb_v.reserve_back(v_cb_entry_tiles);
                            pooled_v_generator.issue_reads(
                                Slice(0, nq, row0, row0 + tiles, 0, vDHt),
                                pooled_rows_t,
                                cb_v_in,
                                cb_v.get_write_ptr(),
                                vDHt * v_tile_bytes,
                                v_tile_bytes,
                                0);
                            noc.async_read_barrier();
                            noc.write_zeros_l1_barrier();
                            cb_v.push_back(v_cb_entry_tiles);
                            for (uint32_t i = i0; i < i1; ++i) {
                                // The chunk's own row back into the plan scratch: compute reads only the header
                                // words of a plan page, and every page has been popped by now.
                                read_chunk_row(nq, res_qc[i]);
                                // Block 0 of every live shard is forced exact by every ring iteration's bits (above),
                                // so mark it exact in the row the bias is generated from (no double counting).
                                {
                                    CoreLocalMem<volatile uint32_t> row(scratch);
                                    for (uint32_t s = 0; s < ring_size; ++s) {
                                        if (block_is_full(s, 0)) {
                                            row[s * mask_nb_pad] = 1u;
                                        }
                                    }
                                }
                                CircularBuffer cb_bias(cb_pooled_bias);
                                cb_bias.reserve_back(Sk_chunk_t);
                                // Block-form row: exact iff the block's own bit is set on a full block.
                                pooled_bias_gen::generate_tiles<pooled_bias_tile_bytes>(
                                    noc,
                                    cb_bias.get_write_ptr(),
                                    row0,
                                    tiles,
                                    geo,
                                    CoreLocalMem<volatile uint32_t>(scratch),
                                    pooled_bias_table_base);
                                cb_bias.push_back(Sk_chunk_t);
                            }
                        }
                        i0 = i1;
                    }
                }
            }

            pooled_pending = false;
            if (packed_sources) {
                // Cores without Q work still consume the group in sequencer order,
                // so the next group and subsequent dispatches start at the right source.
                packed_readiness.drain(source_group_size, wait_packed_source);
            }
        }
    }
}
