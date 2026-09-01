// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "dataflow_common.hpp"
#include "exp_fused_op_indexer.hpp"
#include "metadata_scalar_read.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

void kernel_main() {
    Noc noc;
    noc.async_write_barrier();
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(5);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t logical_n_ct = get_compile_time_arg_val(7);
    constexpr uint32_t logical_nt_ct = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t ring_size = get_compile_time_arg_val(16);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(17);

    constexpr auto q_args = TensorAccessorArgs<18>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gathered_k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gathered_v_args = TensorAccessorArgs<gathered_k_args.next_compile_time_args_offset()>();
    constexpr auto joint_q_args = TensorAccessorArgs<gathered_v_args.next_compile_time_args_offset()>();
    constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
    constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_v_addr = get_arg_val<uint32_t>(argidx++);
    // Head-serial passes: this core owns flat Q chunks q_base + p * q_stride for p in [0, q_count),
    // i.e. one chunk of head (p * grid_rows + my_row) per pass. See the program factory.
    const uint32_t q_base = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_stride = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_count = get_arg_val<uint32_t>(argidx++);

    const uint32_t is_chain_participant = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_injector = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink = get_arg_val<uint32_t>(argidx++);
    argidx += 4;  // skip chain batch/head/chunk_start/count (unused)
    const uint32_t prev_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_y = get_arg_val<uint32_t>(argidx++);
    argidx++;  // skip next_core_q_chunks (unused)
    const uint32_t mcast_num_dests = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_sender_wait = get_arg_val<uint32_t>(argidx++);

    const uint32_t is_mux_writer = get_arg_val<uint32_t>(argidx++);

    // Per-link semaphore addresses for chunk-level sync.
    // Kept as raw L1 pointers (not Semaphore<>) because they're passed as L1 addresses via RT args.
    const uint32_t num_links = get_arg_val<uint32_t>(argidx++);
    volatile tt_l1_ptr uint32_t* per_link_sem_ptrs[2] = {nullptr, nullptr};
    for (uint32_t lnk = 0; lnk < num_links; ++lnk) {
        per_link_sem_ptrs[lnk] =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(argidx++));
    }

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    // Split-head forwarding dedup descriptor (meaningful on injector cores only):
    // 0 = none, 1 = leader (gate on fabric, relay each gate pass to the buddy injector),
    // 2 = follower (this row's remote twin forwards nothing; gate on the leader's relay).
    const uint32_t dedup_role = get_arg_val<uint32_t>(argidx++);
    const uint32_t buddy_injector_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t buddy_injector_y = get_arg_val<uint32_t>(argidx++);

    // After fused-op receiver consumed its runtime args, remaining RT args are S&F chain metadata

    // Compile-time semaphore ids and mcast flag are appended after all TensorAccessorArgs().
    // Semaphore<> wrapper resolves IDs to L1 addrs internally.
    const uint32_t sender_semaphore_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset());
    const uint32_t receiver_semaphore_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 1);
    const uint32_t valid_semaphore_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 2);
    constexpr bool mcast_enabled = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 3) == 1;
    // Streamed Q (host fallback when resident Q does not fit L1): cb_q_in holds one chunk, so each
    // pass's Q is re-read every ring iteration and compute pops it at the end of the pass.
    constexpr bool stream_q = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 4) == 1;
    // Ping-pong valid flag: K broadcasts land on receiver_semaphore_id, V broadcasts on this one.
    // Separate flags per channel let a receiver post its credit (reserve + flag reset + ack) for
    // chunk n+1 immediately after consuming chunk n, so the injector never waits a post-receipt
    // round trip between consecutive mcasts (V of chunk n and K of chunk n+1 overlap in flight).
    const uint32_t receiver_semaphore_b_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 5);
    // V-channel ack counter, separate from the K-channel counter (sender_semaphore_id). K and V
    // ack totals MUST be counted independently: K credits run one chunk ahead of V credits, so a
    // combined total would let fast receivers' K(n+1) credits mask a slow receiver's missing V(n)
    // credit, releasing the V(n) mcast before that receiver reset its V flag (lost VALID ->
    // deadlock). Per channel, a receiver is at most one credit ahead — exactly the awaited event.
    const uint32_t sender_semaphore_v_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 6);
    // Split-head forwarding dedup buddy gate (see the AG gate below).
    const uint32_t buddy_gate_semaphore_id = get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 7);

    // When set, logical_n_ct/logical_nt_ct are worst-case placeholders; the live value is read below.
    constexpr bool has_logical_n_tensor =
        get_compile_time_arg_val(joint_v_args.next_compile_time_args_offset() + 8) == 1;
    constexpr auto logical_n_args = TensorAccessorArgs<joint_v_args.next_compile_time_args_offset() + 9>();

    constexpr uint32_t cb_derived = tt::CBIndex::c_13;

    // Read the live length ONCE, before the ring loop, into locals. Every logical_n-dependent quantity
    // below derives from these — including the chunk-skip predicate that sets this kernel's credit caps
    // and the injector's per-link gate demand. A mid-loop re-read could observe a host rewrite and
    // desynchronize those counts from the writer's, which forwards on the same predicate.
    uint32_t logical_n = logical_n_ct;
    uint32_t logical_nt = logical_nt_ct;
    [[maybe_unused]] uint32_t global_n_partial_col_live = 0;
    if constexpr (has_logical_n_tensor) {
        // Borrow cb_q_in's L1 as read scratch: this runs before any Q is fetched into it.
        logical_n = trace_metadata::read_metadata_scalar_u32(
            noc, logical_n_args, get_common_arg_val<uint32_t>(0), CircularBuffer(tt::CBIndex::c_0).get_write_ptr());
        logical_nt = ring_joint::tiles_for(logical_n);
        global_n_partial_col_live = ring_joint::tile_partial_col(logical_n);

        // Hand compute the values it cannot read itself (compute RISCs cannot NoC-read DRAM).
        CircularBuffer cb_derived_obj(cb_derived);
        cb_derived_obj.reserve_back(1);
        CoreLocalMem<volatile uint32_t> d(cb_derived_obj.get_write_ptr());
        d[ring_joint::kDerivedLogicalNt] = logical_nt;
        d[ring_joint::kDerivedGlobalNPartialCol] = global_n_partial_col_live;
        cb_derived_obj.push_back(1);
    }

    // Receiver flips this to INVALID before each wait; initialize so the first iteration sees it as VALID.
    Semaphore<>(valid_semaphore_id).set(VALID);

    // Injector-side cumulative ack target for the ping-pong protocol. Receivers only ever
    // increment sender_semaphore (one ack per K or V chunk credit); the injector waits for the
    // running total instead of wait(N)+set(0), which would race with credits pre-posted for the
    // next chunk. Reset to 0 at kernel end for cached-program reruns.
    [[maybe_unused]] uint32_t mcast_k_acks_expected = 0;
    [[maybe_unused]] uint32_t mcast_v_acks_expected = 0;

    uint32_t sender_wait_count = 1;
    if constexpr (mcast_enabled) {
        if (is_injector) {
            sender_wait_count = mcast_sender_wait;
        }
    }

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_k_writer_in = tt::CBIndex::c_14;
    constexpr uint32_t cb_v_writer_in = tt::CBIndex::c_15;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto local_v_reader = TensorAccessor(v_args, v_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr);
    const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr);
    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr);
    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr);

    const auto input_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto gathered_kv_input_tile_logical = TensorTileShape(B, NH, padded_Nt, DHt);
    const auto joint_input_tile_logical = TensorTileShape(B, NH, Lt, DHt);

    const auto q_generator = PaddedAddrGenerator(q_reader, input_tile_logical);
    const auto local_k_generator = PaddedAddrGenerator(local_k_reader, input_tile_logical);
    const auto local_v_generator = PaddedAddrGenerator(local_v_reader, input_tile_logical);
    const auto gathered_k_generator = PaddedAddrGenerator(gathered_k_reader, gathered_kv_input_tile_logical);
    const auto gathered_v_generator = PaddedAddrGenerator(gathered_v_reader, gathered_kv_input_tile_logical);
    const auto joint_q_generator = PaddedAddrGenerator(joint_q_reader, joint_input_tile_logical);
    const auto joint_k_generator = PaddedAddrGenerator(joint_k_reader, joint_input_tile_logical);
    const auto joint_v_generator = PaddedAddrGenerator(joint_v_reader, joint_input_tile_logical);

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_n / tt::constants::TILE_HEIGHT, L);

    // Number of Q chunks already pushed to cb_q_in. Q is identical across ring iterations; with
    // resident Q each pass reads its head's chunk exactly once (on the first active ring iteration)
    // and all q_count chunks stay resident until compute pops them after the last pass of the last
    // iteration. With streamed Q (stream_q) every pass re-reads its chunk every active iteration.
    uint32_t q_chunks_pushed = 0;

    uint32_t chunks_signaled_by_remote = 0;

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // find out which is the latest ring_id that synchronized
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        // Iterate over KV blocks gathered on ring.
        // Only the last ring ID will append joint_K, joint_V to K, V.
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;  // Floor division to get tile ID
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);

        const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);

        if (!ring_iter_does_work) {
            continue;
        }

        // Non-skipped KV chunk count this ring iteration (identical for every pass — the skip
        // predicate depends only on ring_id). Caps the ping-pong credit pre-posting so a receiver
        // never acks a chunk the injector will not send: a stale credit would let a later mcast
        // bypass its receiver handshake and clobber an unconsumed CB slot.
        uint32_t chunks_this_iter = 0;
        for (uint32_t kc = 0; kc < num_kv_chunks; ++kc) {
            const bool kc_is_joint = kc >= num_local_k_chunks;
            const uint32_t kc_global_start_tile = local_padded_Nt * ring_id + kc * Sk_chunk_t;
            if (!kc_is_joint && kc_global_start_tile >= logical_nt) {
                continue;
            }
            chunks_this_iter++;
        }

        // Passes are serial within a ring iteration: pass p attends head (p * rows + my_row) against
        // this iteration's K/V shard. Every core of a row runs the same number of passes in the same
        // order, which is what keeps the row's K/V CB pointers in lockstep for the mcast.
        for (uint32_t pass = 0; pass < q_count; ++pass) {
            const uint32_t global_q_chunk = q_base + pass * q_stride;
            // Counted per pass: compute drains its phase-alignment padding per pass too.
            uint32_t KV_chunks_processed_in_iter = 0;
            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const bool is_joint_q = q_chunk >= num_local_q_chunks;

            Slice q_slice;
            uint32_t q_end_seq_tile;
            if (is_joint_q) {
                // Get row index into the joint Q tensor
                const uint32_t joint_q_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                q_slice = Slice(nb, nq, joint_q_row_start_tile, joint_q_row_start_tile + Sq_chunk_t, 0, DHt);
                q_end_seq_tile = Lt;
            } else {
                // Index into the Q input tensor
                q_slice = Slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);
                q_end_seq_tile = local_padded_Nt;
            }

            // Every chunk this core owns is on its row's chain, so participation alone decides.
            const bool should_forward = is_chain_participant && !is_sink;
            const bool should_receive = is_chain_participant && !is_injector;

            // Ping-pong receive credits (mcast path only). A credit = reserve the next chunk's CB
            // slot, reset that channel's valid flag, then ack the injector. Order matters: the
            // reset must precede the ack, because the injector's mcast for that chunk (data +
            // valid relay) is released by the ack — resetting after could erase an already-landed
            // VALID. The reserve must precede the ack so the mcast has a landing slot; it also
            // provides compute-paced backpressure (reserving chunk n+1 waits for chunk n-1's pop).
            // Credits are capped at chunks_this_iter so the ack total exactly matches the
            // injector's cumulative expectation.
            uint32_t k_credits = 0;
            uint32_t v_credits = 0;
            auto post_k_credit = [&]() {
                if (k_credits < chunks_this_iter) {
                    CircularBuffer(cb_k_in).reserve_back(k_chunk_tiles);
                    if (is_mux_writer) {
                        CircularBuffer(cb_k_writer_in).reserve_back(k_chunk_tiles);
                    }
                    Semaphore<>(receiver_semaphore_id).set(INVALID);
                    Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
                    k_credits++;
                }
            };
            auto post_v_credit = [&]() {
                if (v_credits < chunks_this_iter) {
                    CircularBuffer(cb_v_in).reserve_back(v_chunk_tiles);
                    if (is_mux_writer) {
                        CircularBuffer(cb_v_writer_in).reserve_back(v_chunk_tiles);
                    }
                    Semaphore<>(receiver_semaphore_b_id).set(INVALID);
                    Semaphore<>(sender_semaphore_v_id).up(noc, prev_physical_x, prev_physical_y, 1);
                    v_credits++;
                }
            };
            if constexpr (mcast_enabled) {
                if (should_receive) {
                    // Prime the pipeline: credits for the first K and V chunks of this pass.
                    post_k_credit();
                    post_v_credit();
                }
            }

            // Resident Q: read this pass's chunk exactly once, on the first active ring iteration.
            // Streamed Q: read it every pass, every active iteration (the reserve blocks until
            // compute's pass-end pop frees the single slot — a bounded stall, never a deadlock).
            const bool need_q_read = stream_q || (q_chunks_pushed <= pass);

            for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
                /**
                 * Iterate over all KV chunks for this Q chunk.
                 * If this is the last ring ID, we will also read from joint KV.
                 * If this k chunk is in the spatial input and beyond the logical N, we will skip it.
                 */
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                // Global index into the padded KV tensor
                const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                const bool kv_chunk_is_beyond_logical_n = !kv_chunk_is_joint && (kv_global_start_tile >= logical_nt);

                if (kv_chunk_is_beyond_logical_n) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }
                KV_chunks_processed_in_iter++;

                Slice kv_slice;
                uint32_t
                    end_seq_tile;  // further information to `read_block` to determine whether it should pad with zeros.

                if (kv_chunk_is_joint) {
                    const uint32_t joint_k_chunk = k_chunk - num_local_k_chunks;
                    const uint32_t joint_k_row_start_tile = joint_k_chunk * Sk_chunk_t;
                    kv_slice = Slice(nb, nq, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, DHt);
                    end_seq_tile = Lt;
                } else {
                    if (ring_iter == 0) {
                        // Local KV
                        const uint32_t local_k_row_start_tile = k_chunk * Sk_chunk_t;
                        kv_slice = Slice(nb, nq, local_k_row_start_tile, local_k_row_start_tile + Sk_chunk_t, 0, DHt);
                        end_seq_tile = std::min(logical_nt, local_padded_Nt);
                    } else {
                        // Gathered KV
                        const uint32_t gathered_kv_start_tile = ring_iter_kv_start_tile + k_chunk * Sk_chunk_t;
                        kv_slice = Slice(nb, nq, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, DHt);
                        end_seq_tile = std::min(logical_nt, local_padded_Nt * (ring_id + 1));
                    }
                }

                // Per-chunk sync: wait for EACH link's MUX writer to finish writing this chunk
                if (is_injector && ring_iter > 0 && !kv_chunk_is_joint) {
                    chunks_signaled_by_remote++;
                    if (dedup_role == 2) {
                        // Split-head dedup follower: the remote twin of this row forwards nothing.
                        // The leader row of the pair (same device, same head, same deterministic
                        // ring sequence, same gathered region) relays its gate result here.
                        Semaphore<>(buddy_gate_semaphore_id).wait_min(chunks_signaled_by_remote);
                    } else {
                        for (uint32_t lnk = 0; lnk < num_links; ++lnk) {
                            noc_semaphore_wait_min(per_link_sem_ptrs[lnk], chunks_signaled_by_remote);
                        }
                        if (dedup_role == 1) {
                            // Leader: this chunk of the shared head is proven present in the
                            // gathered tensor — release the follower row's injector.
                            Semaphore<>(buddy_gate_semaphore_id).up(noc, buddy_injector_x, buddy_injector_y, 1);
                        }
                    }
                }

                // K: get data into CB buffer. On the mcast path a receiver's slot was already
                // reserved when its credit was posted (see post_k_credit), so it only waits for
                // the valid flag here; the injector reserves/fetches as before.
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_k_writer(cb_k_writer_in);
                uint32_t cb_k_start_address = 0;
                bool k_mcast_receive = false;
                if constexpr (mcast_enabled) {
                    k_mcast_receive = should_receive;
                }
                if (k_mcast_receive) {
                    Semaphore<>(receiver_semaphore_id).wait(VALID);
                } else {
                    cb_k.reserve_back(k_chunk_tiles);
                    if (is_mux_writer) {
                        cb_k_writer.reserve_back(k_chunk_tiles);
                    }
                    cb_k_start_address = cb_k.get_write_ptr();
                    if (should_receive) {
                        // Unicast-chain path (mcast disabled): original 1-deep handshake.
                        Semaphore<> receiver_sem(receiver_semaphore_id);
                        receiver_sem.set(INVALID);
                        Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
                        receiver_sem.wait(VALID);
                    } else {
                        fetch_block(
                            kv_chunk_is_joint ? joint_k_generator
                                              : (ring_iter == 0 ? local_k_generator : gathered_k_generator),
                            kv_slice,
                            end_seq_tile,
                            cb_k_in,
                            cb_k_start_address,
                            k_tile_bytes,
                            true /*transpose*/
                        );
                    }
                }

                // Forward K to next core(s) before push_back — prevents compute from
                // popping the buffer while the mcast is still reading from it.
                if (should_forward) {
                    if constexpr (mcast_enabled) {
                        // Receivers pre-post their credit for this chunk right after consuming the
                        // previous K chunk, so this wait is normally already satisfied — the
                        // post-receipt round trip is off the critical path.
                        mcast_k_acks_expected += sender_wait_count;
                        Semaphore<>(sender_semaphore_id).wait_min(mcast_k_acks_expected);
                    } else {
                        Semaphore<> sender_sem(sender_semaphore_id);
                        sender_sem.wait(sender_wait_count);
                        sender_sem.set(0);
                    }
                    if constexpr (mcast_enabled) {
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_k_start_address),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            mcast_num_dests,
                            {},
                            {.noc_x_start = prev_physical_x,
                             .noc_y_start = prev_physical_y,
                             .noc_x_end = next_physical_x,
                             .noc_y_end = next_physical_y,
                             .addr = cb_k_start_address},
                            true /* linked: semaphore mcast follows */);
                        // Must be back-to-back after the linked data write — any flush between them
                        // deadlocks the linked transaction.
                        Semaphore<>(valid_semaphore_id)
                            .relay_multicast(
                                noc,
                                Semaphore<>(receiver_semaphore_id),
                                prev_physical_x,
                                prev_physical_y,
                                next_physical_x,
                                next_physical_y,
                                mcast_num_dests,
                                /*linked=*/false);
                    } else {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(cb_k_start_address),
                            UnicastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            {},
                            {.noc_x = next_physical_x, .noc_y = next_physical_y, .addr = cb_k_start_address});
                    }
                    noc.async_writes_flushed();
                    if constexpr (!mcast_enabled) {
                        Semaphore<>(valid_semaphore_id)
                            .relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_physical_x, next_physical_y);
                    }
                }

                // Make K available to compute
                cb_k.push_back(k_chunk_tiles);
                if (is_mux_writer) {
                    cb_k_writer.push_back(k_chunk_tiles);
                    ASSERT(cb_k.get_write_ptr() == cb_k_writer.get_write_ptr());
                }
                // Credit the NEXT K chunk now (reserve after push keeps the CB cursor correct),
                // so the injector's next K mcast is released before this core reaches its wait.
                if (k_mcast_receive) {
                    post_k_credit();
                }

                // Download Q on the first K iteration — after K is downloaded and forwarded.
                // Push Q one subblock at a time so compute can start QK matmul incrementally.
                // Placed after K forward so no outstanding NOC writes remain
                // (noc.async_read_barrier inside subblock read would deadlock with in-flight writes).
                if (k_chunk == 0 && need_q_read) {
                    if constexpr (use_q_subblock_push) {
                        const auto& q_gen = is_joint_q ? joint_q_generator : q_generator;
                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                            const uint32_t sb_row_start = q_slice.d2_start + q_sub * qk_subblock_h;
                            const uint32_t sb_row_end = sb_row_start + qk_subblock_h;
                            Slice q_sub_slice(q_slice.d0, q_slice.d1, sb_row_start, sb_row_end, 0, DHt);
                            read_block(
                                q_gen, q_sub_slice, q_end_seq_tile, cb_q_in, q_tile_bytes, false /*transpose*/
                            );
                        }
                    } else {
                        read_block(
                            is_joint_q ? joint_q_generator : q_generator,
                            q_slice,
                            q_end_seq_tile,
                            cb_q_in,
                            q_tile_bytes,
                            false /*transpose*/
                        );
                    }
                    q_chunks_pushed++;
                }

                // V: get data into CB buffer — same ping-pong structure as K, on the second
                // valid flag (receiver_semaphore_b_id).
                CircularBuffer cb_v(cb_v_in);
                CircularBuffer cb_v_writer(cb_v_writer_in);
                uint32_t cb_v_start_address = 0;
                bool v_mcast_receive = false;
                if constexpr (mcast_enabled) {
                    v_mcast_receive = should_receive;
                }
                if (v_mcast_receive) {
                    Semaphore<>(receiver_semaphore_b_id).wait(VALID);
                } else {
                    cb_v.reserve_back(v_chunk_tiles);
                    if (is_mux_writer) {
                        cb_v_writer.reserve_back(v_chunk_tiles);
                    }
                    cb_v_start_address = cb_v.get_write_ptr();
                    if (should_receive) {
                        // Unicast-chain path (mcast disabled): original 1-deep handshake.
                        Semaphore<> receiver_sem(receiver_semaphore_id);
                        receiver_sem.set(INVALID);
                        Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
                        receiver_sem.wait(VALID);
                    } else {
                        fetch_block(
                            kv_chunk_is_joint ? joint_v_generator
                                              : (ring_iter == 0 ? local_v_generator : gathered_v_generator),
                            kv_slice,
                            end_seq_tile,
                            cb_v_in,
                            cb_v_start_address,
                            v_tile_bytes,
                            false /*transpose*/
                        );
                    }
                }

                // Forward V to next core(s) before push_back — prevents compute from
                // popping the buffer while the mcast is still reading from it.
                if (should_forward) {
                    if constexpr (mcast_enabled) {
                        mcast_v_acks_expected += sender_wait_count;
                        Semaphore<>(sender_semaphore_v_id).wait_min(mcast_v_acks_expected);
                    } else {
                        Semaphore<> sender_sem(sender_semaphore_id);
                        sender_sem.wait(sender_wait_count);
                        sender_sem.set(0);
                    }
                    if constexpr (mcast_enabled) {
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_v_start_address),
                            MulticastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            mcast_num_dests,
                            {},
                            {.noc_x_start = prev_physical_x,
                             .noc_y_start = prev_physical_y,
                             .noc_x_end = next_physical_x,
                             .noc_y_end = next_physical_y,
                             .addr = cb_v_start_address},
                            true /* linked: semaphore mcast follows */);
                        // Companion semaphore mcast — see K path above for rationale.
                        // V lands on the second (ping-pong) valid flag.
                        Semaphore<>(valid_semaphore_id)
                            .relay_multicast(
                                noc,
                                Semaphore<>(receiver_semaphore_b_id),
                                prev_physical_x,
                                prev_physical_y,
                                next_physical_x,
                                next_physical_y,
                                mcast_num_dests,
                                /*linked=*/false);
                    } else {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(cb_v_start_address),
                            UnicastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            {},
                            {.noc_x = next_physical_x, .noc_y = next_physical_y, .addr = cb_v_start_address});
                    }
                    noc.async_writes_flushed();
                    if constexpr (!mcast_enabled) {
                        Semaphore<>(valid_semaphore_id)
                            .relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_physical_x, next_physical_y);
                    }
                }

                // Make V available to compute
                cb_v.push_back(v_chunk_tiles);
                if (is_mux_writer) {
                    cb_v_writer.push_back(v_chunk_tiles);
                    ASSERT(cb_v.get_write_ptr() == cb_v_writer.get_write_ptr());
                }
                // Credit the NEXT V chunk (see the K credit above).
                if (v_mcast_receive) {
                    post_v_credit();
                }
            }

            // Phase-alignment padding, per pass. Compute pads once per pass (one sdpa_ring_v2 call
            // per pass, each ending in dummy_kv_chunks_for_phase_alignment), so the reader must pad
            // at the same granularity or the K/V CB phases diverge. The exchange is core-local:
            // reserve/push on our own CBs, no fetch and no mcast.
            if (KV_chunks_processed_in_iter % 2 == 0) {
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                cb_k.reserve_back(k_chunk_tiles);
                cb_v.reserve_back(k_chunk_tiles);
                cb_k.push_back(k_chunk_tiles);
                cb_v.push_back(k_chunk_tiles);
                if (is_mux_writer) {
                    CircularBuffer cb_k_writer(cb_k_writer_in);
                    CircularBuffer cb_v_writer(cb_v_writer_in);
                    cb_k_writer.reserve_back(k_chunk_tiles);
                    cb_v_writer.reserve_back(v_chunk_tiles);
                    cb_k_writer.push_back(k_chunk_tiles);
                    cb_v_writer.push_back(v_chunk_tiles);
                }
            }
        }
    }

    // Reset all per-link out-ready semaphores so they are clean for the next invocation
    if (is_injector) {
        for (uint32_t lnk = 0; lnk < num_links; ++lnk) {
            // noc_semaphore_wait_min(per_link_sem_ptrs[lnk], chunks_signaled_by_remote+1);
            noc_semaphore_set(per_link_sem_ptrs[lnk], 0);
        }
        if constexpr (mcast_enabled) {
            // The cumulative ping-pong ack counters must restart at 0 on a cached-program rerun.
            // Safe: the last forward's waits already observed every ack (credit caps guarantee
            // the receivers' ack totals equal the injector's expected totals), so none are in flight.
            Semaphore<>(sender_semaphore_id).set(0);
            Semaphore<>(sender_semaphore_v_id).set(0);
        }
        if (dedup_role == 2) {
            // Follower's buddy gate is cumulative — restart at 0 for cached reruns. Safe: the
            // final wait_min observed the leader's full count (leader and follower run identical
            // gate schedules), so no incs are in flight.
            Semaphore<>(buddy_gate_semaphore_id).set(0);
        }
    }
    noc.async_writes_flushed();
    noc.async_write_barrier();
}
