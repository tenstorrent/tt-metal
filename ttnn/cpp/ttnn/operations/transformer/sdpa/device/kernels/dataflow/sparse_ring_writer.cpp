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

#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "fused_op_receiver.hpp"
#include "ring_utils.hpp"
#include "sparse_dense_units.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_ring_balance_plan.hpp"

// Output slice of one local Q chunk (the op has no joint tokens).
inline Slice q_chunk_out_slice(uint32_t q_chunk, uint32_t nb, uint32_t nq, uint32_t Sq_chunk_t, uint32_t DHt) {
    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
    return Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
}

// Sparse ring attention writer. Compute keeps every resident Q chunk's accumulators in L1 for the whole op, so there
// is no save / restore between ring iterations: this kernel generates the constant tiles (column identity and
// reduce scaler), follows the ring sequence to stay aligned with the reader and the all-gather, and on the
// last active ring iteration of each pass writes that pass's normalized outputs, in Q-chunk order.
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(25);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(27);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(kv_stripe_split == 1, "the sparse op does not split KV");
    static_assert(L == 0, "the sparse op has no joint tokens");
    static_assert(use_streaming_compute, "the sparse op runs the streaming compute");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");

    // Slots 28-37 hold scalars (32-35, the rank mapping, are unused: the writer never needs the ring id); the output
    // accessor starts at slot 38 and the CB block follows it.
    constexpr uint32_t kFirstAccessorArgOffset = 38;
    constexpr auto out_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    // dense_blocks split across devices: where emitted partials go (the hop staging buffer, or dense_debug's).
    constexpr auto partials_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_arg_offset = partials_args.next_compile_time_args_offset();

    // No Q rotation and no grouped K/V sources.
    static_assert(get_ct_arg<kernel_compile_time_args.size() - 1>() == 0, "no rotated Q split");
    static_assert(GROUPED_KV_SOURCE_COUNT == 1, "no grouped K/V sources");

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    // The static split's units; with load balancing, the range the balance kernel releases (see the reader).
    uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    argidx++;  // logical_nt (unused: the writer only follows the ring order)
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    argidx++;  // single-valid-K-chunk mask (only the save / restore path needs it)
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);
    // Load balancing (cross-device): [swapped-in units' output buffer (laid out as the output, by local slot), balance
    // core NoC x, y].
    constexpr bool balance = get_named_compile_time_arg_val("balance") == 1;
    constexpr bool balance_remote = get_named_compile_time_arg_val("balance_remote") == 1;
    [[maybe_unused]] const uint32_t balance_base = argidx;
    if constexpr (balance_remote) {
        argidx += 3;
    }
    // dense_blocks: the ranges (start, end), then (dense_owners > 0) [first unit, end unit, partials buffer, ready
    // semaphore + , ready semaphore -] and the sparse_dense::Owner records.
    constexpr uint32_t dense_n_ranges = get_named_compile_time_arg_val("dense_n_ranges");
    constexpr uint32_t dense_owners = get_named_compile_time_arg_val("dense_owners");
    constexpr bool res_remap = get_named_compile_time_arg_val("res_remap") == 1;
    constexpr uint32_t mask_nb = get_named_compile_time_arg_val("mask_nb");
    constexpr uint32_t mask_nb_pad = get_named_compile_time_arg_val("mask_nb_pad");
    const uint32_t dense_ranges_base = argidx;
    const uint32_t dense_base = dense_ranges_base + 2 * dense_n_ranges;
    const auto in_dense = [&](uint32_t g) {
        for (uint32_t r = 0; r < dense_n_ranges; ++r) {
            if (g >= get_arg_val<uint32_t>(dense_ranges_base + 2 * r) &&
                g < get_arg_val<uint32_t>(dense_ranges_base + 2 * r + 1)) {
                return true;
            }
        }
        return false;
    };
    const uint32_t shard = fused_op_receiver.seq.ring_index;

    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    Noc noc;

    const auto out_writer = TensorAccessor(out_args, out_addr);
    constexpr bool output_has_no_padding = q_local_padded_Nt % Sq_chunk_t == 0;
    using StaticOutputTileShape = StaticTensorTileShape<B, NH, q_local_padded_Nt, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;
    const auto output_tile_logical = OutputTileShape(B, NH, q_local_padded_Nt, vDHt);
    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    // One normalized Q block from cb_out into the output (the passes' blocks, and the owner's dense blocks).
    const auto write_q_block = [&](uint32_t q_chunk, uint32_t nb, uint32_t nq) SPARSE_DENSE_ONE_COPY {
        write_block_row_grouped_trid<output_has_no_padding>(
            noc,
            out_generator,
            q_chunk_out_slice(q_chunk, nb, nq, Sq_chunk_t, vDHt),
            /*end_seq_tile=*/q_local_padded_Nt,
            cb_out,
            tile_bytes,
            out_subblock_h,
            /*flush_trid=*/0);
    };
    // A block swapped in from another device: into the export buffer (same layout), from where it goes home.
    [[maybe_unused]] const auto export_q_block = [&](uint32_t q_chunk, uint32_t nb, uint32_t nq) {
        write_block_row_grouped_trid<output_has_no_padding>(
            noc,
            PaddedAddrGenerator(TensorAccessor(out_args, get_arg_val<uint32_t>(balance_base)), output_tile_logical),
            q_chunk_out_slice(q_chunk, nb, nq, Sq_chunk_t, vDHt),
            /*end_seq_tile=*/q_local_padded_Nt,
            cb_out,
            tile_bytes,
            out_subblock_h,
            /*flush_trid=*/0);
    };

    {
        // Zero the identity tile first: generate_bcast_col_scalar writes column 0 only, and the compute step
        // multiplies its probabilities by this tile to get row sums, which reads every column of it.
        Noc noc_zero;
        fill_tile_zeros<get_tile_size(cb_col_identity)>(noc_zero, cb_col_identity, 0);
    }
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    {
        // The compute step's row-sum operand: an identity tile (the scalar on the diagonal of faces 0 and 3, which
        // hold rows / columns 0-15 and 16-31), so P x I = P and each sum slot keeps one partial sum per column.
        constexpr uint32_t cb_sum_identity = get_named_compile_time_arg_val("cb_sum_identity");
        Noc noc_zero;
        fill_tile_zeros<get_tile_size(cb_sum_identity)>(noc_zero, cb_sum_identity, 0);
        CircularBuffer cb(cb_sum_identity);
        cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
        const uint16_t one = identity_scalar_packed >> 16;
        for (uint32_t r = 0; r < 16; ++r) {
            p[r * 16 + r] = one;            // face 0
            p[3 * 256 + r * 16 + r] = one;  // face 3
        }
        cb.push_back(1);
    }
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    // Load balancing: the range the balance kernel releases (the reader waits for the same words) and, with
    // cross-device balancing, which of its units are swapped in (their outputs go to the export buffer and are counted
    // on the balance core, which has the gather writers send them home).
    [[maybe_unused]] uint32_t import_bits = 0;
    if constexpr (balance) {
        Semaphore<> range_sem(get_named_compile_time_arg_val("balance_sem_range"));
        range_sem.wait_min(sparse_balance::kRangeValid);
        const uint32_t range = range_sem.value();
        global_q_start = sparse_balance::range_start(range);
        global_q_end = sparse_balance::range_end(range);
        if constexpr (balance_remote) {
            import_bits = Semaphore<>(get_named_compile_time_arg_val("balance_sem_bits")).value();
        }
    }

    // Passes split the core's Q blocks evenly (see the reader); each replays the ring order from the start (the
    // writer never waits on the all-gather, so a fresh copy of the receiver is enough).
    constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
    const uint32_t q_per_core = global_q_end - global_q_start;
    const RingSDPAOpReceiver receiver_start = fused_op_receiver;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        RingSDPAOpReceiver pass_receiver = receiver_start;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            // The ring-id sequence advances on every iteration so the writer stays aligned with the reader, compute
            // and the all-gather; only the pass's last active iteration writes.
            pass_receiver.get_next_ring_id_and_sync();
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0 ||
                !is_last_active_ring_iter(active_ring_iter_mask, ring_iter)) {
                continue;
            }
            [[maybe_unused]] uint32_t n_exported = 0;
            for (uint32_t q_flat = pass_q_start; q_flat < pass_q_end; ++q_flat) {
                const auto decoded_q = decompose_global_q_index(q_flat, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t q_chunk =
                    res_remap ? sparse_dense::resident_block(shard, decoded_q.q_chunk, mask_nb, mask_nb_pad, in_dense)
                              : decoded_q.q_chunk;
                if constexpr (balance_remote) {
                    if (((import_bits >> (q_flat - global_q_start)) & 1u) != 0) {
                        export_q_block(q_chunk, decoded_q.nb, decoded_q.nq);
                        ++n_exported;
                        continue;
                    }
                }
                write_q_block(q_chunk, decoded_q.nb, decoded_q.nq);
            }
            noc.async_write_barrier();
            if constexpr (balance_remote) {
                if (n_exported > 0) {
                    Semaphore<>(get_named_compile_time_arg_val("balance_sem_outputs"))
                        .up(noc,
                            get_arg_val<uint32_t>(balance_base + 1),
                            get_arg_val<uint32_t>(balance_base + 2),
                            n_exported);
                    noc.async_atomic_barrier();
                }
            }
        }
    }

    // Dense units (see the reader): per batch, the owner writes its normalized dense rows to the output; every other
    // device writes the raw partials to the partials buffer and (not in dense_debug) counts each unit, once written, in
    // its direction's ready word (a running count) for the gather writer that forwards it.
    if constexpr (dense_owners > 0) {
        constexpr bool dense_debug = get_named_compile_time_arg_val("dense_debug") == 1;
        constexpr uint32_t kPartialTiles = sparse_dense::partial_tiles<Sq_chunk_t, vDHt>();
        const auto partials_writer = TensorAccessor(partials_args, get_arg_val<uint32_t>(dense_base + 2));
        CircularBuffer cb(cb_out);
        sparse_dense::for_each_batch(
            dense_owners,
            dense_base + 5,
            get_arg_val<uint32_t>(dense_base),
            get_arg_val<uint32_t>(dense_base + 1),
            get_named_compile_time_arg_val("dense_blocks_total"),
            get_named_compile_time_arg_val("resident_slots"),
            [](uint32_t a) { return get_arg_val<uint32_t>(a); },
            [&](uint32_t, const sparse_dense::Owner& o, uint32_t u0, uint32_t u1, bool, bool) {
                const bool emit = dense_debug || o.role != sparse_dense::kRoleOwner;
                for (uint32_t u = u0; u < u1; ++u) {
                    if (!emit) {
                        constexpr uint32_t n_dense_blocks = get_named_compile_time_arg_val("dense_blocks_total");
                        write_q_block(
                            sparse_dense::dense_block(
                                o.shard, u % n_dense_blocks - o.block_offset, mask_nb, mask_nb_pad, in_dense),
                            0,
                            u / n_dense_blocks);
                        continue;
                    }
                    for (uint32_t t = 0; t < kPartialTiles; ++t) {
                        cb.wait_front(1);
                        // dense_debug: unit-major for the test's host merge; else one bank per unit.
                        const uint32_t page =
                            dense_debug ? u * kPartialTiles + t
                                        : sparse_dense::banked_page(
                                              u, t, kPartialTiles, get_named_compile_time_arg_val("dense_banks"));
                        noc.async_write(
                            CoreLocalMem<uint32_t>(cb.get_read_ptr()),
                            partials_writer,
                            tile_bytes,
                            {},
                            {.page_id = page});
                        noc.async_writes_flushed();
                        cb.pop_front(1);
                    }
                    if constexpr (!dense_debug) {
                        noc.async_write_barrier();
                        const uint32_t ready_addr =
                            get_arg_val<uint32_t>(dense_base + (o.role == sparse_dense::kRoleEmitPlus ? 3 : 4));
                        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], ready_addr), 1);
                        noc_async_atomic_barrier();
                    }
                }
                noc.async_write_barrier();
            });
    }
}
