// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa — writer (RISCV_0, NOC1).
//
// boot: the MAX / REDUCE_ROW reduce scaler (bf16 1.0) into cb_reduce_scaler, once.
// store_block: each row's two output tiles go straight to their concatenated-heads pages
//   (b*S_t + st)*heads*HD + h*HD + dt as soon as compute pushes the row (cumulative waits, no pops mid-block);
//   one write barrier a block, then the NOMINAL q_block_rows*HD pages are popped (compute pad-pushes the
//   remainder), so the cb_out ring returns to its base every block.
//
// Split reads (Perf 1, perf_experiments/dual_noc_reads; CT arg split_reads, the default): this RISC is also the
// producer of the Q and V input streams, read on NOC1 so they leave NOC0 (the reader's) — Q into cb_query in the
// reader's lead-rows order; V into cb_value_aug (phantom slots, ones column, own V page reads, multicast receives; on a
// V multicast SENDER core the reader still reads + multicasts on NOC0 and raises local flag kv_sem_base + 2*kv_depth +
// slot, which this RISC waits on before the push). Inputs of block b are loaded BEFORE the outputs of block b-1 are
// stored, so the next block's Q / the next segment's V start as soon as compute frees the ring (like the reader).
// Split startup (Perf 2, perf_experiments/startup_k_first; the reader's file header): for segment 0 this RISC also
// reads key rows [k_split_rows, N) into the reader's cb_key_t slot (it never reserves / pushes cb_key_t) and raises
// flag k_half, and reads only value rows [N / 2, N) of a READ / SEND slot (the reader reads the rest on NOC0): a READ
// slot is pushed once the reader's V flag says its half is in, a SEND slot's half is handed over with flag v_half. The
// reduce scaler is prepared under the first block's reads instead of ahead of them, so the t = 0 reads issue at once.
// Throttled startup (Perf 2; the reader's file header): with CT arg read_chunk > 0 this RISC's segment-0 page reads
// (the first block's Q lead rows, its K part, its V rows) retire in chunks of read_chunk pages, a read barrier after
// each. Raw noc_async_read_page: the same primitive the op's reader uses (no kernel_lib page-read helper is bypassed).
// The reader -> writer flag on one core is a raw volatile L1 semaphore: kernel_lib has no same-core RISC-to-RISC
// signal.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "nomic_sdpa_schedule.hpp"

namespace {
using namespace dataflow_kernel_lib;
constexpr uint32_t kRoleRead = 0, kRoleSend = 1, kRoleReceive = 2;  // host ROLE_* (program descriptor)

template <uint32_t Depth, typename Fn>
FORCE_INLINE void with_slot(uint32_t slot, Fn&& fn) {
    if constexpr (Depth > 0) {
        if (slot == Depth - 1) {
            fn(std::integral_constant<uint32_t, Depth - 1>{});
        } else {
            with_slot<Depth - 1>(slot, fn);
        }
    }
}
template <uint32_t SemId>
using KvReceiver = ReceiverPipe<SemId, false>;
}  // namespace

#ifdef ABLATE_READS
#define READ_PAGE(...) ((void)0)  // ABLATION ONLY: keep the CB protocol, drop the NoC reads
#else
#define READ_PAGE(...) noc_async_read_page(__VA_ARGS__)
#endif

void kernel_main() {
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_reduce_scaler = get_named_compile_time_arg_val("cb_reduce_scaler");
    constexpr uint32_t cb_query = get_named_compile_time_arg_val("cb_query");
    constexpr uint32_t cb_value_aug = get_named_compile_time_arg_val("cb_value_aug");
    constexpr uint32_t Q = get_named_compile_time_arg_val("q_block_rows");
    constexpr uint32_t N = get_named_compile_time_arg_val("key_block_tiles");
    constexpr uint32_t HD = get_named_compile_time_arg_val("head_dim_tiles");
    constexpr uint32_t PV = get_named_compile_time_arg_val("pv_out_tiles");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr uint32_t heads = get_named_compile_time_arg_val("heads");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t kv_depth = get_named_compile_time_arg_val("kv_depth");
    constexpr uint32_t kv_sem_base = get_named_compile_time_arg_val("kv_sem_base");
    constexpr uint32_t q_lead_rows = get_named_compile_time_arg_val("q_lead_rows");
    constexpr bool split_reads = get_named_compile_time_arg_val("split_reads") == 1;
    constexpr bool load_q = split_reads;
    constexpr bool load_v = split_reads;
    constexpr uint32_t q_depth = get_named_compile_time_arg_val("q_depth");
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto q_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_key_t = get_named_compile_time_arg_val("cb_key_t");
    constexpr uint32_t k_split_rows = get_named_compile_time_arg_val("k_split_rows");  // see the reader
    constexpr bool split_k = split_reads && k_split_rows < N;
    constexpr bool split_v = split_reads && N >= 2;
    constexpr uint32_t sem_k_half = kv_sem_base + 3 * kv_depth;
    constexpr uint32_t sem_v_half = kv_sem_base + 3 * kv_depth + 1;

    constexpr uint32_t kTableArg = get_named_compile_time_arg_val("table_arg");  // after the io addresses
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = nomic_sdpa::core_index(grid_x);
    const uint32_t row_begin = get_common_arg_val<uint32_t>(kTableArg + core);
    const uint32_t row_end = get_common_arg_val<uint32_t>(kTableArg + core + 1);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    Noc noc;

    // Throttled startup (file header): a read barrier after every read_chunk segment-0 page reads (0: unthrottled).
    constexpr uint32_t read_chunk = get_named_compile_time_arg_val("read_chunk");
    uint32_t chunk_reads = 0;
    auto throttle = [&](bool segment0) {
        if constexpr (read_chunk > 0) {
            if (segment0 && ++chunk_reads == read_chunk) {
                chunk_reads = 0;
                MaybeDeviceZoneScope("writer_read_chunk");
                noc_async_read_barrier();
            }
        }
    };

    // The reduce scaler: compute needs it only at its first row max. Under split_reads it is prepared under the first
    // block's reads (prepare_scaler is idempotent), so the t = 0 reads issue at kernel start.
    DataflowBuffer scaler_buf(cb_reduce_scaler);
    bool scaler_ready = false;
    auto prepare_scaler = [&]() {
        if (!scaler_ready) {
            MaybeDeviceZoneScope("writer_scaler");
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_reduce_scaler,
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW>(scaler_buf);
            scaler_ready = true;
        }
    };
    if constexpr (!split_reads) {
        prepare_scaler();
    }

    uint32_t roles[kv_depth];
    for (uint32_t k = 0; k < kv_depth; ++k) {
        roles[k] = get_arg_val<uint32_t>(k);
    }
    const uint32_t no_sender_coords[mcast_wire::SENDER_COORD_WORDS] = {};
    constexpr uint32_t value_slot_bytes = N * PV * tile_bytes;
    const uint32_t key_ring_base = get_write_ptr(cb_key_t);  // never reserved here: the reader owns cb_key_t
    constexpr uint32_t key_slot_bytes = N * HD * tile_bytes;
    auto wait_flag = [](uint32_t sem) {  // a same-core flag from the reader, consumed
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
        while (*flag == 0) {
            invalidate_l1_cache();
        }
        *flag = 0;
    };
    auto raise_flag = [](uint32_t sem) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)) = 1; };
    // value rows [first, N) of a pair into a cb_value_aug slot (rows st*PV + dt) for segment `segment`; the caller
    // barriers
    auto issue_value_rows = [&](uint32_t pair, uint32_t v_base, uint32_t first, uint32_t segment) {
        const uint32_t pair_first_page = pair * seq_tiles * HD;
        for (uint32_t st = first; st < N; ++st) {
            for (uint32_t dt = 0; dt < HD; ++dt) {
                READ_PAGE(pair_first_page + st * HD + dt, v_acc, v_base + (st * PV + dt) * tile_bytes);
                throttle(segment == 0);
            }
        }
    };
    const uint32_t first_pair = row_begin / seq_tiles;
    if constexpr (load_v) {
        for (uint32_t i = 0; i < first_pair % kv_depth; ++i) {  // phantom slots (see the reader)
            cb_reserve_back(cb_value_aug, N * PV);
            cb_push_back(cb_value_aug, N * PV);
        }
    }

    auto write_ones_column = [&](uint32_t slot_base) {  // identical to the reader's (its carve-out path)
        const uint32_t ones_tile = slot_base + HD * tile_bytes;
        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_tile);
#pragma GCC unroll 16
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            words[i] = 0x3F803F80;  // bf16 1.0 in every element: P V carries l in every column (normalize in DEST)
        }
        if constexpr (N > 1) {
            dataflow_kernel_lib::set_read_state<tile_bytes>(noc, ones_tile);
            for (uint32_t st = 1; st < N; ++st) {
                dataflow_kernel_lib::read_with_state<tile_bytes>(noc, ones_tile + st * PV * tile_bytes, ones_tile);
            }
            noc_async_read_barrier();
        }
    };

    auto load_query_rows = [&](const nomic_sdpa::Block& blk, uint32_t q_base, uint32_t first_row, uint32_t end_row) {
        for (uint32_t i = first_row * HD; i < end_row * HD; ++i) {
            READ_PAGE(blk.row0 * HD + i, q_acc, q_base + i * tile_bytes);
        }
        noc_async_read_barrier();
    };

    // The block's inputs this RISC produces, in the reader's order: Q lead rows, V (first block of a segment), Q rest.
    auto load_inputs = [&](const nomic_sdpa::Block& blk) {
        uint32_t q_base = 0;
        const uint32_t lead = blk.rows < q_lead_rows ? blk.rows : q_lead_rows;
        // this RISC reads part of segment 0's K (split_k; a K RECEIVE segment gets the whole slot by multicast)
        const bool k_half = split_k && blk.segment == 0 && blk.first_of_segment && roles[0] != kRoleReceive;
        if constexpr (load_q) {
            {
                MaybeDeviceZoneScope("writer_q_reserve");
                cb_reserve_back(cb_query, Q * HD);
            }
            q_base = get_write_ptr(cb_query);
            MaybeDeviceZoneScope("writer_q_lead");
            for (uint32_t i = 0; i < lead * HD; ++i) {
                READ_PAGE(blk.row0 * HD + i, q_acc, q_base + i * tile_bytes);
                throttle(blk.segment == 0 && blk.first_of_segment);
            }
            if (!k_half) {
                prepare_scaler();  // under the first block's Q lead reads (a no-op after); else under the K half's
            }
            noc_async_read_barrier();
            cb_push_back(cb_query, lead * HD);
        }
        if (k_half) {
            // key rows [k_split_rows, N) of segment 0 into the reader's slot (K^T order dt*N + st), then flag k_half
            MaybeDeviceZoneScope("writer_k_half");
            const uint32_t k_base = key_ring_base + (blk.pair % kv_depth) * key_slot_bytes;
            const uint32_t pair_first_page = blk.pair * seq_tiles * HD;
            for (uint32_t st = k_split_rows; st < N; ++st) {
                for (uint32_t dt = 0; dt < HD; ++dt) {
                    READ_PAGE(pair_first_page + st * HD + dt, k_acc, k_base + (dt * N + st) * tile_bytes);
                    throttle(true);
                }
            }
            prepare_scaler();
            noc_async_read_barrier();
            raise_flag(sem_k_half);
        }
        if (load_v && blk.first_of_segment) {
            const uint32_t role = blk.segment < kv_depth ? roles[blk.segment] : kRoleRead;
            const uint32_t slot = blk.pair % kv_depth;
            {
                MaybeDeviceZoneScope("writer_v_reserve");
                cb_reserve_back(cb_value_aug, N * PV);
            }
            const uint32_t v_base = get_write_ptr(cb_value_aug);
            // First real use of this ring slot: the ones tiles (never multicast). A READ segment fills them while its
            // own V reads are in flight.
            const bool ones_now = blk.segment < kv_depth;
            if (ones_now && role != kRoleRead) {
                MaybeDeviceZoneScope("writer_v_ones");
                write_ones_column(v_base);
            }
            MaybeDeviceZoneScope("writer_v");
            if (role == kRoleReceive) {
                with_slot<kv_depth>(slot, [&](auto s) {
                    KvReceiver<kv_sem_base + 2 * decltype(s)::value + 1>(noc, no_sender_coords).receive();
                });
            } else if (role == kRoleSend) {         // the reader pre-read (and multicast) this slot on NOC0
                if (split_v && blk.segment == 0) {  // value rows [N / 2, N) for the reader's multicast
                    issue_value_rows(blk.pair, v_base, N / 2, blk.segment);
                    noc_async_read_barrier();
                    raise_flag(sem_v_half);
                }
                wait_flag(kv_sem_base + 2 * kv_depth + slot);
            } else {
                const bool split = split_v && blk.segment == 0;  // the reader reads value rows [0, N / 2)
                issue_value_rows(blk.pair, v_base, split ? N / 2 : 0, blk.segment);
                if (ones_now) {
                    write_ones_column(v_base);  // under the V reads in flight
                }
                noc_async_read_barrier();
                if (split) {
                    MaybeDeviceZoneScope("writer_v_wait_half");
                    wait_flag(kv_sem_base + 2 * kv_depth + slot);
                }
            }
            cb_push_back(cb_value_aug, N * PV);
        }
        if constexpr (load_q) {
            if (lead < blk.rows) {
                MaybeDeviceZoneScope("writer_q_rest");
                load_query_rows(blk, q_base, lead, blk.rows);
            }
            cb_push_back(cb_query, (Q - lead) * HD);
        }
    };

    constexpr uint32_t out_row_pages = heads * HD;  // pages of one output tile-row (768 columns)
    auto store_issue = [&](const nomic_sdpa::Block& blk) {
        const uint32_t b = blk.pair / heads;
        const uint32_t h = blk.pair % heads;
        const uint32_t st0 = blk.row0 - blk.pair * seq_tiles;
        {
            MaybeDeviceZoneScope("writer_rows");
            for (uint32_t r = 0; r < blk.rows; ++r) {
                cb_wait_front(cb_out, (r + 1) * HD);
                const uint32_t src = get_read_ptr(cb_out) + r * HD * tile_bytes;
                const uint32_t first_page = (b * seq_tiles + st0 + r) * out_row_pages + h * HD;
                for (uint32_t dt = 0; dt < HD; ++dt) {
#ifndef ABLATE_WRITES  // ABLATION ONLY: keep the CB protocol, drop the NoC writes
                    noc_async_write_page(first_page + dt, out_acc, src + dt * tile_bytes);
#endif
                }
            }
        }
    };
    auto store_finish = [&]() {
        {
            MaybeDeviceZoneScope("writer_barrier");
            noc_async_write_barrier();
        }
        cb_wait_front(cb_out, Q * HD);
        cb_pop_front(cb_out, Q * HD);
    };

    // inputs(b) before outputs(b - 1): see the file header. With a 1-deep cb_query (DRAM) block b's Q cannot land
    // before compute is done with block b - 1, so b - 1's output writes are ISSUED first and drained after b's reads.
    nomic_sdpa::Block prev{};
    bool have_prev = false;
    nomic_sdpa::for_each_block<seq_tiles, Q>(row_begin, row_end, [&](const nomic_sdpa::Block& blk) {
        if constexpr (q_depth == 1) {
            if (have_prev) {
                store_issue(prev);
            }
            if constexpr (load_q || load_v) {
                load_inputs(blk);
            }
            if (have_prev) {
                store_finish();
            }
        } else {
            if constexpr (load_q || load_v) {
                load_inputs(blk);
            }
            if (have_prev) {
                store_issue(prev);
                store_finish();
            }
        }
        prev = blk;
        have_prev = true;
    });
    if (have_prev) {
        store_issue(prev);
        store_finish();
    }
}
