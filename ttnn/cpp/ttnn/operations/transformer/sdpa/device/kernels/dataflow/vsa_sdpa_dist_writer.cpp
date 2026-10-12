// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa distributed-window (v18) writer: the K side of the reader's traffic, plus the per-pass
// Q loads and the output drain. Every peer runs the same code. The reader's kreq pages
// (16 B: {kind | group << 8, w1, w2, -}) are served in order:
//   FETCH  w1 = block id, w2 = owned slot      -- K from DRAM (this peer's slice of the window)
//   PULL   w1 = x | y<<8 | src_slot<<16, w2 = gather slot -- K from a peer's owned slot
//   MARK   the group's requests are complete -> one kack page once their trids landed (lazy)
//   END    nothing more will come; exit once the outputs are drained
// Transfers are tagged per group (group g -> trids 4g+1..4g+4; groups 0/1 are the reader's two
// pending gather messages, group 2 its owned slice).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"
#include "vsa_sum_service.hpp"
#include "api/debug/dprint.h"

#if defined(VSA_PROBE) && VSA_PROBE == 9
#define VSA_TICK() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#endif

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint32_t KREQ_FETCH = 1;
constexpr uint32_t KREQ_PULL = 2;
constexpr uint32_t KREQ_MARK = 3;
constexpr uint32_t KREQ_END = 4;
constexpr uint32_t GROUP_OWN = 15;  // owned-slice fetch tag; pull messages are tagged 0..7 (their ring index)
constexpr uint32_t kOwnTrid = 9;    // pull message g -> trid 1 + g

void kernel_main() {
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(0);  // S / 64 per head
    constexpr uint32_t R_MAX = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles_per_row = get_compile_time_arg_val(2);    // Sqt * DHt
    constexpr uint32_t out_tiles_per_row = get_compile_time_arg_val(3);  // Sqt * vDHt
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_head_stride = get_compile_time_arg_val(5);
    constexpr uint32_t q_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(8);

    constexpr uint32_t cb_q_res = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(11);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(12);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(13);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(14);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(15);
    constexpr uint32_t cb_qdone = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);
    constexpr uint32_t cb_shdr = get_compile_time_arg_val(18);  // row-sum service (vsa_sum_service.hpp)
    constexpr uint32_t cb_stiles = get_compile_time_arg_val(19);
    constexpr uint32_t cb_sumback = get_compile_time_arg_val(20);
    constexpr uint32_t cb_sacc = get_compile_time_arg_val(21);
    constexpr uint32_t Sqt = get_compile_time_arg_val(22);

    constexpr auto out_args = TensorAccessorArgs<23, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto q_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    uint32_t argi = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t q_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t head = get_arg_val<uint32_t>(argi++);
    const uint32_t n_passes = get_arg_val<uint32_t>(argi++);
    const uint32_t row_count = get_arg_val<uint32_t>(argi++);
    const uint32_t pass_rows_argi = argi;
    argi += n_passes;
    const uint32_t rows_argi = argi;
    argi += row_count;
    const auto pass_rows_of = [&](uint32_t pass) { return get_arg_val<uint32_t>(pass_rows_argi + pass); };
    const auto q_tile_of = [&](uint32_t ri) { return get_arg_val<uint32_t>(rows_argi + ri); };
    if (n_passes == 0) {
        return;  // surplus core: its reader exits at once and sends no END
    }

    Noc noc;
    vsa_sum::Service sums{cb_shdr, cb_stiles, cb_sumback, get_write_ptr(cb_sacc), R_MAX, Sqt};
    experimental::CB q_cb(cb_q_res), k_cb(cb_k_stream), kreq_cb(cb_kreq), kack_cb(cb_kack);
    experimental::CB qdone_cb(cb_qdone), out_cb(cb_out);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto q = TensorAccessor(q_args, q_addr);

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);
    {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
        experimental::CB(cb_neginf).reserve_back(1);
        fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
        experimental::CB(cb_neginf).push_back(1);
    }

    const uint32_t k_l1_base = k_cb.get_write_ptr();
    const uint32_t k_block_bytes = k_tiles_per_block * k_tile_bytes;
    const uint32_t k_base = head * k_head_stride;

    uint32_t ack_pending[16];  // FIFO of tags awaiting their lazy ack
    uint32_t ack_head = 0, ack_tail = 0;
    bool ended = false;
#if defined(VSA_PROBE) && VSA_PROBE == 9
    uint32_t mark_tick[16];
    uint32_t n_pull = 0, n_mark = 0, n_ack = 0, n_iter = 0, t_mark_to_ack = 0, t_begin = VSA_TICK(), t_serve = 0;
#endif
    // one trid per message (all its K pulls) / per owned slice; landed == that trid flushed
    const auto tag_trid = [](uint32_t g) { return g == GROUP_OWN ? kOwnTrid : 1 + g; };
    const auto group_landed = [&](uint32_t g) {
        return ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), tag_trid(g));
    };
    const auto tagged_read = [&](uint64_t src, uint32_t dst, uint32_t bytes, uint32_t g) {
        experimental::set_read_trid(noc, tag_trid(g));
        noc_async_read(src, dst, bytes, noc.get_noc_id());
        experimental::set_read_trid(noc, 0);
    };
    const auto serve_kreq_if_any = [&]() {
        while (ack_head != ack_tail && group_landed(ack_pending[ack_head & 15])) {
#if defined(VSA_PROBE) && VSA_PROBE == 9
            t_mark_to_ack += VSA_TICK() - mark_tick[ack_head & 15];
            ++n_ack;
#endif
            kack_cb.reserve_back(1);
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kack_cb.get_write_ptr())[0] = ack_pending[ack_head & 15];
            kack_cb.push_back(1);
            ++ack_head;
        }
        while (!ended && cb_pages_available_at_front(cb_kreq, 1)) {
            kreq_cb.wait_front(1);
            uint32_t w0, w1, w2;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                invalidate_l1_cache();  // page written by NCRISC: bypass this RISC's stale L1 read cache
                w0 = rq[0];
                w1 = rq[1];
                w2 = rq[2];
            }
            kreq_cb.pop_front(1);
            const uint32_t kind = w0 & 0xFF;
            const uint32_t g = (w0 >> 8) & 0xFF;
            ASSERT(g < 8 || g == GROUP_OWN);
            if (kind == KREQ_FETCH) {
                const uint32_t k_tile0 = k_base + w1 * k_tiles_per_block;
                experimental::set_read_trid(noc, tag_trid(g));
                for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                    noc.async_read(
                        k,
                        k_cb,
                        k_tile_bytes,
                        {.page_id = k_tile0 + i},
                        {.offset_bytes = w2 * k_block_bytes + i * k_tile_bytes});
                }
                experimental::set_read_trid(noc, 0);
            } else if (kind == KREQ_PULL) {
#if defined(VSA_PROBE) && VSA_PROBE == 9
                ++n_pull;
#endif
                const uint32_t x = w1 & 0xFF, y = (w1 >> 8) & 0xFF, src_slot = w1 >> 16;
                tagged_read(
                    get_noc_addr(x, y, k_l1_base + src_slot * k_block_bytes, noc.get_noc_id()),
                    k_l1_base + w2 * k_block_bytes,
                    k_block_bytes,
                    g);
            } else if (kind == KREQ_MARK) {
                ASSERT(ack_tail - ack_head < 16);
                ack_pending[ack_tail & 15] = g;
#if defined(VSA_PROBE) && VSA_PROBE == 9
                mark_tick[ack_tail & 15] = VSA_TICK();
                ++n_mark;
#endif
                ++ack_tail;
            } else if (kind == KREQ_END) {
                ended = true;
            } else {
                ASSERT(false);
            }
        }
    };

    uint32_t drained = 0;
    uint32_t pass = 0;
    uint32_t pass_base = 0;
    while (!ended || drained < row_count || ack_head != ack_tail) {
#if defined(VSA_PROBE) && VSA_PROBE == 9
        ++n_iter;
#endif
        // next pass's Q rows once the previous pass's outputs left the (reused) Q/O residents
        if (pass < n_passes && drained >= pass_base) {
            const uint32_t pass_rows = pass_rows_of(pass);
            if (pass_rows > 0) {
                for (uint32_t r = 0; r < pass_rows; ++r) {
                    const uint32_t page0 = (head * n_q_tiles + q_tile_of(pass_base + r)) * q_tiles_per_row;
                    for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                        // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                        noc.async_read(
                            q,
                            q_cb,
                            q_tile_bytes,
                            {.page_id = page0 + i},
                            {.offset_bytes = (r * q_tiles_per_row + i) * q_tile_bytes});
                    }
                    serve_kreq_if_any();
                }
                noc.async_read_barrier();
                qdone_cb.reserve_back(1);
                qdone_cb.push_back(1);
            }
            pass_base += pass_rows;
            ++pass;
        }

        serve_kreq_if_any();
#if !defined(VSA_NO_SUMS)
        sums.serve();
#endif

        if (drained < row_count && cb_pages_available_at_front(cb_out, out_tiles_per_row)) {
            out_cb.wait_front(out_tiles_per_row);
            const uint32_t page0 = (head * n_q_tiles + q_tile_of(drained)) * out_tiles_per_row;
            for (uint32_t i = 0; i < out_tiles_per_row; ++i) {
                noc.async_write(
                    out_cb, out, out_tile_bytes, {.offset_bytes = i * out_tile_bytes}, {.page_id = page0 + i});
            }
            noc.async_write_barrier();
            out_cb.pop_front(out_tiles_per_row);
            ++drained;
        }
    }
#if defined(VSA_PROBE) && VSA_PROBE == 9
    if (head == 0 && row_count > 0) {
        DPRINT(
            "VSA_WR total {} iters {} pulls {} marks {} acks {} mark_to_ack_avg {}\n",
            VSA_TICK() - t_begin,
            n_iter,
            n_pull,
            n_mark,
            n_ack,
            n_ack ? t_mark_to_ack / n_ack : 0u);
    }
#endif
}
