// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Length-balanced decode writer (see lean_schedule.hpp). Per piece, in the same order as reader and
// compute: generate the causal mask when the piece holds the segment's last chunk; a child piece sends
// its partial (l, m, o) into its root's staging slot and bumps the root's semaphore; a root waits for
// all its children, feeds them to compute one by one, then writes the finished rows; a whole segment
// writes its rows directly.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/lean_schedule.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t PNHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(7);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(10);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(11);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(16);
    constexpr auto out_args = TensorAccessorArgs<23>();
    constexpr uint32_t B = LEAN_B;

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    // Received child O gets its own CB: c_16 is compute -> writer (outgoing partial), and a CB cannot
    // change producer within a run.
    constexpr uint32_t cb_child_o = tt::CBIndex::c_13;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_intermed_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));

    CircularBuffer cb_index(cb_cur_pos);
    cb_index.wait_front(1);
    volatile tt_l1_ptr uint32_t* pos_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_index.get_read_ptr());
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }
    cb_index.pop_front(1);

    constexpr uint32_t k_chunk_size = Sk_chunk_t * tt::constants::TILE_HEIGHT;
    auto get_n = [&](uint32_t b) -> uint32_t { return lean::chunks_for_pos(pos[b], k_chunk_size); };
    const lean::Schedule sched = lean::make_schedule<B, num_kv_heads>(num_cores, get_n);
    if (!sched.U || !sched.nonempty(core_idx)) {
        return;
    }

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_intermed_out);
    constexpr uint32_t slot_bytes = (out_chunk_tiles + 2 * PNHt) * stats_tile_bytes;
    constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;
    const auto out_writer = TensorAccessor(out_args, out_addr);
    volatile tt_l1_ptr uint32_t* sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(reducer_semaphore_id));
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];

    noc.async_write_barrier();

    lean::PieceIter<B, num_kv_heads> it(core_idx, sched);
    lean::Piece p;
    while (it.next(get_n, p)) {
        if (p.j1 == p.n) {
            generate_mask<cb_mask_in, PNHt>(p.n, Sk_chunk_t, pos[p.b]);
        }

        if (p.is_child) {
            CircularBuffer cb_o(cb_out_o);
            CircularBuffer cb_m(cb_out_m);
            CircularBuffer cb_l(cb_out_l);
            cb_o.wait_front(out_chunk_tiles);
            cb_m.wait_front(PNHt);
            cb_l.wait_front(PNHt);
            constexpr uint32_t ml_bytes = PNHt * stats_tile_bytes;
            constexpr uint32_t o_bytes = out_chunk_tiles * stats_tile_bytes;
            const uint32_t root_x = core_xs[p.root_core];
            const uint32_t root_y = core_ys[p.root_core];
            uint32_t dst = CircularBuffer(cb_intermed_out).get_write_ptr() + p.child_slot * slot_bytes;
            UnicastEndpoint ep;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_l.get_read_ptr()),
                ep,
                ml_bytes,
                {},
                {.noc_x = root_x, .noc_y = root_y, .addr = dst});
            dst += ml_bytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_m.get_read_ptr()),
                ep,
                ml_bytes,
                {},
                {.noc_x = root_x, .noc_y = root_y, .addr = dst});
            dst += ml_bytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_o.get_read_ptr()),
                ep,
                o_bytes,
                {},
                {.noc_x = root_x, .noc_y = root_y, .addr = dst});
            noc.async_write_barrier();
            Semaphore<>(reducer_semaphore_id).up(noc, root_x, root_y, 1);
            noc.async_atomic_barrier();
            cb_o.pop_front(out_chunk_tiles);
            cb_m.pop_front(PNHt);
            cb_l.pop_front(PNHt);
            continue;
        }

        if (p.is_root) {
            while (true) {
                invalidate_l1_cache();
                if (*sem_ptr >= p.num_children) {
                    break;
                }
            }
            constexpr uint32_t ml_bytes = PNHt * stats_tile_bytes;
            constexpr uint32_t o_bytes = out_chunk_tiles * stats_tile_bytes;
            UnicastEndpoint src;
            for (uint32_t slot = 0; slot < p.num_children; ++slot) {
                uint32_t rd = CircularBuffer(cb_intermed_out).get_read_ptr() + slot * slot_bytes;
                CircularBuffer cb_l(cb_l_in);
                cb_l.reserve_back(PNHt);
                noc.async_read(
                    src,
                    CoreLocalMem<uint32_t>(cb_l.get_write_ptr()),
                    ml_bytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = rd},
                    {});
                rd += ml_bytes;
                noc.async_read_barrier();
                cb_l.push_back(PNHt);
                CircularBuffer cb_m(cb_m_in);
                cb_m.reserve_back(PNHt);
                noc.async_read(
                    src,
                    CoreLocalMem<uint32_t>(cb_m.get_write_ptr()),
                    ml_bytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = rd},
                    {});
                rd += ml_bytes;
                noc.async_read_barrier();
                cb_m.push_back(PNHt);
                CircularBuffer cb_o(cb_child_o);
                cb_o.reserve_back(out_chunk_tiles);
                noc.async_read(
                    src,
                    CoreLocalMem<uint32_t>(cb_o.get_write_ptr()),
                    o_bytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = rd},
                    {});
                noc.async_read_barrier();
                cb_o.push_back(out_chunk_tiles);
            }
        }

        CircularBuffer cb_out_buf(cb_out);
        cb_out_buf.wait_front(out_chunk_tiles);
        uint32_t out_tile_id = p.b * out_chunk_tiles;
        barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(
            out_tile_id, out_writer, barrier_count, p.h, num_heads_to_write, out_chunk_tiles);
        noc.async_write_barrier();
        cb_out_buf.pop_front(out_chunk_tiles);
    }
}
