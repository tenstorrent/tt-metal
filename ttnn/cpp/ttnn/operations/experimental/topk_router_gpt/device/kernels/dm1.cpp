// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM1 Kernel: Inter-Core Communication + Helper Tile Generation + Output Writer
// (RISCV_0, NOC 1)
//
// Three paths:
//   Sender: wait for compute partial → NOC write to worker's partial_recv → signal sem
//   Worker (non-collector): generate index tile → wait for sender partials →
//     push to compute → wait for logit+index output → NOC write to collector's gathered_val/gathered_ind → signal sem
//   Collector: generate index/mask/scaler tiles → wait for sender partials →
//     push to compute → wait for logit+index output → copy own to gathered →
//     wait for other workers' results → push gathered to compute →
//     wait for final output → write to DRAM

#include <cstdint>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline uint32_t tile_elem_idx(uint32_t row, uint32_t col) {
    uint32_t face = ((row >= 16) ? 2u : 0u) + ((col >= 16) ? 1u : 0u);
    return face * 256 + (row & 15) * 16 + (col & 15);
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_to_f32(uint16_t bf16) {
    uint32_t u = static_cast<uint32_t>(bf16) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// Generate one index tile where every element at (row, col) = bf16((float)(base + col))
void generate_index_tile(volatile tt_l1_ptr uint32_t* tile32, uint32_t base) {
    uint32_t left_packed[8], right_packed[8];
    for (uint32_t w = 0; w < 8; w++) {
        uint16_t lo = f32_to_bf16(static_cast<float>(base + 2 * w));
        uint16_t hi = f32_to_bf16(static_cast<float>(base + 2 * w + 1));
        left_packed[w] = (static_cast<uint32_t>(hi) << 16) | lo;
        lo = f32_to_bf16(static_cast<float>(base + 16 + 2 * w));
        hi = f32_to_bf16(static_cast<float>(base + 16 + 2 * w + 1));
        right_packed[w] = (static_cast<uint32_t>(hi) << 16) | lo;
    }
    for (uint32_t face = 0; face < 4; face++) {
        uint32_t* src = (face & 1) ? right_packed : left_packed;
        volatile tt_l1_ptr uint32_t* dst = tile32 + face * 128;
        for (uint32_t r = 0; r < 16; r++) {
            for (uint32_t w = 0; w < 8; w++) {
                dst[r * 8 + w] = src[w];
            }
        }
    }
}

void kernel_main() {
    Noc noc;

    // Compile-time args
    constexpr uint32_t tile_size = get_arg(args::tile_size_bf16);
    constexpr uint32_t num_groups = get_arg(args::num_groups);
    constexpr uint32_t num_senders = get_arg(args::num_senders);
    constexpr uint32_t topk_k = get_arg(args::topk_k);
    constexpr uint32_t k_padded = get_arg(args::k_padded);
    constexpr uint32_t collector_phys_x = get_arg(args::collector_physical_x);
    constexpr uint32_t collector_phys_y = get_arg(args::collector_physical_y);

    // Run-time arguments (shared layout with dm0 and compute)
    const auto dram_bank_id = get_arg(args::dram_bank_id);
    const auto vchannel = get_arg(args::vchannel);
    const auto is_sender = get_arg(args::is_sender);
    const auto is_worker = get_arg(args::is_worker);
    const auto is_collector = get_arg(args::is_collector);
    const auto num_k_tiles = get_arg(args::num_k_tiles);
    const auto k_tile_offset = get_arg(args::k_tile_offset);
    const auto n_tile_id = get_arg(args::n_tile_id);
    const auto worker_phys_x = get_arg(args::worker_phys_x);
    const auto worker_phys_y = get_arg(args::worker_phys_y);
    const auto sender_slot = get_arg(args::sender_slot);
    const auto worker_gather_slot = get_arg(args::worker_gather_slot);

    // DFBs
    DataflowBuffer dfb_partial_recv(dfb::partial_recv);
    DataflowBuffer dfb_local_out(dfb::local_out);
    DataflowBuffer dfb_index(dfb::index);
    DataflowBuffer dfb_topk_val(dfb::topk_val);
    DataflowBuffer dfb_gathered_val(dfb::gathered_val);
    DataflowBuffer dfb_gathered_ind(dfb::gathered_ind);
    DataflowBuffer dfb_softmax_mask(dfb::softmax_mask);
    DataflowBuffer dfb_bcast_scaler(dfb::bcast_scaler);
    DataflowBuffer dfb_final_out(dfb::final_out);
    DataflowBuffer dfb_dispatch(dfb::dispatch);

    constexpr uint32_t tile_u32 = tile_size / sizeof(uint32_t);

    // Pre-compute the partial_recv base address.
    // All cores have identical DFB layout (every DFB is placed on all cores), so the base
    // address of partial_recv is at the same L1 offset on every core. We read it from
    // our own DFB interface — valid because partial_recv is allocated on all cores.
    // Then we use this stable base + slot offset for NOC writes to the worker.
    const uint32_t partial_recv_base_addr = dfb_partial_recv.get_write_ptr();

    if (is_sender) {
        // ============================================================
        // SENDER PATH
        // ============================================================
        dfb_local_out.wait_front(1);
        uint32_t local_out_l1 = dfb_local_out.get_read_ptr();

        // NOC write partial tile to worker's partial_recv at our sender_slot.
        // partial_recv_base_addr is the same L1 address on all cores due to uniform DFB layout.
        uint32_t worker_recv_l1 = partial_recv_base_addr + sender_slot * tile_size;

        noc.async_write(
            CoreLocalMem<uint32_t>(local_out_l1),
            UnicastEndpoint{},
            tile_size,
            {},
            {.noc_x = worker_phys_x, .noc_y = worker_phys_y, .addr = worker_recv_l1});
        noc.async_write_barrier();

        // Signal worker that this sender's partial is ready
        Semaphore<> worker_sem(sem::partial_ready);
        worker_sem.up(noc, worker_phys_x, worker_phys_y, 1);
        noc.async_atomic_barrier();

        dfb_local_out.pop_front(1);
        return;
    }

    // ============================================================
    // WORKER PATH (including collector)
    // ============================================================

    // 1. Generate index template tile (expert_base + 0..31)
    uint32_t expert_base = n_tile_id * 32;
    dfb_index.reserve_back(1);
    {
        uint32_t index_l1 = dfb_index.get_write_ptr();
        volatile tt_l1_ptr uint32_t* tile32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1);
        generate_index_tile(tile32, expert_base);
    }
    dfb_index.push_back(1);

    // Collector also generates softmax helper tiles
    if (is_collector) {
        // Softmax mask: cols 0..k-1 = 0.0, cols k..31 = -inf
        dfb_softmax_mask.reserve_back(1);
        {
            uint32_t mask_l1 = dfb_softmax_mask.get_write_ptr();
            volatile tt_l1_ptr uint32_t* mask32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1);
            constexpr uint32_t neg_inf_packed = 0xFF80FF80u;
            for (uint32_t i = 0; i < tile_u32; i++) {
                mask32[i] = neg_inf_packed;
            }
            volatile tt_l1_ptr uint16_t* mask16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1);
            for (uint32_t row = 0; row < 32; row++) {
                for (uint32_t col = 0; col < topk_k; col++) {
                    uint32_t idx = tile_elem_idx(row, col);
                    mask16[idx] = 0x0000;
                }
            }
        }
        dfb_softmax_mask.push_back(1);

        // Broadcast scaler: all 1.0
        dfb_bcast_scaler.reserve_back(1);
        {
            uint32_t scaler_l1 = dfb_bcast_scaler.get_write_ptr();
            volatile tt_l1_ptr uint32_t* scaler32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_l1);
            constexpr uint32_t one_packed = 0x3F803F80u;
            for (uint32_t i = 0; i < tile_u32; i++) {
                scaler32[i] = one_packed;
            }
        }
        dfb_bcast_scaler.push_back(1);
    }

    // 2. Reserve space in partial_recv for the incoming partial tiles
    dfb_partial_recv.reserve_back(num_senders);

    // Wait for all senders' partials to arrive
    Semaphore<> partial_sem(sem::partial_ready);
    partial_sem.wait(num_senders);
    partial_sem.set(0);

    dfb_partial_recv.push_back(num_senders);

    // 3. Wait for compute to produce logit output (topk_val).
    dfb_topk_val.wait_front(1);

    if (!is_collector) {
        // Non-collector worker: send logit+index tiles to collector
        uint32_t val_l1 = dfb_topk_val.get_read_ptr();
        uint32_t ind_l1 = dfb_index.get_read_ptr();

        // Use our own gathered_val/gathered_ind base addresses — identical L1 layout on all worker cores
        uint32_t coll_val_base = dfb_gathered_val.get_write_ptr();
        uint32_t coll_val_dst_l1 = coll_val_base + worker_gather_slot * tile_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(val_l1),
            UnicastEndpoint{},
            tile_size,
            {},
            {.noc_x = collector_phys_x, .noc_y = collector_phys_y, .addr = coll_val_dst_l1});

        uint32_t coll_ind_base = dfb_gathered_ind.get_write_ptr();
        uint32_t coll_ind_dst_l1 = coll_ind_base + worker_gather_slot * tile_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(ind_l1),
            UnicastEndpoint{},
            tile_size,
            {},
            {.noc_x = collector_phys_x, .noc_y = collector_phys_y, .addr = coll_ind_dst_l1});

        noc.async_write_barrier();

        // Signal collector
        Semaphore<> coll_sem(sem::topk_ready);
        coll_sem.up(noc, collector_phys_x, collector_phys_y, 1);
        noc.async_atomic_barrier();

        dfb_topk_val.pop_front(1);
        dfb_index.pop_front(1);
        return;
    }

    // ============================================================
    // COLLECTOR PATH (continues from worker)
    // ============================================================

    // 4. Copy own logit+index tiles to gathered DFBs at slot 0 (collector = group 0)
    {
        uint32_t own_val_l1 = dfb_topk_val.get_read_ptr();
        uint32_t own_ind_l1 = dfb_index.get_read_ptr();

        dfb_gathered_val.reserve_back(num_groups);
        dfb_gathered_ind.reserve_back(num_groups);

        uint32_t gathered_val_base = dfb_gathered_val.get_write_ptr();
        uint32_t gathered_ind_base = dfb_gathered_ind.get_write_ptr();

        volatile tt_l1_ptr uint32_t* src_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_val_l1);
        volatile tt_l1_ptr uint32_t* dst_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gathered_val_base);
        for (uint32_t w = 0; w < tile_u32; w++) {
            dst_val[w] = src_val[w];
        }

        volatile tt_l1_ptr uint32_t* src_ind = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_ind_l1);
        volatile tt_l1_ptr uint32_t* dst_ind = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gathered_ind_base);
        for (uint32_t w = 0; w < tile_u32; w++) {
            dst_ind[w] = src_ind[w];
        }

        dfb_topk_val.pop_front(1);
        dfb_index.pop_front(1);
    }

    // 5. Wait for other 3 workers' topk results
    Semaphore<> topk_sem(sem::topk_ready);
    topk_sem.wait(num_groups - 1);
    topk_sem.set(0);

    dfb_gathered_val.push_back(num_groups);
    dfb_gathered_ind.push_back(num_groups);

    // 6. Wait for compute to produce final output (2 tiles in final_out)
    dfb_final_out.wait_front(2);
    uint32_t final_out_l1 = dfb_final_out.get_read_ptr();

    // 7. Produce dispatch outputs: indices (uint16 RM) + weights (bf16 RM)
    constexpr uint32_t data_size = k_padded * 2;

    // Use dispatch as scratch storage.
    dfb_dispatch.reserve_back(1);
    uint32_t scratch = dfb_dispatch.get_write_ptr();
    uint32_t idx_base = scratch;
    uint32_t wgt_base = scratch + 32 * k_padded * 2;

    volatile tt_l1_ptr uint16_t* idx_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_base);
    volatile tt_l1_ptr uint16_t* wgt_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wgt_base);

    // src0 = weights tile (tile 0), src1 = indices tile (tile 1)
    volatile tt_l1_ptr uint16_t* src0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(final_out_l1);
    volatile tt_l1_ptr uint16_t* src1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(final_out_l1 + tile_size);

    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < topk_k; col++) {
            uint32_t fi = tile_elem_idx(row, col);
            wgt_buf[row * k_padded + col] = src0[fi];
            idx_buf[row * k_padded + col] = static_cast<uint16_t>(bf16_to_f32(src1[fi]));
        }
        for (uint32_t col = topk_k; col < k_padded; col++) {
            wgt_buf[row * k_padded + col] = 0;
            idx_buf[row * k_padded + col] = 0;
        }
    }

    // Complete the DFB reserve/push lifecycle
    dfb_dispatch.push_back(1);

    const auto idx_ag = TensorAccessor(tensor::indices_rm);
    const auto wgt_ag = TensorAccessor(tensor::weights_rm);
    for (uint32_t p = 0; p < 32; p++) {
        noc.async_write(CoreLocalMem<uint32_t>(idx_base + p * data_size), idx_ag, data_size, {}, {.page_id = p});
        noc.async_write(CoreLocalMem<uint32_t>(wgt_base + p * data_size), wgt_ag, data_size, {}, {.page_id = p});
    }
    noc.async_write_barrier();

    // Clean up DFB lifecycle
    dfb_dispatch.wait_front(1);
    dfb_dispatch.pop_front(1);

    dfb_final_out.pop_front(2);
}
