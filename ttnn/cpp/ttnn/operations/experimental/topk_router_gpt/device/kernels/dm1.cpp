// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM1 Kernel: Inter-Core Communication + Helper Tile Generation + Output Writer
// (RISCV_1, NOC 1)
//
// Three paths:
//   Sender: wait for compute partial → NOC write to worker's CB2 → signal sem
//   Worker (non-collector): generate index tile → wait for sender partials →
//     push to compute → wait for logit+index output → NOC write to collector's CB8/CB9 → signal sem
//   Collector: generate index/mask/scaler tiles → wait for sender partials →
//     push to compute → wait for logit+index output → copy own to gathered →
//     wait for other workers' results → push gathered to compute →
//     wait for final output → write to DRAM

#include <cstdint>
#include <cstring>
#include "api/dataflow/dataflow_api.h"

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
    // Compile-time args
    constexpr uint32_t tile_size = get_named_compile_time_arg_val("tile_size_bf16");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t topk_k = get_named_compile_time_arg_val("topk_k");
    constexpr uint32_t k_padded = get_named_compile_time_arg_val("k_padded");
    constexpr uint32_t collector_phys_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_phys_y = get_named_compile_time_arg_val("collector_physical_y");

    // Runtime args
    uint32_t is_sender = get_arg_val<uint32_t>(0);
    uint32_t is_worker = get_arg_val<uint32_t>(1);
    uint32_t is_collector = get_arg_val<uint32_t>(2);
    uint32_t worker_phys_x = get_arg_val<uint32_t>(3);
    uint32_t worker_phys_y = get_arg_val<uint32_t>(4);
    uint32_t sem_partial_ready = get_arg_val<uint32_t>(5);
    uint32_t sem_topk_ready = get_arg_val<uint32_t>(6);
    uint32_t sender_slot = get_arg_val<uint32_t>(7);
    uint32_t worker_gather_slot = get_arg_val<uint32_t>(8);
    uint32_t n_tile_id = get_arg_val<uint32_t>(9);
    uint32_t indices_rm_addr = get_arg_val<uint32_t>(10);
    uint32_t weights_rm_addr = get_arg_val<uint32_t>(11);
    uint32_t aligned_page_size = get_arg_val<uint32_t>(12);
    uint32_t vchannel = get_arg_val<uint32_t>(13);

    constexpr uint32_t CB_PARTIAL_RECV = tt::CBIndex::c_2;
    constexpr uint32_t CB_LOCAL_OUT = tt::CBIndex::c_3;
    constexpr uint32_t CB_INDEX = tt::CBIndex::c_5;
    constexpr uint32_t CB_TOPK_VAL = tt::CBIndex::c_6;
    constexpr uint32_t CB_GATHERED_VAL = tt::CBIndex::c_8;
    constexpr uint32_t CB_GATHERED_IND = tt::CBIndex::c_9;
    constexpr uint32_t CB_SOFTMAX_MASK = tt::CBIndex::c_12;
    constexpr uint32_t CB_BCAST_SCALER = tt::CBIndex::c_15;
    constexpr uint32_t CB_FINAL_OUT = tt::CBIndex::c_16;
    constexpr uint32_t CB_DISPATCH = tt::CBIndex::c_19;

    constexpr uint32_t tile_u32 = tile_size / sizeof(uint32_t);

    // Pre-compute the CB_PARTIAL_RECV base address.
    // All cores have identical CB layout (CB0-CB3 same sizes), so the base
    // address of CB2 is at the same L1 offset on every core. We read it from
    // our own CB interface — valid because we allocated CB2 on all cores.
    // Then we use this stable base + slot offset for NOC writes to the worker.
    const uint32_t cb2_base_addr = get_write_ptr(CB_PARTIAL_RECV);

    if (is_sender) {
        // ============================================================
        // SENDER PATH
        // ============================================================
        cb_wait_front(CB_LOCAL_OUT, 1);
        uint32_t local_out_l1 = get_read_ptr(CB_LOCAL_OUT);

        // NOC write partial tile to worker's CB2 at our sender_slot.
        // cb2_base_addr is the same L1 address on all cores due to uniform CB layout.
        uint32_t worker_recv_l1 = cb2_base_addr + sender_slot * tile_size;
        uint64_t worker_recv_noc = get_noc_addr(worker_phys_x, worker_phys_y, worker_recv_l1);

        noc_async_write(local_out_l1, worker_recv_noc, tile_size);
        noc_async_write_barrier();

        // Signal worker that this sender's partial is ready
        uint64_t worker_sem_noc = get_noc_addr(worker_phys_x, worker_phys_y, get_semaphore(sem_partial_ready));
        noc_semaphore_inc(worker_sem_noc, 1);

        cb_pop_front(CB_LOCAL_OUT, 1);
        return;
    }

    // ============================================================
    // WORKER PATH (including collector)
    // ============================================================

    // 1. Generate index template tile (expert_base + 0..31)
    uint32_t expert_base = n_tile_id * 32;
    cb_reserve_back(CB_INDEX, 1);
    {
        uint32_t index_l1 = get_write_ptr(CB_INDEX);
        volatile tt_l1_ptr uint32_t* tile32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1);
        generate_index_tile(tile32, expert_base);
    }
    cb_push_back(CB_INDEX, 1);

    // Collector also generates softmax helper tiles
    if (is_collector) {
        // Softmax mask: cols 0..k-1 = 0.0, cols k..31 = -inf
        cb_reserve_back(CB_SOFTMAX_MASK, 1);
        {
            uint32_t mask_l1 = get_write_ptr(CB_SOFTMAX_MASK);
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
        cb_push_back(CB_SOFTMAX_MASK, 1);

        // Broadcast scaler: all 1.0
        cb_reserve_back(CB_BCAST_SCALER, 1);
        {
            uint32_t scaler_l1 = get_write_ptr(CB_BCAST_SCALER);
            volatile tt_l1_ptr uint32_t* scaler32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_l1);
            constexpr uint32_t one_packed = 0x3F803F80u;
            for (uint32_t i = 0; i < tile_u32; i++) {
                scaler32[i] = one_packed;
            }
        }
        cb_push_back(CB_BCAST_SCALER, 1);
    }

    // 2. Reserve space in CB2 for the 2 incoming partial tiles
    cb_reserve_back(CB_PARTIAL_RECV, 2);

    // Wait for both senders' partials to arrive
    volatile tt_l1_ptr uint32_t* partial_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_partial_ready));
    noc_semaphore_wait(partial_sem, 2);
    noc_semaphore_set(partial_sem, 0);

    cb_push_back(CB_PARTIAL_RECV, 2);

    // 3. Wait for compute to produce logit output (CB6).
    cb_wait_front(CB_TOPK_VAL, 1);

    if (!is_collector) {
        // ---- Non-collector worker: send logit+index tiles to collector ----
        uint32_t val_l1 = get_read_ptr(CB_TOPK_VAL);
        uint32_t ind_l1 = get_read_ptr(CB_INDEX);

        // Use our own CB8/CB9 base addresses — identical L1 layout on all worker cores
        uint32_t coll_val_base = get_write_ptr(CB_GATHERED_VAL);
        uint32_t coll_val_dst_l1 = coll_val_base + worker_gather_slot * tile_size;
        uint64_t coll_val_noc = get_noc_addr(collector_phys_x, collector_phys_y, coll_val_dst_l1);
        noc_async_write(val_l1, coll_val_noc, tile_size);

        uint32_t coll_ind_base = get_write_ptr(CB_GATHERED_IND);
        uint32_t coll_ind_dst_l1 = coll_ind_base + worker_gather_slot * tile_size;
        uint64_t coll_ind_noc = get_noc_addr(collector_phys_x, collector_phys_y, coll_ind_dst_l1);
        noc_async_write(ind_l1, coll_ind_noc, tile_size);

        noc_async_write_barrier();

        // Signal collector
        uint64_t coll_sem_noc = get_noc_addr(collector_phys_x, collector_phys_y, get_semaphore(sem_topk_ready));
        noc_semaphore_inc(coll_sem_noc, 1);

        cb_pop_front(CB_TOPK_VAL, 1);
        cb_pop_front(CB_INDEX, 1);
        return;
    }

    // ============================================================
    // COLLECTOR PATH (continues from worker)
    // ============================================================

    // 4. Copy own logit+index tiles to gathered CB at slot 0 (collector = group 0)
    {
        uint32_t own_val_l1 = get_read_ptr(CB_TOPK_VAL);
        uint32_t own_ind_l1 = get_read_ptr(CB_INDEX);

        cb_reserve_back(CB_GATHERED_VAL, num_groups);
        cb_reserve_back(CB_GATHERED_IND, num_groups);

        uint32_t gathered_val_base = get_write_ptr(CB_GATHERED_VAL);
        uint32_t gathered_ind_base = get_write_ptr(CB_GATHERED_IND);

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

        cb_pop_front(CB_TOPK_VAL, 1);
        cb_pop_front(CB_INDEX, 1);
    }

    // 5. Wait for other 3 workers' topk results
    volatile tt_l1_ptr uint32_t* topk_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_topk_ready));
    noc_semaphore_wait(topk_sem, num_groups - 1);
    noc_semaphore_set(topk_sem, 0);

    cb_push_back(CB_GATHERED_VAL, num_groups);
    cb_push_back(CB_GATHERED_IND, num_groups);

    // 6. Wait for compute to produce final output (2 tiles in CB16)
    cb_wait_front(CB_FINAL_OUT, 2);
    uint32_t final_out_l1 = get_read_ptr(CB_FINAL_OUT);

    // 7. Produce dispatch outputs: indices (uint16 RM) + weights (bf16 RM)
    constexpr uint32_t data_size = k_padded * 2;  // actual data bytes per page

    // Use a local L1 scratch area. We use CB_DISPATCH as scratch storage.
    // Reserve, write, then push+pop to maintain CB contract.
    cb_reserve_back(CB_DISPATCH, 1);
    uint32_t scratch = get_write_ptr(CB_DISPATCH);
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

    // Complete the CB reserve/push lifecycle
    cb_push_back(CB_DISPATCH, 1);

    const InterleavedAddrGen<true> idx_ag = {.bank_base_address = indices_rm_addr, .page_size = aligned_page_size};
    const InterleavedAddrGen<true> wgt_ag = {.bank_base_address = weights_rm_addr, .page_size = aligned_page_size};
    for (uint32_t p = 0; p < 32; p++) {
        noc_async_write(idx_base + p * data_size, get_noc_addr(p, idx_ag), data_size);
        noc_async_write(wgt_base + p * data_size, get_noc_addr(p, wgt_ag), data_size);
    }
    noc_async_write_barrier();

    // Clean up CB lifecycle
    cb_wait_front(CB_DISPATCH, 1);
    cb_pop_front(CB_DISPATCH, 1);

    cb_pop_front(CB_FINAL_OUT, 2);
}
