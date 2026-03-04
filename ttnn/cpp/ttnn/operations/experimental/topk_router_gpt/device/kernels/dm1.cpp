// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
#include "api/dataflow/dataflow_api.h"

inline uint32_t tile_elem_idx(uint32_t row, uint32_t col) {
    uint32_t face = ((row >= 16) ? 2u : 0u) + ((col >= 16) ? 1u : 0u);
    return face * 256 + (row & 15) * 16 + (col & 15);
}

inline uint16_t f32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    return (uint16_t)(c.u >> 16);
}

// Generate one index tile where every element at (row, col) = bf16((float)(base + col))
void generate_index_tile(volatile tt_l1_ptr uint32_t* tile32, uint32_t base) {
    uint32_t left_packed[8], right_packed[8];
    for (uint32_t w = 0; w < 8; w++) {
        uint16_t lo = f32_to_bf16((float)(base + 2 * w));
        uint16_t hi = f32_to_bf16((float)(base + 2 * w + 1));
        left_packed[w] = ((uint32_t)hi << 16) | lo;
        lo = f32_to_bf16((float)(base + 16 + 2 * w));
        hi = f32_to_bf16((float)(base + 16 + 2 * w + 1));
        right_packed[w] = ((uint32_t)hi << 16) | lo;
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
    uint32_t is_sender = get_arg_val<uint32_t>(0);
    uint32_t is_worker = get_arg_val<uint32_t>(1);
    uint32_t is_collector = get_arg_val<uint32_t>(2);
    uint32_t worker_phys_x = get_arg_val<uint32_t>(3);
    uint32_t worker_phys_y = get_arg_val<uint32_t>(4);
    uint32_t collector_phys_x = get_arg_val<uint32_t>(5);
    uint32_t collector_phys_y = get_arg_val<uint32_t>(6);
    uint32_t sem_partial_ready = get_arg_val<uint32_t>(7);
    uint32_t sem_topk_ready = get_arg_val<uint32_t>(8);
    uint32_t tile_size = get_arg_val<uint32_t>(9);
    uint32_t output_addr = get_arg_val<uint32_t>(10);
    uint32_t topk_k = get_arg_val<uint32_t>(11);
    uint32_t sender_slot = get_arg_val<uint32_t>(12);
    uint32_t worker_gather_slot = get_arg_val<uint32_t>(13);
    uint32_t n_tile_id = get_arg_val<uint32_t>(14);
    uint32_t num_groups = get_arg_val<uint32_t>(15);
    uint32_t untilize_out = get_arg_val<uint32_t>(16);
    uint32_t rm_page_size = get_arg_val<uint32_t>(17);

    constexpr uint32_t CB_PARTIAL_RECV = tt::CBIndex::c_2;
    constexpr uint32_t CB_LOCAL_OUT = tt::CBIndex::c_3;
    constexpr uint32_t CB_INDEX = tt::CBIndex::c_5;
    constexpr uint32_t CB_TOPK_VAL = tt::CBIndex::c_6;
    constexpr uint32_t CB_TOPK_IND = tt::CBIndex::c_7;
    constexpr uint32_t CB_GATHERED_VAL = tt::CBIndex::c_8;
    constexpr uint32_t CB_GATHERED_IND = tt::CBIndex::c_9;
    constexpr uint32_t CB_SOFTMAX_MASK = tt::CBIndex::c_12;
    constexpr uint32_t CB_BCAST_SCALER = tt::CBIndex::c_15;
    constexpr uint32_t CB_FINAL_OUT = tt::CBIndex::c_16;

    uint32_t tile_u32 = tile_size / sizeof(uint32_t);

    if (is_sender) {
        // ============================================================
        // SENDER PATH
        // ============================================================
        // Wait for compute to produce local matmul partial in CB3
        cb_wait_front(CB_LOCAL_OUT, 1);
        uint32_t local_out_l1 = get_read_ptr(CB_LOCAL_OUT);

        // NOC write partial tile to worker's CB_PARTIAL_RECV at correct slot
        // Worker's CB2 base + sender_slot * tile_size
        uint32_t worker_recv_l1 = get_write_ptr(CB_PARTIAL_RECV) + sender_slot * tile_size;
        uint64_t worker_recv_noc = get_noc_addr(worker_phys_x, worker_phys_y, worker_recv_l1);
        noc_async_write(local_out_l1, worker_recv_noc, tile_size);
        noc_async_write_barrier();

        // Signal worker that this sender's partial is ready
        uint64_t worker_sem_noc = get_noc_addr(worker_phys_x, worker_phys_y, get_semaphore(sem_partial_ready));
        noc_semaphore_inc(worker_sem_noc, 1);

        cb_pop_front(CB_LOCAL_OUT, 1);
        // Sender is done
        return;
    }

    // ============================================================
    // WORKER PATH (including collector)
    // ============================================================

    // 1. Generate index template tile (expert_base + 0..31)
    //    Can overlap with DM0 weight reads and compute matmul
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
            uint32_t neg_inf_packed = 0xFF80FF80u;
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
            uint32_t one_packed = 0x3F803F80u;
            for (uint32_t i = 0; i < tile_u32; i++) {
                scaler32[i] = one_packed;
            }
        }
        cb_push_back(CB_BCAST_SCALER, 1);
    }

    // 2. Reserve space in CB2 for the 2 incoming partial tiles (must call
    //    cb_reserve_back before cb_push_back, even though data arrives via NOC)
    cb_reserve_back(CB_PARTIAL_RECV, 2);

    // Wait for both senders' partials to arrive
    volatile tt_l1_ptr uint32_t* partial_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_partial_ready));
    noc_semaphore_wait(partial_sem, 2);
    noc_semaphore_set(partial_sem, 0);

    // Push the 2 received partial tiles to CB2 for compute to consume
    cb_push_back(CB_PARTIAL_RECV, 2);

    // 3. Wait for compute to produce logit output (CB6).
    //    Index tile stays in CB_INDEX (no copy through compute).
    cb_wait_front(CB_TOPK_VAL, 1);

    if (!is_collector) {
        // ---- Non-collector worker: send logit+index tiles to collector ----
        uint32_t val_l1 = get_read_ptr(CB_TOPK_VAL);
        uint32_t ind_l1 = get_read_ptr(CB_INDEX);

        // Write values to collector's CB_GATHERED_VAL at slot worker_gather_slot
        uint32_t coll_val_base = get_write_ptr(CB_GATHERED_VAL);
        uint32_t coll_val_dst_l1 = coll_val_base + worker_gather_slot * tile_size;
        uint64_t coll_val_noc = get_noc_addr(collector_phys_x, collector_phys_y, coll_val_dst_l1);
        noc_async_write(val_l1, coll_val_noc, tile_size);

        // Write indices to collector's CB_GATHERED_IND at slot worker_gather_slot
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
        // Non-collector worker is done
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

        // Copy own results to slot 0
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

    // Push all gathered tiles to compute
    cb_push_back(CB_GATHERED_VAL, num_groups);
    cb_push_back(CB_GATHERED_IND, num_groups);

    // 6. Wait for compute to produce final output (2 tiles in CB16)
    cb_wait_front(CB_FINAL_OUT, 2);
    uint32_t final_out_l1 = get_read_ptr(CB_FINAL_OUT);

    // 7. Write output to DRAM
    if (untilize_out) {
        // Software untilize: rearrange 2 tiles (face layout) → row-major in CB17
        constexpr uint32_t CB_RM_OUT = tt::CBIndex::c_17;
        cb_reserve_back(CB_RM_OUT, 2);
        uint32_t rm_buf = get_write_ptr(CB_RM_OUT);

        volatile tt_l1_ptr uint16_t* src0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(final_out_l1);
        volatile tt_l1_ptr uint16_t* src1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(final_out_l1 + tile_size);
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rm_buf);

        for (uint32_t row = 0; row < 32; row++) {
            for (uint32_t col = 0; col < 32; col++) {
                uint32_t face = ((row >= 16) ? 2u : 0u) + ((col >= 16) ? 1u : 0u);
                uint32_t idx = face * 256 + (row & 15) * 16 + (col & 15);
                dst[row * 64 + col] = src0[idx];
                dst[row * 64 + 32 + col] = src1[idx];
            }
        }

        // Write 32 row-major pages (128 bytes each) to DRAM
        const InterleavedAddrGen</*DRAM=*/true> rm_addrgen = {
            .bank_base_address = output_addr, .page_size = rm_page_size};
        for (uint32_t page = 0; page < 32; page++) {
            uint64_t noc_addr = get_noc_addr(page, rm_addrgen);
            noc_async_write(rm_buf + page * rm_page_size, noc_addr, rm_page_size);
        }
        noc_async_write_barrier();
    } else {
        // Tile-based output path
        const InterleavedAddrGenFast</*DRAM=*/true> output_addrgen = {
            .bank_base_address = output_addr, .page_size = tile_size, .data_format = get_dataformat(CB_FINAL_OUT)};
        noc_async_write_tile(0, output_addrgen, final_out_l1);
        noc_async_write_tile(1, output_addrgen, final_out_l1 + tile_size);
        noc_async_write_barrier();
    }
    cb_pop_front(CB_FINAL_OUT, 2);
}
