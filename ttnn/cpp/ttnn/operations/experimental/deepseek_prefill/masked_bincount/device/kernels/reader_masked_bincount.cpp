// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Masked Bincount Kernel
//
// Counts how many tokens are routed to each expert, producing a per-expert
// histogram masked by which experts are present on this device.
//
// Inputs:
//   - input [sp_dim, topk_dim]: UINT16 height-sharded tensor of expert indices
//     selected for each token (one row per token, one column per top-k slot).
//   - expert_dispatch_table [n_routed_experts]: INT32 tensor mapping experts to
//     chip IDs. Negative (-1) means absent (skip), non-negative values (chip IDs)
//     mean present (count).
//
// Output:
//   - histogram [n_routed_experts]: UINT32 count of token assignments per expert.
//
// The same kernel source is compiled twice per core: once for BRISC
// (is_initializer = true) and once for NCRISC (is_initializer = false). Each
// RISC counts into its own L1 histogram (BRISC: cb_out, NCRISC: cb_gather_tmp)
// and they cooperate through semaphores. The kernel runs in three phases:
//
// Phase 1 — Parallel page reads:
//   Both RISCs read their assigned portion of the shard into separate
//   input CBs (cb_in_brisc / cb_in_ncrisc). The shard's rows are split roughly
//   in half: BRISC gets h_brisc rows starting at h_start, NCRISC gets h_ncrisc
//   rows starting at h_start + h_brisc. All reads are issued together and
//   overlap with phase-2 initialisation.
//
// Phase 2 — Local histogram counting:
//   Each RISC zeroes its own histogram. BRISC (the initializer) fetches the
//   expert mask into cb_mask, then signals NCRISC via init_sem; NCRISC waits for
//   it. Both RISCs then iterate their input rows: for each UINT16 expert index
//   that passes the bounds check (< n_routed_experts) and the mask check
//   (mask[expert_idx] >= 0), the RISC increments its own histogram entry with a
//   plain L1 increment (no NoC atomics). NCRISC then signals done_sem and exits.
//
// Phase 3 — Flat column reduction (BRISC only):
//   BRISC waits for done_sem, adds NCRISC's histogram into cb_out (the core's
//   published histogram) and signals the gather_sem of every reducer core. The
//   first n_routed_experts / 16 cores are reducers: reducer r waits for every
//   core's signal, reads expert column r (16 counts, 64 bytes) of every core's
//   histogram in one batch into c_5, sums it and writes those 16 counts to the
//   output (the last reducer's column can be shorter when n_routed_experts is not
//   a multiple of 16). (A binary tree had ~21 serial remote reads on its critical
//   path.)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

void kernel_main() {
    Noc noc;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t mask_addr = get_arg_val<uint32_t>(2);
    uint32_t h_start = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t h_count = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts_per_token = get_compile_time_arg_val(5);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(6);
    constexpr bool is_initializer = (bool)get_compile_time_arg_val(7);
    constexpr uint32_t init_sem_idx = get_compile_time_arg_val(8);
    constexpr uint32_t done_sem_idx = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_idx = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gather_tmp = get_compile_time_arg_val(11);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(15);
    constexpr uint32_t mask_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t tile_h = get_compile_time_arg_val(17);

    constexpr uint32_t src_accessor_offset = 18;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t dst_accessor_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_args_ct = TensorAccessorArgs<dst_accessor_offset>();
    const auto dst_accessor = TensorAccessor(dst_args_ct, dst_addr);

    constexpr uint32_t mask_accessor_offset = dst_args_ct.next_compile_time_args_offset();
    constexpr auto mask_args_ct = TensorAccessorArgs<mask_accessor_offset>();
    const auto mask_accessor = TensorAccessor(mask_args_ct, mask_addr);

    CircularBuffer cb_in(cb_id_in);
    CircularBuffer cb_out(cb_id_out);
    CircularBuffer cb_mask_obj(cb_mask);
    CircularBuffer cb_gather_tmp_obj(cb_gather_tmp);

    uint32_t in_base_addr = cb_in.get_write_ptr();
    // BRISC counts into cb_out, NCRISC into cb_gather_tmp (each RISC its own histogram: plain L1 increments, no
    // atomics); BRISC then adds NCRISC's into cb_out, the core's published histogram
    uint32_t out_addr = is_initializer ? cb_out.get_write_ptr() : cb_gather_tmp_obj.get_write_ptr();
    uint32_t mask_l1_addr = cb_mask_obj.get_write_ptr();

    // Phase 1: Read the TILE pages covering this RISC's global row range [h_start, h_start + h_count).
    // The input is TILE-interleaved; page_id indexes 32-row tiles (width is padded to a single tile).
    // Row ranges are not tile-aligned in general, so adjacent RISCs/cores may read a shared boundary
    // tile — each reads independently and counts only its own rows.
    uint32_t first_tile = 0;
    if (h_count > 0) {
        first_tile = h_start / tile_h;
        uint32_t last_tile = (h_start + h_count - 1) / tile_h;
        uint32_t num_tiles = last_tile - first_tile + 1;
        for (uint32_t t = 0; t < num_tiles; t++) {
            noc.async_read(
                src_accessor,
                CoreLocalMem<uint32_t>(in_base_addr + t * input_page_size),
                input_page_size,
                {.page_id = first_tile + t},
                {});
        }
    }

    // Phase 2: local counting. BRISC fetches the mask and signals NCRISC (init_sem); each RISC zeroes and fills
    // its own histogram.
    Semaphore<> init_sem(init_sem_idx);
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        counts[i] = 0;
    }

    if constexpr (is_initializer) {
        noc.async_read(mask_accessor, CoreLocalMem<uint32_t>(mask_l1_addr), mask_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        init_sem.set(1);
    } else {
        noc.async_read_barrier();
        init_sem.wait(1);
    }
    invalidate_l1_cache();

    volatile tt_l1_ptr int32_t* mask = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mask_l1_addr);

    const uint8_t my_noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[my_noc_id];
    const uint32_t my_noc_y = my_y[my_noc_id];

    // Untile in place: a 32x32 tile stores four 16x16 faces (face0,face1,face2,face3, row-major within
    // each). Experts occupy columns [0, num_experts_per_token) which fall in the left faces (0 and 2),
    // so element (row r within a tile, column c) sits at uint16 offset:
    //   (r / 16) * 2 * 256 + (r % 16) * 16 + c
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t FACE_SIZE = FACE_DIM * FACE_DIM;  // 256 uint16 per face
    const uint32_t tile_elems = input_page_size / 2;     // uint16 per tile page
    volatile tt_l1_ptr uint16_t* in_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr);
    for (uint32_t j = 0; j < h_count; j++) {
        uint32_t global_row = h_start + j;
        uint32_t tile_local = global_row / tile_h - first_tile;
        uint32_t within = global_row % tile_h;
        uint32_t row_base =
            tile_local * tile_elems + (within / FACE_DIM) * 2 * FACE_SIZE + (within % FACE_DIM) * FACE_DIM;
        for (uint32_t w = 0; w < num_experts_per_token; w++) {
            uint32_t expert_idx = in_u16[row_base + w];
            if (expert_idx < n_routed_experts && mask[expert_idx] >= 0) {
                counts[expert_idx] += 1;
            }
        }
    }
    asm volatile("fence" ::: "memory");

    // Atomically bump the local done_sem on this core: NCRISC signals BRISC that its histogram is complete.
    Semaphore<> done_sem(done_sem_idx);
    if constexpr (!is_initializer) {
        done_sem.up(noc, my_noc_x, my_noc_y, 1);
        noc.async_atomic_barrier();
        return;
    }

    // Phase 3 (BRISC): fold NCRISC's histogram in, publish to every reducer core, then (reducers only) sum the
    // reducer's 16-expert column of every core's histogram and write it to the output.
    done_sem.wait_min(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* other =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_gather_tmp_obj.get_write_ptr());
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        counts[i] += other[i];
    }
    // Drain the fold's stores into L1 before the reducers NoC-read this histogram (an MMIO/NoC signal can race
    // ahead of earlier L1 stores): a blocking load of the last written word.
    asm volatile("fence" ::: "memory");
    (void)ckernel::load_blocking(counts + (n_routed_experts - 1));

    const uint32_t num_cores_rt = get_arg_val<uint32_t>(4);
    const uint32_t num_reducers = get_arg_val<uint32_t>(5);
    const uint32_t my_reducer = get_arg_val<uint32_t>(6);  // 0xFFFFFFFF: not a reducer
    constexpr uint32_t per_reducer = 16;                   // experts per reducer (one 64-byte column)
    constexpr uint32_t column_bytes = per_reducer * sizeof(uint32_t);  // c_5 stride per core
    Semaphore<> gather_sem(gather_sem_idx);
    // core c's NoC xy at args 7 + 2c; the reducers are the first num_reducers cores
    for (uint32_t r = 0; r < num_reducers; r++) {
        gather_sem.up(noc, get_arg_val<uint32_t>(7 + 2 * r), get_arg_val<uint32_t>(7 + 2 * r + 1), 1);
    }
    noc.async_atomic_barrier();
    if (my_reducer == 0xFFFFFFFF) {
        return;
    }
    gather_sem.wait_min(num_cores_rt);
    gather_sem.set(0);
    // stage every core's 64-byte column in c_5 (num_cores x 64 bytes)
    CircularBuffer cb_stage(tt::CBIndex::c_5);
    const uint32_t stage = cb_stage.get_write_ptr();
    const uint32_t col_off = my_reducer * column_bytes;
    // this reducer's experts, and the bytes it moves: the column, cut at the end of the histogram page
    const uint32_t col_experts = n_routed_experts - my_reducer * per_reducer < per_reducer
                                     ? n_routed_experts - my_reducer * per_reducer
                                     : per_reducer;
    const uint32_t xfer_bytes = output_page_size - col_off < column_bytes ? output_page_size - col_off : column_bytes;
    const uint32_t xfer_words = xfer_bytes / sizeof(uint32_t);
    for (uint32_t c = 0; c < num_cores_rt; c++) {
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(stage + c * column_bytes),
            xfer_bytes,
            {.noc_x = get_arg_val<uint32_t>(7 + 2 * c),
             .noc_y = get_arg_val<uint32_t>(7 + 2 * c + 1),
             .addr = out_addr + col_off},
            {});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    uint32_t sum[per_reducer];
    for (uint32_t e = 0; e < per_reducer; e++) {
        sum[e] = 0;
    }
    for (uint32_t c = 0; c < num_cores_rt; c++) {
        volatile tt_l1_ptr uint32_t* col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + c * column_bytes);
        for (uint32_t e = 0; e < col_experts; e++) {
            sum[e] += col[e];
        }
    }
    // stage the 16 sums in cb_gather_tmp (NCRISC's histogram, already folded in) and write them to the output
    const uint32_t res_addr = cb_gather_tmp_obj.get_write_ptr();
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res_addr);
    for (uint32_t e = 0; e < xfer_words; e++) {
        res[e] = sum[e];  // 0 past this reducer's experts (padding of the histogram page)
    }
    (void)ckernel::load_blocking(res + (xfer_words - 1));
    noc.async_write(
        CoreLocalMem<uint32_t>(res_addr), dst_accessor, xfer_bytes, {}, {.page_id = 0, .offset_bytes = col_off});
    noc.async_write_barrier();
}
