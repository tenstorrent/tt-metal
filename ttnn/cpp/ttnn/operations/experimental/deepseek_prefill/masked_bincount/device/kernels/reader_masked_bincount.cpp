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
// (is_initializer = true) and once for NCRISC (is_initializer = false). They
// share a single output histogram buffer (cb_out) in L1 and cooperate through
// semaphores to parallelise the work. The kernel runs in three phases:
//
// Phase 1 — Parallel page reads:
//   Both RISCs read their assigned portion of the shard into separate
//   input CBs (cb_in_brisc / cb_in_ncrisc). The shard's rows are split roughly
//   in half: BRISC gets h_brisc rows starting at h_start, NCRISC gets h_ncrisc
//   rows starting at h_start + h_brisc. All reads are issued together and
//   overlap with phase-2 initialisation.
//
// Phase 2 — Local histogram counting:
//   BRISC (the initializer) zeroes the shared histogram buffer in cb_out,
//   fetches the expert mask into cb_mask, then signals NCRISC via init_sem.
//   NCRISC waits for init_sem before proceeding. Both RISCs then iterate their
//   input rows: for each UINT16 expert index that passes the bounds check
//   (< n_routed_experts) and the mask check (mask[expert_idx] != 0), the count
//   is incremented atomically using noc_semaphore_inc on the local L1 address.
//   This is safe because semaphore increments are atomic even when both RISCs
//   target the same word. After counting, each RISC increments done_sem and
//   waits for the atomic barrier.
//
// Phase 3 — Tree reduction (BRISC only):
//   After both RISCs on a core finish (done_sem reaches 2), BRISC participates
//   in a binary-tree reduction across cores. The tree is structured so that
//   core i receives from children at indices i + 2^L for successive levels L.
//   At each level, BRISC waits for the child's gather_sem signal, reads the
//   child's histogram from remote L1 into a temporary CB (cb_gather_tmp), and
//   element-wise adds it into the local histogram. After processing all
//   children, non-root cores signal their parent's gather_sem. The root core
//   (parent_noc_x == 0xFFFFFFFF) writes the final reduced histogram.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

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
    uint32_t out_addr = cb_out.get_write_ptr();
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

    // Phase 2: Local histogram counting (BRISC/NCRISC cooperate on same core)
    Semaphore<> init_sem(init_sem_idx);

    if constexpr (is_initializer) {
        volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
        for (uint32_t i = 0; i < n_routed_experts; i++) {
            counts[i] = 0;
        }
        noc.async_read(mask_accessor, CoreLocalMem<uint32_t>(mask_l1_addr), mask_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        init_sem.set(1);
    } else {
        noc.async_read_barrier();
        init_sem.wait(1);
    }

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
                // TODO: D2.0 has no wrapper for atomic-increment on an arbitrary L1 word
                // (the target here is a histogram bin, not a registered semaphore). The
                // NoC atomic-inc primitive is exposed only via Semaphore<>::up(noc, x, y, val),
                // which resolves a semaphore-id to an L1 address. Keeping the legacy free
                // function for this specific case is documented as an acceptable fallback.
                uint64_t noc_addr = get_noc_addr(my_noc_x, my_noc_y, out_addr + expert_idx * sizeof(uint32_t));
                noc_semaphore_inc(noc_addr, 1);
            }
        }
    }
    noc.async_atomic_barrier();

    // Atomically bump the local done_sem on this core. up(noc, x, y, val) performs a NoC
    // atomic increment even when the target is the current core, which is needed because
    // BRISC and NCRISC both increment the same semaphore.
    Semaphore<> done_sem(done_sem_idx);
    done_sem.up(noc, my_noc_x, my_noc_y, 1);
    noc.async_atomic_barrier();

    // Phase 3: Tree reduction — BRISC only
    if constexpr (is_initializer) {
        done_sem.wait_min(2);

        uint32_t num_receive = get_arg_val<uint32_t>(4);
        uint32_t parent_noc_x = get_arg_val<uint32_t>(5);
        uint32_t parent_noc_y = get_arg_val<uint32_t>(6);

        Semaphore<> gather_sem(gather_sem_idx);

        uint32_t tmp_addr = cb_gather_tmp_obj.get_write_ptr();
        volatile tt_l1_ptr uint32_t* local_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

        // Wait for ALL children to signal before reading any.
        // A single gather_sem counter does not identify WHICH child signaled,
        // so we must wait for all num_receive increments to guarantee every
        // child's histogram is finalized before reading.
        gather_sem.wait_min(num_receive);
        for (uint32_t level = 0; level < num_receive; level++) {
            uint32_t child_noc_x = get_arg_val<uint32_t>(7 + level * 2);
            uint32_t child_noc_y = get_arg_val<uint32_t>(7 + level * 2 + 1);

            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(tmp_addr),
                output_page_size,
                {.noc_x = child_noc_x, .noc_y = child_noc_y, .addr = out_addr},
                {});
            noc.async_read_barrier();

            volatile tt_l1_ptr uint32_t* remote_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tmp_addr);
            for (uint32_t i = 0; i < n_routed_experts; i++) {
                local_hist[i] += remote_hist[i];
            }
        }

        if (parent_noc_x != 0xFFFFFFFF) {
            // Bump parent's gather_sem (a real semaphore id resolved on each core).
            gather_sem.up(noc, parent_noc_x, parent_noc_y, 1);
            noc.async_atomic_barrier();
        } else {
            noc.async_write(CoreLocalMem<uint32_t>(out_addr), dst_accessor, output_page_size, {}, {.page_id = 0});
            noc.async_write_barrier();
        }
    }
}
