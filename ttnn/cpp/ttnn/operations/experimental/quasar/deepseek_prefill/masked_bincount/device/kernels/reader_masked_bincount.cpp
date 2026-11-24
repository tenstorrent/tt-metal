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
// share a single output histogram buffer (out) in L1 and cooperate through
// semaphores to parallelise the work. The kernel runs in three phases:
//
// Phase 1 — Parallel page reads:
//   Both RISCs read their assigned portion of the shard into separate
//   input scratchpads (in_brisc / in_ncrisc). The shard's rows are split roughly
//   in half: BRISC gets h_brisc rows starting at h_start, NCRISC gets h_ncrisc
//   rows starting at h_start + h_brisc. All reads are issued together and
//   overlap with phase-2 initialisation.
//
// Phase 2 — Local histogram counting:
//   BRISC fetches the expert mask into mask and signals NCRISC via init_sem. Each RISC counts its rows'
//   picks that pass the bounds check (< n_routed_experts) and the mask check (mask[expert] >= 0) into its own
//   RISC-local array (plain increments). NCRISC publishes its array to ncrisc_partial and bumps done_sem;
//   BRISC writes local + NCRISC partial into the core's output page (out).
//
// Phase 3 — Flat reduction (BRISC only):
//   Every core but the collector (core 0) signals the collector's gather_sem once its page is final. The
//   collector waits for all of them, reads every other core's page at once into gather_tmp, sums, and
//   writes the histogram.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include "ckernel.h"

// L1 address the CPU accesses a scratchpad through: on Quasar DM the uncached alias, as DataflowBuffer pointers are,
// since the CPU caches are not coherent with NoC writes to L1; on WH/BH the plain address.
template <typename T>
uint32_t cpu_l1_addr(const Scratchpad<T>& scratch) {
    uintptr_t addr = local_mem::detail::AddressAccess::address(scratch.local_mem());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    return static_cast<uint32_t>(addr);
}

void kernel_main() {
    Noc noc;

    uint32_t h_start = get_arg(args::h_start);

    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t output_page_size = get_arg(args::output_page_size);
    constexpr uint32_t h_count = get_arg(args::h_count);
    constexpr uint32_t num_experts_per_token = get_arg(args::num_experts_per_token);
    constexpr uint32_t n_routed_experts = get_arg(args::n_routed_experts);
    constexpr bool is_initializer = (bool)get_arg(args::is_initializer);
    constexpr uint32_t mask_page_size = get_arg(args::mask_page_size);
    constexpr uint32_t tile_h = get_arg(args::tile_h);

    const auto src_accessor = TensorAccessor(tensor::src);
    const auto dst_accessor = TensorAccessor(tensor::dst);
    const auto mask_accessor = TensorAccessor(tensor::mask);

    Scratchpad<uint32_t> in(scratch::in);
    Scratchpad<uint32_t> out(scratch::out);
    Scratchpad<uint32_t> mask_scratch(scratch::mask);
    Scratchpad<uint32_t> gather_tmp(scratch::gather_tmp);
    // L1 page where NCRISC publishes its partial histogram for BRISC (counts are kept RISC-local, not atomic).
    Scratchpad<uint32_t> ncrisc_partial_scratch(scratch::ncrisc_partial);

    uint32_t in_base_addr = cpu_l1_addr(in);
    uint32_t out_addr = cpu_l1_addr(out);
    uint32_t mask_l1_addr = cpu_l1_addr(mask_scratch);

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

    // Phase 2: Local histogram counting. Each RISC counts into its own RISC-local array (plain increments, no
    // NoC atomics); NCRISC then publishes its array to an L1 page and BRISC adds it into the output page.
    Semaphore<> init_sem(sem::init);

    if constexpr (is_initializer) {
        noc.async_read(mask_accessor, CoreLocalMem<uint32_t>(mask_l1_addr), mask_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        init_sem.set(1);
    } else {
        noc.async_read_barrier();
        init_sem.wait(1);
    }
    invalidate_l1_cache();

    const int32_t* mask = reinterpret_cast<const int32_t*>(mask_l1_addr);

    const uint8_t my_noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[my_noc_id];
    const uint32_t my_noc_y = my_y[my_noc_id];

    static uint32_t local_counts[n_routed_experts];
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        local_counts[i] = 0;
    }

    // Untile in place: a 32x32 tile stores four 16x16 faces (face0,face1,face2,face3, row-major within
    // each). Experts occupy columns [0, num_experts_per_token) which fall in the left faces (0 and 2),
    // so element (row r within a tile, column c) sits at uint16 offset:
    //   (r / 16) * 2 * 256 + (r % 16) * 16 + c
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t FACE_SIZE = FACE_DIM * FACE_DIM;  // 256 uint16 per face
    const uint32_t tile_elems = input_page_size / 2;     // uint16 per tile page
    const uint16_t* in_u16 = reinterpret_cast<const uint16_t*>(in_base_addr);
    for (uint32_t j = 0; j < h_count; j++) {
        uint32_t global_row = h_start + j;
        uint32_t tile_local = global_row / tile_h - first_tile;
        uint32_t within = global_row % tile_h;
        uint32_t row_base =
            tile_local * tile_elems + (within / FACE_DIM) * 2 * FACE_SIZE + (within % FACE_DIM) * FACE_DIM;
        for (uint32_t w = 0; w < num_experts_per_token; w++) {
            uint32_t expert_idx = in_u16[row_base + w];
            if (expert_idx < n_routed_experts && mask[expert_idx] >= 0) {
                local_counts[expert_idx]++;
            }
        }
    }

    Semaphore<> done_sem(sem::done);
    volatile tt_l1_ptr uint32_t* ncrisc_partial =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpu_l1_addr(ncrisc_partial_scratch));
    if constexpr (!is_initializer) {
        for (uint32_t i = 0; i < n_routed_experts; i++) {
            ncrisc_partial[i] = local_counts[i];
        }
        // The stores must land in L1 before BRISC is told to read them (an MMIO/NoC store can race ahead of
        // L1 stores): a blocking load of the last word drains them first.
        (void)ckernel::load_blocking(ncrisc_partial + (n_routed_experts - 1));
        done_sem.up(noc, my_noc_x, my_noc_y, 1);
        noc.async_atomic_barrier();
        return;
    }

    // BRISC from here on.
    done_sem.wait_min(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* local_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        local_hist[i] = local_counts[i] + ncrisc_partial[i];
    }

    // Phase 3: flat reduction. The collector reads every other core's histogram at once and sums them; the other
    // cores only signal it.
    uint32_t num_receive = get_arg(args::num_receive);
    uint32_t parent_noc_x = get_arg(args::parent_noc_x);
    uint32_t parent_noc_y = get_arg(args::parent_noc_y);
    Semaphore<> gather_sem(sem::gather);

    if (parent_noc_x != 0xFFFFFFFF) {
        (void)ckernel::load_blocking(local_hist + (n_routed_experts - 1));
        gather_sem.up(noc, parent_noc_x, parent_noc_y, 1);
        noc.async_atomic_barrier();
        return;
    }

    uint32_t tmp_addr = cpu_l1_addr(gather_tmp);
    gather_sem.wait_min(num_receive);
    for (uint32_t c = 0; c < num_receive; c++) {
        uint32_t child_noc_x = get_arg(args::children_noc)[c * 2];
        uint32_t child_noc_y = get_arg(args::children_noc)[c * 2 + 1];
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(tmp_addr + c * output_page_size),
            output_page_size,
            {.noc_x = child_noc_x, .noc_y = child_noc_y, .addr = out_addr},
            {});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        uint32_t sum = local_counts[i] + ncrisc_partial[i];
        for (uint32_t c = 0; c < num_receive; c++) {
            sum += reinterpret_cast<const uint32_t*>(tmp_addr + c * output_page_size)[i];
        }
        local_hist[i] = sum;
    }
    noc.async_write(CoreLocalMem<uint32_t>(out_addr), dst_accessor, output_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();
}
