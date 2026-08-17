// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "experimental/kernel_args.h"

#include "sort_dataflow_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t start_core_physical_coord_x = get_arg(args::start_core_physical_coord_x);
    const uint32_t start_core_physical_coord_y = get_arg(args::start_core_physical_coord_y);
    const uint32_t end_core_physical_coord_x = get_arg(args::end_core_physical_coord_x);
    const uint32_t end_core_physical_coord_y = get_arg(args::end_core_physical_coord_y);
    // Number of worker cores this coordinator serves. Used as the exact-match target for
    // cores_to_coordinator_ready_sem.wait() and (indirectly, via number_of_confirmations = Wt / 2)
    // for the per-sub-stage done-sem waits.
    const uint32_t number_of_workers = get_arg(args::number_of_workers);
    // Number of NoC destinations for the go-signal multicast. Must match the number of cores in the
    // multicast rectangle (excluding the coordinator itself when the rectangle contains it). In the
    // partial-grid path the multicast rectangle covers a bounding box that may be strictly larger
    // than number_of_workers, so this is tracked as a separate value from the worker count above:
    // using number_of_workers here would drift the NoC's ack counter and hang the op. See
    // SortProgramFactorySingleRowMultiCore for the derivation of both quantities.
    const uint32_t num_multicast_dests = get_arg(args::num_multicast_dests);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr bool is_32_bit_data = get_arg(args::is_32_bit_data) == 1;
    constexpr uint32_t W_tile_bytes = get_arg(args::W_tile_bytes);
    constexpr uint32_t W_index_bytes = get_arg(args::W_index_bytes);
    constexpr uint32_t tile_width = get_arg(args::tile_width);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_H = 32;

    const auto input_tensor_addr_ger = TensorAccessor(tensor::input_tensor);
    const auto output_tensor_addr_gen = TensorAccessor(tensor::output_tensor);
    const auto output_index_tensor_addr_gen = TensorAccessor(tensor::output_index_tensor);

    Noc noc;
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_coord_value_row(dfb::rm_coord_value_row);
    DataflowBuffer rm_coord_index_row(dfb::rm_coord_index_row);
#else
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    // The TILE-branch passthrough below DMAs data raw from the input DRAM buffer into the input
    // buffer and back out to the output DRAM buffer. When the input dtype is promoted for the sort,
    // that buffer's entry is wider than the DRAM tile it mirrors, so its entry size would overrun
    // the source and destination tile boundaries. W_tile_bytes is sized for the raw input dtype, so
    // deriving the byte count from it copies the correct amount in every mode.
    constexpr uint32_t raw_input_tile_bytes = TILE_H * W_tile_bytes;
    const uint32_t index_tensor_tile_size = index_tensor_dfb.get_tile_size();
#endif

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(sem::coordinator_to_cores);
    // Two separate up-channels from the worker cores: the reader's per-row readiness ->
    // ready sem, the writer's per-pair confirmations -> done sem. They are kept on
    // distinct semaphores so each exact-match wait() below has its own monotonic target;
    // folded onto one shared counter, at a tile-row boundary (Ht >= 2) a fast reader's
    // next-row readiness could land during the confirmation window and push the counter
    // past the done target, so the wait would never match and the op would deadlock.
    Semaphore<> cores_to_coordinator_ready_sem(sem::cores_to_coordinator_ready);
    Semaphore<> cores_to_coordinator_done_sem(sem::cores_to_coordinator_done);

    const uint32_t number_of_confirmations = Wt / 2;

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        // Prepare and move data
#ifdef IS_ROW_MAJOR
        const uint32_t row_base = h * TILE_H;
#endif

        for (uint32_t w = 0; w < Wt; w++) {
            // Generate indexes
#ifdef IS_ROW_MAJOR
            {
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_coord_value_row.reserve_back(one_tile);
                    noc.async_read(
                        input_tensor_addr_ger,
                        rm_coord_value_row,
                        W_tile_bytes,
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_tile_bytes)},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    // The row was just deposited at the reserved (write) slot, which is where it has
                    // to be sourced from. A bare DataflowBuffer used as a NoC source resolves to the
                    // read cursor instead, so the write cursor is peeked explicitly here.
                    CoreLocalMem<uint32_t> value_row_src(rm_coord_value_row.get_write_ptr());
                    noc.async_write(
                        value_row_src,
                        output_tensor_addr_gen,
                        W_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_tile_bytes)});
                    noc.async_write_barrier();
                    rm_coord_value_row.push_back(one_tile);
                    rm_coord_value_row.pop_front(one_tile);
                }

                rm_coord_index_row.reserve_back(one_tile);
                const uint32_t l1_idx = rm_coord_index_row.get_write_ptr();
                const uint32_t idx_base = w * tile_width;
                if (is_32_bit_data) {
                    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_idx);
                    for (uint32_t c = 0; c < tile_width; c++) {
                        p[c] = idx_base + c;
                    }
                } else {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_idx);
                    for (uint32_t c = 0; c < tile_width; c++) {
                        p[c] = static_cast<uint16_t>(idx_base + c);
                    }
                }
                // The index buffer above is filled with baby-RISCV stores; the noc.async_write below reads
                // it as its source. A baby-RISCV store can retire before its write-request lands in SRAM,
                // and the RISCV core and NoC are different SRAM clients with no program-order guarantee
                // between them (WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md). load_blocking the
                // 32-bit word holding the last filled byte (blocking load + memory clobber) to drain the
                // fill so the NoC write cannot source a stale index buffer.
                {
                    const uint32_t idx_bytes = tile_width * (is_32_bit_data ? sizeof(uint32_t) : sizeof(uint16_t));
                    (void)ckernel::load_blocking(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_idx) + (idx_bytes - 1) / sizeof(uint32_t));
                }
                CoreLocalMem<uint32_t> index_row_src(l1_idx);
                // All TILE_H writes share the same (already drained) source entry and differ only in
                // destination, so they are all issued before a single barrier. A barrier per row
                // costs a full DRAM round-trip each, on one core, for every one of the Wt tiles.
                // The source entry is not popped until after the barrier, so it cannot be recycled
                // while a write is still in flight.
                for (uint32_t row = 0; row < TILE_H; row++) {
                    noc.async_write(
                        index_row_src,
                        output_index_tensor_addr_gen,
                        W_index_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_index_bytes)});
                }
                noc.async_write_barrier();
                rm_coord_index_row.push_back(one_tile);
                rm_coord_index_row.pop_front(one_tile);
            }
#else
            {
                if (is_32_bit_data) {
                    generate_index_tile<uint32_t>(dfb::index_tensor, w);
                } else {
                    generate_index_tile<uint16_t>(dfb::index_tensor, w);
                }

                index_tensor_dfb.wait_front(one_tile);
                noc.async_write(
                    index_tensor_dfb,
                    output_index_tensor_addr_gen,
                    index_tensor_tile_size,
                    {.offset_bytes = 0},
                    {.page_id = h * Wt + w, .offset_bytes = 0});
                noc.async_write_barrier();
                index_tensor_dfb.pop_front(one_tile);

                input_tensor_dfb.reserve_back(one_tile);
                noc.async_read(
                    input_tensor_addr_ger,
                    input_tensor_dfb,
                    raw_input_tile_bytes,
                    {.page_id = h * Wt + w, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                input_tensor_dfb.push_back(one_tile);

                input_tensor_dfb.wait_front(one_tile);
                noc.async_write(
                    input_tensor_dfb,
                    output_tensor_addr_gen,
                    raw_input_tile_bytes,
                    {.offset_bytes = 0},
                    {.page_id = h * Wt + w, .offset_bytes = 0});
                noc.async_write_barrier();
                input_tensor_dfb.pop_front(one_tile);
            }
#endif
        }  // Wt loop

        // Wait until all cores are ready to start
        cores_to_coordinator_ready_sem.wait(number_of_workers);
        cores_to_coordinator_ready_sem.set(0);  // Reset the semaphore

        // Set signal to start processing
        coordinator_to_cores_sem.set_multicast<NocOptions::DEFAULT>(
            noc,
            start_core_physical_coord_x,
            start_core_physical_coord_y,
            end_core_physical_coord_x,
            end_core_physical_coord_y,
            num_multicast_dests);
        noc.async_write_barrier();

        // Calculate sorting stages
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                // Set signal to start processing next sub-stage
                coordinator_to_cores_sem.set_multicast<NocOptions::DEFAULT>(
                    noc,
                    start_core_physical_coord_x,
                    start_core_physical_coord_y,
                    end_core_physical_coord_x,
                    end_core_physical_coord_y,
                    num_multicast_dests);
                noc.async_write_barrier();

                // Wait until cores will process and save data
                cores_to_coordinator_done_sem.wait(number_of_confirmations);
                cores_to_coordinator_done_sem.set(0);  // Reset the semaphore
            }  // sub loop
        }  // stage loop
    }  // Ht loop
}
