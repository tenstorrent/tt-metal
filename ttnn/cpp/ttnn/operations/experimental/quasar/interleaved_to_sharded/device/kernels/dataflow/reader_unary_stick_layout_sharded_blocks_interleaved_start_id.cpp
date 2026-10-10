// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t block_width_bytes = get_arg(args::block_width_bytes);
    const uint32_t padded_block_width_bytes = get_arg(args::padded_block_width_bytes);
    const bool aligned = static_cast<bool>(get_arg(args::aligned));
    const uint32_t aligned_input_width_offset_bytes = get_arg(args::aligned_input_width_offset_bytes);
    const uint32_t aligned_block_width_bytes = get_arg(args::aligned_block_width_bytes);
    const uint32_t aligned_offset = get_arg(args::aligned_offset);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t num_trids = get_arg(args::num_trids);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in0);
    // Staging area for the unaligned path is a private node-local L1 scratchpad (scratch::pad),
    // constructed inside the unaligned branch below. It is NOT a DFB: the reader both fills it
    // (src->scratch) and drains it (scratch->dest local copy), which would be an unsupported DM
    // producer+consumer self-loop DFB on Gen2/Quasar.

    // The source-buffer base address is bound via the tensor parameter (tensor::src). The
    // legacy reader pre-shifted the base by `aligned_input_width_offset_bytes`; in the typed
    // model that per-core byte shift becomes the source-side `offset_bytes` on each read.
    const auto s0 = TensorAccessor(tensor::src);

    // Thread t of N fills shard rows t, t + N, ...: the strided DFB gives each thread every N-th
    // entry, N entries apart in L1.
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
#ifdef IMPLICIT_SYNC
    // Host enables this with a staging DFB of one unpadded row per entry: each TXN_ID read fills the
    // next ring entry and posts its credit when it lands. Quasar NoC reads take any source byte
    // offset, so the row is read from its exact start.
    const uint32_t row_offset_bytes = aligned_input_width_offset_bytes + aligned_offset;
    for (uint32_t h = thread_id; h < block_height; h += num_threads) {
        noc.async_read<NocOptions::TXN_ID>(s0, cb_in0, {.page_id = start_id + h, .offset_bytes = row_offset_bytes}, {});
    }
#else
    const uint32_t num_my_rows = block_height > thread_id ? (block_height - thread_id - 1) / num_threads + 1 : 0;
    const uint32_t dest_stride_bytes = num_threads * padded_block_width_bytes;
    uint32_t stick_id = start_id + thread_id;
    cb_in0.reserve_back(num_my_rows);
    if (aligned) {
        uint32_t dest_off = 0;
        for (uint32_t h = 0; h < num_my_rows; ++h) {
            noc.async_read(
                s0,
                cb_in0,
                block_width_bytes,
                {.page_id = stick_id, .offset_bytes = aligned_input_width_offset_bytes},
                {.offset_bytes = dest_off});
            stick_id += num_threads;
            dest_off += dest_stride_bytes;
        }
        noc.async_read_barrier();
    } else {
        enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2, SCRATCH_PENDING = 3 };

        constexpr uint32_t trid_base = 1;

        // Private node-local L1 scratchpad (raw memory, no producer/consumer credit semantics).
        // Total size == num_threads * num_trids * scratch_cb_page_size (set by the host ScratchpadSpec):
        // the threads share it, each owning num_trids slots.
        Scratchpad<uint32_t> scratch(scratch::pad);
        uint32_t scratch_cb_page_size = scratch.size_in_bytes() / (num_threads * num_trids);
        SlotState slot_states[num_trids];
        uint32_t dest_offsets[num_trids];
        uint32_t scratch_offsets[num_trids];

        // Initialize slots
        for (uint32_t i = 0; i < num_trids; i++) {
            slot_states[i] = SlotState::IDLE;
            scratch_offsets[i] = (thread_id * num_trids + i) * scratch_cb_page_size;
        }

        // Local NoC coordinates for the scratch->dest reads.
        UnicastEndpoint self_ep;
        const uint32_t my_noc_x = my_x[noc.get_noc_id()];
        const uint32_t my_noc_y = my_y[noc.get_noc_id()];
        // Base L1 address of the scratch region.
        const uint32_t scratch_l1_base = scratch.get_base_address();

        uint32_t dest_off = 0;        // running offset into cb_in0
        uint32_t rows_issued = 0;     // Number of src->scratch transfers started
        uint32_t rows_completed = 0;  // Number of scratch->dest transfers completed

        while (rows_completed < num_my_rows) {
            for (uint32_t slot = 0; slot < num_trids; slot++) {
                uint32_t active_trid = trid_base + slot;

                if (slot_states[slot] == SlotState::IDLE && rows_issued < num_my_rows) {
                    // Start new src->scratch transfer (TRID-tagged). Destination is the raw L1
                    // scratchpad slot, addressed directly (no DFB offset semantics).
                    CoreLocalMem<uint32_t> scratch_dst(scratch_l1_base + scratch_offsets[slot]);
                    noc.async_read<NocOptions::TXN_ID>(
                        s0,
                        scratch_dst,
                        aligned_block_width_bytes,
                        {.page_id = stick_id, .offset_bytes = aligned_input_width_offset_bytes},
                        {.offset_bytes = 0},
                        NocOptVals{.trid = static_cast<uint8_t>(active_trid)});
                    dest_offsets[slot] = dest_off;
                    slot_states[slot] = SlotState::SRC_PENDING;

                    stick_id += num_threads;
                    dest_off += dest_stride_bytes;
                    rows_issued++;
                }
                if (slot_states[slot] == SlotState::SRC_PENDING) {
                    // Check if src->scratch is complete
                    if (noc.is_read_trid_flushed(active_trid)) {
                        slot_states[slot] = SlotState::SCRATCH_READY;
                    }
                }
                if (slot_states[slot] == SlotState::SCRATCH_READY) {
                    // Start scratch->dest transfer: local L1 loopback read tagged with the same trid.
                    noc.async_read<NocOptions::TXN_ID>(
                        self_ep,
                        cb_in0,
                        block_width_bytes,
                        {.noc_x = my_noc_x,
                         .noc_y = my_noc_y,
                         .addr = scratch_l1_base + scratch_offsets[slot] + aligned_offset},
                        {.offset_bytes = dest_offsets[slot]},
                        NocOptVals{.trid = static_cast<uint8_t>(active_trid)});

                    slot_states[slot] = SlotState::SCRATCH_PENDING;
                }
                if (slot_states[slot] == SlotState::SCRATCH_PENDING) {
                    // Check if scratch->dest is complete
                    if (noc.is_read_trid_flushed(active_trid)) {
                        slot_states[slot] = SlotState::IDLE;
                        rows_completed++;
                    }
                }
            }
        }
        // No DFB bookkeeping: the scratchpad is raw L1 with no producer/consumer credits.
    }
    // Reset the sticky NOC_PACKET_TAG register for downstream untagged reads
    UnicastEndpoint self_ep;
    noc.set_async_read_state<NocOptions::TXN_ID>(
        self_ep,
        /*size_bytes=*/0,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = 0},
        NocOptVals{.trid = 0});
    cb_in0.push_back(num_my_rows);
#endif
}
