// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp. Gathers a
// row-major shard stick-by-stick out of an interleaved input tensor, staging through a local scratch
// buffer when the source columns are not alignment-friendly. Only the plumbing changes: the two
// buffer-index compile-time args become dfb::in and scratch::pad, the accessor-args / base-address pair
// becomes the tensor::src binding, the positional runtime args become named ones, and the scratch page
// size is derived from the Scratchpad object instead of the raw CB interface. The TRID-tagged transfer
// pipeline and its slot state machine are untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.
//
// The binding names below (dfb::in, scratch::pad, tensor::src) and the named argument set are this
// fork's interface: every later consumer inherits them, so they are taken from the kernel's own
// vocabulary rather than any one op's locals, and are not renamed once a consumer exists.
//
// [Quasar bring-up: i2s scratchpad] the alignment staging area is scratch::pad, a private
// node-local L1 scratchpad. The reader both fills and drains it, which as a DFB is a
// data-movement PRODUCER+CONSUMER self-loop and is rejected on Gen2/Quasar by
// program_spec.cpp:1502. A factory binding this source declares ONE ScratchpadBinding for it
// (accessor name "pad"), not a pair of DFB endpoints.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
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

    constexpr auto num_trids = get_arg(args::num_trids);

    Noc noc;
    // dfb::in — this core's row-major shard.
    // scratch::pad — the alignment staging area, constructed in the unaligned branch below.
    // It is raw node-local L1, NOT a DFB; see the header comment.
    DataflowBuffer dfb_in(dfb::in);

    const auto s0 = TensorAccessor(tensor::src);
    uint32_t stick_id = start_id;
    dfb_in.reserve_back(block_height);
    if (aligned) {
        uint32_t dest_off = 0;
        for (uint32_t h = 0; h < block_height; ++h) {
            noc.async_read(
                s0,
                dfb_in,
                block_width_bytes,
                {.page_id = stick_id, .offset_bytes = aligned_input_width_offset_bytes},
                {.offset_bytes = dest_off});
            stick_id++;
            dest_off += padded_block_width_bytes;
        }
        noc.async_read_barrier();
    } else {
        enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2, SCRATCH_PENDING = 3 };

        constexpr uint32_t trid_base = 1;

        // Private node-local L1 scratchpad (raw memory, no producer/consumer credit semantics).
        // Total size == num_trids * page size (set by the host ScratchpadSpec), so the per-slot
        // page size is recovered by dividing the total by num_trids.
        Scratchpad<uint32_t> scratch(scratch::pad);
        uint32_t scratch_page_size = scratch.size_in_bytes() / num_trids;
        SlotState slot_states[num_trids];
        uint32_t dest_offsets[num_trids];
        uint32_t scratch_offsets[num_trids];

        // Initialize slots
        for (uint32_t i = 0; i < num_trids; i++) {
            slot_states[i] = SlotState::IDLE;
            scratch_offsets[i] = i * scratch_page_size;
        }

        // Local NoC coordinates for the scratch->dest reads.
        UnicastEndpoint self_ep;
        const uint32_t my_noc_x = my_x[noc.get_noc_id()];
        const uint32_t my_noc_y = my_y[noc.get_noc_id()];
        // Base L1 address of the scratch region.
        const uint32_t scratch_l1_base = scratch.get_base_address();

        uint32_t dest_off = 0;        // running offset into dfb_in
        uint32_t rows_issued = 0;     // Number of src->scratch transfers started
        uint32_t rows_completed = 0;  // Number of scratch->dest transfers completed

        while (rows_completed < block_height) {
            for (uint32_t slot = 0; slot < num_trids; slot++) {
                uint32_t active_trid = trid_base + slot;

                if (slot_states[slot] == SlotState::IDLE && rows_issued < block_height) {
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

                    stick_id++;
                    dest_off += padded_block_width_bytes;
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
                        dfb_in,
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
    dfb_in.push_back(block_height);
}
