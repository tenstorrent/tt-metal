// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
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
    DataflowBuffer cb_in1(dfb::in1);

    // The source-buffer base address is bound via the tensor parameter (tensor::src). The
    // legacy reader pre-shifted the base by `aligned_input_width_offset_bytes`; in the typed
    // model that per-core byte shift becomes the source-side `offset_bytes` on each read.
    const auto s0 = TensorAccessor(tensor::src);
    uint32_t stick_id = start_id;
    cb_in0.reserve_back(block_height);
    if (aligned) {
        uint32_t dest_off = 0;
        for (uint32_t h = 0; h < block_height; ++h) {
            noc.async_read(
                s0,
                cb_in0,
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

        cb_in1.reserve_back(num_trids);
        uint32_t scratch_cb_page_size = get_local_cb_interface(dfb::in1).fifo_page_size;
        SlotState slot_states[num_trids];
        uint32_t dest_offsets[num_trids];
        uint32_t scratch_offsets[num_trids];

        // Initialize slots
        for (uint32_t i = 0; i < num_trids; i++) {
            slot_states[i] = SlotState::IDLE;
            scratch_offsets[i] = i * scratch_cb_page_size;
        }

        // Local NoC coordinates for the scratch->dest reads.
        UnicastEndpoint self_ep;
        const uint32_t my_noc_x = my_x[noc.get_noc_id()];
        const uint32_t my_noc_y = my_y[noc.get_noc_id()];
        // Base L1 address of the scratch CB
        const uint32_t scratch_l1_base = cb_in1.get_write_ptr();

        uint32_t dest_off = 0;        // running offset into cb_in0
        uint32_t rows_issued = 0;     // Number of src->scratch transfers started
        uint32_t rows_completed = 0;  // Number of scratch->dest transfers completed

        while (rows_completed < block_height) {
            for (uint32_t slot = 0; slot < num_trids; slot++) {
                uint32_t active_trid = trid_base + slot;

                if (slot_states[slot] == SlotState::IDLE && rows_issued < block_height) {
                    // Start new src->scratch transfer (TRID-tagged).
                    noc.async_read<NocOptions::TXN_ID>(
                        s0,
                        cb_in1,
                        aligned_block_width_bytes,
                        {.page_id = stick_id, .offset_bytes = aligned_input_width_offset_bytes},
                        {.offset_bytes = scratch_offsets[slot]},
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
    }
    // Reset the sticky NOC_PACKET_TAG register for downstream untagged reads
    UnicastEndpoint self_ep;
    noc.set_async_read_state<NocOptions::TXN_ID>(
        self_ep,
        /*size_bytes=*/0,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = 0},
        NocOptVals{.trid = 0});
    cb_in0.push_back(block_height);
}
