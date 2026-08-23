// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

namespace padded_slice {

constexpr uint32_t MAX_RANK = 8;

struct RMGeometry {
    uint32_t num_unpadded_sticks[MAX_RANK];
    uint32_t num_padded_sticks[MAX_RANK];
    uint32_t id_per_dim[MAX_RANK];
};

FORCE_INLINE RMGeometry load_rm_geometry(uint32_t num_dims) {
    RMGeometry geometry;
    for (uint32_t j = 0; j < num_dims; ++j) {
        geometry.num_unpadded_sticks[j] = get_vararg(j);
        geometry.num_padded_sticks[j] = get_vararg(num_dims + j);
        geometry.id_per_dim[j] = get_vararg(2 * num_dims + j);
    }
    return geometry;
}

FORCE_INLINE void advance_source(uint32_t num_dims, RMGeometry& geometry, uint32_t& src_stick_id) {
    src_stick_id++;
    for (uint32_t j = 0; j < num_dims; j++) {
        geometry.id_per_dim[j]++;
        if (geometry.id_per_dim[j] == geometry.num_unpadded_sticks[j]) {
            geometry.id_per_dim[j] = 0;
            src_stick_id += geometry.num_padded_sticks[j];
        } else {
            break;
        }
    }
}

FORCE_INLINE void read_rm_aligned(
    uint32_t src_byte_offset,
    uint32_t padded_stick_size,
    uint32_t unpadded_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier) {
    RMGeometry geometry = load_rm_geometry(num_dims);
    const uint32_t input_base =
        get_common_arg_val<uint32_t>(decltype(tensor::input)::addr_crta_offset / sizeof(uint32_t));
    // Override the page size at runtime just as the legacy accessor did. This keeps cached programs
    // correct when the bound allocation changes while the program specialization remains reusable.
    const auto input = TensorAccessor(decltype(tensor::input)::args, input_base, padded_stick_size);
    DataflowBuffer output(dfb::output);
    Noc noc;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        const uint32_t remaining_sticks = num_sticks_per_core - sticks_read;
        const uint32_t batch_size = num_read_per_barrier < remaining_sticks ? num_read_per_barrier : remaining_sticks;
        output.reserve_back(batch_size);
        uint32_t l1_offset = 0;
        for (uint32_t i = 0; i < batch_size; ++i) {
            sticks_read++;
            noc.async_read(
                input,
                output,
                unpadded_stick_size,
                {.page_id = src_stick_id, .offset_bytes = src_byte_offset},
                {.offset_bytes = l1_offset});
            l1_offset += stick_size_offset;
            advance_source(num_dims, geometry, src_stick_id);
        }
        noc.async_read_barrier();
        output.push_back(batch_size);
    }
}

template <uint32_t SrcBufferAlignment, uint32_t NumTrids>
FORCE_INLINE void read_rm_non_aligned(
    uint32_t src_byte_offset,
    uint32_t padded_stick_size,
    uint32_t unpadded_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier,
    const ScratchpadBindingToken& alignment_scratch_token) {
    RMGeometry geometry = load_rm_geometry(num_dims);
    const uint32_t input_base =
        get_common_arg_val<uint32_t>(decltype(tensor::input)::addr_crta_offset / sizeof(uint32_t));
    DataflowBuffer output(dfb::output);
    Scratchpad<uint32_t> alignment_scratch(alignment_scratch_token);
    Noc noc;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    const uint32_t misalignment = src_byte_offset % SrcBufferAlignment;
    const uint32_t src_offset_aligned = src_byte_offset - misalignment;
    // Override the page size at runtime just as the legacy accessor did, while aligning the
    // per-core base down so the scratch copy can remove the intra-page offset.
    const auto input = TensorAccessor(decltype(tensor::input)::args, input_base, padded_stick_size);

    // TRID-based pipelined async-read from src->scratch->dst.
    enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2 };
    constexpr uint32_t trid_base = 1;
    const uint32_t scratch_page_size = alignment_scratch.size_in_bytes() / NumTrids;
    const uint32_t scratch_l1_base = alignment_scratch.get_base_address();
    UnicastEndpoint self_ep;

    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        const uint32_t remaining_sticks = num_sticks_per_core - sticks_read;
        const uint32_t batch_size = num_read_per_barrier < remaining_sticks ? num_read_per_barrier : remaining_sticks;
        output.reserve_back(batch_size);
        const uint32_t output_l1_base = output.get_write_ptr();
        SlotState slot_states[NumTrids];
        uint32_t dest_offsets[NumTrids];
        // Initialize every pipeline slot before issuing this batch.
        for (uint32_t i = 0; i < NumTrids; i++) {
            slot_states[i] = SlotState::IDLE;
        }
        uint32_t sticks_issued = 0;
        uint32_t sticks_completed = 0;
        uint32_t dest_offset = 0;

        while (sticks_completed < batch_size) {
            for (uint32_t slot = 0; slot < NumTrids; slot++) {
                const uint8_t active_trid = static_cast<uint8_t>(trid_base + slot);
                const uint32_t scratch_offset = slot * scratch_page_size;
                if (slot_states[slot] == SlotState::IDLE && sticks_issued < batch_size) {
                    // Stage one aligned source page into this slot.
                    CoreLocalMem<uint32_t> scratch_destination(scratch_l1_base + scratch_offset);
                    noc.async_read<NocOptions::TXN_ID>(
                        input,
                        scratch_destination,
                        unpadded_stick_size + misalignment,
                        {.page_id = src_stick_id, .offset_bytes = src_offset_aligned},
                        {.offset_bytes = 0},
                        NocOptVals{.trid = active_trid});
                    dest_offsets[slot] = dest_offset;
                    slot_states[slot] = SlotState::SRC_PENDING;
                    dest_offset += stick_size_offset;
                    advance_source(num_dims, geometry, src_stick_id);
                    sticks_issued++;
                } else if (slot_states[slot] == SlotState::SRC_PENDING) {
                    // Wait until the source-to-scratch transaction completes.
                    if (noc.is_read_trid_flushed(active_trid)) {
                        slot_states[slot] = SlotState::SCRATCH_READY;
                    }
                } else if (slot_states[slot] == SlotState::SCRATCH_READY) {
                    // Wormhole NOC source addresses must be aligned. Copy the requested unaligned
                    // subrange locally after the aligned source read completes; this replaces the
                    // legacy scratch-to-output NOC copy rather than adding another data movement.
                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        noc,
                        output_l1_base + dest_offsets[slot],
                        scratch_l1_base + scratch_offset + misalignment,
                        unpadded_stick_size);
                    slot_states[slot] = SlotState::IDLE;
                    sticks_read++;
                    sticks_completed++;
                }
            }
        }
        output.push_back(batch_size);
    }
    // Reset the sticky TRID tag for downstream untagged reads.
    noc.set_async_read_state<NocOptions::TXN_ID>(
        self_ep, 0, {.noc_x = 0, .noc_y = 0, .addr = 0}, NocOptVals{.trid = 0});
}

}  // namespace padded_slice
