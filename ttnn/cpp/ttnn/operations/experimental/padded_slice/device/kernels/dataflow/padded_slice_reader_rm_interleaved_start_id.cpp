// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// The host defines NON_ALIGNED, and binds the dfb::non_aligned scratch, only when the read window isn't
// aligned to the source buffer; the aligned read path is compiled otherwise.
#ifdef NON_ALIGNED
constexpr bool is_non_aligned = true;
#else
constexpr bool is_non_aligned = false;
#endif

void kernel_main() {
    const uint32_t unpadded_stick_size = get_arg(args::unpadded_stick_size);
    const uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    const uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    // This core's aligned-down start within each input stick, and the bytes to skip after it.
    const uint32_t src_offset_bytes = get_arg(args::src_offset_bytes);
    const uint32_t misalignment = get_arg(args::misalignment);

    const uint32_t num_unpadded_sticks_base = 0;
    const uint32_t num_padded_sticks_base = num_dims;
    const uint32_t id_per_dim_seed_base = 2 * num_dims;

    constexpr uint32_t num_trids = get_arg(args::num_trids);

    uint32_t read_size = unpadded_stick_size;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    // The accessor locates whole input sticks from the buffer's base address. Each read below adds
    // src_offset_bytes to land on this core's part of the stick.
    const auto s0 = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
#ifdef NON_ALIGNED
    DataflowBuffer dfb_non_aligned(dfb::non_aligned);
#endif

    // Per-dimension index counter. The host passes its starting values as varargs, which are read-only,
    // so the kernel copies them into this scratchpad and counts there.
    Scratchpad<uint32_t> id_per_dim(scratch::id_per_dim);
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(id_per_dim_seed_base + j);
    }

#ifdef DEBUG
    DPRINT(
        "unpadded_stick_size: {}, stick_size_offset: {}, num_dims: {}, start_id: "
        "{}, num_sticks_per_core: {}, num_sticks_per_core_read: {}, num_read_per_barrier: {}\n",
        unpadded_stick_size,
        stick_size_offset,
        num_dims,
        start_id,
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);

    DPRINT(
        "non_aligned: {}, read_size: {}, src_stick_id: {}, sticks_read: {}, src_offset_bytes: {}, "
        "misalignment: {}\n",
        is_non_aligned,
        read_size,
        src_stick_id,
        sticks_read,
        src_offset_bytes,
        misalignment);

    DPRINT(
        "num_unpadded_sticks: {} {} {} {}\n",
        get_vararg(num_unpadded_sticks_base + 0),
        get_vararg(num_unpadded_sticks_base + 1),
        get_vararg(num_unpadded_sticks_base + 2),
        get_vararg(num_unpadded_sticks_base + 3));
    DPRINT(
        "num_padded_sticks: {} {} {} {}\n",
        get_vararg(num_padded_sticks_base + 0),
        get_vararg(num_padded_sticks_base + 1),
        get_vararg(num_padded_sticks_base + 2),
        get_vararg(num_padded_sticks_base + 3));
    DPRINT("Out DFB Page size: {}\n", dfb_out.get_entry_size());
#endif
#ifdef NON_ALIGNED
    {
        // TRID-based pipelined async-read from src->scratch->dst. Each slot tags its reads with its own
        // transaction id and polls that id for completion, so the slots progress independently instead
        // of stalling on a full read barrier.
        enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2, SCRATCH_PENDING = 3 };
        constexpr uint32_t trid_base = 1;
        UnicastEndpoint self_ep;

        dfb_non_aligned.reserve_back(num_trids);
        const uint32_t scratch_base = dfb_non_aligned.get_write_ptr();
        const uint32_t scratch_dfb_page_size = dfb_non_aligned.get_entry_size();

        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            dfb_out.reserve_back(num_read_per_barrier);

            SlotState slot_states[num_trids];
            // Byte offset of each slot's destination stick within the block reserved above.
            uint32_t dest_offsets[num_trids];

            // Initialize slots
            for (uint32_t i = 0; i < num_trids; i++) {
                slot_states[i] = SlotState::IDLE;
            }

            uint32_t sticks_issued = 0;
            uint32_t sticks_completed = 0;
            uint32_t current_dest_offset = 0;

            while (sticks_completed < num_read_per_barrier and sticks_read < num_sticks_per_core) {
                for (uint32_t slot = 0; slot < num_trids; slot++) {
                    const uint32_t active_trid = trid_base + slot;
                    const uint32_t scratch_offset = slot * scratch_dfb_page_size;

                    if (slot_states[slot] == SlotState::IDLE && sticks_issued < num_read_per_barrier &&
                        sticks_read < num_sticks_per_core) {
                        // Start new src->scratch transfer

                        dest_offsets[slot] = current_dest_offset;
                        noc.async_read<NocOptions::TXN_ID>(
                            s0,
                            dfb_non_aligned,
                            unpadded_stick_size + misalignment,
                            {.page_id = src_stick_id, .offset_bytes = src_offset_bytes},
                            {.offset_bytes = scratch_offset},
                            {.trid = active_trid});
                        slot_states[slot] = SlotState::SRC_PENDING;

                        current_dest_offset += stick_size_offset;
                        src_stick_id++;
                        for (uint32_t j = 0; j < num_dims; j++) {
                            id_per_dim[j]++;
                            if (id_per_dim[j] == get_vararg(num_unpadded_sticks_base + j)) {
                                id_per_dim[j] = 0;
                                src_stick_id += get_vararg(num_padded_sticks_base + j);
                            } else {
                                break;
                            }
                        }
                        sticks_issued++;

                    } else if (slot_states[slot] == SlotState::SRC_PENDING) {
                        // Check if src->scratch transfer is complete
                        if (noc.is_read_trid_flushed(active_trid)) {
                            slot_states[slot] = SlotState::SCRATCH_READY;
                        }

                    } else if (slot_states[slot] == SlotState::SCRATCH_READY) {
                        if (misalignment % 16 == 0) {
                            // Start scratch->dest transfer
                            noc.async_read<NocOptions::TXN_ID>(
                                self_ep,
                                dfb_out,
                                unpadded_stick_size,
                                experimental::local_addr(
                                    scratch_base + scratch_offset + misalignment, noc.get_noc_id()),
                                {.offset_bytes = dest_offsets[slot]},
                                {.trid = active_trid});
                            slot_states[slot] = SlotState::SCRATCH_PENDING;
                        } else {
                            // The NoC can't copy from a source that isn't 16 B aligned to this 16 B aligned
                            // destination, so copy on the RISC, as slice's RM reader does for misaligned sticks.
                            tt::data_movement::common::tt_memmove<false, false, false, 0>(
                                noc,
                                dfb_out.get_write_ptr() + dest_offsets[slot],
                                scratch_base + scratch_offset + misalignment,
                                unpadded_stick_size);
                            slot_states[slot] = SlotState::IDLE;
                            sticks_read++;
                            sticks_completed++;
                        }

                    } else if (slot_states[slot] == SlotState::SCRATCH_PENDING) {
                        // Check if scratch->dest transfer is complete
                        if (noc.is_read_trid_flushed(active_trid)) {
                            slot_states[slot] = SlotState::IDLE;
                            sticks_read++;
                            sticks_completed++;
                        }
                    }
                }
            }
            dfb_out.push_back(num_read_per_barrier);
        }
        // The non-aligned DFB is reserved once as a TRID scratchpad (no downstream consumer);
        // push the reserved pages so the buffer is left balanced.
        dfb_non_aligned.push_back(num_trids);
    }
#else
    {
        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            dfb_out.reserve_back(num_read_per_barrier);
            uint32_t l1_offset = 0;

            for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
                sticks_read++;
                noc.async_read(
                    s0,
                    dfb_out,
                    unpadded_stick_size,
                    {.page_id = src_stick_id, .offset_bytes = src_offset_bytes},
                    {.offset_bytes = l1_offset});
                l1_offset += stick_size_offset;
                src_stick_id++;
                for (uint32_t j = 0; j < num_dims; j++) {
                    id_per_dim[j]++;
                    if (id_per_dim[j] == get_vararg(num_unpadded_sticks_base + j)) {
                        id_per_dim[j] = 0;
                        src_stick_id += get_vararg(num_padded_sticks_base + j);
                    } else {
                        break;
                    }
                }
            }
            noc.async_read_barrier();
            dfb_out.push_back(num_read_per_barrier);
        }
    }
#endif
}
