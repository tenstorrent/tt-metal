// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(7);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(8);
    // This core's aligned-down start within each input stick, and the bytes to skip after it.
    const uint32_t src_offset_bytes = get_arg_val<uint32_t>(9);
    const uint32_t misalignment = get_arg_val<uint32_t>(10);
    const uint32_t num_unpadded_sticks_addr = get_arg_addr(11);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(num_unpadded_sticks_addr);
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr uint32_t is_non_aligned = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_non_aligned = get_compile_time_arg_val(1);
    constexpr uint32_t num_trids = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    uint32_t read_size = unpadded_stick_size;

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    // The accessor locates whole input sticks from the buffer's base address. Each read below adds
    // src_offset_bytes to land on this core's part of the stick.
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(src_args, src_addr, padded_stick_size);

    Noc noc;
    experimental::CB cb_in0(cb_id_in0);
    experimental::CB cb_non_aligned(cb_id_non_aligned);

#ifdef DEBUG
    DPRINT(
        "src_addr: {}, padded_stick_size: {}, unpadded_stick_size: {}, stick_size_offset: {}, num_dims: {}, start_id: "
        "{}, num_sticks_per_core: {}, num_sticks_per_core_read: {}, num_read_per_barrier: {}\n",
        src_addr,
        padded_stick_size,
        unpadded_stick_size,
        stick_size_offset,
        num_dims,
        start_id,
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);

    DPRINT(
        "non_aligned: {}, cb = {}, read_size: {}, src_stick_id: {}, sticks_read: {}, src_offset_bytes: {}, "
        "misalignment: {}\n",
        is_non_aligned,
        cb_id_non_aligned,
        read_size,
        src_stick_id,
        sticks_read,
        src_offset_bytes,
        misalignment);

    DPRINT(
        "num_unpadded_sticks: {} {} {} {}\n",
        num_unpadded_sticks[0],
        num_unpadded_sticks[1],
        num_unpadded_sticks[2],
        num_unpadded_sticks[3]);
    DPRINT(
        "num_padded_sticks: {} {} {} {}\n",
        num_padded_sticks[0],
        num_padded_sticks[1],
        num_padded_sticks[2],
        num_padded_sticks[3]);
    DPRINT("Out CB Page size: {}\n", get_local_cb_interface(cb_id_in0).fifo_page_size);
#endif
    if constexpr (is_non_aligned) {
        // TRID-based pipelined async-read from src->scratch->dst. Each slot tags its reads with its own
        // transaction id and polls that id for completion, so the slots progress independently instead
        // of stalling on a full read barrier.
        enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2, SCRATCH_PENDING = 3 };
        constexpr uint32_t trid_base = 1;
        UnicastEndpoint self_ep;

        cb_non_aligned.reserve_back(num_trids);
        const uint32_t scratch_base = cb_non_aligned.get_write_ptr();
        const uint32_t scratch_cb_page_size = get_local_cb_interface(cb_id_non_aligned).fifo_page_size;

        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            cb_in0.reserve_back(num_read_per_barrier);

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
                    const uint32_t scratch_offset = slot * scratch_cb_page_size;

                    if (slot_states[slot] == SlotState::IDLE && sticks_issued < num_read_per_barrier &&
                        sticks_read < num_sticks_per_core) {
                        // Start new src->scratch transfer

                        dest_offsets[slot] = current_dest_offset;
                        noc.async_read<NocOptions::TXN_ID>(
                            s0,
                            cb_non_aligned,
                            unpadded_stick_size + misalignment,
                            {.page_id = src_stick_id, .offset_bytes = src_offset_bytes},
                            {.offset_bytes = scratch_offset},
                            {.trid = active_trid});
                        slot_states[slot] = SlotState::SRC_PENDING;

                        current_dest_offset += stick_size_offset;
                        src_stick_id++;
                        for (uint32_t j = 0; j < num_dims; j++) {
                            id_per_dim[j]++;
                            if (id_per_dim[j] == num_unpadded_sticks[j]) {
                                id_per_dim[j] = 0;
                                src_stick_id += num_padded_sticks[j];
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
                                cb_in0,
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
                                cb_in0.get_write_ptr() + dest_offsets[slot],
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
            cb_in0.push_back(num_read_per_barrier);
        }
        // The non-aligned CB is reserved once as a TRID scratchpad (no downstream consumer);
        // push the reserved pages so the buffer is left balanced.
        cb_non_aligned.push_back(num_trids);
    } else {
        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            cb_in0.reserve_back(num_read_per_barrier);
            uint32_t l1_offset = 0;

            for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
                sticks_read++;
                noc.async_read(
                    s0,
                    cb_in0,
                    unpadded_stick_size,
                    {.page_id = src_stick_id, .offset_bytes = src_offset_bytes},
                    {.offset_bytes = l1_offset});
                l1_offset += stick_size_offset;
                src_stick_id++;
                for (uint32_t j = 0; j < num_dims; j++) {
                    id_per_dim[j]++;
                    if (id_per_dim[j] == num_unpadded_sticks[j]) {
                        id_per_dim[j] = 0;
                        src_stick_id += num_padded_sticks[j];
                    } else {
                        break;
                    }
                }
            }
            noc.async_read_barrier();
            cb_in0.push_back(num_read_per_barrier);
        }
    }
}
