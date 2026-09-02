// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar (Metal-2) row-major padded_slice reader. Ported from the shared
// experimental/padded_slice reader to the Metal-2 bound/named model used by the quasar ops:
//   * output CB -> bound DataflowBuffer `dfb::in0` (the reader produces into the sharded output).
//   * source    -> bound TensorAccessor `tensor::src` (the legacy per-core base-address shift
//                  becomes an offset_bytes on each read, mirroring the quasar i2s stick reader).
//   * non-aligned staging -> node-local Scratchpad `scratch::pad` (not a DFB: the reader both
//                  fills and drains it, which would be an unsupported DM self-loop on Gen2).
//   * TRID pipelining uses the Quasar Noc API (noc.async_read<TXN_ID> + is_read_trid_flushed),
//     replacing the legacy noc_async_read_set_trid / ncrisc_..._flushed intrinsics.
// The trailing per-dim arrays (num_unpadded_sticks/num_padded_sticks/id_per_dim) are still read
// via get_arg_addr (kept available in Metal-2, used by the conv2d/matmul quasar readers).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"  // Scratchpad<> (non-aligned TRID staging)
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t src_byte_offset = get_arg(args::src_byte_offset);  // per-core src offset (begins+width-misalign)
    const uint32_t padded_stick_size = get_arg(args::padded_stick_size);
    const uint32_t unpadded_stick_size = get_arg(args::unpadded_stick_size);
    const uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    const uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    // Per-dim geometry passed as positional runtime varargs (Metal-2 has no array-valued named args):
    //   [ num_unpadded[0..num_dims) , num_padded[0..num_dims) , id_per_dim[0..num_dims) ].
    // id_per_dim is mutated locally as the read walks the padded input, so keep all three in local
    // arrays (num_dims <= MAX_RANK; get_vararg is read-only). MAX_RANK covers the max tensor rank.
    constexpr uint32_t MAX_RANK = 8;
    uint32_t num_unpadded_sticks[MAX_RANK];
    uint32_t num_padded_sticks[MAX_RANK];
    uint32_t id_per_dim[MAX_RANK];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_vararg(j);
        num_padded_sticks[j] = get_vararg(num_dims + j);
        id_per_dim[j] = get_vararg(2 * num_dims + j);
    }

    constexpr uint32_t is_non_aligned = get_arg(args::is_non_aligned);
    constexpr uint32_t src_buffer_alignment = get_arg(args::src_buffer_alignment);
    constexpr uint32_t num_trids = get_arg(args::num_trids);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in0);
    const auto s0 = TensorAccessor(tensor::src);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    uint32_t misalignment = 0;
    if constexpr (is_non_aligned) {
        misalignment = src_byte_offset % src_buffer_alignment;
    }
    const uint32_t src_off_aligned = src_byte_offset - misalignment;

    if constexpr (is_non_aligned) {
        // TRID-pipelined src->scratch->dest, mirroring the quasar i2s stick reader's unaligned path.
        enum SlotState : uint8_t { IDLE = 0, SRC_PENDING = 1, SCRATCH_READY = 2, SCRATCH_PENDING = 3 };
        constexpr uint32_t trid_base = 1;

        Scratchpad<uint32_t> scratch_buf(
            scratch::pad);  // NB: local must not be named `scratch` (shadows the scratch:: ns)
        const uint32_t scratch_page_size = scratch_buf.size_in_bytes() / num_trids;
        const uint32_t scratch_l1_base = scratch_buf.get_base_address();
        const uint32_t my_noc_x = my_x[noc.get_noc_id()];
        const uint32_t my_noc_y = my_y[noc.get_noc_id()];
        UnicastEndpoint self_ep;

        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            cb_in0.reserve_back(num_read_per_barrier);

            SlotState slot_states[num_trids];
            uint32_t dest_offsets[num_trids];
            for (uint32_t i = 0; i < num_trids; i++) {
                slot_states[i] = SlotState::IDLE;
            }
            uint32_t sticks_issued = 0;
            uint32_t sticks_completed = 0;
            uint32_t dest_off = 0;

            while (sticks_completed < num_read_per_barrier and sticks_read < num_sticks_per_core) {
                for (uint32_t slot = 0; slot < num_trids; slot++) {
                    const uint8_t active_trid = static_cast<uint8_t>(trid_base + slot);
                    const uint32_t scratch_off = slot * scratch_page_size;

                    if (slot_states[slot] == SlotState::IDLE && sticks_issued < num_read_per_barrier &&
                        sticks_read < num_sticks_per_core) {
                        CoreLocalMem<uint32_t> scratch_dst(scratch_l1_base + scratch_off);
                        noc.async_read<NocOptions::TXN_ID>(
                            s0,
                            scratch_dst,
                            padded_stick_size + misalignment,
                            {.page_id = src_stick_id, .offset_bytes = src_off_aligned},
                            {.offset_bytes = 0},
                            NocOptVals{.trid = active_trid});
                        dest_offsets[slot] = dest_off;
                        slot_states[slot] = SlotState::SRC_PENDING;

                        dest_off += stick_size_offset;
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
                        if (noc.is_read_trid_flushed(active_trid)) {
                            slot_states[slot] = SlotState::SCRATCH_READY;
                        }
                    } else if (slot_states[slot] == SlotState::SCRATCH_READY) {
                        // local L1 loopback: scratch(+misalignment) -> output DFB
                        noc.async_read<NocOptions::TXN_ID>(
                            self_ep,
                            cb_in0,
                            unpadded_stick_size,
                            {.noc_x = my_noc_x,
                             .noc_y = my_noc_y,
                             .addr = scratch_l1_base + scratch_off + misalignment},
                            {.offset_bytes = dest_offsets[slot]},
                            NocOptVals{.trid = active_trid});
                        slot_states[slot] = SlotState::SCRATCH_PENDING;
                    } else if (slot_states[slot] == SlotState::SCRATCH_PENDING) {
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
        // Reset the sticky TRID tag for any downstream untagged reads.
        noc.set_async_read_state<NocOptions::TXN_ID>(
            self_ep, /*size_bytes=*/0, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = 0}, NocOptVals{.trid = 0});
    } else {
        // Aligned path (the resnet stem's path): direct src->output DFB reads.
        for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
            cb_in0.reserve_back(num_read_per_barrier);
            uint32_t dest_off = 0;
            for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
                sticks_read++;
                noc.async_read(
                    s0,
                    cb_in0,
                    unpadded_stick_size,
                    {.page_id = src_stick_id, .offset_bytes = src_off_aligned},
                    {.offset_bytes = dest_off});
                dest_off += stick_size_offset;
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
