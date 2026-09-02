// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the height-only sharded pad reader (private to PadRmShardedHeightOnlyProgramFactory).
// Device-side NoC logic is unchanged; resource access uses Metal 2.0 named handles. Self-loop DFBs are
// no longer permitted on data-movement kernels, so:
//   - The input shard is read via tensor::input (a TensorAccessor over the resident shard): its local L1
//     base address is recovered with NOC_LOCAL_ADDR_OFFSET(s.get_noc_addr(0)) and used as the source offset
//     for the remote NoC reads against neighbouring shards. (No more cb_in0 borrowed fake-CB self-loop.)
//   - cb_pad is now a CROSS-KERNEL DFB: this reader PRODUCES the pad-value stick once (fill moved here
//     from the writer); the writer CONSUMES it. (No more writer self-loop.)
//   - c_16 output shard is written in place via tensor::output (NOC_LOCAL_ADDR_OFFSET(s_out.get_noc_addr(0)));
//     the reader writes the gathered real sticks, the writer writes the pad sticks (no borrowed DFB).
//   - The legacy variable-length arg tail (read_noc_x/y, num_stick_chunks, chunk lists) is runtime varargs.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Pad-value fill helpers (moved here from the writer; identical logic). The reader produces cb_pad so a
// single DM kernel no longer both produces and consumes it.
inline __attribute__((always_inline)) void fill_pad_dfb_with_val(
    Noc& noc, DataflowBuffer& cb, const uint32_t num_bytes_risc, uint32_t num_noc_transfer, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());

    for (uint32_t i = 0; i < num_bytes_risc / 2; ++i) {
        ptr[i] = val;
    }

    uint32_t pad_val_addr = cb.get_write_ptr();
    uint32_t l1_write_addr = pad_val_addr;

    for (uint32_t i = 0; i < num_noc_transfer; ++i) {
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            num_bytes_risc,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = pad_val_addr},
            {.offset_bytes = 0});
        l1_write_addr += num_bytes_risc;
    }
    noc.async_read_barrier();
}

inline __attribute__((always_inline)) void fill_pad_dfb_with_zero(
    Noc& noc, DataflowBuffer& cb, const uint32_t num_bytes_risc, uint32_t num_noc_transfer) {
    noc.async_write_zeros(cb, num_bytes_risc * num_noc_transfer);
    noc.write_zeros_l1_barrier();
}

void kernel_main() {
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    [[maybe_unused]] constexpr uint32_t num_sticks_padded = get_arg(args::num_sticks_padded);

    // Pad-value fill CTAs (moved here from the writer).
    constexpr uint32_t num_zero_pad_sticks_read = get_arg(args::num_zero_pad_sticks_read);
    constexpr uint32_t zero_pad_stick_size = get_arg(args::zero_pad_stick_size);
    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    uint32_t packed_pad_value = 0;
    uint32_t row_major_min_bytes = 0;
    uint32_t num_sticks_padded_read = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
        row_major_min_bytes = get_arg(args::row_major_min_bytes);
        num_sticks_padded_read = get_arg(args::num_sticks_padded_read);
    }

    const uint32_t num_cores_read = get_arg(args::num_cores_read);

    // Vararg tail layout (per core; padded to a uniform max):
    //   [ read_noc_x/y interleaved : 2*num_cores_read ]
    //   [ num_stick_chunks         :   num_cores_read ]
    //   [ (chunk_start_id, chunk_num_sticks) pairs : 2 * sum(num_stick_chunks) ]
    const uint32_t num_stick_chunks_base = num_cores_read * 2;
    const uint32_t chunk_base = num_cores_read * 3;

    DataflowBuffer cb_pad(dfb::cb_pad);
    Noc noc;

    // Produce the pad-value stick for the writer (cross-kernel DFB).
    cb_pad.reserve_back(1);
    if constexpr (not_pad_by_zero) {
        fill_pad_dfb_with_val(noc, cb_pad, row_major_min_bytes, num_sticks_padded_read, packed_pad_value);
    } else {
        fill_pad_dfb_with_zero(noc, cb_pad, zero_pad_stick_size, num_zero_pad_sticks_read);
    }
    cb_pad.push_back(1);

    // Local input-shard base L1 address (the source offset replicated across all sharded cores). Recovered
    // from the resident input TensorAccessor instead of a borrowed fake-CB self-loop.
    const auto s = TensorAccessor(tensor::input);
    uint32_t l1_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s.get_noc_addr(0));

    // Output shard base from the resident output TensorAccessor (written in place; no borrowed
    // co-write DFB — both reader and writer address the output shard directly).
    const auto s_out = TensorAccessor(tensor::output);
    uint32_t l1_write_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s_out.get_noc_addr(0));

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(read_noc_xy_ptr_offset + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_stick_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset + 1);

            uint32_t l1_read_offset = curr_start_id * stick_size_bytes;
            uint32_t read_data_size_bytes = curr_num_sticks * stick_size_bytes;

            if ((curr_start_id != (uint32_t)-1) and (curr_start_id != (uint32_t)-2)) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    read_data_size_bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + l1_read_offset},
                    {.offset_bytes = 0});
            }

            l1_write_addr += read_data_size_bytes;
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
}
