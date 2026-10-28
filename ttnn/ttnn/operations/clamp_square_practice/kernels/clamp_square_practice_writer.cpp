// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice writer (BRISC / NoC1). Same block/CB contract as the op's writer (wait
// block_tiles, `valid` writes, flush, pop; one final barrier). Walks the IDENTICAL segment list as
// the reader, so every page is written back to the index it was read from.
// What changes vs the op is WHICH pages this core owns. The host partitions the DRAM banks into
// groups (a group = an ordered bank list, e.g. the banks NoC-nearest a region of the grid) and hands
// the core a short list of segments over those groups:
//   seg = (g, row0, k0, cnt, bank[0..GMAX-1]); q = row0*g + k0 + i, i in [0, cnt):
//         page = (q / g) * NB + bank[q % g]
// i.e. this core walks the group's pages in natural order, round-robin across the group's banks
// (g = 1 -> one bank, consecutive bank addresses; g = NB with bank[k] = k -> the op's contiguous range).
// The host pre-splits q0 into (row0, k0) (no device divide); the walk is incremental.
// RT args (fixed length, program-cache safe): [addr, core_tiles, nseg, NSLOT slots of (4 + GMAX)].
// No helper bypassed: the op itself uses raw noc_async_{read,write}_page.
// CT args: [cb, block_tiles, page_bytes, num_banks, GMAX, TensorAccessorArgs...]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t gmax = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    // Permanent stage zones (free when the profiler is off). Kernel-level, never per block.
    // writer_write is occupancy: includes cb_wait_front stalls on compute.
    {
        MaybeDeviceZoneScope("writer_write");
        uint32_t seg_arg = 3, banks = 0, g = 1, k = 0, row_base = 0, left = 0;
        for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
            const uint32_t remaining = core_tiles - done;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t i = 0; i < valid; ++i) {
                while (left == 0) {
                    g = get_arg_val<uint32_t>(seg_arg);
                    row_base = get_arg_val<uint32_t>(seg_arg + 1) * num_banks;
                    k = get_arg_val<uint32_t>(seg_arg + 2);
                    left = get_arg_val<uint32_t>(seg_arg + 3);
                    banks = seg_arg + 4;
                    seg_arg += 4 + gmax;
                }
                const uint32_t page = row_base + get_arg_val<uint32_t>(banks + k);
                if (++k == g) {
                    k = 0;
                    row_base += num_banks;
                }
                --left;
                noc_async_write_page(page, output_accessor, l1_read_addr);
                l1_read_addr += page_bytes;
            }
            noc_async_writes_flushed();  // L1 source reads done -> slot can be recycled
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
