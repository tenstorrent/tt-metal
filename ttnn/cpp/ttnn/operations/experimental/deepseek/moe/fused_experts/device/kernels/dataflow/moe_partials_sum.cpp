// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/tensor_accessor.h"

// persistent_moe_request's sum of one part of the row, on that part's sum core (see FusedPersistentParams).
// Every chip of the persistent op sends this part of its partial row into its slot of this core's partials
// buffer, then sets its flag word after the slots. Once every chip's flag is set, the compute sums the slots
// in chip order (cb_in is the buffer's slots), and this kernel writes the sum into its place in the output
// row, then clears the flags: the next request, which is the next time the op sends here, comes after this
// op has finished.
//
// Compile-time args:
//   0: num_slots (the op's chips)   1: part_bytes   2: cb_in (over the slots)   3: cb_out   4: tiles_per_part
//   5+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: skip (a stop request, which gets no partials)   1: partials buffer address   2: output address
//   3: this part's index in the row

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t part_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_part = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2));
    const uint32_t part = get_arg_val<uint32_t>(3);
    auto* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partials_addr + num_slots * part_bytes);

    {
        DeviceZoneScopedN("MOE_SUM_WAIT");
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            do {
                invalidate_l1_cache();
            } while (flags[slot] == 0u);
        }
    }
    cb_push_back(cb_in, num_slots * tiles_per_part);
    {
        DeviceZoneScopedN("MOE_SUM_WRITE");
        cb_wait_front(cb_out, tiles_per_part);
        noc_async_write(get_read_ptr(cb_out), output.get_noc_addr(0, part * part_bytes), part_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, tiles_per_part);
    }
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        flags[slot] = 0u;
    }
}
