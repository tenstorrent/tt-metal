// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 row-major interleaved slice writer. Logic identical to
// slice_writer_unary_stick_layout_interleaved_start_id.cpp (that one stays for the
// legacy/descriptor consumers); only the bindings are Metal 2.0:
//   - CB index            -> dfb::cb_out  (Device 2.0 CircularBuffer preserved)
//   - output accessor     -> ta::dst      (standard 2-arg TensorAccessor; no runtime page-size
//                            override is needed — the destination page base/page size are
//                            spec-derived and in the cache key)
//   - stick_size          -> named CTA (bytes written per output row; the DFB entry_size is the
//                            (possibly larger) aligned CB stride cb_page_size, so the write size
//                            cannot be taken from cb.get_tile_size())
//   - stick_size_offset   -> named CTA (op-level CB stride, in the cache key)
//   - per-core scalars    -> named RTAs (num_sticks_per_core / num_sticks_per_core_read /
//                            num_read_per_barrier / start_id)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = dfb::cb_out;
    // Bytes written per output row (the destination accessor's page size).
    constexpr uint32_t stick_size = get_named_compile_time_arg_val("stick_size");
    // CB read stride between consecutive sticks (op-level; in the cache key).
    constexpr uint32_t stick_size_offset = get_named_compile_time_arg_val("stick_size_offset");

    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    const uint32_t start_id = get_arg(args::start_id);

    const auto s0 = TensorAccessor(ta::dst);

    Noc noc;
    // Create CircularBuffer for Device 2.0 API
    CircularBuffer cb_out0(cb_id_out0);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_out0.wait_front(num_read_per_barrier);
        uint32_t cb_read_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_write(
                cb_out0, s0, stick_size, {.offset_bytes = cb_read_offset}, {.page_id = i_stick, .offset_bytes = 0});
            cb_read_offset += stick_size_offset;
            i_stick += 1;
        }
        noc.async_write_barrier();
        cb_out0.pop_front(num_read_per_barrier);
    }
}
