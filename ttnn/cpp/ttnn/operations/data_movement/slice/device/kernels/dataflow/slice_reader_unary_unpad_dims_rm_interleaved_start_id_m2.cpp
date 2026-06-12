// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 row-major interleaved slice reader. Logic identical to
// slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp (that one stays for the
// legacy/descriptor consumers); only the bindings are Metal 2.0:
//   - CB index                 -> dfb::cb_in   (Device 2.0 CircularBuffer preserved)
//   - num_dims                 -> named CTA (get_named_compile_time_arg_val; sizes the loop)
//   - input accessor           -> ta::src      (standard 2-arg TensorAccessor; page size is
//                                  spec-derived (== padded_stick_size) and in the cache key.
//                                  The within-row column offset is applied via the source
//                                  page-spec offset_bytes, NOT a runtime base-address/page-size
//                                  override — see column_offset_bytes below.)
//   - op-level byte sizes      -> named CTAs (padded_stick_size / unpadded_stick_size /
//                                  stick_size_offset / misalignment / column_offset_bytes;
//                                  pure functions of shape + slice params, all in the cache key)
//   - per-core scalars         -> named RTAs (start_id / num_sticks_per_core /
//                                  num_sticks_per_core_read / num_read_per_barrier)
//   - per-dim shape arrays      -> common varargs (read-only): [num_unpadded[], num_padded[]]
//   - per-dim running index    -> per-core varargs (id_per_dim), copied into a local mutable array

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t num_dims = get_named_compile_time_arg_val("num_dims");

    // Op-level byte sizes (identical across cores; in the cache key).
    constexpr uint32_t unpadded_stick_size = get_named_compile_time_arg_val("unpadded_stick_size");
    constexpr uint32_t stick_size_offset = get_named_compile_time_arg_val("stick_size_offset");
    constexpr uint32_t misalignment = get_named_compile_time_arg_val("misalignment");
    // Byte offset of the slice's first column within each input row, rounded down to the source
    // buffer alignment (== begins_bytes - misalignment in the legacy host). The legacy reader folded
    // this into the accessor's base address; here it is the source page offset_bytes (which keeps the
    // page base/page size spec-derived for the standard 2-arg accessor).
    constexpr uint32_t column_offset_bytes = get_named_compile_time_arg_val("column_offset_bytes");

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);

    // Read-only per-dim shape (common varargs): [num_unpadded[num_dims], num_padded[num_dims]].
    // Mutable running index (per-core varargs): id_per_dim[num_dims] -> local copy.
    uint32_t num_unpadded_sticks[num_dims];
    uint32_t num_padded_sticks[num_dims];
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_common_vararg(j);
        num_padded_sticks[j] = get_common_vararg(num_dims + j);
        id_per_dim[j] = get_vararg(j);
    }

    const uint32_t read_size = unpadded_stick_size + misalignment;

    const auto s0 = TensorAccessor(ta::src);

    Noc noc;
    // Create CircularBuffer for Device 2.0 API
    CircularBuffer cb_in0(cb_id_in0);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = cb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            CoreLocalMem<uint32_t> dst(src_buffer_l1_addr);
            noc.async_read(
                s0, dst, read_size, {.page_id = src_stick_id, .offset_bytes = column_offset_bytes}, {.offset_bytes = 0});
            if (misalignment != 0) {
                noc.async_read_barrier();
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
            }
            src_buffer_l1_addr += stick_size_offset;
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
