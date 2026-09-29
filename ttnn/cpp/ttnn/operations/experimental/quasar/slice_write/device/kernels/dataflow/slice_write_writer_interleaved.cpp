// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar (Metal-2) slice_write writer. Ported from the shared experimental/slice_write writer to the
// Metal-2 bound/named model (the mirror of the quasar padded_slice reader):
//   * input CB  -> bound DataflowBuffer `dfb::in0` (the resident sharded input; drained here).
//   * dst       -> bound TensorAccessor `tensor::dst` (the interleaved output; per-core last-dim/width
//                  byte offset applied via the dst-side offset_bytes on each write).
//   * per-dim geometry (num_input_sticks/num_output_sticks/id_per_dim) -> positional runtime varargs.
// Writes each input stick to the interleaved output at start_id + the padded-dim walk.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t dst_byte_offset = get_arg(args::dst_byte_offset);  // per-core dst offset (begins_last + width)
    const uint32_t output_stick_size = get_arg(args::output_stick_size);
    const uint32_t input_stick_size = get_arg(args::input_stick_size);
    const uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    const uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    // Positional runtime varargs: [ num_input_sticks[0..num_dims) , num_output_sticks[0..num_dims) ,
    //   id_per_dim[0..num_dims) ]. id_per_dim is mutated locally as the write walks the padded output.
    constexpr uint32_t MAX_RANK = 8;
    uint32_t num_unpadded_sticks[MAX_RANK];
    uint32_t num_padded_sticks[MAX_RANK];
    uint32_t id_per_dim[MAX_RANK];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_vararg(j);
        num_padded_sticks[j] = get_vararg(num_dims + j);
        id_per_dim[j] = get_vararg(2 * num_dims + j);
    }

    const auto s0 = TensorAccessor(tensor::dst);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in0);

    uint32_t dst_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.wait_front(num_read_per_barrier);
        uint32_t src_offset = 0;
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_write(
                cb_in0,
                s0,
                noc_write_size,
                {.offset_bytes = src_offset},
                {.page_id = dst_stick_id, .offset_bytes = dst_byte_offset});
            src_offset += stick_size_offset;
            dst_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    dst_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc.async_write_barrier();
        cb_in0.pop_front(num_read_per_barrier);
    }
}
