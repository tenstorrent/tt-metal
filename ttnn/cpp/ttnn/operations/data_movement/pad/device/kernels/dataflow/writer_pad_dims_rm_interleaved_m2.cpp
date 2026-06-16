// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_pad_dims_rm_interleaved.cpp, used by the
// PadRmReaderWriter (single-core) and PadRmReaderWriterMultiCore factories.
// Forked because both factories share this source; the fork lets them move to the Metal 2.0
// named-binding form together while the legacy copy stays available for unmigrated consumers.
// Logic UNCHANGED from the legacy writer; only the access mechanism moves to named bindings:
//   - output dst tensor address -> ta::dst
//   - input CB id               -> dfb::in0   (consumer of the reader-produced rows)
//   - positional runtime args   -> get_arg(args::...)
// The legacy writer read its args from the shared reader/writer RTA vector by positional index;
// only the slots this writer actually consumed survive, now as named args.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_total_W = get_arg(args::num_total_W);
    const uint32_t num_total_Z = get_arg(args::num_total_Z);
    const uint32_t num_total_Y = get_arg(args::num_total_Y);
    const uint32_t num_total_X = get_arg(args::num_total_X);
    const uint32_t padded_X_nbytes = get_arg(args::padded_X_nbytes);
    const uint32_t start_dst_stick_id = get_arg(args::start_dst_stick_id);
    const uint32_t start_dst_stick_wi = get_arg(args::start_dst_stick_wi);
    const uint32_t num_local_Y = get_arg(args::num_local_Y);
    const uint32_t num_local_unpadded_Y = get_arg(args::num_local_unpadded_Y);
    const uint32_t full_padded_X_nbytes = get_arg(args::full_padded_X_nbytes);
    const uint32_t dst_stick_offset = get_arg(args::dst_stick_offset);  // == start_src_stick_wi * elem_size
    const uint32_t num_local_W = get_arg(args::num_local_W);

    constexpr uint32_t cb_id = dfb::in0;
    DataflowBuffer cb(cb_id);

    const auto s1 = TensorAccessor(ta::dst);
    Noc noc;

    uint32_t dst_stick_id = start_dst_stick_id;
    uint32_t dst_stick_wi = start_dst_stick_wi;
    for (uint32_t w = 0; w < num_local_W; ++w) {
        for (uint32_t z = 0; z < num_total_Z; ++z) {
            for (uint32_t y = 0; y < num_local_Y; ++y) {
                // DPRINT("WR: w={} z={} y={}\n", w, z, y);
                cb.wait_front(1);
                noc.async_write(
                    cb,
                    s1,
                    padded_X_nbytes,
                    {.offset_bytes = 0},
                    {.page_id = dst_stick_id, .offset_bytes = dst_stick_offset});
                noc.async_write_barrier();
                ++dst_stick_id;
                cb.pop_front(1);
            }
        }
    }
}
