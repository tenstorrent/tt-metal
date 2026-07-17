// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of pad's row-major writer, shared by PadRmReaderWriterProgramFactory (single core)
// and PadRmReaderWriterMultiCoreProgramFactory (resnet-shaped multi core).  The device-side NoC +
// TensorAccessor logic is identical to the legacy writer_pad_dims_rm_interleaved.cpp; only the
// resource bindings are migrated to the Metal 2.0 namespaces (dfb::/tensor::/args::).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
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

    // The output tensor (dst) flows in through its TensorBinding; its base address and layout
    // metadata are injected by the framework.
    const auto s1 = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer cb(dfb::cb_in0);

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
