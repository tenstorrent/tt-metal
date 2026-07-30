// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) writer kernel for rand: consumer of the intermediate DFB, writes the
// generated tiles to the interleaved output. Port of
// operations/uniform/device/kernels/writer_uniform.cpp to named accessors:
//   - dst buffer address RTA -> TensorAccessor(tensor::output)
//   - CircularBuffer(cb_id)  -> DataflowBuffer(dfb::name)
//   - positional RTAs        -> get_arg(args::name)
// tile_offset/units_per_core are enqueue-invariant runtime args (unchanged across dispatches).
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace tt;

void kernel_main() {
    const uint32_t start_id = get_arg(args::tile_offset);
    const uint32_t num_tiles = get_arg(args::units_per_core);
    const uint32_t end_id = start_id + num_tiles;

    const auto output_addrg = TensorAccessor(tensor::output);

    DataflowBuffer cb_intermed(dfb::intermed);

    Noc noc;

#ifdef OUTPUT_DTYPE_BFLOAT16
    // bfloat16 output: convert each Float32 intermediate tile into the dst staging DFB (a
    // writer self-loop DFB), then NOC-write the packed bf16 page to the output.
    DataflowBuffer cb_dst(dfb::dst);
    const uint32_t page_bytes = cb_dst.get_entry_size();
    cb_dst.reserve_back(1);
    uint32_t dst_cb_write_ptr = cb_dst.get_write_ptr();

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.wait_front(1);
        uint32_t intermed_cb_read_ptr = cb_intermed.get_read_ptr();
        auto intermed_cb_addr = reinterpret_cast<float*>(intermed_cb_read_ptr);

        auto dst_cb_addr = reinterpret_cast<uint8_t*>(dst_cb_write_ptr);
        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_cb_addr;
                uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(&rand_float) + 1;
                *(uint16_t*)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
                intermed_cb_addr += 1;
            }
        }
        cb_intermed.pop_front(1);

        noc.async_write(CoreLocalMem<uint32_t>(dst_cb_write_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
    }
    // dst_cb is a conversion-staging region (consumed only by direct NOC writes); commit the
    // reservation so the self-loop DFB is left balanced.
    cb_dst.push_back(1);
#else
    // float32 output: NOC-write the Float32 intermediate tile directly.
    const uint32_t page_bytes = cb_intermed.get_entry_size();
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.wait_front(1);
        uint32_t intermed_cb_read_ptr = cb_intermed.get_read_ptr();
        noc.async_write(CoreLocalMem<uint32_t>(intermed_cb_read_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb_intermed.pop_front(1);
    }
#endif
}
