// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto unpadded_stick_size = get_arg(args::unpadded_stick_size);
    const auto stick_size_offset = get_arg(args::stick_size_offset);
    const auto num_dims = get_arg(args::num_dims);
    const auto misalignment = get_arg(args::misalignment);
    const auto start_id = get_arg(args::start_id);
    const auto num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const auto num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const auto num_read_per_barrier = get_arg(args::num_read_per_barrier);
    // Sub-row chunking: `num_chunks_per_stick` NOC transfers of `chunk_size` per stick (last = `last_chunk_size`).
    const auto chunk_size = get_arg(args::chunk_size);
    const auto num_chunks_per_stick = get_arg(args::num_chunks_per_stick);
    const auto last_chunk_size = get_arg(args::last_chunk_size);
    // Byte offset of the slice's W-begin within a row, rounded down to the source buffer's alignment;
    // the leftover `misalignment` bytes are trimmed on-device by the tt_memmove below.
    const auto src_offset_bytes = get_arg(args::src_offset_bytes);

    // Three num_dims-long runtime vararg blocks, in host push order:
    //   [0, num_dims)            num_unpadded_sticks per dim
    //   [num_dims, 2*num_dims)   num_padded_sticks per dim
    //   [2*num_dims, 3*num_dims) the per-dim walk counters
    const uint32_t num_unpadded_sticks_base = 0;
    const uint32_t num_padded_sticks_base = num_dims;
    const uint32_t id_per_dim_base = num_dims * 2;

    // The walk counters are seeded by the host and advanced as this kernel walks the input, so they
    // are copied into a local: get_vararg() reads a vararg but cannot write one back. num_dims is a
    // runtime value here, so the local is sized by the accessor's own rank ceiling.
    ASSERT(num_dims <= tensor_accessor::MAX_RANK);
    uint32_t id_per_dim[tensor_accessor::MAX_RANK];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(id_per_dim_base + j);
    }

    uint32_t read_size = unpadded_stick_size + misalignment;

    // The accessor base stays the unshifted buffer base: Metal 2.0 supplies it from the tensor binding
    // and offers no seam for a pre-offset base. The W-begin shift rides each read as `src_offset_bytes`.
    // The per-page stride is the buffer's aligned page size, which the binding supplies; the host
    // pins that it equals the per-shard page size this kernel needs (check_accessor_page_size).
    const auto s0 = TensorAccessor(tensor::src);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in0);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;

    if (num_chunks_per_stick > 1) {
        // Chunked path: batch by `num_read_per_barrier` so the second DFB pair pipelines behind the first.
        for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
            uint32_t c = 0;
            while (c < num_chunks_per_stick) {
                uint32_t batch = num_chunks_per_stick - c;
                if (batch > num_read_per_barrier) {
                    batch = num_read_per_barrier;
                }
                dfb_in0.reserve_back(batch);
                uint32_t src_buffer_l1_addr = dfb_in0.get_write_ptr();
                for (uint32_t k = 0; k < batch; ++k) {
                    const uint32_t cur = c + k;
                    const uint32_t offset = src_offset_bytes + cur * chunk_size;
                    const uint32_t sz = (cur == num_chunks_per_stick - 1) ? last_chunk_size : chunk_size;
                    tt::data_movement::common::noc_async_read_sharded(
                        noc, src_buffer_l1_addr, s0, src_stick_id, offset, sz);
                    src_buffer_l1_addr += chunk_size;
                }
                noc.async_read_barrier();
                dfb_in0.push_back(batch);
                c += batch;
            }
            sticks_read++;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == get_vararg(num_unpadded_sticks_base + j)) {
                    id_per_dim[j] = 0;
                    src_stick_id += get_vararg(num_padded_sticks_base + j);
                } else {
                    break;
                }
            }
        }
        return;
    }

    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        dfb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = dfb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
            // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_read_sharded(
                noc, src_buffer_l1_addr, s0, src_stick_id, /*offset=*/src_offset_bytes, /*size=*/read_size);
            if (misalignment != 0) {
                noc.async_read_barrier();
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
            }
            src_buffer_l1_addr += stick_size_offset;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == get_vararg(num_unpadded_sticks_base + j)) {
                    id_per_dim[j] = 0;
                    src_stick_id += get_vararg(num_padded_sticks_base + j);
                } else {
                    break;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(num_read_per_barrier);
    }
}
