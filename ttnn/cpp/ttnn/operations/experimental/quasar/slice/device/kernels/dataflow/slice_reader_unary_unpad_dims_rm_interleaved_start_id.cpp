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
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Named runtime args (formerly fixed reader slots [0..9]). The legacy slot [0] packed the
    // source buffer base address plus the W-dim slice start; under the spec ABI the base
    // address is bound via the INPUT TensorParameter (TensorAccessor(tensor::in)), so only the
    // W-dim byte offset (begins_offset_bytes = begins_bytes - misalignment) is passed and applied
    // to the accessor page address via the NOC read's `offset` parameter.
    const uint32_t begins_offset_bytes = get_arg(args::begins_offset_bytes);
    const uint32_t unpadded_stick_size = get_arg(args::unpadded_stick_size);
    const uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    // num_dims is a compile-time arg (same on every core; bounds the vararg loops below).
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t misalignment = get_arg(args::misalignment);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);

    // NOTE: the legacy reader slot [1] (padded_stick_size = per-shard page size override) is
    // dropped: the per-shard split in noc_async_read_sharded now uses the bound TensorAccessor's
    // own aligned page size, derived from the input tensor's TensorSpec.

    // num_unpadded_sticks / num_padded_sticks are per-dim arrays read in the inner loop by a
    // runtime-varying index, so they arrive as common runtime varargs: [0, num_dims) is
    // num_unpadded_sticks and [num_dims, 2*num_dims) is num_padded_sticks.
    uint32_t num_unpadded_sticks[num_dims];
    uint32_t num_padded_sticks[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_common_vararg(j);
        num_padded_sticks[j] = get_common_vararg(num_dims + j);
    }

    // id_per_dim is a per-core array advanced in the inner loop by a runtime-varying index → runtime varargs.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // Source buffer is bound via the INPUT TensorParameter; the per-shard page size used by
    // noc_async_read_sharded's multi-shard split is derived from the tensor's TensorSpec.
    const auto s0 = TensorAccessor(tensor::in);

    uint32_t read_size = unpadded_stick_size + misalignment;

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer cb_in0(dfb::cb_in);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = cb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
            // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
            // begins_offset_bytes (the W-dim slice start, aligned down) is applied as the read
            // offset: for an interleaved accessor this lands at base + page + begins_offset_bytes,
            // identical to the legacy fold into the buffer base address.
            tt::data_movement::common::noc_async_read_sharded(
                noc, src_buffer_l1_addr, s0, src_stick_id, /*offset=*/begins_offset_bytes, /*size=*/read_size);
            if (misalignment != 0) {
                noc.async_read_barrier();
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
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
