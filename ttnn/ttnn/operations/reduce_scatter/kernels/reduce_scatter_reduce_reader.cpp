// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — Phase B reduce reader (NCRISC). THE SCATTER LIVES HERE: pure
// local addressing over the N gathered blocks, no fabric, no cross-device sync.
//
// For each owned output-tile position t (dense in device i's output slice), the
// in-shard SOURCE page for slice my_chip_id is
//
//     src = (t / slice_Wt) * tensor_Wt + my_chip_id * slice_Wt + (t % slice_Wt)
//
// emitted by the shared-schedule SliceRowWalker (host-unit-tested; NOT hand-rolled
// arithmetic — an off-by-one here produces a valid-looking wrong slice). The reader
// then pulls the N gathered tiles gather_buffer[c * P_shard + src], c = 0..N-1, in
// ascending block order into cb_gathered_slices — one barrier, one push of N — and
// the compute kernel sums them (sum_blocks waits its whole N-tile input up front).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;

void kernel_main() {
    constexpr uint32_t cb_gathered_slices = get_compile_time_arg_val(0);
    constexpr uint32_t num_devices = get_compile_time_arg_val(1);  // N
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t scatter_dim = get_compile_time_arg_val(3);  // Phase-0: 3
    constexpr auto gather_buffer_args = TensorAccessorArgs<4>();

    static_assert(sched::is_supported_scatter_dim(scatter_dim), "unsupported scatter dim");
    static_assert(scatter_dim == 3, "Phase-0 wires only dim 3 (slice_C/slice_Ht not plumbed)");

    uint32_t ai = 0;
    const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);  // P_shard (full shard)
    const uint32_t start_tile = get_arg_val<uint32_t>(ai++);       // first owned output position
    const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);        // owned output positions
    const uint32_t slice_Wt = get_arg_val<uint32_t>(ai++);         // tiles per output-slice row
    const uint32_t tensor_Wt = get_arg_val<uint32_t>(ai++);        // tiles per full-shard row

    const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
    const uint32_t P = pages_per_shard;

    // Slice-i origin + this core's start offset within the slice (the same
    // start-offset formula the silicon-verified host helper uses).
    sched::SliceRowWalker walker(slice_Wt, tensor_Wt);
    walker.set_base(sched::slice_tile_offset(scatter_dim, my_chip_id, /*slice_C=*/0, /*slice_Ht=*/0, slice_Wt));
    walker.reset_offsets(start_tile % slice_Wt, (start_tile / slice_Wt) * tensor_Wt);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t src = walker.next();
        cb_reserve_back(cb_gathered_slices, num_devices);
        uint32_t l1 = get_write_ptr(cb_gathered_slices);
        for (uint32_t c = 0; c < num_devices; ++c) {
            noc_async_read(gather_buffer.get_noc_addr(c * P + src), l1, page_size);
            l1 += page_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gathered_slices, num_devices);
    }
}
