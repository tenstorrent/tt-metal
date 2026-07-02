// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — per-direction writer (BRISC).
//
// Pure data movement (no compute). Fabric egress goes through the
// safety-by-construction CCL helper (FabricStreamSender -> FabricStream ->
// UnicastWriteChannel / AtomicIncChannel). The op owns the store-and-forward
// composition the helper explicitly does NOT own: the ring slice order, the
// concat-by-gather_dim output page addressing (out_page = c*P + p), and the
// counting atomic-inc that signals each landed block to the downstream reader.
//
// For chip id i, direction d (0=forward -> i+1, 1=backward -> i-1):
//   * Forward writer forwards: seed block i, then relay blocks i-1, i-2, ..., 0
//     (num_targets_backward of them) -> neighbour i+1.
//   * Backward writer forwards: seed block i, then relay blocks i+1, ..., N-1
//     (num_targets_forward of them) -> neighbour i-1.
// Every fabric write lands DIRECTLY into the downstream device's persistent
// output DRAM at the block's canonical page range; one counting inc per block
// tells that device's reader the block has landed.
//
// A line-end worker in its missing direction (my_num_targets==0) opens no
// connection and returns immediately — its reader is a pure receiver.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

// Whole-page concat-by-gather_dim addressing (Refinement 2). See the matching
// comment in all_gather_reader.cpp. Reduces to c*P+p for gather_dim=0.
inline uint32_t gather_out_page(uint32_t c, uint32_t p, uint32_t dim_j, uint32_t inner_stride, uint32_t ring_size) {
    const uint32_t block = dim_j * inner_stride;
    const uint32_t high = p / block;
    const uint32_t rem = p % block;
    const uint32_t mid = rem / inner_stride;
    const uint32_t low = rem % inner_stride;
    return high * (ring_size * block) + (c * dim_j + mid) * inner_stride + low;
}

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t direction = get_compile_time_arg_val(1);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_targets_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t num_targets_bwd = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr auto output_args = TensorAccessorArgs<7>();

    constexpr uint32_t my_num_targets = (direction == 0) ? num_targets_fwd : num_targets_bwd;
    constexpr uint32_t num_relay_blocks = (direction == 0) ? num_targets_bwd : num_targets_fwd;

    // Line end in this direction: no fabric egress, no connection opened.
    if constexpr (my_num_targets == 0) {
        return;
    }

    uint32_t ai = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t num_hops = get_arg_val<uint32_t>(ai++);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t target_noc_x = get_arg_val<uint32_t>(ai++);
    const uint32_t target_noc_y = get_arg_val<uint32_t>(ai++);
    const uint32_t dim_j = get_arg_val<uint32_t>(ai++);         // gathered-axis page size
    const uint32_t inner_stride = get_arg_val<uint32_t>(ai++);  // pages inner to gathered axis

    // Fabric connection arg block (laid out by append_ccl_fabric_rt_args); its
    // leading has_forward flag also encodes the send direction.
    size_t conn_arg_idx = ai;
    const bool dst_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

    const auto output = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t P = pages_per_shard;

    // open(route) binds the stream's route once; arm_* yield the only handles that can issue and
    // reuse it.
    auto stream = sender.open(unicast_route(num_hops));
    auto writer = stream.arm_unicast_write(page_size);  // invariant per-page payload size
    auto counter = stream.arm_inc(1);                   // invariant counting inc value

    const uint64_t neighbor_sem = safe_get_noc_addr(target_noc_x, target_noc_y, counting_sem_addr, 0);

    // Forward each block (seed first, then relays) one hop. The reader pushes the
    // same block order into cb_relay_pages, so a single FIFO drain matches.
    for (uint32_t k = 0; k <= num_relay_blocks; ++k) {
        uint32_t c;
        if (k == 0) {
            c = my_chip_id;  // seed
        } else {
            c = (direction == 0) ? (my_chip_id - k) : (my_chip_id + k);  // relays
        }
        for (uint32_t p = 0; p < P; ++p) {
            cb_wait_front(cb_relay_pages, 1);
            const uint32_t l1 = get_read_ptr(cb_relay_pages);
            const uint32_t out_p = gather_out_page(c, p, dim_j, inner_stride, ring_size);
            writer.write_page(l1, out_p, output);
            noc_async_writes_flushed();  // ensure the page was read from the CB slot before reuse
            cb_pop_front(cb_relay_pages, 1);
        }
        // In-order on the connection: this inc lands after the block's data.
        counter.inc(neighbor_sem);
    }

    stream.close();  // drains (write + atomic barriers) then closes
}
