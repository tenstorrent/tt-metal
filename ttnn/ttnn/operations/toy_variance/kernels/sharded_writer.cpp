// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the WIDTH-SHARDED toy_variance path -- the gather half of the cross-core combine.
//
// Twice (once for the mean, once for the variance) every core unicasts its Ht-tile partial into its
// own slot of the ROOT's gather buffer and bumps a semaphore there. The root waits for all
// num_cores arrivals, then publishes the whole gather buffer to its compute kernel in one push.
// That is `reduce_root_mcast` from ttnn/ttnn/operations/examples/tensix_all_reduce: all fan-in to
// one core, all the reduction work on that core. See the host file for why this op stays on that
// shape and where to look if the shard grid grows.
//
// The root writes to ITSELF over the NoC rather than special-casing a local copy. One NoC hop is
// cheap next to a branch that would make the address, the semaphore count and the slot map all
// differ on one core.
//
// Two addressing facts hold this together:
//   - A DFB's base address is program-global, so this core's `get_write_ptr()` on the gather buffer
//     IS the root's. There is no API for "that buffer, over there", so inferring it from your own
//     is the practice.
//   - Each gather buffer is used exactly ONCE (hence two of them rather than one reused). That
//     keeps the write pointer at the base with no dependence on the ring pointers of different
//     cores staying in lockstep -- the invariant that quietly carries a reused ring.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

// Send this core's Ht-tile partial to its slot in the root's gather buffer, then tell the root it
// arrived. On the root, publish the gathered block once every core has landed.
template <typename SemT>
FORCE_INLINE void gather_to_root(
    Noc& noc,
    DataflowBuffer& dfb_partial,
    DataflowBuffer& dfb_gather,
    SemT& arrived,
    uint32_t Ht,
    uint32_t num_cores,
    uint32_t root_x,
    uint32_t root_y,
    uint32_t slot,
    bool is_root) {
    const uint32_t block_bytes = Ht * dfb_partial.get_tile_size();

    dfb_gather.reserve_back(num_cores * Ht);
    const uint32_t gather_base = dfb_gather.get_write_ptr();

    dfb_partial.wait_front(Ht);
    noc.async_write(
        dfb_partial,
        UnicastEndpoint{},
        block_bytes,
        {.offset_bytes = 0},
        {.noc_x = root_x, .noc_y = root_y, .addr = gather_base + slot * block_bytes});
    noc.async_write_barrier();
    arrived.up(noc, root_x, root_y, 1);
    // A remote semaphore increment is a POSTED atomic: it is fire-and-forget, so nothing else here
    // waits for it. The root happens to be covered by its own down() below, but a non-root core
    // otherwise runs off the end of the kernel with the increment still in flight -- which the
    // firmware's end-of-kernel drain traps under --dev (waypoint NKFW in
    // ncrisc_noc_posted_writes_sent). Barrier it explicitly; the plain-mode run cannot see the
    // difference, which is exactly why it has to be written down rather than discovered twice.
    noc.async_atomic_barrier();
    dfb_partial.pop_front(Ht);

    if (is_root) {
        arrived.down(num_cores);
        dfb_gather.push_back(num_cores * Ht);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t root_x = get_arg(args::root_x);
    constexpr uint32_t root_y = get_arg(args::root_y);
    const uint32_t is_root = get_arg(args::is_root);
    const uint32_t slot = get_arg(args::gather_slot);

    Noc noc;
    DataflowBuffer dfb_partial(dfb::partial);
    DataflowBuffer dfb_gather_mean(dfb::gather_mean);
    DataflowBuffer dfb_gather_var(dfb::gather_var);
    DataflowBuffer dfb_out(dfb::out_tiles);
    Semaphore mean_arrived(sem::mean_arrived);
    Semaphore var_arrived(sem::var_arrived);

    // Round 1: the mean.
    gather_to_root(noc, dfb_partial, dfb_gather_mean, mean_arrived, Ht, num_cores, root_x, root_y, slot, is_root);

    // Round 2: the variance.
    gather_to_root(noc, dfb_partial, dfb_gather_var, var_arrived, Ht, num_cores, root_x, root_y, slot, is_root);

    // Only the root holds a result to write out.
    if (is_root) {
        const auto acc_out = TensorAccessor(tensor::out);
        const uint32_t tile_bytes = dfb_out.get_tile_size();
        for (uint32_t i = 0; i < Ht; ++i) {
            dfb_out.wait_front(1);
            noc.async_write(dfb_out, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            dfb_out.pop_front(1);
        }
    }
}
