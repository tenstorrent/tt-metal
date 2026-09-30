// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-side helpers shared by the reader, sender and copy-writer kernels.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_walk.hpp"

namespace fag = ttnn::operations::experimental::fabric_all_gather::walk;

// Common runtime-arg layout of the geometry block, identical in every kernel (see the factory):
//   [kGeomBase + 0] prefix metadata address (0 = none)   [+1] host active stripe pages
//   [+2] stripes A   [+3] stripe pages B_max   [+4] group size G   [+5] slab (global)   [+6] full extent (global)
//   [+7] pages per slab
struct Geometry {
    uint32_t num_stripes;
    uint32_t stripe_pages;      // active pages per stripe
    uint32_t stripe_pages_max;  // placement stride
    uint32_t group_size;
};

// Read one uint32 from element 0 of a 1-element DRAM metadata tensor.
template <typename Accessor>
FORCE_INLINE uint32_t read_metadata_word(const Accessor& accessor, uint32_t scratch_l1) {
    noc_async_read(accessor.get_noc_addr(0), scratch_l1, sizeof(uint32_t));
    noc_async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
}

template <bool kPrefixFromMetadata, typename PrefixArgs>
FORCE_INLINE Geometry
read_geometry(uint32_t base, uint32_t page_bytes, uint32_t scratch_l1, const PrefixArgs& prefix_args) {
    Geometry g{};
    g.num_stripes = get_common_arg_val<uint32_t>(base + 2);
    g.stripe_pages_max = get_common_arg_val<uint32_t>(base + 3);
    g.group_size = get_common_arg_val<uint32_t>(base + 4);
    if constexpr (kPrefixFromMetadata) {
        const uint32_t prefix_addr = get_common_arg_val<uint32_t>(base + 0);
        const auto prefix = TensorAccessor(prefix_args, prefix_addr, sizeof(uint32_t));
        const uint32_t start = read_metadata_word(prefix, scratch_l1);
        g.stripe_pages = fag::prefix_stripe_pages(
            start,
            get_common_arg_val<uint32_t>(base + 5),
            get_common_arg_val<uint32_t>(base + 6),
            get_common_arg_val<uint32_t>(base + 7));
    } else {
        (void)page_bytes;
        (void)scratch_l1;
        (void)prefix_args;
        g.stripe_pages = get_common_arg_val<uint32_t>(base + 1);
    }
    return g;
}

// Output page of (rank, stripe, stripe-local page).
FORCE_INLINE uint32_t output_page(const Geometry& g, uint32_t rank, uint32_t stripe, uint32_t page) {
    return (stripe * g.group_size + rank) * g.stripe_pages_max + page;
}
