// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "dispatch_fabric2d_placement.hpp"
#include "kernels/dataflow/dispatch_fabric2d_kernel_interface.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/fabric2d_common/fabric2d_assignments.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Who sends what to whom, on which stream and through which channel, is the same ring schedule combine_fabric2d
// runs; see fabric2d_assignments.hpp. A token's destination depends on the data (indices -> dispatch table), so
// the reader builds the per-destination token list on device; an assignment names only the destination and the
// share. The kernel runs all own assignments before any forward, which gives the upstream chip time to write the
// descriptors the forwards consume.
using fabric2d::Assignment;
using fabric2d::forward_chunks_per_stream;
using fabric2d::forwarding_classes;
using fabric2d::generate_assignments;
using fabric2d::num_vcs;
using fabric2d::own_assignments_per_stream;
using fabric2d::vc_after_hop;
using fabric2d::vc_for_own;
static_assert(dspf2d::VC_NONE == fabric2d::VC_NONE, "the host writes fabric2d::VC_NONE into the channel tables");

// Descriptors a stream forwards, in the order the upstream chip emits them into this stream's fwd_section.
std::vector<dspf2d::ChunkDescriptor> forwarding_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links);

// Descriptors this chip writes into the downstream chip's fwd_section for this stream, in emission order:
// its own destinations beyond the downstream chip, furthest first, then the descriptors it forwards. A
// descriptor bound for the downstream chip itself is written straight to that chip's output, so it is not listed.
std::vector<dspf2d::ChunkDescriptor> outgoing_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links);

// The chunks the upstream chip on `stream` writes straight into `my_pos`'s output, one per origin 1 .. extent/2 hops
// upstream: its own share for this chip, then every chunk it forwards whose destination is this chip. Built from
// generate_assignments and forwarding_descriptors for the upstream chip, the lists its program is built from, so the
// receive count the reader derives from them is what the upstream sender sends.
std::vector<dspf2d::ChunkDescriptor> incoming_final_descriptors(
    const std::vector<uint32_t>& ring_chip_ids, StreamId stream, uint32_t my_pos, uint32_t num_links);

// A fwd_section holds no per-chunk addresses, so a chunk is found by counting the chunks before it. What
// a chip writes and what the downstream chip reads must therefore agree in descriptor identity and order,
// or the reader waits for a chunk that is never written. The chips exchange nothing at run time, so this checks
// it on the host.
void validate_descriptor_agreement(uint32_t ring_extent, uint32_t num_links);

// A stream's fwd_section is its slice of the DRAM forwarding buffer: fwd_pages_per_stream pages starting
// at page stream * fwd_pages_per_stream, divided into num_vcs channels of fwd_vc_pages each. Each page is
// one token plus its fwd_meta.
constexpr uint32_t fwd_pages_per_stream(uint32_t ring_extent, uint32_t fwd_vc_pages) {
    return num_vcs(ring_extent) * fwd_vc_pages;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
