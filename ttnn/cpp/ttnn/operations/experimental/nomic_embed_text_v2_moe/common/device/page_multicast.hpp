// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The host side of kernels/dataflow/page_multicast.hpp: one core of an op reads a few pages every core needs
// and multicasts them to the others, rather than every core reading the same pages. 110 cores reading the same
// 24 bias pages from DRAM took 12 us on average and 20 on the slowest core at 4096 tokens (routed_combine).

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// A rectangle of an op's cores in NoC coordinates, and the worker cores the multicast reaches in it.
struct MulticastRectangle {
    CoreCoord start;
    CoreCoord end;
    uint32_t count;
};

// The op's core ranges as NoC rectangles for a multicast on noc, the sender's own counted without it. NOC_1 runs
// the other way, so its rectangles start at the far corner, as ttnn's matmuls swap them. A rectangle may span
// columns without workers (x 8 and 9 on a p300c), as pirotski_matmul's row multicasts do; count is its worker
// cores.
inline std::vector<MulticastRectangle> multicast_rectangles(
    const tt::tt_metal::distributed::MeshDevice& device,
    const CoreRangeSet& cores,
    const CoreCoord& sender,
    tt::tt_metal::NOC noc) {
    std::vector<MulticastRectangle> rectangles;
    for (const CoreRange& range : cores.ranges()) {
        const uint32_t count = static_cast<uint32_t>(range.size()) - (range.contains(sender) ? 1 : 0);
        if (count > 0) {
            CoreCoord start = device.worker_core_from_logical_core(range.start_coord);
            CoreCoord end = device.worker_core_from_logical_core(range.end_coord);
            if (noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(start, end);
            }
            rectangles.push_back({start, end, count});
        }
    }
    return rectangles;
}

// The runtime arguments multicast_pages reads: the rectangles' count, then each one's start x, start y, end x,
// end y and count.
inline std::vector<uint32_t> multicast_rectangle_args(const std::vector<MulticastRectangle>& rectangles) {
    std::vector<uint32_t> args{static_cast<uint32_t>(rectangles.size())};
    for (const MulticastRectangle& rectangle : rectangles) {
        args.insert(
            args.end(),
            {static_cast<uint32_t>(rectangle.start.x),
             static_cast<uint32_t>(rectangle.start.y),
             static_cast<uint32_t>(rectangle.end.x),
             static_cast<uint32_t>(rectangle.end.y),
             rectangle.count});
    }
    return args;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
