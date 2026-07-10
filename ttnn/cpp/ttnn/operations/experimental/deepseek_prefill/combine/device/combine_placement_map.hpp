// // SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #pragma once

// // ============================================================================
// // SENDERS_CUSTOM_PLACEMENT — rule-based sender/untilizer placement
// // ============================================================================
// //
// // HOST-FACTORY ONLY (consumed by combine_program_factory.cpp; device kernels are geometry-agnostic
// // and address cores by NOC coord via runtime args). Flipping this / editing the rule requires a
// // libttnn REBUILD.
// //
// // When set to 1, this OVERRIDES SENDERS_ON_SEPARATE_ROWS (overlap_config.hpp) on every device. For
// // each sender the factory resolves — from the fabric control plane — the physical NOC0 coord(s) of
// // the eth (EDM) core(s) that sender drives (one per combine-axis neighbor; both share a routing
// // plane) and their routing plane, then calls combine_place_sender() below to get the sender +
// // untilizer NOC0 coords. Coords are then mapped NOC0 -> logical (== NOC/virtual on an unharvested
// // BH tensix grid; a coord that doesn't resolve to a worker hard-errors, never silently misplaced).
// //
// // Why: writer_combine (sender->EDM) and writer_untilize (untilizer->sender) both run on NOC_1
// // (-X-then-Y). If a sender's eth core lies -X past its own untilizers, its payload self-congests
// // crossing them. The rule places the sender so its eth is reached without crossing its own
// // untilizers, and puts each sender's group on a row keyed by its eth routing plane.
// #define SENDERS_CUSTOM_PLACEMENT 1

// // How many untilizer cores each sender gets (1..MAX_UNTILIZERS_PER_SENDER=4). Passed to the rule as
// // num_untilizers.
// #define CUSTOM_PLACEMENT_UNTILIZERS_PER_SENDER 4

// #include <algorithm>
// #include <cstdint>
// #include <vector>

// namespace ttnn::operations::experimental::deepseek_prefill::combine {

// // A physical NOC0 core coordinate.
// struct PlacementCoord {
//     uint32_t x;
//     uint32_t y;
// };

// // Rule output: the sender core + its untilizer cores (all physical NOC0), untilizers ordered
// // left-to-right from the sender.
// struct SenderPlacementOut {
//     PlacementCoord sender;
//     std::vector<PlacementCoord> untilizers;
// };

// // Next USABLE worker physical-x strictly greater than x, wrapping to the smallest usable x. The
// // usable set (passed in, sorted ascending) already excludes non-Tensix columns (x=8 router spine,
// // x=9 DRAM) AND columns this op can't use (e.g. dispatch-reserved high-x columns — on BH p150 the
// // compute grid is only x in {1..7,10..13}, so 14/15/16 are absent). Used to lay untilizers to the
// // sender's right.
// inline uint32_t combine_next_usable_x(uint32_t x, const std::vector<uint32_t>& usable_x_sorted) {
//     for (uint32_t cand : usable_x_sorted) {
//         if (cand > x) {
//             return cand;
//         }
//     }
//     return usable_x_sorted.front();  // wrap
// }

// // Snap x to the nearest usable worker x (ties -> lower). Used when an eth core's x is not itself a
// // usable worker column (e.g. eth at x=16 but x=16 is dispatch-reserved), so the sender can't sit
// // directly under it. DEFAULT policy = nearest; revisit if you want "nearest to the left/right".
// inline uint32_t combine_snap_to_usable(uint32_t x, const std::vector<uint32_t>& usable_x_sorted) {
//     uint32_t best = usable_x_sorted.front();
//     uint32_t best_d = (x > best) ? (x - best) : (best - x);
//     for (uint32_t cand : usable_x_sorted) {
//         uint32_t d = (x > cand) ? (x - cand) : (cand - x);
//         if (d < best_d) {
//             best_d = d;
//             best = cand;
//         }
//     }
//     return best;
// }

// // ============================================================================
// // THE PLACEMENT RULE
// // ============================================================================
// // Inputs (all physical NOC0):
// //   connected_eth_noc0 : the 1 or 2 eth cores this sender drives (both share a routing plane).
// //   eth_routing_plane  : that shared routing plane id.
// //   num_untilizers     : how many untilizer cores to place (== CUSTOM_PLACEMENT_UNTILIZERS_PER_SENDER).
// // Returns sender + untilizer NOC0 coords. The factory validates every coord is a real, unused worker.
// inline SenderPlacementOut combine_place_sender(
//     const std::vector<PlacementCoord>& connected_eth_noc0,
//     uint32_t eth_routing_plane,
//     uint32_t num_untilizers,
//     const std::vector<uint32_t>& usable_x_sorted) {
//     // Rule 1 + 2: choose the sender's x from the eth cores' x coords.
//     uint32_t picked_x;
//     if (connected_eth_noc0.size() == 1) {
//         picked_x = connected_eth_noc0[0].x;  // only one eth -> take it
//     } else {
//         uint32_t lo = connected_eth_noc0[0].x;
//         uint32_t hi = connected_eth_noc0[0].x;
//         for (const auto& e : connected_eth_noc0) {
//             lo = std::min(lo, e.x);
//             hi = std::max(hi, e.x);
//         }
//         uint32_t diff = hi - lo;
//         // Discount the non-Tensix gap (x=8 router spine + x=9 DRAM) when the eth cores straddle it.
//         if (lo < 8 && hi > 8) {
//             diff -= 2;
//         }
//         // Wide gap (can't fit the untilizers between) -> anchor on the lower eth; otherwise the higher.
//         picked_x = (diff > num_untilizers) ? lo : hi;
//     }
//     // The eth-derived x may not itself be a usable worker column (e.g. eth at x=16 but x=16 is
//     // dispatch-reserved) -> snap the sender to the nearest usable column.
//     picked_x = combine_snap_to_usable(picked_x, usable_x_sorted);

//     // Rule 5: row from the routing plane (plane 0 -> y=2, plane 1 -> y=3, ...).
//     const uint32_t y = 2 + eth_routing_plane;

//     // Rule 3 + 4: sender at picked_x; untilizers at the next num_untilizers USABLE worker x coords.
//     SenderPlacementOut out;
//     out.sender = {picked_x, y};
//     uint32_t x = picked_x;
//     for (uint32_t i = 0; i < num_untilizers; i++) {
//         x = combine_next_usable_x(x, usable_x_sorted);
//         out.untilizers.push_back({x, y});
//     }
//     return out;
// }

// }  // namespace ttnn::operations::experimental::deepseek_prefill::combine
