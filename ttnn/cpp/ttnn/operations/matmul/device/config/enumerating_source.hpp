// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

// A candidate source that proposes many blockings of each layout instead of one: the configs a sweep times, so the
// heuristic rules (and later an estimator) can be judged by looking up measured times rather than re-running devices.
//
// The layouts (family, per-core split, and the rules a sharded tensor puts on the blocking) are the ones the factory
// blocking source considers, recorded while it runs, plus each multicast layout with its batch looped over instead of
// fused. Within a layout it varies the K depth (in0_block_w, thinned to one divisor of Kt per power of two) and the
// output blocks (the largest that fit L1 at each depth), and per block a few subblock shapes (HeuristicSubblock's, the
// widest row, the tallest column, and the largest-area ones). The heuristic source's own choice is always proposed
// first. Every proposal fits L1 and passes factory_limit_error; the device op's validation (check()) is left to the
// caller.
namespace ttnn::operations::matmul::auto_config {

class EnumeratingSource final : public CandidateSource {
public:
    // Output blocks kept per K depth in a layout: the largest by area, then the tallest and widest of the rest
    static constexpr uint32_t blocks_per_k = 3;

    std::string_view name() const override { return "enumerating"; }

    std::vector<Candidate> propose(const MatmulDesc& matmul, const HardwareDesc& hw) const override;
};

}  // namespace ttnn::operations::matmul::auto_config
