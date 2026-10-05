// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

// The rule ranker: picks among the families' candidates by rules on busy cores and input read, with
// Tuned::one_d_core_advantage as the "clearly better" margin. Where those rules have no basis (2D blocks one tile tall
// or wide), an estimator (the roofline by default) ranks the families instead. Different layouts' blocks don't carry
// the same fixed cost, so this comparison has no per-block term.
namespace ttnn::operations::matmul::auto_config {

// The family rules, on each family's first candidate (the batch fused, as HeuristicGenerator produces them):
//  - B not batched: 2D mcast, unless a 1D layout keeps at least one_d_core_advantage times as many cores busy
//    (small M or small N), or 1D in0-mcast keeps as many cores busy with less input per core;
//  - batched B: Reuse, unless the multicast layout looping over the batch (chosen as above) keeps
//    one_d_core_advantage times as many cores busy, or Reuse would read one_d_core_advantage times as much input;
//  - a 2D choice whose per-core blocks are one tile tall or wide: the lowest estimate instead, among the candidates
//    no other keeps one_d_core_advantage times as many cores busy as; with batched B, Reuse unless that estimate is
//    one_d_core_advantage times better or takes fewer serial steps.
// The chosen family's batch-looping candidate, if it has one, ranks first: the estimate can't see the write overlap
// that looping buys, so the families are compared fused. The other candidates follow in their given order.
class RuleRanker final : public Ranker {
public:
    // Tuned: fitted to benchmark data (see HeuristicBlocking::Tuned)
    struct Tuned {
        // Switching away from the default layout needs at least this many times as many cores busy: 1D over 2D
        // (1D multicasts a whole operand to every core), and for batched B a batch-looping multicast layout over
        // Reuse; where the estimate picks the family, a candidate with this many times fewer cores busy is out.
        // Basis: the Wormhole family sweep; range 1.25 to 2 performs about the same.
        double one_d_core_advantage = 1.5;
    };
    struct Params {
        Tuned tuned;
        // The values for an architecture (the same for every architecture today)
        static Params for_arch(tt::ARCH arch);
    };

    // The values for each matmul's architecture; the roofline for one-tile 2D layouts
    RuleRanker();
    RuleRanker(std::optional<Params> params, std::shared_ptr<const Estimator> estimator);

    std::string_view name() const override { return "rules"; }
    std::vector<const Candidate*> rank(
        const MatmulDesc& matmul, const HardwareDesc& hw, std::span<const Candidate> candidates) const override;

private:
    std::optional<Params> params_;
    std::shared_ptr<const Estimator> estimator_;
};

}  // namespace ttnn::operations::matmul::auto_config
