// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/rule_ranker.hpp"

#include <algorithm>
#include <tuple>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"
#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

namespace {

// Input tiles per K tile each core reads: its rows of A plus its columns of B
uint32_t per_core_input_tiles(const Blocking& b) { return b.per_core_M + b.per_core_N; }

// Input tiles read from memory in total. The mcast layouts read A once per output column block and B once
// per output row block (each shared by multicast); Reuse reads A once and B once per M slice of a batch.
uint64_t total_input_tiles(const MatmulDesc& p, Family family, const Blocking& b) {
    const uint64_t a_tiles = static_cast<uint64_t>(p.batch_a) * p.Mt * p.Kt;
    const uint64_t b_tiles = static_cast<uint64_t>(p.batch_b) * p.Kt * p.Nt;
    if (family == Family::Reuse) {
        return a_tiles + b_tiles * div_up(p.Mt, b.per_core_M);
    }
    return a_tiles * (b.per_core_N / b.out_block_w) + b_tiles * (b.per_core_M / b.out_block_h);
}

// Among the multicast layouts: 2D unless a 1D layout keeps clearly more cores busy (in0-mcast first when both
// do), or 1D in0-mcast keeps as many cores busy with less input per core.
const Candidate* choose_mcast(
    double one_d_core_advantage, const Candidate* two_d, const Candidate* in0, const Candidate* in1) {
    if (!two_d) {
        return in0 ? in0 : in1;
    }
    const double threshold = one_d_core_advantage * two_d->cores;
    if (in0 && in0->cores >= threshold && (!in1 || in0->cores >= in1->cores)) {
        return in0;
    }
    if (in1 && in1->cores >= threshold) {
        return in1;
    }
    if (in0 && in0->cores >= two_d->cores &&
        per_core_input_tiles(in0->blocking) < per_core_input_tiles(two_d->blocking)) {
        return in0;  // a taller, squarer per-core block reads less input
    }
    return two_d;
}

const Candidate* choose_family(double one_d_core_advantage, const MatmulDesc& p, std::span<const Candidate> all) {
    auto find = [&](Family family) -> const Candidate* {
        for (const auto& c : all) {
            if (c.family == family) {
                return &c;
            }
        }
        return nullptr;
    };
    const Candidate* mcast =
        choose_mcast(one_d_core_advantage, find(Family::Mcast2D), find(Family::Mcast1DIn0), find(Family::Mcast1DIn1));
    // Batched B: Reuse, whose cores work on their blocks independently, unless the multicast layout (which
    // loops over the batch across the whole grid) keeps clearly more cores busy, or Reuse would read clearly
    // more input (it re-reads a batch's B for every M slice it splits the batch matrix into).
    if (const auto* reuse = find(Family::Reuse)) {
        if (mcast && (mcast->cores >= one_d_core_advantage * reuse->cores ||
                      total_input_tiles(p, Family::Reuse, reuse->blocking) >=
                          one_d_core_advantage * total_input_tiles(p, mcast->family, mcast->blocking))) {
            return mcast;
        }
        return reuse;
    }
    return mcast;
}

// A 2D layout whose per-core blocks are one tile tall or wide multicasts single tiles along that axis, so the
// reuse the rules above count on isn't there: the estimate picks the family instead (ties to the earlier family: 2D,
// 1D in0, 1D in1, Reuse), among the candidates no other keeps one_d_core_advantage times as many cores busy as. The
// roofline has no per-step latency, so on its own it can prefer a layout that loops serially over many small steps on
// a few cores (2D over a batch) to one that does them at once on many (Reuse). With batched B, Reuse stays the default
// unless the estimate's choice is one_d_core_advantage times better or takes fewer serial steps (output blocks times K
// blocks, per batch loop) on its busiest core.
bool one_tile_2d(const Candidate& c) {
    return c.family == Family::Mcast2D && std::min(c.blocking.per_core_M, c.blocking.per_core_N) == 1;
}

}  // namespace

RuleRanker::Params RuleRanker::Params::for_arch(tt::ARCH /*arch*/) { return {}; }

RuleRanker::RuleRanker() : RuleRanker(std::nullopt, std::make_shared<RooflineEstimator>()) {}

RuleRanker::RuleRanker(std::optional<Params> params, std::shared_ptr<const Estimator> estimator) :
    params_(params), estimator_(std::move(estimator)) {}

std::vector<const Candidate*> RuleRanker::rank(
    const MatmulDesc& p, const HardwareDesc& hw, std::span<const Candidate> candidates) const {
    const Params params = params_.value_or(Params::for_arch(hw.arch));
    const double advantage = params.tuned.one_d_core_advantage;
    // Each family's first candidate is compared; a second one of the same family loops over the batch
    std::vector<Candidate> firsts;
    std::vector<const Candidate*> first_of;  // firsts[i] is a copy of *first_of[i]
    for (const auto& c : candidates) {
        const bool seen =
            std::any_of(firsts.begin(), firsts.end(), [&](const Candidate& f) { return f.family == c.family; });
        if (!seen) {
            firsts.push_back(c);
            first_of.push_back(&c);
        }
    }
    auto cycles = [&](const Candidate& c) {
        const auto e = estimator_ ? estimator_->estimate(p, hw, c) : std::nullopt;
        return e ? e->cycles : 0.0;
    };
    const Candidate* chosen = choose_family(advantage, p, firsts);
    if (chosen && one_tile_2d(*chosen)) {
        uint32_t most_cores = 0;
        for (const auto& c : firsts) {
            most_cores = std::max(most_cores, c.cores);
        }
        auto key = [&](const Candidate& c) {
            const bool clearly_fewer_cores = c.cores * advantage <= most_cores;
            return std::make_tuple(clearly_fewer_cores, cycles(c), static_cast<int>(c.family));
        };
        const Candidate* best = &*std::min_element(
            firsts.begin(), firsts.end(), [&](const Candidate& x, const Candidate& y) { return key(x) < key(y); });
        chosen = best;
        if (p.batch_b > 1 && best->family != Family::Reuse) {
            auto steps = [&](const Candidate& c) {
                const uint64_t k_blocks = div_up(p.Kt, c.blocking.in0_block_w);
                if (c.family == Family::Reuse) {
                    return div_up(p.batch_a * p.Mt / c.blocking.per_core_M, hw.grid.x * hw.grid.y) * k_blocks;
                }
                const uint64_t loops = c.fuse_batch ? 1 : std::max(p.batch_a, p.batch_b);
                return uint64_t{c.blocking.per_core_M / c.blocking.out_block_h} *
                       (c.blocking.per_core_N / c.blocking.out_block_w) * k_blocks * loops;
            };
            const double best_cycles = cycles(*best);
            for (const auto& c : firsts) {
                if (c.family == Family::Reuse && !std::get<0>(key(c)) && steps(c) <= steps(*best) &&
                    cycles(c) < advantage * best_cycles) {
                    chosen = &c;
                    break;
                }
            }
        }
    }
    std::vector<const Candidate*> result;
    if (chosen) {
        const Candidate* first = first_of[chosen - firsts.data()];
        // The chosen family's batch-looping candidate, then its fused one
        for (const auto& c : candidates) {
            if (&c != first && c.family == first->family) {
                result.push_back(&c);
            }
        }
        result.push_back(first);
    }
    for (const auto& c : candidates) {
        if (std::find(result.begin(), result.end(), &c) == result.end()) {
            result.push_back(&c);
        }
    }
    return result;
}

}  // namespace ttnn::operations::matmul::auto_config
