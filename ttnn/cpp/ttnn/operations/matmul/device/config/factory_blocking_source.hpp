// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

// The candidate source built on the factories' blocking rules. For each family that can run the problem it
// fixes the layout's per-core split (the grid the family covers, or a sharded tensor's shards), and three
// policies make the remaining choices:
//  - BlockingPolicy: K depth (in0_block_w) and output blocks within the split, inside the L1 budget;
//  - SubblockPolicy: the output subblock within a block;
//  - FamilyPolicy: which family's candidate to propose.
// The proposal is the chosen candidate and, for interleaved problems, its legal K-depth neighbours, for the
// estimators to rank. Sharded tensors constrain the choice rather than change the rules: a sharded A fixes the
// family, grid and per-core sizes (width -> 1D in0-mcast, height -> 1D in1-mcast or Reuse for batched B,
// block -> 2D), a sharded output (with interleaved inputs) fixes the family (and with a shard spec, the grid
// and per-core sizes), and the policies choose what remains within the layout's BlockRules.
namespace ttnn::operations::matmul::auto_config {

// Extra conditions a layout puts on the blocking.
struct BlockRules {
    uint32_t k_divides = 0;    // in0_block_w must also divide this (a sharded A's shard width)
    uint32_t k_fixed = 0;      // in0_block_w must be exactly this
    bool sharded_out = false;  // out_block_w == per_core_N
    // in0_block_w to use when some block fits with it, even above the K depth limit: a width- or block-sharded
    // A's shard width. Each K block is then a whole shard column, multicast in place; a narrower one makes the
    // sender copy every K block out of the shard first (extract_shard_sub_blocks).
    uint32_t k_preferred = 0;

    bool prefers(uint32_t k) const { return k_preferred != 0 && k == k_preferred; }
    bool prefers_other(uint32_t k) const { return k_preferred != 0 && k != k_preferred; }
};

// A layout's per-core work: per_core_M x per_core_N output tiles, with the batch folded into M or looped over
struct Split {
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    bool fuse_batch = true;
};

// Chooses in0_block_w and the output blocks for a split (subblocks left 0), or nullopt if nothing fits L1
class BlockingPolicy {
public:
    virtual ~BlockingPolicy() = default;
    virtual std::optional<Blocking> block(
        const MatmulDesc& matmul,
        const HardwareDesc& hw,
        Family family,
        const Split& split,
        const BlockRules& rules) const = 0;
};

// Sets the output subblock of a blocking
class SubblockPolicy {
public:
    virtual ~SubblockPolicy() = default;
    virtual Blocking subblock(const MatmulDesc& matmul, Family family, Blocking blocking) const = 0;
};

// Chooses one of the families' candidates (at most one per family, in family order)
class FamilyPolicy {
public:
    virtual ~FamilyPolicy() = default;
    virtual std::optional<Candidate> choose(
        const MatmulDesc& matmul, const HardwareDesc& hw, std::span<const Candidate> candidates) const = 0;
};

// Issue #57884's block-size heuristics with one K block depth rule:
//  - 2D: the largest in0_block_w * out_block_h * out_block_w that fits L1; ties go to the larger output block,
//    then the squarer one. Interleaved 2D then goes no shallower than the legacy selection's K depth where that
//    was measured to pay (see deepen_to_legacy_k_depth);
//  - 1D: the full per-core extent along the multicast dimension, the other one shrunk only if needed, unless
//    that forces single-tile K steps; 1D in0-mcast splits a wide output block into subblock-wide blocks;
//  - Reuse: the deepest in0_block_w within the K depth rule that fits.
// A layout's preferred in0_block_w wins whenever some block fits with it.
class HeuristicBlocking final : public BlockingPolicy {
public:
    struct Params {
        // K block depth, for every family: in0_block_w is at most this. Deeper K blocks stop paying for
        // themselves, and in 2D the block-size heuristic would otherwise trade output-block size (the only source
        // of data reuse) for K depth. The mcast families also keep at least two K blocks, since with a single
        // block they single-buffer the inputs.
        uint32_t max_in0_block_w = 8;
        // 2D output blocks of more than this many tiles may use K blocks up to 2 * max_in0_block_w deep. Every K
        // block ends with a pack of the whole output block (L1 accumulation of the partials), which sits on the
        // compute path; a large block is compute bound, so deeper K blocks amortize that pack. Smaller blocks
        // wait on data, where the pack is hidden and a deeper K block only lengthens the pipeline fill. On the
        // Wormhole 2D sweeps, with the output block fixed, K depth 16 beat 8 on 6 of 10 larger blocks and lost on
        // none, while on smaller blocks 8 won 51 to 19. The threshold is where the sweeps turn, not derived.
        uint32_t large_block_tiles = 64;
        // K block depth is further limited so that the operand a core reads by itself (not by multicast) moves at
        // most this many tiles per K step: B's slice in 1D in0-mcast, A's in 1D in1-mcast, both in Reuse, none in
        // 2D. Small per-step reads keep the double-buffered DRAM stream ahead of math; wide per-core blocks get
        // shallower K blocks, but never below min_in0_block_w (single-tile K steps pay a block handshake per
        // tile).
        uint32_t max_self_read_tiles_per_k_step = 8;
        uint32_t min_in0_block_w = 2;
    };

    HeuristicBlocking() = default;
    explicit HeuristicBlocking(const Params& params) : params_(params) {}
    const Params& params() const { return params_; }

    std::optional<Blocking> block(
        const MatmulDesc& matmul,
        const HardwareDesc& hw,
        Family family,
        const Split& split,
        const BlockRules& rules) const override;

private:
    Params params_;
};

// The largest-area subblock that fits DST and divides the block, two tiles or more on each side unless B's tiles
// are smaller than A's (see the .cpp for why)
class HeuristicSubblock final : public SubblockPolicy {
public:
    Blocking subblock(const MatmulDesc& matmul, Family family, Blocking blocking) const override;
};

// The family rules:
//  - B not batched: 2D mcast, unless a 1D layout keeps at least one_d_core_advantage times as many cores busy
//    (small M or small N), or 1D in0-mcast keeps as many cores busy with less input per core;
//  - batched B: Reuse, unless the multicast layout looping over the batch (chosen as above) keeps
//    one_d_core_advantage times as many cores busy, or Reuse would read one_d_core_advantage times as much input;
//  - a 2D choice whose per-core blocks are one tile tall or wide: the lowest roofline estimate instead.
class HeuristicFamily final : public FamilyPolicy {
public:
    struct Params {
        // Switching away from the default layout needs at least this many times as many cores busy: 1D over 2D
        // (1D multicasts a whole operand to every core), and for batched B a batch-looping multicast layout over
        // Reuse. On the Wormhole sweep anything from 1.25 to 2 performs about the same.
        double one_d_core_advantage = 1.5;
    };

    HeuristicFamily() = default;
    explicit HeuristicFamily(const Params& params) : params_(params) {}

    std::optional<Candidate> choose(
        const MatmulDesc& matmul, const HardwareDesc& hw, std::span<const Candidate> candidates) const override;

private:
    Params params_;
};

class FactoryBlockingSource final : public CandidateSource {
public:
    // The heuristic policies
    FactoryBlockingSource();
    FactoryBlockingSource(
        std::shared_ptr<const BlockingPolicy> blocking,
        std::shared_ptr<const SubblockPolicy> subblock,
        std::shared_ptr<const FamilyPolicy> family);

    std::string_view name() const override { return "factory_blocking"; }

    // The family policy's choice, then (interleaved problems) its K-depth neighbours
    std::vector<Candidate> propose(const MatmulDesc& matmul, const HardwareDesc& hw) const override;

    // The blocked candidate of each family that can run the problem and fits L1, in family order
    std::vector<Candidate> candidates(const MatmulDesc& matmul, const HardwareDesc& hw) const;

private:
    std::vector<Candidate> sharded_candidates(const MatmulDesc& matmul, const HardwareDesc& hw) const;
    std::optional<Blocking> reuse_blocking(const MatmulDesc& matmul, const HardwareDesc& hw) const;

    std::shared_ptr<const BlockingPolicy> blocking_;
    std::shared_ptr<const SubblockPolicy> subblock_;
    std::shared_ptr<const FamilyPolicy> family_;
};

// A candidate's K-depth neighbours: the same candidate at the next deeper and the next shallower in0_block_w
// dividing K within the factory limits (factory_limit_error; select() then checks every proposal in full).
// Interleaved problems only (a sharded layout constrains K depth); empty otherwise.
std::vector<Candidate> k_depth_neighbours(const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate);

}  // namespace ttnn::operations::matmul::auto_config
