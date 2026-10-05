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

// The heuristic generator: one candidate per family, cut by the factories' blocking rules. For each family that can
// run the problem it fixes the layout's per-core split (the grid the family covers, or a sharded tensor's shards),
// and two policies cut the rest:
//  - BlockingPolicy: K depth (in0_block_w) and output blocks within the split, inside the L1 budget;
//  - SubblockPolicy: the output subblock within a block.
// Sharded tensors constrain the layout rather than change the rules: a sharded A fixes the family, grid and per-core
// sizes (width -> 1D in0-mcast, height -> 1D in1-mcast or Reuse for batched B, block -> 2D), a sharded output (with
// interleaved inputs) fixes the family (and with a shard spec, the grid and per-core sizes), and the policies choose
// what remains within the layout's BlockRules. Choosing among the candidates is the ranker's job (rule_ranker.hpp).
//
// The cutting is rules on K depth and block size, and for Reuse's slice height an estimate that adds a fixed cost per
// block and the last block's output write to the roofline (HeuristicBlocking::Tuned::reuse_*), switching from the
// slice that fills the grid at reuse_switch_margin.
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
    // A is read in place (a height-sharded A on 1D in1-mcast): A doesn't count toward the self-read K limit, and K
    // is split only into blocks of the full depth limit
    bool a_in_place = false;

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

// Issue #57884's block-size heuristics with one K block depth rule:
//  - 2D: the largest in0_block_w * out_block_h * out_block_w that fits L1; ties go to the block that moves the
//    fewest input tiles (and, where K blocks are costly, partials) per K block. Where K blocks are costly (packer L1
//    accumulation off or a block-float input), interleaved 2D may use K blocks up to Kt / max_costly_k_blocks deep;
//  - 1D: the full per-core extent along the multicast dimension, the other one shrunk only if needed, unless
//    that forces single-tile K steps; 1D in0-mcast splits a wide output block into subblock-wide blocks;
//  - Reuse: the deepest in0_block_w within the K depth rule that fits.
// A layout's preferred in0_block_w wins whenever some block fits with it.
class HeuristicBlocking final : public BlockingPolicy {
public:
    // Chosen: design limits, set by judgment rather than fitted
    struct Limits {
        // K block depth never goes below this (unless K itself is shallower): single-tile K steps pay a block
        // handshake per tile.
        uint32_t min_in0_block_w = 2;
    };
    // Tuned: fitted to benchmark data. Each records its basis and, where measured, the range over which the
    // choices come out the same; a value outside the range is untested, not wrong.
    struct Tuned {
        // K block depth, for every family: in0_block_w is at most this. Deeper K blocks stop paying for
        // themselves, and in 2D the block-size heuristic would otherwise trade output-block size (the only source
        // of data reuse) for K depth. Basis: 8 against 16 on the Wormhole OOB suite; range not measured.
        uint32_t max_in0_block_w = 8;
        // With packer L1 accumulation off or a block-float input, each 2D K block's fixed cost (packing, and
        // without accumulation reloading, the whole output block's partials) dominates, and K is split into at most
        // this many blocks (see max_in0_block_w): the block search may go that deep. Basis: 8 reproduces, on Wormhole's
        // 8-wide grid, the K depth a 67-case 2D probe found pays under those conditions; without it 13 gist and suite
        // cases (70B qkv_proj fwd T=1024, 1B down_proj fwd T=1024, ...) fall to 0.83-0.95 of legacy.
        uint32_t max_costly_k_blocks = 8;
        // ...but no deeper than this: with a very large K (Kt 512 and up) the K-block count stops mattering and the
        // deeper block only enlarges B's buffer. Basis: a sweep of 18 2D cases with block-float B and Kt 256-1024
        // (generated/matmul_oob/bigk): K 64 was never the fastest, K 32 within 2% of the best or better than 64 on
        // all; test_prefill_mm_interleaved_sharded wo (Kt 512) 0.91x of its hand config at K 64. No suite case goes
        // above 32.
        uint32_t max_costly_in0_block_w = 32;
        // K block depth is further limited so that the operand a core reads by itself (not by multicast) moves at
        // most this many tiles per K block: B's slice in 1D in0-mcast, A's in 1D in1-mcast, both in Reuse, none in
        // 2D. Small per-step reads keep the double-buffered DRAM stream ahead of math; wide per-core blocks get
        // shallower K blocks, but never below Limits::min_in0_block_w. Basis: 8 on Wormhole (8 against 4 on the OOB
        // suite); 12 on Blackhole, the smallest value giving t_matmul_53dd and 4e7d (3 tiles per K step) K 4 in a BH
        // probe, while 1D layouts reading 5 or more tiles per step keep K 2.
        uint32_t max_self_read_tiles_per_k_step = 8;
        // Every K block ends with a pack of the whole output block, which on an architecture that moves data fast
        // relative to compute doesn't hide behind the reads. When on, 2D blocks that fit with in0_block_w at least
        // Limits::min_in0_block_w win over larger blocks that only fit below it, and the 2D tie-break counts the
        // packed partials for every block, not only where K blocks are costly (see block_2d). Basis: off on
        // Wormhole (i29716_dit wants large 2D blocks at K 1, and large bf16 blocks over deeper K at equal work),
        // on for Blackhole (BH probe of g_4096; equal-work 2D ties go 5-8% faster deeper on BH bf16 shapes).
        bool k_depth_over_block_size = false;
        // Reuse's slice estimate (see HeuristicGenerator::reuse_blocking): the fixed cost of each block
        // (cycles), the rate a core writes its block's output at (bytes per cycle; the last block's write doesn't
        // overlap compute), and how much faster another slice must be estimated than the one that fills the grid.
        // Basis: Reuse slice sweeps of 49 batched cases on Wormhole and Blackhole (generated/matmul_oob/rslice;
        // every per_core_M timed): the fixed cost alone is ~10 us per block (2 x 4096x32x256: 1/2/4 blocks per core
        // 41/51/64 us). These values change no case for the worse on either machine; c 16k-32k, 2-3 B/cycle and a
        // margin of 1.1-1.25 behave the same.
        double reuse_block_cycles = 32000;
        double reuse_write_bytes_per_cycle = 3;
        double reuse_switch_margin = 1.25;
    };
    struct Params {
        Limits limits;
        Tuned tuned;
        // The values for an architecture
        static Params for_arch(tt::ARCH arch);
    };

    // The values for each matmul's architecture
    HeuristicBlocking() = default;
    // Fixed values, whatever the architecture
    explicit HeuristicBlocking(const Params& params) : params_(params) {}

    std::optional<Blocking> block(
        const MatmulDesc& matmul,
        const HardwareDesc& hw,
        Family family,
        const Split& split,
        const BlockRules& rules) const override;

private:
    std::optional<Params> params_;
};

// The largest-area subblock that fits DST and divides the block, two tiles or more on each side unless B's tiles
// are smaller than A's (see the .cpp for why)
class HeuristicSubblock final : public SubblockPolicy {
public:
    Blocking subblock(const MatmulDesc& matmul, Family family, Blocking blocking) const override;
};

// One candidate per family that can run the problem and fits L1, in the order Reuse, 2D, 1D in0, 1D in1. With a
// batched A on a layout that splits only M (1D in1, or 2D with blocks one tile wide), the fused candidate is followed
// by the same family looping over the batch, when one batch keeps as many cores busy: each core then gets one shorter
// block per batch, whose output writes overlap the next one's compute. A sharded layout has one candidate.
class HeuristicGenerator final : public Generator {
public:
    // The heuristic policies
    HeuristicGenerator();
    HeuristicGenerator(std::shared_ptr<const BlockingPolicy> blocking, std::shared_ptr<const SubblockPolicy> subblock);

    std::string_view name() const override { return "heuristic"; }
    std::vector<Candidate> generate(const MatmulDesc& matmul, const HardwareDesc& hw) const override;

    // The fused candidate of each family, without the batch-looping ones
    std::vector<Candidate> one_per_family(const MatmulDesc& matmul, const HardwareDesc& hw) const;

private:
    std::vector<Candidate> sharded_candidates(const MatmulDesc& matmul, const HardwareDesc& hw) const;
    std::optional<Blocking> reuse_blocking(const MatmulDesc& matmul, const HardwareDesc& hw) const;
    std::optional<Candidate> looped_over_batch(
        const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& fused) const;

    std::shared_ptr<const BlockingPolicy> blocking_;
    std::shared_ptr<const SubblockPolicy> subblock_;
};

}  // namespace ttnn::operations::matmul::auto_config
