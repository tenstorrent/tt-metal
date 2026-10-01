// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_buffers.hpp"
#include "ttnn/operations/matmul/device/matmul_desc.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

// Default program config selection for matmul (issue #57884), enabled by ttnn.CONFIG.matmul_auto_config_v2.
// It runs only when no program config is given; a measured registry entry (#54943), when present, takes
// precedence because it sets the program config before this is reached.
//
// The selector is a pure function of a MatmulDesc (shapes, formats, compute settings) and a HardwareDesc
// (grid, L1), so it can be exercised for any architecture without a device. It uses structural heuristics
// only, no measured constants:
//  - B not batched: the batch is fused into M and 2D mcast is used, unless a 1D layout keeps at least
//    ONE_D_CORE_ADVANTAGE times as many cores busy (small M or small N), in which case that 1D layout is used,
//    or 1D in0-mcast keeps as many cores busy with less input per core (per_core_M + per_core_N);
//  - batched B: Reuse, unless the multicast layout looping over the batch (chosen as above) keeps
//    ONE_D_CORE_ADVANTAGE times as many cores busy (e.g. large N, where Reuse's per_core_N = N leaves few cores)
//    or Reuse would read ONE_D_CORE_ADVANTAGE times as much input (splitting batch matrices re-reads B);
//  - block sizes follow the #57884 heuristics within the L1 budget, with one K block depth rule
//    (MAX_IN0_BLOCK_W, LARGE_BLOCK_TILES, MAX_SELF_READ_TILES_PER_K_STEP); interleaved 2D then goes no
//    shallower than the legacy selection's K depth (K split into as many blocks as the grid is wide). Precision is left
//    to the compute kernel config: the blocking doesn't change with the output format or packer L1 accumulation. 1D
//    blocks keep the full per-core extent along the multicast dimension unless that forces single-tile K steps,
//    and 1D in0-mcast splits a wide output block into subblock-wide blocks;
//  - subblocks are the largest that fit DST, two tiles or more on each side unless B's tiles are smaller
//    than A's;
//  - K depth is then refined by cost estimators (Estimator): the chosen candidate competes with its legal
//    K-depth neighbours, and the lowest estimate wins, the chosen candidate on ties. The built-in roofline
//    estimate doesn't depend on K depth, so on its own it keeps the heuristics' choice; estimators that do
//    (calibrated, measured, simulated) plug in here.
// Sharded tensors constrain the choice rather than change the rules: a sharded A fixes the family, grid and
// per-core sizes (width -> 1D in0-mcast, height -> 1D in1-mcast or Reuse for batched B, block -> 2D), a sharded
// output (with interleaved inputs) fixes the family (and with a shard spec, the grid and per-core sizes), and
// what remains free (in0_block_w, output blocks, subblocks) is chosen as above within the layout's constraints.
// A width- or block-sharded A's K blocks are whole shard columns when they fit, multicast in place.
// Problems it has no config for return nullopt with the reason; matmul reports them as an error (they are inputs
// the factories can't run).
namespace ttnn::operations::matmul::auto_config {

// Switching away from the default layout needs at least this many times as many cores busy: 1D over 2D (1D
// multicasts a whole operand to every core), and for batched B a batch-looping multicast layout over Reuse.
// On the Wormhole sweep anything from 1.25 to 2 performs about the same.
constexpr double ONE_D_CORE_ADVANTAGE = 1.5;

// K block depth, for every family: in0_block_w is at most this. Deeper K blocks stop paying for themselves,
// and in 2D the block-size heuristic would otherwise trade output-block size (the only source of data reuse)
// for K depth. The mcast families also keep at least two K blocks, since with a single block they
// single-buffer the inputs.
constexpr uint32_t MAX_IN0_BLOCK_W = 8;

// 2D output blocks of more than this many tiles may use K blocks up to 2 * MAX_IN0_BLOCK_W deep. Every K block
// ends with a pack of the whole output block (L1 accumulation of the partials), which sits on the compute
// path; a large block is compute bound, so deeper K blocks amortize that pack. Smaller blocks wait on data,
// where the pack is hidden and a deeper K block only lengthens the pipeline fill. On the Wormhole 2D sweeps,
// with the output block fixed, K depth 16 beat 8 on 6 of 10 larger blocks and lost on none, while on smaller
// blocks 8 won 51 to 19. The threshold is where the sweeps turn, not derived.
constexpr uint32_t LARGE_BLOCK_TILES = 64;

// K block depth is further limited so that the operand a core reads by itself (not by multicast) moves at
// most this many tiles per K step: B's slice in 1D in0-mcast, A's in 1D in1-mcast, both in Reuse, none in 2D.
// Small per-step reads keep the double-buffered DRAM stream ahead of math; wide per-core blocks get
// shallower K blocks, but never below MIN_IN0_BLOCK_W (single-tile K steps pay a block handshake per tile).
constexpr uint32_t MAX_SELF_READ_TILES_PER_K_STEP = 8;
constexpr uint32_t MIN_IN0_BLOCK_W = 2;

// Hardware facts the selector depends on. Tests can describe other architectures directly.
struct HardwareDesc {
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    CoreCoord grid;                // worker grid available to this matmul
    CoreCoord origin{0, 0};        // its first core (a sub-device's worker grid need not start at (0, 0))
    bool pinned_origin = false;    // the configs must name the grid's cores (a sub-device's) explicitly
    uint32_t l1_cb_budget = 0;     // per-core bytes available for circular buffers
    uint32_t dram_alignment = 32;  // bytes; tiles read from DRAM are padded to this
    // Nominal rates for the roofline estimate, per core clock cycle (tech_reports/GEMM_FLOPS and
    // tech_reports/FlashAttention): matrix engine FLOPs at LoFi (8x16 x 16x16 per cycle), divided by the math
    // fidelity; one NoC link into a core; the chip's DRAM bandwidth.
    uint32_t matmul_flops_per_cycle = 4096;
    uint32_t noc_bytes_per_cycle = 32;
    double dram_bytes_per_cycle = 288.0;  // 288 GB/s at 1 GHz

    static HardwareDesc for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget);
};

enum class Family { Mcast2D, Mcast1DIn0, Mcast1DIn1, Reuse };

struct Candidate {
    Family family;
    Blocking blocking;
    uint32_t cores = 0;                     // cores with work
    CoreCoord grid;                         // compute_with_storage_grid_size of the config
    std::optional<CoreRange> worker_cores;  // allowed_worker_cores (sharded layouts: the shard grid)
    bool transpose_mcast = false;           // 2D on a column-major block-sharded A
    bool fuse_batch = true;                 // mcast families: the batch folded into M, else looped over
};

// Per-core L1 bytes the factory for `family` needs with this blocking: the buffers it allocates (sized by the
// factory's own functions in matmul_buffers.hpp), plus a sharded output's shard, which is not allocated yet.
// `fuse_batch`: the batch folded into M (per_core_M and out_block_h count rows of all batches), else looped over.
uint32_t circular_buffer_bytes(
    const MatmulDesc& matmul, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch);

// Per-core roofline terms (cycles) of a blocked candidate, from the rates in HardwareDesc. They depend on the
// output blocks but not on in0_block_w.
struct RooflineTerms {
    double compute = 0;  // the busiest core's tile products
    double noc = 0;      // input bytes the busiest core receives
    double dram = 0;     // bytes read from and written to DRAM, chip-wide
    double cycles() const { return std::max({compute, noc, dram}); }
};
RooflineTerms roofline(
    const MatmulDesc& matmul, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch);

// The program config of a candidate.
MatmulProgramConfig to_program_config(const MatmulDesc& matmul, const Candidate& candidate);

// Whether the factories accept `config` for `matmul` on `hw`: empty if so, else the first rule it breaks
// (K and block divisibility, DST capacity, the grid, what each factory's layout requires, and L1). Covers
// interleaved operands and outputs; a sharded layout's own rules are not checked.
std::string check(const MatmulDesc& matmul, const HardwareDesc& hw, const MatmulProgramConfig& config);

// A cost estimate of a candidate, from one Estimator.
struct Estimate {
    double cycles = 0;      // estimated device time
    double confidence = 0;  // how far to trust it for this problem; the most confident estimate is used
    std::string_view source;
};

// Estimates a candidate's device time, or returns nullopt when it has no estimate for this problem. Estimators
// rank candidates; they never create them, so they can't make an illegal choice.
class Estimator {
public:
    virtual ~Estimator() = default;
    virtual std::string_view name() const = 0;
    virtual std::optional<Estimate> estimate(
        const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate) const = 0;
};

// The roofline estimate (the largest RooflineTerms term), confidence 0: every estimator that has an answer
// outranks it.
class RooflineEstimator final : public Estimator {
public:
    std::string_view name() const override { return "roofline"; }
    std::optional<Estimate> estimate(
        const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate) const override;
};

// The estimators choose_candidate uses: the roofline.
std::span<const Estimator* const> default_estimators();

// The blocked candidate of each family that can run the problem and fits L1, in family order.
std::vector<Candidate> candidates(const MatmulDesc& matmul, const HardwareDesc& hw);

// A candidate's K-depth neighbours: the same candidate at the next deeper and the next shallower in0_block_w
// dividing K that pass check(). Interleaved problems only (a sharded layout constrains K depth); empty
// otherwise.
std::vector<Candidate> k_depth_neighbours(const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate);

// The best of `options` by estimate: per option the most confident estimate (the earlier estimator on ties),
// then the lowest cycles (the earlier option on ties). Options without an estimate lose to those with one.
const Candidate& best_by_estimate(
    const MatmulDesc& matmul,
    const HardwareDesc& hw,
    std::span<const Candidate> options,
    std::span<const Estimator* const> estimators);

// The candidate the heuristics choose, K depth refined by the estimators; nullopt if none fits.
std::optional<Candidate> choose_candidate(const MatmulDesc& matmul, const HardwareDesc& hw);
std::optional<Candidate> choose_candidate(
    const MatmulDesc& matmul, const HardwareDesc& hw, std::span<const Estimator* const> estimators);

// The program config for the chosen candidate, or nullopt if the problem is unsupported or nothing fits.
std::optional<MatmulProgramConfig> select_program_config(const MatmulDesc& matmul, const HardwareDesc& hw);

// Builds the MatmulDesc and HardwareDesc from matmul's inputs and selects a config. Returns nullopt for inputs
// the new selector does not handle, with the reason in `unsupported` when given.
std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported = nullptr);

}  // namespace ttnn::operations::matmul::auto_config
