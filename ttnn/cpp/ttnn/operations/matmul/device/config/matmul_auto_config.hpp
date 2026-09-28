// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

// Default program config selection for matmul (issue #57884), enabled by ttnn.CONFIG.matmul_auto_config_v2.
// It runs only when no program config is given; a measured registry entry (#54943), when present, takes
// precedence because it sets the program config before this is reached.
//
// The selector is a pure function of a Problem (shapes, formats, compute settings) and a HardwareDesc
// (grid, L1, nominal rates), so it can be exercised for any architecture without a device. It has no tuned
// thresholds: the rules are exact constraints (L1, DST, what each factory accepts) plus one physical proxy
// per decision:
//  - blocking (#57884 heuristics): 2D maximizes in0_block_w * out_block_h * out_block_w within L1 (the work per
//    K step, over which every per-step cost is amortized), ties going to the deeper K block; 1D keeps the full
//    per-core extent along the multicast dimension (out_block_w = per_core_N for in0-mcast) and the deepest K
//    block that fits; Reuse the tallest slice of a batch matrix that gives every core a block. The mcast
//    families keep at least two K blocks (with one they single-buffer the inputs);
//  - family (2D, 1D in0/in1-mcast, Reuse): the smallest per-core roofline estimate, the larger of compute
//    (matrix engine rate at the math fidelity), what the busiest core receives over the NoC, and the chip's
//    DRAM traffic, from the blocked candidates;
//  - subblocks are the largest that fit DST, two tiles or more on each side unless B's tiles are smaller
//    than A's.
// Precision is left to the compute kernel config: the blocking doesn't change with the output format or
// packer L1 accumulation.
// Sharded tensors constrain the choice rather than change the rules: a sharded A fixes the family, grid and
// per-core sizes (width -> 1D in0-mcast, height -> 1D in1-mcast or Reuse for batched B, block -> 2D), a sharded
// output (with interleaved inputs) fixes the family (and with a shard spec, the grid and per-core sizes), and
// what remains free (in0_block_w, output blocks, subblocks) is chosen as above within the layout's constraints.
// A width- or block-sharded A's K blocks are whole shard columns when they fit, multicast in place.
// Problems it does not handle yet return nullopt, and the caller falls back to the legacy selection.
namespace ttnn::operations::matmul::auto_config {

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

// Choices made from measurements rather than derived from the hardware or the factories, each with what it rests
// on. The selector's rules read them from here only, so a per-architecture table or a calibrated estimator can
// replace them without touching the rules.
struct EmpiricalDefaults {
    // K block depth cap. The best in0_block_w balances a per-K-block cost (packing the block's partials) against
    // the exposed first K block (its input can't overlap compute), which puts it near sqrt(Kt * h*w/(h+w)):
    // about 8-32 tiles for common shapes. The cost is flat near the balance point (a factor of 2 either way
    // costs only a few percent), so one cap in the middle is close to the best for most shapes. It also keeps
    // the double-buffered input CBs of a typical block to about a third of L1, leaving room for L1-resident
    // tensors. On the Wormhole and Blackhole fast suites it beat 32 and no cap on every suite. Data-bound 1D
    // decode shapes want much shallower K blocks, set by per-block synchronisation rather than this balance,
    // and are not covered.
    uint32_t max_in0_block_w = 16;
    // Kept free below the L1 budget, for allocator alignment and factory-side buffers the CB model doesn't count
    uint32_t l1_headroom_bytes = 16 * 1024;
    // Prefer subblocks at least two tiles on each side (2x4 over 1x8) unless B's tiles are smaller than A's: per
    // K step an h x w subblock unpacks h tiles of A and h * w of B, and the single-row path's per-tile overhead
    // cost up to 10% on large bf16 matmuls on Wormhole
    bool prefer_two_wide_subblocks = true;

    static EmpiricalDefaults for_arch(tt::ARCH arch);
};

enum class Layout { Interleaved, HeightSharded, WidthSharded, BlockSharded };

// Where a tensor lives. Shard dimensions are in tiles; a sharded output may come without a shard spec, in
// which case the program config decides its shard grid.
struct Placement {
    Layout layout = Layout::Interleaved;
    bool in_l1 = false;
    bool has_shard_spec = false;
    CoreRange shard_grid = CoreRange({0, 0}, {0, 0});  // bounding box of the shard grid
    uint32_t shard_cores = 0;
    uint32_t shard_h = 0;
    uint32_t shard_w = 0;
    bool col_major = false;

    bool sharded() const { return layout != Layout::Interleaved; }
};

// The matmul as the selector sees it. Dimensions are in tiles, after transposes: M in A's tiles (in0_tile_h
// rows), N in B's (in1_tile_w columns), K in 32-wide tiles.
struct Problem {
    uint32_t batch_a = 1;  // product of A's leading dims
    uint32_t batch_b = 1;  // product of B's leading dims
    uint32_t Mt = 0;       // per batch
    uint32_t Kt = 0;
    uint32_t Nt = 0;
    uint32_t in0_tile_h = 32;  // A's tiles are in0_tile_h x 32, B's 32 x in1_tile_w
    uint32_t in1_tile_w = 32;
    uint32_t out_tile_h = 32;  // the output tile (in0_tile_h rows; possibly wider than in1_tile_w)
    uint32_t out_tile_w = 32;
    tt::DataFormat in0_format = tt::DataFormat::Float16_b;
    tt::DataFormat in1_format = tt::DataFormat::Float16_b;
    tt::DataFormat out_format = tt::DataFormat::Float16_b;
    uint32_t bias_tile_bytes = 0;  // unaligned tile size of a fused row bias; 0 without bias
    bool transpose_a = false;
    MathFidelity math_fidelity = MathFidelity::HiFi2;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = true;
    bool dst_full_sync_en = false;
    std::optional<unary::UnaryWithParam> activation;
    Placement a;
    Placement b;
    Placement out;
    bool b_shard_matches_a = false;  // B sharded with A's layout, grid and orientation (Reuse only)
    bool no_mcast_1d = false;        // the 1D factories can't run it (a global CB without a gather config)
};

enum class Family { Mcast2D, Mcast1DIn0, Mcast1DIn1, Reuse };

struct Blocking {
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    uint32_t in0_block_w = 0;
    uint32_t out_block_h = 0;
    uint32_t out_block_w = 0;
    uint32_t out_subblock_h = 0;
    uint32_t out_subblock_w = 0;
    bool fuse_batch = true;  // mcast families: batch folded into M, else looped over per batch
};

struct Candidate {
    Family family;
    Blocking blocking;
    uint32_t cores = 0;                     // cores with work
    CoreCoord grid;                         // compute_with_storage_grid_size of the config
    std::optional<CoreRange> worker_cores;  // allowed_worker_cores (sharded layouts: the shard grid)
    bool transpose_mcast = false;           // 2D on a column-major block-sharded A
};

// Per-core L1 bytes the factory for `family` needs with this blocking (32x32 tiles): its circular buffers,
// less those backed by a sharded tensor, plus a sharded output's shard, which is not allocated yet.
uint32_t circular_buffer_bytes(const Problem& problem, const HardwareDesc& hw, Family family, const Blocking& b);

// Per-core roofline terms (cycles) of a blocked candidate, from the rates in HardwareDesc. They depend on the
// output blocks but not on in0_block_w.
struct RooflineTerms {
    double compute = 0;  // the busiest core's tile products
    double noc = 0;      // input bytes the busiest core receives
    double dram = 0;     // input bytes read from DRAM, chip-wide
    bool data_bound() const { return std::max(noc, dram) > compute; }
    double cycles() const { return std::max({compute, noc, dram}); }
};
RooflineTerms roofline(const Problem& problem, const HardwareDesc& hw, Family family, const Blocking& b);

// The roofline estimate: the largest term. The family choice takes the smallest.
double estimated_cycles(const Problem& problem, const HardwareDesc& hw, Family family, const Blocking& b);

// The blocked candidate of each family that can run the problem and fits L1, in family order.
std::vector<Candidate> candidates(
    const Problem& problem, const HardwareDesc& hw, const EmpiricalDefaults& defaults = EmpiricalDefaults{});

// The candidate the heuristics choose, or nullopt if none fits.
std::optional<Candidate> choose_candidate(
    const Problem& problem, const HardwareDesc& hw, const EmpiricalDefaults& defaults = EmpiricalDefaults{});

// The program config for the chosen candidate, or nullopt if the problem is unsupported or nothing fits.
std::optional<MatmulProgramConfig> select_program_config(
    const Problem& problem, const HardwareDesc& hw, const EmpiricalDefaults& defaults = EmpiricalDefaults{});

// Builds the Problem and HardwareDesc from matmul's inputs and selects a config. Returns nullopt for inputs
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
