// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
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
// (grid, L1), so it can be exercised for any architecture without a device. It uses structural heuristics
// only, no measured constants:
//  - B not batched: the batch is fused into M and 2D mcast is used, unless a 1D layout keeps at least
//    ONE_D_CORE_ADVANTAGE times as many cores busy (small M or small N), in which case that 1D layout is used,
//    or 1D in0-mcast keeps as many cores busy with less input per core (per_core_M + per_core_N);
//  - batched B: Reuse, unless the multicast layout looping over the batch (chosen as above) keeps
//    ONE_D_CORE_ADVANTAGE times as many cores busy (e.g. large N, where Reuse's per_core_N = N leaves few cores);
//  - block sizes follow the #57884 heuristics within the L1 budget, with one K block depth rule
//    (MAX_IN0_BLOCK_W, MAX_SELF_READ_TILES_PER_K_STEP).
// Problems it does not handle yet return nullopt, and the caller falls back to the legacy selection.
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
    uint32_t l1_cb_budget = 0;     // per-core bytes available for circular buffers
    uint32_t dram_alignment = 32;  // bytes; tiles read from DRAM are padded to this

    static HardwareDesc for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget);
};

// The matmul as the selector sees it. Dimensions are in 32x32 tiles, after transposes.
struct Problem {
    uint32_t batch_a = 1;  // product of A's leading dims
    uint32_t batch_b = 1;  // product of B's leading dims
    uint32_t Mt = 0;       // per batch
    uint32_t Kt = 0;
    uint32_t Nt = 0;
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
    uint32_t cores = 0;  // cores with work
};

// Per-core circular-buffer bytes the factory for `family` allocates with this blocking (interleaved operands
// and output, 32x32 tiles).
uint32_t circular_buffer_bytes(const Problem& problem, const HardwareDesc& hw, Family family, const Blocking& b);

// The blocked candidate of each family that can run the problem and fits L1, in family order.
std::vector<Candidate> candidates(const Problem& problem, const HardwareDesc& hw);

// The candidate the heuristics choose, or nullopt if none fits.
std::optional<Candidate> choose_candidate(const Problem& problem, const HardwareDesc& hw);

// The program config for the chosen candidate, or nullopt if the problem is unsupported or nothing fits.
std::optional<MatmulProgramConfig> select_program_config(const Problem& problem, const HardwareDesc& hw);

// Builds the Problem and HardwareDesc from matmul's inputs and selects a config. Returns nullopt for inputs
// the new selector does not handle yet (sharded tensors, non-32x32 tiles, global CBs, sub-devices, ...).
std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes);

}  // namespace ttnn::operations::matmul::auto_config
