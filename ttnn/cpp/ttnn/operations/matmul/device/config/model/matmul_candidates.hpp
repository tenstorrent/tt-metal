// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The candidate program configs the matmul config model ranks: a port of the candidate enumeration of the
// default-config selector proposed in #58143 (and the blocking heuristics, roofline and factory buffer sizes it is
// built on), restricted to interleaved A, B and output.
//
// Layouts are the ones the heuristic FactoryBlockingSource asks its blocking policy for (one grid per family, the
// device's), plus each multicast layout with the batch looped over instead of fused. Within a layout: the K depths
// (one divisor of Kt per power of two), up to 3 output blocks per depth that fit the L1 budget, and up to 6 subblocks
// per block. Every candidate fits the L1 budget and the DST limits the factories compute correctly with; the device
// op's own validation is left to the caller. The heuristic's own choice comes first.
//
// Everything here is a pure function of plain data (std library only), so it runs and is tested without a device.
// Sizes are in 32x32 tiles.

#include <cstdint>
#include <string>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/matmul_model_features.hpp"

namespace ttnn::operations::matmul::model {

// Where a tensor lives. Only interleaved tensors are enumerated (a sharded one makes the problem unsupported).
struct Placement {
    bool sharded = false;
    bool in_l1 = false;
};

// A matmul as the enumerator sees it. Dimensions in tiles, after transposes; batch_a / batch_b are the products of
// A's / B's leading dims.
struct MatmulDesc {
    std::uint32_t batch_a = 1, batch_b = 1;
    std::uint32_t rank_a = 2, rank_b = 2;
    std::uint32_t Mt = 0, Kt = 0, Nt = 0;  // Mt per batch
    DataType in0 = DataType::kBfloat16, in1 = DataType::kBfloat16, out = DataType::kBfloat16;
    std::uint32_t bias_tile_bytes = 0;  // tile size of a fused row bias; 0 without one
    std::uint32_t bias_rows = 0;        // the fused bias's height in tiles (1 for a row bias)
    bool transpose_a = false;           // A's tiles transposed (the factories add an in0_transposed buffer)
    Fidelity fidelity = Fidelity::kHiFi2;
    bool fp32_dest_acc = false;
    bool packer_l1_acc = true;
    bool dst_full_sync = false;
    Placement a, b, out_place;
};

// The device and the budget the candidates must fit. Blackhole only (the architecture the model was trained on).
struct HardwareDesc {
    std::uint32_t grid_x = 0, grid_y = 0;  // compute_with_storage_grid_size
    std::uint32_t l1_cb_budget = 0;        // per-core bytes available for circular buffers
    std::uint32_t dram_alignment = 64;     // bytes; tiles read from DRAM are padded to this
    // Nominal rates for the roofline the heuristic family choice uses, per core clock cycle
    std::uint32_t matmul_flops_per_cycle = 4096;
    std::uint32_t noc_bytes_per_cycle = 64;
    double dram_bytes_per_cycle = 512.0 / 1.35;  // 512 GB/s at 1.35 GHz

    static HardwareDesc blackhole(std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t l1_cb_budget);
};

// The per-core work of a config, in tiles. Reuse's output block is the whole per-core block.
struct Blocking {
    std::uint32_t per_core_M = 0, per_core_N = 0, in0_block_w = 0;
    std::uint32_t out_block_h = 0, out_block_w = 0;
    std::uint32_t out_subblock_h = 0, out_subblock_w = 0;
    bool operator==(const Blocking&) const = default;
};

struct Candidate {
    Family family = Family::k2D;
    Blocking blocking;
    std::uint32_t cores = 0;  // cores with work
    std::uint32_t grid_x = 0, grid_y = 0;
    bool fuse_batch = true;  // mcast families: batch folded into M, else looped over
};

// Why the enumerator has no candidates for this matmul (sharded tensors, mismatched batches, ...), or empty.
std::string unsupported_reason(const MatmulDesc& p);

// Per-core L1 bytes of the circular buffers the factory for `family` allocates with this blocking (the factories'
// own sizing rules for interleaved tensors).
std::uint32_t circular_buffer_bytes(const MatmulDesc& p, const HardwareDesc& hw, const Candidate& c);

// The candidates, heuristic choice first, deduplicated as program configs. Empty when unsupported_reason is not.
std::vector<Candidate> enumerate_candidates(const MatmulDesc& p, const HardwareDesc& hw);

// The candidates on the device grid followed by those on the grid one column narrower. A dimension that splits into
// awkward per-core tile counts on the full grid often splits well on one column less (#62076: N = 448 tiles over 11
// columns is 41 per core, a prime, which forces 1x41 output blocks), and the model scores both.
std::vector<Candidate> enumerate_candidates_with_narrower_grid(const MatmulDesc& p, const HardwareDesc& hw);

// A config of one of the enumerated families as a candidate on `hw`'s grid, to score a config the enumerator didn't
// emit (the default selection's). Reuse ignores out_block_* and fuse_batch.
Candidate make_candidate(
    const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch);

// The candidate as the model's input.
MatmulCandidate to_model_candidate(const Candidate& c);

}  // namespace ttnn::operations::matmul::model
