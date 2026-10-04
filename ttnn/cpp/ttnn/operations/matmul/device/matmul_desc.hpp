// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/matmul_validation.hpp"

// A matmul's inputs described as plain data: shapes in tiles, tiles, formats, compute settings, and where each
// tensor lives. Built once from the matmul's specs (describe_matmul); the program config selection's rules read
// only this.
namespace ttnn::operations::matmul {

enum class MemoryLayout { Interleaved, HeightSharded, WidthSharded, BlockSharded, NdSharded };

// Where a tensor lives. Shard dimensions are in the tensor's tiles; a sharded output may come without a shard
// spec, in which case the program config decides its shard grid.
struct Placement {
    MemoryLayout layout = MemoryLayout::Interleaved;
    bool in_l1 = false;
    bool has_shard_spec = false;
    bool shard_whole_tiles = true;                     // the shard shape is a whole number of tiles
    CoreRange shard_grid = CoreRange({0, 0}, {0, 0});  // bounding box of the shard grid
    uint32_t shard_cores = 0;
    uint32_t shard_h = 0;
    uint32_t shard_w = 0;
    bool col_major = false;

    bool sharded() const { return layout != MemoryLayout::Interleaved; }
    bool l1_sharded() const { return sharded() && in_l1; }
    bool dram_sharded() const { return sharded() && !in_l1; }
};

// Dimensions are in tiles, after transposes: M in A's tiles (in0_tile_h rows), N in B's (in1_tile_w columns),
// K in 32-wide tiles.
struct MatmulDesc {
    uint32_t batch_a = 1;  // product of A's leading dims
    uint32_t batch_b = 1;  // product of B's leading dims
    uint32_t rank_a = 2;
    uint32_t rank_b = 2;
    uint32_t Mt = 0;  // per batch
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
    uint32_t bias_rows = 0;        // the bias's height in A's tiles: 1 for a row bias, Mt for a full [M, N] block
    bool transpose_a = false;
    bool in0_tile_transposed = false;  // A's tiles as the matmul reads them are transposed
    bool untilize_out = false;
    MathFidelity math_fidelity = MathFidelity::HiFi2;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = true;
    bool dst_full_sync_en = false;
    std::optional<unary::UnaryWithParam> activation;  // only if the kernels can fuse it
    Placement a;
    Placement b;
    Placement out;
    bool b_shard_matches_a = false;  // B sharded with A's layout, grid and orientation
    bool global_cb = false;          // B streamed through a global circular buffer
};

// Describes the matmul. Fails, with the reason in `why`, only for inputs no matmul runs: a rank below 2, a K tile
// side other than 32, or no compute kernel config.
std::optional<MatmulDesc> describe_matmul(const ttnn::prim::MatmulSpecs& specs, std::string& why);

}  // namespace ttnn::operations::matmul
