// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "device/kernels/moe_ring_common.h"

namespace ttnn::experimental {

namespace {

// Stays in sync with moe_ring_common.h. (Block heights depend on the per-shape transaction size:
// ::moe_ring::block_tiles_h / half_block_tiles_h.)
constexpr uint32_t BLOCK_TILES_W = ::moe_ring::W0_W1_BLOCK_TILES_W;
constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH;

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

ttnn::Tensor reshape_to(const ttnn::Tensor& t, std::vector<int32_t> shape) {
    return ttnn::reshape(t, ttsl::Span<const int32_t>(shape.data(), shape.size()));
}

ttnn::Tensor permute_to(const ttnn::Tensor& t, std::initializer_list<int64_t> dims) {
    return ttnn::permute(t, ttsl::SmallVector<int64_t>(dims));
}

ttnn::Tensor slice_basic(
    const ttnn::Tensor& t, const ttsl::SmallVector<int32_t>& begins, const ttsl::SmallVector<int32_t>& ends) {
    ttsl::SmallVector<int32_t> steps(begins.size(), 1);
    return ttnn::slice(t, begins, ends, steps);
}

ttnn::Tensor zeros_like_dtype(std::initializer_list<uint32_t> shape, const ttnn::Tensor& reference) {
    return ttnn::zeros(
        ttnn::Shape(shape), reference.dtype(), reference.layout(), *reference.device(), reference.memory_config());
}

// `torch.stack` equivalent built from unsqueeze + concat.
ttnn::Tensor stack_along(const std::vector<ttnn::Tensor>& tensors, int dim) {
    std::vector<ttnn::Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(ttnn::unsqueeze(t, dim));
    }
    return ttnn::concat(unsqueezed, dim);
}

// Lay a TP-split shared-expert weight out so each ring core's real TpNt slice
// sits at the FRONT of that core's full-Nt shard, zero-filling the rest. `axis`
// is the intermediate (Nt) dim: last dim for W0/W1, dim -2 for W2. `full_map[c]`
// is core c's tile count under the full-Nt shard (sum = Nt); `tp_map[c]` is core
// c's count under the TpNt shard (sum = TpNt). The real tiles are consumed in
// order, so applying the SAME (full_map, tp_map) pair to W0/W1 (axis=-1) and W2
// (axis=-2) keeps each real intermediate column paired with its W2 row — i.e. a
// correct partial contraction once the kernel walks only the per-core prefixes.
ttnn::Tensor front_pack_per_core(
    const ttnn::Tensor& real, int axis, const std::vector<uint32_t>& full_map, const std::vector<uint32_t>& tp_map) {
    const auto& shape = real.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    const int ax = axis < 0 ? rank + axis : axis;
    const uint32_t num_cores = static_cast<uint32_t>(full_map.size());

    ttsl::SmallVector<int32_t> begins(rank, 0);
    ttsl::SmallVector<int32_t> ends(rank, 0);
    ttsl::SmallVector<uint32_t> zshape(rank, 0);

    std::vector<ttnn::Tensor> pieces;
    pieces.reserve(2 * num_cores);
    uint32_t cursor = 0;  // real tiles consumed so far (along `ax`)
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t r = tp_map[c];
        const uint32_t s = full_map[c];
        TT_FATAL(r <= s, "TpNt shard ({}) exceeds full-Nt shard ({}) at core {}", r, s, c);
        if (r > 0) {
            for (int d = 0; d < rank; ++d) {
                if (d == ax) {
                    begins[d] = cursor * TILE_SIZE;
                    ends[d] = (cursor + r) * TILE_SIZE;
                } else {
                    ends[d] = shape[d];
                    begins[d] = 0;
                }
            }
            pieces.push_back(slice_basic(real, begins, ends));
            cursor += r;
        }
        if (s > r) {
            for (int d = 0; d < rank; ++d) {
                if (d == ax) {
                    zshape[d] = (s - r) * TILE_SIZE;
                } else {
                    zshape[d] = shape[d];
                }
            }
            pieces.push_back(
                ttnn::zeros(ttnn::Shape(zshape), real.dtype(), real.layout(), *real.device(), real.memory_config()));
        }
    }
    auto out = ttnn::concat(pieces, ax);
    for (auto& p : pieces) {
        p.deallocate(/*force=*/true);
    }
    return out;
}

// W2 packer without the trailing N-pad — used by the bias-aware path so the
// bias tile row can be concatenated before the alignment pad is applied.
ttnn::Tensor prepare_w2_no_n_pad(
    const ttnn::Tensor& tt_w2,
    uint32_t L,
    uint32_t E,
    uint32_t N,
    uint32_t K,
    const std::vector<std::pair<uint32_t, uint32_t>>& w2_shard_map,
    const std::vector<uint32_t>& w0_w1_shard_map) {
    const uint32_t Kt = K / TILE_SIZE;
    const uint32_t Nt = N / TILE_SIZE;
    const uint32_t num_cores = static_cast<uint32_t>(w2_shard_map.size());
    const uint32_t first_pair_sum = w2_shard_map[0].first + w2_shard_map[0].second;
    const uint32_t w2_groups_per_core = ceil_div(Kt, num_cores * first_pair_sum);

    std::vector<ttnn::Tensor> each_shard;
    each_shard.reserve(3 * num_cores);
    uint32_t start_col = 0;
    const uint32_t full_block_width = (w2_groups_per_core - 1) * 4 * TILE_SIZE;
    for (const auto& [last_group_tiles, last_group_pad_tiles] : w2_shard_map) {
        if (full_block_width > 0) {
            each_shard.push_back(slice_basic(
                tt_w2,
                {0, 0, 0, static_cast<int32_t>(start_col)},
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(N),
                 static_cast<int32_t>(start_col + full_block_width)}));
            start_col += full_block_width;
        }
        const uint32_t last_group_width = last_group_tiles * TILE_SIZE;
        each_shard.push_back(slice_basic(
            tt_w2,
            {0, 0, 0, static_cast<int32_t>(start_col)},
            {static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(N),
             static_cast<int32_t>(start_col + last_group_width)}));
        start_col += last_group_width;
        if (last_group_pad_tiles > 0) {
            each_shard.push_back(zeros_like_dtype({L, E, N, last_group_pad_tiles * TILE_SIZE}, tt_w2));
        }
    }

    auto reordered = ttnn::concat(each_shard, -1);
    for (auto& s : each_shard) {
        s.deallocate(/*force=*/true);
    }
    each_shard.clear();
    auto grouped_per_bank = reshape_to(
        reordered,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(N),
         static_cast<int32_t>(num_cores),
         static_cast<int32_t>(w2_groups_per_core),
         static_cast<int32_t>(4 * TILE_SIZE)});
    grouped_per_bank = permute_to(grouped_per_bank, {3, 0, 1, 4, 2, 5});
    reordered.deallocate(/*force=*/true);

    auto n_grouped = reshape_to(
        grouped_per_bank,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(w2_groups_per_core),
         static_cast<int32_t>(Nt),
         static_cast<int32_t>(TILE_SIZE),
         static_cast<int32_t>(4 * TILE_SIZE)});

    // Per-core ring rotation of the Nt chunks.
    std::vector<uint32_t> chunk_start_positions = {0};
    chunk_start_positions.reserve(w0_w1_shard_map.size() + 1);
    for (uint32_t s : w0_w1_shard_map) {
        chunk_start_positions.push_back(chunk_start_positions.back() + s);
    }
    std::vector<uint32_t> base_order(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        base_order[i] = num_cores - 1 - i;  // reversed(range(num_cores))
    }
    // `.roll(1)`: last element goes to front.
    std::rotate(base_order.begin(), base_order.begin() + (num_cores - 1), base_order.end());

    std::vector<ttnn::Tensor> per_core_shards;
    per_core_shards.reserve(num_cores);
    std::vector<uint32_t> current_order = base_order;
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        auto core_slab = slice_basic(
            n_grouped,
            {static_cast<int32_t>(core_id), 0, 0, 0, 0, 0, 0},
            {static_cast<int32_t>(core_id + 1),
             static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(w2_groups_per_core),
             static_cast<int32_t>(Nt),
             static_cast<int32_t>(TILE_SIZE),
             static_cast<int32_t>(4 * TILE_SIZE)});
        std::vector<ttnn::Tensor> chunks;
        chunks.reserve(current_order.size());
        for (uint32_t chunk_id : current_order) {
            const uint32_t start_pos = chunk_start_positions[chunk_id];
            const uint32_t end_pos = chunk_start_positions[chunk_id + 1];
            chunks.push_back(slice_basic(
                core_slab,
                {0, 0, 0, 0, static_cast<int32_t>(start_pos), 0, 0},
                {1,
                 static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(w2_groups_per_core),
                 static_cast<int32_t>(end_pos),
                 static_cast<int32_t>(TILE_SIZE),
                 static_cast<int32_t>(4 * TILE_SIZE)}));
        }
        per_core_shards.push_back(ttnn::concat(chunks, 4));
        for (auto& c : chunks) {
            c.deallocate(/*force=*/true);
        }
        core_slab.deallocate(/*force=*/true);
        // Rotate current_order by 1.
        std::rotate(current_order.begin(), current_order.begin() + (current_order.size() - 1), current_order.end());
    }
    n_grouped.deallocate(/*force=*/false);
    grouped_per_bank.deallocate(/*force=*/true);

    auto stacked = ttnn::concat(per_core_shards, 0);
    for (auto& s : per_core_shards) {
        s.deallocate(/*force=*/true);
    }
    per_core_shards.clear();
    auto result = reshape_to(
        stacked,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(w2_groups_per_core),
         static_cast<int32_t>(Nt * TILE_SIZE),
         static_cast<int32_t>(4 * TILE_SIZE)});
    stacked.deallocate(/*force=*/false);
    return result;
}

// Lay the ring-rotated W2 groups (num_cores, L, E, groups, k_tiles * TILE, 4 * TILE) out in DRAM blocks for a
// transaction size (Python mirror: moe_compute_utils.py::_w2_blocks_from_groups). Full a2a iterations stay 4 wide
// with K padded to whole block_tiles_h blocks; a half-width last iteration keeps its 2 valid columns with K padded
// to whole half_block_tiles_h blocks, two consecutive K tile rows side by side per stored row. Without a half
// iteration the grouped shape is kept (the old layout for 14-tile transactions); with one the result is
// (num_cores, L, E, blocks, block_rows, 4 * TILE).
ttnn::Tensor w2_blocks_from_groups(ttnn::Tensor& grouped, uint32_t k_tiles, uint32_t Ht, uint32_t tiles_per_txn) {
    const auto& shape = grouped.logical_shape();
    const uint32_t num_cores = shape[0];
    const uint32_t L = shape[1];
    const uint32_t E = shape[2];
    const uint32_t groups = shape[3];
    const uint32_t rows = shape[4];
    const uint32_t block_h = ::moe_ring::block_tiles_h(tiles_per_txn);
    const uint32_t half_block_h = ::moe_ring::half_block_tiles_h(tiles_per_txn);
    const uint32_t full_rows = ceil_div(k_tiles, block_h) * block_h * TILE_SIZE;
    const bool half = ::moe_ring::w2_last_a2a_iter_half(Ht, num_cores, tiles_per_txn);
    const uint32_t full_groups = groups - (half ? 1 : 0);

    ttnn::Tensor full = grouped;
    if (half && full_groups > 0) {
        full = slice_basic(
            grouped,
            {0, 0, 0, 0, 0, 0},
            {static_cast<int32_t>(num_cores),
             static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(full_groups),
             static_cast<int32_t>(rows),
             static_cast<int32_t>(4 * TILE_SIZE)});
    }
    if (full_groups > 0 && full_rows > rows) {
        auto pad = zeros_like_dtype({num_cores, L, E, full_groups, full_rows - rows, 4 * TILE_SIZE}, grouped);
        auto padded = ttnn::concat({full, pad}, 4);
        pad.deallocate(/*force=*/true);
        if (half) {
            full.deallocate(/*force=*/true);
        }
        full = padded;
    }
    if (!half) {
        return full;
    }

    const uint32_t half_rows = ceil_div(k_tiles, half_block_h) * half_block_h * TILE_SIZE;
    auto last = slice_basic(
        grouped,
        {0, 0, 0, static_cast<int32_t>(groups - 1), 0, 0},
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(groups),
         static_cast<int32_t>(rows),
         static_cast<int32_t>(::moe_ring::W2_HALF_A2A_ITER_TILES_W * TILE_SIZE)});
    if (half_rows > rows) {
        auto pad = zeros_like_dtype(
            {num_cores, L, E, 1, half_rows - rows, ::moe_ring::W2_HALF_A2A_ITER_TILES_W * TILE_SIZE}, grouped);
        auto padded = ttnn::concat({last, pad}, 4);
        pad.deallocate(/*force=*/true);
        last.deallocate(/*force=*/true);
        last = padded;
    }
    // (num_cores, L, E, 1, half_rows, 2*TILE) -> (num_cores, L, E, half_rows / 2, 4*TILE)
    auto last_grouped = reshape_to(
        last,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(half_rows / (2 * TILE_SIZE)),
         2,
         static_cast<int32_t>(TILE_SIZE),
         static_cast<int32_t>(2 * TILE_SIZE)});
    last.deallocate(/*force=*/false);
    auto side_by_side = permute_to(last_grouped, {0, 1, 2, 3, 5, 4, 6});
    last_grouped.deallocate(/*force=*/true);
    auto last_rows = reshape_to(
        side_by_side,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(half_rows / 2),
         static_cast<int32_t>(4 * TILE_SIZE)});
    ttnn::Tensor stream;
    if (full_groups > 0) {
        auto full_rows_t = reshape_to(
            full,
            {static_cast<int32_t>(num_cores),
             static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(full_groups * full_rows),
             static_cast<int32_t>(4 * TILE_SIZE)});
        stream = ttnn::concat({full_rows_t, last_rows}, 3);
        full.deallocate(/*force=*/true);
        side_by_side.deallocate(/*force=*/true);
    } else {
        // The half iteration is the only one (ceil(Ht / num_cores) <= 2).
        stream = last_rows;
    }
    const uint32_t block_rows = block_h * TILE_SIZE;
    const uint32_t stream_rows = full_groups * full_rows + half_rows / 2;
    auto result = reshape_to(
        stream,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(stream_rows / block_rows),
         static_cast<int32_t>(block_rows),
         static_cast<int32_t>(4 * TILE_SIZE)});
    stream.deallocate(/*force=*/false);
    return result;
}

}  // namespace

WeightCoreShardMaps get_weight_core_shard_maps(
    ttnn::MeshDevice* mesh_device, uint32_t hidden_size, uint32_t intermediate_size) {
    const auto in0_core_coords =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    // Matmul ring size = the live DRAM-bank count (12 on Wormhole, 7/8 on Blackhole) to match
    // ttnn.experimental.moe_compute.
    const uint32_t n_dram_banks = static_cast<uint32_t>(in0_core_coords.size());
    const uint32_t target_ring_size = n_dram_banks;

    // Ring ordering: sort the DRAM-bank logical core coords by (y, x) descending.
    std::vector<uint32_t> ring_to_dram_bank(n_dram_banks);
    std::iota(ring_to_dram_bank.begin(), ring_to_dram_bank.end(), 0u);
    std::sort(ring_to_dram_bank.begin(), ring_to_dram_bank.end(), [&](uint32_t a, uint32_t b) {
        const auto& ca = in0_core_coords[a];
        const auto& cb = in0_core_coords[b];
        if (ca.y != cb.y) {
            return ca.y > cb.y;
        }
        return ca.x > cb.x;
    });

    const uint32_t Nt = intermediate_size / TILE_SIZE;
    const uint32_t Ht = hidden_size / TILE_SIZE;
    const uint32_t max_w2_tiles = ceil_div(Ht, target_ring_size);
    const uint32_t groups_per_core = ceil_div(max_w2_tiles, BLOCK_TILES_W);

    WeightCoreShardMaps result;
    result.w0_w1_shard_map.reserve(target_ring_size);
    result.w2_shard_map.reserve(target_ring_size);

    std::vector<ttnn::CoreRange> dram_core_ranges;
    dram_core_ranges.reserve(n_dram_banks);

    for (uint32_t ring_pos = 0; ring_pos < target_ring_size; ++ring_pos) {
        // First n_dram_banks ring positions map to real DRAM-bank-adjacent cores;
        // positions beyond that are synthetic (HEIGHT_SHARDED regroups onto n_dram_banks physical shards).
        if (ring_pos < n_dram_banks) {
            const uint32_t dram_bank_id = ring_to_dram_bank[ring_pos];
            const ttnn::CoreCoord dram_core(dram_bank_id, 0);
            dram_core_ranges.emplace_back(dram_core, dram_core);
        }

        const uint32_t w0_w1_tiles = ::moe_ring::shard_tiles(Nt, ring_pos, target_ring_size);
        result.w0_w1_shard_map.push_back(w0_w1_tiles);

        const uint32_t w2_tiles = ::moe_ring::w2_shard_tiles(Ht, ring_pos, Nt, target_ring_size);
        const uint32_t last_group_tiles = w2_tiles - (groups_per_core - 1) * BLOCK_TILES_W;
        const uint32_t last_group_pad_tiles = groups_per_core * BLOCK_TILES_W - w2_tiles;
        result.w2_shard_map.emplace_back(last_group_tiles, last_group_pad_tiles);
    }

    result.dram_core_range_set = ttnn::CoreRangeSet(std::move(dram_core_ranges));
    return result;
}

WeightMemoryConfigs get_weight_mem_configs(
    ttnn::MeshDevice* mesh_device,
    uint32_t num_layers,
    uint32_t experts_per_device,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    bool has_bias) {
    TT_FATAL(
        hidden_size % TILE_SIZE == 0, "hidden_size ({}) must be divisible by TILE_SIZE ({})", hidden_size, TILE_SIZE);
    TT_FATAL(
        intermediate_size % TILE_SIZE == 0,
        "intermediate_size ({}) must be divisible by TILE_SIZE ({})",
        intermediate_size,
        TILE_SIZE);

    const auto shard_maps = get_weight_core_shard_maps(mesh_device, hidden_size, intermediate_size);
    const auto& w0_w1_shard_map = shard_maps.w0_w1_shard_map;

    // Stored K height for W0/W1: hidden tiles, plus one bias tile row.
    const uint32_t Ht = hidden_size / TILE_SIZE;
    const uint32_t k_dram_tiles = has_bias ? Ht + 1 : Ht;
    const uint32_t Nt_w0_w1 = intermediate_size / TILE_SIZE;

    const uint32_t num_cores = static_cast<uint32_t>(w0_w1_shard_map.size());
    const uint32_t num_banks = shard_maps.dram_core_range_set.num_cores();

    // Per-shape DRAM transaction size (both streams) and the stored rows of one block.
    const uint32_t tiles_per_txn = ::moe_ring::tiles_per_txn_for_shape(Ht, Nt_w0_w1, has_bias, num_cores);
    const uint32_t block_rows = ::moe_ring::block_tiles_h(tiles_per_txn) * TILE_SIZE;

    // Compact W0/W1 layout (prepare_w0_w1_tensor_for_moe_compute): every bank holds the same whole blocks
    // per (layer, expert).
    const uint32_t w0_w1_shard_height =
        num_layers * experts_per_device *
        ::moe_ring::w0_w1_bank_blocks_per_expert(k_dram_tiles, Nt_w0_w1, num_cores, num_banks, tiles_per_txn) *
        block_rows;
    constexpr uint32_t shard_width = 4 * TILE_SIZE;

    const ttnn::MemoryConfig w0_w1_mem_config{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            shard_maps.dram_core_range_set,
            {w0_w1_shard_height, shard_width},
            tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    // W2: every ring core stores the same whole blocks per (layer, expert) (full a2a iterations plus a possible
    // half-width last one; prepare_w2_tensor_for_moe_compute).
    const uint32_t Nt = intermediate_size / TILE_SIZE;
    const uint32_t w2_core_rows =
        ::moe_ring::w2_core_blocks_per_expert(Ht, has_bias ? Nt + 1 : Nt, num_cores, tiles_per_txn) * block_rows;
    const uint32_t w2_total_rows = num_layers * experts_per_device * num_cores * w2_core_rows;
    TT_FATAL(
        w2_total_rows % num_banks == 0,
        "w2 total rows {} not divisible by num_banks {} (num_cores={}, w2_core_rows={})",
        w2_total_rows,
        num_banks,
        num_cores,
        w2_core_rows);
    const uint32_t w2_shard_height = w2_total_rows / num_banks;

    const ttnn::MemoryConfig w2_mem_config{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            shard_maps.dram_core_range_set, {w2_shard_height, shard_width}, tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    return {w0_w1_mem_config, w2_mem_config};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> add_shared_expert_weights(
    const ttnn::Tensor& routed_w0,
    const ttnn::Tensor& routed_w1,
    const ttnn::Tensor& routed_w2,
    const ttnn::Tensor& shared_w0,
    const ttnn::Tensor& shared_w1,
    const ttnn::Tensor& shared_w2,
    const uint32_t cluster_axis) {
    const auto intermediate_dim = static_cast<uint32_t>(routed_w0.logical_shape()[-1]);
    const auto hidden_dim = static_cast<uint32_t>(routed_w0.logical_shape()[-2]);
    const auto tp_axis = 1 - cluster_axis;
    auto* device = routed_w0.device();

    // Per-core shard maps, generated with the SAME moe_ring::shard_tiles the kernel's
    // shard LUT uses (so host layout and kernel geometry agree by construction):
    //  - full_map: how the uniform prep slices EVERY expert's full-Nt intermediate.
    //  - tp_map:   the TpNt sub-shard the shared expert actually contracts.
    // We front-pack each core's real TpNt tiles into the front of its full-Nt shard
    // (zeros after), applying the same mapping to W0/W1 and W2. This keeps the whole
    // downstream prep + DRAM layout uniform (full-Nt per-expert stride) while letting
    // the kernel walk only the real per-core prefixes as a balanced TpNt ring.
    const auto full_map = get_weight_core_shard_maps(device, hidden_dim, intermediate_dim).w0_w1_shard_map;

    auto mp_w0 = ttnn::mesh_partition(shared_w0, -1, tp_axis);
    const auto tp_intermediate = static_cast<uint32_t>(mp_w0.logical_shape()[-1]);
    TT_FATAL(
        tp_intermediate % TILE_SIZE == 0,
        "TP-split intermediate ({}) must be tile-aligned (TILE_SIZE={})",
        tp_intermediate,
        TILE_SIZE);
    const auto tp_map = get_weight_core_shard_maps(device, hidden_dim, tp_intermediate).w0_w1_shard_map;

    auto working_shared_w0 = front_pack_per_core(mp_w0, /*axis=*/-1, full_map, tp_map);
    mp_w0.deallocate(/*force=*/false);
    auto output_w0 = ttnn::concat({routed_w0, working_shared_w0}, 1);
    working_shared_w0.deallocate(/*force=*/false);

    auto mp_w1 = ttnn::mesh_partition(shared_w1, -1, tp_axis);
    auto working_shared_w1 = front_pack_per_core(mp_w1, /*axis=*/-1, full_map, tp_map);
    mp_w1.deallocate(/*force=*/false);
    auto output_w1 = ttnn::concat({routed_w1, working_shared_w1}, 1);
    working_shared_w1.deallocate(/*force=*/false);

    // W2's intermediate (contraction K) is dim -2. Same maps -> each real W2 row pairs
    // with its real W0/W1 column.
    auto mp_w2 = ttnn::mesh_partition(shared_w2, -2, tp_axis);
    auto working_shared_w2 = front_pack_per_core(mp_w2, /*axis=*/-2, full_map, tp_map);
    mp_w2.deallocate(/*force=*/false);
    auto output_w2 = ttnn::concat({routed_w2, working_shared_w2}, 1);
    working_shared_w2.deallocate(/*force=*/false);

    return {output_w0, output_w1, output_w2};
}

namespace {

// tiles_per_txn: the op's per-shape transaction size (moe_ring::tiles_per_txn_for_shape of the hidden size, N and
// the bias flag).
ttnn::Tensor prepare_w0_w1_compact(
    const ttnn::Tensor& tt_w0,
    const ttnn::Tensor& tt_w1,
    uint32_t L,
    uint32_t E,
    uint32_t K,
    uint32_t N,
    uint32_t tiles_per_txn) {
    TT_FATAL(K % TILE_SIZE == 0, "K dimension ({}) must be divisible by TILE_SIZE ({})", K, TILE_SIZE);
    TT_FATAL(N % TILE_SIZE == 0, "N dimension ({}) must be divisible by TILE_SIZE ({})", N, TILE_SIZE);

    const auto shard_maps = get_weight_core_shard_maps(tt_w0.device(), /*hidden_size=*/K, /*intermediate_size=*/N);
    const auto& shard_map = shard_maps.w0_w1_shard_map;
    const uint32_t Nt = N / TILE_SIZE;
    const uint32_t num_cores = static_cast<uint32_t>(shard_map.size());
    const uint32_t num_banks = shard_maps.dram_core_range_set.num_cores();

    // Compact layout (moe_ring_common.h MoeRingConfig; Python mirror in moe_compute_utils.py): ring core c stores
    // only its shard_map[c] columns -- 4-wide block-columns (W0 c, W1 c, W0 c+1, W1 c+1) over K padded to 7-row
    // blocks, then for an odd count the last column as a 2-wide half block-column over K padded to 14-row blocks
    // (two consecutive K tile rows side by side per stored tile row). Per (layer, expert) the cores' slices are
    // laid back to back and cut into num_banks equal pieces (zero-padded); piece b lands in bank b.
    const uint32_t Kt = K / TILE_SIZE;
    const uint32_t block_h = ::moe_ring::block_tiles_h(tiles_per_txn);
    const uint32_t half_block_h = ::moe_ring::half_block_tiles_h(tiles_per_txn);
    const uint32_t blocks_per_col = ceil_div(Kt, block_h);
    const uint32_t blocks_per_half_col = ceil_div(Kt, half_block_h);
    const uint32_t Kp_full = blocks_per_col * block_h * TILE_SIZE;
    const uint32_t Kp_half = blocks_per_half_col * half_block_h * TILE_SIZE;
    const uint32_t Kp = std::max(Kp_full, Kp_half);
    const uint32_t expert_blocks =
        ::moe_ring::w0_w1_core_block_offset(Nt, num_cores, num_cores, blocks_per_col, blocks_per_half_col);
    const uint32_t bank_blocks = ::moe_ring::w0_w1_bank_blocks_per_expert(Kt, Nt, num_cores, num_banks, tiles_per_txn);
    const uint32_t block_rows = block_h * TILE_SIZE;  // every block (full or half) is block_h stored tile rows
    // Today's per-core stride shape is kept where the layout is byte-identical to it (14-tile transactions, every
    // core owning the same even column count, one core per bank).
    const uint32_t stored0 = ::moe_ring::w0_w1_stored_cols(Nt, 0, num_cores);
    bool same_stored = true;
    for (uint32_t c = 1; c < num_cores; ++c) {
        same_stored = same_stored && ::moe_ring::w0_w1_stored_cols(Nt, c, num_cores) == stored0;
    }
    const bool uniform =
        tiles_per_txn == ::moe_ring::DEFAULT_TILES_PER_TXN && num_banks == num_cores && stored0 % 2 == 0 && same_stored;

    ttnn::Tensor working_w0 = tt_w0;
    ttnn::Tensor working_w1 = tt_w1;
    if (K < Kp) {
        auto padding = zeros_like_dtype({L, E, Kp - K, N}, tt_w0);
        working_w0 = ttnn::concat({tt_w0, padding}, 2);
        working_w1 = ttnn::concat({tt_w1, padding}, 2);
        padding.deallocate(/*force=*/true);
    }

    // (L, E, Kp, N) -> (L, E, Kp, Nt, TILE_SIZE)
    auto w0_chunks = reshape_to(
        working_w0,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(Nt),
         static_cast<int32_t>(TILE_SIZE)});
    auto w1_chunks = reshape_to(
        working_w1,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(Nt),
         static_cast<int32_t>(TILE_SIZE)});

    // Stack along new axis 4 so w0/w1 alternate: (L, E, Kp, Nt, 2, TILE_SIZE).
    auto stacked = stack_along({w0_chunks, w1_chunks}, 4);
    w0_chunks.deallocate(/*force=*/false);
    w1_chunks.deallocate(/*force=*/false);
    working_w0.deallocate(/*force=*/false);
    working_w1.deallocate(/*force=*/false);
    auto interleaved = reshape_to(
        stacked,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(Nt),
         static_cast<int32_t>(2 * TILE_SIZE)});

    // Move Nt before Kp: (L, E, Nt, Kp, 2*TILE_SIZE).
    auto permuted = permute_to(interleaved, {0, 1, 3, 2, 4});
    interleaved.deallocate(/*force=*/false);
    stacked.deallocate(/*force=*/true);

    // Each core's compact slice as stored 4-tile rows: (L, E, rows, 4*TILE_SIZE).
    std::vector<ttnn::Tensor> each_slice;
    each_slice.reserve(2 * shard_map.size() + 1);
    uint32_t start_tile = 0;
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        const uint32_t num_tiles = shard_map[core_id];
        // Columns this core stores: its own, then zero columns up to the uniform stride for a non-compact shape.
        const uint32_t stored = ::moe_ring::w0_w1_stored_cols(Nt, core_id, num_cores);
        auto core_cols = slice_basic(
            permuted,
            {0, 0, static_cast<int32_t>(start_tile), 0, 0},
            {static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(start_tile + num_tiles),
             static_cast<int32_t>(Kp),
             static_cast<int32_t>(2 * TILE_SIZE)});
        if (stored > num_tiles) {
            auto zero_cols = zeros_like_dtype({L, E, stored - num_tiles, Kp, 2 * TILE_SIZE}, permuted);
            auto padded = ttnn::concat({core_cols, zero_cols}, 2);
            core_cols.deallocate(/*force=*/true);
            zero_cols.deallocate(/*force=*/true);
            core_cols = padded;
        }
        const uint32_t pairs = stored / 2;
        if (pairs > 0) {
            // (L, E, 2*pairs, Kp_full, 2*TILE) -> (L, E, pairs, Kp_full, 4*TILE): row k = W0 c, W1 c, W0 c+1, W1 c+1
            auto cols = slice_basic(
                core_cols,
                {0, 0, 0, 0, 0},
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(2 * pairs),
                 static_cast<int32_t>(Kp_full),
                 static_cast<int32_t>(2 * TILE_SIZE)});
            auto grouped = reshape_to(
                cols,
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(pairs),
                 2,
                 static_cast<int32_t>(Kp_full),
                 static_cast<int32_t>(2 * TILE_SIZE)});
            cols.deallocate(/*force=*/false);
            auto paired = permute_to(grouped, {0, 1, 2, 4, 3, 5});
            grouped.deallocate(/*force=*/true);
            each_slice.push_back(reshape_to(
                paired,
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(pairs * Kp_full),
                 static_cast<int32_t>(4 * TILE_SIZE)}));
        }
        if (stored % 2 != 0) {
            // (L, E, Kp_half, 2*TILE) -> (L, E, Kp_half / 2, 4*TILE): tile row j = K tile rows 2j and 2j+1 side by side
            const uint32_t col = 2 * pairs;
            auto half = slice_basic(
                core_cols,
                {0, 0, static_cast<int32_t>(col), 0, 0},
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(col + 1),
                 static_cast<int32_t>(Kp_half),
                 static_cast<int32_t>(2 * TILE_SIZE)});
            auto grouped = reshape_to(
                half,
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(Kp_half / (2 * TILE_SIZE)),
                 2,
                 static_cast<int32_t>(TILE_SIZE),
                 static_cast<int32_t>(2 * TILE_SIZE)});
            half.deallocate(/*force=*/false);
            auto side_by_side = permute_to(grouped, {0, 1, 2, 4, 3, 5});
            grouped.deallocate(/*force=*/true);
            each_slice.push_back(reshape_to(
                side_by_side,
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(Kp_half / 2),
                 static_cast<int32_t>(4 * TILE_SIZE)}));
        }
        core_cols.deallocate(/*force=*/true);
        start_tile += num_tiles;
    }
    const uint32_t stream_pad_blocks = num_banks * bank_blocks - expert_blocks;
    if (stream_pad_blocks > 0) {
        each_slice.push_back(zeros_like_dtype({L, E, stream_pad_blocks * block_rows, 4 * TILE_SIZE}, permuted));
    }

    auto stream = ttnn::concat(each_slice, 2);
    for (auto& s : each_slice) {
        s.deallocate(/*force=*/true);
    }
    each_slice.clear();
    permuted.deallocate(/*force=*/true);

    // (L, E, num_banks * bank_blocks * block_rows, 4*TILE) -> (num_banks, L, E, bank_blocks, block_rows, 4*TILE)
    auto banked = reshape_to(
        stream,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(num_banks),
         static_cast<int32_t>(bank_blocks),
         static_cast<int32_t>(block_rows),
         static_cast<int32_t>(4 * TILE_SIZE)});
    auto packed = permute_to(banked, {2, 0, 1, 3, 4, 5});
    banked.deallocate(/*force=*/false);
    stream.deallocate(/*force=*/true);
    if (uniform) {
        // Every core owns the same even column count: byte-identical to the per-core stride layout, keep its shape
        // (num_cores, L, E, groups_per_core, K_padded, 4*TILE_SIZE).
        auto per_core = reshape_to(
            packed,
            {static_cast<int32_t>(num_cores),
             static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(stored0 / 2),
             static_cast<int32_t>(Kp_full),
             static_cast<int32_t>(4 * TILE_SIZE)});
        packed.deallocate(/*force=*/false);
        packed = per_core;
    }
    auto result = ttnn::to_layout(packed, ttnn::Layout::TILE);
    packed.deallocate(/*force=*/false);
    return result;
}

}  // namespace

ttnn::Tensor prepare_w0_w1_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w0, const ttnn::Tensor& tt_w1, uint32_t L, uint32_t E, uint32_t K, uint32_t N) {
    const uint32_t num_cores = static_cast<uint32_t>(
        get_weight_core_shard_maps(tt_w0.device(), /*hidden_size=*/K, /*intermediate_size=*/N).w0_w1_shard_map.size());
    return prepare_w0_w1_compact(
        tt_w0,
        tt_w1,
        L,
        E,
        K,
        N,
        ::moe_ring::tiles_per_txn_for_shape(K / TILE_SIZE, N / TILE_SIZE, /*has_bias=*/false, num_cores));
}

ttnn::Tensor prepare_w2_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w2, uint32_t L, uint32_t E, uint32_t N, uint32_t K) {
    TT_FATAL(N % TILE_SIZE == 0, "N dimension ({}) must be divisible by TILE_SIZE ({})", N, TILE_SIZE);
    TT_FATAL(K % TILE_SIZE == 0, "K dimension ({}) must be divisible by TILE_SIZE ({})", K, TILE_SIZE);

    const auto shard_maps = get_weight_core_shard_maps(tt_w2.device(), /*hidden_size=*/K, /*intermediate_size=*/N);
    const auto& w0_w1_shard_map = shard_maps.w0_w1_shard_map;
    const auto& w2_shard_map = shard_maps.w2_shard_map;

    auto n_reordered_no_pad = prepare_w2_no_n_pad(tt_w2, L, E, N, K, w2_shard_map, w0_w1_shard_map);

    const uint32_t Nt = N / TILE_SIZE;
    const uint32_t Kt = K / TILE_SIZE;

    // Pad N to whole DRAM blocks for the per-shape transaction size (and lay a half-width last a2a iteration out
    // 2 wide).
    const uint32_t tiles_per_txn =
        ::moe_ring::tiles_per_txn_for_shape(Kt, Nt, /*has_bias=*/false, static_cast<uint32_t>(w0_w1_shard_map.size()));
    auto blocks = w2_blocks_from_groups(n_reordered_no_pad, Nt, Kt, tiles_per_txn);
    n_reordered_no_pad.deallocate(/*force=*/false);
    auto result = ttnn::to_layout(blocks, ttnn::Layout::TILE);
    blocks.deallocate(/*force=*/false);
    return result;
}

ttnn::Tensor prepare_w0_w1_tensor_with_bias(
    const ttnn::Tensor& tt_w0,
    const ttnn::Tensor& tt_w1,
    const ttnn::Tensor& tt_b0,
    const ttnn::Tensor& tt_b1,
    uint32_t L,
    uint32_t E,
    uint32_t K,
    uint32_t N) {
    TT_FATAL(K % TILE_SIZE == 0, "K dimension ({}) must be divisible by TILE_SIZE ({})", K, TILE_SIZE);
    TT_FATAL(N % TILE_SIZE == 0, "N dimension ({}) must be divisible by TILE_SIZE ({})", N, TILE_SIZE);

    const uint32_t K_with_bias = (K / TILE_SIZE + 1) * TILE_SIZE;

    // Tile-format bias: (L, E, N) -> (L, E, TILE_SIZE, N) with row 0 populated.
    auto b0_row = ttnn::unsqueeze(tt_b0, 2);  // (L, E, 1, N)
    auto b1_row = ttnn::unsqueeze(tt_b1, 2);
    auto pad_rows = zeros_like_dtype({L, E, TILE_SIZE - 1, N}, tt_b0);
    auto b0_tiled = ttnn::concat({b0_row, pad_rows}, 2);
    auto b1_tiled = ttnn::concat({b1_row, pad_rows}, 2);
    b0_row.deallocate(/*force=*/false);
    b1_row.deallocate(/*force=*/false);
    pad_rows.deallocate(/*force=*/true);

    auto w0_b0 = ttnn::concat({tt_w0, b0_tiled}, 2);  // (L, E, K + TILE_SIZE, N)
    auto w1_b1 = ttnn::concat({tt_w1, b1_tiled}, 2);
    b0_tiled.deallocate(/*force=*/true);
    b1_tiled.deallocate(/*force=*/true);

    const uint32_t num_cores = static_cast<uint32_t>(
        get_weight_core_shard_maps(tt_w0.device(), /*hidden_size=*/K, /*intermediate_size=*/N).w0_w1_shard_map.size());
    const uint32_t tiles_per_txn =
        ::moe_ring::tiles_per_txn_for_shape(K / TILE_SIZE, N / TILE_SIZE, /*has_bias=*/true, num_cores);
    auto result = prepare_w0_w1_compact(w0_b0, w1_b1, L, E, K_with_bias, N, tiles_per_txn);
    w0_b0.deallocate(/*force=*/true);
    w1_b1.deallocate(/*force=*/true);
    return result;
}

ttnn::Tensor prepare_w2_tensor_with_bias(
    const ttnn::Tensor& tt_w2, const ttnn::Tensor& tt_b2, uint32_t L, uint32_t E, uint32_t N, uint32_t K) {
    TT_FATAL(N % TILE_SIZE == 0, "N dimension ({}) must be divisible by TILE_SIZE ({})", N, TILE_SIZE);
    TT_FATAL(K % TILE_SIZE == 0, "K dimension ({}) must be divisible by TILE_SIZE ({})", K, TILE_SIZE);

    const auto shard_maps = get_weight_core_shard_maps(tt_w2.device(), /*hidden_size=*/K, /*intermediate_size=*/N);
    const auto& w0_w1_shard_map = shard_maps.w0_w1_shard_map;
    const auto& w2_shard_map = shard_maps.w2_shard_map;

    const uint32_t Nt = N / TILE_SIZE;
    const uint32_t Kt = K / TILE_SIZE;
    const uint32_t num_cores = static_cast<uint32_t>(w2_shard_map.size());
    const uint32_t first_pair_sum = w2_shard_map[0].first + w2_shard_map[0].second;
    const uint32_t w2_groups_per_core = ceil_div(Kt, num_cores * first_pair_sum);

    // 1) Ring-rotated W2 (without bias) at the kernel's expected layout (no trailing N-pad).
    auto n_reordered_no_pad = prepare_w2_no_n_pad(tt_w2, L, E, N, K, w2_shard_map, w0_w1_shard_map);

    // 2) Bias tile row: (L, E, K) -> (L, E, TILE_SIZE, K) with row 0 populated, then column-shard.
    auto b2_row = ttnn::unsqueeze(tt_b2, 2);  // (L, E, 1, K)
    auto pad_rows = zeros_like_dtype({L, E, TILE_SIZE - 1, K}, tt_b2);
    auto b2_tiled = ttnn::concat({b2_row, pad_rows}, 2);
    b2_row.deallocate(/*force=*/false);
    pad_rows.deallocate(/*force=*/true);

    std::vector<ttnn::Tensor> b2_each_shard;
    b2_each_shard.reserve(3 * num_cores);
    uint32_t start_col = 0;
    const uint32_t full_block_width = (w2_groups_per_core - 1) * 4 * TILE_SIZE;
    for (const auto& [last_group_tiles, last_group_pad_tiles] : w2_shard_map) {
        if (full_block_width > 0) {
            b2_each_shard.push_back(slice_basic(
                b2_tiled,
                {0, 0, 0, static_cast<int32_t>(start_col)},
                {static_cast<int32_t>(L),
                 static_cast<int32_t>(E),
                 static_cast<int32_t>(TILE_SIZE),
                 static_cast<int32_t>(start_col + full_block_width)}));
            start_col += full_block_width;
        }
        const uint32_t last_group_width = last_group_tiles * TILE_SIZE;
        b2_each_shard.push_back(slice_basic(
            b2_tiled,
            {0, 0, 0, static_cast<int32_t>(start_col)},
            {static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(TILE_SIZE),
             static_cast<int32_t>(start_col + last_group_width)}));
        start_col += last_group_width;
        if (last_group_pad_tiles > 0) {
            b2_each_shard.push_back(zeros_like_dtype({L, E, TILE_SIZE, last_group_pad_tiles * TILE_SIZE}, tt_b2));
        }
    }

    auto b2_reordered = ttnn::concat(b2_each_shard, -1);
    for (auto& s : b2_each_shard) {
        s.deallocate(/*force=*/true);
    }
    b2_each_shard.clear();
    b2_tiled.deallocate(/*force=*/true);
    auto b2_grouped = reshape_to(
        b2_reordered,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(TILE_SIZE),
         static_cast<int32_t>(num_cores),
         static_cast<int32_t>(w2_groups_per_core),
         static_cast<int32_t>(4 * TILE_SIZE)});
    b2_grouped = permute_to(b2_grouped, {3, 0, 1, 4, 2, 5});
    b2_reordered.deallocate(/*force=*/true);

    // 3) Concat bias row after weight tiles (NOT rotated).
    auto n_with_bias = ttnn::concat({n_reordered_no_pad, b2_grouped}, 4);
    n_reordered_no_pad.deallocate(/*force=*/true);
    b2_grouped.deallocate(/*force=*/true);

    // 4) Pad to whole DRAM blocks for the per-shape transaction size (and lay a half-width last a2a iteration out
    //    2 wide).
    const uint32_t tiles_per_txn = ::moe_ring::tiles_per_txn_for_shape(Kt, Nt, /*has_bias=*/true, num_cores);
    auto blocks = w2_blocks_from_groups(n_with_bias, Nt + 1, Kt, tiles_per_txn);
    n_with_bias.deallocate(/*force=*/false);
    auto result = ttnn::to_layout(blocks, ttnn::Layout::TILE);
    blocks.deallocate(/*force=*/false);
    return result;
}

// Optionally returns a host tensor to facilitate test quantity caching
ttnn::Tensor quantize_weights_via_host(
    const ttnn::Tensor& device_tensor, ttnn::DataType dtype, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto host_tensor = ttnn::from_device(device_tensor);
    auto cast_tensor = ttnn::to_dtype(host_tensor, dtype);
    // to_dtype is a no-op when the dtype already matches, returning host_tensor itself.
    // Only free host_tensor when the cast produced a distinct tensor; otherwise the
    // deallocate would invalidate cast_tensor (the value we return / pass to to_device).
    if (host_tensor.dtype() != dtype) {
        host_tensor.deallocate(/*force=*/true);
    }

    if (!memory_config.has_value()) {
        return cast_tensor;
    }

    auto result = ttnn::to_device(cast_tensor, device_tensor.device(), memory_config);
    cast_tensor.deallocate(/*force=*/true);
    return result;
}

}  // namespace ttnn::experimental
