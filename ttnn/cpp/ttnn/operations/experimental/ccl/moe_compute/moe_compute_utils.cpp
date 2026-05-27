// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "device/kernels/moe_ring_common.h"

namespace ttnn::experimental {

namespace {

// `BLOCK_TILES_H` in the Python module; both W0/W1 and W2 use the same value.
// Stays in sync with moe_ring_common.h.
constexpr uint32_t BLOCK_TILES_H = ::moe_ring::W0_W1_BLOCK_TILES_H;
constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH;

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

ttnn::Tensor reshape_to(const ttnn::Tensor& t, std::vector<int32_t> shape) {
    return ttnn::reshape(t, ttsl::Span<const int32_t>(shape.data(), shape.size()));
}

ttnn::Tensor permute_to(const ttnn::Tensor& t, std::initializer_list<int64_t> dims) {
    return ttnn::permute(t, ttsl::SmallVector<int64_t>(dims));
}

ttnn::Tensor slice_basic(const ttnn::Tensor& t, ttsl::SmallVector<int32_t> begins, ttsl::SmallVector<int32_t> ends) {
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

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> add_shared_expert_weights(
    const ttnn::Tensor& routed_w0,
    const ttnn::Tensor& routed_w1,
    const ttnn::Tensor& routed_w2,
    const ttnn::Tensor& shared_w0,
    const ttnn::Tensor& shared_w1,
    const ttnn::Tensor& shared_w2) {
    auto output_w0 = ttnn::concat({routed_w0, shared_w0}, 1);
    auto tile_w0 = ttnn::to_layout(output_w0, ttnn::Layout::TILE);
    output_w0.deallocate(/*force=*/true);
    auto output_w1 = ttnn::concat({routed_w1, shared_w1}, 1);
    auto tile_w1 = ttnn::to_layout(output_w1, ttnn::Layout::TILE);
    output_w1.deallocate(/*force=*/true);
    auto output_w2 = ttnn::concat({routed_w2, shared_w2}, 1);
    auto tile_w2 = ttnn::to_layout(output_w2, ttnn::Layout::TILE);
    output_w2.deallocate(/*force=*/true);
    return {tile_w0, tile_w1, tile_w2};
}

ttnn::Tensor prepare_w0_w1_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w0,
    const ttnn::Tensor& tt_w1,
    uint32_t L,
    uint32_t E,
    uint32_t K,
    uint32_t N,
    const std::vector<uint32_t>& shard_map) {
    if (K % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "K dimension (" + std::to_string(K) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }
    if (N % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "N dimension (" + std::to_string(N) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }

    const uint32_t Nt = N / TILE_SIZE;
    // Pad K up to a multiple of (TILE_SIZE * BLOCK_TILES_H) — matches the DRAM read transaction.
    const uint32_t Kp = ceil_div(K / TILE_SIZE, BLOCK_TILES_H) * TILE_SIZE * BLOCK_TILES_H;
    const uint32_t num_cores = static_cast<uint32_t>(shard_map.size());

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

    // Validate shard distribution and pick the padded shard size.
    uint32_t max_shard_size = *std::max_element(shard_map.begin(), shard_map.end());
    max_shard_size = max_shard_size + (max_shard_size % 2);  // round up to even
    for (uint32_t x : shard_map) {
        if (!(x == max_shard_size || (max_shard_size >= 1 && x == max_shard_size - 1) ||
              (max_shard_size >= 2 && x == max_shard_size - 2))) {
            std::string got;
            for (size_t i = 0; i < shard_map.size(); ++i) {
                got += (i ? ", " : "") + std::to_string(shard_map[i]);
            }
            throw std::runtime_error(
                "W0W1 shard sizes must be in [" + std::to_string(max_shard_size - 2) + ", " +
                std::to_string(max_shard_size) + "] (after rounding max to even), got: [" + got + "]");
        }
    }

    std::vector<ttnn::Tensor> each_shard;
    uint32_t start_tile = 0;
    for (uint32_t num_tiles : shard_map) {
        each_shard.push_back(slice_basic(
            permuted,
            {0, 0, static_cast<int32_t>(start_tile), 0, 0},
            {static_cast<int32_t>(L),
             static_cast<int32_t>(E),
             static_cast<int32_t>(start_tile + num_tiles),
             static_cast<int32_t>(Kp),
             static_cast<int32_t>(2 * TILE_SIZE)}));
        const uint32_t pad_tiles = max_shard_size - num_tiles;
        if (pad_tiles > 0) {
            each_shard.push_back(zeros_like_dtype({L, E, pad_tiles, Kp, 2 * TILE_SIZE}, permuted));
        }
        start_tile += num_tiles;
    }

    auto reordered = ttnn::concat(each_shard, 2);
    for (auto& s : each_shard) {
        s.deallocate(/*force=*/true);
    }
    each_shard.clear();
    permuted.deallocate(/*force=*/true);

    // (L, E, num_cores * max_shard_size, Kp, 2*TILE_SIZE) -> (num_cores, L, E, max_shard_size, Kp, 2*TILE_SIZE)
    auto all_groups = reshape_to(
        reordered,
        {static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(num_cores),
         static_cast<int32_t>(max_shard_size),
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(2 * TILE_SIZE)});
    all_groups = permute_to(all_groups, {2, 0, 1, 3, 4, 5});
    reordered.deallocate(/*force=*/true);

    // Pair adjacent (w0, w1) tiles into 4-wide trailing chunks:
    // (num_cores, L, E, groups_per_core, 2, Kp, 2*TILE_SIZE)
    const uint32_t groups_per_core = max_shard_size / 2;
    auto paired = reshape_to(
        all_groups,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(groups_per_core),
         2,
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(2 * TILE_SIZE)});
    paired = permute_to(paired, {0, 1, 2, 3, 5, 4, 6});
    all_groups.deallocate(/*force=*/true);
    auto packed = reshape_to(
        paired,
        {static_cast<int32_t>(num_cores),
         static_cast<int32_t>(L),
         static_cast<int32_t>(E),
         static_cast<int32_t>(groups_per_core),
         static_cast<int32_t>(Kp),
         static_cast<int32_t>(4 * TILE_SIZE)});
    auto result = ttnn::to_layout(packed, ttnn::Layout::TILE);
    packed.deallocate(/*force=*/false);
    paired.deallocate(/*force=*/true);
    return result;
}

ttnn::Tensor prepare_w2_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w2,
    uint32_t L,
    uint32_t E,
    uint32_t N,
    uint32_t K,
    const std::vector<std::pair<uint32_t, uint32_t>>& w2_shard_map,
    const std::vector<uint32_t>& w0_w1_shard_map) {
    if (N % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "N dimension (" + std::to_string(N) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }
    if (K % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "K dimension (" + std::to_string(K) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }

    auto n_reordered_no_pad = prepare_w2_no_n_pad(tt_w2, L, E, N, K, w2_shard_map, w0_w1_shard_map);

    const uint32_t Nt = N / TILE_SIZE;
    const uint32_t num_cores = static_cast<uint32_t>(w2_shard_map.size());
    const uint32_t first_pair_sum = w2_shard_map[0].first + w2_shard_map[0].second;
    const uint32_t Kt = K / TILE_SIZE;
    const uint32_t w2_groups_per_core = ceil_div(Kt, num_cores * first_pair_sum);

    // Pad N up to a multiple of BLOCK_TILES_H tiles for the 7-tile DRAM reads.
    const uint32_t n_padded_tiles = ceil_div(Nt, BLOCK_TILES_H) * BLOCK_TILES_H;
    const uint32_t n_padding = n_padded_tiles * TILE_SIZE - N;
    if (n_padding > 0) {
        auto pad = zeros_like_dtype({num_cores, L, E, w2_groups_per_core, n_padding, 4 * TILE_SIZE}, tt_w2);
        auto padded = ttnn::concat({n_reordered_no_pad, pad}, 4);
        n_reordered_no_pad.deallocate(/*force=*/true);
        pad.deallocate(/*force=*/true);
        auto result = ttnn::to_layout(padded, ttnn::Layout::TILE);
        padded.deallocate(/*force=*/true);
        return result;
    }
    auto result = ttnn::to_layout(n_reordered_no_pad, ttnn::Layout::TILE);
    n_reordered_no_pad.deallocate(/*force=*/true);
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
    uint32_t N,
    const std::vector<uint32_t>& shard_map) {
    if (K % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "K dimension (" + std::to_string(K) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }
    if (N % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "N dimension (" + std::to_string(N) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }

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

    auto result = prepare_w0_w1_tensor_for_moe_compute(w0_b0, w1_b1, L, E, K_with_bias, N, shard_map);
    w0_b0.deallocate(/*force=*/true);
    w1_b1.deallocate(/*force=*/true);
    return result;
}

ttnn::Tensor prepare_w2_tensor_with_bias(
    const ttnn::Tensor& tt_w2,
    const ttnn::Tensor& tt_b2,
    uint32_t L,
    uint32_t E,
    uint32_t N,
    uint32_t K,
    const std::vector<std::pair<uint32_t, uint32_t>>& w2_shard_map,
    const std::vector<uint32_t>& w0_w1_shard_map) {
    if (N % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "N dimension (" + std::to_string(N) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }
    if (K % TILE_SIZE != 0) {
        throw std::invalid_argument(
            "K dimension (" + std::to_string(K) + ") must be divisible by TILE_SIZE (" + std::to_string(TILE_SIZE) +
            ")");
    }

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

    // 4) Pad to BLOCK_TILES_H tile multiple along N.
    const uint32_t n_total_tiles = Nt + 1;
    const uint32_t n_target_tiles = ceil_div(n_total_tiles, BLOCK_TILES_H) * BLOCK_TILES_H;
    const uint32_t n_padding = (n_target_tiles - n_total_tiles) * TILE_SIZE;
    if (n_padding > 0) {
        auto pad = zeros_like_dtype({num_cores, L, E, w2_groups_per_core, n_padding, 4 * TILE_SIZE}, tt_w2);
        auto padded = ttnn::concat({n_with_bias, pad}, 4);
        n_with_bias.deallocate(/*force=*/true);
        pad.deallocate(/*force=*/true);
        n_with_bias = padded;
    }
    auto result = ttnn::to_layout(n_with_bias, ttnn::Layout::TILE);
    n_with_bias.deallocate(/*force=*/true);
    return result;
}

ttnn::Tensor quantize_weights_via_host(
    const ttnn::Tensor& device_tensor, ttnn::DataType dtype, const ttnn::MemoryConfig& memory_config) {
    auto host_tensor = ttnn::from_device(device_tensor);
    auto cast_tensor = ttnn::to_dtype(host_tensor, dtype);
    host_tensor.deallocate(/*force=*/true);
    auto result = ttnn::to_device(cast_tensor, device_tensor.device(), memory_config);
    cast_tensor.deallocate(/*force=*/true);
    return result;
}

}  // namespace ttnn::experimental
