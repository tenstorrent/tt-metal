// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/tensor/tensor_types.hpp>

// Geometry shared by the validation and the program factories, so the size the validation admits is
// the size the factory allocates.
namespace ttnn::operations::toy_scaled_add::detail {

// Streaming circular buffers on the interleaved program hold two tiles: one being filled while the
// other is consumed.
constexpr uint32_t kStreamCbTiles = 2;

struct TileGrid {
    uint32_t rows;   // tile-rows, every leading dim folded in
    uint32_t width;  // tiles per row (Wt)
};

// Every size below comes from the tensor's own spec (its tile, its dtype), not from constants, so a
// page in a circular buffer is exactly a page of the tensor it carries.
inline TileGrid tile_grid(const Tensor& t) {
    const auto& padded = t.padded_shape();
    const auto [tile_h, tile_w] = t.tensor_spec().tile().get_tile_shape();
    const uint32_t width = padded[-1] / tile_w;
    const auto rows = static_cast<uint32_t>(padded.volume() / (padded[-1] * tile_h));
    return {rows, width};
}

inline uint32_t tile_bytes(const Tensor& t) {
    return t.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(t.dtype()));
}

// Circular-buffer bytes one core needs. Shard-backed buffers live in the shards themselves, so on the
// sharded program only gamma's row takes circular-buffer space.
inline uint32_t cb_bytes_per_core(
    const Tensor& a, const Tensor& b, const std::optional<Tensor>& gamma, uint32_t out_tile_bytes, bool sharded) {
    const uint32_t gamma_bytes = gamma.has_value() ? tile_grid(a).width * tile_bytes(*gamma) : 0;
    if (sharded) {
        return gamma_bytes;
    }
    return (kStreamCbTiles * (tile_bytes(a) + tile_bytes(b) + out_tile_bytes)) + gamma_bytes;
}

}  // namespace ttnn::operations::toy_scaled_add::detail
