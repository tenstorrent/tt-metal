// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"  // tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "nlp_create_qkv_heads.hpp"

#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/small_vector.hpp>

namespace ttnn::experimental {
namespace {

// The TM kernels walk every head as a whole number of TILE_WIDTH-wide tiles, so a head
// dim that is not a multiple of TILE_WIDTH both drops the partial tile of each head and
// mis-spaces the per-head tile addresses, which silently writes the wrong values rather
// than failing. Route those through a row-major round trip instead: in row major the
// head/dim split of the last dim is a pure view, and the final to_layout pads every
// head's tail to TILE_WIDTH on its own, so each head keeps exactly its own data.
ttnn::Tensor to_heads(
    const ttnn::Tensor& row_major_tensor,
    bool transpose_k_heads,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    // [B, S, NH, D] -> [B, NH, S, D], or [B, NH, D, S] when the caller wants K transposed.
    ttsl::SmallVector<int64_t> permute_dims({0, 2, 1, 3});
    if (transpose_k_heads) {
        permute_dims = {0, 2, 3, 1};
    }
    ttnn::Tensor heads = ttnn::permute(row_major_tensor, permute_dims);

    return ttnn::to_layout(
        heads,
        ttnn::Layout::TILE,
        /*dtype=*/std::nullopt,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
}

// Slices a contiguous run of head sections out of a row-major [B, S, TOT, D] view, on the
// head dim (2), leaving dims 0, 1 and 3 whole.
ttnn::Tensor slice_heads(
    const ttnn::Tensor& row_major_heads, uint32_t begin_head, uint32_t num_heads) {
    const auto& shape = row_major_heads.logical_shape();
    return ttnn::slice(
        row_major_heads,
        ttsl::SmallVector<int64_t>({0, 0, (int64_t)begin_head, 0}),
        ttsl::SmallVector<int64_t>(
            {(int64_t)shape[0], (int64_t)shape[1], (int64_t)(begin_head + num_heads), (int64_t)shape[3]}),
        ttsl::SmallVector<int64_t>({1, 1, 1, 1}));
}

// [B, 1, S, TOT * D] -> a [B, S, TOT, D] row-major view, so the last dim is the head dim
// and dim 3 indexes whole heads.
ttnn::Tensor heads_view(const ttnn::Tensor& input_tensor, uint32_t num_heads, uint32_t head_dim) {
    return ttnn::reshape(
        input_tensor,
        ttnn::Shape(
            {input_tensor.logical_shape()[0],
             input_tensor.logical_shape()[2],
             num_heads,
             head_dim}));
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_subtile_head_dim(
    const ttnn::Tensor& input_tensor_q,
    const std::optional<ttnn::Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    bool kv_tied,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    // Layouts and the head views are pure reshapes of the logical elements, so nothing has
    // to be reinterpreted against the padded width: the head dim only has to fit the
    // tensor's own width, and each head's tail gets padded to TILE_WIDTH on its own.
    ttnn::Tensor q_row_major = ttnn::to_layout(input_tensor_q, ttnn::Layout::ROW_MAJOR);

    ttnn::Tensor q_heads;
    ttnn::Tensor k_heads;
    ttnn::Tensor v_heads;
    if (input_tensor_kv.has_value()) {
        // Q stands alone, so its whole width is the Q section.
        q_heads = heads_view(q_row_major, num_q_heads, head_dim);
        // The KV tensor holds kv_sections * num_kv_heads sections: one shared K/V section
        // when tied, separate K and V sections otherwise.
        const uint32_t kv_sections = kv_tied ? 1 : 2;
        ttnn::Tensor kv_heads = heads_view(
            ttnn::to_layout(input_tensor_kv.value(), ttnn::Layout::ROW_MAJOR),
            kv_sections * num_kv_heads,
            head_dim);
        k_heads = slice_heads(kv_heads, 0, num_kv_heads);
        v_heads = kv_tied ? k_heads : slice_heads(kv_heads, num_kv_heads, num_kv_heads);
    } else {
        // Q, then K/V follow in the same tensor, so the view has to span the full fused
        // width and Q is the first num_q_heads sections of it. Using num_q_heads alone
        // here would describe B*S*num_q_heads*D elements for a B*S*(sections*D) tensor.
        const uint32_t fused_sections = kv_tied ? num_q_heads + num_kv_heads
                                                : num_q_heads + 2 * num_kv_heads;
        ttnn::Tensor fused_heads = heads_view(q_row_major, fused_sections, head_dim);
        q_heads = slice_heads(fused_heads, 0, num_q_heads);
        k_heads = slice_heads(fused_heads, num_q_heads, num_kv_heads);
        v_heads = kv_tied ? k_heads : slice_heads(fused_heads, num_q_heads + num_kv_heads, num_kv_heads);
    }

    return {
        to_heads(q_heads, /*transpose_k_heads=*/false, memory_config),
        to_heads(k_heads, transpose_k_heads, memory_config),
        to_heads(v_heads, /*transpose_k_heads=*/false, memory_config)};
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_q_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const bool kv_tied,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors,
    const bool runtime_rows) {
    TT_FATAL(num_q_heads > 0, "num_q_heads must be greater than 0");
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    // Every head takes at least one column, so bounding the head counts by the width they split also keeps
    // the uint32_t section sums below from wrapping to 0.
    const uint64_t q_width = input_tensor_q.padded_shape()[3];
    // The head dim is a property of the data, not of the padded tile: a section count that does
    // not fill the last tile leaves the logical width short of the padded one, and dividing the
    // padded width would derive a head_dim larger than the one the caller actually packed.
    const uint64_t q_logical_width = input_tensor_q.logical_shape()[3];
    const uint64_t kv_logical_width =
        input_tensor_kv.has_value() ? input_tensor_kv.value().logical_shape()[3] : 0;
    uint32_t head_dim;
    if (input_tensor_kv.has_value()) {
        TT_FATAL(num_kv_heads_val > 0, "num_kv_heads must be greater than 0 when a KV tensor is given");
        // Tied, so the KV tensor holds one K/V section rather than two.
        const uint32_t kv_sections = kv_tied ? 1 : 2;
        TT_FATAL(num_q_heads <= q_width, "num_q_heads ({}) exceeds the Q width ({})", num_q_heads, q_width);
        TT_FATAL(
            uint64_t{kv_sections} * num_kv_heads_val <= input_tensor_kv.value().padded_shape()[3],
            "{} * num_kv_heads ({}) exceeds the KV width ({})",
            kv_sections,
            num_kv_heads_val,
            input_tensor_kv.value().padded_shape()[3]);
        TT_FATAL(
            q_logical_width % num_q_heads == 0,
            "Unsupported input shape: Q width ({}) is not divisible by num_q_heads ({})",
            q_logical_width,
            num_q_heads);
        TT_FATAL(
            kv_logical_width % (kv_sections * num_kv_heads_val) == 0,
            "Unsupported input shape: KV width ({}) is not divisible by {} * num_kv_heads ({})",
            kv_logical_width,
            kv_sections,
            num_kv_heads_val);
        head_dim = static_cast<uint32_t>(q_logical_width / num_q_heads);
        TT_FATAL(
            kv_logical_width / (kv_sections * num_kv_heads_val) == head_dim,
            "Head dims must be the same for Q and K, V");
    } else if (kv_tied) {
        // One K/V section instead of two, so the fused width is (q + kv), not (q + 2*kv).
        TT_FATAL(
            uint64_t{num_q_heads} + num_kv_heads_val <= q_width,
            "num_q_heads ({}) + num_kv_heads ({}) exceeds the fused width ({})",
            num_q_heads,
            num_kv_heads_val,
            q_width);
        const uint32_t tied_sections = num_q_heads + num_kv_heads_val;
        TT_FATAL(
            q_logical_width % tied_sections == 0,
            "Unsupported input shape: fused width ({}) is not divisible by tied sections ({})",
            q_logical_width,
            tied_sections);
        // Only bounded above via q + kv, so the untied count can exceed uint32_t; keep it 64-bit.
        const uint64_t untied_sections = uint64_t{num_q_heads} + 2 * uint64_t{num_kv_heads_val};
        TT_FATAL(
            q_logical_width % untied_sections != 0,
            "Ambiguous kv_tied fused input shape: width {} is divisible by both {} (tied) and {} (untied) sections",
            q_logical_width,
            tied_sections,
            untied_sections);
        head_dim = static_cast<uint32_t>(q_logical_width / tied_sections);
    } else {
        // 64-bit: a uint32 sum can wrap to 0, pass the bound and then divide by zero.
        const uint64_t untied_sections = uint64_t{num_q_heads} + 2 * uint64_t{num_kv_heads_val};
        TT_FATAL(
            untied_sections <= q_width,
            "num_q_heads ({}) + 2 * num_kv_heads ({}) exceeds the fused width ({})",
            num_q_heads,
            num_kv_heads_val,
            q_width);
        TT_FATAL(
            q_logical_width % untied_sections == 0,
            "Unsupported input shape: fused width ({}) is not divisible by num_q_heads ({}) + 2 * num_kv_heads ({})",
            q_logical_width,
            num_q_heads,
            2 * num_kv_heads_val);
        head_dim = static_cast<uint32_t>(q_logical_width / untied_sections);
    }

    // The fused TM kernels only move whole TILE_WIDTH-wide tiles per head, so a sub-tile
    // head dim both loses the tail of every head and mis-spaces the per-head tile
    // addresses silently. Interleaved inputs take the row-major route instead.
    if (head_dim % tt::constants::TILE_WIDTH != 0) {
        TT_FATAL(
            !input_tensor_q.is_sharded() && !(input_tensor_kv.has_value() && input_tensor_kv.value().is_sharded()),
            "nlp_create_qkv_heads does not support head_dim ({}) with sharded inputs: the sharded TM kernels take a "
            "sub-tile head_dim silently wrong. Use an interleaved input, or a head_dim that is a multiple of "
            "TILE_WIDTH ({}).",
            head_dim,
            tt::constants::TILE_WIDTH);
        TT_FATAL(
            !optional_output_tensors.has_value(),
            "nlp_create_qkv_heads does not support output_tensors with a sub-tile head_dim ({}); the row-major "
            "fallback allocates its own outputs",
            head_dim);
        return nlp_create_qkv_heads_subtile_head_dim(
            input_tensor_q,
            input_tensor_kv,
            num_q_heads,
            num_kv_heads_val,
            head_dim,
            transpose_k_heads,
            kv_tied,
            memory_config);
    }

    return ttnn::prim::nlp_create_qkv_heads(
        input_tensor_q,
        input_tensor_kv,
        num_q_heads,
        num_kv_heads,
        head_dim,
        transpose_k_heads,
        kv_tied,
        memory_config,
        optional_output_tensors,
        std::nullopt,
        runtime_rows);
}

std::tuple<Tensor, Tensor> nlp_create_q_heads_split(
    const Tensor& input,
    uint32_t num_heads,
    uint32_t split_head_dim,
    const std::optional<MemoryConfig>& memory_config,
    const bool runtime_rows) {
    TT_FATAL(input.logical_shape().rank() == 4 && num_heads > 0, "Expected rank-4 input and nonzero head count");
    TT_FATAL(
        input.logical_shape()[3] == input.padded_shape()[3] && input.logical_shape()[3] % num_heads == 0,
        "Q width must be unpadded and divisible by num_heads");
    auto [first, second, unused] = ttnn::prim::nlp_create_qkv_heads(
        input,
        std::nullopt,
        num_heads,
        0,
        input.logical_shape()[3] / num_heads,
        false,
        false,
        memory_config,
        std::nullopt,
        split_head_dim,
        runtime_rows);
    return {first, second};
}

}  // namespace ttnn::experimental
