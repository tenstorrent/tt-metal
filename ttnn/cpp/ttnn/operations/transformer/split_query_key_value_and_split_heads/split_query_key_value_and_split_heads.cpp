// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads.hpp"

#include "ttnn/operations/core/core.hpp"

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads.hpp"

#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ttnn::operations::transformer::detail {
std::tuple<Tensor, Tensor, Tensor> reshape_outputs_of_split_query_key_value_and_split_heads(
    const std::tuple<Tensor, Tensor, Tensor>& outputs,
    const uint32_t sequence_size,
    const uint32_t sequence_size_padded,
    const bool transpose_key) {
    auto [query, key, value] = outputs;

    auto batch_size = query.logical_shape()[0];
    auto num_heads = query.logical_shape()[1];
    auto head_size = query.logical_shape()[-1];
    auto head_size_padded = query.padded_shape()[-1];

    auto num_kv_heads = value.logical_shape()[1];

    query = ttnn::reshape(
        query,
        ttnn::Shape({batch_size, num_heads, sequence_size, head_size}),
        ttnn::Shape({batch_size, num_heads, sequence_size_padded, head_size_padded}));

    if (transpose_key) {
        key = ttnn::reshape(
            key,
            ttnn::Shape({batch_size, num_kv_heads, head_size, sequence_size}),
            ttnn::Shape({batch_size, num_kv_heads, head_size_padded, sequence_size_padded}));
    } else {
        key = ttnn::reshape(
            key,
            ttnn::Shape({batch_size, num_kv_heads, sequence_size, head_size}),
            ttnn::Shape({batch_size, num_kv_heads, sequence_size_padded, head_size_padded}));
    }

    value = ttnn::reshape(
        value,
        ttnn::Shape({batch_size, num_kv_heads, sequence_size, head_size}),
        ttnn::Shape({batch_size, num_kv_heads, sequence_size_padded, head_size_padded}));
    return {query, key, value};
}
}  // namespace ttnn::operations::transformer::detail

namespace ttnn::transformer {

// Implementation notes — see issue #41718 and the matrix test in
// tests/ttnn/unit_tests/gtests/test_split_qkv_heads_matrix.cpp.
//
// QKV LAYOUT CONVENTION (implicit, not validated):
//
// Two QKV input layouts exist in the framework:
//
//   CONCATENATED — what nn.Linear(d, 3d) produces by default. Per (batch, seq) row:
//     [Q_h0, Q_h1, ..., Q_h(n_q-1), K_h0, ..., K_h(n_kv-1), V_h0, ..., V_hn]
//
//   GROUPED — what the existing sharded `experimental::create_qkv_heads` reader
//   expects. Per (batch, seq) row, with n = n_q / n_kv Q heads per KV group:
//     [Q_g0_h0,...,Q_g0_h(n-1), K_g0, V_g0, Q_g1_h0,..., K_g1, V_g1, ...]
//
// The two paths in this op handle different layouts:
//   - The interleaved path (`experimental::nlp_create_qkv_heads`) reads Q, then K,
//     then V — i.e. CONCATENATED layout.
//   - The sharded path (`experimental::create_qkv_heads`, used below) reads
//     [Q heads per group, K, V] per group — i.e. GROUPED layout.
//
// Both production callers of the sharded path explicitly repack their nn.Linear
// QKV weights into GROUPED layout to match what the kernel expects:
//   - SD U-Net cross-attention: see `concatenate_qkv()` in
//     models/demos/vision/generative/stable_diffusion/wormhole/tt/ttnn_functional_cross_attention.py:81-117
//   - ViT WH: see the `query_key_value` weight construction in
//     models/demos/vision/classification/vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py:559-566
//
// The "19-month silent corruption" claim in #41718 turned out to be wrong for
// both production models — they were correctly using grouped weights the whole
// time. The actual original-bug case from #41526 is tt-mlir greedy optimizer at
// opt_level=2 feeding CONCATENATED layout to the sharded kernel; this is reliably
// reproduced by Cell 13 of the matrix test (PCC ~ 0.1 vs CPU reference).
//
// ADDITIONAL CONSTRAINT (the `sequence_size_padded == sequence_size` FATAL below):
//
// The sharded reader's address arithmetic at stride `block_wt_size_bytes` per seq
// tile assumes a tile-aligned sequence dimension. For non-tile-aligned seq lengths
// (e.g. ViT seq=197) it produces silent corruption regardless of layout. Reject
// those cases at validation time so callers see a clear error instead of garbage.
//
// LIMITATION OF THE CURRENT FIX:
//
// The refined FATAL only catches non-tile-aligned cases. A caller that feeds
// CONCATENATED layout with a tile-aligned sequence length (e.g. seq=64 like the
// SD U-Net case) is not caught and the kernel silently corrupts the output.
// Cell 13 of the matrix test demonstrates this. The proper long-term fix is an
// explicit `qkv_layout` parameter on this op so the convention is part of the API
// and not a kernel-comment-level convention. Tracked in the comments of #41718.
std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_key,
    const std::optional<MemoryConfig>& memory_config,
    const bool use_falcon7b_backend) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto& padded_input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 3, "Invalid input tensor: expected 3 dimensions, but found {}.", input_shape.rank());

    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Invalid layout: input tensor must use TILE_LAYOUT, but found {}.",
        static_cast<int>(input_tensor.layout()));

    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Invalid storage type: input tensor must be on a device, but found {}.",
        static_cast<int>(input_tensor.storage_type()));

    const uint32_t sequence_size = input_shape[1];
    const uint32_t sequence_size_padded = padded_input_shape[1];

    if (use_falcon7b_backend) {
        TT_FATAL(
            num_kv_heads.has_value() && !input_tensor_kv.has_value(),
            "Invalid configuration: use_falcon7b_backend requires num_kv_heads to be set and no separate KV tensor.");

        TT_FATAL(
            !transpose_key,
            "Invalid configuration: Transpose is set to true, but this is not supported when separate num_kv_heads is "
            "used.");

        uint32_t qkv_heads_times_head_dim = input_shape[2];
        uint32_t qkv_heads_times_head_dim_padded = padded_input_shape[2];
        auto head_size = qkv_heads_times_head_dim / (num_heads + (num_kv_heads.value() * 2));
        auto padded_head_size = qkv_heads_times_head_dim_padded / (num_heads + (num_kv_heads.value() * 2));

        TT_FATAL(
            head_size % TILE_SIZE == 0,
            "Invalid head size: {} is not a multiple of tile size {}. Update the preceding matmul to include padding "
            "in the weights.",
            head_size,
            TILE_SIZE);

        TT_FATAL(
            padded_head_size == head_size,
            "Invalid padding: Head size {} should not have additional tile padding, but padded size is {}.",
            head_size,
            padded_head_size);

        const auto input_4d = ttnn::experimental::view(
            input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});

        auto outputs = ttnn::experimental::nlp_create_qkv_heads_falcon7b(
            input_4d, memory_config.value_or(input_tensor.memory_config()));
        return ttnn::operations::transformer::detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)},
            sequence_size,
            sequence_size_padded,
            transpose_key);
    }

    uint32_t hidden_dim_padded = 0, hidden_dim = 0;
    if (input_tensor_kv.has_value()) {
        const auto input_shape_kv = input_tensor_kv.value().logical_shape();
        TT_FATAL(
            input_shape_kv[0] == input_shape[0],
            "Dimension mismatch: KV tensor batch dimension ({}) must match Q tensor batch dimension ({}).",
            input_shape_kv[0],
            input_shape[0]);

        TT_FATAL(
            input_shape_kv[1] == input_shape[1],
            "Dimension mismatch: KV tensor sequence length ({}) must match Q tensor sequence length ({}).",
            input_shape_kv[1],
            input_shape[1]);

        TT_FATAL(
            (input_shape[2] % num_heads) == 0,
            "Query dimension ({}) must be divisible by num_heads ({}).",
            input_shape[2],
            num_heads);

        TT_FATAL(
            (input_shape_kv[2] % (2 * num_kv_heads.value_or(num_heads))) == 0,
            "KV dimension ({}) must be divisible by 2 * num_kv_heads ({}).",
            input_shape_kv[2],
            (2 * num_kv_heads.value_or(num_heads)));

        TT_FATAL(
            (input_shape_kv[2] / (2 * num_kv_heads.value_or(num_heads))) == (input_shape[2] / num_heads),
            "Dimension mismatch: KV head size ({}) must be equal to query head size ({}).",
            (input_shape_kv[2] / (2 * num_kv_heads.value_or(num_heads))),
            (input_shape[2] / num_heads));

        hidden_dim = input_shape[2];
        hidden_dim_padded = padded_input_shape[2];
    } else {
        hidden_dim = input_shape[2];
        hidden_dim_padded = padded_input_shape[2];
    }

    // For separate Q and KV tensors: hidden_dim is Q's dimension, so head_size = hidden_dim / num_heads
    // For fused QKV tensor: hidden_dim contains Q+K+V, so head_size = hidden_dim / (num_heads + 2*num_kv_heads)
    uint32_t head_size_divisor =
        input_tensor_kv.has_value() ? num_heads : (num_heads + (2 * num_kv_heads.value_or(num_heads)));
    uint32_t head_size = hidden_dim / head_size_divisor;
    uint32_t padded_head_size = hidden_dim_padded / head_size_divisor;
    TT_FATAL(
        head_size % tt::constants::TILE_WIDTH == 0,
        "Invalid head size: {}. The head size must be a multiple of the tile width ({}). Please adjust the dimensions "
        "accordingly.",
        head_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        padded_head_size == head_size,
        "Padding error: Head size {} should not include additional tile padding, but padded head size was found to be "
        "{}. Ensure that no extra padding is applied.",
        head_size,
        padded_head_size);

    if (input_tensor.is_sharded()) {
        // Issue #41526 / #41718: the sharded `create_qkv_heads` reader walks address
        // arithmetic that assumes a tile-aligned sequence dimension. For non-tile-aligned
        // seq lengths (e.g. ViT seq=197) this produces silent corruption (PCC ~0.08).
        // For tile-aligned seq lengths the reader works correctly and the SD U-Net
        // cross-attention path has been relying on it since 2024-08. PR #41550 added a
        // blanket FATAL on any sharded input which broke the working SD U-Net path
        // (#41718). Re-allow tile-aligned sharded inputs and only FATAL when the seq
        // length would actually trigger the corruption.
        TT_FATAL(
            sequence_size_padded == sequence_size,
            "Sharded input is not supported for split_query_key_value_and_split_heads when the "
            "sequence length is not tile-aligned (sequence length = {}, padded to {}). The "
            "sharded reader's address arithmetic assumes tile-aligned source offsets and produces "
            "corrupted output (PCC ~0.08) otherwise. The caller should unshard the input before "
            "calling this op, or pad the sequence dimension to a multiple of {}.",
            sequence_size,
            sequence_size_padded,
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            !input_tensor_kv.has_value(),
            "Invalid operation: KV tensor should not be provided when the input tensor is sharded. "
            "The KV tensor is only used in non-sharded configurations.");
        const auto input_tensor_4d_sharded = ttnn::experimental::view(
            input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});
        return ttnn::operations::transformer::detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            ttnn::experimental::create_qkv_heads(
                input_tensor_4d_sharded,
                num_heads,
                num_kv_heads.value_or(num_heads),
                transpose_key,
                memory_config.value_or(input_tensor.memory_config())),
            sequence_size,
            sequence_size_padded,
            transpose_key);
    }
    const auto input_tensor_4d = ttnn::experimental::view(
        input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});
    std::optional<Tensor> input_tensor_kv_4d = std::nullopt;
    if (input_tensor_kv.has_value()) {
        auto padded_input_shape_kv = input_tensor_kv.value().padded_shape();
        input_tensor_kv_4d = ttnn::experimental::view(
            input_tensor_kv.value(),
            ttnn::Shape{padded_input_shape_kv[0], 1, padded_input_shape_kv[1], padded_input_shape_kv[2]});
    }
    const auto outputs = ttnn::experimental::nlp_create_qkv_heads(
        input_tensor_4d,
        input_tensor_kv_4d,
        num_heads,
        num_kv_heads.value_or(num_heads),
        transpose_key,
        memory_config.value_or(input_tensor.memory_config()));
    return ttnn::operations::transformer::detail::reshape_outputs_of_split_query_key_value_and_split_heads(
        outputs, sequence_size, sequence_size_padded, transpose_key);
}

}  // namespace ttnn::transformer
