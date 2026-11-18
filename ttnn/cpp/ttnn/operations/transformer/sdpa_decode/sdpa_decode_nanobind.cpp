// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "sdpa_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::transformer {

namespace {
// return_lse=False returns the output tensor (the op is unchanged); True returns (output, lse).
std::variant<ttnn::Tensor, std::tuple<ttnn::Tensor, ttnn::Tensor>> paged_flash_mla_decode_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool return_lse,
    uint32_t spec_multi_pos_tiles,
    bool spec_multi_pos_packed) {
    if (return_lse) {
        return ttnn::transformer::paged_flash_multi_latent_attention_decode_with_lse(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            head_dim_v,
            page_table_tensor,
            is_causal,
            attn_mask,
            cur_pos_tensor,
            attention_sink,
            scale,
            sliding_window_size,
            memory_config,
            program_config,
            compute_kernel_config,
            spec_multi_pos_tiles,
            spec_multi_pos_packed);
    }
    return ttnn::transformer::paged_flash_multi_latent_attention_decode(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        head_dim_v,
        page_table_tensor,
        is_causal,
        attn_mask,
        cur_pos_tensor,
        attention_sink,
        scale,
        sliding_window_size,
        memory_config,
        program_config,
        compute_kernel_config,
        spec_multi_pos_tiles,
        spec_multi_pos_packed);
}
}  // namespace

void bind_sdpa_decode(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        A version of scaled dot product attention specifically for decode.
        The implementation is Flash-Decode and it currently only supports MQA on decoding single token.


        Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension.


        Shape identifiers used below: `b` is the batch size, `nh` the number of Q heads on this device
        (`pnh` the same count padded up to the tile height), `nkv` the number of KV heads, `s` the KV
        sequence length and `dh` the head dimension.


        Args:
            input_tensor_q (ttnn.Tensor): the input tensor [1 x b x nh x dh]
            input_tensor_k (ttnn.Tensor): the input tensor [b x nkv x   s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor [b x nkv x   s x dh]


        Keyword args:
            is_causal (bool): whether the attention is is_causal. Defaults to `True`.
            attn_mask (ttnn.Tensor, optional): the input tensor [1 x b x nh x s]; `nh` must match Q's.
                `s` must be a multiple of `k_chunk_size`: unpaged decode defaults it to
                `min(512, largest power-of-two divisor of s)`. Paged decode uses 32 when `program_config`
                is omitted; a custom config for paged non-causal attention must set a positive
                `k_chunk_size`. Only valid with `is_causal=False`. Defaults to `None`.
            cur_pos (List of int, optional): list of integers of length b. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            cur_pos_tensor (ttnn.Tensor, optional): [b] tensor of integers of length b. Defaults to `None`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            sliding_window_size (int, optional): The size of sliding window for sliding window attention. Defaults to `None`.
            share_cache (bool, optional): KV cache sharing across batch for decode. `True` enables, `False` disables. Defaults to `None` (distinct program-cache key from explicit `False`).


        Returns:
            ttnn.Tensor: the output tensor [1 x b x pnh x dh].


        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        "If a position is given as (-1), compute for the corresponding index in the batch is skipped."
        )doc";

    ttnn::bind_function<"scaled_dot_product_attention_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::scaled_dot_product_attention_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal").noconvert() = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos") = nb::cast(std::vector<uint32_t>()),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("share_cache") = nb::none());

    ttnn::bind_function<"paged_scaled_dot_product_attention_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::paged_scaled_dot_product_attention_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v").noconvert(),
        nb::arg("page_table_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal") = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        // Read the K/V cache through a different (block_size, num_kv_heads, head_dim) view
        // than its declared shape; needed for vLLM's shared kv-cache groups. Same type as
        // chunked prefill SDPA. See ttnn.PagedCacheGeometryOverride.
        nb::arg("paged_cache_geometry") = nb::none(),
        // cache_position_modulo (in tokens) treats the cache as a circular buffer:
        // every page_table lookup uses cur_pos % cache_position_modulo. Required when
        // the cache is sized for sliding-window-only allocation (vLLM
        // SlidingWindowSpec); without it, positions past the bounded capacity collapse
        // onto physical block 0 and silently corrupt the cache. Must be a multiple of
        // (effective) block_size and >= sliding_window_size. Paged-mode only.
        nb::arg("cache_position_modulo") = nb::none(),
        // spec_multi_pos_tiles = Tg (0 = off) folds Tg speculative-decode candidates that
        // share one KV cache onto a single batch row, so that row's KV is read from DRAM
        // once instead of Tg times. Q is [1, B, Tg*32, DH] (batch b's row-tile j is
        // candidate b*Tg + j, its valid q-heads in rows j*32..), page_table has B rows, and
        // cur_pos_tensor holds B*Tg positions, ascending within each group of Tg; candidate
        // b*Tg+j attends to [0, cur_pos[b*Tg+j]] inclusive. Output is [1, B, Tg*32, DH] —
        // byte-identical to the legacy B*Tg-row output [1, B*Tg, 32, DH]. Requires causal +
        // paged + num_kv_heads==1, unsharded bf16 TILE Q, no sliding window, no MLA.
        nb::arg("spec_multi_pos_tiles") = 0u);

    ttnn::bind_function<"flash_multi_latent_attention_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::flash_multi_latent_attention_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v") = nb::none(),
        nb::arg("head_dim_v").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal").noconvert() = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos") = nb::cast(std::vector<uint32_t>()),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    const auto* const paged_mla_doc =
        R"doc(
        Paged flash decode for multi-latent attention (MLA): V is the first `head_dim_v` columns of the K cache,
        or a separate paged V cache `input_tensor_v` addressed by the same page table.

        Shape identifiers used below: `b` is the batch size (the page table's rows), `nh` the number of Q heads
        (`pnh` padded up to the tile height), `dh` the K head dimension, `block` the cache block size.

        Args:
            input_tensor_q (ttnn.Tensor): Q [1 x b x nh x dh]; DRAM interleaved or height-sharded in L1.
            input_tensor_k (ttnn.Tensor): paged K cache [num_blocks x 1 x block x dh].
            input_tensor_v (ttnn.Tensor, optional): paged V cache [num_blocks x 1 x block x head_dim_v].
            head_dim_v (int): the V head dimension.
            page_table_tensor (ttnn.Tensor): int32 [b x max_blocks_per_user].

        Keyword args:
            is_causal (bool): Defaults to `True`.
            attn_mask (ttnn.Tensor, optional): only with `is_causal=False`. Defaults to `None`.
            cur_pos_tensor (ttnn.Tensor, optional): int32 [b]: user i attends to cache positions
                [0, cur_pos[i]]; -1 skips the user. Required when causal. Defaults to `None`.
            attention_sink (ttnn.Tensor, optional): Defaults to `None`.
            scale (float, optional): Defaults to `None` (1 / sqrt(dh)).
            sliding_window_size (int, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): of the output. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            return_lse (bool): also return the log-sum-exp of every (user, head). Defaults to `False`, which
                leaves the op (programs and results) unchanged. Needs one KV head, no attention sink and a TILE
                layout Q.
            spec_multi_pos_tiles (int): T > 0 folds T candidate positions (a speculative verify) onto each page
                table row, which then scans its KV once for all of them: Q [1 x b x T*32 x dh] DRAM interleaved,
                candidate j's heads in its own 32-row tile j, and cur_pos_tensor int32 [b * T], ascending within a
                row's group except for -1 entries (a candidate with no visible key: output 0, lse -1e30). Needs
                `input_tensor_v`. Defaults to 0 (off).
            spec_multi_pos_packed (bool): with spec_multi_pos_tiles even and candidates of at most 16 heads, compute
                two candidates per tile (candidate 2t + h in rows 16h..16h+15 of tile t: half the QK / PV work) while
                Q, the output and lse keep a tile per candidate; output rows 16..31 of a candidate's tiles repeat rows
                0..15 and its lse rows 16..31 are -1e30. Defaults to `False`.

        Returns:
            ttnn.Tensor: the output [1 x b x nh x head_dim_v], or with `return_lse=True` the tuple
            (output, lse):

            - lse: float32 [1 x b x nh x 1], TILE layout: `memory_config` when that is interleaved, otherwise
              interleaved in the same buffer type as the output (L1 for an L1-sharded output). Each head's value is in column 0 of its tile row (the tile padding is 0), so
              it broadcasts against the output along the last dim with ttnn eltwise ops. lse = m + log(l) in the kernel's logit space `scale * q.k`:
              the log of the softmax denominator sum_k exp(scale * q.k) over the user's visible keys.
            - A user with cur_pos = -1 (no visible key) gets output rows 0 and lse = -1e30 (finite).

            Two partial attentions (o_a, lse_a), (o_b, lse_b) over disjoint key sets merge exactly as
            w_x = exp(lse_x - max(lse_a, lse_b)), o = (w_a * o_a + w_b * o_b) / (w_a + w_b).
        )doc";

    ttnn::bind_function<"paged_flash_multi_latent_attention_decode", "ttnn.transformer.">(
        mod,
        paged_mla_doc,
        &paged_flash_mla_decode_wrapper,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v") = nb::none(),
        nb::arg("head_dim_v").noconvert(),
        nb::arg("page_table_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal").noconvert() = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("return_lse") = false,
        nb::arg("spec_multi_pos_tiles") = 0u,
        nb::arg("spec_multi_pos_packed") = false);
}
}  // namespace ttnn::operations::transformer
