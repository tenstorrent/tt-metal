// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "sdpa_decode.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::transformer {

void bind_sdpa_decode(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Scaled dot product attention for decode (single-token generation).

        Implements Flash-Decode, parallelizing over batch ``b``, query heads ``nh``, and
        key-value heads ``nkv`` (when possible). The op parallelizes over the KV sequence
        length ``s`` across a group of cores associated with a batch/kv head
        ``max_cores_per_head_batch``, then uses tree reduction to compute softmax correction.
        Supports MQA (Multi-Query Attention) and GQA (Grouped-Query Attention).

        Accepts a ``SDPAProgramConfig`` which specifies the grid size and chunk tiles in the
        K/V/Mask sequence lengths.


        Supported data types:
            - Q: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``. When using GQA (nkv > 1), Q must be ``bfloat16``.
            - K: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``.
            - V: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``. Must match K's dtype.
            - attn_mask: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``.
            - cur_pos_tensor: ``int32``.
            - attention_sink: ``bfloat16`` only.


        Memory configurations:
            - Q: DRAM interleaved or L1 ``HEIGHT_SHARDED`` (sharded by batch, shard shape ``[padded_num_heads, dh]``).
            - K: DRAM interleaved only. Must be ``TILE`` layout.
            - V: DRAM interleaved only. Must be ``TILE`` layout.
            - attn_mask: DRAM interleaved. Must be ``TILE`` layout.
            - cur_pos_tensor: DRAM interleaved or L1 ``HEIGHT_SHARDED``. Must be ``ROW_MAJOR`` layout.
            - attention_sink: DRAM interleaved only. Must be ``TILE`` layout.
            - output: DRAM interleaved or L1 ``HEIGHT_SHARDED``. GQA (nkv > 1) does not support sharded output.


        Args:
            input_tensor_q (ttnn.Tensor): Query tensor with shape ``[1, b, nh, dh]``.
                Layout: ``TILE`` or ``ROW_MAJOR`` (if ``ROW_MAJOR``, the kernel tilizes internally).
            input_tensor_k (ttnn.Tensor): Key tensor with shape ``[b, nkv, s, dh]``.
                Layout: ``TILE``. Sequence length ``s`` must be divisible by ``k_chunk_size``.
            input_tensor_v (ttnn.Tensor): Value tensor with shape ``[b, nkv, s, dh]``.
                Layout: ``TILE``. Must have same dtype and shape constraints as K.


        Keyword args:
            is_causal (bool): Whether to apply causal masking. When ``True``, positions beyond
                ``cur_pos`` are masked out. When ``False``, ``attn_mask`` must be provided.
                Defaults to ``True``.
            attn_mask (ttnn.Tensor, optional): Attention mask tensor with shape ``[b, nh, 1, s]``.
                Required when ``is_causal=False``. Must not be provided when ``is_causal=True``.
                Use large negative values (e.g. ``torch.finfo(torch.float32).min``) to mask positions.
                Defaults to ``None``.
            cur_pos (List[int], optional): List of current positions of length ``b``. Each value
                indicates the current decode position for that batch element. A value of ``-1``
                (i.e. ``UINT32_MAX``) skips computation for that batch element. Mutually exclusive
                with ``cur_pos_tensor``. Defaults to ``[]``.
            cur_pos_tensor (ttnn.Tensor, optional): Tensor of current positions with shape ``[b]``,
                dtype ``int32``, layout ``ROW_MAJOR``. Alternative to ``cur_pos`` list. A value of
                ``-1`` skips computation for that batch element. Defaults to ``None``.
            attention_sink (ttnn.Tensor, optional): Attention sink tensor with shape
                ``[padded_num_heads, 32]``, dtype ``bfloat16``, layout ``TILE``, in DRAM.
                Used for streaming/infinite-length attention where initial tokens act as sinks.
                The sink values should be pre-divided by the scale factor. Defaults to ``None``.
            scale (float, optional): Scaling factor applied to QK^T. Typically ``1/sqrt(dh)``.
                If ``None``, defaults to ``1/sqrt(dh)``. Defaults to ``None``.
            sliding_window_size (int, optional): Size of the sliding window for local attention.
                When set, only the most recent ``sliding_window_size`` tokens are attended to
                (positions ``[max(0, cur_pos+1-window), cur_pos]``). Must be used with causal mode.
                Defaults to ``None``.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Supports
                DRAM interleaved or L1 ``HEIGHT_SHARDED``. Defaults to ``None`` (DRAM interleaved).
            program_config (SDPAProgramConfig, optional): Specifies compute grid size and chunk
                sizes. Fields: ``compute_with_storage_grid_size`` (tuple), ``q_chunk_size`` (int),
                ``k_chunk_size`` (int, must be power of 2 and multiple of 32, max 512),
                ``exp_approx_mode`` (bool). Defaults to ``None`` (auto-configured), ``max_cores_per_head_batch`` (int, optional). Defaults to 16.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel
                configuration (math fidelity, fp32 accumulation, etc.). Defaults to ``None``.


        Returns:
            ttnn.Tensor: Output tensor with shape ``[1, b, pnh, dh]`` where ``pnh`` is the
            padded number of heads (rounded to nearest power of 2 and multiple of 32).


        Limitations:
            - GQA with sharded output is not supported. Use DRAM output for GQA (nkv > 1).
            - Attention sink only supports ``bfloat16`` dtype and DRAM memory.
            - ``k_chunk_size`` must be a power of 2, a multiple of 32, and must divide the
              sequence length evenly. Maximum auto-calculated chunk size is 512.
            - Causal mode (``is_causal=True``) and ``attn_mask`` are mutually exclusive.
            - Non-causal mode (``is_causal=False``) requires ``attn_mask`` and a ``k_chunk_size > 0``.
            - Maximum tree reduction depth is 6 rounds (up to 64 cores per head).
            - ``cur_pos`` list and ``cur_pos_tensor`` are mutually exclusive; provide only one.
            - Q unpadded heads must be divisible by K heads for GQA (``nh % nkv == 0``).
            - Share cache (K/V batch=1 shared across Q batch) requires Q batch divisible by KV batch.
            - When using ``fp32_dest_acc_en=True`` in compute config, the DST accumulator size is
              reduced from 8 to 4, which limits the maximum dynamic chunk size and can reduce throughput.
            - Output untilization uses the fast ``pack_untilize`` path only when
              ``Sq_chunk_t * vDHt <= 8`` tiles; otherwise the slower ``untilize`` path is used.
            - Half-tile optimization (16x32 tiles) is automatically enabled when ``is_causal=True``,
              ``nh <= 16``, and Q dtype is ``bfloat16``.


        Example (causal decode with cur_pos_tensor):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s, dh = 4, 32, 8, 8192, 128
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos_tensor = ttnn.from_torch(
                torch.tensor([4096, 4096, 4096, 4096]), dtype=ttnn.int32, device=device)

            output = ttnn.transformer.scaled_dot_product_attention_decode(
                Q, K, V,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=32, k_chunk_size=128),
            )


        Example (non-causal decode with attention mask):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s, dh = 1, 64, 8, 2048, 128
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Mask: 0 for attended, large negative for masked positions
            mask = torch.zeros(b, nh, 1, s)
            mask[:, :, :, s // 2:] = torch.finfo(torch.float32).min
            tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b,
                                      layout=ttnn.TILE_LAYOUT, device=device,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

            output = ttnn.transformer.scaled_dot_product_attention_decode(
                Q, K, V,
                is_causal=False,
                attn_mask=tt_mask,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=32, k_chunk_size=128),
            )


        Example (causal decode with sliding window):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s, dh = 8, 8, 1, 32768, 128
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 16384, dtype=torch.int32), dtype=ttnn.int32, device=device)

            # Only attend to last 4096 tokens in causal mode
            output = ttnn.transformer.scaled_dot_product_attention_decode(
                Q, K, V,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                sliding_window_size=4096,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 6),
                    q_chunk_size=32, k_chunk_size=256),
            )


        Example (HEIGHT_SHARDED Q and output):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s, dh = 4, 32, 8, 8192, 128
            padded_nh = 32  # nearest power of 2 and multiple of 32
            scale = 1.0 / (dh ** 0.5)

            # HEIGHT_SHARDED: one shard per batch, shard shape [padded_nh, dh]
            shard_grid = ttnn.CoreRangeSet({ttnn.num_to_corerange(b)})
            shard_spec = ttnn.ShardSpec(shard_grid, (padded_nh, dh),
                                        ttnn.ShardOrientation.ROW_MAJOR)
            sharded_memcfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=sharded_memcfg)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 4096, dtype=torch.int32), dtype=ttnn.int32, device=device)

            output = ttnn.transformer.scaled_dot_product_attention_decode(
                Q, K, V,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                memory_config=sharded_memcfg,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=32, k_chunk_size=128),
            )

        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::scaled_dot_product_attention_decode,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::vector<uint32_t>& cur_pos,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<SDPAProgramConfig>& program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    is_causal,
                    attn_mask,
                    cur_pos,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
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
            nb::arg("compute_kernel_config") = nb::none()});

    const auto* paged_doc =
        R"doc(
        Paged scaled dot product attention for decode (single-token generation).

        Same as ``scaled_dot_product_attention_decode`` but uses paged KV cache via a
        ``page_table_tensor`` that maps logical sequence positions to physical page indices.
        This enables non-contiguous KV cache storage for dynamic memory management of
        variable-length sequences.

        The page table maps logical block indices to physical block indices in the KV cache.
        K and V tensors represent the physical block pool, and the page table tells the kernel
        which physical blocks to read for each batch element's logical sequence.


        Supported data types:
            - Q: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``. GQA (nkv > 1) requires Q as ``bfloat16``.
            - K, V: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``. V must match K's dtype.
            - page_table_tensor: ``int32`` (DRAM interleaved) or ``uint16`` (L1 ``HEIGHT_SHARDED``).
            - cur_pos_tensor: ``int32``.
            - attn_mask: ``bfloat16``, ``bfloat8_b``, ``bfloat4_b``.
            - attention_sink: ``bfloat16`` only.


        Memory configurations:
            - Q: DRAM interleaved or L1 ``HEIGHT_SHARDED`` (shard shape ``[padded_num_heads, dh]``).
            - K, V: DRAM interleaved only. ``TILE`` layout. Shape represents physical block pool:
              ``[num_blocks, nkv, block_size, dh]``.
            - page_table_tensor: DRAM interleaved (``int32``) or L1 ``HEIGHT_SHARDED`` (``uint16``).
              Must be ``ROW_MAJOR`` layout. Shape: ``[b, max_num_blocks_per_seq]`` where
              ``max_num_blocks_per_seq = max_seq_len / block_size``.
            - cur_pos_tensor: DRAM interleaved or L1 ``HEIGHT_SHARDED``. Must be ``ROW_MAJOR`` layout.
            - attention_sink: DRAM interleaved only. Must be ``TILE`` layout.
            - output: DRAM interleaved or L1 ``HEIGHT_SHARDED``. GQA does not support sharded output.


        Args:
            input_tensor_q (ttnn.Tensor): Query tensor ``[1, b, nh, dh]``.
                ``TILE`` or ``ROW_MAJOR`` layout.
            input_tensor_k (ttnn.Tensor): Key cache block pool ``[num_blocks, nkv, block_size, dh]``.
                ``TILE`` layout, DRAM. ``block_size`` is the page size (e.g. 64).
            input_tensor_v (ttnn.Tensor): Value cache block pool ``[num_blocks, nkv, block_size, dh]``.
                ``TILE`` layout, DRAM. Must match K's block_size and dtype.
            page_table_tensor (ttnn.Tensor): Page table mapping logical to physical blocks.
                Shape: ``[b, max_num_blocks_per_seq]``. ``ROW_MAJOR`` layout.
                Dtype: ``int32`` when in DRAM, ``uint16`` when ``HEIGHT_SHARDED`` in L1.
                Each entry maps a logical block index to the physical block index in K/V.


        Keyword args:
            is_causal (bool): Whether to apply causal masking. Defaults to ``True``.
            attn_mask (ttnn.Tensor, optional): Mask ``[b, nh, 1, s]``.
                Required when ``is_causal=False``. Defaults to ``None``.
            cur_pos_tensor (ttnn.Tensor, optional): Current positions ``[b]``, dtype ``int32``,
                ``ROW_MAJOR``. Required for causal paged attention. A value of ``-1`` skips
                that batch element. Defaults to ``None``.
            attention_sink (ttnn.Tensor, optional): Sink tensor ``[padded_num_heads, 32]``,
                ``bfloat16``, DRAM. Defaults to ``None``.
            scale (float, optional): Scaling factor for QK^T. Defaults to ``1/sqrt(dh)``.
            sliding_window_size (int, optional): Sliding window size for local attention.
                Defaults to ``None``.
            memory_config (ttnn.MemoryConfig, optional): Output memory config.
                Defaults to ``None`` (DRAM interleaved).
            program_config (SDPAProgramConfig, optional): Grid and chunk sizes. ``k_chunk_size``
                can be 0 for paged attention (auto-determined by kernel). Defaults to ``None``.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Defaults to ``None``.


        Returns:
            ttnn.Tensor: Output ``[1, b, pnh, dh]``.


        Limitations:
            - Share cache (K/V batch=1 shared across Q batch) is **not supported** with paged attention.
            - ``cur_pos`` list is not available for paged attention; use ``cur_pos_tensor`` instead.
            - GQA with sharded output is not supported.
            - When ``block_size < 32``, a block padding mask is applied per chunk, adding overhead.
              The kernel converts ``cur_pos`` from original sequence space to padded tile space.
            - Non-causal: ``page_table.shape[1] * block_size`` must be divisible by ``k_chunk_size``.
            - Causal mode and ``attn_mask`` are mutually exclusive.
            - Max tree reduction depth: 6 rounds (up to 64 cores per head).
            - Attention sink: ``bfloat16`` and DRAM only.
            - Q unpadded heads must be divisible by K heads for GQA (``nh % nkv == 0``).
            - K shape[2] (block_size) must equal V shape[2].


        Example (causal paged attention):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, dh = 8, 16, 4, 128
            block_size, max_seq_len = 64, 4096
            max_num_blocks = max_seq_len // block_size
            num_physical_blocks = b * max_num_blocks  # total blocks in pool
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            # K/V are physical block pools, not per-batch
            K = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, dh),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, dh),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Page table: each row maps batch element's logical blocks to physical blocks
            page_table = torch.randint(0, num_physical_blocks,
                                       (b, max_num_blocks), dtype=torch.int32)
            tt_page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, device=device)

            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 2048, dtype=torch.int32), dtype=ttnn.int32, device=device)

            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                Q, K, V, tt_page_table,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 2),
                    q_chunk_size=32, k_chunk_size=128),
            )


        Example (non-causal paged attention with mask):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, dh = 4, 8, 1, 128
            block_size, max_seq_len = 64, 2048
            max_num_blocks = max_seq_len // block_size
            num_physical_blocks = b * max_num_blocks
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, dh),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, dh),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tt_page_table = ttnn.from_torch(
                torch.randint(0, num_physical_blocks,
                              (b, max_num_blocks), dtype=torch.int32),
                dtype=ttnn.int32, device=device)

            mask = torch.zeros(b, nh, 1, max_seq_len)
            mask[:, :, :, max_seq_len // 2:] = torch.finfo(torch.float32).min
            tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b,
                                      layout=ttnn.TILE_LAYOUT, device=device,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                Q, K, V, tt_page_table,
                is_causal=False,
                attn_mask=tt_mask,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    q_chunk_size=32, k_chunk_size=64),
            )

        )doc";

    using PagedOperationType = decltype(ttnn::transformer::paged_scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::paged_scaled_dot_product_attention_decode,
        paged_doc,
        ttnn::nanobind_overload_t{
            [](const PagedOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& page_table_tensor,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<SDPAProgramConfig>& program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    page_table_tensor,
                    is_causal,
                    attn_mask,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
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
            nb::arg("compute_kernel_config") = nb::none()});

    const auto* mla_doc =
        R"doc(
        Flash Multi-Latent Attention (MLA) for decode (single-token generation).

        Variant of ``scaled_dot_product_attention_decode`` where the value head dimension
        (``head_dim_v``) can differ from the key/query head dimension. This is used in
        architectures like DeepSeek-V2/V3 where the latent compressed KV representation
        has a different dimension than the output projection.

        When ``input_tensor_v`` is ``None``, the value is derived from ``input_tensor_k``
        using the first ``head_dim_v`` columns (K is reused for V internally).


        Supported data types:
            Same as ``scaled_dot_product_attention_decode``.


        Memory configurations:
            Same as ``scaled_dot_product_attention_decode``.


        Args:
            input_tensor_q (ttnn.Tensor): Query tensor ``[1, b, nh, dh]``.
            input_tensor_k (ttnn.Tensor): Key tensor ``[b, nkv, s, dh]``. ``TILE`` layout, DRAM.
            input_tensor_v (ttnn.Tensor, optional): Value tensor ``[b, nkv, s, head_dim_v]``.
                If ``None``, V is derived from K using the first ``head_dim_v`` columns.
                Defaults to ``None``.
            head_dim_v (int): Value head dimension. Must be ``<= dh``.
                When V is provided, ``V.shape[-1]`` must equal ``head_dim_v``.


        Keyword args:
            is_causal (bool): Whether to apply causal masking. Defaults to ``True``.
            attn_mask (ttnn.Tensor, optional): Mask ``[b, nh, 1, s]``. Defaults to ``None``.
            cur_pos (List[int], optional): Current positions list of length ``b``.
                Defaults to ``[]``.
            cur_pos_tensor (ttnn.Tensor, optional): Current positions ``[b]``, ``int32``.
                Defaults to ``None``.
            attention_sink (ttnn.Tensor, optional): Sink tensor. Defaults to ``None``.
            scale (float, optional): Scaling factor. Defaults to ``1/sqrt(dh)``.
            sliding_window_size (int, optional): Sliding window size. Defaults to ``None``.
            memory_config (ttnn.MemoryConfig, optional): Output memory config.
                Defaults to ``None``.
            program_config (SDPAProgramConfig, optional): Grid and chunk sizes.
                Defaults to ``None``.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Defaults to ``None``.


        Returns:
            ttnn.Tensor: Output ``[1, b, pnh, head_dim_v]`` (note: last dim is ``head_dim_v``,
            not ``dh``).


        Limitations:
            - Only supported in causal mode (``is_causal=True``).
            - ``head_dim_v`` must be ``<= dh`` (query/key head dimension).
            - Q shape[-1] must equal K shape[-1] (both use ``dh``).
            - When parallelizing Q heads with sharded configs, ``nkv`` must be 1.
            - All other limitations from ``scaled_dot_product_attention_decode`` apply.


        Example (MLA decode):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s = 4, 16, 1, 8192
            dh, head_dim_v = 192, 128  # compressed KV dim vs output dim
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # V has head_dim_v columns, not dh
            V = ttnn.from_torch(torch.randn(b, nkv, s, head_dim_v), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 4096, dtype=torch.int32), dtype=ttnn.int32, device=device)

            # Output shape: [1, b, pnh, head_dim_v]
            output = ttnn.transformer.flash_multi_latent_attention_decode(
                Q, K, V, head_dim_v,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    q_chunk_size=32, k_chunk_size=128),
            )


        Example (MLA decode without V, reusing K):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv, s = 4, 16, 1, 8192
            dh, head_dim_v = 192, 128
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(torch.randn(b, nkv, s, dh), dtype=ttnn.bfloat8_b,
                                layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 4096, dtype=torch.int32), dtype=ttnn.int32, device=device)

            # V=None: kernel derives V from K's first head_dim_v columns
            output = ttnn.transformer.flash_multi_latent_attention_decode(
                Q, K, None, head_dim_v,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
            )

        )doc";

    using MLAOperationType = decltype(ttnn::transformer::flash_multi_latent_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::flash_multi_latent_attention_decode,
        mla_doc,
        ttnn::nanobind_overload_t{
            [](const MLAOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const std::optional<const Tensor>& input_tensor_v,
               const uint32_t head_dim_v,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::vector<uint32_t>& cur_pos,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<SDPAProgramConfig>& program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    head_dim_v,
                    is_causal,
                    attn_mask,
                    cur_pos,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
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
            nb::arg("compute_kernel_config") = nb::none()});

    const auto* paged_mla_doc =
        R"doc(
        Paged Flash Multi-Latent Attention (MLA) for decode (single-token generation).

        Combines paged KV cache with MLA's separate ``head_dim_v``. Uses a ``page_table_tensor``
        for non-contiguous KV storage and supports different Q/K vs V head dimensions.


        Supported data types:
            Same as ``paged_scaled_dot_product_attention_decode``.


        Memory configurations:
            Same as ``paged_scaled_dot_product_attention_decode``.


        Args:
            input_tensor_q (ttnn.Tensor): Query tensor ``[1, b, nh, dh]``.
            input_tensor_k (ttnn.Tensor): Key cache block pool ``[num_blocks, nkv, block_size, dh]``.
                DRAM, ``TILE`` layout.
            input_tensor_v (ttnn.Tensor, optional): Value cache block pool
                ``[num_blocks, nkv, block_size, head_dim_v]``. If ``None``, derived from K.
                Defaults to ``None``.
            head_dim_v (int): Value head dimension. Must be ``<= dh``.
            page_table_tensor (ttnn.Tensor): Page table ``[b, max_num_blocks_per_seq]``.
                ``ROW_MAJOR`` layout. Dtype: ``int32`` (DRAM) or ``uint16`` (HEIGHT_SHARDED).


        Keyword args:
            is_causal (bool): Defaults to ``True``.
            attn_mask (ttnn.Tensor, optional): Mask ``[b, nh, 1, s]``. Defaults to ``None``.
            cur_pos_tensor (ttnn.Tensor, optional): Current positions ``[b]``, ``int32``.
                Defaults to ``None``.
            attention_sink (ttnn.Tensor, optional): Sink tensor. Defaults to ``None``.
            scale (float, optional): Defaults to ``1/sqrt(dh)``.
            sliding_window_size (int, optional): Defaults to ``None``.
            memory_config (ttnn.MemoryConfig, optional): Defaults to ``None``.
            program_config (SDPAProgramConfig, optional): Defaults to ``None``.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Defaults to ``None``.


        Returns:
            ttnn.Tensor: Output ``[1, b, pnh, head_dim_v]``.


        Limitations:
            - Only supported in causal mode (``is_causal=True``).
            - Share cache is not supported with paged attention.
            - ``cur_pos`` list is not available; use ``cur_pos_tensor``.
            - ``head_dim_v`` must be ``<= dh``.
            - All other limitations from ``paged_scaled_dot_product_attention_decode`` apply.


        Example (paged MLA decode):

        .. code-block:: python

            import ttnn, torch

            b, nh, nkv = 4, 16, 1
            dh, head_dim_v = 192, 128
            block_size, max_seq_len = 64, 4096
            max_num_blocks = max_seq_len // block_size
            num_physical_blocks = b * max_num_blocks
            scale = 1.0 / (dh ** 0.5)

            Q = ttnn.from_torch(torch.randn(1, b, nh, dh), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)
            K = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, dh),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            V = ttnn.from_torch(
                torch.randn(num_physical_blocks, nkv, block_size, head_dim_v),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            page_table = ttnn.from_torch(
                torch.randint(0, num_physical_blocks,
                              (b, max_num_blocks), dtype=torch.int32),
                dtype=ttnn.int32, device=device)
            cur_pos_tensor = ttnn.from_torch(
                torch.full((b,), 2048, dtype=torch.int32), dtype=ttnn.int32, device=device)

            output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                Q, K, V, head_dim_v, page_table,
                cur_pos_tensor=cur_pos_tensor,
                scale=scale,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 2),
                    q_chunk_size=32, k_chunk_size=64),
            )

        )doc";

    using PagedMLAOperationType = decltype(ttnn::transformer::paged_flash_multi_latent_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::paged_flash_multi_latent_attention_decode,
        paged_mla_doc,
        ttnn::nanobind_overload_t{
            [](const PagedMLAOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const std::optional<const Tensor>& input_tensor_v,
               const uint32_t head_dim_v,
               const ttnn::Tensor& page_table_tensor,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<SDPAProgramConfig>& program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
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
                    compute_kernel_config);
            },
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
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::transformer
