# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TextEncoder (Qwen3) TTNN model — clean LightweightModule implementation.

Architecture (Qwen3Model from Tongyi-MAI/Z-Image-Turbo / text_encoder):
  - 36 transformer layers, hidden_size=2560, num_heads=32, num_kv_heads=8, head_dim=128
  - intermediate_size=9728, rms_norm_eps=1e-6, rope_theta=1000000
  - Causal self-attention (decoder-style, no KV cache)
  - 4-way tensor parallelism on (1,4) mesh:
      Q/K/V/gate/up projections: column-parallel (ShardTensorToMesh dim=0)
      O/down projections: row-parallel (ShardTensorToMesh dim=1) + all-reduce
      Norms and embedding: replicated

Weights are loaded directly from the HuggingFace PyTorch model — no tensorbin files.
RoPE tables and causal mask are computed at forward time for each sequence length.
"""

import math

import torch
import ttnn


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ── Constants ──────────────────────────────────────────────────────────────────

DRAM_MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

NORM_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Main model class ───────────────────────────────────────────────────────────


class TextEncoderTTNN(LightweightModule):
    """
    Qwen3 text encoder in TTNN with 4-way tensor parallelism.

    Loads weights directly from a PyTorch Qwen3Model and converts them
    to TTNN with TP sharding. Supports variable sequence lengths.

    Usage::

        model = TextEncoderTTNN(mesh_device, pt_model)
        tt_out = model(input_ids_tt)  # [1, seq_len] INT32 on device
    """

    # Architecture constants
    HIDDEN_SIZE = 2560
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    INTERMEDIATE_SIZE = 9728
    NUM_LAYERS = 36
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 1_000_000.0
    TP = 4  # tensor parallelism degree
    ATTN_SCALE = 1.0 / math.sqrt(128)  # 1/sqrt(head_dim) ≈ 0.08839

    def __init__(self, mesh_device, pt_model=None):
        """
        Initialize and load all weights.

        Args:
            mesh_device: TTNN MeshDevice ((1,4) mesh, already opened).
            pt_model: PyTorch Qwen3Model in eval mode. If None, loads from
                      Tongyi-MAI/Z-Image-Turbo automatically.
        """
        self.mesh_device = mesh_device
        self.q_per_dev = self.NUM_HEADS // self.TP  # 8 Q heads per device
        self.kv_per_dev = self.NUM_KV_HEADS // self.TP  # 2 KV heads per device

        if pt_model is None:
            from transformers import AutoModel

            print("  Loading Qwen3 text encoder from HuggingFace ...")
            pt_model = AutoModel.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                use_cache=False,
            ).eval()

        print("  Converting weights to TTNN (TP sharding) ...")
        self.weights = self._load_weights(pt_model)
        print("  Weight loading complete.")

    # ── Weight loading ─────────────────────────────────────────────────────────

    def _load_weights(self, pt_model):
        """
        Load all weights from the PyTorch model and convert to TTNN.

        Column-parallel (Q/K/V/gate/up): ShardTensorToMesh(dim=0)
          - q_proj [4096, 2560] → [1024, 2560] per device  (8 Q heads × 128)
          - k_proj [1024, 2560] → [ 256, 2560] per device  (2 KV heads × 128)
          - v_proj [1024, 2560] → [ 256, 2560] per device
          - gate_proj [9728, 2560] → [2432, 2560] per device
          - up_proj   [9728, 2560] → [2432, 2560] per device

        Row-parallel (O/down) + all-reduce: ShardTensorToMesh(dim=1)
          - o_proj   [2560, 4096] → [2560, 1024] per device
          - down_proj [2560, 9728] → [2560, 2432] per device

        Replicated: embed_tokens, all RMSNorm weights
        """
        sd = {k: v.bfloat16() for k, v in pt_model.state_dict().items()}
        weights = {}

        col_par = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        row_par = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)

        def _tile(t, mapper):
            """Convert a bfloat16 PyTorch tensor to TILE BF16 on device."""
            return ttnn.from_torch(
                t,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.Layout.TILE,
                device=self.mesh_device,
                memory_config=DRAM_MC,
                mesh_mapper=mapper,
            )

        # ── Embedding table: ROW_MAJOR for ttnn.embedding ──────────────────────
        weights["embed_tokens.weight"] = ttnn.from_torch(
            sd["embed_tokens.weight"],
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.mesh_device,
            memory_config=DRAM_MC,
            mesh_mapper=replicate,
        )

        # ── Final RMSNorm ──────────────────────────────────────────────────────
        weights["norm.weight"] = _tile(sd["norm.weight"], replicate)

        # ── RoPE inverse frequencies (CPU float32 for RoPE computation) ────────
        weights["inv_freq"] = 1.0 / (
            self.ROPE_THETA ** (torch.arange(0, self.HEAD_DIM, 2, dtype=torch.float32) / self.HEAD_DIM)
        )  # [64] float32 on CPU

        # ── Per-layer weights ──────────────────────────────────────────────────
        for i in range(self.NUM_LAYERS):
            p = f"layers.{i}"

            # RMSNorm weights — replicated
            weights[f"layers.{i}.input_layernorm.weight"] = _tile(sd[f"{p}.input_layernorm.weight"], replicate)
            weights[f"layers.{i}.post_attention_layernorm.weight"] = _tile(
                sd[f"{p}.post_attention_layernorm.weight"], replicate
            )

            # q_norm and k_norm per head — replicated
            weights[f"layers.{i}.self_attn.q_norm.weight"] = _tile(sd[f"{p}.self_attn.q_norm.weight"], replicate)
            weights[f"layers.{i}.self_attn.k_norm.weight"] = _tile(sd[f"{p}.self_attn.k_norm.weight"], replicate)

            # Attention projections
            weights[f"layers.{i}.self_attn.q_proj.weight"] = _tile(
                sd[f"{p}.self_attn.q_proj.weight"], col_par
            )  # [4096, 2560] → [1024, 2560] per device
            weights[f"layers.{i}.self_attn.k_proj.weight"] = _tile(
                sd[f"{p}.self_attn.k_proj.weight"], col_par
            )  # [1024, 2560] → [256, 2560] per device
            weights[f"layers.{i}.self_attn.v_proj.weight"] = _tile(
                sd[f"{p}.self_attn.v_proj.weight"], col_par
            )  # [1024, 2560] → [256, 2560] per device
            weights[f"layers.{i}.self_attn.o_proj.weight"] = _tile(
                sd[f"{p}.self_attn.o_proj.weight"], row_par
            )  # [2560, 4096] → [2560, 1024] per device

            # MLP projections
            weights[f"layers.{i}.mlp.gate_proj.weight"] = _tile(
                sd[f"{p}.mlp.gate_proj.weight"], col_par
            )  # [9728, 2560] → [2432, 2560] per device
            weights[f"layers.{i}.mlp.up_proj.weight"] = _tile(
                sd[f"{p}.mlp.up_proj.weight"], col_par
            )  # [9728, 2560] → [2432, 2560] per device
            weights[f"layers.{i}.mlp.down_proj.weight"] = _tile(
                sd[f"{p}.mlp.down_proj.weight"], row_par
            )  # [2560, 9728] → [2560, 2432] per device

        return weights

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(self, input_ids):
        """
        Run the Qwen3 text encoder forward pass.

        Args:
            input_ids: [1, seq_len] INT32 TTNN tensor on device (replicated).

        Returns:
            [seq_len, hidden_size] bfloat16 TTNN tensor — last hidden state.
        """
        seq_len = input_ids.shape[-1]

        # Compute RoPE tables and causal mask for this sequence length
        cos, sin = self._compute_rope_tables(seq_len)
        attn_mask = self._compute_causal_mask(seq_len)

        # Token embedding: [1, seq_len] INT32 → [seq_len, 2560] BF16
        hidden_states = self._embedding(input_ids, seq_len)  # [seq_len, 2560]

        # 36 Qwen3 decoder layers
        for layer_idx in range(self.NUM_LAYERS):
            hidden_states = self._decoder_layer(hidden_states, cos, sin, attn_mask, layer_idx)

        # Return pre-norm hidden states (equivalent to hidden_states[-2] in HuggingFace
        # Qwen3Model with output_hidden_states=True). The ZImagePipeline uses
        # hidden_states[-2] as cap_feats input, not last_hidden_state.
        # The cap_embedder in the transformer applies its own RMSNorm, so the
        # text encoder's final norm should NOT be applied here.
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=DRAM_MC)

        return hidden_states

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embedding(self, input_ids, seq_len):
        """
        Token embedding lookup.

        Input:  [1, seq_len] INT32 TTNN tensor on device
        Output: [seq_len, 2560] BF16 TTNN tensor

        Follows the traced op sequence:
          INT32 → TILE → UINT32 (for embedding) → ROW_MAJOR → embedding → reshape
        """
        x = ttnn.to_layout(input_ids, ttnn.Layout.TILE, None, memory_config=None)
        x = ttnn.typecast(x, ttnn.DataType.UINT32, memory_config=DRAM_MC)
        x = ttnn.to_layout(x, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

        x = ttnn.embedding(
            x,
            self.weights["embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [1, seq_len, 2560]

        x = ttnn.reshape(x, [seq_len, self.HIDDEN_SIZE], memory_config=DRAM_MC)
        return x

    # ── RoPE ───────────────────────────────────────────────────────────────────

    def _compute_rope_tables(self, seq_len):
        """
        Compute RoPE cosine and sine tables for the given sequence length.

          inv_freq [64] × positions [0..seq_len-1] → angles [seq_len, 64]
          → doubled [seq_len, 128] → cos/sin → BF16 → [1, 1, seq_len, 128]

        Args:
            seq_len: number of tokens in the sequence

        Returns:
            cos, sin: each [1, 1, seq_len, 128] BF16 TTNN tensors (replicated)
        """
        inv_freq = self.weights["inv_freq"]  # [64] float32 on CPU
        positions = torch.arange(seq_len, dtype=torch.float32)  # [seq_len]

        # [seq_len, 64] = positions[:, None] * inv_freq[None, :]
        angles = torch.outer(positions, inv_freq)

        # Double: [seq_len, 128]
        angles_doubled = torch.cat([angles, angles], dim=-1)

        cos_table = torch.cos(angles_doubled).bfloat16().unsqueeze(0).unsqueeze(0)
        sin_table = torch.sin(angles_doubled).bfloat16().unsqueeze(0).unsqueeze(0)
        # Each: [1, 1, seq_len, 128]

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)

        cos_tt = ttnn.from_torch(
            cos_table,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=DRAM_MC,
            mesh_mapper=replicate,
        )
        sin_tt = ttnn.from_torch(
            sin_table,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=DRAM_MC,
            mesh_mapper=replicate,
        )
        return cos_tt, sin_tt

    # ── Attention mask ──────────────────────────────────────────────────────────

    def _compute_causal_mask(self, seq_len):
        """
        Build a causal (lower-triangular) attention mask.

        Valid positions (i >= j): 0.0
        Masked positions (i < j): -inf

        Output: [1, 1, seq_len, seq_len] BF16 TTNN tensor (replicated)
        """
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(
            mask,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=DRAM_MC,
            mesh_mapper=replicate,
        )

    # ── Decoder layer ──────────────────────────────────────────────────────────

    def _decoder_layer(self, hidden_states, cos, sin, attn_mask, layer_idx):
        """
        Single Qwen3 decoder layer (pre-norm, Post-RoPE).

        Structure:
            residual = hidden_states
            x = input_layernorm(hidden_states)
            x = self_attention(x, cos, sin, attn_mask)      → [1, S, H]
            hidden_states = residual + x                     broadcasts

            x2d = reshape(hidden_states, [S, H])
            residual = x2d
            x = post_attention_layernorm(x2d)
            x = mlp(x)                                       → [1, S, H]
            hidden_states = residual + x                     broadcasts

        Args:
            hidden_states: [S, H] or [1, S, H] BF16 TTNN tensor
            cos, sin:      [1, 1, S, 128] BF16 RoPE tables
            attn_mask:     [1, 1, S, S] BF16 causal mask
            layer_idx:     0-35

        Returns:
            [1, seq_len, 2560] BF16 TTNN tensor
        """
        seq_len = hidden_states.shape[-2] if len(hidden_states.shape) == 3 else hidden_states.shape[0]

        # Ensure [seq_len, hidden_size] for layernorm
        x = ttnn.reshape(hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=DRAM_MC)

        # ── Attention sub-layer ───────────────────────────────────────────────
        residual = x

        x = ttnn.rms_norm(
            x,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MC,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [seq_len, 2560]

        x = self._attention(x, cos, sin, attn_mask, seq_len, layer_idx)
        # x: [1, seq_len, 2560]

        hidden_states = ttnn.add(
            residual,
            x,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [1, seq_len, 2560] (broadcast from [S, H] + [1, S, H])

        # ── MLP sub-layer ─────────────────────────────────────────────────────
        x = ttnn.reshape(hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=DRAM_MC)
        residual = x

        x = ttnn.rms_norm(
            x,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MC,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [seq_len, 2560]

        x = self._mlp(x, seq_len, layer_idx)
        # x: [1, seq_len, 2560]

        hidden_states = ttnn.add(
            residual,
            x,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # [1, seq_len, 2560]

        return hidden_states

    # ── Self-Attention ──────────────────────────────────────────────────────────

    def _attention(self, x, cos, sin, attn_mask, seq_len, layer_idx):
        """
        Qwen3 grouped-query self-attention with RoPE and 4-way TP.

        Per device:
          Q: 8 heads × 128 dim  (col_par on q_proj)
          K: 2 heads × 128 dim  (col_par on k_proj)
          V: 2 heads × 128 dim  (col_par on v_proj)
          O: row_par on o_proj  + all-reduce

        GQA ratio: 4:1  (8 Q heads per 2 KV heads per device)

        Args:
            x:          [seq_len, 2560] BF16
            cos, sin:   [1, 1, seq_len, 128] BF16
            attn_mask:  [1, 1, seq_len, seq_len] BF16
            seq_len:    sequence length (int)
            layer_idx:  layer index

        Returns:
            [1, seq_len, 2560] BF16 (after all-reduce)
        """
        li = layer_idx  # shorthand

        # ── Q projection (col_par: 8 Q heads per device) ─────────────────────
        q = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.q_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 1024] per device

        q = ttnn.reshape(q, [1, seq_len, self.q_per_dev, self.HEAD_DIM], memory_config=DRAM_MC)  # [1, seq_len, 8, 128]

        q = ttnn.rms_norm(
            q,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{li}.self_attn.q_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MC,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [1, seq_len, 8, 128]

        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)
        # [1, 8, seq_len, 128]

        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=DRAM_MC)
        q = ttnn.slice(
            q, [0, 0, 0, 0], [1, self.q_per_dev, seq_len, self.HEAD_DIM], [1, 1, 1, 1], memory_config=DRAM_MC
        )  # [1, 8, seq_len, 128] — trims TILE padding from rotary_embedding

        # ── K projection (col_par: 2 KV heads per device) ────────────────────
        k = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.k_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 256] per device

        k = ttnn.reshape(k, [1, seq_len, self.kv_per_dev, self.HEAD_DIM], memory_config=DRAM_MC)  # [1, seq_len, 2, 128]

        k = ttnn.rms_norm(
            k,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{li}.self_attn.k_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MC,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [1, seq_len, 2, 128]

        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)
        # [1, 2, seq_len, 128]

        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=DRAM_MC)
        k = ttnn.slice(
            k, [0, 0, 0, 0], [1, self.kv_per_dev, seq_len, self.HEAD_DIM], [1, 1, 1, 1], memory_config=DRAM_MC
        )  # [1, 2, seq_len, 128]

        # ── V projection (col_par: 2 KV heads per device) ────────────────────
        v = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.v_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 256] per device

        v = ttnn.reshape(v, [1, seq_len, self.kv_per_dev, self.HEAD_DIM], memory_config=DRAM_MC)  # [1, seq_len, 2, 128]

        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)
        # [1, 2, seq_len, 128] — no RoPE on V

        # ── Scaled Dot-Product Attention (GQA: 4 Q heads per KV head) ─────────
        # Expand K/V from [1, 2, S, 128] to [1, 8, S, 128] using grouped
        # (repeat_interleave) style: each KV head is repeated 4 times.
        # NOTE: ttnn.transformer.scaled_dot_product_attention with GQA (unequal
        # Q/KV head counts) produces incorrect results in this TTNN version.
        # Manual SDPA (matmul + scale + mask + softmax + matmul) is used instead.
        grp = self.q_per_dev // self.kv_per_dev  # 4
        k_exp = self._expand_kv(k, self.kv_per_dev, grp, seq_len)
        v_exp = self._expand_kv(v, self.kv_per_dev, grp, seq_len)
        # k_exp, v_exp: [1, 8, seq_len, 128] each

        scores = ttnn.matmul(
            q,
            k_exp,
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
        )  # [1, 8, seq_len, seq_len]
        scores = ttnn.multiply(
            scores,
            self.ATTN_SCALE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        scores = ttnn.add(
            scores,
            attn_mask,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # causal mask broadcast: [1,1,S,S] → [1,8,S,S]
        attn_w = ttnn.softmax(scores, dim=-1, memory_config=DRAM_MC)
        out = ttnn.matmul(
            attn_w,
            v_exp,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
        )  # [1, 8, seq_len, 128]

        out = ttnn.transformer.concatenate_heads(out, memory_config=DRAM_MC)
        # [1, seq_len, 1024]

        out = ttnn.reshape(out, [seq_len, self.q_per_dev * self.HEAD_DIM], memory_config=DRAM_MC)  # [seq_len, 1024]

        # ── O projection (row_par) + TP all-reduce ────────────────────────────
        out = ttnn.matmul(
            out,
            self.weights[f"layers.{li}.self_attn.o_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2560] — partial sum per device

        return self._all_reduce(out, seq_len)  # [1, seq_len, 2560]

    # ── KV head expansion for GQA ──────────────────────────────────────────────

    def _expand_kv(self, kv, kv_heads, grp, seq_len):
        """
        Expand KV from [1, kv_heads, seq_len, head_dim] to
        [1, kv_heads * grp, seq_len, head_dim] using grouped repeat_interleave
        style: each KV head is repeated 'grp' times contiguously.

        Example (kv_heads=2, grp=4):
          KV[0] KV[0] KV[0] KV[0] KV[1] KV[1] KV[1] KV[1]

        Args:
            kv:       [1, kv_heads, seq_len, head_dim] BF16 TTNN tensor
            kv_heads: number of KV heads per device (2)
            grp:      repeat count per KV head (4 = q_per_dev // kv_per_dev)
            seq_len:  sequence length

        Returns:
            [1, kv_heads * grp, seq_len, head_dim] BF16 TTNN tensor
        """
        slices = []
        for kv_i in range(kv_heads):
            s = ttnn.slice(
                kv,
                [0, kv_i, 0, 0],
                [1, kv_i + 1, seq_len, self.HEAD_DIM],
                [1, 1, 1, 1],
                memory_config=DRAM_MC,
            )
            for _ in range(grp):
                slices.append(s)
        return ttnn.concat(slices, dim=1, memory_config=DRAM_MC)

    # ── MLP ────────────────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, layer_idx):
        """
        Qwen3 SwiGLU MLP with 4-way TP.

        Structure:
            gate = SiLU(gate_proj(x))   col_par: 2432 per device
            up   = up_proj(x)           col_par: 2432 per device
            h    = gate * up
            out  = down_proj(h)         row_par + all-reduce

        Args:
            x:         [seq_len, 2560] BF16
            seq_len:   sequence length (int)
            layer_idx: layer index

        Returns:
            [1, seq_len, 2560] BF16 (after all-reduce)
        """
        li = layer_idx

        # Gate projection with SiLU fused (col_par)
        gate = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.mlp.gate_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )  # [seq_len, 2432]

        # Up projection (col_par)
        up = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.mlp.up_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2432]

        # Element-wise product (SwiGLU)
        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        # [seq_len, 2432]

        # Down projection (row_par) + all-reduce
        out = ttnn.matmul(
            h,
            self.weights[f"layers.{li}.mlp.down_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2560] — partial sum per device

        return self._all_reduce(out, seq_len)  # [1, seq_len, 2560]

    # ── TP All-Reduce (reduce_scatter + all_gather) ────────────────────────────

    def _all_reduce(self, x, seq_len):
        """
        Tensor-parallel all-reduce via reduce_scatter + all_gather (ring topology).

        Matches the traced pattern exactly:
          [S, H] → reshape [1,1,S,H] → reduce_scatter(dim=3) → [1,1,S,H//4]
          → reshape [S, H//4] → all_gather(dim=1) → [S, H]
          → reshape [1, S, H]

        Args:
            x:       [seq_len, hidden_size] BF16 — partial sum on each device
            seq_len: sequence length (int)

        Returns:
            [1, seq_len, hidden_size] BF16 — fully summed, same on all devices
        """
        H = self.HIDDEN_SIZE  # 2560

        x = ttnn.reshape(x, [1, 1, seq_len, H], memory_config=DRAM_MC)

        x = ttnn.reduce_scatter(
            input_tensor=x,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MC,
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [1, 1, seq_len, H//4] per device

        x = ttnn.reshape(x, [seq_len, H // self.TP], memory_config=DRAM_MC)
        # [seq_len, 640]

        x = ttnn.all_gather(
            input_tensor=x,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MC,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )  # [seq_len, H]

        x = ttnn.reshape(x, [1, seq_len, H], memory_config=DRAM_MC)
        # [1, seq_len, 2560]

        return x
