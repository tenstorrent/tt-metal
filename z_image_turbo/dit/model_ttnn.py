# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZImageTransformer TTNN — optimized inference model.

All performance optimizations are active:
  - ttnn.rms_norm           (replaces manual 9-op F32 sequence, ×153/pass)
  - ttnn.layer_norm         (final layer)
  - minimal_matmul          (attention + MLP projections, pre-transposed weights)
  - minimal_matmul_split    (fused Q/K/V in one kernel)
  - nlp_create_qkv_heads    (fused V head-reshape)
  - CCLManager async CCL    (reduce_scatter_minimal_async + all_gather_async)

4-way tensor parallelism, (1,4) MeshDevice, 8 heads/device.
Fixed resolution: 512×512 px (64×64 latent, 1024 image patches, 32 caption tokens).
"""

import importlib.util
import json
import math
import os
import sys

import torch
import ttnn

from dit import consteval  # run_const_evals, CONSTEVAL_MAP
from dit import model_pt

HERE = os.path.dirname(os.path.abspath(__file__))

# ── tt_dit library path ────────────────────────────────────────────────────────
_TT_DIT_PATH = os.path.normpath(
    os.path.join(
        HERE,
        "../..",
        "models",
    )
)
if _TT_DIT_PATH not in sys.path:
    sys.path.insert(0, _TT_DIT_PATH)

from tt_dit.parallel.manager import CCLManager

# tt_dit minimal_matmul config (shape lookup for blocking parameters)
_matmul_path = os.path.join(_TT_DIT_PATH, "tt_dit/utils/matmul.py")
_spec = importlib.util.spec_from_file_location("tt_dit_matmul", _matmul_path)
_mm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm_mod)
_get_matmul_config = _mm_mod.get_matmul_config
del _spec, _mm_mod

# ── Architecture constants ─────────────────────────────────────────────────────

HIDDEN_DIM = 3840
PADDED_HEADS = 32  # 30 original → padded to 32 for TP
ORIGINAL_HEADS = 30
HEAD_DIM = 128
MLP_HIDDEN = 10240
TP = 4
HEADS_PER_DEV = PADDED_HEADS // TP  # 8
MLP_PER_DEV = MLP_HIDDEN // TP  # 2560
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)

EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256
IMG_PATCHES = 1024  # 32×32 patches
CAP_TOKENS = 128
PATCH_SIZE = 2
PATCH_DIM = 16 * PATCH_SIZE * PATCH_SIZE  # 64
ADALN_EMBED_DIM = 256

RMS_EPS = 1e-5  # all RMSNorm layers
LN_EPS = 1e-6  # final LayerNorm

# ── TTNN config ────────────────────────────────────────────────────────────────

DRAM_MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Weight loading helpers ─────────────────────────────────────────────────────


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _pad_col(w):
    return torch.cat([w, torch.zeros(EXTRA_DIM, w.shape[1], dtype=w.dtype)], dim=0)


def _pad_row(w):
    return torch.cat([w, torch.zeros(w.shape[0], EXTRA_DIM, dtype=w.dtype)], dim=1)


def _to_ttnn(pt, layout, dtype, stype, mesh_device, on_device):
    ttnn_layout = ttnn.Layout.TILE if layout == "TILE" else ttnn.Layout.ROW_MAJOR
    ttnn_dtype = ttnn.DataType.BFLOAT16 if dtype == "BFLOAT16" else ttnn.DataType.FLOAT32
    if stype in ("col_par_attn", "col_par_mlp"):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    elif stype in ("row_par_attn_out", "row_par_mlp"):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    kwargs = dict(dtype=ttnn_dtype, layout=ttnn_layout, mesh_mapper=mapper)
    if on_device:
        kwargs["device"] = mesh_device
        kwargs["memory_config"] = DRAM_MC
    return ttnn.from_torch(pt, **kwargs)


def _make_const_device(pt, mesh_device, dtype=ttnn.DataType.BFLOAT16):
    pt_cast = pt.float() if dtype == ttnn.DataType.FLOAT32 else pt.bfloat16()
    return ttnn.from_torch(
        pt_cast,
        dtype=dtype,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_MC,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def load_static_inputs(mesh_device, transformer):
    config_path = os.path.join(HERE, "tensor_load_config.json")
    with open(config_path) as f:
        config = json.load(f)
    state_dict = transformer.state_dict()
    inputs = [None] * 529
    for param_name, cfg in config.items():
        arg_idx = cfg["arg_idx"]
        pt = state_dict.get(param_name)
        if pt is None:
            raise KeyError(f"Parameter '{param_name}' not found in state_dict.")
        pt = pt.bfloat16()
        stype = cfg["stype"]
        if stype == "col_par_attn" and pt.shape[0] == ORIGINAL_HEADS * HEAD_DIM:
            pt = _pad_col(pt)
        elif stype == "row_par_attn_out" and pt.shape[1] == ORIGINAL_HEADS * HEAD_DIM:
            pt = _pad_row(pt)
        inputs[arg_idx] = _to_ttnn(pt, cfg["layout"], cfg["dtype"], stype, mesh_device, cfg["on_device"])
    inputs[330] = _make_const_device(torch.tensor(1.0), mesh_device)
    inputs[331] = _make_const_device(torch.tensor(0.0), mesh_device)
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    rope = transformer.rope_embedder
    freqs = RopeEmbedder.precompute_freqs_cis(rope.axes_dims, rope.axes_lens, getattr(rope, "theta", 256.0))
    inputs[334] = _make_const_device(freqs[0], mesh_device, dtype=ttnn.DataType.FLOAT32)
    inputs[332] = _make_const_device(freqs[1], mesh_device, dtype=ttnn.DataType.FLOAT32)
    inputs[333] = _make_const_device(freqs[2], mesh_device, dtype=ttnn.DataType.FLOAT32)
    return inputs


# ── Main model class ───────────────────────────────────────────────────────────


class ZImageTransformerTTNN(LightweightModule):
    """Optimized TTNN ZImageTransformer with all perf improvements baked in."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

        # ── Load + pad the reference PyTorch model ─────────────────────────────
        tr_pt = model_pt.load_model()
        model_pt.pad_heads(tr_pt)

        # ── Load weights ───────────────────────────────────────────────────────
        print("  Loading static inputs ...")
        self._static_inputs = load_static_inputs(mesh_device, tr_pt)
        del tr_pt
        print("  Running consteval ...")
        self._cached = consteval.run_const_evals(self._static_inputs, mesh_device)
        print("  Consteval complete.")

        config_path = os.path.join(HERE, "tensor_load_config.json")
        with open(config_path) as f:
            _config = json.load(f)
        _arg_to_ce = {arg_idx: ce_idx for ce_idx, (arg_idx, _) in consteval.CONSTEVAL_MAP.items()}
        self.weights = {}
        for param_name, cfg in _config.items():
            arg_idx = cfg["arg_idx"]
            if not cfg["on_device"]:
                ce_key = f"main_const_eval_{_arg_to_ce[arg_idx]}"
                t = self._cached[ce_key][0]
            else:
                t = self._static_inputs[arg_idx]
            self.weights[param_name] = t

        # RoPE tables
        def _to_bf16_rm(t):
            host = ttnn.to_torch(
                ttnn.from_device(t),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
            )
            host = host[: host.shape[0] // 4]
            if host.dim() > 2:
                host = host.reshape(host.shape[0], -1)
            return ttnn.from_torch(
                host.bfloat16(),
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.Layout.ROW_MAJOR,
                device=mesh_device,
                memory_config=DRAM_MC,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.weights["_freqs_F"] = _to_bf16_rm(self._static_inputs[334])
        self.weights["_freqs_H"] = _to_bf16_rm(self._static_inputs[332])
        self.weights["_freqs_W"] = _to_bf16_rm(self._static_inputs[333])

        # Position IDs, consteval scalars
        self.weights["_img_f_ids"] = self._cached["main_const_eval_158"][0]
        self.weights["_img_h_ids"] = self._cached["main_const_eval_158"][1]
        self.weights["_img_w_ids"] = self._cached["main_const_eval_158"][2]
        self.weights["_cap_hw_ids"] = self._cached["main_const_eval_158"][3]
        self.weights["_cap_f_ids"] = self._cached["main_const_eval_96"][0]
        self.weights["_t_freqs"] = self._cached["main_const_eval_219"][0]
        self.weights["_t_scale"] = self._cached["main_const_eval_367"][0]
        self.weights["_eps_hidden"] = self._cached["main_const_eval_208"][0]
        self.weights["_eps_cap"] = self._cached["main_const_eval_208"][1]
        self.weights["_eps_qk"] = self._cached["main_const_eval_208"][2]
        self.weights["_scale_hidden"] = self._cached["main_const_eval_230"][0]
        self.weights["_scale_head"] = self._cached["main_const_eval_408"][0]
        self.weights["_one"] = self._cached["main_const_eval_454"][0]
        self.weights["_scale_cap"] = self._cached["main_const_eval_88"][0]
        self.weights["_x_pad_token"] = self._static_inputs[368]
        self.weights["_cap_pad_token"] = self._static_inputs[329]

        # ── Detect compute grid ────────────────────────────────────────────────
        try:
            d = mesh_device.get_device(0) if hasattr(mesh_device, "get_device") else mesh_device
            self._core_grid = d.compute_with_storage_grid_size()
        except Exception:
            self._core_grid = ttnn.CoreCoord(8, 8)
        print(f"  Compute grid: {self._core_grid.x}×{self._core_grid.y}")

        # ── Pre-process weights for optimized ops ──────────────────────────────
        print("  Pre-processing weights ...")
        self._prep_qk_norm_weights()
        self._prep_fused_qkv_weights()
        self._prep_parallel_weights()

        # ── Async CCL (reduce_scatter_minimal_async + all_gather_async) ────────
        self._ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
        print("  Async CCL initialized.")

        # ── cap_feats persistent buffer (updated per-prompt) ───────────────────
        self._cap_feats_buf = None

    # ── Weight preprocessing ───────────────────────────────────────────────────

    def _prep_qk_norm_weights(self):
        """Reshape QK norm weights [1,1,1,64,2] → [1,1,1,128] for ttnn.rms_norm."""
        for key in list(self.weights):
            if (".norm_q." in key or ".norm_k." in key) and "weight" in key:
                w = self.weights[key]
                w_host = ttnn.to_torch(
                    ttnn.from_device(w),
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
                )
                shard = w_host[: w_host.shape[0] // 4].reshape(1, 1, 1, HEAD_DIM).bfloat16()
                self.weights[key + "_flat"] = ttnn.from_torch(
                    shard,
                    dtype=ttnn.DataType.BFLOAT16,
                    layout=ttnn.Layout.TILE,
                    device=self.mesh_device,
                    memory_config=DRAM_MC,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

    def _prep_fused_qkv_weights(self):
        """Fuse Q/K/V weights → [HIDDEN, 3*N] for minimal_matmul_split."""
        prefixes = (
            [f"noise_refiner.{i}" for i in range(2)]
            + [f"context_refiner.{i}" for i in range(2)]
            + [f"layers.{i}" for i in range(30)]
        )
        for prefix in prefixes:
            q_key = f"{prefix}.attention.to_q.weight"
            if q_key not in self.weights:
                continue
            q_T = ttnn.permute(self.weights[q_key], [1, 0], memory_config=DRAM_MC)
            k_T = ttnn.permute(self.weights[f"{prefix}.attention.to_k.weight"], [1, 0], memory_config=DRAM_MC)
            v_T = ttnn.permute(self.weights[f"{prefix}.attention.to_v.weight"], [1, 0], memory_config=DRAM_MC)
            self.weights[f"{prefix}.attention.qkv_fused_mmT"] = ttnn.concat(
                [q_T, k_T, v_T], dim=1, memory_config=DRAM_MC
            )

    def _prep_parallel_weights(self):
        """Pre-transpose to_out, w1/w2/w3 weights for minimal_matmul."""
        suffixes = (
            "attention.to_out.0.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        )
        for key in list(self.weights):
            if any(key.endswith(s) for s in suffixes) and len(self.weights[key].shape) == 2:
                self.weights[key + "_mmT"] = ttnn.permute(self.weights[key], [1, 0], memory_config=DRAM_MC)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_tile(self, x):
        if x.get_layout() != ttnn.Layout.TILE:
            x = ttnn.to_layout(x, ttnn.Layout.TILE, memory_config=DRAM_MC)
        return x

    def _mm(self, x, weight, M, K, N, dtype=ttnn.DataType.BFLOAT16):
        """minimal_matmul with auto-configured blocking."""
        config = _get_matmul_config(M, K, N, self._core_grid)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            config=config,
            compute_kernel_config=REDUCE_KERNEL,
            dtype=dtype,
            memory_config=DRAM_MC,
        )

    # ── Optimized norm ─────────────────────────────────────────────────────────

    def _rms_norm_f32(self, x, norm_weight, scale_inv_dim, eps, hidden_dim):
        """Fused RMS norm — replaces the manual 9-op F32 sequence."""
        if x.dtype == ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        x = self._ensure_tile(x)
        return ttnn.rms_norm(
            x,
            epsilon=RMS_EPS,
            weight=norm_weight,
            memory_config=DRAM_MC,
            compute_kernel_config=REDUCE_KERNEL,
        )

    def _qk_norm(self, qk, norm_weight, seq_len, num_heads):
        """Fused RMS norm on head dimension using pre-reshaped [1,1,1,128] weight."""
        # Find the flat weight (reshaped from [1,1,1,64,2] to [1,1,1,128] at init)
        flat_w = None
        for key, val in self.weights.items():
            if val is norm_weight:
                flat_w = self.weights.get(key + "_flat")
                break
        if flat_w is None:
            # Fallback: manual F32 RMSNorm (should not happen after _prep_qk_norm_weights)
            return self._qk_norm_manual(qk, norm_weight, seq_len, num_heads)

        dtype_in = qk.dtype
        if dtype_in == ttnn.DataType.FLOAT32:
            qk = ttnn.typecast(qk, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        qk = self._ensure_tile(qk)
        out = ttnn.rms_norm(
            qk,
            epsilon=RMS_EPS,
            weight=flat_w,
            memory_config=DRAM_MC,
            compute_kernel_config=REDUCE_KERNEL,
        )
        if dtype_in == ttnn.DataType.FLOAT32:
            out = ttnn.typecast(out, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        return out

    def _qk_norm_manual(self, qk, norm_weight, seq_len, num_heads):
        """Fallback manual F32 per-head RMSNorm (matches base model_ttnn.py exactly)."""
        qk_sq = ttnn.pow(qk, 2.0, memory_config=DRAM_MC)
        qk_sum = ttnn.sum(qk_sq, dim=3, keepdim=False, memory_config=DRAM_MC)
        qk_sum = ttnn.reshape(qk_sum, [1, seq_len, num_heads, 1, 1], memory_config=DRAM_MC)
        qk_mean = ttnn.multiply(qk_sum, self.weights["_scale_head"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        qk_var = ttnn.add(qk_mean, self.weights["_eps_qk"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        qk_rsqrt = ttnn.rsqrt(qk_var, memory_config=DRAM_MC)
        qk_r = ttnn.reshape(qk, [1, seq_len, num_heads, HEAD_DIM // 2, 2], memory_config=DRAM_MC)
        qk_normed = ttnn.multiply(qk_r, qk_rsqrt, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        w_f32 = ttnn.typecast(norm_weight, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        qk_normed = ttnn.multiply(qk_normed, w_f32, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        return ttnn.reshape(qk_normed, [1, seq_len, num_heads, HEAD_DIM], memory_config=DRAM_MC)

    # ── Optimized attention ────────────────────────────────────────────────────

    def _attention(self, x, seq_len, block_prefix, is_caption=False):
        """Attention with fused QKV (minimal_matmul_split) + nlp_create_qkv_heads for V."""
        x_2d = ttnn.reshape(x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        N = HEADS_PER_DEV * HEAD_DIM  # 1024

        # ── Fused QKV projection ──────────────────────────────────────────────
        fused_qkv = self.weights[f"{block_prefix}.attention.qkv_fused_mmT"]
        config = _get_matmul_config(seq_len, HIDDEN_DIM, 3 * N, self._core_grid)
        q_2d, k_2d, v_2d = ttnn.experimental.minimal_matmul_split(
            x_2d,
            fused_qkv,
            chunks=3,
            dim=-1,
            config=config,
            compute_kernel_config=REDUCE_KERNEL,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )  # each [seq, N] BF16

        # Q: reshape → QK norm → RoPE
        q = ttnn.reshape(q_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
        q = self._qk_norm(q, self.weights[f"{block_prefix}.attention.norm_q.weight"], seq_len, HEADS_PER_DEV)
        q = self._apply_rope(q, seq_len, HEADS_PER_DEV, is_caption=is_caption)

        # K: same
        k = ttnn.reshape(k_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=DRAM_MC)
        k = self._qk_norm(k, self.weights[f"{block_prefix}.attention.norm_k.weight"], seq_len, HEADS_PER_DEV)
        k = self._apply_rope(k, seq_len, HEADS_PER_DEV, is_caption=is_caption)

        # V: nlp_create_qkv_heads handles head-reshape in one fused op
        v_4d = ttnn.reshape(v_2d, [1, 1, seq_len, N], memory_config=DRAM_MC)
        v_4d = self._ensure_tile(v_4d)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_4d,
            num_heads=HEADS_PER_DEV,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=DRAM_MC,
        )  # [1, HEADS_PER_DEV, seq, HEAD_DIM] BF16

        # ── SDPA ─────────────────────────────────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=ATTN_SCALE,
            sliding_window_size=None,
            memory_config=DRAM_MC,
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out, memory_config=DRAM_MC)
        attn_out = ttnn.reshape(attn_out, [seq_len, N], memory_config=DRAM_MC)

        # ── to_out projection (row_par) ───────────────────────────────────────
        out_wT = self.weights[f"{block_prefix}.attention.to_out.0.weight_mmT"]
        attn_out = self._mm(attn_out, out_wT, seq_len, N, HIDDEN_DIM)
        return self._all_reduce(attn_out, seq_len)

    # ── Optimized MLP ──────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, block_prefix):
        """SwiGLU MLP with minimal_matmul for all three projections."""
        w1T = self.weights[f"{block_prefix}.feed_forward.w1.weight_mmT"]
        w3T = self.weights[f"{block_prefix}.feed_forward.w3.weight_mmT"]
        w2T = self.weights[f"{block_prefix}.feed_forward.w2.weight_mmT"]

        gate = self._mm(x, w1T, seq_len, HIDDEN_DIM, MLP_PER_DEV)
        gate = ttnn.silu(gate, memory_config=DRAM_MC)
        up = self._mm(x, w3T, seq_len, HIDDEN_DIM, MLP_PER_DEV)
        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        out = self._mm(h, w2T, seq_len, MLP_PER_DEV, HIDDEN_DIM)
        return self._all_reduce(out, seq_len)

    # ── Async all-reduce ───────────────────────────────────────────────────────

    def _all_reduce(self, x, seq_len):
        """Async ring all-reduce via CCLManager (persistent ping-pong buffers)."""
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        x = self._ccl.reduce_scatter(x, dim=3, mesh_axis=1, use_persistent_buffer=True)
        x = self._ccl.all_gather(x, dim=3, mesh_axis=1, use_hyperparams=True, use_persistent_buffer=True)
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        x = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        return x

    # ── Optimized final layer ──────────────────────────────────────────────────

    def _final_layer(self, x, adaln_input, seq_len):
        """Final norm (ttnn.layer_norm) + adaLN scale + linear projection."""
        final_prefix = "all_final_layer.2-1"

        cond = ttnn.silu(adaln_input, memory_config=DRAM_MC)
        scale_raw = ttnn.matmul(
            cond,
            self.weights[f"{final_prefix}.adaLN_modulation.1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        scale_raw = ttnn.add(
            scale_raw,
            self.weights[f"{final_prefix}.adaLN_modulation.1.bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )
        scale_raw_bf16 = ttnn.typecast(scale_raw, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        one = ttnn.typecast(self.weights["_one"], ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        scale = ttnn.add(one, scale_raw_bf16, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        scale = ttnn.reshape(scale, [1, 1, HIDDEN_DIM], memory_config=DRAM_MC)

        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)
        if x_3d.dtype == ttnn.DataType.FLOAT32:
            x_3d = ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        x_3d = self._ensure_tile(x_3d)

        x_norm = ttnn.layer_norm(
            x_3d,
            epsilon=LN_EPS,
            memory_config=DRAM_MC,
            compute_kernel_config=REDUCE_KERNEL,
        )

        x_scaled = ttnn.multiply(x_norm, scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        x_2d = ttnn.reshape(x_scaled, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        out = ttnn.matmul(
            x_2d,
            self.weights[f"{final_prefix}.linear.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )
        out = ttnn.add(
            out, self.weights[f"{final_prefix}.linear.bias"], dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC
        )
        return ttnn.reshape(out, [1, seq_len, PATCH_DIM], memory_config=DRAM_MC)

    # ── Unchanged infrastructure (patchify, embed, RoPE, blocks, unpatchify) ──

    def _patchify_and_embed(self, latent):
        x = ttnn.reshape(latent, [16, 1, 1, 32, 2, 32, 2], memory_config=DRAM_MC)
        x = ttnn.permute(x, [1, 3, 5, 2, 4, 6, 0], memory_config=DRAM_MC, pad_value=0.0)
        x = ttnn.reshape(x, [IMG_PATCHES, PATCH_DIM], memory_config=DRAM_MC)
        x = ttnn.to_layout(x, ttnn.Layout.TILE, memory_config=DRAM_MC)
        x = ttnn.matmul(
            x,
            self.weights["all_x_embedder.2-1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        x = ttnn.add(x, self.weights["all_x_embedder.2-1.bias"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        return ttnn.reshape(x, [1, IMG_PATCHES, HIDDEN_DIM], memory_config=DRAM_MC)

    def _cap_embed(self, cap_feats):
        if len(cap_feats.shape) == 3:
            cap_feats = ttnn.reshape(cap_feats, [CAP_TOKENS, 2560], memory_config=DRAM_MC)
        x = self._rms_norm_f32(
            cap_feats,
            norm_weight=self.weights["cap_embedder.0.weight"],
            scale_inv_dim=self.weights["_scale_cap"],
            eps=self.weights["_eps_cap"],
            hidden_dim=2560,
        )
        x = ttnn.matmul(
            x,
            self.weights["cap_embedder.1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        x = ttnn.add(x, self.weights["cap_embedder.1.bias"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        return ttnn.reshape(x, [1, CAP_TOKENS, HIDDEN_DIM], memory_config=DRAM_MC)

    def _timestep_embed(self, timestep):
        t = ttnn.multiply(timestep, self.weights["_t_scale"], dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        t = ttnn.reshape(t, [1, 1], memory_config=DRAM_MC)
        freqs = ttnn.multiply(t, self.weights["_t_freqs"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        t_emb = ttnn.concat(
            [ttnn.cos(freqs, memory_config=DRAM_MC), ttnn.sin(freqs, memory_config=DRAM_MC)],
            dim=1,
            memory_config=DRAM_MC,
        )
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        t_emb = ttnn.matmul(
            t_emb,
            self.weights["t_embedder.mlp.0.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        t_emb = ttnn.add(
            t_emb, self.weights["t_embedder.mlp.0.bias"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC
        )
        t_emb = ttnn.silu(t_emb, memory_config=DRAM_MC)
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        t_emb = ttnn.matmul(
            t_emb,
            self.weights["t_embedder.mlp.2.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        t_emb = ttnn.add(
            t_emb, self.weights["t_embedder.mlp.2.bias"], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC
        )
        t_emb = ttnn.typecast(t_emb, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        return ttnn.reshape(t_emb, [1, ADALN_EMBED_DIM], memory_config=DRAM_MC)

    def _adaLN_modulation(self, adaln_input, block_prefix):
        if block_prefix.startswith("all_final_layer"):
            w_key = f"{block_prefix}.adaLN_modulation.1.weight"
            b_key = f"{block_prefix}.adaLN_modulation.1.bias"
        else:
            w_key = f"{block_prefix}.adaLN_modulation.0.weight"
            b_key = f"{block_prefix}.adaLN_modulation.0.bias"
        mod = ttnn.matmul(
            adaln_input,
            self.weights[w_key],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
        )
        mod = ttnn.add(mod, self.weights[b_key], dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        mod = ttnn.typecast(mod, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        mod = ttnn.reshape(mod, [1, 1, 4 * HIDDEN_DIM], memory_config=DRAM_MC)
        H = HIDDEN_DIM
        s = lambda a, b: ttnn.slice(mod, [0, 0, a], [1, 1, b], [1, 1, 1], memory_config=DRAM_MC)
        scale_msa = ttnn.add(self.weights["_one"], s(0, H), dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        gate_msa = ttnn.tanh(s(H, 2 * H), memory_config=DRAM_MC)
        scale_mlp = ttnn.add(self.weights["_one"], s(2 * H, 3 * H), dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        gate_mlp = ttnn.tanh(s(3 * H, 4 * H), memory_config=DRAM_MC)
        return scale_msa, gate_msa, scale_mlp, gate_mlp

    def _apply_rope(self, q_f32, seq_len, num_heads, is_caption=False):
        freqs_cis = self._build_freqs_cis(seq_len, is_caption)
        freqs_cis = ttnn.to_layout(freqs_cis, ttnn.Layout.TILE, memory_config=DRAM_MC)
        q = ttnn.reshape(q_f32, [1, seq_len, num_heads, HEAD_DIM // 2, 2], memory_config=DRAM_MC)
        q_real = ttnn.slice(
            q, [0, 0, 0, 0, 0], [1, seq_len, num_heads, HEAD_DIM // 2, 1], [1, 1, 1, 1, 1], memory_config=DRAM_MC
        )
        q_imag = ttnn.slice(
            q, [0, 0, 0, 0, 1], [1, seq_len, num_heads, HEAD_DIM // 2, 2], [1, 1, 1, 1, 1], memory_config=DRAM_MC
        )
        f_real = ttnn.slice(
            freqs_cis, [0, 0, 0, 0, 0], [1, seq_len, 1, HEAD_DIM // 2, 1], [1, 1, 1, 1, 1], memory_config=DRAM_MC
        )
        f_imag = ttnn.slice(
            freqs_cis, [0, 0, 0, 0, 1], [1, seq_len, 1, HEAD_DIM // 2, 2], [1, 1, 1, 1, 1], memory_config=DRAM_MC
        )
        out_real = ttnn.subtract(
            ttnn.multiply(q_real, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            ttnn.multiply(q_imag, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )
        out_imag = ttnn.add(
            ttnn.multiply(q_real, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            ttnn.multiply(q_imag, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MC),
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )
        q_rot = ttnn.concat([out_real, out_imag], dim=4, memory_config=DRAM_MC)
        q_rot = ttnn.reshape(q_rot, [1, seq_len, num_heads, HEAD_DIM], memory_config=DRAM_MC)
        q_rot = ttnn.permute(q_rot, [0, 2, 1, 3], memory_config=DRAM_MC, pad_value=0.0)
        return ttnn.typecast(q_rot, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)

    def _build_freqs_cis(self, seq_len, is_caption=False):
        if seq_len == IMG_PATCHES:
            return self._build_freqs_img()
        elif seq_len == CAP_TOKENS:
            return self._build_freqs_cap()
        else:
            return ttnn.concat([self._build_freqs_img(), self._build_freqs_cap()], dim=1, memory_config=DRAM_MC)

    def _build_freqs_img(self):
        ff = self._embed_freq(self.weights["_img_f_ids"], self.weights["_freqs_F"], IMG_PATCHES, 16)
        fh = self._embed_freq(self.weights["_img_h_ids"], self.weights["_freqs_H"], IMG_PATCHES, 24)
        fw = self._embed_freq(self.weights["_img_w_ids"], self.weights["_freqs_W"], IMG_PATCHES, 24)
        return ttnn.concat([ff, fh, fw], dim=3, memory_config=DRAM_MC)

    def _build_freqs_cap(self):
        ff = self._embed_freq(self.weights["_cap_f_ids"], self.weights["_freqs_F"], CAP_TOKENS, 16)
        fhw = self._embed_freq(self.weights["_cap_hw_ids"], self.weights["_freqs_H"], CAP_TOKENS, 24)
        return ttnn.concat([ff, fhw, fhw], dim=3, memory_config=DRAM_MC)

    def _embed_freq(self, ids, freq_table, seq_len, out_half_dim):
        ids_flat = ttnn.reshape(ids, [seq_len], memory_config=DRAM_MC)
        emb = ttnn.embedding(
            ids_flat,
            freq_table,
            padding_idx=None,
            layout=ttnn.Layout.ROW_MAJOR,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        return ttnn.reshape(emb, [1, seq_len, 1, out_half_dim, 2], memory_config=DRAM_MC)

    def _block_with_adaLN(self, x, adaln_input, seq_len, block_prefix):
        if x.dtype != ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._adaLN_modulation(adaln_input, block_prefix)
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        norm1_x = self._rms_norm_f32(
            x_3d,
            self.weights[f"{block_prefix}.attention_norm1.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        norm1_x = ttnn.multiply(norm1_x, scale_msa, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        attn_out = self._attention(norm1_x, seq_len, block_prefix)
        norm2_out = self._rms_norm_f32(
            attn_out,
            self.weights[f"{block_prefix}.attention_norm2.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        x = ttnn.add(
            ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            ttnn.multiply(gate_msa, norm2_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )

        norm3_x = self._rms_norm_f32(
            x,
            self.weights[f"{block_prefix}.ffn_norm1.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        norm3_x = ttnn.multiply(norm3_x, scale_mlp, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        mlp_out = self._mlp(ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC), seq_len, block_prefix)
        norm4_out = self._rms_norm_f32(
            mlp_out,
            self.weights[f"{block_prefix}.ffn_norm2.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        x = ttnn.add(
            x,
            ttnn.multiply(gate_mlp, norm4_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        return x

    def _block_no_adaLN(self, x, seq_len, block_prefix, is_caption=False):
        if x.dtype != ttnn.DataType.FLOAT32:
            x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=DRAM_MC)

        norm1_x = self._rms_norm_f32(
            x_3d,
            self.weights[f"{block_prefix}.attention_norm1.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        attn_out = self._attention(norm1_x, seq_len, block_prefix, is_caption=is_caption)
        norm2_out = self._rms_norm_f32(
            attn_out,
            self.weights[f"{block_prefix}.attention_norm2.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        x = ttnn.add(
            ttnn.typecast(x_3d, ttnn.DataType.BFLOAT16, memory_config=DRAM_MC),
            norm2_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )

        norm3_x = self._rms_norm_f32(
            x,
            self.weights[f"{block_prefix}.ffn_norm1.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        mlp_out = self._mlp(ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=DRAM_MC), seq_len, block_prefix)
        norm4_out = self._rms_norm_f32(
            mlp_out,
            self.weights[f"{block_prefix}.ffn_norm2.weight"],
            self.weights["_scale_hidden"],
            self.weights["_eps_hidden"],
            HIDDEN_DIM,
        )
        x = ttnn.add(x, norm4_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MC)
        return x

    def _unpatchify(self, x):
        x = ttnn.reshape(x, [IMG_PATCHES, PATCH_DIM], memory_config=DRAM_MC)
        x = ttnn.reshape(x, [1, 32, 32, 1, 2, 2, 16], memory_config=DRAM_MC)
        x = ttnn.permute(x, [6, 0, 3, 1, 4, 2, 5], memory_config=DRAM_MC, pad_value=0.0)
        return ttnn.reshape(x, [16, 1, 64, 64], memory_config=DRAM_MC)

    # ── Forward ────────────────────────────────────────────────────────────────

    def set_cap_feats(self, cap_cpu_bf16):
        """Upload caption features for a new prompt.

        Must be called before each prompt's denoising loop.  from_torch with
        ReplicateTensorToMesh is the only API that reliably writes to all 4
        BH devices; in-place copy helpers (copy_host_to_device_tensor without
        mesh_mapper) only reach device 0, leaving the other 3 with stale
        caption data that then propagates through the all_reduce ring.

        Args:
            cap_cpu_bf16: CPU BF16 tensor of shape [1, CAP_TOKENS, 2560].
        """
        self._cap_feats_buf = ttnn.from_torch(
            cap_cpu_bf16,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.mesh_device,
            memory_config=DRAM_MC,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_impl(self, latents, timestep):
        """Full forward pass."""
        latent = latents[0]
        cap_feats = self._cap_feats_buf
        x = self._patchify_and_embed(latent)
        adaln_input = self._timestep_embed(timestep)
        cap = self._cap_embed(cap_feats)

        x = ttnn.reshape(x, [1, IMG_PATCHES, HIDDEN_DIM], memory_config=DRAM_MC)
        cap = ttnn.reshape(cap, [1, CAP_TOKENS, HIDDEN_DIM], memory_config=DRAM_MC)

        for i in range(2):
            x = self._block_with_adaLN(x, adaln_input, IMG_PATCHES, f"noise_refiner.{i}")
        for i in range(2):
            cap = self._block_no_adaLN(cap, CAP_TOKENS, f"context_refiner.{i}", is_caption=True)

        joint = ttnn.reshape(
            ttnn.concat([x, cap], dim=1, memory_config=DRAM_MC),
            [1, IMG_PATCHES + CAP_TOKENS, HIDDEN_DIM],
            memory_config=DRAM_MC,
        )
        for i in range(30):
            joint = self._block_with_adaLN(joint, adaln_input, IMG_PATCHES + CAP_TOKENS, f"layers.{i}")

        x = ttnn.slice(joint, [0, 0, 0], [1, IMG_PATCHES, HIDDEN_DIM], [1, 1, 1], memory_config=DRAM_MC)
        x = self._final_layer(x, adaln_input, IMG_PATCHES)
        return [self._unpatchify(x)]

    def forward(self, latents, timestep):
        """Forward pass. Call set_cap_feats(cap_cpu_bf16) before each new prompt."""
        return self._forward_impl(latents, timestep)
