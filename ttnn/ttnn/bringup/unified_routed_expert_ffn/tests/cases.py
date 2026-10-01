# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.unified_routed_expert_moe call a model makes (bringup-fork-tests skill). Append
only; never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # MiMo-V2.6 routed experts (SwiGLU), EP=4 on a 1x4 mesh: 64 local experts per chip, bfp8 weights.
        "id": "mimo_v2_6_d_p-1x4-silu-hp-hifi4-h4096-i2048-epc64",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "31daae9fc6",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # dispatched buffer [42976, 4096] BFLOAT16 ROW_MAJOR; regions / counts [1, 256] UINT32 ROW_MAJOR;
        # global_expert_idx_table [64] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer_rows": 42976,
        "emb_dim": 4096,
        "hidden_dim": 2048,
        "num_routed_experts": 256,
        "experts_per_chip": 64,
        # The routing the buffer comes from: 5120 tokens x top-8, every chip's own random ids.
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 8,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT8_B", "layout": "TILE"},  # gate/up [4096, 2048], down [2048, 4096] per expert
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "Silu",
        "high_precision": True,
        "seed": 0,
        # x ~ N(0, 1), weights ~ N(0, 1/fan_in). Measured on this case (seed 0, 4 chips, ~10.2k routed rows each):
        # PCC 0.999996, rel Frobenius error 0.00303-0.00304 on every chip (bf16 intermediates vs the float32
        # reference). Limits leave ~2.6x on rel; an output scaled by 1.01 (rel ~0.0105) fails.
        "pcc": 0.9999,
        "rel": 0.008,
    },
    {
        # Gemma-4 A4B routed experts (GeluTanh gate), EP=4 on a 1x4 mesh: 32 local experts per chip, bf16 weights,
        # HiFi2, default precision path (high_precision not passed: bfp8 activations / output).
        "id": "gemma4_a4b_d_p-1x4-gelutanh-hifi2-h2816-i704-epc32",
        "model": "gemma4_a4b_d_p",
        "task": "O.1",
        "sig": "befc6cf758",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # dispatched buffer [41952, 2816] BFLOAT16 ROW_MAJOR; regions / counts [1, 128] UINT32 ROW_MAJOR;
        # global_expert_idx_table [32] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer_rows": 41952,
        "emb_dim": 2816,
        "hidden_dim": 704,
        "num_routed_experts": 128,
        "experts_per_chip": 32,
        # The routing the buffer comes from: 5120 tokens x top-8, every chip's own random ids.
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 8,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT16", "layout": "TILE"},  # gate/up [2816, 704], down [704, 2816] per expert
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi2",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "GeluTanh",
        "high_precision": False,
        "seed": 0,
        # x ~ N(0, 1), weights ~ N(0, 1/fan_in). Measured (seeds 0 and 1, 4 chips, ~10.1k routed rows each):
        # PCC 0.999780-0.999781, rel Frobenius error 0.02408-0.02411 on every chip; the floor is the bfp8 output
        # (and bfp8 activations) of the default precision path vs the float32 reference. Limits leave ~1.7x on rel;
        # an output scaled by 1.05 (measured rel 0.066) fails.
        "pcc": 0.999,
        "rel": 0.04,
    },
    {
        # MiMo-V2.6 routed experts (SwiGLU) on a 2x2 mesh: dispatch groups are the 2 columns (2 chips each), chip
        # (r, c) holds the 64 experts (2c + r) * 64 .. + 63 (ExpertMapping col-major), bfp8 weights.
        "id": "mimo_v2_6_d_p_2x2-2x2-silu-hp-hifi4-h4096-i2048-epc64",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "86afb6a289",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # dispatched buffer [42976, 4096] BFLOAT16 ROW_MAJOR; regions / counts [1, 256] UINT32 ROW_MAJOR;
        # global_expert_idx_table [64] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer_rows": 42976,
        "emb_dim": 4096,
        "hidden_dim": 2048,
        "num_routed_experts": 256,
        "experts_per_chip": 64,
        # The routing the buffer comes from: a dispatch group's 2 x 2560 = 5120 tokens x top-8 over its 128 experts.
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 8,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT8_B", "layout": "TILE"},  # gate/up [4096, 2048], down [2048, 4096] per expert
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "Silu",
        "high_precision": True,
        "seed": 0,
        # x ~ N(0, 1), weights ~ N(0, 1/fan_in); Measured (seed 0, 4 chips, 10.2-10.4k routed
        # rows each): pcc 0.999996, rel 0.00303-0.00304; limits as the 1x4 case.
        "pcc": 0.9999,
        "rel": 0.008,
    },
    {
        # GLM-5.3-Flash routed experts (ClampedSiluGlu = silu(min(gate, 10)) * clamp(up, +/-10)) on a 2x2 mesh:
        # dispatch groups are the 2 columns (2 chips each), chip (r, c) holds the 72 experts (2c + r) * 72 .. + 71
        # (ExpertMapping col-major), bfp8 weights (the checkpoint is FP8), HiFi4 + fp32 dest, high_precision.
        "id": "glm53_flash_d_p-2x2-clampedsiluglu-hp-hifi4-h4096-i2048-epc72",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "820d12513b",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # dispatched buffer [43232, 4096] BFLOAT16 ROW_MAJOR; regions / counts [1, 288] UINT32 ROW_MAJOR;
        # global_expert_idx_table [72] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer_rows": 43232,
        "emb_dim": 4096,
        "hidden_dim": 2048,
        "num_routed_experts": 288,
        "experts_per_chip": 72,
        # The routing the buffer comes from: a dispatch group's 2 x 2560 = 5120 tokens x top-8 over its 144 experts.
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 8,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT8_B", "layout": "TILE"},  # gate/up [4096, 2048], down [2048, 4096] per expert
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "ClampedSiluGlu",
        "high_precision": True,
        # gate / up weights ~ N(0, 16 / fan_in): gate and up ~ N(0, 16), so ~0.6% of gates and ~1.2% of ups hit the
        # +/-10 clamp; the reference without the clamp is off by rel 0.073 (CPU, 512 rows), so a missing clamp fails.
        "gate_up_scale": 4.0,
        "seed": 0,
        # Measured (seed 0, 4 chips, 10.0-10.5k routed rows each): pcc 0.999997, rel 0.00250 on every chip. Limits
        # leave ~3x on rel; a 1.01 output scale fails (checked by hand), and so would a missing clamp (rel ~0.07).
        "pcc": 0.9999,
        "rel": 0.008,
    },
    {
        # Hy4 (preview) routed experts (SwiGLU clamped at 10: ClampedSiluGlu) on a 2x2 mesh: dispatch groups are the 2
        # columns (2 chips each), chip (r, c) holds the 64 experts (2c + r) * 64 .. + 63 (ExpertMapping col-major),
        # bfp8 weights.
        "id": "hy4_preview_d_p-2x2-clampedsiluglu-hp-hifi4-h6144-i2048-epc64",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "4fe753a443",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # dispatched buffer [42976, 6144] BFLOAT16 ROW_MAJOR; regions / counts [1, 256] UINT32 ROW_MAJOR;
        # global_expert_idx_table [64] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer_rows": 42976,
        "emb_dim": 6144,
        "hidden_dim": 2048,
        "num_routed_experts": 256,
        "experts_per_chip": 64,
        # The routing the buffer comes from: a dispatch group's 2 x 2560 = 5120 tokens x top-8 over its 128 experts.
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 8,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT8_B", "layout": "TILE"},  # gate/up [6144, 2048], down [2048, 6144] per expert
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "ClampedSiluGlu",
        "high_precision": True,
        # x ~ N(0, 16): the gate / up projections are ~N(0, 16), so about 0.6% of the gate values sit above the limit
        # 10 and 1.2% of the up values outside +-10; dropping the clamps moves the output by rel 0.075 (host check).
        "x_scale": 4.0,
        "seed": 0,
        # x ~ N(0, 16), weights ~ N(0, 1/fan_in). Measured (seed 0, 4 chips, 10.2-10.4k routed rows each): pcc
        # 0.999997, rel 0.00250; limits as the other 2x2 case. An output scaled by 1.01 fails (checked by hand).
        "pcc": 0.9999,
        "rel": 0.008,
    },
    {
        # Xing4.0 routed experts (tt/experts.py) on the 4x2 mesh: 8 experts per chip, SwiGLU (Silu), hidden 3584,
        # moe intermediate 1024, high_precision, HiFi4 + fp32 dest + packer_l1_acc. Captured per device: buffer
        # [20704, 3584] bf16 RM, regions / counts [1, 64] uint32 RM, global ids [8] uint32 RM, 8 x gate / up
        # [3584, 1024] and 8 x down [1024, 3584] bfp8 TILE (the model's bfp8 weights). seq_len_per_chip here is the
        # dispatch group's token count (4 chips x 1280), so the counts are as loaded as the model's.
        "id": "xing40_a4b_d_p-4x2-silu-hp-hifi4-h3584-i1024-epc8",
        "model": "xing40_a4b_d_p",
        "task": "O.1",
        "sig": "c30d4661c3",
        "mesh": [4, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "buffer_rows": 20704,
        "emb_dim": 3584,
        "hidden_dim": 1024,
        "num_routed_experts": 64,
        "experts_per_chip": 8,
        "seq_len_per_chip": 5120,
        "num_experts_per_tok": 4,
        "buffer": {"dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "weights": {"dtype": "BFLOAT8_B", "layout": "TILE"},
        "max_dispatched_tokens_per_expert": 8192,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        },
        "activation": "Silu",
        "high_precision": True,
        "seed": 0,
        # measured (seed 0, 8 chips, ~2500 routed rows each): pcc 0.999996, rel 0.0030-0.0031 vs the reference on the
        # bfp8-rounded weights; limits as the other HiFi4 high_precision cases (a 1.01 output scale, rel ~0.01, fails).
        "pcc": 0.9999,
        "rel": 0.008,
    },
]
