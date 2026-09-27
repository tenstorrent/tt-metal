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
]
