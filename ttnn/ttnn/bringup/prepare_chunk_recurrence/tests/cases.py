# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.prepare_chunk_recurrence call a model makes (bringup-fork-tests skill). Append
only; never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # GLM-5.3-Flash KDA chunk preparation (tt/kda_attention.py, GLM_KDA_DECAY=fork: _ForkPrepRecurrence wraps
        # deepseek_v3_d_p/tt/kda/recurrence.py:_prepare_chunk_terms with precise_gate_factors=True) on the 2x2 mesh,
        # SP over axis 0: this chip's 2560 rows of a 5120-token chunk, the 32 heads it holds (TP 2 of 64), K = V = 128.
        # q / k / v / gate [1, 2560, 4096] bf16 TILE; beta = the fused input projection [1, 2560, 16544] bf16 TILE
        # holding the 32 pre-sigmoid beta logits at columns 16512 .. 16543 (beta_logits_column_offset). The gate input
        # is the sigmoid of the forget-gate projection, scaled on device by gate_scale = -5 (GLM's lower bound).
        # actual_start = the chunk start (51200, the ladder's last chunk), no actual_end: every row valid.
        # HiFi4 + fp32 DEST, no approx; outputs DRAM interleaved, bf16 for kd / q_decay / final_decay (mask 0x26).
        "id": "glm53_flash_d_p-2x2-sp0-t2560-h32-k128-v128-gs-5-betacol16512-precise",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "09fddf83cd",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "rows": 2560,
        "num_heads": 32,
        "key_dim": 128,
        "value_dim": 128,
        "beta_columns": 16544,
        "beta_logits_column_offset": 16512,
        "gate_scale": -5.0,
        "actual_start": 51200,
        "sequence_parallel_axis": 0,
        "output_bf16_mask": 38,
        "precise_gate_factors": True,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # vs the float64 reference on the same bf16 inputs, per output. Measured (seed 0, 4 chips), rel L2 / pcc:
        # v_beta 4.0e-4 / 1.0000000, kd 3.9-4.0e-3 / 0.9999982 (bf16 out), q_decay 3.0-3.1e-3 / 0.9999983 (bf16 out),
        # intra 4.56e-3 / 0.9999994, k_dec_t 2.84-2.88e-3 / 0.9999997, final_decay 1.9-2.0e-3 / 0.9999984 (bf16 out),
        # t_inv 2.0e-4 / 1.0000000 and strictly-lower max abs 1.2-1.3e-3. Most of the kd / q_decay / k_dec_t / intra
        # error is a systematic low bias (about -0.3 / -0.2 / -0.2 / -0.45 %, the same on every chip). Limits about
        # 1.5x the rel L2 (t_inv max abs 0.003; the source op's test allows 0.01). Checked by hand: an output scale of
        # 0.99 fails every limit; 1.01 fails all but intra (it cancels intra's bias: rel 0.0056).
        "tol": {
            "v_beta": {"pcc": 0.99999, "rel": 0.0008},
            "kd": {"pcc": 0.9999, "rel": 0.006},
            "q_decay": {"pcc": 0.9999, "rel": 0.005},
            "intra": {"pcc": 0.9999, "rel": 0.007},
            "k_dec_t": {"pcc": 0.9999, "rel": 0.0045},
            "final_decay": {"pcc": 0.9999, "rel": 0.004},
            "t_inv": {"pcc": 0.99999, "rel": 0.0004, "lower_max_abs": 0.003},
        },
    },
]
