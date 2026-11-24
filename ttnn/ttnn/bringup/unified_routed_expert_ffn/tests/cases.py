# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of ttnn.bringup.unified_routed_expert_moe: one dict per captured call, every model's, appended
(pipelines/prefill-stepwise/method/FORK_TESTS.md). Values are written out literally, never imported from a model.

Keys (per device unless said otherwise):
    mesh, fabric_config         how the mesh is opened
    tokens                      rows of the dispatched buffer (max_dispatch), ROW_MAJOR bf16 DRAM interleaved
    emb, hidden                 the experts' K and N (gate / up [emb, hidden], down [hidden, emb])
    n_experts, experts_per_chip global expert count (length of counts / region offsets) and local experts per chip
    expert_layout               local slot -> global id: "column_major" = (col * mesh_rows + row) * experts_per_chip + l
    weight_dtype                the experts' weights, TILE DRAM interleaved
    max_dispatched_tokens_per_expert, compute_kernel_config, high_precision, activation   the op's arguments
    group_tokens, top_k, hot_experts, cold_experts, bias_std   routing the counts are built from: each mesh column is
        one dispatch group of group_tokens tokens, each picking its top_k experts of n_experts; a few experts forced
        hot (every token picks them: count = group_tokens) and cold (count 0), the rest skewed by a random bias
    weight_scale                randn scale of the weights
    pcc, max_rel_err            tolerance (max_rel_err: ||dev - ref|| / ||ref|| of the worst (chip, expert))
"""

CASES = [
    {
        "id": "qwen38_flash_next-experts-fused-hp-8x4",
        "model": "qwen38_flash_next",
        "task": "P.1",
        "sig": "25fc228d4e",
        "op": "ttnn.bringup.unified_routed_expert_moe",
        "mesh": [8, 4],
        "fabric_config": "FABRIC_2D",
        "tokens": 51680,
        "emb": 2560,
        "hidden": 640,
        "n_experts": 512,
        "experts_per_chip": 16,
        "expert_layout": "column_major",
        "weight_dtype": "bfloat8_b",
        "max_dispatched_tokens_per_expert": 5120,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "high_precision": True,
        "activation": "Silu",
        "group_tokens": 5120,
        "top_k": 10,
        "hot_experts": 2,
        "cold_experts": 8,
        "bias_std": 0.5,
        "weight_scale": 0.02,
        "seed": 0,
        # measured (HiFi4 + fp32 dest, bfp8 weights, fp32 reference on the same rounded weights): min pcc 0.999995,
        # max rel err 0.0035 (bias_std 0.5, 48131 rows) and 0.0038 (bias_std 1.5, 47019 rows); the device output
        # scaled by 1.01 gives 0.0105 and fails
        "pcc": 0.9999,
        "max_rel_err": 0.006,
    },
]
