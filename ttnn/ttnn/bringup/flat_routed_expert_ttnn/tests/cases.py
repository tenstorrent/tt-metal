# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.flat_routed_expert / flat_routed_expert_with_program call a model makes
(bringup-fork-tests skill). Append only; never edit or loosen another model's case. Shapes are per device, exactly as
captured (fork_calls.json); the op's plan, weight layout and done words come from the fork's FlatRoutedExpert with the
literal configuration below."""

_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    # the model's mesh: 2D fabric, l1_small, the weighted combine's 10880 B fabric packets (bringup/hooks.py)
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    "H": 5120,
    "act": "clamped_silu",
    "pin": 1,
    "weights_nk": True,
    "x_bf16": False,
    "h_bf16": False,
    "down_fp32": True,
    "math_fidelity": 0,  # LoFi (EXPERTS_PRECISION glm)
    # the FlatRoutedExpert's layout: a flat_layouts preset name or a dict of FlatLayoutOptions fields. LEGACY (the plan
    # before the DRISC-only / even-split defaults) for the calls made before settings.EXPERTS_LAYOUT=marko (2026-10-27)
    "layout": "LEGACY",
}

CASES = [
    {
        # deepseek_v41_flash_d_p tt/shared_expert.py (P.12, SE_PARTS 1): the chip's 1152 intermediate columns of the
        # shared expert (zero-padded to the op's 256-column down block: 1280) as ONE always-on expert over the chip's
        # 1024 own rows, read by token index (r 1024 + t) from the column's gathered rows [2048, 5120] bf16 ROW_MAJOR;
        # bfp8 weights (W1 | W3 | W2 as stored [N, K]), y bf16 TILE. Random weights rounded to exact bfloat8_b values
        # (check_exact), x ~ N(0, 1) bf16.
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-shared-expert-s1024-i1280-bfp8-lofi",
        "sig": "9df16e185a",
        "op": "flat_routed_expert",
        "I": 1280,
        "epc": 1,
        "n_global": 1,
        "m": 1024,
        "S": 1024,
        "wdtype": "bf8",
        "w_std": 0.02,
        "y_bf16": True,
        "seed": 0,
        "noise": {"inputs": ["bf16"], "out": "bf16"},
    },
    {
        # deepseek_v41_flash_d_p tt/experts.py + tt/weighted_combine.py (P.8-P.11, FIX.9; EXPERTS_IMPL bfp4,
        # COMBINE_ACC bf16, COMBINE_PLANNER multi, CCL_IMPL fabric_bf16): the chip's 24 routed experts (of the column's
        # 48, of 96) over the column's 2048 tokens (top 6 of 96 each), and the weighted combine to the home chips in the
        # same program (extra = the combine's MeshProgramDescriptor on the first call of a layer), one pass of 1024
        # home tokens, bf16 running sums and output. Ids: 6 distinct random experts a token; weights U(0.5, 1.5) bf16;
        # x ~ 0.3 N(0, 1) bf16; expert weights random, exact bfloat4_b values (check_exact).
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-routed-weighted-combine-s1024-e96-top6-i2304-bfp4-lofi",
        # F179 stable sig (extra_key, an id()-derived program-cache key, left out; DS41-110). Same call as the
        # pre-F179 sigs 284a4a9ead / 31755c8401, which differ only in extra_key, and as 3fc158f72b before the rebase
        # onto llk_helper_library 6b80daf43e3, whose l1_args list the same cores in another order.
        # EXPERTS_LAYOUT=marko (2026-10-27): the same call on the op's default layout plus the model's rows_hint
        # (round(chunk 2048 x top 6 / 96 experts)) and one x helper; was sig 2aefb66a5a on LEGACY.
        "sig": "c6e50c3022",
        "op": "flat_routed_expert_with_program",
        "layout": {"rows_hint": 128, "x_helpers": 1},
        "I": 2304,
        "epc": 24,
        "E": 96,
        "K": 6,
        "S": 1024,
        "wdtype": "bf4",
        "w_std": 0.02,
        "out_dtype": "BFLOAT16",
        "acc_dtype": "BFLOAT16",
        "planner": "multi",
        "seed": 1,
        "noise": {"inputs": ["bf16", "bf16"], "out": "bf16"},
    },
    {
        # deepseek_v41_flash_d_p tt/shared_expert.py after FIX.1 (no padding to get around a kernel limit): the same
        # call as the shared-expert case above, with the chip's 1152 intermediate columns at their real width (36
        # tiles, down weights [6912, 256] per bank shard), no zero padding to a 256-column down block.
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-shared-expert-s1024-i1152-bfp8-lofi",
        # EXPERTS_LAYOUT=marko (2026-10-27): the same call on the op's default layout plus rows_hint = the chip's 1024
        # rows and one x helper (no reader down weights on that plan); was sig a274529a9f on LEGACY.
        "sig": "e2065c8d71",
        "op": "flat_routed_expert",
        "layout": {"rows_hint": 1024, "x_helpers": 1},
        "I": 1152,
        "epc": 1,
        "n_global": 1,
        "m": 1024,
        "S": 1024,
        "wdtype": "bf8",
        "w_std": 0.02,
        "y_bf16": True,
        "seed": 2,
        "noise": {"inputs": ["bf16"], "out": "bf16"},
    },
    {
        # deepseek_v41_flash_d_p MOE_IMPL=block (tt/moe_block_adapter.py, Marko's MoE block; MOE_HEAD / MOE_TAIL fused,
        # EXPERTS_LAYOUT marko): the block's token exchange (models/demos/moe_block TokenExchange ->
        # tt_k3_token_split_moe.py routed_only) calls the op with its own fused weighted combine (sparse_combine_rf,
        # fp32 sums, each term snapped to 2^-19, accumulators planned from the experts' free cells) over the 2x2's
        # 96 experts, 24 a chip in row-major chip order (block ids), capacity 2048 rows an expert, y rows bf16 in L1,
        # no math_fidelity argument (LoFi). Built as the block builds it: TokenExchange on a FlatRoutedExpert with the
        # block's ids; the routing gather, planner and dispatch are the exchange's ops path (the fused head does the
        # same three in its own program; the experts' call is the same). Per chip 512 own tokens, x ~ 0.3 N(0, 1) bf16,
        # top 6 distinct random experts a token, weights U(0.5, 1.5) normalised to sum 1.5 (route_scale), bf16, as
        # the joint routing rows; expert weights random, exact bfloat4_b values (check_exact).
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-block-exchange-s512-e96-top6-i2304-bfp4-lofi",
        "sig": "03f82eaff0",
        "op": "flat_routed_expert_with_program",
        "via": "moe_block_exchange",
        "layout": {"rows_hint": 128, "x_helpers": 1},
        "I": 2304,
        "E": 96,
        "K": 6,
        "chunk": 2048,
        "links": 2,
        "route_scale": 1.5,
        "combine_snap": 2.0**-19,
        "wdtype": "bf4",
        "w_std": 0.02,
        "x_std": 0.3,
        "seed": 3,
        "noise": {"inputs": ["bf16", "bf16"], "out": "bf16"},
    },
]
