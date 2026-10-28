# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct program a model runs through ttnn.bringup.prepare_program / run_prepared (bringup-fork-tests
skill, F201). A program is named by fork_capture.program_summary's digest (its kernels, defines, compile-time args,
cores, CBs; never an address); a case covers that program's prepare_program call ("sigs") and its run_prepared call
("sig"). Append only; never edit or loosen another model's case.

io positions are the program's io_tensors list: "move" the ones the model's call site moves between calls (prepared
patching), "restore" the DRAM ones whose data is reloaded before every call (written in place by the program, or
a persistent input buffer the call site fills), "compare" its outputs."""

_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    # the model's mesh: 2D fabric, l1_small, the weighted combine's 10880 B fabric packets (bringup/hooks.py)
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    # MOE_IMPL=block (tt/moe_block_adapter.py): Marko's MoE block, MOE_HEAD fused, MOE_TAIL fused, EXPERTS_LAYOUT marko
    "E": 96,
    "K": 6,
    "H": 5120,
    "I": 2304,
    "limit": 10.0,  # (clamped SiLU, routed and shared)
    "route_scale": 1.5,  # (sqrtsoftplus router)
    "chunk": 2048,
    "links": 2,
    "combine_snap": 2.0**-19,
    "shared_fidelity": "LoFi",
    "layout": {"rows_hint": 128, "x_helpers": 1},
    "w_std": 0.02,
}

CASES = [
    {
        # deepseek_v41_flash_d_p MOE_IMPL=block: the fused MoE head (models/demos/moe_block/tt/fused_head.py HeadSE3, one
        # program a chip): x [1024, 2560] (the chip's column split) gathered over TP on its ports into the shared expert
        # engine (bfp8 weights, LoFi, clamped SiLU 10) with its row reduce-scatter to the owners (y [512, 5120]), the
        # sqrt(softplus) router (top 6 of 96, weights x 1.5) into the joint routing rows, the routing exchange, the
        # combine planner (counts, regions, token index, table, plan) and the dispatch prep (xbuf, route). Prepared once
        # per layer set; every layer's call moves the shared expert's and the router's weights (layers mode) - x is the
        # head's own buffer, its data per call (restored: A's or B's); moved here too (its args are its base address).
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-fused-head-s2048-e96-top6-i2304-lofi",
        "sig": "ef9baff5aa",  # run_prepared
        "sigs": ["f0e39b029d"],  # prepare_program
        "digest": "dfe7d6fa1d59",
        "program": "fused_head",
        "n_io": 16,
        # io: 0 x, 1 shared expert weights, 2 y, 3 arena (L1), 4 rows, 5 xbuf, 6 route, 7-9 router weight / bias /
        # (its index table), 10 routing slots (the exchange's own scratch), 11-15 counts, regions, token_index, table, plan
        "move": [0, 1, 7, 8, 9],
        "restore": [0],
        "compare": [2, 4, 5, 6, 11, 12, 13, 14, 15],
        # vs float64 (test _check_head; owner 2026-10-28: limits = the measured error with a margin). Measured on seeds
        # 0 / 1 / 2, A and B each (6 calls, worst chip): shared rows rel L2 0.0494-0.0495, PCC 0.99964 (bfp8 weights,
        # LoFi); picks equal on 100% of the tokens clear of noise, 99.22-99.80% of all (near ties); weights max abs err
        # 1.000-1.016e-3 (~1 bf16 ulp at 0.25). Limits: rel 0.06, PCC 0.9995, all picks 0.98, weights 2e-3; clear 100%.
        "ref_limits": {"se_rel": 0.06, "se_pcc": 0.9995, "picks_all": 0.98, "w_err": 2e-3},
        "seed": 0,
    },
    {
        # deepseek_v41_flash_d_p MOE_IMPL=block: the token exchange's sparse 2D dispatch (combine_rf
        # sparse_all_gather_2d, prepared by ttnn.bringup.prepared_program.generic_op; the exchange arena's zeroing
        # folded in): each chip's own rows of xbuf [2048, 5120] bf16 to the chips its route words [512] name, into their
        # xbuf. Prepared once; the model reuses its xbuf / route (the head's outputs), so the moves here go beyond the
        # model's own use: the prepared contract (exact-address patching).
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-exchange-dispatch-s2048-h5120",
        "sig": "bef5e0fc64",  # run_prepared
        "sigs": ["54aa3bbe0e"],  # prepare_program
        "digest": "befea287463f",
        "program": "exchange_dispatch",
        "n_io": 2,
        # io: 0 route words, 1 xbuf (in place: the own rows in, every chip's rows out)
        "move": [0, 1],
        "restore": [0, 1],
        "compare": [1],
        "seed": 1,
    },
]
