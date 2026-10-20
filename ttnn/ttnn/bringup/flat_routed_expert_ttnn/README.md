# ttnn.bringup.flat_routed_expert, ttnn.bringup.flat_routed_expert_plan, ttnn.bringup.flat_combine_overlap, ttnn.bringup.flat_routed_expert_with_program  (+ flat_drisc_stop, flat_drisc_dump; Python: `ttnn.bringup.flat_routed_expert_ttnn.flat_expert.FlatRoutedExpert`, `FlatExpert`)

## What it does
New op (Marko Staletovic's, no TTNN source on this tree). Blackhole only.
- `flat_routed_expert`: every local expert's gated FFN (gate/up -> act -> down) in ONE program per chip that streams
  each expert's weights from DRAM once through a spatial pipeline (DRAM bank readers -> gate/up cores, x relays,
  down cores writing y at each expert's region). Per-expert counts / regions are read on device, so one cached
  program serves every routing (no host sync, no tilize of x).
  In: x = row-major bf16 dispatch buffer [rows, H] in DRAM (or the gathered tokens + `token_index`), counts /
  regions [1, NG] uint32, the chip's global expert ids, weights in the plan's bank layout (bfp8 or bfp4).
  Out: y [rows, H], bfp8 TILE by default (bf16 TILE with `y_bf16`, bf16 ROW_MAJOR with `y_row_major`), only the
  active experts' rows written.
- `flat_routed_expert_plan`: the C++ layout plan as a dict (role cores, bank layout, done-word cores); the weight
  packer (`flat_expert.flat_weight_regions`) needs it.
- `flat_combine_overlap`: the flat expert and `combine_fabric2d` (ttnn/cpp/.../deepseek_prefill/hybrid_routed_expert_ffn)
  in one program per chip; combine on grid rows 0-1 starts on an expert once all y writers report it. "initial
  version" (docstring).
- `flat_routed_expert_with_program`: the flat expert with another op's MeshProgramDescriptor appended (one program per
  chip; the y writers report each expert to a collector core). Built for `combine_rf/sparse_combine_rf.py` (the
  router-forwarded fp32 combine; see ../combine_rf/README.md). The caller owns the arena.
- `FlatRoutedExpert` (flat_expert.py): the model-facing wrapper. Builds the plan, lays out / caches the weights
  (`cache_prefix`), owns the expert-id table and done words; `__call__(x, counts, regions, ...)` calls the op.
  `FlatExpert` is the Python ProgramDescriptor builder kept as the parity reference.
- Activations: 0 silu, 1 swigluoai, 2 situ, 3 clamped_silu (limit 10), 4 gelu_tanh (nanobind docstring; `ACTS` in
  flat_expert.py).

## Consider it instead of
- `ttnn.bringup.unified_routed_expert_ffn` / `ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`,
  `routed_expert_ffn`, `hybrid_routed_expert_ffn`, or a per-expert `ttnn.linear` loop, for MoE prefill on
  Blackhole with many local experts per chip (<= 64) and >= 256-row expert capacity.
- `ttnn.experimental.deepseek_prefill.hybrid_routed_expert_ffn` + `combine_fabric2d` overlap: `flat_combine_overlap`.
- Faster than unified at every measured M on GLM 5.3 / Kimi K2.7 / K3 shapes (table below); it LOST to unified on K2
  TP2 at M 512 / 2048 and on K2 TP4 at M 2048 (FLAT_EXPERT_WORKLOG.md "TP2 vs existing", "Final defaults").

## Options
Realistic (use by default):

`flat_routed_expert(dispatched_buffer, expert_token_counts, expert_region_offsets, global_expert_ids, gate_up_weights, down_weights, reader_down_weights=None, done_words, intermediate, max_tokens_per_expert, *, ...)`
- `reader_down_weights=None`: required iff the plan has reader tails (`plan["rdown"]`).
- `done_words`: persistent zeroed L1 tensor on `plan["coords"]` (FlatRoutedExpert allocates it).
- `max_tokens_per_expert`: capacity per expert, >= 256.
- `activation=0`: 0 silu, 1 swigluoai, 2 situ, 3 clamped_silu, 4 gelu_tanh.
- `pin=1`: pin the largest expert's weights in chunks of >= pin sub-blocks; 0 off.
- `token_index=None`: [1, rows] uint32; indexed mode, flat row r reads x row token_index[r]; y has token_index's rows.
- `x_pages_per_row=1`: x stored as [rows * P, H / P] (P 8 at H 4096 spreads each row over the 8 DRAM banks).
- `y_row_major=False`: y bf16 ROW_MAJOR, pack-untilized on the down cores.
- `down_fp32=False`: (fork addition) fp32 DEST, full-sync down projection. Realistic: rule 7's reference point
  (GLM-5.3: experts LoFi with fp32 accumulation; glm53_flash_d_p `tt/experts_ag.py` GLM_AG_DOWN_FP32=1); with bfp4
  weights bf16 DEST over K = I inflates y (Measured).
- `math_fidelity=0`: matmul fidelity (0 LoFi; 2 HiFi2; the DeepSeek-V4.1 model passes HiFi4). Rule 7: LoFi with bfp4 /
  bfp8 weights where the end-to-end check holds, else HiFi2; HiFi4 is the max_precision value.
- `routing_weights=None`: (fork addition) [rows of x, 32] FLOAT32 TILE DRAM; h *= the row's weight for its local
  expert (fp32 SFPU) before down. Needs a plan built with `routed=True`.
- `fp8_act_quant=False`: (fork addition) fp8 e4m3 q-dq of h per 32 values, ue8m0 scale, RNE, before down.
- `weights_nk=False`: (fork addition) weights laid out from nn.Linear's [N, K]; every matmul transposes in1 back.
- `hidden=0`: (fork addition) H when x rows are wider; relays read each row's first H values.

`flat_routed_expert_plan(device, hidden, intermediate, experts_per_chip, num_global_experts, max_tokens_per_expert, weights_bf8=False, pin=1, x_bf16=False, h_bf16=False, routed=False)` -> dict. `routed`: (fork addition) the h-options plan (32-row sub-blocks).

`flat_combine_overlap(<flat args as above>, dispatched_metadata, expert_offsets, replicated_global_expert_idx_table, *, num_experts_per_tok, seq_len_per_chip, combine_axis=0, combine_num_links=2, fwd_arrived_semaphore, final_arrived_semaphore, expert_go_semaphore, activation=0, pin=1, down_fp32=False, x_bf16=False, h_bf16=False, y_row_major=True, y_out=None)` -> combine output [1, 1, seq_len_per_chip, num_experts_per_tok, H] bf16 RM. No routing_weights / fp8_act_quant / weights_nk / hidden.

`flat_routed_expert_with_program(<flat args>, *, arena, y, extra, extra_key, collector, counts_addr, writers, activation=0, pin=1, token_index=None, down_fp32=False, x_bf16=False, h_bf16=False, l1_args=[], relay_words=None, math_fidelity=0, weights_nk=False)` -> y. `extra`: the other op's MeshProgramDescriptor (empty = "cache hit expected"); `l1_args`: per appended program, the L1 hand-off args (`sparse_combine_rf` `run.l1_args`). `math_fidelity` / `weights_nk` (P.8; defaults LoFi / [K, N]): as for flat_routed_expert. No routing_weights / fp8_act_quant / hidden.

`FlatRoutedExpert(device, weights, *, m, H, I, gids, n_global, wdtype="bf4", act="silu", pin=1, cache_prefix=None, x_bf16=False, h_bf16=False, weights_nk=False, routed=False, check_exact=False)`; `weights="fake"` = uninitialised weights (perf only). `__call__(x, counts, regions, token_index=None, x_pages_per_row=1, y_row_major=False, down_fp32=False, pack_stochastic_rounding=False, y_bf16=False, math_fidelity=0, routing_weights=None, fp8_act_quant=False, hidden=0)`.

Many `MIMO_FL_*` environment knobs change the plan (flat_expert.py, device/flat_routed_expert_plan.cpp); defaults are the tested ones.

Precision only: per-step checks or a measured end-to-end gap (rule 7) (the same names in every entry point above):
- `x_bf16=False`, `h_bf16=False`: x / h tiles bf16 instead of bfp8; must match the plan the weights were laid out with.
- `y_bf16=False`: tiled y as bf16 instead of bfp8 (not with y_row_major).
- `pack_stochastic_rounding=False`: stochastic rounding in the packers (x tilize, h, y); analysis only in glm53_flash_d_p
  (`tt/experts_ag.py`, GLM_AG_PACK_SRND).

## Measured
- GLM 5.3 at H 4096 / I 2048, 36 experts, bfp4, one chip: `down_fp32` output scale 1.0337 -> 1.0105 vs the quantized
  reference (rel 0.071 -> 0.057); layer-4 golden through the MoE 1.048 -> 1.024; 2.34 ms per MoE call at chunk 2048
  either way (CHANGELOG.md "fp32 DEST down projection").
- GLM 5.3 preset (tests/test_flat_bench.py glm53flash, 4096 x 2048, 36 experts): 2048 rows 9.558 ms, 5120 rows
  23.721 ms (CHANGELOG.md "DeepSeek-V4.1 expert numerics", 2x2 p150 box).
- DeepSeek-V4.1 P.3, 2x2 p150, 11 x 10 grid, chunk 2048, 24 experts / chip, HiFi4, fp32 DEST: op 3.76 ms bfp8 /
  3.66 ms bfp4 per layer vs unified fork 6.37 / 5.99; model experts section 149.3 / 148.0 ms per chunk vs 176.8 /
  173.7 (CHANGELOG.md; known_issues.md DS41-62). Accuracy of the options at V4.1 shapes vs float64: rel
  0.0032-0.0053, PCC >= 0.99998 (CHANGELOG.md, tests/unit/test_ds41_h_options.py).
- DeepSeek-V4.1 EXPERTS_PRECISION=glm + bfp4: layer-0 experts vs CPU rel 0.131, experts 149.2 -> 128.9 ms per chunk
  (deepseek_v41_flash_d_p/bringup/known_issues.md DS41-64).
- Flat vs unified, us per expert, LoudBox (FLAT_COMBINE_OVERLAP_WORKLOG.md section 3): GLM 5.3 M 32 / 512 / 2048:
  52.5 / 114.3 / 436.1 vs unified best 76.5 / 130.7 / 484.9; Kimi K2.7 M 2048: 437.8 vs 557.8.
- One p150a, bfp4, LoFi, 8 balanced experts, us per expert flat / best existing (FLAT_EXPERT_WORKLOG.md "Final
  defaults"): K2 7168x2048 M 2048 453.0 / 573.0; DSv4-F 4096x2048 M 32 35.6 / 77.5; K2 TP4 7168x512 M 2048 310.1 /
  286.3 (flat slower); K2 TP2 M 2048 438.0 vs unified 346.2 (flat slower, "TP2 vs existing").
- MiMo layers, 2x2, traced ms per layer unified -> flat C++: L5 GA+MoE 640 tok/chip 9.85 -> 7.56; L1 SWA+MoE 2048
  tok/chip 11.62 -> 10.17 (FLAT_EXPERT_WORKLOG.md "C++ op"). Host enqueue 0.154 ms per call on 2x2 (same section).
- DeepSeek-V4.1 shape H 5120 I 2304 E 24, one chip, us per expert M 32 / 80 / 256 / 1024: 43.2 / 48.2 / 62.9 / 211.7
  (FLAT_EXPERT_WORKLOG.md "V4.1 large M").
- flat_combine_overlap, LoudBox 8 x 1 ring, us: Kimi balanced 1140.7 vs PR hybrid overlap 1365.6; GLM balanced 810.4
  vs 936.7 (FLAT_COMBINE_OVERLAP_WORKLOG.md 6.19).
- flat_routed_expert_with_program + combine_rf, 2 x 4 LoudBox, V4.1 dims, uniform routing: flat alone 1089 us, fused
  1238 us, serial 1735 us; worst per-token rel err 2.4e-3 (FLAT_COMBINE_OVERLAP_WORKLOG.md "DeepSeek V4.1-Flash: flat +
  combine overlap").
- deepseek_v41_flash_d_p P.8, 2x2 QuietBox, layer 0, chunk 2048 (column combine, 2 passes of 512 home tokens): the
  fused call ~2.05 ms a pass (EXPERTS_PRECISION=glm) / ~2.14 ms (full: HiFi4, bf16 x / h) vs the plain flat op over
  every row 2.2 ms: weight-stream bound, so a pass costs about a whole call (../combine_rf/README.md "Measured").

## Limits
- Blackhole only; plan needs the BH DRAM-reader layout (TT_FATALs in device/flat_routed_expert_plan.cpp).
- H % 256 == 0, I % 32 == 0, capacity % 32 == 0 and >= 256, 1..64 local experts (flat_routed_expert_plan.cpp:251-253).
- x: ROW_MAJOR bf16 DRAM interleaved; page width a multiple of 1 KB (or wide rows with `hidden`). counts / regions:
  [1, NG] uint32 RM DRAM. y_bf16 and y_row_major are exclusive.
- routing_weights / fp8_act_quant: no token_index, no row-major y, routing_weights FLOAT32 TILE DRAM [rows, 32], <= 32
  local experts; the h options refuse subgrids, M-groups, bf16 gate/up DEST, h units, MIMO_FL_X2 (CHANGELOG.md).
- `x_bf16` / `h_bf16` must match the plan the weights were laid out with.
- Known unit failures: 13 in tests/unit/ with options off (19 pass / 13 fail, 2026-10-18): 12 TP2 / TP4 subgrid cases
  TT_FATAL `p.nsg == 1`, and the MiMo `indexed_perf` band (runs faster than its lower bound, 1710341 ns vs 1720090)
  (CHANGELOG.md; known_issues.md DS41-63). After the e4d239c6d81 sync: 26 passed / 13 failed, the same 13; test_ds41_h_options
  7/7; deepseek_v41 test_c_swa_experts vs CPU rel 0.00224936 (CHANGELOG.md, "Sync to e4d239c6d81").
- flat_combine_overlap carries none of the DeepSeek-V4.1 options; flat_routed_expert_with_program has math_fidelity and
  weights_nk only (no routing_weights / fp8_act_quant / hidden: the weighted combine applies the routing weight).
- With the combine_rf L1 hand-off, rows wider than SPC_PIECE (default 7168 B) hang unless SPC_PIECE >= row bytes
  and the fabric payload >= row + 32 B (FLAT_COMBINE_OVERLAP_WORKLOG.md; asserted in sparse_combine_rf.py).
- GLM's experts_ag path only builds at <= 64 experts per chip; it was never run on a 2x2 (known_issues.md DS41-63).
- Not tested on Wormhole or on meshes other than 1 chip, 2x2, LoudBox 8x1 / 2x4 (tests and worklogs).

## How to call
```python
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert
fe = FlatRoutedExpert(mesh, weights, m=cap, H=H, I=I, gids=gids, n_global=NG, wdtype="bf4", act="clamped_silu", pin=1)
y = fe(x_rm_bf16, counts, regions, down_fp32=True)          # y [rows, H] bfp8 TILE
```
(deepseek_v41_flash_d_p/tt/experts.py `_load_flat` / `_flat`; glm53_flash_d_p/tt/experts_ag.py.)
Tests: tests/unit/ (test_flat_routed_expert_op.py, test_flat_expert_indexed_shapes.py, test_flat_expert_mesh.py,
test_flat_expert_yrm.py, test_flat_uneven_split.py, test_ds41_h_options.py), tests/test_flat_bench.py (perf, fake
weights); tests/source.yaml is empty. flat_combine_overlap:
models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py. flat_routed_expert_with_program: deepseek_v41_flash_d_p tt/perf_experts.py::test_weighted_vs_separate /
test_weighted_combine_routes.

## Used by
- deepseek_v41_flash_d_p P.3 / P.4: tt/experts.py `_flat` (EXPERTS_KERNEL=flat, tt/settings.py default): clamped_silu,
  routing_weights, fp8_act_quant, weights_nk, hidden, routed plan, HiFi4, bfp8 / bfp4.
- deepseek_v41_flash_d_p P.8: tt/weighted_combine.py (EXPERTS_COMBINE=weighted): flat_routed_expert_with_program with
  combine_rf's sparse_combine_rf appended, token_index x (the column's gathered q(x)), math_fidelity, weights_nk,
  down_fp32, the plan without the h options (routed=False).
- deepseek_v41_flash_d_p P.12: tt/shared_expert.py (SHARED_EXPERT_IMPL=flat): the shared expert's TP slice as one
  always-on expert (counts [S], regions [0]; intermediate zero-padded to a multiple of 256, the down kernel's K block:
  se6_dcompute.cpp `kt_d % 8`), bfp8 weights_nk, token_index into the routed experts' gathered q(x), y_bf16,
  down_fp32, LoFi / bfp8 x / h (EXPERTS_PRECISION=glm); official numerics: routing_weights = 1 + fp8_act_quant. 2x2,
  1024 rows x 1280 x 5120: 250 us a layer vs 1431 us for the ttnn.linear chain.
- glm53_flash_d_p (via glm53_flash_d_p_lb): tt/experts_ag.py (GLM_EXPERTS_MODE=ag): indexed x, y_row_major,
  down_fp32 (GLM_AG_DOWN_FP32=1 default), clamped_silu.
- glm53_flash_d_p_lb tests (test_flat_*.py); deepseek_v3_d_p test_flat_combine_overlap.py (flat_combine_overlap).

## History
CHANGELOG.md, FLAT_EXPERT_WORKLOG.md, FLAT_COMBINE_OVERLAP_WORKLOG.md (paths in the worklogs are the source branch's).
