# ttnn.bringup.moe_grouped_topk

## What it does
Fork of `ttnn.experimental.deepseek_prefill.moe_grouped_topk`
(`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk` @ `f3149480fb1`, which holds Marko
Staletovic's `fp32_selection` commits). Adds `fp32_activation`: the score activation is compiled with the fp32 SFPU
polynomials (`INP_FLOAT32`, what the stock unary op uses for an fp32 input), so `sqrt(softplus(x))` matches stock
`ttnn.sqrt(ttnn.softplus(x))` on fp32 bit for bit. The source compiles the bf16-grade softplus (degree 6 on [0, 5],
0 below -5): ~8e-4 relative off on [-3, 9], up to 0.08 absolute below, which swaps near-tied experts even with
`fp32_selection`. Everything else as the stock op: activation (sigmoid / sqrtsoftplus), bias add for the selection,
grouped or single-group top-k, gather of the unbiased scores, normalization, route scale, in one program.

## Consider it instead of
- `ttnn.experimental.deepseek_prefill.moe_grouped_topk(score_func="sqrtsoftplus")` (DeepSeek-V4.x routers): with
  `fp32_activation=True` the keys are the fp32 softplus (a fix; the reference computes softplus in fp32).
- a router op chain after the gate matmul (`ttnn.softplus` / `ttnn.sigmoid`, `ttnn.sqrt`, `ttnn.add`, `ttnn.topk`,
  `ttnn.gather`, `ttnn.sum`, `ttnn.add`, `ttnn.multiply`, `ttnn.div`): one program, ~10x less host time.

## Options
`moe_grouped_topk(scores, bias, *, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
route_scale=1.0, epsilon=1e-20, stable_sort=False, score_func="sigmoid", memory_config=None, padding_config=None,
biased_scores=None, weights_layout=TILE, fp32_selection=False, fp32_activation=False)`
(`moe_grouped_topk_nanobind.cpp`).

Realistic (use by default):
- `fp32_activation=False` (fork, opt-in): the fp32 SFPU softplus (sqrtsoftplus). False = the source program
  (the define is absent). Sigmoid is unchanged either way (no INP_FLOAT32 branch). A fix, not precision only: the
  bf16-grade polynomial is below the fp32 the reference (and stock ttnn.softplus) computes.
- `fp32_selection=False` (source): fp32 DEST reads, SFPU bias add, fp32 sort keys and SFPU normalization.
- All other arguments as the source. Outputs: weights BFLOAT16 (TILE or ROW_MAJOR), indices UINT16 TILE. The bias
  has the scores' full shape.

Precision only: none.

## Measured
- `tests/unit/test_fp32_activation.py` (one p300c chip): keys == stock `ttnn.sqrt(ttnn.softplus(x))` fp32 bit for
  bit on [-30, 30], max 2.5e-5 from float64 (off: 8.2e-2); DeepSeek-V4.1 routing (1 group, top 6 of 96 / 384,
  1024 / 64 rows): selected sets == float64 on every row with a 6th / 7th gap > 1e-5, bf16 weights rel <= 2.5e-3
  (one bf16 rounding ~1.6e-3); off: 1-2 of 1024 rows select other experts on random logits.
- DeepSeek-V4.1 scaled golden (2x2, 1024 rows a chip, 96 experts, 8 layers; probes tests/ttnn/unit_tests/operations/ds41_router/probes/probe_005.py (keys) and probe_006.py (routing),
  `models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md` P.22): selected sets == the model's old op chain on
  2048 / 2048 rows of every layer, weights rel 1.5e-3..1.6e-3 vs its fp32 weights (= one bf16 rounding); off: 0-13
  rows a layer differ (keys max error 8.2e-2). One call 0.070 ms host vs the 9-op chain's 0.58 ms.

## Limits
- As the source: TILE fp32 / bf16 scores and a full-shape bias; n_groups 1 (any tile-aligned expert count, k <= 64)
  or 8 x 32 = 256 experts with the DeepSeek-V3 group parameters; weights bf16 only.
- `fp32_activation` only changes sqrtsoftplus; tested single-chip and in DeepSeek-V4.1's 2x2 model (replicated
  router, 1024 rows a chip).
- Source tests carried: the op's five nightly files, 203 cases, 203 / 203 pass on the source and the fork
  (`tests/source.yaml`, `tests/source_baseline.json`).

## How to call
```python
wts, idx = ttnn.bringup.moe_grouped_topk(
    logits, bias_rows,                      # [1, 1, S, E] fp32 TILE; the bias repeated over S
    n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=6,
    route_scale=1.5, epsilon=1e-20, stable_sort=True, score_func="sqrtsoftplus",
    fp32_selection=True, fp32_activation=True,
)  # wts [1, 1, S, 6] bf16 TILE, idx [1, 1, S, 6] uint16 TILE
```
Caller: `models/demos/deepseek_v41_flash_d_p/tt/router.py:TtRouter._fused`. Tests: `tests/unit/`; model cases
`tests/cases.py` + `tests/test_moe_grouped_topk.py` (O.1).

## Used by
deepseek_v41_flash_d_p (P.22)

## History
CHANGELOG.md
