# Operation Requirements: ring_joint_scaled_dot_product_attention

## Definition
- **Formula**: per group of G devices along `cluster_axis`, device p:
  `out[p] = softmax(s · Q[p] · keysᵀ + M) · values`, keys / values = every member's K / V at its global positions,
  then the joint keys (rear); `M = −inf` on keys at global position ≥ `logical_n` (and joint rows ≥ the valid
  joint length; causal: key position > query position); `joint_out = softmax(s · JQ · keysᵀ + M) · values`
  (non-causal). `s = scale or 1/sqrt(dh)`. Head mapping: Q head h reads K head `h // (hq/hk)`, V head `h // (hq/hv)`.
- **PyTorch Reference**: `eval/golden_tests/ring_joint_scaled_dot_product_attention/helpers.py::pytorch_ring_joint_scaled_dot_product_attention`
- **Import Path**: `from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention`
- **Function Signature**:
  ```python
  ring_joint_scaled_dot_product_attention(
      input_tensor_q, input_tensor_k, input_tensor_v,
      joint_tensor_q=None, joint_tensor_k=None, joint_tensor_v=None, *,
      logical_n: int, logical_l: int = 0, cluster_axis: int,
      topology=Topology.Linear, num_links: int | None = None,
      is_causal: bool = False, is_balanced: bool = False, scale: float | None = None,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None, memory_config: ttnn.MemoryConfig | None = None,
  ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]
  ```

Kind labels (fused CCL op): every cross-device item is **TRANSPORT** (how bytes cross the fabric) or **OVERLAP**
(hiding the transport behind the attention). Blocking-model classes: **knob-turn** (bundle) vs **scheme-change**
(stands alone) — `op_design.md` § Blocking Model.

## Measured per-chip breakdown (the ranking input for the perf queue)

One call of the perf-flagged shape (Wan2.2 480p: per device (1, 40, 8192, 128), 4-chip ring 1×4
`FABRIC_2D_TORUS_XY`, Ring, 2 links, padded `logical_n = 32760`) at the only SUPPORTED proxy
(`fp32_dest_acc_en=True` → HiFi3 by the fp32 precision policy, Sq = Sk = 8, 106 compute cores / chip), device
profiler kernel zones (probe: `tests/ttnn/unit_tests/operations/ring_joint_scaled_dot_product_attention/probes/focus_breakdown.py`):

| Part | Measured (every chip within 0.1 ms) | vs roofline |
|------|-------------------------------------|-------------|
| op span | ≈ 198.4 ms | compute roofline 18.1 ms (HiFi2) / ≈ 27 ms (HiFi3); target 24.1 ms |
| TRANSPORT (port readers: own sends + every relay) | done at 2.3–4.5 ms | links roofline 3.46 ms — at the link bound, entirely inside the first 2.3 % of the call |
| local compute (compute cores) | end at 183.1 ms (median) / 198.4 ms (max) | ≈ 7× the HiFi3 compute roofline; per unit ≈ 15 ms vs ≈ 2.3 ms of FPU work |
| tail | last wave: 1280 units / 106 cores = 12.08 waves → 8 units run alone for ≈ 15 ms (7.6 %) | — |

**Reading:** local compute dominates with large headroom against the roofline, and the transport is hidden.
Under the fused-CCL rule ("local-compute perf comes after the cross-device ones *unless* the breakdown shows the
local part dominating with headroom"), the first perf refinement is local compute. The DRAM K/V re-read
(≈ 10.7 GB/call at Sq = 16, ≈ 24 ms at the measured 447 GB/s) becomes the binding resource once compute nears its
roofline, so the reuse-multicast scheme-change comes next. OVERLAP comes last, re-measured then. No TRANSPORT
refinement is filed: the measured transport already runs at the link bound and is fully hidden (see
`verification_report.md` § Recommendations).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Golden-run note**: run the golden suite with `EVAL_CAPTURE_L1=0` until the harness limitation in `verification_report.md` § Verifier CLI Summary is resolved — with L1 capture on, every supported cell of this multi-program mesh op fails with `L1ProfilingError: peak_total=0` (a `thread_local` graph tracker missing the mesh thread-pool compile), independent of the op.
> **Test-run note**: under `--dev`, run one fabric config / mesh shape per `run_safe_pytest.sh` invocation (e.g. `-k linear_axis0`, `-k ring_mock`); switching the fabric mesh mid-session under the watcher stalls at teardown (implementer's `test_ring_joint_scaled_dot_product_attention_debug.py`).

### [x] Phase 0 — Core Implementation

- **Regime**: R1 `qstationary_dram_kv` (Q-stationary flash attention, K/V streamed per unit from DRAM; ring all-gather of K/V into persistent DRAM landing buffers on dedicated port cores, landing gated per (unit, remote segment)).
- **SUPPORTED dtype / kv_dtype**: [bfloat16] / [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (compute runs at ≥ HiFi3 when fp32 DEST is on — precision floor policy)
- **SUPPORTED layout / alignment**: [TILE] / [tile_aligned]
- **SUPPORTED shape-derived axes**: attention_kind=[self], kv_heads_mode=[mha], joint_len=[none, aligned]
- **SUPPORTED op-specific axes**: joint_mode ∈ {none, replicated}, mask_mode=[none], is_balanced=[False], seq_padding ∈ {exact, padded}, cluster_axis ∈ {0, 1}, topology ∈ {Linear, Ring}
- **Cores**: every worker core minus `2·num_links` port cores per device (106 of 110 on a QuietBox 2 at 2 links); attention units round-robin head-major
- **Golden baseline**: 110 supported_pass / 0 supported_fail / 1705 xfail_expected / 5984 invalid_skipped (per `verifier_report.json`, `EVAL_CAPTURE_L1=0`)

### [x] Refinement 1 — Numerical configurability: bf16 DEST (perf-focus enabler) + bfloat8_b Q and K/V

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (16-bit DEST: `T_acc` = Float16_b for scores, stats and O; `choose_block_extents()` then picks Sq = 16, Sk = 8 at dh = dv = 128), add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (Q and joint Q) and to `SUPPORTED["kv_dtype"]` (K, V, joint K/V — may differ from Q's dtype), with the per-CB formats derived from the dtypes (the landing buffers, the port CB and `run_pages` already follow the K/V dtype). Cells that fail out of the box go to `EXCLUSIONS`, not their own refinement. This lands the **full config contract of the perf-flagged case** (bf16 / bf16 / `fp32_dest_acc_en=False` / HiFi2 / TILE / Ring / `cluster_axis=1` / padded / MHA / self / no joint — every other knob is already SUPPORTED), so Refinement 3 measures the real config, never a proxy.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Anchor for the perf queue: after this refinement, `test_golden.py::test_op_loose` (the Wan2.2 FOCUS case) must run and pass its gate (bf16/bf16/False tolerance 0.995 / 0.05, 256 sampled rows) — it is xfail today only because of `fp32_dest_acc_en=False`.
- Build the bf16-DEST path performantly (it is what Refinement 3 optimizes): keep the existing grid fill, block-batched reader/writer, and the single `compute_cb_plan` source; the bf16 subblock budget is 8 DEST tiles (the compute kernel's `subblock_w` reads `DEST_AUTO_LIMIT`).
- Precision policy introduced by the verifier (`FP32_DEST_MIN_FIDELITY = HiFi3` in the program descriptor): with fp32 DEST the compute kernel runs at ≥ HiFi3 because HiFi2's Q·Kᵀ operand truncation fails `test_regression::test_large_logits` (rel-RMS 0.0206 vs 0.02; HiFi3 0.0045). Keep that policy for fp32 DEST. Do not apply it to bf16 DEST (the DiT production config runs HiFi2). Measure `test_large_logits` at bf16 DEST against the bf16/bf16/False gate (0.995 / 0.05) and record the result.
- Ledger: `cb_scores`, stats and O follow `fp32_dest_acc_en` both ways (audit 3). Keep it that way. No Float32 page while DEST is 16-bit.

**Done when**: `fp32_dest_acc_en=False` and `bfloat8_b` (dtype and kv_dtype) are in SUPPORTED, the golden suite (`EVAL_CAPTURE_L1=0`) shows 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode, and the perf-flagged LOOSE_CASE passes its golden gate.

**Outcome**: SUPPORTED widened only (dtype/kv_dtype += bfloat8_b, fp32_dest_acc_en += False): every CB format, the landing buffers, the port CB and `run_pages` already followed the dtypes and `T_acc` already followed `fp32_dest_acc_en`, so no kernel or descriptor change. Golden slice (bf8b or bf16-DEST cells, EVAL_CAPTURE_L1=0): 280 passed / 0 failed / 0 XPASS; `test_op_loose` (Wan2.2 FOCUS) passes at Sq=16, Sk=8. `test_large_logits` at bf16 DEST (not a golden cell) measures rel-RMS 0.067 out / 0.085 joint (gate 0.05) at HiFi2 and 0.071 at HiFi4: Q·Kᵀ accumulating in 16-bit DEST at logit std ~36 (host model of 16-bit per-face accumulation reproduces 0.058–0.066). Inherent to the config; held to a 0.10 regression ceiling in the unit precision test.
