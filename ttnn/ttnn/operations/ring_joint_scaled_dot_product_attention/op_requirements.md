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


### [x] Refinement 1b — Numerical configurability: bf16 DEST (perf-focus enabler) + bfloat8_b Q and K/V (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 36/391). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome**: not an op defect. The gate runs with `EVAL_CAPTURE_L1=1`, and graph capture could not see this op's CBs: the
op is a heterogeneous (one program per chip) MeshWorkload, which `MeshWorkloadImpl::compile` built on the mesh thread
pool while the GraphTracker is `thread_local`. `peak_total` was then 0 (-> `L1ProfilingError`) unless the call happened to
create the cached global semaphores, so which cells "passed" depended on order (36/391 both phases, a different 36). Fixed
in the runtime, capture-only: compile on the calling thread while a capture is active; report a workload program's CBs
only on the devices of its range; key the per-core peak by (device, core). Harness runner (`eval/eval_test_runner.sh`,
L1 capture on): 391/391 supported pass, 0 regressions, 0 hangs, real per-chip L1 peak 113 KB - 1.27 MB. Also bullet 2:
the unit-test files each imported the session fabric-mesh fixture, giving one session cache per module and a
`!devices_still_open` setup error when run together; the fixtures now live in the directory conftest (91/91 pass).

<!-- Restored by the run operator: the Refinement 1 agent's commit bb7819f002d deleted the verifier's
     pending Refinements 2-6 from this queue, which made the orchestrator treat the queue as drained. -->

### [x] Refinement 2 — Causal prefill family: causal + zig-zag balanced, GQA / MQA / MLA heads

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`, `True` to `SUPPORTED["is_balanced"]`, and `"gqa"`, `"mqa"`, `"mla"` to `SUPPORTED["kv_heads_mode"]`, natively in the kernels:
- **Causal (design regime R4)**: extend the shared `key_block_plan` (reader and compute use the same function, so push/wait counts stay equal by construction). Each (unit, key block) is classified {skip, full, diagonal-masked} from global positions, contiguous (`p·n + i`) or zig-zag balanced (chunk p then chunk 2G−1−p, `helpers.global_positions`). Skipped blocks are neither read nor computed, so K/V reads scale with keys_seen. Diagonal blocks get a causal mask tile pattern. The first processed block of every unit must still hold ≥ 1 valid key (the online-softmax finite-max cap, `op_design.md` § Mechanism caps). The output stays in the caller's token layout (prompt rule: no cross-device reordering).
- **Head sharing (design regime R3, a knob-turn of `q_heads_per_unit`)**: Q head h reads K head `h // (hq/hk)` and V head `h // (hq/hv)`; the transport already moves (b, kv head) entries, so the remaining change is the unit's K/V head index plus MLA's `dv ≠ dh` (DHt up to 18 at dh = 576: `choose_block_extents()` must still fit L1 and fall back to a smaller Sq). Units of `hq/hk` heads sharing one key-block stream (K/V DRAM reads ÷ hq/hk) are the intended realization. Expose the factor as a parameter from one source; don't collapse it.

**Verifier notes**:
- Ordering: the hardest remaining generality work (reshapes the key plan and the unit decode), so it goes before the joint/cross variants (Refinement 4), which extend the same plan.
- Every causal INPUTS shape is GQA / MQA / MLA, and the ring-mock causal case is GQA 16:1 with `kv_dtype=bfloat8_b`. That is why heads and causal ship together, and why this follows Refinement 1 (bf8b K/V).
- Causal work is imbalanced across chips in the contiguous layout (p = G−1 sees the most keys): measure the busiest chip.
- A causal unit's remote segments that are entirely in its future are skipped: no landing wait for them (OVERLAP: fewer waits, not more).

**Done when**: those five axis values are in SUPPORTED, the golden suite (incl. `test_ring_mock.py` causal-balanced cells and `test_regression.py`) shows 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode, and no regression on prior cells.
**Outcome**: all five values in SUPPORTED. Busiest chip, GQA 8:1, n=2048/chip, 4-chip ring, bf8b K/V, bf16 DEST: contiguous 1.23 ms (chips 1.23/0.86/0.55/0.35) → balanced 1.04 ms (0.84/0.89/1.04/0.97). Balancing does not reach the mean (~0.75 ms) at this size because each core holds about one unit and the round-robin split assumes equal-cost units, so the costliest unit sets the wall. Next lever: causal-cost-aware unit assignment (pair early and late units per core); left for the perf rounds.

### [ ] Refinement 3 — Speed up the perf-flagged Wan2.2 case: local compute (attention) — the measured dominant part

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (attention: PERF FOCUS) — Wan2.2 A14B 480p, per device (1, 40, 8192, 128), bf16 / bf16, `fp32_dest_acc_en=False`, HiFi2, 1×4 `FABRIC_2D_TORUS_XY`, Ring, `cluster_axis=1`, 2 links, `logical_n=32760`; `roofline_us = 18076.5`, `target_us = 24101.9`, soft gate (0.995 / 0.05 on 256 sampled rows). The measured breakdown above puts ≈ 98 % of the call in local compute at ≈ 7× its roofline (per unit ≈ 15 ms vs ≈ 2.3 ms of FPU work at the fp32 proxy). Speed up the per-key-block compute chain on exactly this config. Levers to weigh from `ttnn/ttnn/operations/examples/master.md`:
- the unmodelled SFPU work, chiefly the full-tile exact `exp` over Sq·Sk score tiles per block (design lamps "exp approximation" and "scale placement", gated on the soft precision gate);
- `compute_block_size` (≈ 10 helper phases per key block, each with an init and data-format reconfig; reconfig ablation where formats are constant at bf16 DEST);
- `sfpu_tile_scope` (column-0 stats ops: α exp, recip, max on Sq tiles);
- `compute_fusion` / `row_reduce_accumulate` (row sum of P: second L1-acc pack vs `reduce<SUM>`);
- `matmul_output_subblock` (score / PV subblocks are height 1 today);
- the block co-tune Sq × Sk with the "single O accumulator" lamp, which frees `Sq·DVt·T_acc` of L1 (ledger fold-in);
- the partial last wave (640 units / 106 cores = 6.04 waves at Sq = 16).
No SUPPORTED change.

**Done when**: measured device time (eval `ccl_perf` protocol: fastest chip, slowest chip guarded) improves on the perf-flagged case toward `target_us` with its soft precision gate still holding, the golden suite is green, and there is no regression across the config-spanning guard set (one representative per distinct kernel path × placement: no-joint / replicated-joint, Linear G=2 / Ring G=4, fp32 / bf16 DEST).

### [ ] Refinement 4 — DiT joint / cross variants: sharded joint, non-aligned replicated joint, cross-attention

**Goal**: add `"sharded"` to `SUPPORTED["joint_mode"]`, `"non_aligned"` to `SUPPORTED["joint_len"]` and `"cross"` to `SUPPORTED["attention_kind"]`, natively:
- **Sharded joint (design regime R5)**: each device holds an L_dev = roundup32(ceil(L/G)) slice and `logical_l = L`. The joint K/V slices travel the ring as extra transport entries into joint landing slots, `key_block_plan` gains remote joint segments (rows ≥ `logical_l` masked), and joint Q stays local, so `joint_out` is the device's own slice. **TRANSPORT** (new entry kind on the existing relay) + **OVERLAP** (joint remote segments gated like the main ones).
- **Non-aligned replicated joint**: L not a multiple of 32 (SD3.5 333, Motif 334, Mochi 118). Second key-tail mask tile (`MASK_TILES` → 2, ledger symbol `mask_tiles ≤ 2`), the joint q-block tail, and `joint_out` rows ≥ L unspecified. Inputs arrive tile-padded; mask in-kernel, never through a host pad / `ttnn.fill_implicit_tile_padding` wrapper.
- **Cross-attention**: `n_q ≠ n_kv` (short Q, LTX-2 video→audio). Decouple the Q tile count from the K/V tile count in the unit decode and the key plan; non-causal; `logical_n` bounds the keys only; every query row is valid.

**Verifier notes**:
- Ordering: after Refinement 2 (it extends the same `key_block_plan` and unit decode) and Refinement 3 (the joint-heavy short-sequence Flux cases sit near the link bound, so build on the faster compute path).
- Sharded joint is the only item here that changes the transport. Reuse the existing entry / slot / arrival-count protocol: one more entry kind, the count stays an entry index (`blackhole-fabric.md` rule 5).

**Done when**: those three axis values are in SUPPORTED, the golden suite (incl. `test_ring_mock.py` sharded-joint cells and `test_fabric_configs.py`) shows 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode, and no regression on prior cells.

### [ ] Refinement 5 — Perf-flagged case: K/V reuse multicast across a grid row (design regime R2) — scheme-change

**Type**: perf

**Goal**: on the perf-flagged Wan2.2 case (same full config contract as Refinement 3), remove the K/V DRAM re-read. R1 streams each head's K/V from DRAM once per Q block: Uq = 16 times at Sq = 16, ≈ 10.7 GB/call ≈ 24 ms at the measured 447 GB/s, which is the target itself. Once Refinement 3 brings compute near its 18 ms roofline, DRAM is the binding resource. Implement the deferred R2 `qstationary_kv_row_mcast`: the cores of one grid row hold consecutive Q blocks of the same (b, head) per wave (already true of the head-major round-robin order). One injector per row (or `injectors_per_row`, each owning every k-th key block) reads each K/V block once and multicasts it along the row (`mcast_pipe` `SenderPipe` / `ReceiverPipe`); `await_landing` moves to the injector only. This is a new dataflow topology, so it stands alone. Patterns: `shared_input_reuse` (1.71×) and `mcast_topology` in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: validate by building the variant and measuring device time, not by a remove-a-stage ablation (a broadcast's benefit is under-shown by one). Re-measure the per-chip breakdown first. If, after Refinement 3, the case is still compute-bound with DRAM well below its bound, record that and keep the scheme for when it binds. The row group is the compute grid minus the port cores (the port cores sit in logical grid row 0 next to the Ethernet links — measured placement on the 1×4 ring — so derive the row groups on the grid that is left).

**Done when**: measured device time (fastest chip; slowest chip guarded) improves on the perf-flagged case with its soft precision gate holding, the golden suite is green, and there is no regression across the config-spanning guard set (as in Refinement 3, plus a GQA / causal representative from Refinement 2).

### [ ] Refinement 6 — Perf-flagged case: OVERLAP — exposed transport per chip (first-wave landing, landing notification, tail chain)

**Type**: perf

**Goal**: **OVERLAP**, measured in the op on the perf-flagged Wan2.2 case (same full config contract), after Refinements 3 and 5 have shrunk the local work: measure the transport time the op leaves exposed per chip, then remove it. Exposed transport = remote-segment landing waits of the first wave(s) plus the chain from the last landing to the output write. Phase 0 already processes local blocks before any remote one, but a faster unit leaves less local work to hide the first landings behind (heads arrive head-major at ≈ 9 heads/ms against a 3.46 ms link roofline). Levers from `op_design.md` § Perf lamps / Stall-shadow:
- unit / wave ordering that front-loads heads whose remote entries land first;
- landing notification: compute readers poll the port counter by NoC read today (no back-off), against the port reader multicasting landed counts;
- the landing increment interval (`inc_every` within an entry, TRANSPORT-side knob, from `fabric_all_gather`);
- launch-skew tolerance between chips (`ccl_perf` spread metric).

No SUPPORTED change.

**Verifier notes**: measured today (fp32 proxy) the transport is done by 2.3–4.5 ms of a 198 ms call, so exposure is unobservable until the local work shrinks. Measure first, with a per-chip breakdown via device-profiler zones around the landing waits. If the measured exposure is within noise of zero, record the measurement and close with no change. TRANSPORT itself (link rate, ring balance split, packet size) is not queued: the port readers already finish at the 3.46 ms link roofline (`fabric_all_gather` measured 35.9 GB/s busiest link at 2 links).

**Done when**: the exposed transport per chip on the perf-flagged case is measured and reported; any change lands only with a measured device-time win (fastest chip; slowest chip and spread guarded), the golden suite green, and no regression across the guard set.
