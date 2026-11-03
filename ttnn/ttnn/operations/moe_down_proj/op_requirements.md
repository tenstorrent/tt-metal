# Operation Requirements: moe_down_proj

## Definition
- **Formula**: `y[0, 0, r, :] = x[0, 0, r, :] @ M_e^T` for every real row r of expert region e, with
  `M_e = w[0, 0, :, e*F:(e+1)*F]` and regions `start_e = start_{e-1} + 32 * ceil(counts[e-1] / 32)`. Non-real rows
  are unspecified. E = 8, H = 768, F = 3072.
- **PyTorch Reference**: `pytorch_moe_down_proj` in `eval/golden_tests/moe_down_proj/helpers.py` (fp64, returns
  `(y, real)`; compare only rows where `real` is True).
- **Import Path**: `from ttnn.operations.moe_down_proj import moe_down_proj`
- **Function Signature**:
  ```python
  moe_down_proj(
      input_tensor: ttnn.Tensor,            # x, (1, 1, R, 3072), bfloat8_b TILE
      weight: ttnn.Tensor,                  # w, (1, 1, 768, 8 * 3072), bfloat8_b TILE, DRAM
      expert_counts: ttnn.Tensor,           # (1, 8) uint32 ROW_MAJOR, DRAM, read on device only
      *,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # None -> LoFi, fp32_dest_acc_en=True
      memory_config: ttnn.MemoryConfig = None,                     # None -> input_tensor's
  ) -> ttnn.Tensor                          # (1, 1, R, 768) bf16 ROW_MAJOR interleaved
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory - the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, ... (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements - a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

Queue shape. `TARGET - SUPPORTED` has exactly two values, `buffer_type = DRAM` and `out_buffer_type = DRAM`, both
knob-turns in the Blocking Model, so they form one generality refinement. The perf-flagged contract (every
`LOOSE_CASES` row: x and output in L1, weight and counts in DRAM, default LoFi + fp32 DEST, bfloat8_b TILE) is
already fully in SUPPORTED at Phase 0, so no unlock is needed in front of the perf slots. Once generality is
exhausted, the remaining phases are perf. The primary perf gap is large: 173-179 us against a 90.8 us target at
t = 4096, and 103-140 us against 74.6 us below.

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat8_b] (x and weight)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **SUPPORTED buffer_type (x)**: [L1]
- **SUPPORTED out_buffer_type**: [L1]
- **SUPPORTED shape-derived axes**: none (`INPUT_TAGGERS = {}`; R is always a tile multiple)
- **Cores**: 108 / 110 (L layout: 12 n-lines x 9 m-groups, `core_n_tiles` = 2); x multicast per m-group, w multicast per n-line, rotating senders
- **Compute config**: caller-provided honored; default LoFi + fp32_dest_acc_en; `fp32_dest_acc_en=False` refused
- **Golden baseline**: 27 supported_pass / 54 xfail_expected / 0 loud (93 total, 12 regression tests untagged and passing)
- **Device kernel time (us)**: t4096 L0..L5 173.2 / 174.5 / 175.0 / 178.9 / 173.7 / 173.1; t3072 140.0; t2048 113.7; t1024 102.8 (all under stock; L1 is 0.3 us under stock)

### [x] Refinement 1 - DRAM placement for x and the output

**Goal**: add `ttnn.BufferType.DRAM` to `SUPPORTED["buffer_type"]` (x) and to `SUPPORTED["out_buffer_type"]`
(output), covering all three combinations (DRAM x / L1 out, L1 x / DRAM out, DRAM both). Both are knob-turns in the
Blocking Model: the x senders and the y writer already go through placement-agnostic `TensorAccessor`s. A verifier
probe with the gate patched open passed numerically on all three combinations at t = 37 and t = 4096 layer 0
(PCC 0.99991, rel RMS 1.90e-2, identical to the L1 path). The work is the gate flip plus making the DRAM path
performant:
- DRAM x adds 27.5 MB of DRAM reads at R = 8416 next to the 20 MB w stream (47.5 MB, about 106 us at 448 GB/s). The
  x sender's staged prefetch (`PREFETCH_ISSUE_STEPS` quota) must keep enough x reads in flight that the DRAM-x
  cells are not paced by read latency. Measure, and retune the x-side issue quota / staging if needed. The NoC
  split is the opposite of the w channel's (x reads on NoC1), so measure the NoC assignment for DRAM x as well.
- DRAM output: the 128 B row-segment writes (32 per tile row) go to DRAM pages. Measure, and if the store becomes
  exposed (last wave's drain), coarsen it. The `num_waves >= 2` overlap already hides earlier waves.
- `CB_BUDGET_BYTES` (960 KiB) was set against x and y both resident in L1 at R = 8416. When either lives in DRAM,
  more L1 is free, and `select_blocking` may take one wave (no straddling-expert w re-read) or a deeper buffer.
  This is a knob-turn. Derive the budget from the placement (shape- and config-derived only, so the program cache
  stays valid) rather than one constant, and keep the production L1 / L1 budget unchanged.

**Verifier notes**: no skill in the inventory covers interleaved DRAM vs L1 placement (`/memory-layouts` is RM vs
TILE). Native kernel support only. No `ttnn.to_memory_config` wrapper on the entry point: the prompt forbids
host-side tensor transformations, and the kernels already read and write through accessors. Land this before the
perf refinements so their guard set includes the DRAM placements.

**Done when**: all 54 currently `xfail_expected` golden cells (`buffer_type=DRAM` and/or `out_buffer_type=DRAM`)
pass, `test_dram_placement` (6 cells) passes instead of skipping, `verify_supported` is clean, device-ns for
t = 4096 layer 0 and t = 1024 in each DRAM combination is recorded in the changelog, and the production L1 / L1
rows do not regress.

**Outcome**: all 54 DRAM golden cells and `test_dram_placement` pass. Device us (t = 4096 layer 0 / t = 1024):
L1/L1 172.0 / 102.4, DRAM x 221.2 / 117.2 (gate-flip only: 218.2 / 107.1, noisy at t = 1024), DRAM y 220.4 / 112.2
(was 253.9 / 123.9), DRAM both 258.8 / 117.2 (was 291.3 / 126.7). The DRAM y store now runs on the reader RISC
(NoC1). A write-stubbed ablation puts the DRAM y store at about 105 us of work at t = 4096 (128 B segments). The
next lever is gathering each row's 12 segments onto one core and writing full 1536 B rows, a cross-core change. DRAM x
is bound by its 27.5 MB of NoC1 DRAM reads: the x issue quota (4 / 8 / 11 / 16) and one wave at a smaller K block
were measured flat. Moving x reads to NoC0 needs DM_DYNAMIC_NOC, which the mcast helper's TXN_ID flush does not
support yet.



### [x] Refinement 1b — DRAM placement for x and the output (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 2 FAIL: acceptance/refinement tests failing:
  - tests/ttnn/unit_tests/operations/moe_down_proj/test_moe_down_proj.py::test_moe_down_proj[t1] - AssertionError: pcc nan < 0.9998
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome**: not a kernel bug. The inline precompile collect pass (`run_safe_pytest.sh` on a whole file/dir) runs bodies with stand-in `torch.randn` / shape-only `from_torch`, and `host_weight()`'s `lru_cache` carried that stand-in weight into the real pass (constant output and reference, pcc nan). The conftest now clears module-level caches per pass via a module-scoped autouse fixture. Unit dir 72/72 with precompile on; golden 93/93 (graded mode, no hangs).
### [ ] Refinement 2 - Cut the multicast scaffold on the t = 4096 rows

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `t4096_layer0` (8x512, R = 8416, x and output in L1, weight and counts
in DRAM, default LoFi + fp32 DEST) as the primary perf target: stock 179.0 us, `target_ns` 90.8 us. Today it runs
at 173.2 us. Layers 1-5 share the shape and target, and layer 1 is only 0.3 us under its 174.8 us stock. The
implementer's ablation puts the sync + multicast floor at about 130 us of the 173 us at t = 4096, with compute about
15 us and w reads about 2 us. The multicast scaffold binds, not the math or the DRAM stream. Each core receives
about 3.1 MB of x (low x reuse: `core_n_tiles` = 2) and about 1.9 MB of w, mostly chunks for experts its rows never
touch. In the L layout the x sender also sends each round twice, serially (row multicast plus strip copy). Levers,
most structural first:
- **Range-pruned w delivery (regime R2, deferred in `op_design.md`)**: deliver expert e's chunk only to the
  n-line cores whose rows touch e. That cuts w receive per core from about 1.9 MB to about 0.3 MB. This is a new
  delivery topology (runtime sub-rectangle multicast with per-expert readiness, since `ttnn.Mcast` groups are
  static), which is why it gets its own phase.
- **Layout re-solve for the pruned traffic**: once w receive no longer scales with `E / num_n_lines`, the
  receive-byte optimum moves to fewer n-lines (larger `core_n_tiles`, more x reuse per received tile). Re-score
  `select_layout` on the busiest core's received bytes as well as its output tiles.
- **x strip copy**: take the serial second send off the x sender's critical path (for example, a different core or
  NoC issues it, or the strip core joins the row multicast).
- Patterns: `ttnn/ttnn/operations/examples/master.md` (`mcast_topology`, `shared_input_reuse`, `noc_placement`,
  `bank_stagger`). These are T3 levers, so this is a whole phase. No SUPPORTED change.

**Verifier notes**: scheme-change: a reuse-multicast topology change, standalone. Any change to the per-core K round
order changes the fp32 summation order. LoFi rel RMS is 1.90e-2 against a 2.0e-2 bar, so gate on
`test_moe_down_proj_precision_baseline.py`. Validate by building and measuring device-ns, not by ablation.

**Done when**: device-ns improves on `t4096_layer0` toward the 90.8 us `target_ns`, every t = 4096 layer row
(0-5) improves or holds and stays under its stock, PCC >= 0.9998 and rel RMS <= 2.0e-2 still hold on the golden
inputs, the golden suite is green, and there is no regression across the config-spanning guard set: t = 1, 37,
1858, 4096 one_expert and zeros_between, plus each placement combination from Refinement 1, on both layouts via
`layout_override`.

### [ ] Refinement 3 - Speed up the DRAM-bound rows (t = 1024 / 2048 / 3072)

**Type**: perf

**Goal**: the flagged `t3072_scaled`, `t2048_scaled` and `t1024_scaled` rows (same production config) have a
74.6 us `target_ns`, set by the DRAM term: 20.05 MB of w at 448 GB/s / 0.60. They run at 140.0, 113.7 and 102.8 us.
The implementer's t = 1024 ablation: sync floor about 46 us, w reads about +50 us, so the w stream runs at roughly
200 GB/s. These are T1/T2 knob co-tunes on the surface the planner exposed, so several fit in one phase:
- w-sender lookahead depth and `PREFETCH_ISSUE_STEPS` (lamp L4, catalog `trid_double_issue`), and `W_DEPTH` 2 vs 3:
  keep more DRAM reads in flight across the 12 rotating line senders.
- DRAM bank stagger for the w page walk (lamp L7 analogue, catalog `bank_stagger`, 1.32x measured on BH).
- `num_waves` / `k_block_tiles` co-tune (lamps L1, L2). At t = 3072 two waves re-read one straddling expert
  (+2.5 MB, about +5.6 us of DRAM time). Check whether one wave at a smaller K block wins there.
- Store issue cost (lamp L6): stateful `noc_async_write_one_packet_with_state` for the 128 B row segments.
- Block-size floor: whole tiles minimum, coarser amortizes. Name the axis and direction from measurement, not a
  fixed number. Pointer: `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: runs after Refinement 2 because R2 changes who receives w chunks and can shift the sender
balance. Tune these knobs on the final topology. If `/perf-ceiling-dm` shows the w stream at ceiling after R2,
record that and close the phase.

**Done when**: device-ns improves on at least one of `t1024_scaled`, `t2048_scaled` and `t3072_scaled` toward
74.6 us, none of the three regresses, every t = 4096 row holds, the precision gates hold, the golden suite is green,
and there is no regression across the same guard set as Refinement 2.
