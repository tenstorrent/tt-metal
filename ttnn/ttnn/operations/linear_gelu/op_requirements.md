# Operation Requirements: linear_gelu

## Definition
- **Formula**: `z = x[b, 0, s, :] @ W + b`; `y = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))`, K = 768, N = 3072
- **PyTorch Reference**:
  ```python
  def linear_gelu_ref(x, w, b):
      B, _, S, K = x.shape
      z = x.reshape(B, S, K).double() @ w.double() + b.double().reshape(-1)
      return torch.nn.functional.gelu(z, approximate="tanh").reshape(B, 1, S, -1)
  ```
- **Import Path**: `from ttnn.operations.linear_gelu import linear_gelu`
- **Function Signature**:
  ```python
  linear_gelu(
      input_tensor: ttnn.Tensor,            # (B, 1, S, 768)
      weight: ttnn.Tensor,                  # (768, 3072)
      bias: ttnn.Tensor,                    # (3072,) or (1, 3072)
      *,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # None -> default_compute_kernel_config()
      memory_config: ttnn.MemoryConfig = None,                     # None -> input_tensor's
  ) -> ttnn.Tensor                                                 # (B, 1, S, 3072)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory, the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, ... (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements: a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf measurement**: device time is `device_kernel_ns` in `<results_dir>/test_results.json` from `eval/eval_test_runner.sh eval/golden_tests/linear_gelu/ <results_dir>` (the `test_op_loose[...]` rows are the flagged perf profiles). Host wall-clock is useless here: descriptor build costs about 300 us per call. Clear the program cache (`device.disable_and_clear_program_cache()`) between A/B variants in a probe.

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = [tile_aligned]
- **SUPPORTED placement axes**: memory_layout = [INTERLEAVED], buffer_type (x) = [L1], out_buffer_type = [L1]
- **SUPPORTED precision**: fp32_dest_acc_en = [True]; any caller math_fidelity honored, default HiFi2
- **Cores**: 2D grid (m over grid rows, n over grid columns), up to 110 on p100a. Regime R1 (W slice resident per core via column multicast) + R2 (x row-block multicast along each grid row).
- **Compute config**: `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=True, math_approx_mode=False; accurate tanh GELU on the packer thread (`GELU_FAST = 0`).
- **Golden baseline**: 23 / 23 supported cells pass (15 grid + 8 flagged loose), 73 xfail_expected, 0 loud. Regression file: 2 pass, 4 refused (`h_non_aligned`).
- **Perf baseline (device us, production placement)**: 8x512 195.6, 16x256 195.7, 32x128 195.7, 8x384 155.0, 4x512 114.1, 8x256 114.2, 2x512 73.1, 4x256 73.0. Beats stock on every row; 2.2x to 3.2x above target.

### [ ] Refinement 1 - Non-aligned S + DRAM placement for x and output

**Goal**: add `"h_non_aligned"` to `SUPPORTED["alignment"]`, `ttnn.BufferType.DRAM` to `SUPPORTED["buffer_type"]` and `ttnn.BufferType.DRAM` to `SUPPORTED["out_buffer_type"]`. All three are native kernel paths:
- `h_non_aligned`: design stage W3. The writer zeroes rows `valid_rows(m)..31` of every tile of a tail tile row (`st == seq_tiles - 1` with `S % 32 != 0`) in the `cb_output_tiles` page after `wait_front` and before the NoC write. Use `noc.async_write_zeros` on the partial faces, the same primitive the reader uses for the bias padding. Gate it with a CT arg (`last_valid_rows = S % 32`) so aligned S compiles it out. Do not mask x instead: a padded x row gives z = b, so GELU(b) != 0 would still land in the padding, and NaN/Inf in x's padding must stay confined to padded rows (matmul rows are independent).
- `buffer_type = DRAM`: R2 (x row multicast) is already built, so DRAM x crosses DRAM once. Correctness is a gate flip; measured as correct (8x512, 2x37, 8x500 at PCC 0.99997).
- `out_buffer_type = DRAM`: the writer's `TensorAccessor` is placement-agnostic; also verified correct.

**Verifier notes**: these are knob-turns, not scheme changes, and they share one writer edit plus three gate flips, so they ship as one refinement. The perf-1 contract (8x512, L1/L1, default config) is already fully in SUPPORTED at Phase 0, so this is the only generality refinement left (TARGET - SUPPORTED is exactly these three values). Every refinement after it is perf. Ship it performant: the zero fill is only on tail-row tiles and must not add a per-tile barrier on aligned rows. Measure DRAM output at 8x512 against the L1 number: 25.2 MB of output at 448 GB/s is about 56 us, streamed over the whole op, so it should not bind at today's 195 us, but record it as a guard row for the perf refinements.

**Done when**: every golden cell with `alignment = h_non_aligned`, `buffer_type = DRAM` or `out_buffer_type = DRAM` passes (73 cells move from xfail_expected to supported_pass), the 4 `h_non_aligned` regression tests in `eval/golden_tests/linear_gelu/test_regression.py` pass (incl. `test_padded_rows_zero` with Inf padding), the acceptance tests' skipped non-aligned / DRAM cases run and pass, verifier CLI loud categories stay 0, and 8x512 L1/L1 device time does not regress.

### [ ] Refinement 2 - Speed up the flagged 8x512 profile: per-tile compute cost

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags 8x512 (4096 tokens) as the mandatory perf target. Exact config: x and output in L1 interleaved, W and bias in DRAM, bf16 TILE, default compute config (HiFi2, fp32_dest_acc_en=True, math_approx_mode=False). Stock 247.5 us, target <= 90.8 us, today 195.6 us. The same kernel path serves 16x256 and 32x128 (identical 195.7 us), so they move together. Region: the per-core compute pipeline. Device time is linear in tile rows per core (about 41 us per +3 rows, i.e. about 2050 cycles per output tile) against about 910 FPU cycles per tile (25 tile products at HiFi2). The sibling op on the same board and K measured its matmul at the FPU roofline, so the packer-thread accurate tanh GELU (`ckernel_sfpu_gelu.h:351`, about 38 SFPU ops per iteration, design lamp L1) is the likely binder. Confirm that by ablation before choosing a lever. Candidate levers from `op_design.md` perf lamps and `ttnn/ttnn/operations/examples/master.md` (`compute_fusion`, `matmul_output_subblock`):
- GELU cost (lamp L1). The `GELU_FAST` variant (`z / (1 + exp(-2u))`) is the same function by an exact identity, but see the contract note below. A cheaper accurate evaluation is also in scope, as is moving part of the activation off the packer if the packer binds and the math thread has slack.
- Subblock / `block_w` co-tune (lamp L2). `block_w = 3` forces `sb_w in {1, 3}`, so the fp32 4-tile DEST budget only reaches 4x1; `block_w in {2, 4, 6}` admits 2x2 / 1x4. Re-solve `block_h` from the ledger formula, do not hand-set it.
- `block_h` / `ACT_DEPTH` (lamp L6): 13 rows per core are blocked 4,4,4,1, and the 1-row tail re-inits the matmul.
- Bias as a DEST seed instead of the 25th K tile (lamp L5): 1/25 of the FPU work, only worth it once the FPU binds.

No SUPPORTED change.

**Verifier notes**: prompt rule (MUST): "the accurate tanh GELU... A faster but different activation does not count", and "A speedup obtained by lowering precision does not count". `GELU_FAST` may be adopted only if it computes the same tanh GELU (not a piecewise / LUT form) and, on the golden model-distributed inputs, PCC and relative RMS against fp64 are no worse than the accurate path (today 0.99997 / 1.175e-2; the bar in the prompt is stock's 1.21e-2..1.30e-2). Keep `GELU_FAST` a single host constant. Precision knobs (fidelity below HiFi2, fp32_dest_acc_en off) are off the table. Knob-turn class: every lever here is a parameter the design already exposed, except the bias seed (needs a `matmul_block` mode that accumulates onto a pre-loaded DEST, which the helper does not expose today; extending the helper is allowed). Any L1 headroom freed by a smaller `block_h` should go to `ACT_DEPTH` / `OUT_DEPTH`, recorded in `l1_ledger.md`.

**Done when**: measured device-ns at 8x512 (exact flagged config) improves and moves toward 90.8 us; precision gate holds (PCC >= 0.9999, rel RMS <= 1.4e-2, and no worse than the Phase 0 baseline in `verification_report.md`); golden suite green; no regression across the config-spanning guard set: one representative per distinct path x placement x alignment (e.g. 2x512 L1/L1, 8x512 DRAM x, 8x512 DRAM out, 8x500 h_non_aligned, 1x32 single-row column group).

### [ ] Refinement 3 - Speed up the flagged small-token rows (1024 / 2048 tokens)

**Type**: perf

**Goal**: the flagged `LOOSE_CASES` 2x512 / 4x256 (1024 tokens: today 73 us, stock 83.5, target 22.7) and 4x512 / 8x256 (2048 tokens: 114 us, stock 135.5, target 45.4), exact production config as in Refinement 2. These are furthest from target (3.2x and 2.5x). Region: a fixed cost of about 32 us (the intercept of the time-vs-rows line) plus under-fill. 1024 tokens uses 88 of 110 cores (8 grid rows x 11 columns, 4 rows each), and the whole W slice of a column (450 KB) is read serially by one sender per column before compute can run past chunk 0. Candidate levers from `op_design.md` and `ttnn/ttnn/operations/examples/master.md` (`shared_input_reuse`, `mcast_topology`, `noc_placement`, `double_buffer`, `trid_double_issue`):
- W sender spread (lamp L3): `McastRotatingSenderConfig` inside each column so chunk j is read by row `j mod active_rows`, putting the 450 KB in parallel. The sibling op measured the single W sender as its small-M binder.
- Grid shape at small M (lamp L7): fewer columns with a larger `core_n_tiles` vs today's 8x11, so more of the 110 cores carry rows.
- First-chunk wait: smaller `block_w` for row-block 0 only, or bank-staggered W reads.

No SUPPORTED change.

**Verifier notes**: runs after Refinement 2 because the per-tile compute cost Refinement 2 attacks is also most of the small-M time (36 tiles x about 2050 cycles is about 55 us of the 73). Re-measure the intercept after Refinement 2 before choosing the lever. All candidates are knob-turns on the existing R1/R2 topology (sender choice, grid split). R6 (K-split + cross-core combine) is a scheme change and is not in scope: it only pays below `tensor_m_tiles < grid_y` (< 320 tokens), which no flagged row reaches. If a lever is specific to small M, gate it by the predicate in `select_blocking` and keep the 4096-token path unchanged.

**Done when**: measured device-ns improves on 2x512 / 4x256 and 4x512 / 8x256 (exact flagged config) toward their targets, golden suite green, precision gate holds, and no regression on 8x512 or on the guard set named in Refinement 2.

### [ ] Refinement 4 - Speed up the flagged 4096-token profile: fill, drain and data-movement residue

**Type**: perf

**Goal**: after Refinement 2 has lowered the per-tile compute cost, return to the flagged 8x512 profile (and 16x256 / 32x128 / 8x384 on the same path) and attack what remains outside the steady-state compute loop. Region: pipeline fill (row-block 0 waits on W chunk 0 by DRAM read + column multicast), the per-row-block x multicast rendezvous (the slowest core of a grid row gates the row), and the store stream (25.2 MB from 110 cores, written per 4-tile subblock). Profile first: if compute is still the binder, file the remaining compute lever here instead. Candidate levers from `ttnn/ttnn/operations/examples/master.md` (`bank_stagger` = lamp L4, `trid_double_issue`, `double_buffer`, `noc_placement`, `split_reader`) and `op_design.md` lamps L4 / L6: x bank stagger across the cores of a row, `ACT_DEPTH = 3` (frees about 114 KB at `block_h = 3`), and write batching across subblocks.

No SUPPORTED change.

**Verifier notes**: filed on cadence as a separate phase because its levers are data-movement (T2) while Refinement 2's are compute. Packing them together would mix two kinds of measurement. Skip with a note in the changelog if the Refinement 2 profile shows no headroom here (the data path is already hidden under compute).

**Done when**: measured device-ns improves on 8x512 (exact flagged config), golden suite green, precision gate holds, no regression on the guard set named in Refinement 2.
