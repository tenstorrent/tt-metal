# Changelog: onorm

## Phase 0 — Core Implementation

- **Date**: 2026-07-27
- **Arch**: Blackhole p150 (`ARCH_NAME=blackhole`), 11×10 = 110-core compute grid
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Fused KDA s6 tail
  `out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)` in a single
  compute kernel, with the head-major → flat re-tile performed **in-kernel**
  (`untilize<V_TILES>` → row-major stripe → `tilize<FLAT_TILES>`) so neither `o_norm`
  nor `sigmoid(gate)` round-trips through DRAM. Multi-core from day one:
  `split_work_to_cores(grid, B*ceil(T/32), row_wise=True)`.

- **SUPPORTED at Phase 0**: `dtype=[ttnn.bfloat16]`, `layout=[ttnn.TILE_LAYOUT]`.
  `INPUT_TAGGERS={}`, `EXCLUSIONS=[]`. This **equals TARGET** — `feature_spec.py` pins a
  single KDA bringup cell by design, so there is no generality backlog.

- **Accuracy achieved**: PCC = **0.999993**, max_abs_err = 0.0330, mean_abs_err = 0.0012,
  rel_rms_err = **0.0037**, got/true ratio median = **0.9997** (p5 0.9936 / p95 1.0058,
  std 0.0037 — broad-spread-around-1.0, i.e. ordinary bf16 noise, **no scale bug**).
  Measured on 4 shapes (B×T = 1×32, 1×128, 1×640, 4×256) via
  `test_onorm_precision_baseline.py`. Flat across a 20× range in token count, so the
  multi-core split introduces no accumulation drift. Against the golden bf16 bar
  (PCC 0.995 / RMS 0.04) there is ~10× margin on RMS and three orders of magnitude on
  `1 − PCC`; the design's `cb_rstd → Float32` precision contingency (risk 11) is not needed.

- **Golden suite at Phase 0**: **5 / 5 registry cells passing** (`verifier_report.json`).
  `supported_pass=5`, `supported_fail=0`, `xpass_drift=0`, `xfail_wrong_mode=0`,
  `xfail_expected=0`, `invalid_skipped=0`, plus 6 non-registry numerics regression tests
  passing (`no_axes_found`). 11/11 golden tests, 0 hangs.

- **Device-ns baseline** (median of 5 trial-major interleaved trials, default knobs
  `DM_BLOCK_TILES=8 / DM_DEPTH=4 / O_DEPTH=2 / NORM_CHUNK_TOKENS=8 / GATE_CHUNK_TILES=64 /
  TOKENS_PER_BLOCK=32`, default compute config):

  | Shape | token-blocks | cores | device ns | achieved GB/s |
  |---|---|---|---|---|
  | B=1, T=64  | 2   | 2   | 239,587 | 6.3 |
  | B=1, T=128 | 4   | 4   | 239,690 | 12.6 |
  | B=1, T=640 | 20  | 20  | 244,110 | 61.3 |
  | B=8, T=640 | 160 | 110 | 538,989 | 221.4 |

  Per-core phase breakdown at B=1/T=640: **P7b sigmoid 152.7 µs (63.9 %)**, P4 29.3,
  P7c 19.2, P1 11.9, P5 9.4, P2 8.6, P6 4.3, P7a 3.6 (TRISC total 239.1 µs).

- **Key verification finding**: the op is **SFPU-bound, not DRAM-bound** — per-core achieved
  bandwidth is 3.1 GB/s against a ~17.9 GB/s single-core NoC ceiling, and every compute-only
  config knob moves the runtime (`fp32_dest_acc_en=False` 1.095×, LoFi 1.047×, `math_approx`
  1.030×, all three 1.112×). This contradicts `op_design.md`'s central premise and voids its
  "that lever will not pay because we are DRAM-bound" dismissals. The whole refinement queue
  is re-derived from it. Also independently confirmed the design's risk 14:
  `dst_full_sync_en=True` is a silent **0.931×** (7 % slower) with no correctness signal.

- **Issues encountered** (all fixed in this pass unless noted):
  1. **CB slot map restated in two places** — host `CB_*` constants vs. twelve literals in the
     three kernels. Renumbering either side would have silently mis-addressed a buffer. Fixed by
     injecting the map through `KernelDescriptor(defines=…)` from a single host-side `_CB_SLOTS`;
     kernels now read `ONORM_CB_*`.
  2. **`V` never checked against the tile width** — a non-multiple-of-32 `V` interleaves TILE
     padding into the flat feature span and mis-maps every output index, with no numeric signal.
     Added `head_dim % TILE_W == 0` assert.
  3. **No placement contract** — a sharded input would have been read correctly by
     `TensorAccessor` but its resident L1 is invisible to the CB budget. Added an interleaved-only
     host assert (`memory_config` is not a TARGET axis, so an assert, not an axis refusal).
  4. **Page-size assert silent and incomplete** — omitted `weight`, no message. Fixed.
  5. **`DM_BLOCK_TILES` divisibility assert did not name the escape route.** Fixed.
  6. **Golden-suite defect** (`eval/golden_tests/onorm/feature_spec.py`): `INPUTS[4]` declared
     `weight` as `(2, 1, 1, V)`, contradicting the contract documented in the same file and the
     reference itself — the case raised a torch broadcast error inside the harness *before any
     device tensor was built*, so no op-side change could ever have cleared it. This was the run's
     only `supported_fail`. Corrected to `(1, 1, 1, V)`; the planner had flagged it in
     `op_design.md` §13.
  - **Not changed, noted only**: a caller who supplies `dst_full_sync_en=True` gets a measured
    7 % regression silently; `PROPERTIES["math_fidelity"]` describes only the default config;
    `_L1_CB_BASE_RESERVE = 72 KB` is a Blackhole-measured constant that needs re-deriving on
    Wormhole. See `verification_report.md` → Recommendations.

- **Blocking-model audit**: clean. All six knobs (`TOKENS_PER_BLOCK`, `NORM_CHUNK_TOKENS`,
  `GATE_CHUNK_TILES`, `DM_BLOCK_TILES`, `DM_DEPTH`, `O_DEPTH`) are live, single-source-of-truth
  parameters; no CB is sized by a whole-op dimension; no half-turned split (per-core loops run at
  8 tokens / 64 tiles, not 1); the L1 budget assert fires before the runtime's bare throw and
  names the knobs to lower in order. Verified by `test_onorm_knobs.py`, 14/14.

- **Prompt-rule audit**: all hard (MUST / MUST NOT) rules satisfied. One soft rule deliberately
  and correctly not followed — the rules prescribe `tilize<StreamMode::PerTile>` with a 2-tile
  `cb_flat`, which would **deadlock** here because tilize and its consumer share the same compute
  kernel and therefore the same three TRISCs. `StreamMode::Atomic` with a full-block
  `cb_flat_tiles` is the correct deviation and produces bit-identical bytes.

- **Tests added**: `test_onorm_precision_baseline.py` (new, this pass — PCC / abs / rel-RMS /
  got-true-ratio scale-bug detector over 4 shapes). Pre-existing and retained:
  `test_onorm.py` (acceptance, immutable spec, 11/11), `test_onorm_knobs.py` (14/14),
  `test_onorm_trials.py` (40/40), `test_onorm_perf.py` (5/5), `test_onorm_dmsweep.py`.

- **Refinement queue**: 3 phases, **all perf** (`TARGET − SUPPORTED = ∅`, so there is no
  generality work to file). R1 — move `sigmoid(gate)` off the saturated MATH thread (the 64 %
  phase). R2 — cross-core re-tile so small-`T` shapes stop leaving 96–99 % of the grid idle
  (scheme-change, stands alone). R3 — reconfig-off + block-factor co-tune against the final
  structure. See `op_requirements.md`.
