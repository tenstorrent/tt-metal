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

---

## Refinement 1 — Get `sigmoid(gate)` off the saturated MATH thread

- **Date**: 2026-07-27
- **Arch**: Blackhole p150 (`ARCH_NAME=blackhole`), 11×10 = 110-core compute grid
- **Outcome**: `[~]` partial. Both named leads were built and measured; both are closed
  with evidence. A third lever was found, measured and adopted. The one large remaining
  lever is now priced and filed as **Refinement 1b**.

### What was done

1. **`SIGMOID_ENGINE` knob** (`onorm_program_descriptor.py`, one source of truth → kernel
   preprocessor defines, exactly like the existing `_CB_SLOTS` map). Three values:
   - `"math"` (**default**, unchanged behaviour) — `unary<Sigmoid<>>`, SFPU on TRISC1.
   - `"pack"` — the design's lead 1. SFPU on TRISC2 at the pack stage, via
     `ActivationInitHelper<SIGMOID>::init()` + `apply_activation_from_pack<SIGMOID>()`
     from `sfpu_activation_helpers.hpp`. Correct, live, non-default.
   - `"ablate"` — measurement only. Drops the sigmoid, keeps every CB wait/push, DEST
     window and NoC transfer. Numerically wrong by construction, so it is double-gated
     behind `ALLOW_SIGMOID_ABLATION` and covered by a test that proves the gate holds.

2. **`GATE_DEST_TILES` knob (adopted at 4)** — the third lever, and the only one that
   won. Phase 0 opened one DEST window **per tile** in both gate phases, because
   `InputLifecycle::Streaming` clamps the chain's `block_size` to 1. Both P7b **and its
   1:1 twin P7c** moved to `InputLifecycle::Chunked` / `OutputLifecycle::Chunked` +
   `OperandKind::Block`, so tiles-per-DEST-window is now a live tunable capped by
   `_DEST_TILE_LIMIT` (= `DEST_AUTO_LIMIT` = 4 under `fp32_dest_acc_en` + half sync).
   Both halves were coarsened in the same pass — blocking one alone would only have
   pushed the per-window cost onto the other.

3. **Shared-helper bug fix** — `ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp`:
   the SIGMOID branch of `ActivationApplyHelper` declared `constexpr int vec_mode` but
   `sigmoid_tile_pack`'s first template parameter is typed `VectorMode` (a scoped enum),
   so the branch failed to compile the first time it was ever instantiated. Latent dead
   code in a shared header; fixed in place.

### Measurements (median of 5 trial-major interleaved trials, `test_onorm_sigmoid_engine.py`)

Engine × block factor, `DEVICE KERNEL DURATION [ns]`:

| Shape | cores | math/1 (Phase 0) | math/2 | math/4 | pack/1 | ablate/1 |
|---|---|---|---|---|---|---|
| B=1,T=128 | 4 | 239,803 | 238,761 (1.004×) | **238,256 (1.006×)** | 241,979 (0.991×) | 83,978 (2.856×) |
| B=1,T=640 | 20 | 244,561 | 243,606 (1.004×) | **243,387 (1.005×)** | 246,733 (0.991×) | 92,145 (2.654×) |
| B=8,T=640 | 110 | 538,677 | 537,713 (1.002×) | **535,756 (1.005×)** | 543,377 (0.991×) | 305,906 (1.761×) |

Non-regression across the full **config-spanning guard set**, paired (dest=1 vs dest=4)
inside one process — every cell improves, none regresses:

| Shape | config | cores | dest=1 | dest=4 | speedup | max spread |
|---|---|---|---|---|---|---|
| B=1,T=32  | default | 1   | 239,551 | 238,119 | 1.0060× | 0.03 % |
| B=1,T=128 | default | 4   | 239,699 | 238,366 | 1.0056× | 0.03 % |
| B=1,T=640 | default | 20  | 244,317 | 243,301 | 1.0042× | 0.16 % |
| B=8,T=640 | default | 110 | 538,161 | 536,625 | 1.0029× | 0.47 % |
| B=1,T=640 | `math_approx_mode=True` | 20 | 236,413 | 235,456 | 1.0041× | 0.11 % |
| B=1,T=640 | `MathFidelity::LoFi`    | 20 | 233,626 | 232,484 | 1.0049× | 0.18 % |

Per-phase zone breakdown re-recorded at the new defaults (B=1/T=640, per TRISC):
P7b sigmoid **155.5 µs of a 238.5 µs TRISC kernel (65.2 %)**, then P2 21.4, P4 17.0,
P7c 15.0, P1 11.2, P5 9.3, P6 4.9, P7a 3.4. **P7b's share did not fall** — that is the
honest result, and it is the finding: the phase is the SFPU payload itself, not the
structure around it.

### Findings — why the two named leads are closed

- **The sigmoid is genuine SFPU work, not a `cb_wait_front` stall.** The ablation settles
  what a per-phase zone cannot (the zone wraps the helper's own `cb_wait_front`): removing
  only the payload drops B=1/T=640 from 244.5 µs to 92.2 µs. 37 ns/vector-op × 32
  vector-ops/tile × 128 tiles/block matches the catalog's measured Blackhole SFPU rate
  (`sfpu_tile_scope`: ~24 ns rsqrt, ~28 ns recip per vector op). Normal throughput, nothing
  misconfigured. The ablation also exposes the floor: with the sigmoid gone the op is
  reader-bound at ~92 µs (NCRISC 84.8 µs vs TRISC 90.3 µs).
- **Lead 1 (pack thread) is 0.991–0.994× — a small, consistent loss, and structurally so.**
  A Tensix has **one SFPU**, shared by MATH and PACK, and DEST half-sync pins the two
  threads to within one window. Relocating the SFPU volume therefore relocates the
  bottleneck rather than overlapping it, and the pack path additionally pays
  `apply_activation_from_pack`'s SEMWAIT + dest-offset flip + WAIT_SFPU stall. The
  refinement's premise — "PACK is comparatively idle" — is true of the *thread* and false
  of the *engine*, and there is no MATH-side work of comparable size inside P7b/P7c to hide
  the sigmoid behind. Kept as a correct, live, non-default knob (not reverted): it is the
  ready-made vehicle if a future refinement ever creates concurrent MATH work here.
- **Lead 2 (`fast_and_approx`) cannot be built at all.** `_calculate_sigmoid_` and
  `_init_sigmoid_` take `APPROXIMATION_MODE` as a template parameter and **ignore it** on
  both Blackhole and Wormhole B0 — accurate and fast are the same 6-entry LUT
  (`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_sigmoid.h`).
  The measured 1.030× global `math_approx_mode` win in the Phase-0 report therefore came
  from `rsqrt`, not the sigmoid. Closed by source inspection; no device time spent.
- **The remaining lever is the DEST width, and it is now priced.** `fp32_dest_acc_en=False`
  is **1.208×** overall at B=1/T=640 (244,312 → 202,256 ns) and **1.39× on the sigmoid
  alone** (152,167 → 109,354 ns), with the non-sigmoid remainder flat (92.1 → 92.9 µs).
  Constraint (a) forbids reaching for it here, so it was measured, not taken, and filed as
  Refinement 1b with the mechanism options spelled out.

- **Accuracy achieved**: unchanged at the shipped defaults — PCC = **0.999993**,
  rel-RMS = **0.0037**, max_abs_err 0.0234–0.0330, got/true ratio median **0.9997** across
  B×T = 1×32 / 1×128 / 1×640 / 4×256. Neither adopted change touches the arithmetic:
  `GATE_DEST_TILES` only regroups DEST windows, and `SIGMOID_ENGINE` stays at `"math"`.
  `test_onorm_precision_baseline.py` passes unchanged (4/4).
  For Refinement 1b's benefit, the 16-bit-DEST numbers were also measured (probe_005):
  PCC 0.999988, rel-RMS 0.0056, ratio median 1.0026 — inside the PCC ≥ 0.9995 /
  rel-RMS < 0.02 bar, broad-spread-around-1.0, i.e. rounding and not a scale bug.

- **Golden test progress**: **11 / 11 passing** (5 / 5 registry cells + 6 numerics
  regression tests), unchanged from Phase 0. No SUPPORTED / EXCLUSIONS change — this is a
  perf refinement.

- **Issues encountered**:
  1. `ActivationApplyHelper`'s SIGMOID branch did not compile (scoped-enum `VectorMode`
     deduced as `int`) — the design's own named mechanism for lead 1 had never been
     instantiated by anything. Fixed in the shared header.
  2. `eltwise_chain` has **no packer-activation slot** (only `matmul_block` /
     `add_bias_bcast_rows` do), so the `"pack"` engine could not be expressed as a
     `unary<>` call. It is the file's one helper substitution, declared at the top of
     `onorm_compute.cpp` with the specific limitation that forces it; the activation itself
     still runs through the `sfpu_activation_helpers.hpp` helpers. If a packer-activation
     slot ever lands on `eltwise_chain`, the branch collapses to one `unary<>` call.
  3. `InputLifecycle::Chunked` is legal only with `OperandKind::Block` — a chunk-scaled
     wait on a `Scalar`-kind operand out-runs the window and deadlocks. Both gate phases
     pass `OperandKind::Block` explicitly, with the reason in a comment.

- **Tests added**: `tests/ttnn/unit_tests/operations/onorm/test_onorm_sigmoid_engine.py`
  (125 cases) — the shipping-engine × block-factor correctness cross product, the two
  guard tests (ablation opt-in, over-budget `GATE_DEST_TILES` rejected naming the limit),
  the engine/block-factor measurement sweep, the DEST-width pricing sweep, and the
  config-spanning guard-set non-regression sweep. Plus `probes/probe_005.py` (precision
  under both DEST widths). Full op suite: **240 / 240**; golden suite **11 / 11**; `--dev`
  clean (no watcher asserts, no races).

- **Follow-up filed**: `### [ ] Refinement 1b — Cash the 16-bit DEST`, ordered immediately
  after this one and before Refinement 2, naming the exact next lever with its measured
  prize (1.208×) and its measured precision cost.
