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

---

## Refinement 1b — Cash the 16-bit DEST: 1.208×, with P1's fp32 sum-of-squares preserved

- **Date**: 2026-07-27
- **Arch**: Blackhole p150 (`ARCH_NAME=blackhole`), 11×10 = 110-core compute grid
- **Outcome**: `[x]` full. The 16-bit DEST is landed as the default and is worth
  **1.18–1.28× on every cell of the config-spanning guard set**. Route 1
  (preserve P1's fp32 accumulation by another mechanism) was investigated and is
  **not available on this hardware**; **route 2** — the explicit documented
  deviation the refinement provides for — was taken, with the deviation written
  into `default_compute_kernel_config()`'s docstring naming the contract line.
  The named free rider (`GATE_DEST_TILES` 4 → 8) was re-swept and **does not
  pay**; 4 remains the measured optimum.

### Why route 1 is closed (this is the substance of the refinement, not the flag flip)

`fp32_dest_acc_en` is a **whole-kernel compile-time** config, so "fp32 for P1
only" necessarily means moving the sum-of-squares accumulation *off* DEST. All
three candidates the refinement named were run down:

1. **Packer L1 accumulator (`PackTileL1Accumulation`)** — the only fp32
   accumulation datapath in a Tensix that bypasses DEST, and therefore the only
   candidate that could have worked. It is **fp32-DEST-only hardware**. This is
   not inferred: the catalog's on-device example
   `ttnn/ttnn/operations/examples/row_reduce_accumulate` measured it and pins it
   in three places — "L1-accumulate is **fp32-DEST-only** hardware" (README
   method table), "the packer L1-accumulate datapath is fp32-DEST-only (**a bf16
   DEST corrupts the accumulate**), so `l1_accum` ALWAYS uses fp32 DEST"
   (program descriptor), and its accuracy row is best "only because the packer
   forces fp32 DEST". Enabling it under a 16-bit DEST does not buy fp32
   accumulation — it buys a corrupt one.
2. **A `Float32` `cb_sumsq` with the accumulation moved out of DEST** — there is
   nowhere else to move it to. Every path from the FPU or SFPU into L1
   traverses DEST, so with a 16-bit DEST each `o²` term is already rounded to
   bf16 before it can reach an fp32 CB. A wider intermediate CB below a
   narrower DEST buys nothing: the pack is exact and the next unpack rounds
   straight back down.
3. **`dest_accum_pairs`** — a *tree* rather than *sequential* accumulation. Real,
   but it is bf16-accumulation-error reduction, not fp32 preservation, and the
   catalog measures its benefit at W=32 tiles. P1's cross-tile depth here is
   `V_TILES = 4`, where the difference is negligible. Not worth a restructure of
   a passing phase for an effect below the measurement floor.

Structurally there is a fourth blocker even had (1) worked: the chain's
`OutputLifecycle::L1Accumulation` owns **one** accumulator for the whole chain
(`OneUpfront` / `OneAtEnd`), while P1 needs one per token (`nb` per chunk) —
which is why it pairs with `DestAccumulation`'s `PerOuter` today. Getting
per-token L1 accumulation would have meant caller-managed CB scaffolding wrapped
around the helper, i.e. a helper substitution, for a mechanism that is
hardware-invalid at the target DEST width anyway.

**Conclusion**: at a 16-bit DEST there is no fp32 accumulator in the machine.
Route 1 is not a cost/benefit call, it is unavailable. Route 2 taken.

### The documented deviation (route 2)

`eval/prompts/onorm.txt` → `## Rules` → `Precision`:

> "When reducing for RMSNorm: accumulate the sum-of-squares in fp32 in DST
> (`fp32_dest_acc_en=True`) even though the I/O dtype is bf16 — the reduction is
> the precision-sensitive step."

`default_compute_kernel_config()` now returns `fp32_dest_acc_en=False`. The
deviation is written out in full in that factory's docstring (prize, why route 1
is unavailable, measured precision cost, and the fact that the field itself is
still honoured). **The contract's mechanism is not removed** — it is a public
`compute_kernel_config` field, and a caller who passes
`ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True)` still gets the 32-bit
DEST and the fp32 sum-of-squares accumulation, bit-for-bit as R1 shipped it.
Only the `None` default moved. That path is now a guard-set cell in its own
right (`fp32on` below) so it cannot silently rot.

### What was done

1. **`default_compute_kernel_config()` → `fp32_dest_acc_en=False`** with the
   deviation block in its docstring. Also corrected two stale claims in the same
   docstring: "the op is DRAM-bound" (Phase 0's verification disproved it) and
   the implication that `math_approx_mode` relaxes the sigmoid (R1 proved the
   LLK ignores `APPROXIMATION_MODE` for sigmoid).
2. **`_DEST_TILE_LIMIT` (a hardcoded `4`) → `_dest_tile_limit(fp32_dest_acc_en)`**
   over `_DEST_TILE_LIMIT_FP32 = 4` / `_DEST_TILE_LIMIT_16B = 8`. `DEST_AUTO_LIMIT`
   is a function of the DEST width, and the DEST width is a *caller* input — so a
   hardcoded ceiling was simply wrong under the new default (it would have
   rejected a legal 8). This was a required correctness fix, not a knob turn.
3. **`GATE_DEST_TILES` became a clamped REQUEST.** The descriptor derives
   `gate_dest_tiles = min(GATE_DEST_TILES, _dest_tile_limit(cfg))` and everything
   downstream (the CB-capacity assert, the divisibility assert, the compute CT
   arg) reads the *effective* value. This is what lets one module-level knob
   serve both DEST widths: a caller asking for fp32 DEST gets a legal 4-tile
   window from the op's own 8-capable default instead of an assert. **No kernel
   change was needed** — `gate_dest_tiles` already travelled as a CT arg.

### Measurements

All medians of 5 **trial-major interleaved** repetitions inside one process
(`test_onorm_sigmoid_engine.py`), `DEVICE KERNEL DURATION [ns]`.

**Config-spanning guard set — paired inside one process, R1's shipped pair
(32-bit DEST, `GATE_DEST_TILES=4`) vs R1b's (16-bit DEST, `GATE_DEST_TILES=4`):**

| cell | cores | R1 shipped | R1b shipped | speedup | max spread |
|---|---|---|---|---|---|
| B=1,T=32 default | 1 | 238,002 | 195,749 | **1.2159×** | 0.07 % |
| B=1,T=128 default | 4 | 238,346 | 196,079 | **1.2156×** | 0.07 % |
| B=1,T=640 default | 20 | 243,207 | 200,692 | **1.2118×** | 0.48 % |
| B=8,T=640 default | 110 | 535,686 | 453,649 | **1.1808×** | 3.59 % |
| B=1,T=640 `math_approx_mode=True` | 20 | 243,337 | 192,920 | **1.2613×** | 0.79 % |
| B=1,T=640 `MathFidelity::LoFi` | 20 | 243,308 | 189,979 | **1.2807×** | 0.28 % |
| B=1,T=640 **`fp32_dest_acc_en=True`** | 20 | 243,170 | 243,183 | 0.9999× | 0.34 % |

Every cell improves; none regresses. The last row is the caller-override path
and is unchanged to within 0.01 % — the clamp makes R1b invisible to a caller
who asks for the 32-bit DEST, which is exactly the required behaviour.

**The win is the SFPU's, confirmed by ablation at BOTH DEST widths** (the R1
`ablate` engine removes the sigmoid payload and keeps every CB wait/push, DEST
window and NoC transfer):

| B=1/T=640 | whole kernel | ablated (no sigmoid) | sigmoid payload |
|---|---|---|---|
| 32-bit DEST | 243,103 | 91,620 | **151,483** |
| 16-bit DEST | 200,667 | 92,426 | **108,241 (1.399×)** |

| B=8/T=640 | whole kernel | ablated | sigmoid payload |
|---|---|---|---|
| 32-bit DEST | 534,816 | 305,888 | **228,928** |
| 16-bit DEST | 452,478 | 305,279 | **147,199 (1.555×)** |

The ablated (non-sigmoid) remainder is **flat across the two DEST widths** —
91.6 vs 92.4 µs, and 305.9 vs 305.3 µs — so the entire saving is the SFPU
payload, not the scaffolding around it. This reproduces R1's prediction (1.208×
whole-kernel, 1.39× on the payload) essentially exactly, from an independent
measurement.

**Free rider — `GATE_DEST_TILES` re-swept against the doubled `DEST_AUTO_LIMIT`**
(all at the shipping 16-bit DEST; `vs d1`):

| shape | d1 | d2 | d4 | d8 | max spread |
|---|---|---|---|---|---|
| B=1,T=128 | 1.0000 | 1.0054 | **1.0078** | 1.0046 | 0.26 % |
| B=1,T=640 | 1.0000 | 1.0031 | **1.0046** | 1.0044 | 0.56 % |
| B=8,T=640 | 1.0000 | 0.9891 | **1.0009** | 0.9960 | 5.2 % |

The curve **turns over at 4**: R1's monotonic 1 → 2 → 4 continues, but 4 → 8 is
a tie at best and a small loss at worst, on all three shapes. So the free rider
was measured and **not** taken — `GATE_DEST_TILES` stays at its measured optimum
of 4. The *knob* keeps the headroom it gained (the ceiling is now derived, so 8
is reachable and legal); only the shipped *value* stayed put. Refinement 3
inherits a live knob with a doubled ceiling and this curve as its starting point.

The `pack` engine was re-measured at the new DEST width and is still a
consistent loss (0.985–0.988× across the three shapes), unchanged from R1's
finding — it remains a correct, live, non-default knob.

- **Accuracy achieved**: PCC = **0.999988**, rel-RMS = **0.0056**,
  max_abs 0.0390–0.0487, mean_abs 0.0018, got/true ratio median = **1.0026**
  (p5 0.9947 / p95 1.0110, std 0.0049) across B×T = 1×32 / 1×128 / 1×640 /
  4×256. `test_onorm_precision_baseline.py` passes 4/4.

  | | PCC | rel-RMS | got/true median |
  |---|---|---|---|
  | R1 shipped (fp32 DEST) | 0.999993 | 0.0037 | 0.9997 |
  | **R1b shipped (16-bit DEST)** | **0.999988** | **0.0056** | **1.0026** |

  This lands exactly on the numbers R1 predicted from `probe_005`. It is 3.5×
  inside the refinement's own bar (PCC ≥ 0.9995, rel-RMS < 0.02) and ~7× inside
  the golden bf16 bar (PCC ≥ 0.995, RMS 0.04). The ratio stays a broad spread
  centred on 1.0 (std 0.0049 against a 0.0026 median offset), i.e. rounding, not
  a scale bug — the baseline's own scale-bug guard (0.98 ≤ median ≤ 1.02) passes
  with ~7× margin.

- **Golden test progress**: **11 / 11 passing** (5 / 5 registry cells + 6
  numerics regression tests), unchanged. No SUPPORTED / EXCLUSIONS change — this
  is a perf refinement.

- **Issues encountered**:
  1. `_DEST_TILE_LIMIT` was a hardcoded `4`, which silently encoded the *old*
     default's DEST width. Flipping the default without fixing it would have
     made a legal `GATE_DEST_TILES=8` unreachable while the assert claimed it was
     a hardware limit. Fixed by deriving the ceiling from the caller's config.
  2. The R1 test file's `_FP32_ON = default_compute_kernel_config()` bound a
     *name* to a *default* that this refinement moves. It now pins the fp32
     config explicitly and `_FP32_OFF` is the factory, with an import-time assert
     that the factory really does ship the 16-bit DEST — so the next default move
     fails loudly here instead of silently retitling a measurement column.
  3. Not an issue, recorded: the got/true median moves from 0.9997 to 1.0026, a
     +0.26 % systematic offset (the 16-bit DEST slightly under-estimates the
     positive sum of squares, so `rsqrt` comes out slightly large). It is well
     inside the scale-bug guard and an order of magnitude below the golden bar,
     but it is a *bias* rather than pure noise and is recorded here so a future
     precision refinement has the signature.

- **Tests added / changed** (all in `test_onorm_sigmoid_engine.py`, extending
  R1's harness rather than forking a new file):
  - `test_gate_dest_tiles_clamped_for_fp32_dest` (**new**, 4 cases) — the guard on
    the new clamp: every `GATE_DEST_TILES` in {1,2,4,8}, including the op's own
    default, must produce a correct answer under `fp32_dest_acc_en=True`.
  - `test_gate_dest_tiles_over_limit_rejected` — retargeted from 8 (now legal) to
    16 (above the widest DEST budget), so it still proves the host assert fires.
  - `DEST_TILES` extended to include 8; `CANDIDATES` extended with `math/d8` and
    `ablate/d8` so the block-factor curve reaches the new ceiling.
  - `test_dest_acc_trial` — rebuilt around a `DEST_ACC_VARIANTS` table covering
    both shapes and both DEST widths, with an `ablate` cell **at each width** so
    the payload can be priced separately at 16 and 32 bits.
  - `test_guard_set_trial` — the "new" arm now reads `GATE_DEST_TILES` from the
    module instead of restating it, so the guard always measures what actually
    ships; added the `fp32on` cell that keeps the public override on the guard set.
  - Full op suite: **368 / 368**; golden suite **11 / 11**; `--dev` clean (no
    watcher asserts, no races).

- **Left for Refinement 3**: `GATE_DEST_TILES`' ceiling is now 8 and derived, and
  the 4-vs-8 curve above is measured at the *current* structure — R3 should
  re-sweep it against the post-R2 structure, not assume this result carries.

---

## Refinement 2 — Cross-core re-tile: stop leaving 108 of 110 cores idle at small `T`

- **Date**: 2026-07-27
- **Arch**: Blackhole p150 (`ARCH_NAME=blackhole`), 11×10 = 110-core compute grid
- **Outcome**: `[x]` full. The cross-core re-tile is built, is the shipping default
  through a measured dispatch policy, and is worth **16.09× / 13.32× / 8.48×** on the
  three under-filled shapes the refinement targeted and **2.72× / 1.22×** on the two
  that were already core-saturated. Every cell of the config-spanning guard set
  improves; none regresses. `CORE COUNT` rises 1 → 32, 2 → 64, 4 → 64, 20 → 96.

### What was done

**The scheme: one exchange, TWO split axes.** The refinement's own sketch — each core
normalizes a token subset and NoC-writes its row-major rows into *the owning core's*
`cb_rm_flat_rows` — parallelizes only the normalize half (P1–P6). After R1b that half is
~64 µs of a ~196 µs kernel, so its ceiling is **1.48×**: below the "several-fold" bar. The
shipped scheme splits the **gate half as well**, and both fall out of the *same* exchange:

| half | split axis | what each core reads / writes | why it needs no amplification |
|---|---|---|---|
| normalize (P1–P6) | **tokens** | its own `tokens_per_core` slice of `o` | consecutive tokens are consecutive `o` tiles |
| gate (P7a–P7c) | **flat columns** | its own `cols_per_core` slice of `gate` and of `out` | one consecutive run per tile-row |

The join: `pack_untilize` emits token `t`'s features at linear index `h*V + c`, which **is**
the flat feature index — so the features that column-owner `d` needs are exactly ONE
contiguous `chunk_bytes = FLAT*2 / group_cores` slice of every token's row. The exchange is
therefore "for each of my token rows, send its `d`-th chunk to member `d`'s
`cb_rm_flat_rows` at row offset `t * chunk_bytes`", and each core ends up holding a
`[TOKENS_PER_BLOCK, cols_per_core*32]` row-major stripe of row stride `chunk_bytes` — the
*same* contract the local untilize honoured, at `1/G` the width. That is precisely what
`op_design.md` §1.5's lamp #1 promised, and it is why nothing is re-read and nothing is
re-written: only the row-major intermediate crosses the NoC, and only once.

1. **`RETILE_GROUP_CORES` knob** (`"auto"` default, or an int to pin) — how many cores share
   one token-block. `1` is the **trivial, byte-identical** value: no exchange, no semaphores,
   `ONORM_CB_RM_LOCAL` *aliases* `ONORM_CB_RM_FLAT_ROWS` so compute untilizes straight into
   the stripe it later tilizes, and every derived quantity collapses to its pre-R2 value.
2. **`_retile_group_cores()` dispatch policy.** `auto` minimises the slowest group's
   critical-path work in whole-block units, `ceil(blocks / num_groups(g)) / g`, ties to the
   smaller group. One objective, two regimes: occupancy when blocks < cores, load balance
   when blocks > cores.
3. **Writer-side scatter + a two-counter handshake** (`onorm_writer.cpp`). Per block:
   `cb_reserve_back` my stripe → tell every member it is free → wait all `G` → scatter
   (`norm_chunk_tokens × group_cores` writes, ONE barrier per normalize chunk) → announce →
   wait all `G` → `cb_push_back` the stripe for P7a. `TOKENS_PER_BLOCK` writes per core per
   block regardless of `G`; the bytes fall as `1/G`.
4. **`cb_rm_local`** — the staging CB between compute's untilize (producer) and the writer's
   scatter (consumer), depth `RM_LOCAL_DEPTH = 2` so compute builds chunk *i+1* while the
   writer scatters chunk *i*. Single-producer / single-consumer holds in both paths.
5. **Per-stream `DM_BLOCK_TILES` clamp** (a required correctness fix, not a knob turn) — see
   Issues.

**The exchange is L1-POSITIVE.** `cb_rm_flat_rows` and `cb_flat_tiles` divide by `G`:

| `G` | norm_chunk | gate_chunk | cols/core | CB pages | bytes |
|---|---|---|---|---|---|
| 1 | 8 | 64 | 128 | 533 | 1,091,584 (1066 KB) |
| 4 | 8 | 32 | 32 | 373 | 763,904 (746 KB) |
| 8 | 4 | 16 | 16 | 221 | 452,608 (442 KB) |
| 32 | 1 | 4 | 4 | **107** | **219,136 (214 KB)** |

### Measurements

Medians of 5 **trial-major interleaved** repetitions inside one process,
`DEVICE KERNEL DURATION [ns]`.

**Group-size curve** (`test_onorm_retile_group.py::test_group_trial`), speedup vs `G=1`:

| shape | blocks | g=1 (cores) | g=2 | g=4 | g=8 | g=16 | g=32 |
|---|---|---|---|---|---|---|---|
| B=1,T=32 | 1 | 195,927 (1) | 1.93× | 3.58× | 6.32× | 10.62× | **16.12×** (32 cores) |
| B=1,T=128 | 4 | 195,998 (4) | 1.89× | 3.35× | 5.55× | **8.59×** (64) | 7.67× (96) |
| B=1,T=640 | 20 | 201,020 (20) | 1.68× | 2.57× | 2.62× | 2.53× | **2.72×** (96) |
| B=8,T=640 | 160 | 454,099 (110) | 1.22× | **1.26×** (108) | 1.05× | 0.94× | 0.87× |

Max spread 4.4 %. The curve is monotone up to where the grid runs out of groups; B=8/T=640
turns over at 4 and **loses** past 8, which is exactly what the `_work` objective keeps it
away from.

**Config-spanning guard set — paired inside one process, R1b's shipped config
(`RETILE_GROUP_CORES=1`) vs R2's shipped `auto`** (`test_onorm_retile_guard.py`):

| cell | cores R1b → R2 | R1b ns | R2 ns | speedup | max spread |
|---|---|---|---|---|---|
| B=1,T=32 default | 1 → 32 | 195,748 | 12,164 | **16.092×** | 0.99 % |
| B=1,T=64 default | 2 → 64 | 195,859 | 14,704 | **13.320×** | 2.56 % |
| B=1,T=128 default | 4 → 64 | 195,905 | 23,098 | **8.481×** | 1.92 % |
| B=1,T=640 default | 20 → 96 | 200,657 | 73,819 | **2.718×** | 2.03 % |
| B=8,T=640 default | 110 → 110 | 452,179 | 371,193 | **1.218×** | 3.49 % |
| B=1,T=640 `math_approx_mode=True` | 20 → 96 | 193,110 | 73,968 | **2.611×** | 0.72 % |
| B=1,T=640 `MathFidelity::LoFi` | 20 → 96 | 190,113 | 72,957 | **2.606×** | 7.15 % |
| B=1,T=640 **`fp32_dest_acc_en=True`** | 20 → 96 | 243,199 | 83,091 | **2.927×** | 1.19 % |

Every cell improves; none regresses. The caller-override row (`fp32on`) gains the most of the
T=640 cells (2.93×) — the split shrinks the per-core SFPU volume, which is the term the 32-bit
DEST inflates, so R2 and R1b compound rather than compete.

### Findings

- **`mcast_pipe.hpp` cannot express this exchange** — three independent reasons, each one of
  the helper's own stated preconditions: (a) `SenderPipe::send(src_l1, dst_l1, size)`
  multicasts ONE block to a rectangle at ONE landing address ("`dst_l1` is identical across
  all receivers"), while here every destination gets *different* bytes from a *different*
  source offset — a scatter, not a broadcast; (b) "single sender per receiver" — this is an
  all-to-all, so each data-ready cell has `group_cores` writers, and neither `Flag` nor
  `Counter` expresses "wait for `G` distinct contributors"; (c) the payload rows are strided
  on **both** sides (source `local_row_bytes`, destination `chunk_bytes`), so even a
  degenerate 1×1 rect could not carry a block in one `send()`. The substitution is declared
  at the head of `onorm_writer.cpp`. What *is* used is the layer `mcast_pipe` itself is built
  on — the `Noc` and `Semaphore<>` object APIs — not raw `noc_semaphore_set/wait/inc`.
- **Monotone counters, not reset flags.** Both semaphores are host-initialised to 0 and
  waited with `wait_min((blk+1)*group_cores)`; no kernel ever resets them. A member that
  finishes block *k* and races into block *k+1* therefore cannot clobber an increment another
  member has not yet observed — which a set-to-0 reset would. `test_repeated_dispatch_is_stable`
  guards the other half of that argument (the host really does re-initialise per launch).
- **Flow control is `cb_reserve_back`, not a hand-built credit scheme.** A receiver claims its
  stripe *before* telling anyone it is free, so "free" is literally "the CB says there is
  room" — the same mechanism the local path used, lifted across the NoC.
- **`auto` needs the load-balance term, not just occupancy.** The intuitive policy ("spend
  only cores a per-block split leaves idle") gives `G=1` on B=8/T=640 and misses a measured
  1.22×: 160 blocks on 110 cores puts TWO blocks on fifty cores, and `G=2` rebalances to 55
  groups × 3 blocks at half a block each (critical-path work 2 → 1.5). The
  `ceil(blocks/num_groups)/g` objective covers that and the small-`T` case in one number.
- **The exchange is numerically inert, and that is asserted exactly.**
  `test_group_size_is_bit_identical_to_single_core` demands `torch.equal` against the `G=1`
  output at every group size on every shape (15 cells, all exact). Which core normalizes a
  token and which core gates a column changes; the arithmetic applied to either does not.

- **Accuracy achieved**: PCC = **0.999988**, rel-RMS = **0.0056**, max_abs 0.0390–0.0487,
  mean_abs 0.0018, got/true ratio median = **1.0026** (p5 0.9947 / p95 1.0111, std 0.0049)
  across B×T = 1×32 / 1×128 / 1×640 / 4×256. **Identical to R1b's recorded numbers to every
  digit** — as required, since the exchange is bit-exact.
  `test_onorm_precision_baseline.py` passes 4/4.

- **Golden test progress**: **11 / 11 passing** (5 / 5 registry cells + 6 numerics regression
  tests), unchanged. No SUPPORTED / EXCLUSIONS change — this is a perf refinement.

- **Issues encountered**:
  1. **`DM_BLOCK_TILES` divisibility became a live bug.** The assert required the streaming
     CBs' page counts to be a multiple of the raw knob, but the cross-core split *shortens
     every stream*: at `G=32` a normalize chunk is 1 token = 4 pages, so `cb_o_tiles` is
     `4*1*O_DEPTH` and `O_DEPTH=3` gave 12 pages — not a multiple of 8, and the op refused a
     previously legal knob setting (caught by `test_onorm_knobs.py::test_knob_turn[O_DEPTH-3]`
     and two `test_onorm_dmsweep.py` cases). Fixed by making `DM_BLOCK_TILES` a **request
     clamped per stream** to that stream's own consumption granularity — one normalize chunk
     for `o`, one column slice for `gate`/`out` — which is the same idiom `GATE_DEST_TILES`
     already used for the DEST budget, and keeps the two knobs independent. Inactive (hence
     byte-identical) at every shipped `G=1` setting.
  2. **The L1-budget frontier MOVED, so its guard tests had to be re-pinned.** The two
     re-tile buffers dominate the footprint and now divide by `G`, so `NORM_CHUNK_TOKENS=32`
     — legitimately over budget at `G=1` — fits comfortably at `G≥2`, and
     `test_knob_over_budget_is_rejected_with_guidance` stopped seeing its assert. The
     over-budget combos now pin `RETILE_GROUP_CORES=1`, where the assert still binds, and the
     assert's own message gained the third escape route the refinement created ("raising
     RETILE_GROUP_CORES divides both of them by the group size"). A fourth `COMBOS` entry
     records the flip side: `TOKENS_PER_BLOCK=64, NORM_CHUNK_TOKENS=16` needs two knobs
     lowered at `G=1` and fits untouched at `G=4`.
  3. **`_grid_assignment` had to lift the split one level, cores → groups.**
     `split_work_to_cores` still owns grid → core-set (asked for exactly
     `num_groups * group_cores` cores, one unit each, so it returns them in row-wise order);
     the blocks-over-groups distribution is the same `base`/`remainder` split it would apply
     itself. `row_wise=True` now matters twice over: the exchange group is a *contiguous run*
     of that order, so a group sits inside one grid row wherever the row is wide enough and
     the all-to-all stays a short-hop exchange along one row.
  4. Not an issue, recorded: B=1/T=640 shows ~6 % between the policy's pick (`G=32`, 2.72×)
     and the flat `G=4..32` plateau, and B=8/T=640 ~3.5 % between `G=2` (the tie-break's pick)
     and `G=4`. Both are inside the "pick a simple, shape-general rule" tolerance and are
     recorded so R3 can revisit the tie-break against its final block factors.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/onorm/test_onorm_retile_group.py` (**new**, 41 cases) —
    correctness at every legal group size × 3 shapes plus a core-saturated shape; the 15
    **bit-identical-to-`G=1`** cells; the `auto` policy against an independent restatement of
    its objective plus a block-count sweep straddling the grid boundary asserting it never
    loses to `G=1`; the illegal-group-size host guard; and `test_repeated_dispatch_is_stable`
    (5 dispatches in one process must agree — the monotone-semaphore / host-re-init argument).
    `test_group_trial` is the trial-major interleaved group-size measurement sweep.
  - `tests/ttnn/unit_tests/operations/onorm/test_onorm_retile_guard.py` (**new**) — the paired
    old-vs-new device-ns guard set over all 8 config-spanning cells, with the "new" arm
    reading `RETILE_GROUP_CORES` from the module so it always measures what ships.
  - `test_onorm_knobs.py` — `RETILE_GROUP_CORES` ∈ {1, 2, 8, 32}, `MAX_RETILE_GROUP_CORES=4`
    and `RM_LOCAL_DEPTH=3` added to `KNOB_SETTINGS`; the new group-funded `TOKENS_PER_BLOCK=64`
    combo added; the over-budget combos re-pinned to `G=1` (see issue 2).
  - Full op suite: **418 / 418**; golden suite **11 / 11**; `--dev` clean (no watcher asserts,
    no races) and the production-timing run agrees.
