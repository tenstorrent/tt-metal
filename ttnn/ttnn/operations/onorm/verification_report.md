# Verification Report: onorm

Verified on **Blackhole p150** (`ARCH_NAME=blackhole`, 11×10 = 110-core compute grid),
2026-07-27, against commit `44c07aa47f`.

`onorm` is the Kimi-Linear KDA **s6** tail:
`out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)`.

---

## Verdict

The op is **correct, honest, and shippable**. All three loud verifier categories are
zero, the acceptance suite is 11/11, the knob suite is 14/14, and PCC is 0.99999 on
every measured shape — three orders of magnitude of headroom over the bf16 golden bar.

The op is **not yet fast**, and the reason is a design-premise miss that only shows up
under measurement: `op_design.md` builds its whole lever ranking on the op being
*DRAM-bandwidth-bound*. **It is not.** It is **SFPU-bound on the gate sigmoid**, which is
**64 % of the kernel**, and per-core achieved bandwidth is **3.1 GB/s against a
~17.9 GB/s single-core NoC ceiling**. Every conclusion in the design of the form "the
catalog says this lever does not pay because we are DRAM-bound" is therefore void and is
re-opened by the refinement queue. Details in *Measured Performance Profile* below.

---

## Code Review

Everything in this section was **fixed in place**. Nothing was deferred except the items
under *Recommendations*.

### Fixed

1. **CB slot map was restated in two places (DRY violation).** The host declared
   `CB_O_TILES = 0 … CB_GATE_SIG = 31` in `onorm_program_descriptor.py`, and all three
   kernels re-declared the same twelve integers as literals. Renumbering a slot on either
   side would have silently pointed a kernel at a differently-sized buffer — a hang or a
   wrong answer with no compile-time signal.
   **Fix**: added `_CB_SLOTS` / `_CB_DEFINES` in the descriptor and injected the map via
   `KernelDescriptor(defines=…)`. The kernels now read `ONORM_CB_*` preprocessor defines.
   The host block is the single source of truth; a kernel referencing a slot the host does
   not define now fails to compile.

2. **`V` was never checked against the tile width.** The whole op rests on the in-kernel
   head-major → flat re-tile, whose correctness requires head `h`'s values to be *exactly*
   the row-major span `[h*V, (h+1)*V)`. A `V` that is not a multiple of 32 interleaves TILE
   padding into that span and mis-maps every flat feature index — and it does so with **no
   numeric signal**, because the RMSNorm scaler `1/V` would still be arithmetically right.
   **Fix**: added `assert head_dim % TILE_W == 0` in `onorm()` with that reasoning in the
   message.

3. **No placement contract.** `memory_config` is not a TARGET axis, so a sharded input was
   neither refused nor budgeted: `TensorAccessor` would still have read it correctly, but
   the shard's resident L1 is invisible to the descriptor's CB budget, so it could OOM with
   a bare runtime throw.
   **Fix**: added an explicit host assert that `o` / `gate` / `weight` are interleaved,
   worded as a placement contract rather than an axis refusal.

4. **Page-size assert was silent and incomplete.** `assert gate.buffer_page_size() ==
   tile_bytes and output.buffer_page_size() == tile_bytes` had no message and omitted
   `weight`, even though the reader addresses `weight` with the same shared `page_bytes`
   compile-time arg.
   **Fix**: looped over all three, added a message naming the shared-CT-arg reason.

5. **`DM_BLOCK_TILES` divisibility assert did not name the escape.** It reported the
   violation without telling the caller which knob to move.
   **Fix**: the message now names both routes (pick a dividing `DM_BLOCK_TILES`, or raise
   `O_DEPTH` / `NORM_CHUNK_TOKENS`), matching the style of the L1-budget assert.

6. **Golden-suite defect (`eval/golden_tests/onorm/feature_spec.py`).** `INPUTS[4]` declared
   `weight` as `(2, 1, 1, V)`, contradicting the contract documented four lines above it
   (`weight : [1, 1, 1, V]`) and the reference itself (`pytorch_onorm` does
   `weight.reshape(-1)` → 256 elements → a torch broadcast error *before any device tensor
   exists*). This produced the run's only `supported_fail`, and **no op-side change could
   ever have cleared it** — the cell never reached the op. The planner flagged this in
   `op_design.md` §13 and asked for it to be folded in.
   **Fix**: corrected to `(1, 1, 1, V)` with a comment explaining that `weight` is a
   per-`head_dim` channel scale shared across `(b, t, h)` and carries no batch dim.
   *This is the one edit made outside the op directory; see* Recommendations *if you would
   rather re-derive it through `/golden-tests`.*

### Reviewed and found correct (no change needed)

- **Helper usage is complete.** Every compute mechanism goes through `kernel_lib`: P1
  `eltwise_chain` + `DestAccumulation`, P2 `reduce<>` with a fused `post_reduce_op`, P4/P5/P7c
  `mul<>`, P6 `untilize<>`, P7a `tilize<>`, P7b `unary<Sigmoid<>>`, reader
  `prepare_reduce_scaler` in its pool-type-aware form. There is no raw `tile_regs_*`, no raw
  `reduce_tile`, no raw `pack_tile`, and no CB op wrapped around a helper call. The only raw
  compute LLKs are the three inside P2's `post_reduce_op` lambda — and that lambda *is* the
  helper's documented epilogue hook.
- **`mcast_pipe.hpp` correctly not used.** Phase 1 shares no operand across cores (every
  stream is disjoint per core), so there is no multicast to perform. The `weight` re-read is
  the one candidate and it is ~0.7 % of traffic; it is filed as a rider on Refinement 3.
- **Broadcast dims are right.** P4 uses `BroadcastDim::Col` + `OperandKind::Col` against a
  `REDUCE_ROW` (column-0-valid) `cb_rstd`; P5 uses `BroadcastDim::Row` + `OperandKind::Row`
  against a `[1, V]` weight. Neither pre-broadcasts a full tile of repeated data. The reduce
  is `REDUCE_ROW` over `V` with `Ht = 1`, `batches = NB` — it does not reduce across heads or
  tokens.
- **CB sync ledger balances.** Every CB's push count equals its wait/pop count per token-block
  (`op_design.md` §8.1); the held CBs (`cb_weight`, `cb_scaler`) are re-waited and never
  popped; `cb_o_tiles` is `HeldBulk` in P1 and popped by P4's `Bulk`. Verified against the
  kernel and confirmed empirically — 14/14 knob turns run clean, including the settings that
  change every CB's page count.
- **`TensorAccessor` everywhere**, no `InterleavedAddrGen`. `void kernel_main()` in all three
  kernels. Includes use `api/dataflow/dataflow_api.h`.
- **Both dataflow halves are batched.** Reader and writer both move `DM_BLOCK_TILES`-sized
  groups with **one** barrier per group, driven by the *same* compile-time knob — no
  "reader coalesces, writer dribbles" asymmetry. Reads are on the reader (NoC0), writes on the
  writer (NoC1); the writer performs no reads.
- **The grid is filled to the extent the design's work unit allows.**
  `split_work_to_cores(..., row_wise=True)` over `B * ceil(T/TOKENS_PER_BLOCK)` units, with
  `corerange_to_cores(..., True)` using the same ordering. Measured `CORE COUNT` = 2 / 4 / 20 / 110
  for the four profiled shapes — exactly `num_token_blocks` capped at the grid. That small-`T`
  shapes under-fill is a *work-unit* limit (the 32-token re-tile floor), not an implementation
  bug; going finer is the scheme-change filed as Refinement 2.

### Blocking-model fidelity — audited, clean

Every knob in `op_design.md` §1.1/§1.2 is a live, single-source-of-truth parameter:

| Knob | Where defined | How it reaches the kernel | Verified live |
|---|---|---|---|
| `TOKENS_PER_BLOCK` | descriptor, once | CT arg + derived `tile_rows_per_block` | ✅ `test_knob_combo[TOKENS_PER_BLOCK64-…]` |
| `NORM_CHUNK_TOKENS` | descriptor, once | CT arg `nb` + derived `norm_chunks_per_block` | ✅ 4 and 16 |
| `GATE_CHUNK_TILES` | descriptor, once | CT arg + derived `gate_chunks_per_block` | ✅ 32 and 128 |
| `DM_BLOCK_TILES` | descriptor, once | CT arg to **both** reader and writer | ✅ 1, 2, 8 |
| `DM_DEPTH`, `O_DEPTH` | descriptor, once | host CB sizing only | ✅ 4 / 3 |

No collapsed knobs found. Specifically checked and cleared:

- **No CB is sized by a whole-op dimension.** Every page count in `cb_pages` derives from
  `V_TILES` / `NORM_CHUNK_TOKENS` / `O_DEPTH` / `DM_BLOCK_TILES` / `DM_DEPTH` /
  `GATE_CHUNK_TILES` / `flat_tiles_per_block`. Nothing scales with `B` or `T`. The two
  256 KB re-tile buffers (`cb_rm_flat_rows`, `cb_flat_tiles`) scale with `TOKENS_PER_BLOCK`,
  which is a knob, not a whole-op dimension — and they are the *documented* irreducible
  transpose working set (§6.1), not an unconditional op-sized CB.
- **No half-turned split.** The per-core compute loop runs at `NORM_CHUNK_TOKENS = 8` tokens
  and `GATE_CHUNK_TILES = 64` tiles per invocation — coarse, not the minimal unit of 1. The
  cross-core split is at 32 tokens because that is the atomic re-tile granularity (one output
  tile-row), not a collapsed minimum.
- **No duplicate literals.** `flat_tiles_per_block` is derived identically on the host, in the
  reader and in the writer from the same two CT args. The CB slot map was the one genuine
  duplication and is now fixed (item 1 above).
- **The L1 budget assert is real and actionable**, and `test_knob_over_budget_is_rejected_with_guidance`
  proves it fires *before* the runtime's bare "beyond max L1 size" throw and names the knobs to
  lower in the design's prescribed order.

### Prompt-rule audit (`eval/prompts/onorm.txt` `## Rules`)

| Rule | Class | Status |
|---|---|---|
| Reduce over last dim `V`, per `(b,t,h)`; **MUST NOT** reduce across tokens/heads | hard | ✅ `REDUCE_ROW`, `Ht=1`, `batches=NB` |
| Output flat token-major; re-tile **MUST** be in-kernel; **MUST NOT** use `ttnn.reshape`/`to_layout`/`tilize`/`untilize` in Python | hard | ✅ entry point calls none of them; P6+P7a do it in-kernel |
| Re-tile via untilize → row-major → tilize; **MUST NOT** use a tile transpose | hard | ✅ `untilize<V_TILES>` → `cb_rm_flat_rows` → `tilize<FLAT_TILES>` |
| Op owns the sigmoid; normalize before gating | hard | ✅ P1–P6 then P7b/P7c |
| fp32 sum-of-squares in DST; expose `compute_kernel_config` through one exported factory | precision | ✅ `default_compute_kernel_config()` sets `fp32_dest_acc_en=True`; P1 uses `DestAccumulation::Enabled` |
| Single fused kernel, no DRAM round-trip for `o_norm` / `sigmoid(gate)` | perf | ✅ one compute kernel; only `o`/`gate`/`weight` read, only the flat output written |
| Use `untilize_helpers.hpp` / `tilize_helpers.hpp`, no hand-rolled LLK | perf | ✅ |
| Bound per-core CB footprint by a constant in `T`; stream `o`/`gate`/out in small double buffers; do **not** double-buffer the row-major intermediate | perf | ✅ CB total is `O(TOKENS_PER_BLOCK · FLAT)`, constant in `B`/`T`; `cb_rm_flat_rows` is single-buffered |
| Double-buffer the streamed CBs so the reader runs ahead | perf | ✅ `DM_DEPTH = 4` on `cb_gate_tiles`/`cb_out_tiles`, `O_DEPTH = 2` on `cb_o_tiles` |

**One advisory — unfollowed soft rule, and the deviation is correct.** The rules prescribe
`tilize<…, StreamMode::PerTile>` with a 2-tile `cb_flat` and say "do NOT use `Atomic` here".
The implementation uses `Atomic` with a full-block `cb_flat_tiles`. This is a *soft* (perf
section) rule and the deviation is **justified and load-bearing**: `PerTile` reserves/packs/pushes
one output tile at a time, but tilize and its consumer both run in **this** compute kernel, so
all three TRISCs execute them in program order — PACK would block in `cb_reserve_back(cb_flat, 1)`
at the third tile while UNPACK is still inside the tilize loop and can never reach the consumer's
`cb_pop_front`. `PerTile` only pays when the consumer is a *different* RISC. I verified the
helper's own header states exactly this constraint boundary. The design records the reasoning
(§6.1) and the kernel repeats it at the call site. **No action.**

---

## Registry Conformance

- **`INPUT_TAGGERS`** — present, `{}`. Correct: the KDA s6 geometry is fixed (`HV = 32` = one
  tile height, `V = 128` = 4 tile widths) and `T` is tile-aligned by contract, so there is no
  categorical shape facet to project. No tagger signatures to check.
- **`SUPPORTED`** — present: `{"dtype": [bfloat16], "layout": [TILE_LAYOUT]}`. Matches
  `feature_spec.TARGET` axis-for-axis. These are the only two axes the kernel gates on, and
  `INPUT_TAGGERS` adds none. No missing axis.
- **`EXCLUSIONS`** — present, `[]`. Correct: TARGET is a single cell and it is implemented, so
  there is nothing inside the SUPPORTED rectangle to refuse.
- **`validate()`** — present, correctly ordered (SUPPORTED per-axis first, then EXCLUSIONS
  cell-level), raises `UnsupportedAxisValue` / `ExcludedCell` from
  `ttnn.operations._op_contract`, and checks **all three** input tensors. It is the **first
  line** of `onorm()`, before any shape assert or kernel work.
- **`INVALID` is absent from the op file** ✅ — confirmed by grep. It lives only in
  `feature_spec.py`, as the registry model requires.
- **No auto-fixes to SUPPORTED were needed** — `xpass_drift` was zero on both runs, so there is
  no under-claim to promote.

### INVALID audit (`eval/golden_tests/onorm/feature_spec.py`)

`INVALID = []`, and that is **correct, not an omission**:

- *Single-tensor coupling* — vacuous, no entries.
- *Universe-must-change* — TARGET is the single cell `(bfloat16, TILE_LAYOUT)`; the cartesian
  product has exactly one member. Any INVALID entry would either be inert or would empty the
  universe.
- *Canonicalization-only multi-axis exception* — vacuous.
- *Canonical `bfloat8_b` + `ROW_MAJOR` activation entry* — **legitimately absent**: neither
  `bfloat8_b` nor `ROW_MAJOR_LAYOUT` is in TARGET, so that cell is never generated. Adding it
  would be dead weight, not safety. The file says so explicitly at line 52–54.
- *No-weight canonicalization for norm-like ops* — not applicable: `weight` is mandatory in the
  signature, so there is no "no-weight" cell to canonicalize.
- *No cross-tensor-axis entries* — vacuous.

The one **defect** found in this file was in `INPUTS`, not `INVALID`, and is fixed (Code Review
item 6).

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/onorm/test_onorm_precision_baseline.py`, bf16 in / bf16 out,
`epsilon = 1e-5`, default compute config (HiFi4, exact, `fp32_dest_acc_en=True`).

| Shape (B×T) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|---|---|---|---|---|
| 1 × 32   | 0.9999932 | 0.023425 | 0.001186 | 0.003685 |
| 1 × 128  | 0.9999931 | 0.028038 | 0.001200 | 0.003715 |
| 1 × 640  | 0.9999932 | 0.031019 | 0.001192 | 0.003702 |
| 4 × 256  | 0.9999933 | 0.032964 | 0.001185 | 0.003682 |

**got/true ratio spread** (`r = actual / expected` over finite elements with
`|expected| > 0.1 · median|expected|`) — the scale-bug detector:

| Shape (B×T) | median r | p5 | p95 | std |
|---|---|---|---|---|
| 1 × 32   | 0.999687 | 0.993616 | 1.005814 | 0.003693 |
| 1 × 128  | 0.999720 | 0.993587 | 1.005870 | 0.003725 |
| 1 × 640  | 0.999676 | 0.993576 | 1.005789 | 0.003706 |
| 4 × 256  | 0.999741 | 0.993668 | 1.005844 | 0.003682 |

**Assessment.** Textbook bf16 round-trip noise and **no scale bug**. The ratio is centred on
1.0 to within 3 × 10⁻⁴ with a symmetric ±0.6 % spread — the broad-spread-around-1.0 signature,
not the tight-cluster-around-a-constant signature. `rel_rms ≈ 0.0037` matches `std(r)` to three
digits, which is what you expect when the error is per-element quantization rather than a
systematic factor. The numbers are **flat across shapes** (0.003682 → 0.003715 over a 20×
range in token count), confirming the multi-core split and the block loop introduce no
accumulation drift, and that the design's one bf16 statistic round-trip (`cb_rstd`) costs
essentially nothing. Max abs error grows slowly with tensor size purely as an extreme-value
effect on a fixed distribution.

The design's contingency — "if measured PCC lands marginal, promote `cb_rstd` alone to
`Float32`" (`op_design.md` risk 11) — is **not needed**. PCC is 0.99999 against a 0.995 bar;
relative RMS is 0.0037 against a 0.04 bar. There is ~10× margin on RMS and three orders of
magnitude of margin on `1 − PCC`.

**Recommended tolerances**: `PCC >= 0.9995`, `rtol = 0.02`, `atol = 0.02` for regression gates
(the golden suite's 0.995 / 0.04 is correct as a *floor* but is ~10× looser than observed, so it
would not catch a real regression). The precision-baseline test itself asserts the tighter
`rel_rms < 0.02` plus a `0.98 ≤ median r ≤ 1.02` scale-bug guard.

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/onorm/` → `python3 -m eval.verify_supported … ttnn.operations.onorm`
(artifact: `verifier_report.json`, copied next to the op).

| Category | Count |
|---|---|
| `supported_pass` | **5** |
| `xfail_expected` | 0 |
| `invalid_skipped` | 0 |
| `infeasible_skipped` | 0 |
| **`supported_fail`** | **0** ✅ |
| **`xpass_drift`** | **0** ✅ |
| **`xfail_wrong_mode`** | **0** ✅ |
| `supported_marked_xfail` | 0 |
| `no_axes_found` | 6 (the `test_regression.py` numerics tests — not registry-driven; all passed) |

11/11 golden tests pass, 0 hangs.

`xfail_expected = 0` is **expected and correct here, not a queue gap**: TARGET is a single cell
and SUPPORTED equals it, so the cartesian product generates no unsupported cells to xfail.
`TARGET − SUPPORTED = ∅` on both axes — see *Refinement Queue Shape* below.

**First run** (before the fix) had `supported_fail = 1`: the `2×64` cell, failing inside the
harness with `RuntimeError: The size of tensor a (128) must match the size of tensor b (256)`
at `helpers.py:50`, *before any device tensor was built*. Root-caused to the `feature_spec.INPUTS[4]`
weight-shape defect (Code Review item 6), not to the kernel. After the fix: 5/5.

### Other suites

| Suite | Result |
|---|---|
| `test_onorm.py` (acceptance, immutable spec) | **11/11 pass** |
| `test_onorm_knobs.py` (every blocking knob turned + 3 combos + 2 over-budget guards) | **14/14 pass** |
| `test_onorm_precision_baseline.py` (new) | **4/4 pass** |
| `test_onorm_trials.py::test_shape_trial` (40 interleaved perf trials) | **40/40 pass** |
| `test_onorm_perf.py` (5 compute-config variants) | **5/5 pass** |
| `eval/golden_tests/onorm/test_regression.py` (numerics: tiny `o`, saturating gate, large `o`) | **6/6 pass** |

---

## Measured Performance Profile

This is the section that reshapes the refinement queue, so the method is stated first.

**Method.** Single-shot profiler numbers for this op are known-unreliable (the implementer
recorded a 248 µs vs 102 µs swing for the same config across processes). Everything below is
either (a) a median of 5 **trial-major interleaved** repetitions inside one process, or (b) a
wall-clock mean of 20 back-to-back dispatches after a 3-dispatch warm-up with explicit
`synchronize_device`. Both methods agree to within 5 %, and the intra-process spread is < 1 %.

### Baseline device-kernel time (median of 5 interleaved trials, default knobs)

| Shape | token-blocks | cores used | device ns | per-core µs/block | achieved GB/s |
|---|---|---|---|---|---|
| B=1, T=64  | 2   | 2   | 239,587 | 240 | 6.3 |
| B=1, T=128 | 4   | 4   | 239,690 | 240 | 12.6 |
| B=1, T=640 | 20  | 20  | 244,110 | 244 | 61.3 |
| B=8, T=640 | 160 | 110 | 538,989 | ~337 | 221.4 |

**The op takes ~240 µs to process one 32-token block on one core, essentially independent of
core count.** Wall-clock (profiler off) confirms: 250.9 / 248.8 / 249.6 / 256.5 / 568.4 µs for
T=32 / 64 / 128 / 640 (B=1) and B=8,T=640.

### Where the time goes (B=1, T=640, per core, device zones, UNPACK TRISC)

| Phase | µs | share |
|---|---|---|
| **P7b `sigmoid(gate)` (SFPU)** | **152.7** | **63.9 %** |
| P4 normalize (`mul` bcast Col) | 29.3 | 12.3 % |
| P7c gate multiply (FPU) | 19.2 | 8.0 % |
| P1 sum-of-squares | 11.9 | 5.0 % |
| P5 weight scale | 9.4 | 3.9 % |
| P2 reduce + eps + rsqrt | 8.6 | 3.6 % |
| P6 untilize | 4.3 | 1.8 % |
| P7a tilize | 3.6 | 1.5 % |
| **TRISC total** | **239.1** | |

Dataflow, same run: NCRISC `read_o` 33.0 µs, `read_gate` 160.1 µs, `read_weight` 0.7 µs,
`fill_scaler` 0.3 µs; BRISC `write_out` 242.4 µs.

### Reading it

- **`read_gate`'s 160 µs is stall, not transfer.** `read_o` moves the *same* 256 KB through the
  *same* code path with the *same* `DM_BLOCK_TILES` in **33 µs**. The 127 µs difference is
  `cb_reserve_back` blocking behind P7b. Likewise BRISC's 242 µs is the writer waiting for
  `cb_out_tiles`, which only fills at the very end of each block.
- **P7b's 153 µs is real SFPU work, not a mirror stall.** Proof: it moves when you change
  compute-only knobs that cannot touch the NoC (table below), and it is 1.19 µs per tile ≈
  37 ns per SFPU vector op with a 32-bit DEST — the right order for `calculate_sigmoid` under
  `DST_ACCUM_MODE`.
- **The op is nowhere near DRAM-bound.** Per-core achieved bandwidth is **3.1 GB/s** against the
  ~17.9 GB/s single-core NoC ceiling the design itself cites, and aggregate is 61 GB/s at 20
  cores. Even at grid fill (110 cores) it reaches 221 GB/s, and per-core throughput *falls*
  there, so contention starts before the roofline does.

### Compute-config sensitivity (B=1, T=640; wall clock, 5 interleaved trials × 10 reps)

| Config | median | vs default |
|---|---|---|
| **default** (HiFi4, exact, `fp32_dest_acc_en=True`, half-sync) | 268.5 µs | 1.000× |
| `math_approx_mode=True` | 260.6 µs | 1.030× |
| `MathFidelity::LoFi` | 256.5 µs | 1.047× |
| `fp32_dest_acc_en=False` | 245.2 µs | 1.095× |
| all three together | 241.5 µs | 1.112× |
| **`dst_full_sync_en=True`** | 288.4 µs | **0.931×** |

Two conclusions:

1. **Every compute knob moves the number** — final confirmation that the op is compute-bound.
   If it were DRAM-bound, none of these would register.
2. **No config knob is the answer.** Even stacking all three precision relaxations buys only
   1.11×, so the sigmoid's cost is the **raw SFPU volume** (128 tiles × 32 vector ops per block
   on the one MATH thread), not the polynomial or the DEST width. The lever with real headroom
   is therefore *moving the work off the MATH thread*, not making each call cheaper — which is
   exactly what Refinement 1 targets.

**Design claim independently confirmed:** `dst_full_sync_en=True` is a **7 % silent regression**
with no correctness signal, exactly as `op_design.md` risk 14 predicts (it drops P7a off the
`can_use_fast_tilize()` path). The default correctly pins it to `False`. This is a real trap for
any caller who builds their own `WormholeComputeKernelConfig` — see Recommendations.

---

## Refinement Queue Shape

`TARGET − SUPPORTED = ∅` on **both** axes:

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| `dtype` | `[bfloat16]` | `[bfloat16]` | — |
| `layout` | `[TILE_LAYOUT]` | `[TILE_LAYOUT]` | — |

There are **no generality refinements to file**, and no undocumented omissions. This is by
design, not by neglect: `feature_spec.py`'s docstring says so explicitly ("DELIBERATELY NARROW…
TARGET is a single cell… there is no refinement backlog; the task is to make this one shape
correct and fast"), and the prompt repeats it. Widening to more dtypes / layouts / head counts
is stated as out of scope for this suite.

Consequently `op_requirements.md` is **all perf**, which the queue rules explicitly sanction
once generality candidates are exhausted. `feature_spec.py` declares no `LOOSE_CASES`, so no
shape is perf-flagged and the regions below are verifier-selected from the measurements above.

---

## Recommendations

**Priorities.** Refinement 1 (get the sigmoid off the saturated MATH thread) is worth more than
everything else combined — it is 64 % of the kernel on *every* supported cell. Refinement 2
(cross-core re-tile) is worth up to ~16–30× on the small-`T` shapes that a decode-path caller
actually issues, and nothing else can touch them: at T ≤ 128 the op occupies 2–4 cores of 110
and takes the same 240 µs as T=640. Refinement 3 is the cleanup pass that re-tunes the knobs
against whatever structure 1 and 2 leave behind.

**Cross-cutting concerns for whoever picks these up.**

- **Do not trust a single profiled dispatch on this op.** Use `test_onorm_trials.py`'s
  trial-major interleave, or a wall-clock loop with explicit `synchronize_device`. Two methods
  agreeing is the bar.
- **The per-phase device zones include each helper's `cb_wait_front`.** A starved phase reads as
  a slow phase. Cross-check any phase attribution against a compute-only knob that cannot touch
  the NoC (the config table above is the template).
- **`op_design.md`'s "we are DRAM-bound so this lever will not pay" reasoning is void.** That
  premise is measurably wrong, so the levers it waves off — `RECONFIG_MODE = off` (§1.5),
  compute block-size amortization (§6.2) — are back in play and are Refinement 3's content.
- **`fp32_dest_acc_en=True` is a precision-rule requirement, not a free knob.** It is worth
  1.095× if dropped, and the measured PCC margin (0.99999 vs a 0.995 bar) would absorb it — but
  `eval/prompts/onorm.txt` directs fp32 sum-of-squares accumulation in DST. Any refinement that
  wants that 9.5 % must either preserve fp32 accumulation by another mechanism or get an
  explicit documented deviation. Do not flip it silently.

**Observations that are not refinements.**

- **Caller foot-gun: `dst_full_sync_en`.** A caller-built `WormholeComputeKernelConfig` defaults
  this to `False`, so the common path is safe — but a caller who sets it `True` gets a measured
  7 % regression with no correctness signal. The op accepts it silently. Worth a `logger.warning`
  or a docstring callout if this op ever ships to external callers; not fixed here because
  warning on a legal config in a hot path is its own nuisance.
- **`PROPERTIES["math_fidelity"] = ["HiFi4"]` is `source: declared`** and describes only the
  *default* config; the op honours any caller-supplied `math_fidelity`. Harmless, but the
  property under-describes the op.
- **L1 headroom.** At the current knobs the CB footprint is 533 pages = 1,091,584 B
  (~74 % of the ~1.43 MB CB-available L1 after `_L1_CB_BASE_RESERVE`). No OOM today. The two
  re-tile buffers are 256 pages (512 KB, 47 % of the total) and scale linearly with
  `TOKENS_PER_BLOCK` — which is why `TOKENS_PER_BLOCK = 64` only fits once `NORM_CHUNK_TOKENS`
  or `GATE_CHUNK_TILES` comes down, as `test_knob_combo` demonstrates. Anyone raising
  `TOKENS_PER_BLOCK` in Refinement 3 must pay for it on one of those two knobs; the budget
  assert enforces it and names the order.
- **`_L1_CB_BASE_RESERVE = 72 KB` is a Blackhole-measured constant.** It is documented in place
  with the measurement that produced it, but it is arch-specific and would need re-deriving on
  Wormhole. Not a bug; a portability note.
- **`test_onorm_dmsweep.py`, `test_onorm_knobs.py`, `test_onorm_perf.py`, `test_onorm_trials.py`**
  are the implementer's measurement vehicles and are all marked DO NOT DELETE. They are the
  right harnesses for the perf refinements below — reuse them rather than writing new ones.

**On the `feature_spec.py` edit.** I changed one tuple in `INPUTS` (item 6). It is outside the op
directory, so if you would rather have it re-derived through `/golden-tests`, the required change
is exactly `INPUTS[4][2]: (2, 1, 1, V) → (1, 1, 1, V)`. `INVALID`, `TARGET` and the other four
`INPUTS` entries are untouched and correct as authored.
