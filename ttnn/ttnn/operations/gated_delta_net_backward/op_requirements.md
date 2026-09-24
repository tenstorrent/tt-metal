# Operation Requirements: gated_delta_net_backward

## Definition

- **Formula** — the vector–Jacobian product of the chunked gated delta rule. The forward being
  differentiated is, per batch element `b` and head `h`, over the sequence, with state `S [K,V]`:

  ```
  S_t = exp(g_t) · S_{t-1} + beta_t · k_t (v_t - S_{t-1}ᵀ k_t)ᵀ
  o_t = S_tᵀ q_t
  ```

  computed chunk-parallel: inside a chunk of `chunk_size` tokens the dependencies are resolved in
  closed form by a UT transform (Woodbury), `Tinv = (I − A)^{-1}` with
  `A = −((k⊙β) @ kᵀ) ⊙ L ⊙ strict_tril` and `L[t,s] = exp(decay[t] − decay[s])` for `s ≤ t`;
  across chunks `S` propagates sequentially. The backward runs in **both** directions — an
  intra-chunk part that is parallel over chunks, and an inter-chunk **reverse** scan carrying `dS`
  from the last chunk to the first:

  ```
  dv_new_i = u_i + P_i @ dS_{i+1}                  u_i = intra_iᵀ @ do_i
  dS_i     = Γ_i·dS_{i+1} + c_i − kcd_iᵀ @ dv_new_i    c_i = Q_iᵀ @ do_i
  dh0      = dS_0                                  (only when initial_state is given)
  ```

  and produces six gradients. The full per-chunk assembly (`dq, dk, dv, dg, dbeta`) is written out
  in `op_design.md` → *Backward derivation*.

- **PyTorch Reference** (the oracle — float64 autograd through the reference forward, passes
  `torch.autograd.gradcheck` at eps 1e-6 / atol 1e-7 / rtol 1e-5):

  ```python
  from eval.golden_tests.gated_delta_net_backward.helpers import (
      pytorch_gated_delta_net_backward,   # (q,k,v,g,beta,do,*,dht,initial_state,chunk_size,scale)
      _chunk_gated_delta_rule_fwd,        # the forward it differentiates — the definition
      make_reference_inputs,              # L2-normalizes q/k; use it, do not hand-roll inputs
  )
  ```

- **Import Path**: `from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward`

- **Function Signature**:

  ```python
  gated_delta_net_backward(
      q: ttnn.Tensor,                    # [B, T, H, K]  L2-normalized along K (CALLER CONTRACT)
      k: ttnn.Tensor,                    # [B, T, H, K]  L2-normalized along K (CALLER CONTRACT)
      v: ttnn.Tensor,                    # [B, T, H, V]
      g: ttnn.Tensor,                    # [B, T, H]     log-space decay gate (<= 0)
      beta: ttnn.Tensor,                 # [B, T, H]     write strength, in (0, 1)
      do: ttnn.Tensor,                   # [B, T, H, V]  gradient of the output
      *,
      dht: ttnn.Tensor = None,           # [B, H, K, V]  gradient of the final state
      initial_state: ttnn.Tensor = None, # [B, H, K, V]
      chunk_size: int = 64,              # multiple of 32
      scale: float = None,               # defaults to K ** -0.5, applied ON DEVICE
      compute_kernel_config = None,      # ttnn.ComputeKernelConfig — math fidelity, fp32 dest acc
      memory_config: ttnn.MemoryConfig = None,   # OUTPUT placement; defaults to q's
  ) -> tuple   # (dq, dk, dv, dg, dbeta, dh0);  dh0 is None when initial_state is None
  ```

  `g` comes **before** `beta`, matching the in-tree torch reference and the shipping forward
  binding. Six forward tensors positional, everything else keyword-only.

---

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### Why this queue is one generality refinement and then all perf

`TARGET − SUPPORTED` is **empty** except `dtype = bfloat8_b`, which `feature_spec.py` declares
`INVALID` (one `dtype` axis covers both the activations and the per-`(B,T,H)` gate sequences, and a
block-quantized shared exponent across heads makes `exp(g)` diverge). The verifier CLI confirms it:
`xfail_expected = 0`, `xpass_drift = 0`. There is no `(axis, missing_value)` pair left to add, so
the 2:1 generality:perf cadence degenerates after the single non-perf entry — Refinement 1, which
owns the two named `numerical-precision` failures — and every later phase is measured perf.
Widening the universe further needs a **TARGET** change upstream (splitting `gate_dtype` off the
`dtype` axis, which would retire the INVALID entry); that is a `/golden-tests` decision, not a
refinement, and it is written up in `verification_report.md` → *Recommendations*.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[float32, bfloat16]` (`bfloat8_b` is INVALID at TARGET, not a candidate)
- **SUPPORTED layout**: `[TILE]` — the whole layout universe for this op; ROW_MAJOR is deliberately
  absent from TARGET (it would be a different algorithm, not a different address map)
- **SUPPORTED shape-derived axes**: `chunk_size ∈ {32, 64}`, `seq_alignment ∈
  {chunk_aligned, chunk_ragged}`, `head_dims ∈ {square, wide_v}` — all values of all three
- **SUPPORTED op-specific axes**: `state_mode ∈ {do_only, with_h0, with_h0_and_dht}` — all three
- **EXCLUSIONS**: empty
- **Cores**: multi-core — `split_work_to_cores(grid, BH·NC, row_wise=True)` for stages P and G,
  one core per `(b,h)` for the sequential scan in stage S. Measured engagement on the 110-core
  Blackhole part: 4–32 cores, because `max(BH·NC)` over the whole `INPUTS` corpus is 32.
- **Compute config**: `compute_kernel_config` exposed; defaults `HiFi4`, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`, `dst_full_sync_en=False`. Every internal CB (`Tinv`, decay, `L`, the
  state, the accumulators) is `Float32` regardless of input dtype.
- **Blocking**: `block_chunk_tiles = Ct = chunk_size/32`, `block_key_tiles = Kt = ceil(K/32)`,
  `block_val_tiles = Vb` **solved on host** from the `l1_ledger.md` closed form (largest divisor of
  `Vt` that fits; 1..4 over the corpus), `block_chunks = 1`, `gather_stage_tokens = 16`,
  `gather_depth = egress_depth = accum_depth = 2`.
- **Golden baseline**: **145 / 147 runnable cells passing** (218 collected − 66 INVALID-skipped −
  5 harness-defect host-only tests). `supported_fail = 2` — both the same saturated-gate
  `numerical-precision` cell, owned by Refinement 1.
- **Precision baseline**: worst PCC 0.99999077, worst relative RMS 8.1e-3 over 4 shapes × 2 dtypes ×
  6 gradients (bands: fp32 0.999/0.02, bf16 0.99/0.12).
- **Perf baseline** (warm, best-of-3 `min`): 0.429–1.731 ms across the guard set; see
  `verification_report.md` → *Perf Baseline*.

---

### [ ] Refinement 1 — Close the saturated-gate `dg` precision floor

**Goal**: move these two named cells out of `numerical-precision` into passing, without touching
SUPPORTED, EXCLUSIONS, or any tagger:

```
eval/golden_tests/.../test_golden.py::test_op_loose[1x128x2x64x64-chunk_size=32-dtype=FLOAT32
                                     -head_dims=square-layout=TILE-seq_alignment=chunk_aligned
                                     -state_mode=with_h0_and_dht0]        (LOOSE_CASES g_scale=8.0)
eval/golden_tests/.../test_regression.py::test_gate_saturation[8.0-saturated_decay]
```

Both fail on `dg` only, and only on the RMS half of the band:
`pcc = 0.999741` (gate 0.999 — passes) but `rms = 0.024075` (gate 0.020 — fails, 1.20×),
`max_abs = 2.56e-3`, `median_abs = 1.54e-5`, no inf/NaN. Triaged as **genuine precision**, not a
scale bug: the error is heavy-tailed (`max/median ≈ 166`) rather than a uniform ratio, and the
precision baseline's `got/true` ratio spread is broad and centred on 1.0 everywhere else.

**Root cause, isolated — read this before choosing an approach.** At `g_scale = 8` the intra-chunk
`decay` cumsum reaches `|242|`. Every consumer wants `exp(decay[t] − decay[s])`, and `build_L()`
forms that as `X − Xᵀ` where `X = decay ⊗ 1` is produced by an outer-product **matmul**, so `decay`
enters through an FPU source register at ~tf32 width (10 explicit mantissa bits ⇒ ~0.2 absolute
resolution at that magnitude). The loss is at the *consumer's* source register, **not** at the
packer — so the op file's own suggested fix ("carry `decay` as a coarse + fine pair") attacks the
wrong surface, and keeping `decay` in fp32 in L1 changes nothing.

**Validated reformulation** (modelled on host at 10-bit operand width, `C = 32`, `g_scale = 8`):
build the difference matrix directly from `g`, whose entries are small, instead of from the
cumulative sum. With `LT[t,u] = 1 iff u ≤ t` (this is `CST_LT`) and `SL[u,s] = 1 iff u > s`
(this is `−CST_NSTRICT`, already resident):

```
D = LT @ diag(g) @ SL          ⇒  D[t,s] = Σ_{u=s+1..t} g_u = decay[t] − decay[s]  for s ≤ t,
                                  and exactly 0 for s > t (so the existing −1e4 BIAS + exp()
                                  masking of the upper triangle is unchanged)
```

Measured in the model, relative error of `L` at `g_scale = 8`:

| `L` construction | max rel-err | rel-RMS |
|---|---|---|
| current: `decay ⊗ 1 − (decay ⊗ 1)ᵀ` | 1.13e-1 | 3.45e-2 |
| proposed: `LT @ diag(g) @ SL` | 1.30e-2 | **5.99e-3** |
| proposed, pessimistic (accumulator re-rounded every step too) | 3.83e-2 | 1.21e-2 |

5.8× on the quantity `dg`'s whole chain hangs off, and `dg` currently misses by 1.20×.

**Plumbing — the only real cost.** `build_L()` runs in **both** stage P and stage G, and stage G has
no `g`: `sc[vec]` is `ST_VEC = 4*Ct` and its four slots are `decay, beta, dc1, w`. Two ways out,
both fine, pick one and say which in the changelog:
  * add a fifth slot (`ST_VEC → 5*Ct`, `LITEM → Ct*Ct + 5*Ct`, `NUM_VECA_SLOTS → 6`, stage-G load
    and `egress_vec4 → egress_vec5`), or
  * **replace** the stored `decay` with `g` and store `gamma = exp(decay)` in its place — stage G's
    only other use of `decay` is `gamma`, and `Γ = exp(dc1)` is unaffected (`dc1` is already the
    constant `log Γ` column). This keeps four slots and adds no DRAM.

`diag(g)` is `(g ⊗ CST_ROWONES) ⊙ CST_EYE` — one outer-product matmul plus one mask multiply, both
already-instantiated block ops. Net cost per item: +2 `[C,C]` matmuls and +1 mask multiply in
`build_L`, ×2 stages. Budget it: the largest shape has ~20 KB of L1 headroom (see below).

**Verifier notes**:
- **Ordering**: first, unconditionally. It is the only entry that moves a failing cell, and every
  later phase is perf work gated on "golden still green" — which is easiest to assert once the
  suite is fully green.
- **L1**: `l1_ledger.md`'s closed form is exact (matched to device-measured peak L1 within 0.1 KB).
  Headroom at `(1,128,2,64,128)` c64 fp32 is **20 KB** of a 1416 KB budget. If your approach adds a
  resident buffer, re-run `_solve_block_val_tiles` on host **before** writing a kernel: the solve
  will silently drop `Vb` from 4 to 2 on that shape and quadruple its stage-S re-read traffic long
  before it OOMs. Update the ledger row and the closed form in the same commit.
- **Do not** silence these cells. `EXCLUSIONS`, a soft `pcc_threshold` in `LOOSE_CASES`, or a
  `shape_size`-style tagger would delete the only instrument that can tell whether this worked. The
  recorded `rms` is the pass/fail signal.
- **Secondary signal worth watching**: the precision baseline shows a systematic `got/true` ratio
  median 0.4–0.8% **below** 1.0 on every gradient, shape and dtype — the signature of truncating
  behaviour at the same source-register width, compounded over ~25 phase boundaries. Closing this
  cell should shrink that bias; if it does not, you have fixed a symptom.
- **While you are in the precision surface**: `PROPERTIES["math_fidelity"]` claims all four
  fidelities and none but `HiFi4` is exercised anywhere (there is no `math_fidelity` axis in
  TARGET, so the golden cartesian cannot cover it). Add `HiFi2` and `LoFi` against the `dg` gate to
  this refinement's own test matrix so the claim stops being unbacked.

**Done when**: both named cells pass (`rms ≤ 0.020` on `dg` at `g_scale = 8`), the whole golden
suite is green apart from the five host-only harness failures (`no_axes_found`), `supported_fail`
is **0**, the precision baseline's worst PCC / rel-RMS do not regress, and the perf guard set shows
no regression beyond the instrument's ~3% drift.

---

### [ ] Refinement 2 — Split the value dimension across cores (regime R4)

**Type**: perf

**Goal**: this op leaves most of the machine idle on every shape in the corpus. The work unit is
`(bh, chunk)` and `max(BH·NC)` over all 22 `INPUTS` entries is **32**, so on the 110-core Blackhole
part measured engagement is **4–32 cores (3.6%–29%)**:

| Shape | items | cores | stage-S cores | measured min |
|---|---|---|---|---|
| `(1,256,4,128,256)` c64 fp32 | 16 | 16 | 4 | **1.731 ms** |
| `(1,128,2,64,128)` c64 fp32 | 4 | **4** | 2 | 0.685 ms |
| `(1,100,2,64,64)` c64 fp32 | 4 | **4** | 2 | 0.593 ms |
| `(2,64,4,64,64)` c32 fp32 | 16 | 16 | 8 | 0.429 ms |

Implement the regime `op_design.md` already enumerated and deferred: **R4 — V-split across cores**,
whose predicate `BH·NC < grid_area and Vt > 1` is true for nearly every corpus shape here. The
state factorizes **exactly** by V-column (this is what the shipping forward op's `distribute_scan`
exploits), so each core owns a `(bh, v_block)` pair and runs its own scan over a `[Kt, Vb]` state.
`block_val_tiles` already exists as an extent knob and the V loop is already the **outer** loop of
stages S and G, so the reachable part is a core-assignment change on an existing extent — plus the
combine that makes it a scheme-change rather than a knob-turn.

**The topology is the work** (there is no implementation skill for this class yet, so it is spelled
out here). Four of the six gradients reduce over `V` and therefore need a per-chunk cross-core sum
once the V axis is split:

```
per (bh, chunk):  Mraw [C,C]  +  d_attn [C,C]  +  dQ [C,K]  +  dP [C,K]  +  ndkcd [C,K]
                  + the column scalars (dβ_v, dΓ)
```

`dv` is V-local and needs no combine. At `Ct=2, Kt=4` that is ~22 tiles ≈ 88 KB per chunk fanned in
over `num_v_blocks` cores. Today those same partials are accumulated **for free in L1** by the
`mm_accum(..., first)` path inside the V loop, so the combine is the entire cost of the trade — it
is genuinely uncertain whether it wins, which is exactly why this is gated on measurement and not
on the argument above. Either a fan-in to the `v_block = 0` core followed by a re-broadcast, or a
ring/tree reduction; the existing barrier machinery (per-group semaphore fan-in with a host-computed
unicast target list, all increments issued before any wait) is the pattern to extend, not to
replace.

**Verifier notes**:
- **Hardest first, and it reshapes the ground under Refinement 4.** If V is split across cores,
  each core owns one V block by construction, `Vb = 1` per core stops being a compromise, and
  Refinement 4's whole premise (raise `Vb` above its L1 floor) may become moot. Do this one first
  and **re-measure before starting Refinement 4**.
- **Do not attack the gather.** Three independent measurements say this op is not
  bandwidth-limited: the reader is RISC-issue bound (~280 ns per gathered row against a ~1 KB
  read), removing ~250 KB per item of local zero-fill traffic measured neutral-to-1.03×-*worse*,
  and doubling `gather_depth` won 1.01–1.02×. Regime R3 (de-interleave stage 0, `÷H` on the gather
  term) is the *bandwidth* lever and is correctly deferred — it is not this phase.
- **Deadlock discipline is non-negotiable.** The existing two barriers are deadlock-free by
  construction because **all** prep is issued before **any** wait, and the rendezvous is
  per-`(bh)`-group rather than global. A third rendezvous inside stage G must preserve that
  property; get it wrong and you get a device hang, not a wrong answer. `op_design.md` →
  *Dataflow Strategy* → Barrier 1/2 rows state the argument that must keep holding.
- **Regime-pinned tests are required**, per the design: a regime that only triggers on some grid
  sizes passes on one device and fails on another. The predicate is host-checkable, so pin at least
  one shape on each side of it. Keep the single-dispatch contract: this is a core-assignment and
  semaphore change inside the one `generic_op`, never a second dispatch.
- `ttnn/ttnn/operations/examples/master.md` **does not exist on this branch** (`ttnn/ttnn/operations/examples/`
  is absent), so there is no measured-pattern catalog to consult; `op_design.md` →
  *Provenance of the perf figures* records the same absence and argues every knob structurally
  instead. Argue from this op's own measurements.

**Done when**: measured device time improves on `(1,256,4,128,256)` c64 and on at least one of the
two 4-item shapes (`(1,128,2,64,128)` c64, `(1,100,2,64,64)` c64), by more than the instrument's
~3% drift — use `test_gated_delta_net_backward_perf_baseline.py`, best of 3 runs, `min` column —
with **no regression** across the rest of that config-spanning guard set (one representative per
distinct kernel path × dtype: `NVB == 1` chunk-32, `NVB == 1` chunk-64 wide_v, `NVB > 1`, ragged,
multi-batch/multi-head, and bf16), the golden suite still green, and `supported_fail` still 0. If
the cross-core combine dominates and no configuration wins, **record the measurement in the
changelog and stop** — a negative result here is a real result and saves Refinement 4 from
inheriting a wrong premise.

---

### [ ] Refinement 3 — Amortize the per-item fixed cost on the low-occupancy shapes

**Type**: perf

**Goal**: below roughly 16 work items this op is dominated by its per-item serial chain, not by
throughput — `(1,128,2,64,128)` c64 costs **0.685 ms on 4 cores** while `(2,64,4,64,64)` c32 costs
**0.429 ms on 16 cores** for comparable total work. Each `(bh, chunk)` block pays ~25 phase
boundaries (an LLK init + a data-format reconfig each), ~25 CB handshakes, and a reader pipeline
fill, and with one item per core none of it is amortized or overlapped with a neighbouring block.
Speed up the ≤ 8-item shapes by paying those fixed costs fewer times. The levers, all of which the
planner already exposed as knobs (so this is a knob-turn phase — several cheap levers, not one
restructure):

- **`BLOCK_CHUNKS > 1`** — the design's *overlap* perf lamp, and the largest of these. It exists as
  a host constant wired to a CT arg and to `NC = ceil(T / (chunk·BLOCK_CHUNKS)) · BLOCK_CHUNKS`,
  but is pinned to 1. At small `K·V` the per-chunk working set is a fraction of L1, so 2 or 4
  chunks per block amortizes the ~25 inits across them. Watch the interaction with `NC` and with
  the scan owner's `sem_prep == NC · num_owned` count.
- **Buffer-depth co-tune** — `BLOCK_DEPTH` (1 today) on the four gathered input CBs
  (`cb_qin/kin/vin/doin`) lets the reader run ahead into item `n+1`; `EGRESS_DEPTH` (2) is the
  compute→writer overlap. Both trade L1 for overlap, and `block_val_tiles` is what funds them —
  changing either re-runs the extent solve, so co-tune them rather than turning one.
- **Reconfig elision** — `mmx()` issues `reconfig_data_format(cbb, cba)` per block (and again per
  subblock on the accumulate path). Many of this op's ~25 boundaries are fp32→fp32 and the reconfig
  is then a no-op that still costs issue time. Elide **only** on boundaries that cannot touch an
  `in_dtype` CB; the comment at `mmx()` explains exactly why the matmul operand order makes this
  dangerous (`in0 → SrcB`, `in1 → SrcA`), and eliding across a real format change is silent
  corruption, not a failure.
- **The granularity floor still applies**: whole tiles minimum, coarser amortizes up to the point
  where it stops fitting. `Ct` is 1 or 2 and is *not* a free knob (it is `chunk_size / 32`, a
  caller-visible support axis) — `BLOCK_CHUNKS` is the knob that coarsens the block here.

**Verifier notes**:
- Order after Refinement 2 because R2 may change how many items a core holds, which is the very
  quantity these levers amortize over. Re-read the item counts before choosing values.
- These kernels are **kernel-config-ring-buffer bound** — the reason all three translation units
  carry `#pragma GCC optimize("Os")`, and at `-O3` the three binaries were ~105 KB against a ~70 KB
  budget. A lever that duplicates an LLK init sequence (a new template instantiation, an unrolled
  variant) can fail to *fit* rather than fail to be fast. `L1_KERNEL_CONFIG_RESERVE = 80 KB` in the
  descriptor is the slack the extent solve budgets against; if you grow the binaries, that constant
  is what has to move, and it comes straight out of `block_val_tiles`.
- `master.md` is absent from this branch (see Refinement 2's note); argue from this op's own
  measurements.

**Done when**: measured device time improves by more than ~3% on both 4-item shapes
(`(1,128,2,64,128)` c64, `(1,100,2,64,64)` c64) using
`test_gated_delta_net_backward_perf_baseline.py` (best of 3, `min`), with no regression across the
rest of the guard set, the golden suite green, and `supported_fail` still 0. Record which levers
moved the number and which did not — a measured no-op is worth writing down, as the zero-fill
experiment in `verification_report.md` shows.

---

### [ ] Refinement 4 — Lift `block_val_tiles` off its floor at `Kt = 4`

**Type**: perf

**Goal**: **re-measure first — Refinement 2 may have made this moot.** At `K = 128` the host extent
solve lands on `block_val_tiles = 1`, the minimum, on both of the two largest shapes:

| Shape | `Ct` | `Kt` | `Vt` | solved `Vb` | `NVB` | footprint | `Vb = Vt` would be |
|---|---|---|---|---|---|---|---|
| `(1,128,2,128,128)` c64 | 2 | 4 | 4 | **1** | 4 | 1368 KB | 1940 KB — over the 1416 KB budget |
| `(1,256,4,128,256)` c64 | 2 | 4 | 8 | **1** | 8 | 1368 KB | 2884 KB — over budget |

`num_v_blocks` then costs two things: stage G walks its V loop `NVB` times per item, and — the term
`l1_ledger.md` singles out as *the one that scales with `num_v_blocks`* — stage S re-reads the
**V-independent** `sc[kcd]` and `sc[p]` blocks (`Ct·Kt` tiles each) once per V block, because its V
loop is outside its chunk loop. `(1,256,4,128,256)` c64 measures **1.731 ms**, 2.5–4× every other
guard-set case, and it is the only corpus cell with `NVB > 1`. Bring `Vb` to at least 2 there, or
remove the `NVB` scaling another way. Two levers:

- **Free working-set bytes to buy `Vb = 2`** (~190 KB needed against 48 KB of slack). The funding
  is visible in the closed form: the seven `[C,K]` buffers are `7·Da·Ct·Kt = 448 KB` and the five
  `[C,C]` buffers `(4Da+1)·Ct² = 144 KB` of the 1368 KB, all at `Da = ACCUM_DEPTH = 2`. `Da` is
  **not** a depth knob to turn down — `l1_ledger.md` → *The two capacity patterns* explains that it
  is the only way a compute→compute CB can express `X ← f(X)` while keeping the TRISC pack and
  unpack threads synchronized, and turning it to 1 is a hang. Freeing bytes therefore means
  **shortening a lifetime**: Audit 3 currently argues all six `[C,K]` roles are concurrent in
  stage G, so this is a stage-G phase-reordering question, not a resizing one. Two fewer `[C,K]`
  buffers is 128 KB.
- **Or kill the `NVB` re-read directly** — restructure stage S so the V-independent blocks are read
  once. Note the obvious version does not work: swapping the loops needs all `NVB` state blocks
  resident (`Da·Kt·Vt = 256 KB` extra at the largest shape), which is the same L1 wall. Something
  cleverer, or nothing.

**Verifier notes**:
- **Check the premise before spending the phase.** If Refinement 2 landed, each core owns one
  `v_block`, `Vb = 1` per core is correct by construction rather than a compromise, and the
  measured 1.731 ms should already have moved. Re-run the guard set and only continue if the
  `NVB > 1` shape is still the outlier.
- The closed form in `l1_ledger.md` is exact — matched to device-measured peak L1 within 0.1 KB —
  so every candidate can be evaluated on host by running `_solve_block_val_tiles` before any kernel
  is written. Do that first; it is free and it tells you whether a candidate reaches `Vb = 2` at
  all.
- Any buffer change must update the ledger's row, its axis accounting, the closed form, and the
  footprint table in the same commit. The verification pass already found one stale row (`cb_veca`
  documented at `6*Ct` where the code allocates `5*Ct`), and the footprint table is only trustworthy
  because it now agrees with the measurement.

**Done when**: either (a) `block_val_tiles ≥ 2` at `(1,256,4,128,256)` c64 with measured device
time improving by more than ~3% and the ledger updated to match, or (b) the `NVB` re-read term is
removed with the same measured improvement, or (c) the premise is re-measured as already resolved
by Refinement 2 and that measurement is recorded in the changelog. In cases (a) and (b): no
regression across the config-spanning guard set, golden suite green, `supported_fail` still 0.
