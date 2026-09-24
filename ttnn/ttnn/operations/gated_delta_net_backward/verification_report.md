# Verification Report: gated_delta_net_backward

Phase-0 verification of the one-dispatch backward pass of the chunked gated delta rule.
Device: **Blackhole, 11 × 10 = 110 worker cores**, `get_max_worker_l1_unreserved_size() = 1 531 904 B`.

Artifacts produced by this pass:

| Artifact | Where |
|---|---|
| Golden results + categorized verifier report | `/tmp/gdnb_v1/{junit.xml,test_results.json,verifier_report.json}` |
| Precision baseline test | `tests/.../gated_delta_net_backward/test_gated_delta_net_backward_precision_baseline.py` |
| Perf baseline / no-regression guard set | `tests/.../gated_delta_net_backward/test_gated_delta_net_backward_perf_baseline.py` |
| Refinement queue | `op_requirements.md` |
| Phase-0 changelog | `changelog.md` |

---

## Headline

The op is **functionally complete against its own TARGET**: `TARGET − SUPPORTED` is empty except
`dtype = bfloat8_b`, which `feature_spec.py` declares `INVALID`. `xfail_expected = 0` and
`xpass_drift = 0` — the SUPPORTED rectangle is exactly the honest one, and there is **no generality
refinement to file**. Two golden cells fail, both the same `numerical-precision` cell at the
saturated-gate setting; per the registry-model routing rule they stay failing and are owned by
Refinement 1 rather than being silenced into `EXCLUSIONS`.

Consequently the refinement queue is **one precision refinement followed by measured perf work**,
and the perf work has a clear target: on a 110-core part, **no shape in `INPUTS` engages more than
32 cores and most engage 4–16**, because the work unit is `(bh, chunk)` and `BH·NC ≤ 32` across the
whole corpus.

---

## Code Review

Everything in this section was **fixed in place**; nothing was deferred except where stated.

### Fixed

1. **Environment-variable override of math fidelity — removed.**
   `_compute_config()` read `os.environ["GDN_FID"]` as the default `math_fidelity`. A knob that
   changes the numerical answer must travel with the call, not with the shell: an env var silently
   re-grades every test in the process and is invisible to the program hash's caller. The default
   is now the literal `ttnn.MathFidelity.HiFi4`, overridable only through
   `compute_kernel_config`. (A stale commit message claimed this had been dropped; it had not.)

2. **DRY / single source of truth for the uniform CB block sizes.**
   `MAXV`, `LVB`, `LITEM`, `MAXBLK`, `MAXBLK_G`, `NCOL`, `NCONST` were each written out **twice** —
   once in `_cb_pages()` (host, which sizes the CB) and once as a `constexpr` formula in
   `gdn_common.hpp` (device, which pushes/pops that many pages) — held together only by a
   `// mirrors _cb_pages()` comment. Same for the three named-slot counts (`6` const masks,
   `5` `cb_veca` slots, `9` `cb_vecb` slots). These are dependent quantities of the block extents,
   and a divergence between the two sides is a **fifo wrap, i.e. a hang**, not a warning. Routed
   through the one source: a new `_cb_blocks()` on the host is the only definition, `_cb_pages()`
   consumes it, and the same values reach the kernels as compile-time args 40..49. The kernel now
   `static_assert`s the two slot counts against its own slot layout, and `cb_const`'s offsets are
   derived from `NUM_CONST_MASKS` instead of a repeated `6`. A knob change now lands in exactly one
   place.

3. **`issued[2]` in the reader's gather pipeline → `issued[GATHER_DEPTH]`.**
   The per-slot in-flight counter array was sized by a literal `2` while its index is
   `w % GATHER_DEPTH`. Raising the depth knob (the design's own buffer-depth knob, and a candidate
   in a perf refinement) would have written past the array — a silent stack overrun, not an error.
   Now sized from the knob.

4. **`transpose_block` (the block entry point) replaces a per-tile transpose loop.**
   `tr_blk()` ran `tile_regs_acquire / transpose_tile / commit / wait / pack / release` **per
   tile**, paying a full DEST handshake for every one of `Rt·Nt` tiles. `api/compute/transpose.h`
   exposes `transpose_block(icb, start_itile, start_idst, ntiles)` as "the uniform block entry
   point of the transpose op group"; `tr_blk` now walks `DEST_LIMIT`-sized groups with one
   handshake each and packs the group with the tile-grid permutation applied at pack time.
   Transposes are hot here (materialized `Aᵀ` operands — `matmul` transposes `in1` only), ~9 per
   stage-P/G item plus one per scan step. **Measured** on the perf guard set: 1.813 → 1.726 ms at
   `(1,256,4,128,256)`, 1.620 → 1.567 ms at the same shape in bf16, 0.468 → 0.439 ms at
   `(2,64,4,64,64)`; neutral at `Ct = 1`, where a `[C,C]` block is a single tile and there is
   nothing to batch. (Run-to-run drift on this instrument is ~3%, so read this as
   "1.0–1.05×, never worse", not as a precise figure.)

5. **Dead debug scaffolding removed** — `STAGE_MASK` (and the `DO_P/DO_S/DO_G` stage gates and
   `ABLATE_GATHER`/`ABLATE_COMPACT` bits it carried). No test referenced it, its ablation bits
   "produce wrong results by design", and reaching it required editing the source constant anyway.
   These kernels are *kernel-config-ring-buffer bound* (the reason for `#pragma GCC optimize("Os")`
   on all three translation units), so removing branches and a CT arg is a small real win and
   removes any chance of shipping with a non-`0b111` mask.

6. **Dead code**: the unused `src` NoC address in `zero_l1()` and the unused `ntiles` in
   `gather_block()`.

7. **Stale comment corrected (compute kernel).** It claimed `cb_gatein` "carries `UnpackToDestFp32`
   (see the program descriptor)". Nothing in the descriptor configures that, and there is no such
   field: the datacopy init selects the unpack-to-DEST path automatically from the operand's DEST
   format (`llk_unpack_A<..., UnpackToDestEn>` when it is 32-bit). The comment now states the real
   mechanism — which matters, because that mechanism is exactly what keeps `g` at full width and is
   the thing Refinement 1 leans on.

8. **`l1_ledger.md` currency** (details under *L1 Ledger Audit*): `cb_veca` was documented as
   `6*Ct` pages where the code allocates `5*Ct`, which inflated every row of the footprint table by
   `Ct` f32 tiles. Corrected, and the corrected table now matches the **device-measured** peak L1
   to 0.1 KB. Audit 2's mechanism attribution was also corrected (the truncation is at the FPU
   source registers of a `Float32` page's *consumer*, not at the packer).

### Measured and reverted (recorded so it is not re-attempted blind)

- **Skipping the reader's zero-fill on whole chunks.** `gather_block()` zero-fills the whole
  destination block before gathering, but the re-pack provably overwrites every row of every tile
  unless the chunk runs past `T`. Restricting the fill to the ragged case removes ~250 KB of
  local NoC traffic per item at the largest shape. **Measured: neutral to 1.03× *worse*** across
  the guard set. The reader is RISC-issue bound, not bandwidth bound (~280 ns per gathered row
  against a ~1 KB read), so the DM engine's zeroing is effectively free while the added branch in
  the hot template is not. Reverted; the measurement is now a comment at the site so a perf
  refinement does not spend itself here.

### Advisories (soft prompt rules, not violations)

`eval/prompts/gated_delta_net_backward.txt` `## Rules`, checked against what is implemented:

| Rule | Verdict |
|---|---|
| **MUST** L2-normalize `q`/`k` in every test and probe | Honoured. Acceptance, golden, precision-baseline and perf-baseline inputs all come from `make_reference_inputs` / an in-file `l2()`. |
| **MUST** start the reverse scan from `dht` when supplied, else zero | Honoured — `load_final_state_grad` is gated on the `HAS_DHT` CT flag; pinned by `test_dht_only_path` and `test_oracle_dht_changes_gradients`. |
| **MUST** return `None` (not zeros) in `dh0` when `initial_state is None` | Honoured — the host allocates five outputs in that case, so the two shapes are distinct programs. Pinned by `test_dh0_is_none_without_initial_state`. |
| *prefer* parallelizing over `(B,H)` and chunks; do not parallelize the reverse scan in Phase 0 | Followed exactly. |
| *prefer* multiple `generic_op` dispatches over one kernel that does not fit L1 | **Not followed, deliberately and correctly.** The run's one-dispatch mandate is hard, and the fit argument the rule is protecting against is answered a different way: the `block_val_tiles` extent solve plus the `l1_ledger.md` closed form guarantee a fit at every INPUTS shape (worst case 1396 KB of a 1416 KB budget), and the closed form is now confirmed against device-measured peak L1. The ledger the rule asks for exists and predates the kernels. |
| *prefer* checking `dg` last and `dh0`/`dht` first when a gradient is wrong | Process advice; followed during triage. |

### Not fixed — architectural, filed or reported

- **`PROPERTIES["math_fidelity"]` claims all four fidelities, and none but HiFi4 is exercised.**
  There is no `math_fidelity` axis in `feature_spec.py` TARGET, so the golden cartesian cannot
  cover it, and the claim is `"source": "declared"`. The op does pass a caller-supplied
  `math_fidelity` straight through. Folded into Refinement 1's test matrix rather than left as a
  bare claim.
- **`H ≤ 32` is a hard `ValueError`, not a support axis.** The source page-index formula assumes
  `ceil(H/32) == 1`. Every INPUTS entry has `H ≤ 8`, and `H` is not a TARGET axis, so this is
  neither a queue entry nor a drift signal — but a real model with more than 32 heads per batch
  element would hard-error rather than degrade. Documented omission; it needs a TARGET/axis
  decision first, which is upstream of this queue.
- **`_LAST_SCRATCH` debug hook** holds the previous invocation's DRAM scratch tensors alive until
  the next call (`test_..._debug.py` reads them back with `ttnn.to_torch`). Not in the public
  contract, DRAM not L1, and load-bearing for the scratch-readback harness that isolated the `dg`
  precision term — kept as is.

---

## Registry Conformance

**Confirmed in the op file** (`gated_delta_net_backward.py`):

- `INPUT_TAGGERS` — three taggers, every one with the `(inputs, axes)` signature, in an order that
  matters: `tag_chunk_size` first so `tag_seq_alignment` can read `axes["chunk_size"]`, then
  `tag_head_dims`. ✓
- `SUPPORTED` — six axes: `dtype`, `layout`, `state_mode`, and all three tagger keys
  (`chunk_size`, `seq_alignment`, `head_dims`). Every axis the kernel gates on is present; nothing
  extra. ✓
- `EXCLUSIONS` — present and **empty**, which is the honest value: no cell inside SUPPORTED is
  refused. ✓
- `validate()` — checks SUPPORTED per-axis (`UnsupportedAxisValue`) **then** EXCLUSIONS
  (`ExcludedCell`), both from `ttnn.operations._op_contract`, in that order; mechanism caps
  (`chunk_size % 32`, `H ≤ 32`) raise `ValueError` *after* the support gate, which is the right
  split — a cap is an error, not a refusal. Mixed dtype/layout across the eight input tensors is
  also caught (the taggers only see `q`). ✓
- `validate()` is the **first statement** of the public entry point, before any program build. ✓
- **No `INVALID` symbol in the op file.** ✓ (grep-confirmed.)

**Auto-fixes applied to SUPPORTED based on XPASS evidence: none required** — `xpass_drift = 0`.

### INVALID audit (`eval/golden_tests/gated_delta_net_backward/feature_spec.py`)

One entry: `{"dtype": ttnn.bfloat8_b}`. Well-formed on all three rules:

1. **Single-tensor coupling** — single axis, no coupling at all, so the canonical cross-tensor
   mistake cannot arise. ✓
2. **Universe-must-change** — `g` and `beta` are per-`(B,T,H)` gate *sequences* whose last
   dimension is the head count, so a bf8b shared exponent lets one head's magnitude flush another
   head's decay, and `exp()` of that error is unbounded. As long as `dtype` is **one** axis for all
   eight tensors, that is a property of the data-format definition, not of the kernel. ✓
   (The entry's own note says a future axis split — `gate_dtype` separate from activation `dtype` —
   would remove it. That is a TARGET/axis-set change authored upstream via `/golden-tests`, not a
   refinement; see *Recommendations*.)
3. **Canonicalization** — n/a, and correctly so: the canonical `{bf8b, ROW_MAJOR}` activation entry
   is subsumed because bf8b is excluded wholesale and `ROW_MAJOR` is deliberately not in TARGET
   (a RM variant of this op would be a different algorithm, not a different address map). There
   are no weight-like optional-tensor axes (`dht`/`initial_state` presence is folded into
   `state_mode`), so no redundant-cell canonicalization is owed. ✓

**Cartesian-collapse check** (the trap for tagger axes): `INPUTS` spans both `chunk_size` values,
both `seq_alignment` buckets and both `head_dims` geometries, so no tagger axis is silently pinned.
Arithmetic checks out: 22 `INPUTS` × 3 `state_mode` × 3 `dtype` = 198 cartesian cells, of which
22 × 3 = 66 are bf8b and skipped, + 6 loose + 14 regression = **218 collected**. ✓

**No changes requested to `feature_spec.py`.**

### Golden-suite defect (not the op's — reported, not edited)

Five tests in `eval/golden_tests/gated_delta_net_backward/test_regression.py` fail in a graded run
with `eval.l1_profiling.L1ProfilingError: L1 profiling is enabled for a graded hardware test, but
the test has no device or mesh_device fixture`:

```
test_oracle_matches_in_tree_forward   test_oracle_gradcheck
test_l2_normalization_is_required     test_oracle_dh0_absent_without_initial_state
test_oracle_dht_changes_gradients
```

All five are **host-only oracle tests** (they never touch a device) and all five **pass standalone**.
`eval_test_runner.sh` exports `EVAL_CAPTURE_L1=1` for every graded run, and `l1_profiling`
raises when a graded test has no `device` fixture. They land in the uncharged `no_axes_found`
bucket, so they are not a loud category and do not gate shipping — but they are five permanent reds
in every future refinement run. The one-line fix is to give each of them the (unused) `device`
fixture, or to teach `l1_profiling` to skip an item with no device rather than raise. Not edited
here: it is graded-suite/harness code, and making graded tests pass is not the verifier's call.

---

## L1 Ledger Audit

`l1_ledger.md` is unusually complete — 32 CB rows with per-axis `spans`/`streams` accounting, a
symbol table with bounds, a closed form, and a per-tensor DRAM crossing budget. The five checks:

1. **Ledger currency — one defect, fixed.** Every CB the implementation declares has a row, every
   row is a live CB, and 31 of the 32 size expressions match `_cb_pages()` exactly. `cb_veca` said
   `6*Ct` pages and listed six slots (`decay, γ, w, decayᵀ, β, dc1`) where the code allocates
   `5*Ct` and the kernel lays out five (`decay, γ, w, β, dc1` — there is no stored `decayᵀ`; the
   transpose is a `cb_cd` transient inside `build_L`). The error propagated into the closed form and
   into all five rows of the footprint table. Fixed both. **The corrected closed form now matches
   the device-measured peak exactly**: 1396 KB predicted vs **1396.1 KB measured**
   (`metric.device_l1_peak_bytes`, `(1,128,2,64,128)` c64 fp32), and 344 KB vs 344.1 KB at the
   smallest shape — so the ledger is now a measurement, not an estimate.
2. **Capacity vs live set, both directions — clean.**
   *Over*: three CBs exceed their live set for pipelining (`cb_gather` at `Dg`, `cb_egr`/`cb_gegr`
   at `De`), each with its overlap mechanism named. Fifteen are `Da × block`, which the ledger
   correctly classifies as a **correctness** mechanism rather than a depth knob: a compute→compute
   CB is the only synchronization between the TRISC pack and unpack threads, so `X ← f(X)` needs
   two aligned halves to alternate between. That is the right call and it is the single largest
   term in the footprint (592 KB of 1368 KB at the largest shape), which is why it is the funding
   source the perf refinement is pointed at.
   *Under*: no CB `spans` an axis its capacity does not scale with. Spot-checked the load-bearing
   ones: `cb_kc`/`cb_kd`/`cb_cc`/`cb_cd` are tagged `V: streams` because the V-loop *accumulates*
   into them, and their capacity correctly carries no `Vb`; `cb_sa`/`cb_sb` are tagged
   `n: streams` because the running state is re-entered per chunk, not stacked, and their capacity
   is one `[Kt,Vb]` block — the claim the kernel actually implements (the state is never packed out
   to DRAM between scan steps). No capacity expression mentions `B`, `T`, `H` or `NC`. ✓
3. **Page format vs DEST width — clean, with the mechanism re-attributed.**
   `fp32_dest_acc_en = True` by default, so every `Float32` page is an fp32-DEST value: no wide
   page is carrying a value that DEST already rounded. The `in_dtype` pages are exactly the
   boundary buffers (`cb_qin/kin/vin/doin/gatein`, `cb_gegr`) — values that were `in_dtype` in DRAM
   or are about to be written back as `in_dtype`, so a wider page would buy nothing and would
   double the packer/unpacker bytes. The `Float32` pages are kept even when a caller passes
   `fp32_dest_acc_en = False`; that is a *declared precision floor*, stated as the one deliberate
   exception, not an oversight. Audit 2's *explanation* was wrong, though, and is corrected: the
   ~tf32 resolution it observes is imposed by the FPU **source registers of a page's consumer**, not
   by the packer — which matters because it is the difference between "carry `decay` as a
   coarse+fine pair" (the op file's suggestion) and "stop making a large-magnitude value an FPU
   operand at all" (Refinement 1, and 5.8× better in the model).
4. **Disjoint lifetime with no justification — none.** Every `Shares with / why not` cell is
   filled; 19 of 32 CBs carry two or more roles across disjoint lifetimes, and each "No share"
   names the concurrent phase that forbids the merge (e.g. `cb_ke`: `q̃` is live from the head of a
   stage-G item to `dk`'s first term, concurrently with all five other `[C,K]` buffers). The
   merges already taken are real: nine CB indices saved by folding the constants into one block and
   the per-item / per-V-block scratch loads into one transfer each. Nothing to fix.
5. **Bounds and closed form — clean.** Every symbol in a capacity expression is in the symbol table
   with a bound and the predicate that establishes it. `Kt` and `Vt` are honestly marked *not*
   hard-bounded: the solve consumes them and shrinks `Vb`, bounding the *product*, and the host
   raises a `RuntimeError` naming the minimum footprint if even `Vb = 1` does not fit. That is a
   proven bound, which is the requirement. Verified by running the solve over every INPUTS shape —
   it fits everywhere, worst case 1396 KB of a 1416 KB budget.

**Block-size defaults.** Interleaved-memory default = spread the split's work units across the full
grid, then take the coarsest block that fits. The spread is implemented (`split_work_to_cores(...,
row_wise=True)` over `BH·NC`), and the coarsest-that-fits is implemented as a genuine host extent
solve rather than a constant. The departure from `Vb = Vt` at `Kt = 4` is forced by L1 with the
inventory demonstrably minimized first (19 shared CBs, nine indices saved, three values recomputed
instead of stored, two phase boundaries expressed in place) — not by a budget solve that "settled".
The `Vb = 1` floor and its `NV`-scaling traffic term are the live question and are filed as a
measured perf refinement.

**Data-movement budget.** Present, per tensor, and consistent with the implemented split: the
face-row gather is counted once per input, the compact copies are counted as the extra write+read
they are, and — the one that is easy to under-count — `sc[kcd]`/`sc[p]` are counted at **`NV` reads**
because stage S's V loop is outside its chunk loop and both blocks are V-independent. The cheapest-
traffic split is **not** implemented and is correctly carried as a `deferred` regime row with a
positive reason: R3 (de-interleave stage 0) divides the gather term by `H` and is reachable *on top
of* what is built, because R1 already materializes exactly the head-major compact copies R3 would
produce. The dead end is enumerated, costed (84 MB vs 2.6 MB at the largest shape) and marked
`rejected` without being written. Occupancy is explicitly used as a tiebreaker, not as a
justification. Nothing to file.

**Per-core footprint** (closed form; `Da = De = Dg = 2`):

```
(6Ct² + 2 + Ct) + max(Ct,Kt,Vb) + (Ct² + 4Ct) + LVB      constants, colones, item/V-block loads
+ 7·Da·Ct·Kt                                             the seven [C,K] buffers      448 KB @ Ct2/Kt4
+ 3·Da·Ct·Vb                                             the three [C,Vb] buffers
+ (4Da+1)·Ct²                                            the five [C,C] buffers       144 KB @ Ct2
+ 2·Da·Kt·Vb                                             the two [K,Vb] state buffers
+ 5Ct + 9Ct + Da·Ct                                      the three column CBs
+ De·MAXBLK                        f32 tiles
+ (2Ct·Kt + MAXV + Ct·Vb + 2Ct + De·MAXBLK_G)  in_dtype tiles
+ Dg·(GATHER_STAGE_TOKENS·row_span_stride + 64)          the face-row staging window
```

Scaling: quadratic in `block_chunk_tiles` on the `[C,C]` terms, `Ct×Kt` and `Ct×Vb` on the working
buffers, `Kt×Vb` on the state (the only `Ct`-independent term), and **nothing** scales with `B`,
`T`, `H` or `NC`. `block_batch`, `block_head` and `block_chunks` select *which* block, not how big
it is.

---

## Design Conformance

| Binding dimension | Verdict |
|---|---|
| **Algorithm** | Matches. The backward uses the stored `Tinv` and the algebraic inverse VJP `dA = Tinvᵀ·d_attn·Tinvᵀ` — two `[C,C]` matmuls — and never re-runs the forward-substitution loop; `(I−A)⁻¹` is built by Neumann doubling with `neumann_steps = ceil(log2(chunk_size))` as a CT constant derived from `chunk_size` (5 at C=32, 6 at C=64), which is the correctness constant the design flags, not a tuning knob. `dq` carries no `dS`-derived term, which is why the `zero_do` regression passes exactly. |
| **Data-pipeline topology** | Matches: P (grid-parallel over `(bh,chunk)`) → per-group semaphore fan-in → S (one core per `bh`, sequential in chunk) → per-group fan-out → G (grid-parallel, V-accumulated). Reader owns every DRAM→L1 load and both waits; writer owns every L1→DRAM store and both increments — the RISC ownership the design specifies. |
| **Single dispatch** | Verified: one `ttnn.generic_op` call, all six (or five) outputs pre-allocated on the host and passed in `io_tensors`, return handle discarded. The four `[C,C]` constant masks are **built in the reader at boot** rather than uploaded, precisely so no `ttnn.zeros`/`ttnn.eye` becomes a second dispatch. |
| **Inter-core communication** | Matches, including the deadlock argument: **all** prep increments are issued before **any** wait anywhere in the program, the rendezvous is per-`(bh)` group (fan-in `NC` unicast increments, fan-out a host-computed unicast target list), and cores with no work in a group wait for 0. Two semaphores, no payload, twice per program. |
| **Work distribution / does it fill the machine** | Implemented as designed — `split_work_to_cores(grid, BH·NC, row_wise=True)`, with `row_wise=True` as the design requires. **But see the finding below: the design's own work unit under-fills this device on every corpus shape.** Both dataflow halves are batched (the reader issues a whole block's reads then one barrier; the writer issues a whole block's writes then one barrier; the gather is software-pipelined across `GATHER_DEPTH` staging slots with per-slot transaction ids, not a plain barrier). |
| **Blocking-model fidelity** | Every knob the planner named is a parameter with one definition: `BLOCK_CHUNKS`, `GATHER_STAGE_TOKENS`, `GATHER_DEPTH`, `EGRESS_DEPTH`, `BLOCK_DEPTH`, `ACCUM_DEPTH` as host constants, `Ct`/`Kt`/`Vt` derived from the shape, and `block_val_tiles` **solved on host** from the ledger's closed form. No CB page count scales with a whole-op dimension: grep-confirmed that no capacity expression contains `B`, `T`, `H` or `NC`. The DRY defect found (dependent quantities restated per language) is fixed above. `GATHER_STAGE_TOKENS` is 16 rather than the design's 32 — a deliberate, recorded knob-turn that funded `GATHER_DEPTH` 1→2 at constant L1 (measured 1.01–1.02×, small because the reader is issue-bound); that is a knob moving, not a knob collapsing. |
| **Block expression (reader / compute / writer, at the scheduling boundary)** | Clean in all three. Compute's `mmx()` is a real block matmul: one `reconfig_data_format` + one `matmul_block_init` per block, then a DEST-sized subblock walk with `ct_dim`/`rt_dim` derived from `DEST_LIMIT` (itself derived from `fp32_dest_acc_en` and `dst_full_sync_en`) — subblocking is left where it belongs, below the block. Every phase is a "reserve the block, run one op into it, finish it" triple funnelled through `blk_end()`, so the per-phase handshake is once per block. The reader's gather acquires a whole `[C,D]` block per handshake and pipelines the staging windows underneath it; the writer drains a whole block per `cb_wait_front` and issues every NoC write before one barrier. The per-tile loops that remain (`scatter_rows`'s two 16-element face-row runs, `gather_gate`'s 32 scalar extracts) are *inside* a block operation and are forced by the `[B,T,H,D]` tiling — a head is one **row** of every page — not by unit-at-a-time scheduling. The one genuine per-unit DEST handshake found, `tr_blk`, is fixed above. |
| **Axis accounting** | Present in every ledger row and consistent with the code (check 2 above). |

### Finding: the designed work unit under-fills this device

Not a deviation — the implementation spreads exactly what the design told it to — but the
performance consequence is large enough to be the queue's centre of gravity:

| Shape | `BH·NC` items | cores engaged (of 110) | stage-S cores | measured |
|---|---|---|---|---|
| `(1,256,4,128,256)` c64 | 16 | 16 | 4 | 1.73 ms |
| `(1,512,2,64,64)` c32 | 32 | 32 | 2 | 0.71 ms |
| `(1,128,2,64,128)` c64 | 4 | **4** | 2 | 0.69 ms |
| `(1,100,2,64,64)` c64 | 4 | **4** | 2 | 0.59 ms |
| `(2,64,4,64,64)` c32 | 16 | 16 | 8 | 0.43 ms |

`max(BH·NC)` over the entire 22-entry `INPUTS` corpus is **32**, so no golden cell ever uses more
than 29% of the grid, and the two 4-item shapes use 3.6%. The design anticipated this precisely —
regime **R4** (V-split across cores) carries the predicate `BH·NC < grid_area and Vt > 1`, which is
true for nearly every corpus shape on a 110-core part — and deferred it on the grounds that "R1
covers every shape". It does, correctly; it just leaves 70–96% of the machine idle while doing so.
That is Refinement 3.

---

## Precision Baseline

`test_gated_delta_net_backward_precision_baseline.py`, 4 shapes × 2 dtypes × 6 gradients, against
the float64 autograd oracle, `state_mode=with_h0_and_dht`, `g_scale=0.02`. `q`/`k` L2-normalized
(caller contract). PCC via `comp_pcc`/`assert_with_pcc`, abs errors via `comp_allclose`.

**float32**

| Shape | grad | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ratio med (p5 / p95) |
|---|---|---|---|---|---|---|
| (1,32,1,32,32) c32 | dq | 0.99999816 | 5.74e-03 | 1.19e-03 | 5.14e-03 | 0.9954 (0.983 / 1.005) |
| | dk | 0.99999706 | 6.15e-02 | 8.06e-03 | 5.00e-03 | 0.9957 (0.981 / 1.009) |
| | dv | 0.99999871 | 8.08e-03 | 8.90e-04 | 3.56e-03 | 0.9968 (0.988 / 1.005) |
| | dg | 0.99999538 | 5.69e-02 | 2.47e-02 | 5.85e-03 | 0.9950 (0.989 / 1.009) |
| | dbeta | 0.99999860 | 6.24e-02 | 1.61e-02 | 4.61e-03 | 0.9957 (0.988 / 1.005) |
| | dh0 | 0.99999555 | 6.04e-03 | 9.75e-04 | 2.99e-03 | 0.9999 (0.985 / 1.017) |
| (1,128,2,64,64) c32 | dq | 0.99999802 | 1.08e-02 | 1.43e-03 | 5.77e-03 | 0.9947 (0.983 / 1.006) |
| | dk | 0.99999606 | 1.24e-01 | 7.34e-03 | 5.53e-03 | 0.9952 (0.978 / 1.012) |
| | dv | 0.99999816 | 1.30e-02 | 6.07e-04 | 3.90e-03 | 0.9963 (0.983 / 1.009) |
| | dg | 0.99999158 | 2.26e-01 | 5.12e-02 | 8.09e-03 | 0.9930 (0.973 / 1.054) |
| | dbeta | 0.99999602 | 1.10e-01 | 1.38e-02 | 5.38e-03 | 0.9953 (0.980 / 1.017) |
| | dh0 | 0.99999491 | 1.57e-03 | 3.03e-04 | 3.66e-03 | 0.9981 (0.979 / 1.018) |
| (1,128,2,64,128) c64 | dq | 0.99999815 | 1.44e-02 | 1.93e-03 | 5.43e-03 | 0.9950 (0.984 / 1.006) |
| | dk | 0.99999665 | 1.74e-01 | 9.22e-03 | 4.98e-03 | 0.9958 (0.979 / 1.012) |
| | dv | 0.99999819 | 1.07e-02 | 5.39e-04 | 3.69e-03 | 0.9969 (0.985 / 1.010) |
| | dg | 0.99999911 | 2.49e-01 | 6.07e-02 | 5.35e-03 | 0.9949 (0.987 / 1.005) |
| | dbeta | 0.99999744 | 1.53e-01 | 1.59e-02 | 4.70e-03 | 0.9961 (0.977 / 1.011) |
| | dh0 | 0.99999273 | 1.76e-03 | 3.18e-04 | 3.83e-03 | 0.9995 (0.976 / 1.022) |
| (1,256,4,128,256) c64 | dq | 0.99999789 | 1.38e-02 | 1.89e-03 | 6.79e-03 | 0.9936 (0.982 / 1.005) |
| | dk | 0.99999712 | 3.59e-01 | 8.81e-03 | 5.87e-03 | 0.9944 (0.982 / 1.007) |
| | dv | 0.99999862 | 1.38e-02 | 3.25e-04 | 3.65e-03 | 0.9965 (0.986 / 1.007) |
| | dg | 0.99999692 | 8.03e-01 | 7.09e-02 | 7.56e-03 | 0.9923 (0.973 / 1.013) |
| | dbeta | 0.99999758 | 2.02e-01 | 1.84e-02 | 6.01e-03 | 0.9942 (0.979 / 1.011) |
| | dh0 | 0.99999786 | 6.86e-04 | 1.05e-04 | 3.26e-03 | 0.9975 (0.985 / 1.009) |

**bfloat16** (same shapes; full table in the test output)

| Shape | worst grad | worst PCC | worst rel-RMS | ratio med range |
|---|---|---|---|---|
| (1,32,1,32,32) c32 | dq | 0.99999185 | 4.78e-03 | 0.9969 – 1.0000 |
| (1,128,2,64,64) c32 | dk | 0.99999077 | 5.75e-03 (dg) | 0.9927 – 0.9988 |
| (1,128,2,64,128) c64 | dq | 0.99999218 | 4.72e-03 | 0.9974 – 0.9997 |
| (1,256,4,128,256) c64 | dbeta | 0.99999168 | 6.41e-03 (dg) | 0.9943 – 0.9987 |

**Assessment.** Uniformly strong and well inside both bands: worst PCC over all 48
shape×dtype×gradient measurements is **0.99999077** (bands: 0.999 fp32 / 0.99 bf16) and worst
relative RMS is **8.1e-3** (bands: 0.02 / 0.12). `dg` is the weakest gradient on 4 of 8
shape×dtype combinations, as the design predicts — it is the only one through `exp()` of a
cumulative sum. bfloat16 is *not* materially worse than float32 here, which corroborates the
design's decision to keep every internal CB (`Tinv`, decay, `L`, the state, the accumulators)
`Float32` regardless of input dtype: bf16 only touches the boundary buffers.

**Scale-bug triage (the ratio-spread column).** The `got/true` ratio has p5/p95 spreads of roughly
±2% around a median of 0.992–1.000 — a *broad* spread centred near 1.0, which is rounding noise,
**not** the tight-cluster-off-1.0 signature of a uniform scale or structural bug. So no
`supported_fail` here is misfiled: the two failures below are genuinely precision.

Worth recording, though: the ratio median is consistently **0.4–0.8% below 1.0** across every
gradient, shape and dtype. That is a systematic magnitude *deficit*, not noise (noise would
straddle 1.0), and it is the expected signature of truncating rather than round-to-nearest
behaviour at the FPU source-register width, compounded over the ~25 phase boundaries each block
passes through. It costs nothing at the current bands (rel-RMS 5e-3 against a 2e-2 gate) but it is
the same mechanism that makes the one failing cell fail, and closing that cell should shrink this
bias too — a useful secondary signal for Refinement 1.

**Recommended tolerances** (unchanged from the golden suite — the measurements justify them, do not
tighten): float32 `PCC ≥ 0.999`, rel-RMS ≤ 0.02; bfloat16 `PCC ≥ 0.99`, rel-RMS ≤ 0.12.
Equivalent `rtol/atol`: the measured per-element ratio band is ±2%, so `rtol = 2e-2` with
`atol = 1e-3 · |ref|max` is the honest elementwise statement — but PCC + relative RMS is the right
gate for a six-gradient VJP, because `max_abs` here tracks the (large) dynamic range of `dk`/`dg`
rather than any error mechanism.

---

## Perf Baseline

`test_gated_delta_net_backward_perf_baseline.py`. The real-time device profiler is **inactive** in
this build (`ttnn.device.IsProgramRealtimeProfilerActive() == False`, so the golden run's
`metric.device_kernel_ns` is empty), therefore: warm program cache, `synchronize_device` after each
call, 3 warm-up + 20 timed iterations, best of 3 runs. The op is one dispatch, so this is a faithful
relative instrument; run-to-run drift is ~3% on `min`.

| Guard-set case | `BH·NC` | `Vb` / `NVB` | min (ms) |
|---|---|---|---|
| `(1,512,2,64,64)` c32 fp32 | 32 | 2 / 1 | 0.709 |
| `(1,128,2,64,128)` c64 fp32 | 4 | 4 / 1 | 0.685 |
| `(1,256,4,128,256)` c64 fp32 | 16 | 1 / **8** | 1.731 |
| `(1,100,2,64,64)` c64 fp32 (ragged) | 4 | 2 / 1 | 0.593 |
| `(2,64,4,64,64)` c32 fp32 | 16 | 2 / 1 | 0.429 |
| `(1,256,4,128,256)` c64 bf16 | 16 | 1 / 8 | 1.574 |

Two readings drive the perf refinements:

- **`(1,256,4,128,256)` is 2.5–4× every other case** and it is the only one with `NVB > 1`. It
  pays the V loop 8 times in stage G, re-reads the V-independent `sc[kcd]`/`sc[p]` 8 times in
  stage S, and runs at `Vb = 1` — the extent at its floor — because the footprint at `Vb = 2`
  would need ~190 KB more than the budget has.
- **The 4-item shapes cost as much as the 16-item ones** (0.69 ms at 4 cores vs 0.43 ms at 16),
  i.e. below ~16 items the op is dominated by the per-item serial chain (two barriers, the
  sequential scan, ~25 phase boundaries per block) rather than by throughput. That is the
  low-occupancy regime R4 exists for.

---

## Verifier CLI Summary

`python3 -m eval.verify_supported /tmp/gdnb_v1 ttnn.operations.gated_delta_net_backward`
(218 collected, identical categories before and after the code-review fixes — no regressions, no
drift introduced):

```
supported_pass:         145
xfail_expected:           0     <- TARGET - SUPPORTED is empty except the INVALID dtype
invalid_skipped:         66
supported_fail:           2     <- both numerical-precision, SAME cell, owned by Refinement 1
xpass_drift:              0     (must be 0 to ship)  OK
xfail_wrong_mode:         0     (must be 0 to ship)  OK
supported_marked_xfail:   0                          OK
no_axes_found:            5     <- the harness defect above, not the op
```

**On the two `supported_fail`.** Both are the saturated-gate cell — `feature_spec.LOOSE_CASES`
`g_scale=8.0` at `(1,128,2,64,64)` c32 fp32, and its twin `test_regression.py::test_gate_saturation
[8.0-saturated_decay]`. The failing quantity is `dg`:

```
pcc = 0.999741   (gate 0.999 — PASSES)
rms = 0.024075   (gate 0.020 — FAILS, 1.20x)
max_abs = 2.56e-3   median_abs = 1.54e-5   inf = False   nan = False
```

Triaged per the scale-vs-precision rule and classified **genuine precision**, not a scale bug:
PCC is high but relative RMS is only 1.2× the band (not ≳0.1), the error distribution is heavy-
tailed rather than uniform (`max_abs / median_abs ≈ 166`), and the precision baseline's ratio
spread is broad and centred on 1.0 on every non-saturated cell. The causal chain is isolated, not
inferred: at `g_scale = 8` the intra-chunk `decay` cumsum reaches |242|, every consumer needs
`exp(decay[t] − decay[s])`, and a value of that magnitude read as an FPU operand carries ~0.2
absolute resolution at the ~tf32 source-register width. Modelling the two algebraically identical
`L` constructions at 10-bit operand width reproduces it and quantifies the fix
(3.4e-2 → 6.0e-3 relative RMS on `L`) — that model is Refinement 1's starting point.

Per the registry-model routing rule these two cells **stay failing**: the failure category and the
PCC/RMS pair *are* the signal, and an `EXCLUSIONS` entry or a shape-bucketing tagger would delete
the only instrument that can tell whether the refinement worked. Nothing else in the suite regresses
because of them.

---

## Recommendations

1. **Take Refinement 1 first and treat its model as load-bearing.** The
   `LT @ diag(g) @ strict_lower` reformulation of the decay-difference matrix was validated on the
   host at the same operand width the FPU uses and is **5.8× better** on `L`'s relative RMS. Note
   the op file's own suggestion — "carry `decay` as a coarse + fine pair" — attacks the wrong
   surface (it assumes the packer rounds; the consumer's source register does), and is both harder
   and worse. The real plumbing cost is that stage G needs `g`, which today is not in the
   four-slot `sc[vec]` block; both ways out are sketched in the queue entry.
2. **The op's perf story is occupancy, not bandwidth.** Three measurements point the same way: the
   reader is RISC-issue bound (~280 ns per gathered row against a ~1 KB read), removing ~250 KB per
   item of local zero-fill traffic was neutral-to-worse, and doubling the gather depth won 1.01×.
   Do not spend a perf refinement on the gather's byte count. Spend it on the 70–96% of the grid
   that is idle (Refinement 3) and on the `Vb = 1` floor at `Kt = 4` (Refinement 2).
3. **`(1,256,4,128,256)` c64 is the perf-critical shape and no loose case flags it.**
   `feature_spec.LOOSE_CASES` has no `attention:` PERF-FOCUS marker, so the perf refinements
   free-select their region; all of them target this shape, because it is the only corpus cell with
   `NVB > 1` and it costs 2.5–4× every other cell. If a hand-authored perf profile is ever added
   upstream, it should be this geometry (and it would then pin the target automatically).
4. **Consider splitting `gate_dtype` off the `dtype` axis upstream.** `feature_spec.py`'s INVALID
   entry says so itself: the entry exists only because one `dtype` axis covers both the activations
   and the per-`(B,T,H)` gate sequences. With `gate_dtype` separate, `bfloat8_b` activations + fp32
   gates becomes a legitimate cell and the INVALID entry goes away. That is a `/golden-tests`
   TARGET decision, not a refinement, and it is the single largest available expansion of this op's
   universe — the internal CBs are already `Float32` regardless of input dtype, so the kernel side
   is a boundary-format change.
5. **Fix the five host-only regression tests' fixtures** (or `l1_profiling`'s no-device path) so
   graded runs stop showing five permanent reds that have nothing to do with the op.
6. **`math_fidelity` is claimed but unexercised.** No golden axis covers it. Refinement 1's test
   matrix is the natural place to pin at least `HiFi2` and `LoFi` against the `dg` gate, since that
   refinement is already about the precision surface.
7. **L1 headroom is 20 KB at the largest `Vb=Vt` shape** (1396 KB of 1416 KB at
   `(1,128,2,64,128)` c64 fp32). Any refinement that adds a resident buffer must re-run the extent
   solve — it will silently drop `Vb` from 4 to 2 on that shape and quadruple its stage-S re-read
   traffic before it ever OOMs. The closed form in `l1_ledger.md` is exact (matched to measured peak
   L1 within 0.1 KB), so this is checkable on host before writing a kernel.
