# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O = softmax(Q·Kᵀ·scale + mask) · V`, computed via FlashAttention-2
  (tiled online-softmax over KV blocks; the `S_q × S_kv` score matrix is never
  materialized). `scale` defaults to `1/sqrt(D)`.
- **PyTorch Reference**:
  ```python
  def sdpa(q, k, v, attn_mask=None, scale=None):
      B, H, Sq, D = q.shape
      Hkv = k.shape[1]
      if scale is None:
          scale = 1.0 / math.sqrt(D)
      if Hkv != H:                                   # GQA / MQA
          rep = H // Hkv
          k = k.repeat_interleave(rep, dim=1)
          v = v.repeat_interleave(rep, dim=1)
      scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
      if attn_mask is not None:
          scores = scores + attn_mask.float()
      return torch.matmul(torch.softmax(scores, dim=-1), v.float())
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                          # (B, H, S_q, D)
      key: ttnn.Tensor,                            # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                          # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor | None = None,        # (B, {1,H}, S_q, S_kv) additive
      is_causal: bool = False,                     # mutually exclusive with attn_mask
      scale: float | None = None,                  # None → 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor                                 # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A `[~]` partial's sharper follow-up appends a lowercase letter to the parent's number (`Refinement 1b`, `Refinement 1c`, …), ordered immediately after its parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.

> **TARGET − SUPPORTED gap covered by this queue** (everything else in TARGET is
> already SUPPORTED and green):
> - `dtype`: float32, bfloat8_b → **R2**
> - `fp32_dest_acc_en`: False → **R2**
> - `alignment`: w_non_aligned, h_non_aligned → **R1**
> - `mask_mode`: causal → **R4**
>
> No `memory_layout`/sharding axis exists in the golden TARGET, so flash-decode
> (cross-core `S_kv` split) and GQA KV-multicast are **not** queued — they have no
> golden cell to unlock (see verification_report.md "Observations").

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha, gqa, mqa]
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **EXCLUSIONS**: none
- **Cores**: multi-core (`split_work_to_cores` over `B·H·n_q_chunks`, interleaved DRAM)
- **Blocking**: `Sq_chunk_t`/`Skv_chunk_t`/`KV_DEPTH` fitted once by `_fit_l1`
- **Compute config**: HiFi4 + fp32 DEST (default), knobs threaded from `compute_kernel_config`
- **Golden baseline**: **212 / 212** supported cells passing (verifier CLI: supported_fail=0, xpass_drift=0, xfail_wrong_mode=0)

---

### [x] Refinement 1 — Non-tile-aligned shapes (w_non_aligned + h_non_aligned)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`,
handled **natively in the kernel** (no `ttnn.tilize`/`to_layout` wrapper). Covers
both non-aligned legs in one refinement (they share the reader/compute edge
machinery):
- **`w_non_aligned` (D % 32 ≠ 0)**: zero-pad the last D-tile in the reader so the
  Q·Kᵀ contraction (over `Dt`) and the P·V free-dim are correct; drop the padded
  output columns of the last D-tile in the writer.
- **`h_non_aligned` (S_q or S_kv % 32 ≠ 0)**: the last Q-chunk writes only valid
  rows; **the last KV tile's padding rows must be masked to −∞ before the
  row-max/exp/sum** so they fall out of the softmax denominator — use the reduce
  helper's partial-scaler / mask path (`ReducePartialScaler::last_tile_at` +
  `calculate_and_prepare_partial_reduce_scalers` / `prepare_reduce_mask` on the
  reader). This is the structural part — attention's softmax reduction spans the
  padded KV axis, so a plain edge-pad is not enough.

**Implementation skill**: /memory-layouts

**Verifier notes**: **hardest generality refinement — land it first, on the
smallest test surface** (before R2 adds float32/bf8b, so the edge machinery is
validated on bf16 alone). While here, replace the descriptor's `_chunk_size`
largest-divisor trick with a coarse chunk (`min(axis_t, 4)`) + a masked partial
remainder (the divisor trick avoids partial chunks today but would collapse to a
1-tile chunk for a prime tile-count > 4 — see verification_report.md). The skill
body owns the last-tile zero-pad/mask pattern; do not re-derive it here.
`bfloat8_b + non_tile_aligned` is expected to fail and becomes an EXCLUSION armed
by R2, not by this refinement.

**Done when**: the `alignment ∈ {w_non_aligned, h_non_aligned}` golden cells
(currently `xfail_expected`) pass; the mask-reduction correctly excludes KV
padding (verify on `test_regression`-style partial-S_kv shapes); prior phases green.

**Landed (R1)**: both alignment values added to `SUPPORTED`; golden **252 passed**
(212 prior + 40 non-aligned), **0 failed / 0 xpass**; prior unit + regression
unchanged (the 9 `test_regression` misses are the same pre-existing aligned-
adversarial precision cells, R2's target). `w_non_aligned` (D%32) rides the
`from_torch` tile zero-padding through the QKᵀ contraction and PV free dim (no
reader change); `h_non_aligned` S_q writes only valid rows (whole-tile write +
logical slice); the structural piece — S_kv%32 — is an additive **−∞** mask
(bf16 `0xFF80`, face-aware `fill_vertical_mask_tile` in the reader, mirroring
production `fill_vertical_tile_bf16`) added to the last KV chunk's boundary tile
before the row-max/exp/row-sum, reusing the existing additive-mask `add` path.
The `_chunk_size` divisor trick was **kept** (correct + DRY + general for every
tested/realistic shape) and its generalization deferred to R1b below.

### [x] Refinement 1b — Coarse-chunk + partial-remainder (replace `_chunk_size` divisor trick)

**Goal** (verifier-noted "while here" from R1, deferred): replace
`_chunk_size(axis_t, target)`'s largest-divisor rule with a coarse chunk
`min(axis_t, 4)` plus a **partial last chunk** (fewer whole tiles than
`Sq_chunk_t`/`Skv_chunk_t`). Today the divisor trick keeps every chunk whole
(so the *only* partial unit is the last S_kv tile's columns, handled by R1's
mask), but for a **prime tile-count > 4** (e.g. `S_kv = 32·101 → Skv_t=101`) it
collapses to a 1-tile chunk — a granularity-floor violation that repays per-chunk
reconfig/init/fill-drain overhead every tile.

**Exact lever**: thread a per-chunk runtime tile count
`min(chunk_t, axis_t − j·chunk_t)` into the reader's read counts, the compute
`MatmulBlockShape`/`ReduceInputBlockShape`/`EltwiseShape` (all take runtime
extents) with a re-derived matmul subblock decomposition for the partial `n`,
and the writer's tile counts — for **both** the Sq q-chunk and the Skv loop. This
is a core-loop restructuring touching all three kernels (regression risk to the
212 aligned + perf-flagged shape), so it is split out rather than bundled into R1.

**Verifier notes**: **no current test exercises a prime tile-count > 4** (golden
`S_kv` are all composite or chunk-4-divisible; worst real case `Skv_t=6 → 3`), so
this is a generality/perf hardening with no golden cell to unlock — add a
`test_regression`-style shape (e.g. `S_kv=3232`, `Skv_t=101`) alongside it.
Gate on the full golden suite staying green (no regression) as the acceptance net.

**Done when**: `_chunk_size` uses the coarse-chunk + partial-remainder scheme; a
prime-`Skv_t`(>4) shape runs at chunk 4 (not 1) and is correct; the full golden
suite + unit + regression stay green.

**Landed (R1b)**: `_chunk_size` now returns a coarse chunk `min(axis_t, target)`
with a **partial last chunk** of `axis_t % chunk` tiles; the reader/compute/writer
thread a per-chunk runtime tile count `min(chunk, axis_t − j·chunk)` — `sq_valid`
(M extent, per q-chunk work unit; compute decodes `qc` from `start_wu`) and
`skv_valid` (QKᵀ N / PV K, per KV chunk) — through the read counts, the
`MatmulBlockShape`/`ReduceInputBlockShape`/`EltwiseShape` runtime extents, and the
write counts, for **both** axes. The matmul N-subblock decomposition is derived
**on-device** (`decomp_n`, replacing the host `_matmul_subblocks` — single source
of truth). Prime `Skv_t`=101 → **chunk 4** (26 chunks, last=1), not the divisor
trick's chunk 1; no prime collapses to 1. **Constraint discovered on device:** the
partial remainder must **divide** the chunk (`rem | chunk` ⇔ `2·rem ≤ chunk`) — the
score-block CBs (`cb_scores`/`cb_exp`) are ring buffers read by linearly-indexed
compute (row-max reduce + exp), and the in-place mask `add` rotates the read pointer
by the per-chunk tile count; a remainder that doesn't divide the chunk offsets the
reduce window past the ring wrap → OOB unpack → packer wedge (`Skv_t=7` at chunk 4
gave remainder 3, `2·3>4`, and hung). So `_chunk_size` picks the largest straddle-safe
chunk ≤ target (`Skv_t=7 → 3`, `Skv_t=11 → 2`, `Skv_t=101 → 4`, `Skv_t=296 → 4`);
the granularity floor is raised from 1 to ≥2 (and 4 wherever `rem | 4`). QKᵀ
`out_subblock_w` is held **constant** across the KV loop (optimal when no partial
chunk — incl. the perf-flagged `Skv_t=296` shape; `1` for partial-chunk shapes) so
`mm_block_init_short` never reconfigs the packer to a partial width mid-loop.
Golden **252 passed / 2017 xfailed** (0 fail, 0 xpass — no SUPPORTED change); unit
**62/62** (prod mode); `test_regression` unchanged (same 9 R2-target precision
misses). A strict *chunk-4-for-every-prime* variant would need a pad-to-full +
`−∞`-mask scheme (compute always full blocks) — deferred; no test/golden cell needs
it and the perf delta is marginal.

### [x] Refinement 2 — Numerical configurability (dtype + compute-config + intermediate precision)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add
`False` to `SUPPORTED["fp32_dest_acc_en"]`, expose the caller's
`compute_kernel_config` (`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`)
end-to-end, and correct intermediate-CB precision (`cb_scores`/`cb_exp` fp32 where
it pays, `UnpackToDestFp32` tagging). Cells that fail out of the box land in
`EXCLUSIONS`, **not** their own refinement — arm at minimum:
- `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` — maxed input + non-maxed
  accumulation is lossy; refuse it (honor the caller's flag, don't silently force
  True). The golden `TOLERANCES` omits this combo for the same reason.
- `{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` (and `h_non_aligned`)
  if bf8b + non-aligned misses tolerance — the canonical `/numeric-formats-metal`
  EXCLUSION. (Depends on R1 having added the non-aligned values first.)

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **pulled ahead of the (harder) causal refinement (R4)** because
the perf-flagged loose case `1×10×9472×128` pins `fp32_dest_acc_en=False` — that
shape is `xfail` until this refinement lands it in SUPPORTED, and **R3 (perf)
cannot run against it until then**. Also fixes/relaxes the 9 `test_regression.py`
precision failures (float32 dtype + fp32 intermediates push the adversarial
distributions back under tolerance — see verification_report.md; they are genuine
bf16 precision, ruled not-a-bug). Order after R1 so the bf8b + non-aligned
EXCLUSIONS have their axis values to reference. Pass condition per the skill: zero
kernel changes when helpers/descriptor precision are wired correctly.

**Landed (R2)**: `SUPPORTED["dtype"] += {float32, bfloat8_b}` and
`SUPPORTED["fp32_dest_acc_en"] += False`, with **zero compute-kernel changes** (the
pass condition held — every phase is helper-based, no hard-coded formats). All
descriptor-level: `cb_scores`/`cb_exp` promoted to **Float32 under fp32-DEST**
accumulation (bf16 otherwise, so the perf-flagged bf16+16-bit-DEST shape stays
byte-identical); consumed by FPU ops (add/reduce/sub/matmul) so **not**
`UnpackToDestFp32`-tagged — the L1 format alone lifts bf16→TF32 through the softmax.
`_fit_l1`/`_working_set_bytes` now use real per-dtype tile bytes + the intermediate
format (fp32 input no longer under-counts L1). `compute_kernel_config` threaded
end-to-end (added `dst_full_sync_en`; math_fidelity/fp32_dest_acc_en/math_approx_mode
were already wired). **EXCLUSIONS armed** (all verifier-pre-authorized): `{float32,
fp32_dest_acc_en=False}` (maxed input + non-maxed acc, honored not forced) and
`{bfloat8_b, w_non_aligned}` + `{bfloat8_b, h_non_aligned}` — the canonical
block-float × partial-tile incompatibility (measured PCC ≈ 0.2–0.5 on the `S_kv%32≠0`
additive-−∞ mask path; both alignment tags carry it, so both refused; bf8b +
tile_aligned is fully supported). Golden **1181 passed / 1088 xfailed** (0 failed,
0 xpass — no drift), up from 252 at R1b; the perf-flagged loose case
(`1×10×9472×128`, bf16, `fp32_dest_acc_en=False`) now **runs and passes**, unblocking
R3. Unit **62/62**; translated bf16 sanity green. New
`test_scaled_dot_product_attention_precision_matrix.py` (8 shapes × 3 dtype × 4
fidelity × 2 acc × 2 dist, EXCLUSION cells skipped): **272 passed / 112 skipped**;
worst non-degenerate PCC 0.9905 (bf8b/LoFi). `test_regression.py`: the fp32
intermediates fixed **2 of the 9** pre-existing precision misses (the genuine-precision
`×10`-magnitude peaked-softmax cases); the remaining **7** (uniform/negative) are
`max_abs = 1 bf16 ULP`, `ulp_p99=1` — the documented normalized-RMS-on-near-constant-
reference metric artifact, floored by the **bf16 output** quantization (the regression
tests hard-code bf16, so R2's float32 path cannot reach them). Not a bug, never green
in prior phases, outside the registry cartesian (no golden gate).

### [~] Refinement 3 — Speed up the perf-flagged profile (data-movement)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `1×10×9472×128` (bf16, MHA, self-attn,
`fp32_dest_acc_en=False`, HiFi2) as the mandatory perf target, with
`expected_math_util = 0.35` (≈2.16 ms kernel time) as the goal and a **soft**
`pcc_threshold = 0.997` as the sole gate. Optimize **this** shape. The dominant
lever is data movement: the reader currently issues **one async read + one barrier
+ one push per tile** (the `double_buffer` anti-pattern) even though the KV CBs are
double-buffered, leaving the NoC latency-bound. Apply the relevant patterns from
`ttnn/ttnn/operations/examples/master.md` — `double_buffer` (batch a block of
async reads then one barrier; the CBs are already `KV_DEPTH`-deep) and
`reader_placement` (the split already uses `row_wise=True` — confirm it's optimal
here) look applicable. No SUPPORTED change.

**Verifier notes**: **depends on R2** — the flagged shape's `fp32_dest_acc_en=False`
must be in SUPPORTED (added by R2) before this shape can run at all (it is `xfail`
today). This is the reason R2 precedes R4 in the queue. Headroom is confirmed
qualitatively: the largest measured supported cell (`1×8×4096×128`, 3.47 ms /
110 cores) extrapolates to well above the 2.16 ms target for the ~5× larger
flagged shape, and the shape is data-movement-bound. `/perf-roofline-dm` can
confirm before investing.

**Done when**: measured device-ns improves on the flagged shape (moving
`expected_math_util` toward 0.35), its soft `pcc_threshold=0.997` still holds, the
golden suite is green, and no regression across the config-spanning guard set (one
representative per distinct kernel path × layout × placement — here: mask
none/custom × a small and a large shape, DRAM and L1 output).

**Landed (R3, partial)**: the double_buffer anti-pattern was removed — the reader
now batches a whole KV/Q chunk of async reads behind **one** `noc_async_read_barrier`
(the writer twin batches the whole output chunk behind one `noc_async_write_barrier`),
via a `read_tiles<cb,batch>` / `write_tiles<cb,batch>` helper. Batching is gated on a
straddle-safety predicate (`batch_kv = Skv_t % Skv_chunk_t == 0`, `batch_q = Sq_t %
Sq_chunk_t == 0`) so the multi-page reserve is always slot-aligned in the
`KV_DEPTH`/`OUT_DEPTH`-slot ring (never crosses the CB wrap); partial-chunk shapes keep
the byte-identical per-tile path. Predicates derive from existing CT args — no new
descriptor arg, no hardcoded block/tile literals (DRY). `reader_placement` is already
`row_wise=True` (confirmed optimal). **But device-ns did NOT improve** (baseline
11.06 ms → batched 11.01 ms, within noise). **Ablation (`/perf-measure` no-DM: stub
ALL reader NoC transfers, keep CB reserve/push/barrier)** measures **11.01 ms —
unchanged** → the reader's data movement is **entirely hidden behind compute** by the
existing `KV_DEPTH=2` double-buffer. **The flagged shape is compute-bound, not
data-movement-bound** (current FPU util ≈ 0.07 vs the 0.35 target): reads are off the
critical path, so a DM lever cannot move wall-time here. The batching is **kept** (a
correct, non-regressing removal of the flagged anti-pattern that will surface once
compute no longer dominates), not reverted. Golden **1181 passed / 1088 xfailed**
(0 fail, 0 xpass — no SUPPORTED change); unit **55/55** (incl. non-aligned batched +
partial-chunk per-tile fallback); precision-matrix **272/112**; new guard set
**8/8** (mask none/custom × small/medium × DRAM/L1). The win is gated on the
compute-side work → **R3a** below (which converges with R5).

> **Correction (R3b)**: the `batch=true` path was **parked at its per-tile default**,
> not left active — see R3b's Landed note. The `read_tiles`/`write_tiles` scaffolding
> stays as a live knob for R3a to re-enable + re-measure; the shipped runtime is
> byte-identical to R2's per-tile reader/writer.



### [x] Refinement 3b — Speed up the perf-flagged profile (data-movement) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 3 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 1180/1181). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Landed (R3b)**: root-caused the regression to R3's `batch=true` NoC-batching lever
and **parked it at its trivial (per-tile) default** — the shipped reader/writer are
now runtime **byte-identical to the gate-passing R2** state. Evidence chain:
- The full golden suite **passes clean locally** at R3's committed HEAD
  (`1181 passed / 1088 xfailed / 0 failed`, production mode), so the gate's 1-cell
  regression is **not deterministic** on this hardware.
- A `ttnn-static-analyzer` structural review of `batch=true` found **no hazard**:
  the batched reserve/read/barrier/push is byte-identical in L1 layout and
  producer/consumer counts to the per-tile path for every supported shape
  (full-slot, slot-aligned, non-straddling reserves; `get_tile_size ==
  buffer_page_size` for all supported dtypes; no deadlock cycle; balanced counts).
- The lever is **ablation-proven zero-win** (R3: the flagged shape is compute-bound,
  reads hidden behind `KV_DEPTH=2`), so parking it costs no measured perf.
The only plausible mechanism for the intermittent gate failure is a rare bursty-NoC
transient stall on silicon (n≤64 async reads before one barrier) or an infra flake;
neither reproduces here. Parking the knob (compile-time `batch_* = false`) removes
the bursty-read pattern **and** guarantees no regression by construction (== R2).
The `read_tiles<cb,batch>` / `write_tiles<cb,batch>` scaffolding is retained so R3a
flips a one-line predicate to re-enable + re-measure once compute-side work (R5)
exposes the reads on the critical path.

**Verification**: full golden suite **1181 passed / 1088 xfailed / 0 failed**
(ran to completion, no hang — bullets 1 & 3); acceptance + nonaligned + coarse-chunk
+ debug unit **58 passed**; precision baseline + matrix + perf/guard **285 passed /
112 skipped** (flagged-shape soft `pcc≥0.997` held, guard set `pcc≥0.99`) — bullet 2.
No SUPPORTED change (perf refinement).

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression. ✔
### [~] Refinement 3a — Close the perf win on the compute-bound flagged shape (re-measure DM after compute-side)

**Type**: perf

**Goal** (sharper follow-up from R3's ablation finding): R3 proved — by no-DM
ablation (stubbing every reader NoC transfer leaves device-ns unchanged at 11.0 ms) —
that the flagged `1×10×9472×128` shape is **compute-bound**, so R3's (correct, kept)
reader/writer batching is off the critical path and produced no wall-time win. The
exact next lever is **compute-side**: grow the QKᵀ/PV matmul output subblocks toward
the `fp32_dest_acc_en=False` DEST budget (8 bf16 tiles) and/or coarsen `Sq_chunk_t`/
`Skv_chunk_t` to amortize the ~10 sequential per-chunk helper phases (per-phase
reconfig/init/fill-drain) over more tiles — i.e. lift FPU util from ≈0.07 toward 0.35.
This is exactly **R5's** lever class (`matmul_output_subblock`, `compute_block_size`,
reconfig ablation); R3a is the ordered marker that R5 is the step that closes R3's
target. **After R5 lands, RE-ENABLE + re-measure the R3 DM batching** (parked at its
per-tile default in R3b — flip the reader/writer `batch_q`/`batch_kv` predicates back
to the divisor rules documented in the reader kernel, then re-run the full golden
suite under `--dev` to confirm no intermittent hang before measuring): if the faster
compute exposes the reads (reader becomes the critical path), the re-enabled batching
should now win — tune `KV_DEPTH` / the read-block size; if reads stay hidden, leave the
knob parked (the DM batching is complete-but-dormant, no measured win to bank).

**Verifier notes**: no SUPPORTED change (perf). Do R5 first (or fold R5 into this),
then confirm on the flagged shape that device-ns moves toward the 0.35 util goal with
the soft `pcc_threshold=0.997` holding and the golden suite green.

**Done when**: measured device-ns improves on the flagged shape via the compute-side
lever, and the R3 DM batching is confirmed (still hidden → complete; or exposed →
`KV_DEPTH`/read-block tuned); prior phases green.

**Landed (R3a, partial)**: the compute-side coarsen lever landed and **won** — the
`_fit_l1` block-factor target is raised 4→8 (single-source module constants
`SQ_CHUNK_TARGET`/`SKV_CHUNK_TARGET`; the shrink loop still L1-caps and `_chunk_size`
still per-axis-caps, so it stays "coarsest that fits"). On the flagged shape this
halves `n_kv_chunks` (74→37) and grows the QKᵀ matmul `out_subblock_w` 4→8 (full
`fp32_dest_acc_en=False` DEST budget). **Measured (Blackhole p150b, 110 cores; device
FW ns, warm median of 5, fresh cache): 9.666 ms → 9.006 ms = 1.073×** (FPU util
0.078→0.084). The **DM batching half is complete**: re-enabled the divisor predicates,
re-measured the flagged shape at 9.05 ms — **flat vs 9.01 ms** → the reads are STILL
hidden behind `KV_DEPTH=2` (the shape stays compute-bound even after the coarsen), so
per the refinement's "reads stay hidden → leave parked" branch the reader/writer knob
is **parked at its per-tile default** (byte-identical to gate-passing R2/R3b; the
`read_tiles`/`write_tiles` scaffolding stays a live tunable). **Regression found + fixed
in the same pass**: the coarser target made `_fit_l1` shrink some previously-clean
shapes (e.g. `Q1x8x4096x128` GQA, 128%4==0) onto a **partial** KV chunk (`_chunk_size`
returned 6), and a partial chunk pushes a *fraction* of the `cb_scores`/`cb_exp` ring —
so on shapes with **>1 work unit per core** (`total_work=128 > 110`) the ring pointer
carried across work units and the reduce's linear read window straddled the wrap →
catastrophic garbage (6 golden cells, pcc≈0/rms=inf). Root-caused by isolation
(discriminator = `partial_kv`, not batching, not L1 — batching-parked reproduced it) —
a **latent R1b bug** (its `rem|chunk` guard only covers the *within*-work-unit straddle;
the cross-work-unit carry was never exercised because R1b's prime tests ran ≤1 wu/core).
Fixed by making `_chunk_size` **prefer the largest exact divisor ≤ target** (no partial
→ ring realigns to slot 0 per work unit), keeping R1b's coarse+partial only as the
prime-tile-count>target fallback (reachable only at 1 wu/core). Golden **1181 passed /
1088 xfailed / 0 failed** (restored, no SUPPORTED change); unit **343 passed / 112
skipped**. **Why partial**: the flagged shape is nowhere near the aspirational
`expected_math_util=0.35` (0.084 measured) — the knob-turn levers are now **exhausted**
(`Sq_chunk_t` L1-capped at 8; `Skv_chunk_t` DEST/divisor-capped at 8), and the ~11×
residual gap is **structural**: the ~10 sequential per-chunk online-softmax vector
phases (exp/reduce/sub/mul over S² tiles) run serially with the QKᵀ/PV matmul (each
helper owns all 3 TRISCs), so the FPU idles during the softmax — a **scheme-change** to
overlap them, not a knob-turn. Filed as **R3c**.

### [x] Refinement 3c — Overlap the softmax vector phases with the matmul (lift FPU util past ~0.08)

**Type**: perf

**Goal** (sharper follow-up from R3a — the exact next lever): R3a exhausted the
compute-side **knob-turns** on the flagged `1×10×9472×128` shape (block factor coarsened
to the L1 ceiling `Sq_chunk_t=8` and the DEST/divisor ceiling `Skv_chunk_t=8`, QKᵀ
subblock at the full 8-tile budget) and banked +7% (9.67→9.01 ms), but FPU util is still
only ≈0.084 vs the 0.35 goal. Ablation (R3 + R3a) proves the shape is **compute-bound**
with the FPU **idle during the softmax**: per KV chunk the kernel runs ~10 phases
*sequentially* (QKᵀ matmul → mask → row-max reduce → running-max/α → exp → row-sum
reduce → O-rescale → PV matmul → O-accumulate), each a kernel_lib helper that owns all 3
TRISCs, so the two matmuls (the only FPU work) never overlap the vector-engine softmax
(exp/reduce/sub/mul over the S² score tiles — the dominant wall-time). The lever is a
**scheme-change**: pipeline the matmul of KV chunk *j+1* against the softmax of chunk *j*
(e.g. split the KV loop across compute passes / deepen `cb_scores`/`cb_exp` so the FPU
and SFPU run concurrently), so the FPU stops idling. This is distinct from R5's remaining
knob-turn (reconfig-ablation), which R3a judged low-value (coarsening halved the phase
count for only +7%, so per-phase fixed overhead is small — the cost is the vector math
itself, not reconfig).

**Verifier notes**: no SUPPORTED change (perf). This is the structural step R3a's
heading ("close the perf win") could not reach with knob-turns. Gate on `/perf-measure`
ablation (confirm the softmax phases are the idle-FPU cost) and the soft
`pcc_threshold=0.997`. Reference `tech_reports/FlashAttention/FlashAttention.md` (§3.2.2
overlap) and `ttnn/ttnn/operations/examples/master.md` `compute_fusion` /
`double_buffer` (compute-side depth). If the overlap exposes the reads on the critical
path, **re-enable the R3 DM batching** (the parked `read_tiles`/`write_tiles` knob) and
re-measure — it may finally win.

**Done when**: measured device-ns improves further on the flagged shape beyond R3a
(toward the 0.35 util goal), soft `pcc_threshold=0.997` holds, golden green, no
regression across the config-spanning guard set.

**Landed (R3c, partial)**: the named **overlap scheme was NOT implemented** — it is
architecturally infeasible with the helper library (FPU and SFPU are both driven by
the single MATH RISC + share DST, so consecutive helpers serialize on MATH; the design's
own CB rationale states cb_scores/cb_exp are depth-1 because "consecutive helpers cannot
pipeline"; `FlashAttention.md` §5 lists cross-unit matmul/softmax pipelining as
**unimplemented future work** needing FA-3 warp-specialization — the same raw-LLK
dual-DST-bank restructure that failed the gate in the two prior R3c attempts). Instead
achieved the heading's **titular measurable goal** (lift FPU util past ~0.08) with a
different, **winning, non-regressing** lever: the **fast/approximate SFPU exp**
(`Exp<Approx::Fast>`) for the dominant P=exp phase, **gated (compile-time
`fast_exp = !fp32_dest`) to the fp32_dest_acc_en=False throughput regime**. Evidence:
**DeviceZoneScopedN clock-invariant zone profiling** showed the exact exp is **54% of
per-chunk compute** (QKᵀ 12.6%, PV 16.3%, reduces 8.4%+8.4%); the fast exp is ~75%
cheaper. **Flagged `1×10×9472×128` (fp32_dest_acc_en=False): 9.006 → 5.796 ms = 1.55×,
FPU util 0.084 → 0.131** (past the ~0.08 goal; short of the aspirational 0.35). Soft
PCC≥0.997 held; **golden 1181/1181 (0 fail, no hang), zero regression**; guard set 8/8.
The unconditional fast exp first regressed 201 tight-tolerance cells (fp32 RMS 0.02,
bf16+fp32-DEST 0.05) → gated to fp32_dest_acc_en=False (loose 0.12; the max-precision
regime keeps exact exp, byte-identical). `[~]` because the named overlap scheme was not
built and the 0.35 aspiration has large headroom — filed as **Refinement 3d**.

### [~] Refinement 3d — Close the flagged-shape util gap after the fast-exp win (matmul + reduce, or FA-3 overlap)

**Type**: perf

**Goal** (sharper follow-up from R3c): after R3c's fast-exp lever the flagged
`1×10×9472×128` shape is at **FPU util 0.131 / 5.796 ms** (was 0.084 / 9.006 ms), still
short of the aspirational 0.35. R3c's DeviceZoneScopedN profiling shows the **new**
per-chunk cost split (fast-exp in effect): QKᵀ matmul 21%, row-max reduce 15%, exp 23%,
row-sum reduce 14%, PV matmul+accum 28% — so the two **matmuls (~49%)** and the two
**reduces (~29%)** now dominate; the exp is no longer the bottleneck. Concrete next levers,
in priority order:
1. **Fold the row-sum into the PV matmul (V-ones-column trick).** Append a ones column to
   V so `O_augmented[:, D] = rowsum(P)` falls out of the PV matmul on the FPU — this
   **eliminates the separate row-sum reduce pass (~14%)** and moves that work onto the
   already-running matmul engine (a genuine "lift FPU util" move). Touches reader (V
   augmentation), the PV matmul N (Dt→Dt+1), and normalize (read the extra column instead
   of cb_row_sum). Moderate restructure; validate PCC across the golden suite.
2. **Short-K matmul efficiency.** QKᵀ (K=Dt=4) and PV (K=Skv_chunk_t=8) are operand-load-
   bound (per `matmul_output_subblock`: short-K matmuls can't hide the per-tile operand
   load); subblocks are already maxed (R3a). Investigate `num_k_blocks`/in0-reuse or a
   larger effective K (e.g. process multiple KV chunks' PV in one matmul) to raise matmul
   efficiency.
3. **True FPU∥SFPU overlap (the original R3c scheme).** Only reachable via raw-LLK FA-3
   warp-specialization (async matmul on one DST bank while the SFPU softmax runs on the
   other) — abandons the helper library, T3-advanced, gate-risky (failed twice as the
   prior R3c approach). Deprioritized behind (1)/(2) until the safer levers are exhausted.
4. **Re-measure the parked R3 DM batching** at the faster (5.8 ms) compute level — R3a
   found reads still hidden at 9.0 ms; at 5.8 ms they may start to surface. Flip the
   `batch_q`/`batch_kv` predicates and re-run the full golden suite under `--dev` (confirm
   no intermittent hang) before measuring.

**Verifier notes**: no SUPPORTED change (perf). Reuse the fast-exp gating pattern
(fp32_dest_acc_en=False) if any lever trades precision for speed. Gate on `/perf-measure`
(clock-invariant DeviceZoneScopedN cycles — the box's AICLK drifts ~1.8× between fresh
invocations, so ns A/B across separate runs is invalid; use same-process back-to-back or
zone cycles) and the soft `pcc_threshold=0.997`.

**Done when**: measured device-ns improves further on the flagged shape beyond R3c's
5.796 ms (toward the 0.35 util goal), soft `pcc_threshold=0.997` holds, golden green, no
regression across the config-spanning guard set.

**Landed (R3d, partial)**: implemented lever 1 (the **V-ones-column trick**) in full and
proved by **clock-controlled measurement** that it is a **REGRESSION** on the flagged shape,
then reverted it and filed the correct next lever (**R3e**). Evidence chain:
- **Full implementation, gated to bf16 + fp32_dest_acc_en=False** (the flagged regime, reusing
  R3c's fast-exp gating so every other cell stays byte-identical): reader appends a ones
  tile-column to V (`fill_col_ones_tile`, mirroring production `generate_bcast_col_scalar`);
  the PV matmul N grows `Dt → Dt+1`; the O rescale/accumulate carry `l` in the extra column
  for free through the flash recurrence (no separate row-sum reduce); normalize strided-gathers
  `l` (raw `copy_tile`, the only step with no helper — confirmed) and divides. **Verified
  numerically correct** (all-ones → 1.0; random-V matches torch to maxdiff 0.002; flagged shape
  soft PCC≥0.997 held).
- **Measured (Blackhole p150b, 110 cores; device FW ns, warm median of 5, SAME-SESSION
  back-to-back A/B via an `SDPA_FOLD_ROWSUM` env toggle to defeat the AICLK drift):
  fold OFF 5.803 ms vs fold ON 6.701 ms = 0.866× — a 15% REGRESSION.** (Baseline reproduced
  R3c's 5.796 ms, so the clock was steady; not drift.)
- **Root cause (structural, not fixable):** the flagged shape has **tile-aligned D=128** (Dt=4),
  so the rowsum needs a **whole extra tile-column** on the PV matmul. PV is **short-K**
  (K=Skv_chunk_t=8, operand-load-bound), so adding one N tile-column costs a full K operand-load
  pass (+25% of PV) plus +25% on the O rescale/accumulate — **more** than the cheap 1-wide
  `reduce<SUM,REDUCE_ROW>` (the ~14% it replaced). The waste is intra-tile (31 of the ones
  tile's 32 columns are ×0), which the tile-granular FMA cannot avoid — no `last_in1_subblock_w_valid`
  or subblock knob fixes it. **Production SDPA deliberately avoids the V-ones trick for exactly
  this reason** (it keeps `l` in a separate CB, L1-accumulates the partial sum **during the exp
  pack**, and finishes with a single 1-wide ones-vector `matmul_reduce` after the loop).
- **The correct lever is production's L1-accumulate-during-exp**, which eliminates the row-sum
  reduce with **no added matmul work** — but it needs a **raw-LLK dual-pack** (pack exp→cb_exp
  AND `pack_tile<true>`+`llk_pack_reconfig_l1_acc` accumulate rowsum→cb_sum in the SAME DEST
  window). The kernel_lib chain is **single-terminal** (`OutputLifecycle::L1Accumulation` /
  `DestAccumulation` pack to one output), so it **cannot** co-pack exp and the rowsum — the fusion
  is not expressible with helpers. Filed as **R3e**. Levers 2/3/4 unchanged: lever 4 (DM batching)
  is ablation-proven hidden at 9.0 ms (R3a) and still hidden at 5.8 ms (util 0.13, reads ≪ compute
  per chunk); lever 3 (FA-3 overlap) is helper-infeasible (R3c). `[~]` because the priority lever
  was tried in full + measured as a dead end and the winning lever (R3e) is a raw-LLK scheme-change,
  not a knob-turn; no perf win banked, no SUPPORTED change, op byte-identical to R3c (green).

### [x] Refinement 3e — Eliminate the per-chunk row-sum reduce via L1-accumulate-during-exp (raw-LLK dual-pack)

**Type**: perf

**Goal** (sharper follow-up from R3d — the exact next lever): R3d proved the V-ones-column
trick loses on the flagged tile-aligned shape (adds a full tile-column to the short-K PV matmul,
+15%). The **correct** way to eliminate the ~14% per-chunk `reduce<SUM,REDUCE_ROW>` (still one of
the two dominant reduces at ~29%) is **production SDPA's approach**: fuse the partial row-sum into
the **exp pack** via packer **L1 accumulation** — while the exp results are in DEST, pack them to
`cb_exp` (for PV) AND `pack_tile<true>` + `llk_pack_reconfig_l1_acc(1)` accumulate each row's
tile-columns into a (rows×1) column of `cb_sum_chunk`, then finish the intra-tile reduction with a
single **1-wide ones-vector `matmul_reduce`** ONCE after the KV loop. This adds **no matmul N work**
(unlike V-ones) and re-uses the exp's already-loaded DEST (no cb_exp re-read, unlike the current
dedicated reduce), so it removes the reduce nearly for free.

**Verifier notes**: no SUPPORTED change (perf). This is a **raw-LLK helper substitution** of the
exp phase — the kernel_lib eltwise chain is single-terminal (`PackTile` packs to ONE output;
`OutputLifecycle::L1Accumulation`/`DestAccumulation` exist but cannot co-pack `cb_exp` AND the
rowsum in one DEST window), so the fused dual-pack must be hand-written in raw LLK. Document the
substitution + argument at the kernel-file head (helper limitation: no dual-terminal pack). Gate any
precision trade to fp32_dest_acc_en=False (R3c's pattern). This is the same regime that failed the
gate in R3c's overlap attempts, so land it carefully: build a deterministic debug test first
(all-ones → l = effective count), validate the full golden suite for no regression (the exp phase
is central), then measure clock-invariantly (DeviceZoneScopedN cycles or same-session A/B).
Reference production `compute_common.hpp:344-366` (`sub_exp_block_bcast_cols_inplace` do_reduce path)
and `:1042,1879` (`matmul_reduce`). If it exposes the reads, re-measure the parked R3 DM batching.

**Done when**: measured device-ns improves on the flagged shape beyond R3c's 5.796 ms via the
L1-accumulate-during-exp row-sum fusion, soft `pcc_threshold=0.997` holds, golden green, no
regression across the config-spanning guard set.

**Landed (R3e)**: implemented production SDPA's L1-accumulate-during-exp row-sum fusion as a
**raw-LLK dual-pack** (compile-gated `fuse_rowsum = !fp32_dest`, the fp32_dest_acc_en=False
throughput regime; the max-precision path keeps the exact per-chunk `reduce<SUM>`,
byte-identical → zero regression). New compute helper `fused_exp_dual_pack`: for each Q row it
subtracts the running max (bcast-col), fast-exps into DEST, then packs that ONE DEST window
**twice** — a normal `pack_tile<true>` to `cb_exp` (for PV) AND a `pack_reconfig_l1_acc(1)` +
`pack_tile<true>` L1-accumulation of the row's `skv` column tiles into a (rows×1, 32-col,
un-reduced) `cb_sum_chunk[i]`. The running sum `l` is then kept in that partial form across the
KV loop (per-chunk `alpha`-rescale bcast-col + `add`), and collapsed to the scalar denominator
**once** after the loop with a single `reduce<SUM,REDUCE_ROW>` (the FPU matmul-with-ones) — so
the per-chunk row-sum reduce is eliminated (exact: rowsum is linear, so it commutes with the
alpha rescale). `cb_sum_chunk` is promoted to `interm_format` (bf16 here) so the two pack
targets share a format (no per-pack `pack_reconfig_data_format`). **Why raw LLK**: the
kernel_lib eltwise chain is single-terminal — a chain using `OutputLifecycle::L1Accumulation`
static-asserts that *every* pack be L1-accumulating to *one* CB, so it cannot co-pack `cb_exp`
+ the L1-acc `cb_sum_chunk` from one DEST window; documented at the compute-kernel head.
**Measured (Blackhole p150b, 110 cores; same-session A/B via `SDPA_FUSE_ROWSUM` env toggle to
defeat AICLK drift; warm median of 5, fresh cache):** flagged `1×10×9472×128` (bf16,
fp32_dest_acc_en=False) **5.804 ms (reduce, reproduces R3c 5.796) → 5.440 ms (fused) = 1.067×
(6.27%, −364 µs)**; gap ≈29× the fused-run std → above noise; soft PCC≥0.997 held.
**Bug found + fixed during bring-up**: the first cut used a full `init_bcast` (complete packer
hw-configure) each KV chunk, which clobbered the boot-time `matmul_block_init` packer state
that the per-chunk matmul `InitMode::Short` relies on (Short doesn't fully re-issue packer
cfg); this drifted across chunks×work-units and regressed 4 golden cells
(`Q1x71x2048x64` MQA custom, >1 wu/core, pcc≈0.90). Isolated by probe to the multi-work-unit
regime; fixed by switching to the lightweight reconfig the eltwise_chain itself uses
(`reconfig_data_format` + `sub_bcast_cols_init_short` + `pack_reconfig_data_format`, no full
pack re-init). **Golden 1181 passed / 1088 xfailed / 0 failed** (no SUPPORTED change — perf);
unit + fused-debug + precision-matrix + perf/guard **351 passed / 112 skipped**; guard set
8/8. The parked R3 DM-batching knob stays parked: the reads were ablation-proven hidden at
5.8 ms (R3d), and a 6% compute speedup to 5.44 ms (FPU util ≈0.14, still compute-bound) does
not expose them.

### [x] Refinement 4 — Causal masking (mask_mode = causal)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`, generating the triangular
bias **on-device** (no mask tensor) driven by the `is_causal` compile-time flag.
Two parts, both reachable from phase-0's per-chunk KV loop:
1. **Block-skip** whole future KV chunks (`j·Skv_chunk_t > qc_end` ⇒ the chunk is
   fully masked) — roughly halves the KV work for causal self-attention.
2. **Per-element diagonal mask** on the single straddling KV chunk (the additive
   `−∞` upper-triangular bias), applied to the scaled scores *before* the row-max —
   reuse the existing `has_mask` additive-mask compute path (phase 3), fed by an
   on-device-generated mask instead of a streamed tensor.

Arm `EXCLUSIONS += {"mask_mode": "causal", "attention_kind": "cross"}` (causal
requires `S_q == S_kv`; the rectangular case corresponds to no real workload). The
`is_causal ∧ attn_mask` ValueError is already in `validate()`.

**Verifier notes**: standalone scheme-change (design Lamp 1) — no inventory skill
covers on-device causal masking, so this is verifier-authored. Land **after** R2
so the causal path is validated against the full dtype set already in SUPPORTED
(bf16 first is smallest, but R2's dependency pull-up forces R2 before this; the
causal path is dtype-agnostic and inherits R2's cells). Reuses R1's KV-edge masking
for the straddling-chunk diagonal when `S_q`/`S_kv` are non-aligned. Reference the
design's "Lamps → 1. Causal masking" section and
`tech_reports/FlashAttention/FlashAttention.md` (causal load-balancing).

**Done when**: `mask_mode=causal` self-attention golden cells pass; `causal + cross`
is xfail via the new EXCLUSION; block-skip verified to reduce device-ns on causal
vs an equivalent full-mask custom run; prior phases green.

**Landed (R4)**: `"causal"` added to `SUPPORTED["mask_mode"]`; the triangular −∞
bias is generated **on-device** (no mask tensor) from an `is_causal` compile-time
flag, **reusing R1's generated-mask CB (`cb_kv_mask`) + the phase-3b additive-mask
compute path** (causal is a generalization of R1's KV-padding mask). Two parts, both
in the phase-0 per-chunk KV loop: (1) **block-skip** — reader + compute cap the KV
loop at `n_kv_active = ceil((sq_off+sq_valid)/Skv_chunk_t)`, eliding K/V DRAM reads +
compute for fully-future chunks (identical predicate on both sides keeps the CB
counts matched); (2) **per-element diagonal mask** on straddling chunks — the reader
generates a score-block-shaped triangular mask (`fill_zeros_tile` below-diagonal,
`fill_neg_mask_tile` above, `fill_causal_diag_tile` on-diagonal), added before the
row-max. Causal **subsumes** the KV-padding mask (a padding key is always in the
future of every valid query), so `is_causal` disables R1's vertical-pad path.
`EXCLUSIONS += {mask_mode:causal, attention_kind:cross}` (causal needs `S_q==S_kv`);
the `is_causal ∧ attn_mask` ValueError was already present. Golden **1685 passed /
584 xfailed / 0 failed** (+504 causal cells vs R2's 1181; 0 xpass — SUPPORTED honest);
unit **370 passed / 112 skipped** (incl. 18 new causal + R1/R1b/R2/R3e regression).
Causal is **dtype-agnostic** — bf16/fp32/bf8b all pass (PCC ≥ 0.995). **bf8b bug found
+ fixed** (static-analyzer F1): the reader sized the `cb_kv_mask` fill word-count from
`get_tile_size(cb_q_in)` (input dtype) — for bf8b (1088 B) that under-filled the bf16
mask tile (2048 B), leaving stale L1 in the tail rows and leaking attention across
masked columns (bf8b causal PCC 0.29–0.40; bf16/fp32 unaffected). Fixed by sizing from
the mask CB (`get_tile_size(cb_kv_mask)/4 = 512`) at both the causal and R1 KV-pad
fill sites → bf8b 0.9999 (also fixes a latent fp32 over-fill OOB in R1's path). Causal
mask value is a large finite `-1e9` (the reference convention, NaN-safe), not true −∞.
**Block-skip device-ns verified**: same-process A/B on `(16,8,1024,64)` — causal
**1.577 ms** vs equivalent full-mask custom **2.310 ms = 1.465× (31.7% reduction)**,
both proven numerically equal to the torch causal reference. (On low-B·H shapes the
win dilutes to ~1.11× because naive contiguous work assignment puts near-full high-qc
work units on the critical-path core; the full ~2× needs **causal load-balancing** —
work-unit reassignment, a future perf refinement per FlashAttention.md.)

### [~] Refinement 5 — Speed up the perf-flagged profile (compute-side)

**Type**: perf

**Goal**: continue optimizing the flagged `1×10×9472×128` profile toward
`expected_math_util = 0.35`, this time with **compute-side** levers distinct from
R3's data-movement work. From `ttnn/ttnn/operations/examples/master.md`:
`matmul_output_subblock` (grow the Q·Kᵀ and P·V output subblocks toward the DEST
budget — with `fp32_dest_acc_en=False` the budget is the full 8 bf16 tiles, so the
`_matmul_subblocks` cap can widen), `compute_block_size` (co-tune the
`Sq_chunk_t`/`Skv_chunk_t` block factors — coarser amortizes per-phase
reconfig/init over more tiles, whole-tile-minimum floor) and its reconfig-ablation
second lever (drop the per-phase data-format reconfig where the format is constant
across the boundary). These are knob-tunes on the block surface `_fit_l1` already
exposes. No SUPPORTED change.

**Verifier notes**: generality is exhausted after R4, so the remaining phases are
all perf (still on the flagged shape — it is the sole flagged case, so R5 uses a
different lever class than R3). Co-tune block size against L1 (the `_fit_l1` budget)
and DEST budget together; the `fp32_dest_acc_en=False` regime (flagged) doubles the
DEST budget vs the phase-0 default, which is exactly what unlocks a wider matmul
subblock. Gate on measurement — if `/perf-roofline-dm` shows the shape is already
data-movement-saturated after R3, fold any residual compute win into R3's guard-set
re-measure instead of shipping a no-op phase.

**Done when**: measured device-ns improves further on the flagged shape beyond R3
(toward the 0.35 util goal), soft `pcc_threshold=0.997` holds, golden green, and no
regression across the config-spanning guard set.

**Landed (R5, partial)**: all three named compute-side levers **measured on device**;
none produced a win — the flagged shape's compute-bound cost is the serialized SFPU
softmax, not the block/subblock/reconfig fixed overhead this lever class touches.
- **`matmul_output_subblock` — `out_subblock_h` (the one genuinely-untried instance).**
  R3a grew `out_subblock_w` toward the DEST budget but left `out_subblock_h=1`, so the PV
  matmul (`N=Dt=4`, `dest_limit=8` in the flagged `fp32_dest_acc_en=False` regime) used
  only **half** the 8-tile bf16 DEST per subblock pass. Added **`decomp_h`** (mirrors R1b's
  `decomp_n`, single-source): grows `out_subblock_h` to `dest_limit/out_subblock_w` **only
  when the output is single-N-subblock** (`out_subblock_w == N` ⇒ SubblockMajor is
  tile-row-major for any height, which the downstream reduce/Col-bcast require) and `h | M`
  — self-gating (fp32-DEST `dest_limit=4` → `h=1`, byte-identical). For the flagged shape
  PV grows to `h=2`. **MEASURED same-session A/B** (steady AICLK, warm median of 5):
  `h=2` **5.461 ms** vs `h=1` **5.443 ms** = **FLAT / marginally negative**. Root cause
  (principled, not noise): filling the full 8-tile **half-sync** DEST section per subblock
  **defeats the intra-DEST math/pack pipeline** that `h=1` (4-tile subblocks, 4 tiles free)
  enables — `h=1` is optimal for half-sync DEST. Correct, general, self-gating lever
  **PARKED at its trivial default** (`grow_subblock_h=0` → `h=1`, byte-identical to R4);
  `decomp_h` retained as a **live knob** (`SDPA_PV_SB_H=1` re-enables same-session — worth
  re-measuring under a future full-sync-DEST or FPU∥SFPU-overlap scheme that would expose
  the pack overhead). `out_subblock_w` + block-size were exhausted by R3a (QKᵀ `w=8`=full,
  PV `w=4`=N-capped, blocks L1/divisor-capped at 8) — confirmed still maxed.
- **`compute_block_size`** — exhausted by R3a (`Sq_chunk_t` L1-capped at 8, `Skv_chunk_t`
  divisor-capped at 8 for `Skv_t=296=8·37`). No further headroom.
- **reconfig-ablation** — **directly measured** (throwaway ablation, reverted): dropping
  **both** matmul reconfigs (the highest-frequency reconfig sites, input+output) saves
  **9.3 µs = 0.17%**, below the ~6 µs noise floor. Unlike master.md's all-bf16 tiny-kernel
  example (1.19×), this kernel is **mixed-format** (only a few boundaries constant) with
  big Blackhole matmuls (reconfig ≪ FMA), so the reconfig class has ~zero headroom — not
  worth the mixed-format silent-corruption risk. Not shipped.

Golden **1685 passed / 584 xfailed / 0 failed** (identical to R4 — the shipped runtime is
byte-identical, knob parked); core unit + guard set 8/8 green; R5 A/B PCC≥0.997 both
variants. `[~]` because R5's Done-when requires a **measured device-ns win** and every
named knob-turn lever measured flat — the residual gap to `util=0.35` (0.14 measured) is
**structural** (the ~serialized SFPU softmax exp can't overlap the FPU matmul; FA-3
FPU∥SFPU warp-specialization is helper-infeasible per R3c). The knob-turn lever class is
now **exhausted with measurement**; the next lever is a matmul-efficiency scheme-change,
filed as **R5a**.

### [ ] Refinement 5a — Short-K PV matmul batching (raise effective K past the operand-load floor)

**Type**: perf

**Goal** (sharper follow-up from R5 — the exact next lever after the subblock knob-turns
are exhausted): R5 measured that the PV matmul's per-subblock/pack overhead is **not** the
bottleneck (growing `out_subblock_h` to fill DEST was flat), and R3d found the PV matmul is
**short-K, operand-load-bound** (`K = Skv_chunk_t = 8`: the per-K-block operand load can't
be hidden behind so few FMA steps). The untried lever is to **raise the effective K** of the
PV matmul: batch several consecutive KV chunks' `P·V` into one matmul with a larger
`in0_block_k`/`num_k_blocks` (e.g. accumulate `cb_exp`/`cb_v` for 2–4 chunks before the PV
matmul fires, K=16–32), so each operand load amortizes over more FMA. This is R3d's lever 2
(deprioritized then behind the V-ones trick, which R3d/R3e closed). It is a matmul-efficiency
**scheme-change** (restructures the KV loop's PV phase + deepens `cb_exp`/`cb_v` — L1 budget
must be re-fit), distinct from R5's block/subblock knob-turns.

**Verifier notes**: no SUPPORTED change (perf). Gate on `/perf-measure` clock-invariant
DeviceZoneScopedN cycles (or same-session A/B — the box's AICLK drifts ~1.8×) on the flagged
`1×10×9472×128` shape, soft `pcc_threshold=0.997`, golden green, no guard-set regression.
Watch the online-softmax recurrence: batching PV over multiple chunks means the running-max
`α` rescale must still be applied per-chunk **before** each chunk's PV is folded in — either
rescale `cb_exp` per chunk before the batched matmul, or keep the recurrence per-chunk and
only widen the matmul's K within a chunk (the safer first cut). The remaining structural
ceiling beyond this — true FPU∥SFPU overlap (softmax exp concurrent with the matmul) — is
**helper-infeasible** (R3c: single MATH RISC + shared DST serialize consecutive helpers;
needs raw-LLK FA-3 warp-specialization, which failed the gate twice) and is the last resort.

**Done when**: measured device-ns improves on the flagged shape beyond R5's 5.44 ms via the
wider-K PV matmul, soft `pcc_threshold=0.997` holds, golden green, no guard-set regression.
