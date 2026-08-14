# Self-Reflection: rms_norm

_Advisory only. Nothing here is auto-applied; every item is a proposal for a human to ratify._

## Summary

Final blind pass: **golden 5325 passed / 96 failed**, **translated 106/106**, **regression 15/15**;
`xpass_drift = 0`, `supported_fail = 96`. All 96 failures are **one** cluster — `failure_category=OOM`,
a CB-vs-L1-buffer clash on sharded inputs whose per-core resident row is large — and they are
**bit-for-bit the same 96 nodeids as `golden_refinement_2`**, i.e. they survived refinements 3–5 and
both perf rounds untouched. The single most important finding is therefore **process, not kernel**:
Refinement 2 correctly left them failing per the verifier's OOM rule but never filed the follow-up
refinement the rule implies, and nothing downstream is allowed to notice. Secondary theme is
framework-level: ~28 % of the live axes universe is parked in `INVALID` as "out of scope for now",
and at least one of those groups **demonstrably passes**. The op itself looks strong (13/13 perf cells
at or under `achievable_ns`, zero precision failures, zero hangs).

---

## 1. Golden coverage → `eval/golden_tests/rms_norm/feature_spec.py`

### 1.1 `{bfloat8_b × non-aligned}` is parked in `INVALID` but passes on device
**What.** `INVALID` carries `{dtype: bfloat8_b, alignment: w_non_aligned}` / `h_non_aligned` as
author-scoped ("out of scope for now"). The translated suite runs exactly that corner and it **passes**.
That region — 288 axes-cells, 13 % of the 2160 non-structurally-invalid cells — is invisible to every count.
**Evidence.** `golden_blind_final/test_results.json`:
`test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned-bfloat8_b]`
→ `passed`, pcc `0.99997` (also `[w200_c3_nonaligned-bfloat8_b]`, pcc `0.99997`); these are the two
`invalid_unexpected` rows in `verifier_report.json`. `feature_spec.py:100-101` — "bf8b
block-quantization + a masked/padded reduce is out of scope for now".
**Recommendation.** Propose deleting both entries from `INVALID`; if a cautious step is wanted first,
add two pinned loose cases and promote after they go green:
`{"inputs": ((1,1,32,72),), dtype=bfloat8_b, layout=TILE, gamma_dtype=bfloat8_b, gamma_layout=TILE, gamma_mode="gamma", fp32_dest_acc_en=False, memory_layout=INTERLEAVED}` and the same at `((1,1,72,64),)` (h-non-aligned).
**Confidence.** high for `w_non_aligned` (measured); med for `h_non_aligned` (untested by anything).

### 1.2 `ROW_MAJOR × sharded × TILE-gamma` is hidden in `INVALID`, and it is the run's riskiest region
**What.** Three `INVALID` entries couple the **activation's** `layout` to the **gamma tensor's**
`gamma_layout` (the cross-tensor coupling the verifier already flagged), removing 324 axes-cells.
The op refuses none of them — `SUPPORTED` admits every `layout × gamma_layout × memory_layout`
combination and `EXCLUSIONS` has one unrelated entry — so the op *claims* this region while the
harness skips it. It is also adjacent to the only failure class in the run: **every** ROW_MAJOR
HEIGHT_SHARDED cell at a wide row fails (see §2.1), because the RM path adds two staging CBs on top
of the resident shard.
**Evidence.** `feature_spec.py:94-96`; `verification_report.md:57` — "By the registry model that is
`EXCLUSIONS` … not `INVALID` (harness skip, invisible)"; `rms_norm.py:92-109` (no refusal).
**Recommendation.** Propose dropping the three entries and adding one small-W probe per scheme, e.g.
`{"inputs": ((1,1,64,256),), layout=ROW_MAJOR, gamma_layout=TILE, gamma_mode="gamma", gamma_dtype=bfloat16, dtype=bfloat16, fp32_dest_acc_en=False, memory_layout=HEIGHT_SHARDED}` (× WIDTH, × BLOCK);
if they fail, the honest home is one `EXCLUSIONS` cell in the op, not `INVALID`.
**Confidence.** high (the coupling and the invisibility are structural facts).

### 1.3 The failure discriminant is a facet no tagger carries, and no case pins its boundary
**What.** The 96 failures separate from passes on **per-core resident bytes**, not on any declared axis:
`dtype_bytes × W` (+ gamma bytes at the margin, + 2 staging buffers for ROW_MAJOR). Measured
differential over HEIGHT_SHARDED cells: every cell at ≤ 6 KB/row passes; 8 KB/row is mixed
(bf16 W=4096 → 90/90 pass, bf8b W=8192 → 12 fail / 24 pass); ≥ 16 KB/row is mostly fail
(fp32 W=4096 25 fail/20 pass, bf16 W=8192 42/12); ≥ 22 KB/row is 4/4 fail. Secondary separators, also
measured: `layout=ROW_MAJOR` fails at every wide row, `gamma_mode=no_gamma` **always** passes.
Golden does catch the region (that is where the 96 came from), so this is not a detection gap — but
there is no *minimal* pinned case, so the threshold is not a tracked number and a fix cannot be
regression-guarded cheaply.
**Evidence.** derived from `golden_blind_final/test_results.json` `observed_axes` + shape;
`changelog.md:170-176` states the mechanism ("a HEIGHT shard pins `slice_hidden_tiles = Wt` on *every*
core, so x + out + gamma alone are ≈ 3·W·2 B").
**Recommendation.** Propose one boundary pair as `LOOSE_CASES` (`group="l1_boundary"`), minimal rather
than model-sized: `((1,1,32,4096),)` HEIGHT_SHARDED at `dtype=float32` (fails today, 16 KB/row) and the
same shape at `bfloat16` (passes, 8 KB/row). Do **not** promote the facet to an `INPUT_TAGGERS` axis
unless a fix lands a genuinely different scheme — per `incremental-verifier.md:207` a tagger invented
to bucket a failure mode would delete the signal.
**Confidence.** med (the threshold is measured; whether a boundary pair is worth two cells is a judgement call).

---

## 2. SUPPORTED honesty → `rms_norm.py` `SUPPORTED` / `EXCLUSIONS`

### 2.1 `supported_fail = 96` — one cluster, open for five consecutive phases
**What.** Every one of the 96 is `failure_category=OOM` with the same message; 95 are
`memory_layout=HEIGHT_SHARDED`, 1 is `WIDTH_SHARDED` (`13x777x1023`, whose per-core shard is 1.33 MB
by itself). Shapes: `1x1x32x8192`/`1x32x8192`/`128x8192` (21 each), the W=4096 family (5 each), plus
6 loose cells at W ∈ {4064, 5119, 6144, 11008}. The op declares all four placements SUPPORTED with no
carve-out, so this is an over-claim on a resource boundary, not on a value.
**Evidence.** `verifier_report.json` summary `supported_fail: 96`; failure message
`"Statically allocated circular buffers in program … clash with L1 buffers … L1 buffer allocated at 524288 and static circular buffer region ends at 648064"`; identical nodeid set in
`golden_refinement_2/` … `golden_perf_2/` … `golden_blind_final/` (96 ∩ 96 = 96).
**Recommendation.** **Fix, do not demote** — the region is in `TARGET`, the mechanism is understood,
and the exit is already named (`op_requirements.md:131`, the design's lamped **TwoPassStreaming**:
sub-chunk the hidden axis, re-read x with `Accumulate::at`). Propose filing it as **`Refinement 2b`**
in `op_requirements.md` — today **no queue entry exists for it at all**, which is why five phases
passed it by. Done-when: every cell in the `OOM` category passes; no `EXCLUSIONS` edit.
**Confidence.** high.

### 2.2 No under-claim
`xpass_drift = 0`, `supported_marked_xfail = 0`, `xfail_wrong_mode = 0` — nothing to promote.
`infeasible_skipped = 21` are S1/S2 shard-geometry skips (harness × device), correctly uncharged.
**Confidence.** high.

### 2.3 Reporting note (framework, not the op)
The 96 cells are **one** defect amplified by the cartesian: `memory_layout` multiplies the resident
footprint corner across `dtype × gamma_dtype × gamma_layout × layout × fp32_dest_acc_en`. Any
dashboard/queue that ranks by `supported_fail` count will read this op as 96× worse than an op with
one distinct bug. Consider reporting `supported_fail` alongside a **distinct-cluster** count.
**Confidence.** med.

---

## 3. Helper / reference docs

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `ckl::reduce` — `ReduceWithinTile::Skip` | capability | **missing** (documented, compile-blocked) | `reduce_helpers_compute.inl:885-891` — the `static_assert(within_tile == Collapse, …)` sits at function scope *after* the `if constexpr (resolved_algorithm == AccumulateViaAdd) { …; return; }` at `:808`, so it is not in a discarded statement; `reduce_helpers_compute.hpp:158` documents Skip as valid with AccumulateViaAdd and `:168` names *this* use case ("summing per-core PARTIALS"). Hit **5×** independently: Phase-0 implementer breadcrumb (`reduce_helpers_compute.inl:886`), `verification_report.md:27`, Perf-1 I2 + I3, Perf-2 P1 + P2 | move the assert into the `else` branch and add a kernel_lib test instantiating `<AccumulateViaAdd, …, Skip>` so the path stays compiled |
| `ckl::reduce` — `post_reduce_op` scope hint | ergonomics | **missing** (highest-value ask of the round) | `reduce_helpers_compute.hpp:492` — "post_reduce_op callback receives dst_idx parameter" is the *only* thing it receives; measured 989.7 → 240.1 ns (4.12×) once the caller drops to raw sfpi (`rms_norm_finalize.hpp:106`, `rms_norm_compute.cpp:287`) | add a `PostReduceScope{Col0, Row0, Scalar}` hint that maps to `VectorMode` + parity stride (sketch below) |
| `rsqrt_tile`, `mul_unary_tile`, `add_unary_tile` | capability | **missing**, but cheaper than the row implies — the *convention already exists in-family* | `eltwise_unary/rsqrt.h:43-45` hardcodes `(APPROX, 8 /*ITERATIONS*/, …), idst, VectorMode::RC`; `binop_with_scalar.h:39-42,65-68` identical. Siblings already take the parameter: `recip.h:38`, `exp.h:72`, `rdiv.h:36`, `rpow.h:35`, `binary_max_min.h:126` all `VectorMode vector_mode = VectorMode::RC` | extend the existing defaulted `vector_mode` parameter to `rsqrt_tile` / `binop_with_scalar` family. Caveat from the run's own menu: `VectorMode::C` alone recovers 989.7→514.8 (1.9×), not the 4.12× — the parity stride still needs the reduce-side hint above |
| `ckl::eltwise_chain` / `convenience.hpp` (4 hits, one cluster) | ergonomics ×3 + capability | **undocumented** ×2, **too-hard** ×1 | (a) `chain.hpp:257-277` documents `TileOffset::Unset` as "no offset, zero overhead" and says nothing about `pack_tile<out_of_order_output=false>`'s internal counter — implementer breadcrumb: "SILENTLY DROPS its output when a previous chain already packed to the same CB … no error, no hang". (b) `chain.hpp:293-302` `DestAccumulation` docs are silent that the accumulator cannot survive `tile_regs_release` at `dst_full_sync_en=False` (Perf-2 row, "only discoverable by reading `llk_pack_common.h`"). (c) wrappers take only compile-time specs + `shape` (`convenience.hpp:86`) so they cannot carry a runtime tile base — **3rd independent hit**, and **34591 vs 34598 ns: exactly zero device time was bought** | (a)+(b) two doc lines: "for `(None,None)` output policies `TileOffset::Set` (base 0) is REQUIRED if another chain may have packed to this CB"; "a `DestAccumulation` window cannot span a `tile_regs_release` at `dst_full_sync_en=False`". (c) add a runtime `base` parameter to the convenience wrappers, or at minimum list "a runtime tile base" in `convenience.hpp:32`'s drop-to-`eltwise_chain` reasons |
| `ckl::ReduceInputMemoryLayout` | capability | **missing** — but the equal ns is *not* an API tax | `reduce_helpers_compute.hpp:260-268` carries `row_stride` only, so `addr(r,k) = r + k*own_rows` is inexpressible. The row's `26156 vs 26150` is equal because **idea P2 was itself a NULL** (`changelog.md:824`), not because the bypass paid for nothing — do not read it as the ergonomics signal | add an optional `col_stride` to `ReduceInputMemoryLayout`; priority low until an idea that *wins* needs it |
| `ckl::transpose` (absent) + `post_reduce_op` MATH re-init | capability | **missing** | no `transpose` symbol anywhere in `kernel_lib` (only a comment at `reduce_helpers_compute.hpp:364`); `reduce_init` is called once outside the per-output loop (`:491-495`), so a re-configuring post-op corrupts later output tiles | either `reduce_transposed<…>` fusing `reduce_tile + transpose_dest` in one window, or document that `post_reduce_op` must not reconfigure MATH |
| `mcast_pipe::ReceiverPipe` — wait for N senders | capability | **missing**, correctly low priority | `mcast_pipe.hpp:72` `enum class DataReadySignal { Flag, Counter }`; `:277-278` "Flag — a plain doorbell … Counter — returns the monotone round number". Neither expresses "N distinct senders, one round". The variant that needed it measured a NULL (57219 vs 28890) | leave as-is; record the ask only |
| kernel_lib dataflow — no gather/scatter helper | capability | **missing** (pre-existing, 3× re-confirmed) | no `gather`/`scatter` symbol in `mcast_pipe.hpp` / `dfb_helpers_dataflow.hpp` / `reduce_helpers_dataflow.hpp`; the op hand-rolls landing + paging + arrival counters (`rms_norm_reader.cpp:576`, `rms_norm_writer.cpp:139`) | a `GatherPipe` owning *(windows × pages)* of landing plus a **per-window** arrival barrier (P6's sharpened ask); the per-window part matters — Perf-2's finding #1 shows a single cumulative counter silently sags PCC once contributors run ahead |

**Record completeness.** Grepped all four shipped kernels for raw LLK / sfpi / NoC usage: raw appears at
exactly three sites — the sfpi finalize (`rms_norm_finalize.hpp:89-107`), the owner→root funnel
(`rms_norm_reader.cpp:576`) and the contributor gather (`rms_norm_writer.cpp:139`) — and **all three are
carried by a row** (or by Perf 1's explicit "same gather idiom the op already documents"). No unrecorded
bypass found; the record matches the code. Confidence high.

**API sketch** (proposal, derived from the raw sequence actually written in `rms_norm_finalize.hpp`):

```cpp
// what the caller wanted, instead of ~40 lines of sfpi + _llk_math_eltwise_unary_sfpu_params_:
enum class PostReduceScope { Whole, Col0, Row0, Scalar };   // -> VectorMode + dst_reg parity stride
reduce</*…as today…*/, PostReduceScope::Col0>(shape, layout, scaler,
    [](uint32_t dst_idx) { rsqrt_tile(dst_idx, /*already scoped by the helper*/); });
```
`confidence: low` on the exact spelling — extrapolated from one call site (two invocations: the owner
combine and the `s == 1` collapse).

### 3.1 A reference doc documents four bindings that do not exist — and it cost L1
**What.** `ttnn-python-utility-bindings.md` documents `ttnn.div_up`, `ttnn.round_up`,
`ttnn.find_max_divisor` (with examples and summary-table rows) and `device.l1_size_per_core()`; none is
bound on this build. The last one is load-bearing: the op fell back to a hardcoded conservative budget,
which Perf 2 then re-discovered as a **1.06× win left on the floor**, and which is the same budget the
96 OOM cells clash against.
**Evidence.** implementer breadcrumb, `ref: .claude/references/ttnn-python-utility-bindings.md:91-101,146-157` — "documented … but are NOT bound in ttnn on this build — hasattr is False for all three";
`verification_report.md:76` — "`l1_working_budget` is `1 MB − 96 KB = 928 KB`, because
`device.l1_size_per_core()` is not bound to Python"; `changelog.md:845` (P5) — "`device.l1_size_per_core`
is **not bound** to Python, so the 1 MB fallback has been running everywhere against a part that
reports 1,461,376 B per bank".
**Recommendation.** Mark all four C++-only in the reference with the Python fallback shown inline
(`ttnn.get_max_worker_l1_unreserved_size()` plus its kernel-config caveat), or bind them.
**Confidence.** high.

### 3.2 A toolchain trap with no home in any doc
**What.** Holding two live scalar constants across an unrolled `_calculate_sqrt_body_` is an sfpi
**ICE** ("cannot store sfpu register (register spill)"), not a diagnostic; the fix
(`#pragma GCC unroll 1`) is free but undiscoverable.
**Evidence.** `changelog.md:955` (Perf-2 bypass table), site `rms_norm_finalize.hpp:84`.
**Recommendation.** One line in a kernel-authoring reference (e.g. `.claude/references/ttnn-cb-memory-fundamentals.md`
or a new sfpi section): hand-written sfpi bodies that hold >1 live constant should carry
`#pragma GCC unroll 1`.
**Confidence.** med (single occurrence, but the symptom is a compiler crash).

---

## 4. Agent prompts → `.claude/agents/*.md`

### 4.1 The OOM rule leaves a cluster failing but nothing obliges anyone to schedule it
**What.** `incremental-verifier.md:207` says an `OOM` `supported_fail` "stay[s] failing and become[s] [a]
refinement entr[y]". Refinement 2 followed the first half exactly (`changelog.md:170-176`, "left failing
on purpose … rather than silenced with an `EXCLUSIONS` entry") and the second half never happened: there
is **no `Refinement 2b`** in `op_requirements.md`, the queue continued 3 → 4 → 5 (all perf, per the
one-perf-per-two-generality cadence at `incremental-verifier.md:361`), and both perf tournaments then
*re-confirmed* the same 8 resilience failures in their guard sets and shipped anyway
(`changelog.md:665, 933`). The rule's escape hatch ("EXCLUSIONS would delete the signal") is right, but
with no filing obligation the signal is preserved and ignored for five phases.
**Recommendation.** Two one-line prompt edits: (a) `incremental-verifier.md:207` — an `OOM`/`precision`
cell left failing **must** be filed as a named follow-up (`Refinement Nb`) in the same commit, and the
partial-tick ordering rule at `op_requirements.md:39` then guarantees it is picked next; (b)
`perf-coordinator.md` — before entering a tournament, report any open `supported_fail` cluster and
require an explicit human waiver to proceed (do not gate, just make it loud).
**Confidence.** high.

### 4.2 The prescribed `perf_experiments/` location is unusable in this repo
**What.** Three prompt surfaces prescribe `ttnn/ttnn/operations/<op>/perf_experiments/<idea_slug>/`;
a pytest file there trips the repo's no-global-torch-import hook and makes pytest import ttnn twice, so
both rounds relocated to `tests/ttnn/unit_tests/operations/<op>/perf_experiments/`.
**Evidence.** `perf-part-optimizer.md:74`, `perf-coordinator.md:131`,
`.claude/eval/prompts/perf_tournament_prompt.txt:17`; commit `f437f03196` — "Relocated under tests/ to
match Perf 1's convention: a test file inside the ttnn package tree trips the repo's
no-global-torch-import hook".
**Recommendation.** Change the prescribed path to `tests/ttnn/unit_tests/operations/<op>/perf_experiments/<idea_slug>/`
in those three files, and in the two reader-side surfaces (`self-reflection.md:65`,
`self_reflection_prompt.txt:26`) so a later pass looks in the right place.
**Confidence.** high.

### 4.3 The op prompt's `INPUT_TAGGERS` MUST would have broken the suite
**What.** `eval/prompts/rms_norm.txt` requires `tag_gamma_dtype` / `tag_gamma_layout` in
`INPUT_TAGGERS`. Complying would have collapsed both gamma axes to one value, deleting ~2/3 of the
gamma cartesian, because `cartesian()` removes tagger keys from the iterated finite axes and gamma
format is not derivable from the input shape tuple. The verifier had to *disobey* and justify it.
**Evidence.** verifier breadcrumb (`ref: eval/prompts/rms_norm.txt:95-105 vs eval/feature_matrix.py:106-113`,
"Following that instruction would BREAK the golden suite"); `verification_report.md:36` — "One advisory,
deliberately not 'fixed'".
**Recommendation.** Reword that prompt section, and add the general rule to the registry-model reference:
declare a tagger **only** for an axis computable from the `inputs` tuple, because a tagger key overrides
TARGET iteration.
**Confidence.** high.

### 4.4 Two smaller verifier-prompt mismatches, both cost a cycle
**What.** (a) The prompt says "The CLI emits a categorized report. Read every loud category" —
`eval/verify_supported.py` prints nothing to stdout and exits 1 on a *clean* run, so the verifier
hand-loaded a 25 MB JSON and initially read the exit code as a crash. (b) The prompt says the mandatory
perf target is marked by an `attention:` note on a `LOOSE_CASES` entry; this op's spec marks it with
`group="perf"` + `extras.achievable_ns` / `minimum_expected_speedup` instead.
**Evidence.** verifier breadcrumbs (`ref: eval/verify_supported.py (CLI main)`; `ref: verifier prompt,
Perf refinements: "an attention: note on the entry"`); `feature_spec.py` `_perf_case(...)` carries
`group="perf"` and no `attention:` marker anywhere.
**Recommendation.** (a) Have the CLI print the seven category counts + loud nodeids and exit 0 when the
loud categories are zero; until then say "report is JSON-only" in the prompt. (b) Teach the verifier
prompt the real convention: `group == "perf"` marks the mandatory targets and
`extras.minimum_expected_speedup` marks the decisive one. Related: the prompt's foreground
`eval_test_runner.sh` idiom exceeds the 10-minute tool timeout for this op (~20 min wall) — state the
nohup+poll idiom.
**Confidence.** high (both are directly quoted breadcrumbs).
