# Self-Reflection: tilize

## Summary

Blind final: **515 passed / 1 failed / 2 errors / 118 skipped of 636** — registry golden
`test_golden.py` **126/126** supported cells (90 INVALID skipped), consolidated main-branch
`test_golden_main_tests.py` 105 pass + 28 skip + **2 collection errors**, `test_regression.py` 9/9,
`test_translated.py` **275/276**. Verifier: `supported_fail=0`, `xpass_drift=0`.
The single translated failure is a **known-bad reference test** that upstream itself fixed as a test
bug (PR #49581) one day after the pinned commit — the op refuses it correctly, so there is no
evidence-backed op-level correctness gap in this run.
The substantive findings are therefore **framework-level**: (a) the golden matrix was already
saturated at Phase 0 and returned byte-identical results for all 10 phases, so it carried zero
refinement signal; (b) the bullet-3 responsible-cell denominator counts INVALID *skips*, capping
this op at 0.583 and making any expansion phase structurally unpassable; (c) the two hangs found in
this run were caught by the static analyzer and a `--dev` assert, not by any of the 636 cells,
because no cell in the universe reaches the narrow-shard geometry that triggers them.

---

## 1. Golden coverage → `eval/golden_tests/tilize/feature_spec.py`

Note up front: `feature_spec.py` declares **no `LOOSE_CASES`** and `test_golden.py` does not import
it, so every proposal below also needs the two-line wiring (`LOOSE_CASES` in the spec +
`golden_harness.parametrize_loose_cases`, as in `eval/golden_tests/softmax/test_golden.py:41`).
See §4 finding 6 — the planner prompt never mentions the mechanism.

**F1.1 — Axis-blind: no sharded cell is narrower than 64 elements, and that is exactly where a
reachable hang lives.** The only hang reachable from a plain public call in this run needed a
HEIGHT-sharded RM input with a **32-wide shard** (⇒ `chunk_row_bytes == 64`, one block/core, which
is precisely the `split_read_pays` gate). Every sharded scenario in `INPUTS` uses shard width 64 or
128, so no cell in the 636-cell universe can reach it. Shard width / row-bytes is not an axis, so
the axes-tuple looks "covered".
- **Evidence** `changelog.md:2187` — "**Invisible to a 316-test suite whose sharded cells are all 64
  wide.**"; `agent_logs/ttnn-implementer_breadcrumbs.jsonl` #77 `hang_detected` "BRISC
  `noc_semaphore_wait_min(sem_reserve)`"; fixed in commit `9baf7fa6a9`.
- **Recommendation** Add `LOOSE_CASES` entry pinning the narrowest sharded geometry (in-TARGET:
  legacy_2d / HEIGHT / l1_to_dram / rank 4 / multicore):
  `{"inputs": (({"input_shape": [1,1,128,32], "use_multicore": True, "shard_api": "legacy_2d", "in": _sh(_L1, _crs(((0,0),(3,0))), (32,32), _ROW, _HEIGHT), "out": _il(_DRAM)},),), "dtype": bf16, "output_dtype": bf16}`
  — and consider promoting the facet to an axis, e.g. `tag_shard_row_bytes → "le_64"|"gt_64"`.
- **Confidence** high.

**F1.2 — In-TARGET region never exercised: sharded tensors in DRAM.** All 15 distinct axis combos
the golden matrix reaches put every *sharded* spec in L1 (`INPUTS` uses `_sh(_L1, …)` exclusively);
`(shard_api=legacy_2d, out_scheme=WIDTH/HEIGHT_SHARDED, buffer=dram_to_*)` is declared in `TARGET`
but has zero cells. The one blind failure sits in exactly that region, and DRAM-sharded I/O is
otherwise only covered by untagged translated cells.
- **Evidence** blind combo census (from `verifier_report.json` axes): `legacy_2d/WIDTH_SHARDED` only
  as `l1_to_l1`; nodeid `test_translated.py::test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107`
  is the sole DRAM-sharded→L1-sharded cell and it fails.
- **Recommendation** Add a `LOOSE_CASES` entry mirroring upstream's fixed shape — rank-2 DRAM
  width-sharded in → L1 width-sharded out with the **output** grid mapped over
  `compute_with_storage_grid_size()`: `input_shape [32, 256]`, `in: _sh(_DRAM, _crs(((0,0),(3,0))), (32,64), _ROW, _WIDTH)`,
  `out: _sh(_L1, _crs(((0,0),(3,0))), (32,64), _ROW, _WIDTH)`.
- **Confidence** high.

**F1.3 — Axis-blind: a ROW_MAJOR-sharded input whose shard width does not divide W.** Such an input
stores `padded_shape[-1] = ceil(W/shard_W)*shard_W > W`; deriving the tile grid from it corrupted
every writer page index. The registry matrix cannot see this — every `INPUTS` shard width divides
its tensor width exactly — so the defect surfaced only in the untagged
`test_golden_main_tests.py` cells (`[3,160,160]` with shard `[2,64,64]`).
- **Evidence** breadcrumbs #3/#9 (`cost: high`) "26 reference-grader cells failed with whole-tensor
  mismatch (Max ATOL 0.996) plus OOM"; fix commit `b1a485489c` "derive tile grid from the OUTPUT
  padded shape"; breadcrumb #5 "79 passed/26 failed -> 105 passed/0 failed".
- **Recommendation** `LOOSE_CASES` entry with a non-dividing shard width, minimal:
  `input_shape [1,1,64,96]`, `shard_api "nd"`, `in/out: _sh(_L1, _crs(((0,0),(1,0))), (1,1,64,64), _ROW, None)`
  (RM padded W = 128 ≠ 96); plus a candidate axis `tag_in_row_padded → bool`.
- **Confidence** high.

**F1.4 — `double_buffer` is a phantom axis: the value the harness labels is never requested.**
`helpers.run_tilize` calls the op with `memory_config / dtype / use_multicore` only — it never reads
`scenario["use_double_buffer"]` nor forwards `use_double_buffer=`. So the 24 cells tagged
`double_buffer=False` actually run the public default (`None` → planner picks), and
`SUPPORTED["double_buffer"] = False` is verified only vacuously by golden.
- **Evidence** `eval/golden_tests/tilize/helpers.py:219-222` — `tilize(tt_input, memory_config=…,
  dtype=out_dtype, use_multicore=use_multicore)`; `test_golden.py:66` "`Its **_ swallows the tagger
  axes`"; `feature_spec.py:246` sets `"use_double_buffer": False`.
- **Recommendation** In `helpers.run_tilize`, forward
  `use_double_buffer=scenario.get("use_double_buffer")` (None when absent) so the tagged axis value
  is the value actually requested.
- **Confidence** high.

**F1.5 — Minor: `ulp_p99` is meaningless for block-float outputs and dominates the dashboard row.**
36 golden cells report `ulp_p99 ≈ 9.7e8` while `pcc ≥ 0.99997`, `max_abs_diff = 0.047`: the metric
compares an fp32 bf8b readback against a bf16 expected at **fp32** bit granularity, so
quantize-to-zero elements produce ~1e9 ULP.
- **Evidence** blind `test_results.json` cell `…output_dtype=BFLOAT8_B…`: `"pcc": 0.99996839,
  "ulp_p99": 984350720.0`; `eval/metrics.py:83-92` picks granularity from `readback_dtype`.
- **Recommendation** Record `ulp_p99=None` (or measure at the coarser of the two dtypes) when the
  tested output dtype is block-float.
- **Confidence** med.

---

## 2. SUPPORTED honesty → `ttnn/ttnn/operations/tilize/tilize.py`

Verifier on the blind dir: `supported_pass=126`, **`supported_fail=0`**, `supported_marked_xfail=0`,
`xpass_drift=0`, `invalid_skipped=90`, `no_axes_found=420`. Among *measured* cells the declaration
is honest — no over-claim to demote, no under-claim to promote. The findings are about declarations
nothing in the universe can measure.

**F2.1 — Declared-but-unexercised axis values.** `SUPPORTED["rank"] = [2,3,4,5,6]` and
`SUPPORTED["dtype"]` ⊃ {uint16, int32} extend beyond `TARGET`; of these, uint16/int32 *are* measured
(`test_regression.py` 9/9), but **rank 5 and 6 are exercised by zero of the 636 blind cells** — the
only evidence is the op's own acceptance test, which the blind pass does not run.
- **Evidence** `tilize.py:104-113` — "declared so a rank-5/6 caller is not refused by a contract the
  kernel actually satisfies"; blind axes census: `rank ∈ {2,3,4}` only.
- **Recommendation** Either mirror the uint16/int32 pattern and add a rank-5 scenario to
  `test_regression.py` (outside TARGET, so *not* a `TARGET`/`INPUTS` widening), or drop 5/6 from
  `SUPPORTED` until an eval-visible cell covers them. Framework side: have
  `eval/verify_supported.py` emit a `declared_unexercised` category — today an over-claim on a value
  **no cell takes** is structurally invisible (and 420/636 blind cells are `no_axes_found`, so
  honesty is measured on 20 % of the suite).
- **Confidence** high.

**F2.2 — `use_multicore=False` × sharded *input* is claimed and measured nowhere.** `EXCLUSIONS = []`
deliberately admits it, justified by the reference suite; but the reference suite skips exactly that
cell, golden has no single-core sharded cell, and `test_translated.py` contains no
`use_multicore=False` at all. Single-core with a sharded *output* (interleaved input) is covered by
main tests, so the claim is half-verified.
- **Evidence** `tilize.py:156-162` — "the reference suite exercises exactly that cell" vs
  `test_golden_main_tests.py:198` `pytest.skip("Singlecore is not supported for sharded input")`
  (12 skips in the blind run); golden combo census has `use_multicore=False` only on
  `none/interleaved/dram_to_dram/rank4`.
- **Recommendation** Add one `LOOSE_CASES` cell (`use_multicore: False`, HEIGHT-sharded L1→L1) to
  substantiate the claim, or demote with `EXCLUSIONS = [{"use_multicore": False, "shard_api": "legacy_2d"}, {…"nd"}]`.
- **Confidence** high.

**F2.3 — `double_buffer=False` is nominally supported and nominally covered, but not actually
requested** — see F1.4. Until `helpers.py` forwards the flag, treat `supported_pass` on those 24
cells as evidence about the planner default, not about depth-1.
- **Confidence** high.

---

## 3. Helper / reference docs

**F3.1 — Absence: nothing documents that a ROW_MAJOR *sharded* tensor's `padded_shape[-1]` is a
shard-width-rounded stride, not the logical width.** The references teach
`padded_shape[-1] * element_size()` as *the* RM stick width with no sharding caveat, and the op
design's geometry table did not say which tensor's padded shape to use — the highest-cost defect of
the run (26 failing cells + OOM).
- **Evidence** `.claude/references/ttnn-python-utility-bindings.md:62` "`page_size =
  tensor.padded_shape[-1] * tensor.element_size()`" and `:276` "When you need the raw stick width for
  RM-specific math"; breadcrumb #9 (`surface: design`, `cost: high`) on `op_design.md:84-99`.
- **Recommendation** Add one line to `ttnn-python-utility-bindings.md` §RM math and to
  `.claude/references/op-design-template.md`'s geometry table: "for a sharded ROW_MAJOR tensor
  `padded_shape[-1]` is rounded up to a whole shard width — it is a source *stride*; take the tile
  grid from the OUTPUT tensor's padded shape."
- **Confidence** high.

**F3.2 — Absence: the CB doc teaches a multi-page `cb_push_back` without the FIFO-straddle rule.**
`ttnn-cb-memory-fundamentals.md:77` presents `cb_push_back(cb, ntiles_per_block)` as the correct
pattern but never says a single push may not straddle the FIFO end (only the exact-hit wrap is
legal, `dataflow_api.h:213-222`). The run hit exactly that: silent pointer overrun in the default
build, an ebreak under `--dev`.
- **Evidence** `changelog.md:2193-2198` — "a single push may not **straddle** the end of the FIFO …
  the **default build silently ran the write pointer past the limit**"; breadcrumb #78
  (`hang_detected`, reader stuck at `tilize_reader.cpp:529`).
- **Recommendation** Add to that section: "A single `cb_push_back(cb, N)` must not straddle the FIFO
  end — only an exact-hit wrap is handled; issue one push per contiguous window."
- **Confidence** high.

**F3.3 — Misleading default: `Fp32Mode` steers a standalone layout op into a silent precision
regression.** The enum doc's three "Lossless is correct ONLY when…" conditions all presuppose a
same-kernel FPU/SFPU consumer; the case where the tiled output *is* the op's user-visible result
(bit-identity oracle) is absent, and for fp32→fp32 `Fast` breaks bit-exactness.
- **Evidence** `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:49` "You almost never want Lossless,
  even in \"max-precision\" kernels."; verifier breadcrumb #0 (`surface: helper`, `cost: med`) — "I
  had to re-derive the exception from the oracle."
- **Recommendation** Add a 4th bullet: "Exception — when the tiled output IS the op result (a
  standalone layout conversion with a value-preserving contract), Lossless is required for fp32
  input."
- **Confidence** high.

**F3.4 — The shared kernel helper did not compile at HEAD.** `has_unpack_to_dest_fp32` was defined
twice byte-identically, so *every* kernel including `tilize_helpers.hpp` failed to JIT-compile; the
run had to delete the duplicate before Phase 0 could produce a single green cell.
- **Evidence** commit `5c2ce9f817` — "Fixed blocking compile bug in
  `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl`: `has_unpack_to_dest_fp32` was defined twice";
  breadcrumb #12.
- **Recommendation** Upstream the deletion and add a compile smoke test that includes every
  `kernel_lib` header in one TU (a CI run of the existing `toy_tilize_untilize` op would also have
  caught it).
- **Confidence** high.

**F3.5 — Two smaller doc-accuracy items (both cost-limited only because `op_design.md` pre-warned):**
(a) `.claude/references/ttnn-python-utility-bindings.md:78,91,146` documents `ttnn.round_up`,
`ttnn.div_up`, `ttnn.find_max_divisor` — none exist on this build (`hasattr == False`, probed), and
`.claude/references/generic_op_template/template_op_program_descriptor.py:44-46` repeats two of them;
(b) `.claude/skills/perf-measure/SKILL.md` (Layer A1) documents only the out-of-process profiler
route, so the in-process `ttnn.ReadDeviceProfiler` path returns an empty dict unless three env vars
are set before device open — "Cost two failed bench runs and a probe" (breadcrumbs #11, #10).
Also undocumented and worth one line somewhere: `noc_async_read_one_packet_with_state` **hangs every
core on a watcher build** (`op_requirements.md:160-162`, commit `dc17e21d6d`) — a `--dev`-only trap
with no doc anywhere.
- **Recommendation** Mark the three bindings NOT BOUND with inline plain-Python equivalents; add an
  A1b "in-process measurement recipe" subsection; add the watcher caveat to the dataflow/CB
  debugging reference.
- **Confidence** high (a, c) / med (b).

---

## 4. Agent prompts → `.claude/agents/*.md` (+ the refinement harness)

**F4.1 — The golden matrix was saturated at Phase 0, so 10 phases produced identical numbers and
the golden gate carried no signal.** Every phase from `golden_phase0` to `golden_refinement_5`
reports `PASSED=240 FAILED=0 ERRORS=2 SKIPPED=118` with a byte-identical verifier summary, and the
Phase-0 registry snapshot is `diff`-identical to the blind one. `feature_spec.py` had been authored
on the opposite assumption ("Refused … until Refinement 6 wires the single-buffered CB path",
`feature_spec.py:90`; "Refused until the crossover refinement lands (xfail), then passes",
`:232`) — none of those xfail states ever existed (`xfail_expected=0` in every phase).
- **Evidence** the 10 `golden_results.txt` files; `diff` of `registry_snapshot.json`
  (phase0 vs blind) = identical; verifier breadcrumb #1 — "The queue-building guidance assumes
  SUPPORTED is a proper subset of TARGET. This op arrived with SUPPORTED == TARGET on every axis …
  every grouping rule … was inapplicable."
- **Recommendation** In `incremental-verifier.md`, add the perf-only playbook the breadcrumb asks
  for (and add `/perf-measure` + `/perf-ceiling-dm` to the skill-inventory table so "attach a
  pointer if it matches the inventory" stops contradicting "must carry a `/perf-measure` pointer").
  In `incremental-planner.md`, stop encoding a *predicted refinement order* in `feature_spec.py`
  comments — the spec is the universe, not a schedule. Consider having the harness detect
  `SUPPORTED == TARGET` at Phase 0 and re-point the phase gate at translated + bench instead of
  re-running a saturated matrix 10 times.
- **Confidence** high.

**F4.2 — The bullet-3 responsible-cell denominator counts INVALID *skips*, making the expansion
threshold unreachable for this op.** `eval/run_refinements.py:823` increments `responsible_total`
for every `is_supported` non-xfail cell — including the 90 cells the harness itself skipped as
INVALID — so the ceiling is 126/216 = **0.583**: above `GOLDEN_MAJORITY_FIX=0.50`, permanently below
`GOLDEN_MAJORITY_EXPANSION=0.75`. Any genuine cartesian-expansion refinement on tilize would have
failed bullet 3 no matter how good the implementation. It cost a whole extra debug phase.
- **Evidence** recomputed with the harness's own `feature_matrix.is_supported` on
  `golden_phase0`: `responsible 126/216 = 0.583, nonpassing statuses {'skipped': 90}`;
  `refinement_1_debug_output.json` — "The 90 INVALID-skipped cells sit in the harness's denominator
  … 0.583 clears 0.50, fails 0.75"; commits `a88ebcfb43`, `401227f367`.
- **Recommendation** Exclude harness-skipped INVALID cells from `responsible_total` in
  `run_refinements.py` (they never reach the op). Separately, `_next_sub_refinement_id` reused ID
  `1b` because it scans the pre-edit phase list, producing two headings with one ID that
  `_set_phase_checkbox` resolves first-match — worth a fix while in there.
- **Confidence** high.

**F4.3 — The implementer ran the whole blind translated suite, against its own prompt rule, so the
translated column is not an independent measurement.** `ttnn-implementer.md:163` says: "Run
`test_translated.py` only if your refinement body pins a specific translated cell nodeid; then run
exactly that cell, not the whole file." The breadcrumbs record full-file runs in at least six phases
("275 passed, 1 failed"), i.e. the exact blind result was known from 11:11 on day one.
- **Evidence** breadcrumbs #8, #79, #83, #90, #99, #107 — e.g. #8 `test: eval/golden_tests/tilize/test_translated.py`,
  "275 passed, 1 failed (… reference-test portability bug)".
- **Recommendation** Either enforce it mechanically (withhold `test_translated.py` from the clone
  until the blind pass) or legalize a **one-shot hazard scan** explicitly in the prompt and record
  it in the run metadata so the dashboard does not present the translated column as blind. Note the
  scan was *productive* here — it is how the CQ-wedging L1 shard-grid hazard was found — so
  legalize-and-label may beat prohibit.
- **Confidence** high.

**F4.4 — The blind suite's only failure is a reference-test bug that upstream fixed independently;
the corpus pin has no refresh path.** The translated copy reuses one `ShardSpec` (grid sized from
`dram_grid_size().x` = 12) for both a DRAM input and an **L1** output on an 8×8 compute grid.
Upstream fixed exactly this as a *test* bug — and the golden harness already has the guard the
translated corpus lacks (`helpers.py:60-77` `_check_grid_bounds` → `pytest.skip`).
- **Evidence** nodeid `test_translated.py::test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107`
  "output L1 shard grid ends at (11,0) but the device compute grid is (8,8)"; upstream commit
  `38eb908f94` (PR #49581) — "coordinates such as `(8, 0)` that are valid for DRAM banking but
  invalid as TENSIX worker coordinates"; pinned commit `25d5bac9` predates it;
  `op_requirements.md:179` — "the reference test needs a device-portability gate (it is unmodifiable
  from here)".
- **Recommendation** Apply the upstream fix (separate output grid via
  `ttnn.num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size(), True)`) or reuse
  `helpers._check_grid_bounds` in `test_translated.py`; and give the translation step a rule:
  device-portability of a derived shard grid must be checked at translation time, since a
  commit-pinned corpus cannot inherit upstream test fixes.
- **Confidence** high.

**F4.5 — `PROPERTIES` is load-bearing but absent from the verifier's Registry Conformance
checklist.** The checklist enumerates four declarations and instructs deleting extra
registry-looking symbols; `PROPERTIES` (consumed by `eval/dump_op_metadata.py` /
`eval/features_table.py`) is legitimate, and the verifier had to grep `eval/` to avoid deleting it.
- **Evidence** verifier breadcrumb #2 (`surface: prompt`) — "the \"delete non-template symbols\"
  instinct points at a load-bearing declaration"; `tilize.py:171` declares it and
  `op_metadata.json` consumes it.
- **Recommendation** Mention `PROPERTIES` in `incremental-verifier.md` §Registry Conformance as an
  optional fifth declaration whose `source: "verified"` tags should be audited, not deleted.
- **Confidence** high.

**F4.6 — `LOOSE_CASES` is never mentioned in the planner prompt, so the mechanism is missing from
this op's suite.** `eval/golden_harness.py:87 parametrize_loose_cases` exists and softmax / matmul /
rms_norm specs declare `LOOSE_CASES`, but `incremental-planner.md:180-215`'s `feature_spec.py`
template documents only TARGET / INPUTS / INVALID — and tilize's spec has no `LOOSE_CASES` at all,
which is why every §1 proposal needs harness wiring first.
- **Evidence** `grep -rl LOOSE_CASES .claude/agents/` matches only `self-reflection.md`;
  `eval/golden_tests/tilize/feature_spec.py` has no such symbol.
- **Recommendation** Add `LOOSE_CASES = []` (with a one-line "pin a concrete boundary shape without
  multiplying the cartesian" note) to the planner's feature_spec template and to the generated
  `test_golden.py` skeleton.
- **Confidence** high.
