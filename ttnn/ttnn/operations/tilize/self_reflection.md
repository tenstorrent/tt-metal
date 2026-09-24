# Self-Reflection: tilize (run_1)

## Summary

- **Blind final:** 1763 passed, 5 failed, 18 xfailed, 2604 skipped, 0 hangs (4390 tests). Golden `test_op`: 771 pass / 1 fail / 18 xfail. Regression: 10/10. Translated: 982 pass / 4 fail; its 294 skips all come from the reference tests' own arch gates (252 are "LLK for tiny tiles not fully supported on Wormhole").
- **`supported_fail = 5`, but none of them is a kernel numerics bug.** Three are one translation defect (INT32). One is an unseeded golden cell sitting on the PCC floor. One is a real op crash on a zero-volume input, which is the only op-level finding.
- **Most important finding (framework-level): the "blind" translated suite was not blind.** `test_translated.py` sits in `eval/golden_tests/tilize/`, which the implementer, verifier and perf prompts all tell agents to run. Agents read it from Refinement 1 onward and changed op code because of it. So treat the 982/986 translated pass rate as a development-set number, not a blind one.
- **Perf phase:** 2 rounds. Perf 1 graduated 0 of 5 ideas; Perf 2 graduated 4 of 4. Of the three recorded bypasses, all are `missing` capabilities. The ergonomics note is really `undocumented`: the tilize helper's default reconfig mode is wrong right after `compute_kernel_hw_startup`. One raw internal NoC call (from a pre-perf refinement) has no bypass record.
- Overall, the problems are mostly framework- and harness-level: suite placement, test seeding, one translation, and one harness tagger.

---

## 1. Golden coverage → `eval/golden_tests/tilize/`

**G1. Zero-volume input is an axis-blind, in-TARGET gap, and it crashes (high-severity singleton).**
- **What:** Neither golden nor any tagger captures "some dimension is 0". A zero-volume tensor projects onto ordinary SUPPORTED cells (rank 1 → `single_tile`; `[1,1,0,32]` → `small`, `tile_aligned`), so golden looks covered. The op raises `TypeError` instead of returning an empty tensor or refusing cleanly.
- **Evidence:** `test_translated.py::test_to_from_01d[0]` → `tilize_program_descriptor.py:635: TypeError: 'NoneType' object is not subscriptable`. `grid_2d_split` loops over `range(1, min(row_units, …)+1)` (`:628`). With R = 0 tile-rows or C = 0 tile-columns the loop never runs, `best` stays `None`, and `return best[1], …` crashes. The shape set has no zero dimension (`feature_spec.py` INPUTS / LOOSE_CASES), and op_requirements / op_design never mention zero volume.
- **Recommendation:**
  - Propose two `LOOSE_CASES` entries: `{"input_shape": [1, 1, 0, 32], "shard_api": "none", "in": _il(_DRAM), "out": _il(_DRAM)}` (the minimal rank-4 case) and `{"input_shape": [0], …, "pad_mode": "auto", "pad_value": 0}` (the rank-1 case that failed).
  - Consider promoting the facet to an axis value, e.g. `tag_tile_grid → "empty"` when R·C == 0, so the op must declare it SUPPORTED or excluded.
- **Confidence:** high for rank 1 (observed). Med for `[1,1,0,32]`: derived from the code, not run.

**G2. Golden inputs are unseeded, so lossy-output cells near the floor flip between phases.**
- **What:** `make_torch_input` (`helpers.py:69-87`) draws `torch.randn` / `randint` with no seed. `test_op` requests no seed fixture, and the root `reset_seeds` (`conftest.py:34`) is not autouse. Each run therefore draws different data. The cells that flap share one pattern: a lossy output dtype, plus either almost no valid datums or a nonzero fill sharing bfp exponent blocks with data.
  - **Cluster A (bfloat4_b, 1x1x50x50, nonzero fill):** the 1x1x50x50 bfloat4_b cells with a positive or negative fill fail in turn: fp32/auto/neg in R7, R8 and Perf 1; fp32/explicit/pos in R7, R8 and Perf 2; bf16/auto/neg in blind. PCC is 0.9789–0.9799 against a 0.98 floor.
    - Every zero-fill cell passes in every phase. So do W-aligned cells like `1x1x50x64` and `1x1x30x32`.
    - The discriminator is **bfloat4_b ∧ nonzero fill ∧ W % 16 ≠ 0**. At W = 50, row datums 48..63 form one 16-datum shared-exponent block holding 2 data datums and 14 fill datums, and the fill sets the exponent.
  - **Cluster B (bfloat8_b, rank 0):** the rank-0 bfloat8_b cell is a single datum, so PCC is degenerate. It fails as `PCC: 0.0` whenever that datum is not exactly representable (R7 fp32; Perf 2 bf16, `Max ATOL Delta: 0.0039`).
- **Evidence:** per-phase `test_results.json` (`golden_refinement_7/8`, `golden_perf_1/2`, `golden_blind_final`). The changelog already says so (R8): "fails on ~1 run in 3 because the golden input is unseeded randn". Also: "HEAD and R7 outputs are bit-identical on 60 seeded … cases".
- **Recommendation:**
  - Seed `make_torch_input` deterministically per cell, e.g. `torch.manual_seed(hash(nodeid) & 0xffffffff)`.
  - For lossy outputs with fewer than ~1024 valid datums (rank 0/1), replace PCC with an allclose using a format-derived atol (one bfp mantissa ulp at the block exponent).
  - For bfloat4_b ∧ nonzero fill ∧ W % 16 ≠ 0, either pin the floor through `extras`, or compare against a host-bfp4-quantized expected tensor, the way the fp8 path already quantizes on host (`helpers.py:81-86`).
- **Confidence:** high (seed); med (tolerance approach).

**G3. A translation defect: `test_to_layout_pad_value_dtype[INT32-*]` (3 cells).**
- **What:** The translation dropped the reference's INT32 branch. The test now feeds `torch.rand` bf16 in [0, 1) into an INT32 tensor. The device correctly returns 0 in the data region and the right fill in the pad region, but the expected tensor keeps the float values. The op is correct.
- **Evidence:** the reference has `if ttnn_dtype == ttnn.int32: torch_input_tensor = torch.randint(-100, 100, …)` (`tests/ttnn/unit_tests/base_functionality/test_to_layout.py:996-997`). The translation keeps only `if ttnn_dtype in (ttnn.uint32, ttnn.uint16)` (`eval/golden_tests/tilize/test_translated.py:1941`). The failure shows expected `0.6719` against device `0.`, with pad `5.`/`-2.` matching.
- **Recommendation:** restore the INT32 `randint` branch in `test_translated.py`. Also flag the translator: it should preserve per-dtype input-generation branches verbatim.
- **Confidence:** high.

**G4. The harness tagger mislabels legacy 2-D sharding as `nd` (88 cells).**
- **What:** `axes.py:_spec_of` treats `nd_shard_spec is not None` as ND. But an allocated legacy-sharded tensor also carries a derived `nd_shard_spec`. The op fixed this on its side (`tilize.py:313-331`, via `created_with_nd_shard_spec` read from `to_json()`); the harness did not.
- **Evidence:** `golden_blind_final/test_axes_consistency.json` has 88 entries, all `shard_api: declared legacy_2d, captured nd`. The verifier logged it as friction (`incremental-verifier_breadcrumbs.jsonl:0`).
- **Recommendation:** port the op's `_created_with_nd_shard_spec` check into `axes.py:_spec_of`. It is harmless now only because both `legacy_2d` and `nd` are SUPPORTED. Any run where their SUPPORTED status differs will mis-classify translated cells.
- **Confidence:** high.

## 2. SUPPORTED honesty → `tilize.py` `SUPPORTED` / `EXCLUSIONS`

`verifier_report.json` (blind): supported_pass 1753, **supported_fail 5**, xfail_expected 18, **xpass_drift 0**, supported_marked_xfail 0. The 5 failures form three clusters:

**S1. Zero-volume (1 cell, `rank=1, pad_mode=explicit, single_tile`): fix.**
- **What:** A real op bug in a declared SUPPORTED cell. It also escapes `validate()` as a `TypeError` rather than an `UnsupportedAxisValue` or `ExcludedCell`.
- **Evidence:** G1.
- **Recommendation:** fix. In `tilize()`, short-circuit when `logical_volume() == 0` and return the allocated (empty) `Layout::TILE` output with no dispatch; also guard `grid_2d_split` for R·C == 0. If the fix is deferred, raise `NotImplementedError` from `validate()` for zero volume so the gate is honest.
- **Confidence:** high.

**S2. INT32 pad-value translated cells (3 cells): neither fix nor demote.**
- **What:** A test defect (G3). The op's INT32 output is correct.
- **Recommendation:** no SUPPORTED change.
- **Confidence:** high.

**S3. bfloat4_b + nonzero fill on 1x1x50x50 (1 golden cell): neither fix nor demote; fix the test.**
- **What:** Format-bound and seed-dependent (G2). The sibling cells with the same axes pass in the same run (fp32/auto/neg, bf16/auto/pos, `1x2x1x50x50` neg).
- **Why not demote:** demoting `{output_dtype: bfloat4_b, pad_value: positive|negative, alignment: w/hw_non_aligned}` would only create xpass_drift.
- **Recommendation:** no SUPPORTED change; apply G2.
- **Confidence:** med.

**S4. EXCLUSIONS audit: two entries are oracle impossibilities, not "unsupported for now".**
- **What:** `{dtype: uint16|uint8, pad_value: negative}` is excluded because "no output can match the expected tensor". The fill is unrepresentable in an unsigned dtype, and `F.pad` rejects it for uint8. No refinement can ever promote these cells, so they are structurally impossible and belong in the feature spec's INVALID.
- **Evidence:** `tilize.py:274-282` ("oracle (unrepresentable)"). `ttnn-implementer.md:644` tells the implementer to "tell the user" rather than edit INVALID, so it parked them in EXCLUSIONS.
- **Recommendation:** propose moving both entries to `feature_spec.py` `INVALID`. Keep the tile_h = 16 block-float packer entries and the bfloat4_b rank 0/1 entries as EXCLUSIONS: those are genuinely fixable upstream.
- **Confidence:** med.

## 3. Helper / reference docs

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::tilize` (column slice of a wider tile-row) | capability | **missing** | The helper passes only `block_width_tiles` to `fast_tilize_block` / `tilize_block` (`tilize_helpers.inl:230-232`) and exposes no tile index. Under it, the compute API has `input_tile_index` but hard-wires stride = width (`tilize.h:387` `uint32_t full_dim = block;`). Raw sites: `tilize_compute.cpp:109` (`tilize_cols_fast`), `:164` (`tilize_cols_slow`). A real win: LOOSE 7 16997 → 15568 ns (−8.4 %). | Add a slice entry point: the row stride (the `full_dim` programmed at init) is a template argument; the sub-block width and column offset are runtime arguments. Sketch below. |
| `compute_kernel_lib::tilize` `ReconfigureRegisterDatatypeMode` default (the changelog's "ergonomics note") | ergonomics (not a bypass) | **undocumented**: the default is wrong for the documented consumer | The header's own prerequisite is "Call compute_kernel_hw_startup(input_cb, output_cb)" (`tilize_helpers.hpp:89-93`). The default `UnpackAndPackReconfigure` (`:26`, `:193-194`) then re-issues identical config (`.inl:150-167`), costing 7.9 % on a 4×4-tile shard and 15 % on a 2-tile shard (changelog Perf 2). By contrast, `eltwise_chain` "compile-time-elide[s]" unchanged reconfig (`ttnn-implementer.md:165`). | Doc line at `:21`: "NoReconfigure is correct when the previous init on these DFBs was `compute_kernel_hw_startup(input_dfb, output_dfb)` or a tilize of the same formats." Better: elide the reconfig at compile time the way `eltwise_chain` does. |
| dataflow API: this Tensix core's physical NoC0 coordinate | capability | **missing** (confidence med) | Firmware fills `my_x`/`my_y` from `MY_NOC_ENCODING` (`tt-1xx/risc_common.h:185-188`), which the kernel comment says is translated on WH (18..25). There is no dataflow-API accessor for the physical id. Raw `NOC_CMD_BUF_READ_REG(0,0,NOC_NODE_ID)` at `tilize_stick_reads.hpp:277` and `:362`. Cost: 65 cycles (co-read select) and 451 cycles (hop init). | Add `get_physical_noc_xy(noc)` to `dataflow_api.h`, or a host binding that returns physical coordinates. With the latter the host could emit a single list instead of up to 3 candidate lists (`tilize_stick_reads.hpp:263-270`). |
| dataflow API: one RISC-V issuing on the other NoC under `DM_DEDICATED_NOC` | capability | **missing** | Raw `noc_local_state_init(1 - noc_index)` on BRISC (`tilize_writer.cpp:225`). NCRISC then re-syncs its counters with `noc_local_state_init(noc_index)` behind a semaphore (`tilize_stick_reads.hpp:390-397`). Without the re-sync, `--dev` halts NCRISC at NKFW. `DM_DYNAMIC_NOC` costs +13..18 %. | A supported "borrow the other NoC" scope, e.g. `NocBorrow<1 - noc_index> b;`, that snapshots the counters on entry and hands NIU ownership back on exit. The cross-RISC-V re-sync should not be the caller's job. |
| dataflow API: non-blocking read-trid poll (**no bypass record**) | — (unrecorded) | **missing**; record gap | `tilize_stick_reads.hpp:699` and `:703` call the internal `ncrisc_noc_read_with_transaction_id_flushed`. `dataflow_api.h` offers only the blocking `noc_async_read_barrier_with_trid` (`:2424-2427`). Added in Refinement 3 (`c6d44fdc597`), before any bypass table existed. | Add `bool noc_async_read_done_with_trid(trid, noc)` to `dataflow_api.h`. Also propose that refinement changelogs carry the same bypass table as perf rounds, so pre-perf raw calls get recorded. |

No `too-hard` row with roughly equal helper and raw ns: every recorded bypass bought a measured win, and the one zero-gain cost (the reconfig default) is recorded as a note, not a bypass. The compute raw sites (`llk_unpack_fast_tilize_block`, `llk_pack_fast_tilize_block`, …) are all covered by row 1.

Sketch for row 1, derived from `tilize_sub_blocks::tilize_rows` (confidence: low; one call site):
```cpp
// init once with the row stride; the caller owns the per-row wait/pop, the helper owns each slice's reserve/push
template <uint32_t row_stride_tiles, uint32_t input_dfb, uint32_t output_dfb, Fp32Mode = Fp32Mode::Fast>
ALWI void tilize_slice(uint32_t width_tiles, uint32_t col_offset_tiles);   // + tilize_slice_init/uninit<…>()

// call site as it would then read:
tilize_slice_init<block_width, cb_in, cb_out>();
cb_wait_front(cb_in, block_width);
for (uint32_t p = 0; p < SB::n; ++p) { auto k = SB::at(j0, p); tilize_slice<block_width, cb_in, cb_out>(SB::width(k), SB::first(k)); }
cb_pop_front(cb_in, block_width);
```

**D1. The dataflow tilize helper gap (design phase, verified).**
- **What:** Phase 0 rejected `dataflow_kernel_lib::read_sticks_for_tilize`, and the stated reasons match the `.inl`.
  - The push quantum and L1 stride both derive from `row_bytes` (`tilize_helpers_dataflow.inl:92-93`), so a ragged last column block breaks the nominal-quantum ring-wrap.
  - The helper has no fill.
  - ROW granularity issues one barrier per stick (`:148-157`).
- **Evidence:** `op_design.md:377-382`.
- **Recommendation:** adopt op_design's own proposal: a `read_sticks_for_tilize` overload taking a nominal `block_width_tiles` separately from the valid `row_bytes`, plus an optional fill value and a stick-index functor.
- **Confidence:** high.

**D2. A block-float pack bug at tile [16, 32] is undocumented anywhere.**
- **What:** A bfloat8_b or bfloat4_b output at tile height 16 comes back with each mantissa row paired with the next row's exponent (PCC ≈ 0). Heights 32/8/4/2/1 are fine. `ttnn-op-constraints.md:13-23,71` covers block-float only against ROW_MAJOR and is silent on tiny tiles.
- **Evidence:** `ttnn-implementer_breadcrumbs.jsonl:15`; changelog R7 EXCLUSIONS.
- **Recommendation:**
  - Add a constraints row: "BFLOAT8_B/BFLOAT4_B output with Tile [16,32] mis-pairs exponents on WH B0 (tilize R7)."
  - File an LLK issue.
- **Confidence:** med (measured in one run).

**D3. Profiler and measurement caveats found the hard way, missing from the zone reference.**
- **What:** `device-zone-scope-attribution.md` has none of these:
  - A profiling session with more than ~5 zoned variant kernel dirs collides on the profiler's 16-bit zone hash and writes no report.
  - Splitting reader and writer into two kernel groups lowers DEVICE KERNEL DURATION by 5–13 % with no real change; judge on FW duration instead.
  - `--profile` gives no logs when a test fails, and ablation stubs always fail their value asserts.
  - `--profile` splits a quoted `-k` expression on spaces.
- **Evidence:** changelog Perf 1 "Measurement notes"; `ttnn-implementer_breadcrumbs.jsonl:2-3`.
- **Recommendation:** add a "pitfalls" block with those four lines to `device-zone-scope-attribution.md` and to the `/perf-measure` ablation section.
- **Confidence:** high.

**D4. The legacy-vs-ND discriminator is not exposed in Python.**
- **What:** `created_with_nd_shard_spec` is reachable only through `MemoryConfig.to_json()`. Both the op and the harness first got this wrong (G4).
- **Recommendation:** bind it as a property, or document the `to_json` workaround in `ttnn-python-utility-bindings.md`.
- **Confidence:** med.

## 4. Agent prompts → `.claude/agents/*.md`

**P1. The blind translated suite is exposed to every development phase.**
- **What:** The pipeline's own phase runs cover only `test_golden.py` and `test_regression.py` (3110 tests per phase). But the prompts tell agents to run the whole op directory, and that directory contains `test_translated.py`.
  - The implementer iterated against it: R1 "Translated sharded / ND / L1 slice: 191 passed / 1 failed"; R4 "`test_translated.py`: 723 passed".
  - It drove code changes: `_resolve_output_memory_config` was "needed by the translated `tilize_with_val_padding` sharded cases" (`changelog.md` R4). The spec-derivation gap "surfaced here because those translated cases were previously refused".
  - Perf 1 and Perf 2 report golden as "test_golden + test_regression + test_translated".
- **Evidence:**
  - `incremental-verifier.md:204` (`eval/eval_test_runner.sh eval/golden_tests/{op_name}/`).
  - `perf-coordinator.md:200` (`scripts/run_safe_pytest.sh eval/golden_tests/<op>/`).
  - `ttnn-implementer.md:320`.
- **Recommendation:**
  - Stage `test_translated.py` into the op dir only at blind time, or keep it outside `eval/golden_tests/<op>/` until then.
  - Change the three prompt lines to name `test_golden.py test_regression.py` explicitly.
- **Confidence:** high.

**P2. The perf coordinator saw a deterministic crash and carried it forward without routing it.**
- **What:** Perf 1 identified `test_to_from_01d[0]` and the INT32 cells as "4 fail deterministically". Perf 2 repeated this. Neither round filed anything for a correctness pass, because the perf charter freezes correctness. (Seeing them at all is a symptom of P1.)
- **Evidence:** `changelog.md` Perf 1 "Golden" and Perf 2 "Golden and unit nets".
- **Recommendation:** add to `perf-coordinator.md`: "A deterministic pre-existing failure you observe is recorded as a follow-up item with its nodeid and root-cause guess (op bug or test bug) in the changelog `### Handoff`; do not just re-report it."
- **Confidence:** med.

**P3. The planner has no degenerate-shape checklist item.**
- **What:** Zero volume, rank 0 and rank 1 are the classic edge shapes for a layout op. op_requirements and op_design handle rank 0/1 carefully but never state zero-volume behavior. That silence let the `grid_2d_split` `None` path ship.
- **Evidence:** G1. `incremental-planner.md` has no mention of empty or zero-size tensors.
- **Recommendation:** add a Step-level checklist line to `incremental-planner.md`: "State the behavior for a zero-volume input (any dim = 0): empty output with no dispatch, or an explicit refusal in `validate()`."
- **Confidence:** med.

**P4. The verifier prompt asks to commit a report that exceeds the repo's size hook.**
- **What:** The full `verifier_report.json` is 3.3 MB, against the 500 KB pre-commit limit.
- **Evidence:** `incremental-verifier_breadcrumbs.jsonl:1`.
- **Recommendation:** have `incremental-verifier.md` commit only the summary and category counts, or add `--summary-only` to `eval.verify_supported`.
- **Confidence:** high (low impact).
