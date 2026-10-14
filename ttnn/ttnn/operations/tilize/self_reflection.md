# Self-Reflection: tilize

## Summary
- Blind pass: golden `test_op` 908 pass / 56 fail / 46 xfail (2090 INVALID-skipped). Translated: 1201 pass / 8 fail. Regression: 10/10. Verifier: `supported_fail` 56, `xfail_wrong_mode` 8, `xpass_drift` 0.
- Most important finding: **all 64 golden failures are fp8_e4m3 inputs the harness cannot build** (a sharded fp8 input, or a TILE-layout fp8 input). The op never runs on these cells. The verifier diagnosed this in Phase 0 and asked for a feature_spec fix. Nobody applied it, and from Refinement 5 onward the cells sat in SUPPORTED as `supported_fail` for 6 phases.
- Of the 8 translated failures, 7 are translation bugs in the test file. 1 is a real op crash: a zero-volume input hits a `ZeroDivisionError` in the program descriptor.
- The problems are mostly at framework / test-universe level (feature_spec INVALID, harness axis capture, no category for failures in harness input creation, a misleading catalog entry). Only one is an op-level bug.
- Perf phase ran (2 rounds, focus −7 % total, up to −45 % on short_wide shapes). It recorded 3 helper bypasses. All 3 hold up as `missing`, and two smaller gaps are missing from the record.

## 1. Golden coverage → `eval/golden_tests/tilize/feature_spec.py`

**1a. fp8_e4m3 × (TILE input | sharded input) are unbuildable but not in INVALID** — confidence: high
- **What:** The failures cluster on `dtype=FP8_E4M3` ∧ (`in_tile_height≠none` OR input-side sharded, i.e. `buffer∈{l1_to_l1,l1_to_dram}` ∧ `shard_api≠none`). fp8 cells that are interleaved, or sharded only on the output side, all pass: 145 dram_to_dram, 16 dram_to_l1 with sharded output, 12 L1-interleaved. So the cluster is a property of the input combination, not of fp8 itself.
- **Evidence:** retile cells fail with `tensor_spec.cpp:144: Only ROW_MAJOR layout is supported for FP8_E4M3 dtype!`. Sharded-input cells fail with `pytensor.cpp:167: parse_py_tensor: ndarray_import failed.` Both errors come from `from_torch`, before the op is called. The spec itself says fp8 is ROW_MAJOR-only (`feature_spec.py:98-99`), but `INVALID` (`:253-256`) only prunes the fp8→int output crosses. `verification_report.md:38-39` already lists both gaps.
- **Recommendation:** Propose adding `{"dtype": ttnn.fp8_e4m3, "in_tile_height": h}` for h∈{32,16,8,4,2,1} to `INVALID`, since this is structural on every arch. Also propose adding `{"dtype": ttnn.fp8_e4m3, "buffer": b, "shard_api": s}` for b∈{l1_to_l1,l1_to_dram}, s∈{legacy_2d,nd} to `INVALID_FOR_ARCH["blackhole"]`, unless the harness can build a sharded fp8 input another way (e.g. build interleaved, then `to_memory_config`). If it can, prefer fixing `helpers.py` so these become real tests.

**1b. Zero-volume input is an axis-blind gap** — confidence: med
- **What:** `test_to_from_01d[0]` (input shape `[0]`, rank 1, auto pad) crashes in the op. Its axes tuple (`rank=1, pad_mode=auto, alignment=hw_non_aligned`) counts as covered, but no golden case has a zero-length dim, and no tagger records volume. The reference supports this round trip (`tests/ttnn/unit_tests/base_functionality/test_to_layout.py:290`).
- **Evidence:** `tilize_program_descriptor.py:156: ZeroDivisionError: division by zero` (`num_col_blocks = ceil(core_cols_tiles / block_width_tiles)` with 0 cols).
- **Recommendation:** Propose a LOOSE_CASES entry `{"inputs": ({"input_shape": [0], "shard_api": "none", "in": _il(_DRAM), "out": _il(_DRAM), "pad_mode": "auto", "pad_value": 0},)}`, plus one rank-2 sibling `[0, 64]`. Consider an `empty` facet in `tag_alignment` (or a new `volume` axis) if more zero-dim shapes are expected in TARGET.

**1c. Harness `shard_api` capture mis-tags every legacy shard** — confidence: high
- **What:** `axes._spec_of` checks `nd_shard_spec` first, but a live legacy 2D shard carries both specs, so 88 cells are captured as `nd`. The op's own tagger was fixed for the same ambiguity (`tilize.py:266-271`); the harness was not.
- **Evidence:** `golden_blind_final/test_axes_consistency.json`: 88 × `{"shard_api": {"declared": "legacy_2d", "captured": "nd"}}`, unchanged from phase 0. `eval/golden_tests/tilize/axes.py:44`.
- **Recommendation:** Propose that `_spec_of` prefer `shard_spec` when `memory_layout` is HEIGHT/WIDTH/BLOCK, mirroring `tilize.py:_side_dict`. Also propose documenting in the golden-tests skill that legacy_2d vs nd cannot be told apart at runtime.

**1d. Translated-test defects (7 cells; test bugs, not op bugs)** — confidence: high
- `test_tilize_retile_dtype_conversion[*-fp32_to_bf16]` ×4: `test_translated.py:1158` calls `assert_with_ulp(a, b, ulp_threshold=1)` positionally, but the signature is keyword-only (`tt_py_test_utils_common/utils_for_testing.py:230`, `def assert_with_ulp(*, expected_result, actual_result, ...)`). Propose passing `expected_result=`, `actual_result=` by keyword.
- `test_to_layout_pad_value_dtype[*-INT32-*]` ×3: the translation dropped the reference's `if ttnn_dtype == ttnn.int32: torch.randint(-100, 100, ...)` branch (`test_to_layout.py:1004`). int32 then gets `torch.rand` bf16 data in [0,1), which truncates to 0 on device. Propose restoring the branch at `test_translated.py:~1940`.

## 2. SUPPORTED honesty → `tilize.py` `SUPPORTED` / `EXCLUSIONS`

Counts: `supported_fail` 56 = 48 golden fp8 (§1a) + 7 translated test bugs (§1d) + 1 real crash (§1b). `xfail_wrong_mode` 8 = fp8 TILE input × bfp output × tiny tile, the same cause as §1a. `xpass_drift` 0.

**2a. fp8 over-claim: demote until INVALID lands** — confidence: high
- **What:** Refinement 5 put `fp8_e4m3` in `SUPPORTED["dtype"]` (`tilize.py:202`) while knowing that 48 cells in the in_tile_height and sharded-input crosses fail before the op runs (`changelog.md` Refinement 5: "56 are fp8_e4m3 on sharded / retile scenarios where the harness's `from_torch` cannot build the input"). The count sat at `supported_fail` 48 from `golden_refinement_5` through `golden_perf_2`.
- **Recommendation:** If §1a is ratified, nothing is needed in the op. Until then, propose `EXCLUSIONS += [{"dtype": ttnn.fp8_e4m3, "in_tile_height": h} for h in (32,16,8,4,2,1)]` plus the fp8 × sharded-input crosses, with a comment that points to the pending INVALID request. This keeps `supported_fail` meaning "op bug".

**2b. Zero-volume crash: fix, don't demote** — confidence: med
- **What:** The crash is in host code, and no axis can express "volume 0" in EXCLUSIONS.
- **Recommendation:** Propose either an early return in the descriptor or `tilize()` when `volume==0` (allocate the padded output and fill it with `pad_value`), or a clean `ValueError` in `validate()`. Site: `tilize_program_descriptor.py:154-156`.

## 3. Helper / reference docs

**3a. The `read_coalescing` catalog entry and README say interleaved sources cannot be coalesced. That is wrong for same-bank stride reads, and it cost Phase 0 its biggest lever.** — confidence: high
- **Evidence:** `ttnn/ttnn/operations/examples/master.md:150` says "invalid across interleaved pages", and `read_coalescing/README.md:86-87` says "Interleaved input is rejected: consecutive logical pages there are on different banks". The planner copied this into `op_design.md:143` ("no coalescing across sticks: catalog `read_coalescing/`"). The implementer then found that pages p, p+num_banks, … are contiguous within one bank. It measured 2.1–2.5× on the focus shapes (commit `9ab2a35ba13`, breadcrumb `friction surface=design`).
- **Recommendation:** Propose a line in master.md/README: "Interleaved RM: pages p, p+B, p+2B (B = num banks) are contiguous in bank p%B when page size is bank-aligned; read them as one wide call per bank, then reorder locally."

**3b. perf-measure skill is silent on mismatched timer clusters (BH p150b)** — confidence: high
- **Evidence:** The implementer breadcrumb reports `DEVICE KERNEL DURATION reads ~1e11 ns ... offset timer base`. Refinement 5a: "Tracy CSV returned invalid ~2e10 ns durations". Perf 1 had to build `perf_experiments/_tools/zones.py`. `.claude/skills/perf-measure/SKILL.md:144` gives column 19 as the measurement with no caveat. The read_coalescing README covers this, but perf-measure does not.
- **Recommendation:** Propose adding to perf-measure a check ("if DEVICE KERNEL DURATION > 100× expected, the board has offset timer clusters") and the per-cluster-span recipe from `perf_experiments/_tools/zones.py`.

**3c. `tilize` helper: default `reconfig_mode` runs on every call, including `InitUninitMode::Neither`, and clobbers fast-tilize state** — confidence: med
- **Evidence:** `tilize_helpers.hpp:29-37` says to use "Neither on middle calls" but does not say that the reconfig at `tilize_helpers.inl:150,164` is independent of init mode. The perf part-optimizer hit this: "per-row tilize calls must not reconfig srcA/pack after init (clobbers fast-tilize config -> rows>=1 wrong)". The workaround is at `perf_experiments/dual_noc_writes/kernels/tilize_compute.cpp:123-128`.
- **Recommendation:** Propose a doc line on `InitUninitMode`: "Neither/UninitOnly calls must pass `ReconfigureRegisterDatatypeMode::NoReconfigure` (reconfig after fast_tilize_init corrupts output)". Better still, default reconfig to off when init is skipped.

**3d. No eval surface for failures in harness input creation** — confidence: high
- **Evidence:** incremental-verifier breadcrumb: "Cells that fail inside harness input creation ... classified xfail_wrong_mode ... No guidance". `verify_supported` lumps them with op failures, which is how §2a stayed hidden in plain sight.
- **Recommendation:** Propose a `harness_input_failure` category in `eval/verify_supported.py`: an exception raised before the op call, detected by a marker set in `helpers.create_ttnn_input_tensor`. Report it separately from `supported_fail`.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `noc_async_write_one_packet_with_state` (dataflow_api) and `dataflow_kernel_lib` local-copy helpers | capability: no multi-cmd-buf / batched-counter form for a stream of same-size local copies | **missing** | `dataflow_api.h:1163` hardcodes `update_counter=true` on a single `write_cmd_buf`. `local_copy_helpers_dataflow.inl:73-93` `read_with_state` is also single-buf, per-call. Real gain: reorder zone 3270→2010 ns/core, op −3.3 % (Perf 1) | Add a batched one-packet local-copy helper (alternating cmd bufs, counters bumped once per batch) |
| `compute_kernel_lib::tilize` (unpack step) | capability: no 2-level row stride for bank-grouped rows | **missing** | `tilize_helpers.hpp:121-135` geometry is a constant-stride `1×block_width_tiles` tile-row. No address-pattern parameter in `.inl`. Real gain: op 11016→10534 ns (Perf 2). Bypass at `tilize_compute.cpp:111,139` | Add a `RowLayout` / source-stride option to the fast path |
| `fast_tilize_init` / `fast_tilize_uninit` (compute API) | capability: ct==1 routed to standard tilize on BH; no hook between init and first block | **missing** | `tilize.h:242` `if (full_dim == 1) { tilize_init(...); return; }` under `ARCH_BLACKHOLE`. The changelog reports unit dim 1 through the fast-tilize MOPs as bit-exact, so the fallback may be over-conservative (confidence: low) | Revisit the BH ct==1 fallback. Expose an unpack-config hook, or fold it into the `RowLayout` option above |

None of the rows is an ergonomics row with ~equal ns. All three paid for a measured speedup.

**Missing from the record:**
- The Perf 2 row for `compute_kernel_lib::tilize` cites only the unpack override. The whole block loop is hand-written at LLK level, including math/pack and DEST sync (`tilize_compute.cpp:166-184`: `llk_math_fast_tilize_block_`, `llk_pack_fast_tilize_row_chunk`, `llk_*_dest_section_done`). The helper's lifecycle and sync handling were bypassed too, not just its unpack (confidence: high).
- The segment path's stateful DRAM reads that graduated in Perf 2 (`tilize_reader.cpp:611-615`) have no row. They sidestep `dataflow_kernel_lib::read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp`), which has no stateful/per-bank-group form. These are public API calls, not LLK (confidence: low).
- `tilize_reader.cpp:27` includes `local_copy_helpers_dataflow.hpp` but never calls it. All L1→L1 copies use self-aimed writes, which the helper's header argues against (`:12-23`). This is a dead include, or an undocumented reason for writes over reads (confidence: low).

```cpp
// Sketch (confidence: low — single call site). Derived from rft_unpack_override/rft_unpack_block.
namespace compute_kernel_lib::tilize_config {
struct BankGroupedRows { uint32_t group_rows; uint32_t stick_bytes; };  // row i at (i%G)*group + (i/G)*stick
}
compute_kernel_lib::tilize<ct, cb_rm_sticks, cb_tiles_out, InitUninitMode::InitAndUninit,
                           WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                           Fp32Mode::Fast>(rows, std::nullopt, BankGroupedRows{num_banks, stick_bytes});

// Sketch (confidence: med). Derived from raw_copy walk at tilize_reader.cpp:370-423.
dataflow_kernel_lib::LocalCopyBatch<seg_bytes, /*cmd_bufs=*/2> batch(noc, src_base);
for (...) batch.copy(src_off, dst_addr);   // no per-copy counter RMW
batch.commit();                            // bump write counters by N once
```

## 4. Agent prompts → `.claude/agents/*.md`

**4a. Implementer/verifier: widening SUPPORTED over known-unbuildable cells** — confidence: high
- **What:** `op_requirements.md:98` explicitly carved the fp8 sharded/retile cells out of "Done-when". The implementer still widened `SUPPORTED["dtype"]` wholesale, and no later verifier gate demoted them, despite `incremental-verifier.md:601` ("supported_fail: 0 (must be 0 to ship)"). The request to change feature_spec had no owner, so it was never applied.
- **Recommendation:** Propose adding to `incremental-verifier.md` (supported_fail row, `:215`) and `ttnn-implementer.md`: "Cells that fail before the op call are out of scope for the op. Add a temporary EXCLUSIONS entry that cites the INVALID request, and file the INVALID request where the runner surfaces it to a human. Never leave them as supported_fail."

**4b. Planner: trusted a catalog negative claim without checking the bank math** — confidence: med
- **What:** See §3a. The planner prompt led to a design that named the read count as "the fewest possible" and ruled out the biggest perf lever.
- **Recommendation:** Propose a planner-prompt line: "A catalog entry's *rejection* of a technique is scoped to its own example layout. For interleaved sources, derive the per-bank page contiguity (page p → bank p%B, offset ⌊p/B⌋) before ruling out coalescing."
