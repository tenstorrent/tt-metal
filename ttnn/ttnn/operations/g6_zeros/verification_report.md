# Verification Report: g6_zeros

Phase 0 state: one writer kernel (BRISC, NoC1), no reader and no compute kernel, regime `bank_run`.
- Writers: 104 cores, 13 per DRAM bank, 2 row-wise rectangles.
- Each core fills a 16 KiB `cb_zero_source` from `MEM_ZEROS_BASE` by L1 loopback.
- Each core then streams one contiguous same-bank slot run as ≤ 16 KiB `noc_async_write`s, ending with one write barrier.
- SUPPORTED: dtype [float32], layout [TILE].

## Code Review

Fixed in this pass:

1. **One plan derivation (DRY, BB-08).** `describe_plan()` and the dispatch path had separate page-count and page-stride derivations. Both now call `_plan_for(shape, dtype, device)`. On a plan-cache miss the dispatch path asserts that the host values equal `output.buffer_num_pages()` and `output.buffer_aligned_page_size()`, so the allocator stays the source of truth.
2. **Knobs single-sourced at call time and keyed (BB-07).** `plan_bank_runs` read `WRITERS_PER_BANK`, `ZERO_SOURCE_BYTES` and `ZERO_SOURCE_REGIONS` as default arguments, which froze them at import. They now resolve from the module constants at call time. `knob_values()` is part of the host plan-cache key, so turning a knob can never replay a stale descriptor.
3. **Every piece of work in a named zone (BB-21).** The max-run scan that sizes the fill moved into `G6Z-FILL`. The `TensorAccessor` construction, its stride `ASSERT` and the zero-source address moved into `G6Z-STORE`, the only place they are used. Only two argument reads remain outside a zone.
4. **Cache-hit address re-patch test.** `test_g6_zeros_cache_hit_new_address` was added. The first output stays alive and the second call lands on a poisoned range, so a stale common runtime arg would zero the wrong buffer and fail. The p2 dirty-destination case already covers cross-shape cache hits: it runs after p3 on the shared program, over poisoned memory.
5. **Stale tests removed.** `test_g6_zeros_debug.py`, `test_g6_zeros_bank_run_debug.py`, `test_g6_zeros_perf_knobs.py` and `test_g6_zeros_bank_run_perf.py` targeted the APIs of two superseded rebuilds: `Knobs`/`get_program`, `CORE_ORDER`, `WRITER_CORES_PER_BANK`. They no longer match the module. `test_g6_zeros_perf.py` is the current knob-sweep harness.
6. **Precision baseline bounded.** It accumulates every metric over leading-dim slices. Before this, p1 (2.7e8 elements) would have built several GB of float64 temporaries once bfloat8_b enters SUPPORTED.

Checked and clean, no change needed:
- `kernel_main()` form, `api/dataflow/...` include paths, `TensorAccessor`, non-posted writes, and one `noc_async_write_barrier` before return (BB-03).
- No CB handshake, and none is needed: `cb_zero_source` is a raw scratch region owned by one kernel.
- **Helpers.** The fill uses `Noc::async_write_zeros` (CB overload) plus `write_zeros_l1_barrier`. The store is a custom block operation over raw `noc_async_write`, because the DRAM overload of `async_write_zeros` is capped at one page per call (`noc_zero_dram.inl:24`). That would force 262,648 packets on p1 instead of 17,472. No compute or mcast helper applies.
- **Broadcast efficiency.** Nothing to broadcast. A per-core local fill is cheaper than any transfer.

Deferred: none. The one material finding, write-stream geometry at 51 % of DRAM peak, is a measured perf refinement (Refinement 2), not a code defect.

## Prompt Rules (`eval/prompts/g6_zeros.txt` → `## Rules`)

| Rule | Status |
|------|--------|
| Write every logical element with the all-zero pattern, never rely on a zeroed buffer, never -0.0 | ✓ Exercised by the dirty-destination cases (acceptance, golden loose and regression) on p2 and p3, and by probe_029 on p4 and p1. 0 non-zero bits. Only `0x00` bytes are ever written. |
| Output newly allocated per call | ✓ `allocate_tensor_on_device` on every call. `test_g6_zeros_new_storage` and the golden `new_storage` case pass. |
| Dispatch the op's own program; never delegate or transfer host values | ✓ One `ttnn.generic_op` over the op's descriptor. Nothing else runs. |
| Output crosses DRAM once as a write; no DRAM read | ✓ Each page sits in exactly one core's run (plan check: runs tile `[0, slots_b)` per bank). Zero bytes come from core-local L1 loopback. |
| Mesh: allocate and write on every device | ✓ by construction: mesh allocation plus `generic_op`'s SPMD path, and the common-arg address is the same on every chip. Only a single device is available here, so a 4-chip run is not verifiable on this board. |
| No size/shape/rank axis, no extra tagger, no EXCLUSIONS refusing a production case | ✓ `INPUT_TAGGERS = {dtype, layout}` and `EXCLUSIONS = []`. |
| OOM on a golden shape is a defect | ✓ None. L1 per core is 16,384 B on every shape. |
| Core count is the blocking decision's choice, not a shape limit | ✓ 104 writers on every production shape, p4 included (128 pages). |
| BB-01 one program / BB-02 no wrapping | ✓ |
| BB-05 L1 ≤ 1,347,776 B per core | ✓ 16,384 B. The host asserts the bound. |
| BB-06 bit-identical repeats | ✓ golden `test_repeat`. |
| BB-07 no descriptor rebuild; complete cache key | ✓ Host plan cache keyed on (device, shape, dtype, knobs). `generic_op` hashes the structure. Golden `test_cache_order` (p3→p2→p3) adds no entry. |
| BB-08 report the selected plan | ✓ `describe_plan()` returns regime, knobs, core ranges and a per-core role map. |
| BB-16 trace | ✓ All allocation is the output itself, at capture time. The descriptor is built on the first call. |
| BB-17 few rectangular core ranges | ✓ 2 rectangles: {(0,0)–(10,8), (0,9)–(4,9)}. |
| BB-21 zones and role dump | ✓ after fix 3: `G6Z-FILL`, `G6Z-FILL-WAIT`, `G6Z-STORE`, `G6Z-DRAIN`. Roles come from `describe_plan`. |
| BB-22 tests limited to production cases | ✓ Every test uses p1–p4 plus the argument-check cases. |

No soft ("prefer / consider / avoid") rule is unfollowed.

## Registry Conformance

- `INPUT_TAGGERS`: `tag_dtype(inputs, axes)` and `tag_layout(inputs, axes)` read `inputs[0]`. Both signatures are correct.
- `SUPPORTED`: `{"dtype": [float32], "layout": [TILE_LAYOUT]}`. Every axis the kernel gates on is present.
- `EXCLUSIONS`: `[]`.
- `validate()`: argument checks (ValueError / RuntimeError) run first, then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). The order is correct, and it is the first line of `g6_zeros()`.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fix was needed: `xpass_drift = 0`.
- **INVALID audit** (`eval/golden_tests/g6_zeros/feature_spec.py`): `INVALID = []`. That is correct.
  - TARGET layout is `[TILE]` alone, so the canonical bf8b + ROW_MAJOR cell is outside the universe and needs no entry.
  - There is no weight tensor, so there are no cross-tensor couplings and no canonicalization cells.

## Design Conformance (`op_design.md`)

| Dimension | Status |
|-----------|--------|
| Algorithm | ✓ `bank_run`: core-local zero source, contiguous same-bank runs, ≤ `zero_source_bytes` per write. |
| Pipeline topology | ✓ Writer only, BRISC/NoC1. |
| Work distribution | ✓ `writers_per_bank = max(1, min(grid // N, ceil(P / N)))` = 13, giving 104 cores. Bank `w mod N`, slot base/extra split, `first_page_id = b + first_slot·N`. This matches the design's table for p1–p4. |
| Inter-core communication | ✓ None, as designed. |
| Blocking-model fidelity | ✓ `writers_per_bank`, `zero_source_bytes` and `zero_source_regions` are each defined once and reach the kernel as CT args, the CB size and the plan. `num_blocks` is a runtime loop bound, so the multi-bank-per-core unlock is a knob-turn. No CB scales with an op dimension. The per-chunk loop inside `store_bank_run` is the `NOC_MAX_BURST_SIZE` mechanism cap, not a half-turned split. |
| Expression | ✓ Inside a run, every write is issued back-to-back with no per-chunk barrier or handshake. There is one barrier per core per call. |
| Performance conformance | The implementation fills the grid exactly as designed. Measured, that geometry reaches 259 GB/s on p1, 51 % of DRAM peak, because 13 NoC1 streams per bank converge from every grid row. The design flagged this as its grid-sync and core↔bank placement lamps. A measured knob-turn reaches 510.6 GB/s (below). It is filed as Refinement 2 (perf), not fixed in place: it moves off the design's Phase 0 values, regresses p4 unless handled by shape, and must be measured on bfloat8_b p1, which Refinement 1 unlocks. |

## L1 Ledger Audit

- **Ledger currency.** There is one CB, `cb_zero_source`, index 0, `total_size = zero_source_regions · zero_source_bytes = 16384`, `page_size = zero_source_bytes`, format = output dtype tag. The ledger row matches the code.
- **Capacity vs live set.** The live set is `fill_bytes = min(zero_source_bytes, core_run_bytes)`. It is 16,384 B on p1/p2/p3. On p4 it is 2–4 KiB: capacity exceeds the live set there, and the ledger justifies it because the fill writes only the live prefix. *Under*: none. The buffer streams over `slot` (the run is walked in chunks) and spans no block axis.
- **Page format vs DEST.** Not applicable, and recorded with a mechanism reason: a dataflow-only payload with no compute kernel and no DEST.
- **Disjoint lifetime.** Single CB, nothing to pair.
- **Bounds.** `zero_source_bytes` is bounded by the model L1 share and DRAM alignment, both host-asserted. No op dimension appears in a capacity expression.
- **Total per core**: `L1 = zero_source_regions · zero_source_bytes = 16,384 B`. It scales only with the `zero_source_bytes` knob; shape, dtype, grid and mesh size do not affect it.
- **Data-movement budget.** Present and consistent with the code: output written once, 0 B DRAM read, 0 B cross-core. The cheapest-traffic split (bank-major) is implemented.
- **Block-size defaults.** Interleaved: the split spreads over 104 of 110 cores (13 × 8, equal streams per bank) at the coarsest block, the whole per-core share. The ledger ranks candidates on bytes and packet count. Measurement shows the binding cost is NoC-path convergence on each bank endpoint, which the ledger's transfer-shape lens does not see. The fix changes writer count and placement only, not any CB, so it needs no ledger change and is folded into Refinement 2.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_zeros/test_g6_zeros_precision_baseline.py` was run under `--dev`. The reference is all zero, so PCC and relative RMS are degenerate: the reference std is 0, and `comp_pcc` falls back to allclose. The gate is bitwise: max ULP 0 and no -0.0. The got/true ratio is undefined, since every reference element is 0. The scale-bug triage does not apply: the op computes nothing.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | RMS Err | Relative RMS | Max ULP | -0.0 count |
|-------|-------|-----|-------------|--------------|---------|--------------|---------|------------|
| (8,12,128,128) | float32 | 1.0 | 0.0 | 0.0 | 0.0 | undefined (ref std 0) | 0 | 0 |
| (16,12,128,128) | float32 | 1.0 | 0.0 | 0.0 | 0.0 | undefined (ref std 0) | 0 | 0 |
| (1,4,256,128) | bfloat16 | xfail (outside SUPPORTED). Probe_029 past `validate()`: 0 / 131,072 non-zero bits | | | | | | |
| (32831,1,32,256) | bfloat8_b | xfail (outside SUPPORTED). Probe_029 past `validate()`: 0 / 268,951,552 non-zero bits | | | | | | |

**Assessment**: exact. Every element is the +0.0 bit pattern, including on poisoned ranges.
**Recommended tolerances**: exact for every dtype. `helpers.TOLERANCES` already holds `("exact", None)`, so the defaults hold.

## Verifier CLI Summary

From `verifier_results/verifier_report.json`, golden run 13/19 passed, 0 failed, 0 hangs:
- supported_pass: 13
- xfail_expected: 6. All are bfloat16 / bfloat8_b cells, queued as Refinement 1.
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Unit tests (`tests/ttnn/unit_tests/operations/g6_zeros/`, `--dev`): 40 passed, 6 xfailed. The xfails are the same bf16/bf8b refusals.

## Perf Baseline (device kernel duration, Blackhole, 11×10 grid, 8 DRAM banks)

| plan | p1 bf8b 285.8 MB | p2 fp32 12.6 MB | p3 fp32 6.3 MB | p4 bf16 0.26 MB |
|------|------------------|-----------------|----------------|-----------------|
| Phase 0 (13/bank, row-wise, 104 cores) | 1,102,674 ns (259 GB/s) | 48,693 | 24,827 | 1,603 |
| 1/bank on its NoC1-optimal worker (8 cores) | 559,640 ns (510.6 GB/s) | 25,807 | 13,471 | 1,775 |

p1 target: ≤ 837,190 ns. The full writers-per-bank sweep is in `op_requirements.md` → Refinement 2. Every variant was checked over a poisoned range.

## Recommendations

- **Order.** Refinement 1 (dtypes) comes first, because it lands the perf-flagged p1 contract. Then Refinement 2 (perf on p1). TARGET − SUPPORTED is exactly `dtype ∈ {bfloat16, bfloat8_b}`, so the queue holds one generality refinement, and the cadence runs straight into perf.
- **No third refinement.** The probe puts p1 at 510.6 GB/s once Refinement 2 lands. That is at the 512 GB/s nominal DRAM peak and below the lead's 620.14 µs roofline. The remaining design lamps have no device-ns headroom on the flagged shape, so none is filed:
  - stateful `noc_async_write_one_packet_with_state`
  - an NCRISC/NoC0 split of each run
  - `zero_source_bytes` 8 KiB / 32 KiB

  If Refinement 2 lands short of about 510 GB/s, its partial-tick follow-up is where these belong.
- **Host time on p4**, the most-called case. `custom_program_hash` is unset, so every untraced call hashes the descriptor. This is host time, not device ns, and traced replay skips it. It is not a queue item. Measure untraced host time on p4 if host dispatch shows up in the model.
- **p4 is launch-dominated**: 1.6 µs for 256 KiB. Any p1-driven placement change must keep p4 in its guard set (see Refinement 2's notes).
