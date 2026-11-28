# Verification Report: g6_zeros

Phase 0 state: regime R1, bank-affine coalesced zero runs.
- **Kernels**: one writer kernel (BRISC, NoC1). No reader and no compute kernel: the op has no input and does no math.
- **Cores**: 104 = 8 DRAM banks × `cores_per_bank` 13, the first 104 cores in row-major order (2 rectangles).
- **Per core**: fill a `burst_pages·aps` (≤16 KiB) `cb_zero_source` once by NoC loopback from `MEM_ZEROS_BASE`. Then write one contiguous run of bank-rows of bank `k % 8` as ≤16 KiB `noc_async_write`s, with no barrier inside the run. End with one `noc_async_write_barrier`.
- **SUPPORTED**: dtype [float32], layout [TILE].

## Code Review

Fixed in this pass:

1. **The plan report no longer allocates on the device.** `g6_zeros_plan()` used to allocate, and then free, a full output tensor just to read `buffer_num_pages()` and `buffer_aligned_page_size()`. For p1 that is 286 MB of DRAM. A new helper, `_plan_inputs(shape, dtype, device)`, derives `N = d0·d1·(d2/32)·(d3/32)`, `aps = ttnn.tile_size(dtype)`, the grid and `B` on the host, with no device work. The plan report and the dispatch path now share this one derivation (DRY, BB-08). On a descriptor-cache miss, the dispatch path asserts that `N` and `aps` equal the allocated buffer's values, so the allocator stays the source of truth. The report also returns the core ranges now.
2. **The descriptor-cache key is the plan input itself.** It is now `(dtype, N, aps, gx, gy, B)`. The old key `(id(device), N, dtype, aps, gx, gy)` had two problems. It left out `B`, which the plan depends on. And `id()` of a Python object can be reused after a device closes. The descriptor holds nothing device-specific beyond those values, and only `common_runtime_args[0]` (the buffer address) is patched per call. `custom_program_hash` already covers dtype plus the full plan key.
3. **The perf harness was rewritten.** `test_g6_zeros_perf.py` called `pdm.plan_bank_runs` and `grid_fill_writers_per_bank`, which do not exist in the current module, so it could not run. It now sweeps the real knob, `make_plan(..., cores_per_bank=…)`, over the four production cases. That sweep produced the numbers behind Refinement 2.

Checked and clean, no change needed:
- **Kernel basics**: `void kernel_main()`, `api/dataflow/...` includes and a `TensorAccessor` for interleaved DRAM. There is one `noc_async_write_barrier` before return (BB-03), and a core with `run_pages == 0` issues no NoC transaction before returning.
- **Dev-mode ASSERTs**: the bank stride check (`contiguous_page_stride() == B`) and the run contiguity check (`num_contiguous_pages`) are exercised under `--dev`. The acceptance suite passes there with watcher and NoC sanitizer on.
- **Write bounds**: every write is `≤ burst_pages·aps = zero_source_bytes` by construction, so no write reads past the zeroed source.
- **CB use**: `cb_zero_source` has no `reserve/push/wait/pop`, by design. It is single-thread scratch whose content never changes.
- **Helpers**: the fill uses `Noc::async_write_zeros` (CB overload) plus `write_zeros_l1_barrier`. The store is a raw `noc_async_write` per burst, because the DRAM overload of `async_write_zeros` is limited to one page per call (`noc_zero_dram.inl:24`). Using it would mean 262,648 transactions on p1 instead of about 17.5 k. No mcast or compute helper applies, and there is nothing to broadcast.

### Design conformance

- **Algorithm and topology**: match `op_design.md` exactly (R1, a single writer, a single CB, the three block operations in named zones `fill_zero_source` / `store_zero_run` / `drain_writes`, BB-21).
- **Work distribution**: as designed: `cores_per_bank = min(floor(G/B), ceil(N/B))`, core `k` → bank `k % B`, slot `k // B`, row-major order, 2 rectangles (BB-17). The host asserts `Σ run_pages == N`.
- **Blocking-model fidelity**: `cores_per_bank`, `burst_pages` and `MAX_WRITE_BURST_BYTES` are each defined once, in `make_plan` and the module constant. Everything else is derived from them: the RT args `(first_page, run_pages)`, the core ranges, `zero_source_bytes` (the CB size) and the kernel's `burst_bytes`. `burst_pages`, `B` and `aps` reach the kernel as CT args, and no kernel loop bound is a literal.
- **Expression**: the per-core block is the whole run, with no per-unit synchronization inside it.
- **Placement** is the one design lamp that is not yet a knob: `core_coord` is fixed row-major. Refinement 2 turns it into a knob.
- **Performance conformance**: the design's grid fill is honored (104 cores). The sweep shows that this default is *slower* than fewer writers per bank: 1 writer per bank is 1.20× faster on p1 and 1.16× on p2. This is the design's own grid-sync lamp measuring against its default. It is filed as Refinement 2, which is a knob-turn, not a defect.

## Prompt Rules (`eval/prompts/g6_zeros.txt` → `## Rules`)

| Rule | Status |
|------|--------|
| Every logical element written with +0.0 bits; never rely on a zeroed buffer; never -0.0 | ✓ The dirty-destination cases (acceptance, golden loose) pass on p2 and p3. probe_031 covers p4 bf16 and p1 bf8b on poisoned ranges with 0 wrong bits. Only `0x00` bytes are written. |
| Output newly allocated per call | ✓ `allocate_tensor_on_device` runs on every call. The `new_storage` tests pass. |
| Own program; no delegation or host transfer | ✓ One `ttnn.generic_op` over the op's own descriptor. |
| Output crosses DRAM once; no DRAM read | ✓ Each page sits in exactly one run (host assert). The zero bytes come from core-local loopback. |
| Mesh: write every device | ✓ by construction: one SPMD descriptor through `generic_op`, with the same address on every chip. This board has a single device, so a 4-chip run is not verifiable here. |
| No size/shape/rank axis, no extra tagger, no refusing EXCLUSIONS | ✓ `INPUT_TAGGERS = {dtype, layout}` and `EXCLUSIONS = []`. |
| OOM on a golden shape is a defect | ✓ None. L1 per core is ≤16,384 B on every shape. |
| Core count is the blocking decision's choice | ✓ `cores_per_bank` is a plan knob, not a shape limit. |
| BB-01 one program, BB-02 no wrapping, BB-03 drain | ✓ |
| BB-05 L1 ≤ 1,347,776 B | ✓ 16,384 B maximum. |
| BB-06 bit-identical repeats, BB-07 no rebuild + complete key | ✓ Acceptance `cache_order_and_repeat` and golden `then_shapes` (p3→p2→p3) add no program on the repeat. The key is fixed per item 2 above. |
| BB-08 plan reporting | ✓ `g6_zeros_plan()` reports the plan on the host only (item 1). |
| BB-16 trace | ✓ The only allocation is the output, at capture. The descriptor is built on the first call. |
| BB-17 few rectangles | ✓ 2 ranges: `{(0,0)-(10,8), (0,9)-(4,9)}`. |
| BB-21 zones | ✓ All three block operations are zoned. Roles come from `g6_zeros_plan`. |
| BB-22 tests limited to production cases | ✓ Every test uses p1–p4 plus the argument checks. |

No soft ("prefer / consider / avoid") rule applies unfollowed.

## Registry Conformance

- `INPUT_TAGGERS` has `{dtype, layout}`, both with the `(inputs, axes)` signature, reading the scenario dict. `SUPPORTED` has `{dtype: [float32], layout: [TILE]}`. `EXCLUSIONS = []`.
- `validate()` runs the structural checks (`ValueError`), then the taggers, then SUPPORTED (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). It is the first line of `g6_zeros`.
- The op file declares no `INVALID`.
- No auto-fix to SUPPORTED was needed: xpass_drift is 0. bf16 and bf8b are known to work past `validate()` (probe_031), but `validate()` refuses them, so they are xfail rather than XPASS. Adding them is Refinement 1.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. TARGET has TILE only, so the canonical bf8b+ROW_MAJOR entry has no layout to couple with. There is no weight tensor and no cross-tensor coupling.

## L1 Ledger Audit

- **Currency**: one CB in the code, `cb_zero_source`, index 0, `total_size = page_size = burst_pages·aps`. It is one ledger row with the same expression. ✓
- **Capacity vs live set**: the capacity is one burst's source, and the live set is the same, because every burst reads all of it. It spans `burst` and streams over `bank_row`, `bank` and `mesh`. Nothing is oversized or collapsed. ✓
- **Page format vs DEST**: n/a. The payload is dataflow-only and never unpacked (the `dtype` label is cosmetic). ✓
- **Disjoint lifetimes**: n/a. It is the only buffer, and its "why not removed" is justified: `MEM_ZEROS_BASE` is 512 B, so using it directly would mean 32× the write count. ✓
- **Bounds**: `zero_source_bytes ≤ 16384`, closed form in `burst_pages` and `aps`. No op dimension appears in any capacity. ✓
- **Traffic budget**: output 1 DRAM crossing (write), 0 reads, 0 cross-core traffic. This is consistent with the code. R1 is the cheapest-traffic split, so nothing is deferred. ✓
- **Block-size defaults**: the interleaved default (full grid, coarsest block = whole run) held as designed. The measurement says fewer cores are faster for this write-only, bank-bound op. That departure is folded into Refinement 2, which must record it in the ledger. It gets no queue entry of its own.
- **Per-core footprint**: `burst_pages·aps ≤ 16 KiB`. It scales only with `burst_pages` and with `dtype` through `aps`.

## Precision Baseline

`test_g6_zeros_precision_baseline.py`. The reference is all zero, so relative RMS is undefined (the reference std is 0), and so is the got/true ratio (every reference element is zero). The real gate is the bit pattern: max ULP 0 and no -0.0.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | RMS Err | Max ULP | -0.0 count |
|-------|-------|-----|-------------|--------------|---------|---------|-----------|
| (8,12,128,128) | float32 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| (16,12,128,128) | float32 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| (1,4,256,128) | bfloat16 | xfail (outside SUPPORTED); probe_031: 0 non-zero bits / 131,072 | | | | | |
| (32831,1,32,256) | bfloat8_b | xfail (outside SUPPORTED); probe_031: 0 non-zero bits / 268,951,552 | | | | | |

**Assessment**: exact. The op writes `0x00` bytes and computes nothing.
**Recommended tolerances**: exact for every dtype (`helpers.TOLERANCES`). The scale-versus-precision triage is n/a: there are no failures, and the reference is all zero.

## Verifier CLI Summary

`ttnn/ttnn/operations/g6_zeros/verifier_results/verifier_report.json`, re-run after the fixes:
- supported_pass: 13
- xfail_expected: 6. These are the p4 bf16 ×3 and p1 bf8b ×3 cells, which go to Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

## Device time (Blackhole, `test_g6_zeros_perf.py`, device kernel duration, ns)

| case | default (cpb 13, 104 cores) | cpb 8 | cpb 4 | cpb 2 | cpb 1 (8 cores) |
|------|----|----|----|----|----|
| p1 bf8b (286 MB) | 1,092,860 | 1,047,264 | 942,677 | 971,191 | 906,971 |
| p2 fp32 (12.6 MB) | 47,114 | 44,731 | 41,344 | 44,024 | 40,619 |
| p3 fp32 (6.3 MB) | 24,271 | 22,514 | 21,830 | 22,205 | 20,781 |
| p4 bf16 (0.26 MB) | 1,807 | 1,799 | 1,874 | 1,875 | 1,757 |

The p1 target is ≤ 837.19 µs (Refinement 2).

## Recommendations

- **Queue coverage**: `TARGET − SUPPORTED = {dtype: bfloat16, bfloat8_b}`. Refinement 1 covers both values. `layout` is complete, and there are no INVALID or INVALID_FOR_ARCH values.
- **Perf**: one perf refinement is filed (Refinement 2, p1, writer count plus bank-near placement). A second perf slot is not filed yet. The perf flag requires every perf phase to target p1, and p1's remaining headroom after Refinement 2 is unknown: the roofline is 620 µs, and the levers left (`burst_pages`, a split BRISC+NCRISC writer per core per catalog `split_reader`) only pay if p1 is still issue-bound and not DRAM-bound. Re-measure after Refinement 2. If p1 is above about 1.1× the roofline, file a follow-up perf refinement on p1 with those levers.
- **p4** (128 pages, about 1.8 µs) is dominated by per-call overhead. It is not a flagged shape, so it is a guard, not a target.
