# tilize — changelog

## Phase 0 — initial implementation (2026-07-29)

### What was done

Implemented `tilize` (ROW_MAJOR → TILE layout conversion) from `op_design.md` on the Python
`ttnn.generic_op` / `ProgramDescriptor` path.

Files:

| File | Contents |
|---|---|
| `tilize.py` | `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `PROPERTIES`, `validate()`, structural checks, entry point |
| `tilize_program_descriptor.py` | host planner (2 paths) + CBs + kernel descriptors + `ComputeConfigDescriptor` |
| `kernels/tilize_reader.cpp` | `read_sticks_for_tilize<TILE>` + raw strided fallback |
| `kernels/tilize_compute.cpp` | `compute_kernel_lib::tilize` |
| `kernels/tilize_writer.cpp` | raw whole-TILE-page `noc_async_write` |
| `tests/.../tilize/_bench_tilize.py` | perf bench + ablation harness (no PCC asserts) |

**Prerequisite fix (`op_design.md` Risk #1, confirmed):** `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl`
defined `has_unpack_to_dest_fp32` **twice, byte-identically** (lines 47-63 and 65-81) — a hard
`error: redefinition` that made *every* kernel including `tilize_helpers.hpp` fail to JIT-compile.
Deleted the duplicate. This was the first thing done; nothing else could build until it was.

**Two dataflow paths, as designed:**

- **Path A/C — generic.** `read_sticks_for_tilize<cb_rm_input, TilizeGranularity::TILE>` (32 strided
  stick-reads per barrier, lever B7) → `compute_kernel_lib::tilize<chunk_wt, …>` → raw whole-tile-page
  `noc_async_write` (`chunk_wt` writes per barrier). Work unit = a *chunk-block* (32 rows × `chunk_wt`
  tile-columns); each core owns a 2D rectangle, height-first with width filling the remainder. When a
  ROW_MAJOR-*sharded* input puts more than one page on a logical row, the reader switches to a raw
  strided read (the helper hard-codes one page per row — `tilize_helpers_dataflow.inl:121`).
- **Path B — zero-copy aliased CBs** for same-spec L1-sharded I/O. Both CBs built with
  `cb_descriptor_from_sharded_tensor` + the page-size read-modify-write-back idiom, so the CB base
  address *is* the shard base address. Reader = one `cb_push_back`, writer = one
  `cb_wait_front`/`cb_pop_front`, **zero NoC traffic** (verified on device — see the ablation table).

**Helper conformance.** Both compute and read phases use the `kernel_lib` helpers named by the design.
The only raw-LLK/raw-API substitution is the **writer**, documented at the top of `tilize_writer.cpp`
with the design's own justification: the sole write helper in `kernel_lib`
(`write_sticks_after_untilize`) emits ROW_MAJOR *sticks* (`tilize_helpers_dataflow.inl:232-236`) — it
is the untilize partner, the wrong direction, and would scatter tile bytes to stick addresses.

### Accuracy achieved

| Suite | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/tilize/test_tilize.py` | **60 / 60** in both `--dev` and production mode (no race) |
| `eval/golden_tests/tilize/test_golden.py` | **126 passed, 90 skipped (INVALID), 0 failed, 0 xfail/xpass** |
| `eval/golden_tests/tilize/test_regression.py` | **9 / 9** |
| `eval/golden_tests/tilize/test_golden_main_tests.py` (reference/grader) | **105 passed, 28 skipped, 0 failed** (2 collection ERRORs are internal to the golden file — `use_module_device` marker vs a `device_params` parametrize on `test_deepseek_v3_mla_tilize_trace_mode`) |
| `eval/golden_tests/tilize/test_translated.py` | **275 passed, 1 failed** — the one failure requests a hardware-invalid config (see issue 6) |

Measured accuracy is **bit-exact** wherever the format allows it, per the identity oracle:

| Transition | Oracle | Result |
|---|---|---|
| bf16→bf16, fp32→fp32, uint32/uint16/int32 passthrough | `comp_equal` (exact) | exact |
| bf16→fp32 (widening) | `comp_equal` (exact) | exact |
| fp32→bf16 (narrowing) | PCC 0.999 | pass |
| bf16→bf8b, fp32→bf8b | PCC 0.99 | pass |

fp32 is bit-exact because the compute kernel selects `Fp32Mode::Lossless` from the CB format
(`is_fp32_input_format<cb_rm_input>()`), and the host satisfies both of its `static_assert`s
(`fp32_dest_acc_en=True` and `unpack_to_dest_mode[cb_rm_input] = UnpackToDestFp32`, assigned as a
whole 32-element list because the bound vector copies on `__getitem__`).

### Perf gate

Bench: `tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py` (measurement only, no PCC).
5 rounds × 10 launches, median, warm-up discarded. All CVs ≤ 1.9 %, so the noise floor is ~2 %.
Box: WH B0, 8×8 compute grid, AICLK ≈ 985 MHz.

#### 1. Bound classification (ablation: `TILIZE_BENCH_ABLATE=1`)

Payload stubbed via `TILIZE_SKIP_DM` / `TILIZE_SKIP_COMPUTE` compile-time flags; every CB op, barrier
and loop trip count kept identical (the address-gen is kept behind a `volatile` sink so DCE cannot
delete the timed loop).

| regime | full | no_compute | no_dm | sync_only | **verdict** |
|---|---|---|---|---|---|
| a_square `[1,1,2048,2048]` bf16 | 86.2 µs | 87.3 µs (=full) | 5.8 µs | 3.1 µs | **DM-bound** |
| b_wide_short `[1,1,32,16384]` | 13.3 µs | 13.2 µs (=full) | 2.2 µs | 1.2 µs | **DM-bound** |
| e_square_fp32 | 182.0 µs | 185.9 µs (=full) | 13.2 µs | 5.7 µs | **DM-bound** |
| e_square_bf8b_out | 64.2 µs | 65.1 µs (=full) | 4.5 µs | 3.1 µs | **DM-bound** |
| c_single_core `[1,1,512,512]` | 31.6 µs | 27.6 µs | 17.5 µs | 10.5 µs | DM-bound, **large sync floor (33 %)** |
| d_tall_narrow `[1,1,2048,32]` | 3.70 µs | 2.79 µs | 1.48 µs | 1.20 µs | **sync/dispatch-bound** (2 tiles/core) |
| f_sharded_small (Path B) | 1.37 µs | 0.69 µs | **1.36 µs** | 0.69 µs | **compute+sync-bound, zero DM** |
| f_sharded_large (Path B) | 2.07 µs | 0.90 µs | **2.07 µs** | 0.90 µs | **compute+sync-bound, zero DM** |

**The zero-copy claim is proven on device, not asserted:** on both Path-B regimes `no_dm == full` to
within 0.2 % — removing every `noc_async_read`/`write` changes nothing because there are none. That is
lever **C14** confirmed by measurement.

#### 2. Ceiling vs measured

`noc_estimate` (NPE) brackets per core, depth-2 ⇒ read/write pipeline ⇒ `max(reader, writer)`; then
`op_target = max(per_core_noc_bound, dram_floor)` (Step 4b). DRAM spec peak 288 GB/s.

| regime | traffic | measured | achieved GB/s | NPE bracket [no-cont … full-cont] | DRAM floor @288 | **achieved vs floor** | in-tree copy reference |
|---|---|---|---|---|---|---|---|
| a_square | 16.78 MB | **85.6 µs** | 196.1 | 6.3 … 113.8 µs | 58.3 µs | **0.68** | **86.6 µs @ 193.8 GB/s** → ratio **1.01** |
| e_square_fp32 | 33.55 MB | 182.2 µs | 184.2 | — | 116.5 µs | 0.64 | — |
| e_square_bf8b_out | 12.85 MB | 64.2 µs | 200.0 | — | 44.6 µs | 0.70 | — |
| b_wide_short | 2.10 MB | 13.4 µs | 156.5 | — | 7.3 µs | 0.54 (launch-latency bound at 2 MB) | — |
| c_single_core | 1.05 MB | 31.5 µs | 33.3 | 25.2 µs (1-core) | — | **0.80** | 17.9–18.3 GB/s (`double_buffer/report.md`) → **1.8× better** |

**Headline:** on the grid-filling square, tilize runs at **196.1 GB/s in 85.6 µs**, i.e. **1.01× the
in-tree measured 64-core DRAM→DRAM *copy* of the same 16.78 MB** (86.6 µs @ 193.8 GB/s,
`examples/dram_saturation/report.md`) — while additionally running a tilize LLK on every tile. The
operationally meaningful DM ceiling is the achievable-copy number, and the op is **at** it. The
remaining 0.68 → 1.0 of DRAM *spec* peak is interleaved round-robin congestion that the same in-tree
report measures as the best a 64-core spread achieves.

#### 3. A0 active-core gate (machine-checked by the bench, not eyeballed)

| regime | `total_tiles` | required `min(grid, total_tiles)` | **measured cores** | ✓ |
|---|---|---|---|---|
| a_square | 4096 | 64 | 64 | ✓ |
| **b_wide_short** (`nt_h = 1`) | 512 | 64 | **64** | ✓ |
| c_single_core (`use_multicore=False`) | 256 | 1 (forced) | 1 | ✓ |
| d_tall_narrow (`Wt = 1`) | 64 | 64 | 64 | ✓ |
| f_sharded_small | — | shard's own cores | 4 | ✓ |
| f_sharded_large | — | shard's own cores | 64 | ✓ |

The wide-short gate is the one that matters: `nt_h = 1`, so a height-only
`split_work_to_cores(nt_h)` would strand it on **one** core. It runs on 64.

Per-core CB L1 is **bounded by a constant in `W`** in every regime (`depth * chunk_wt * (tile_in +
tile_out)`, `chunk_wt ≤ WT_CHUNK_MAX = 16`): max observed 131 072 B/core on a `W = 16384` tensor.

#### 4. Mode-C used-optimization ledger

| lever | id | predicted Δ | **measured Δ** | verdict |
|---|---|---|---|---|
| 2D height-first split | A0 | wide-short: 64 cores vs 1 | b_wide_short **13.4 µs (64c) vs 61.5 µs (1c) = 4.59×** | **KEEP** — the single biggest lever |
| one barrier per block (32 reads / `chunk_wt` writes) | B7 | read 113.8→173.1 µs, write 53.8→92.2 µs (+52 % composed) | not re-measured on device (model + design reference) | KEEP |
| width coalescing (`chunk_wt` 16 ⇒ 1024 B reads vs 64 B) | B5/B6 | read 113.8→193.3 µs (+70 %) | not re-measured on device | KEEP |
| zero-copy aliased CBs | C14 | DRAM traffic → 0 | **`no_dm == full` (±0.2 %) on both sharded regimes** | **KEEP — proven** |
| depth-2 CBs | C16 | +47 % vs the per-core NoC bound | a_square **85.6 µs (d2) vs 85.9 µs (d1) = +0.3 %, inside the 2 % noise floor**; sharded-small 1.39 vs 1.36 µs | **NEUTRAL** — see below |
| `row_wise=True` placement | A1 | 2.91× (`noc_placement/report.md`) | not re-measured | KEEP (design reference) |
| reads NoC0 / writes NoC1 | B9 | 4.8× / 4.3× (`noc_placement/report.md`) | not re-measured | KEEP (design reference) |
| CT `TensorAccessorArgs`, base-address-only RT args | D18/D19 | program-cache hit | `test_tilize_program_cache` passes (2nd call adds no entry) | KEEP |

**C16 is the interesting negative result.** The NPE model predicts depth-2 should pay (+47 % against
the per-core NoC bound), but on device it is inside the noise floor. Reason: the binding resource is
**DRAM aggregate bandwidth** (~196 GB/s measured, ~193.8 GB/s achievable-copy ceiling), which *both*
depths already reach — so removing the reader/writer serialization cannot help. Per `/perf-ceiling-dm`
Step 5, "predicted-pays-but-measured-doesn't means the bound wasn't the binding resource."

Practical consequence: **`use_double_buffer=False` halves per-core CB L1 (131 072 → 65 536 B) at zero
measured perf cost on the DRAM-bound interleaved regime.** The default is left at `True` because
`op_design.md` and the acceptance test specify it; gating the default on the bound class is a concrete
follow-up (below).

#### 5. Perf-bench baseline (cumulative set — carry forward and re-measure every phase)

| regime | shape | dtype | cores | ns (median) | GB/s |
|---|---|---|---|---|---|
| a_square | `[1,1,2048,2048]` | bf16 | 64 | **85 569** | 196.1 |
| b_wide_short | `[1,1,32,16384]` | bf16 | 64 | **13 404** | 156.5 |
| c_single_core | `[1,1,512,512]` | bf16 | 1 | **31 465** | 33.3 |
| d_tall_narrow | `[1,1,2048,32]` | bf16 | 64 | **3 666** | 71.5 |
| e_square_fp32 | `[1,1,2048,2048]` | fp32 | 64 | **182 189** | 184.2 |
| e_square_bf8b_out | `[1,1,2048,2048]` | bf16→bf8b | 64 | **64 238** | 200.0 |
| f_sharded_small | `[1,1,512,64]` H-sharded | bf16 | 4 | **1 388** | 0 DRAM |
| f_sharded_large | `[1,1,2048,512]` B-sharded | bf16 | 64 | **2 072** | 0 DRAM |
| x_square_depth1 | `[1,1,2048,2048]` d1 | bf16 | 64 | **85 849** | 195.4 |
| x_wide_short_1core | `[1,1,32,16384]` 1c | bf16 | 1 | **61 526** | 34.1 |
| x_sharded_small_depth1 | `[1,1,512,64]` d1 | bf16 | 4 | **1 359** | 0 DRAM |

### Issues encountered

1. **`tilize_helpers.inl` redefinition** (Risk #1) — fixed before anything else; see above.
2. **Tile grid must come from the OUTPUT padded shape.** The first implementation derived
   `folded_H / W / nt_h / Wt` from the *input's* padded shape. A ROW_MAJOR-*sharded* input rounds its
   last dim up to a whole number of shard widths (logical `W = 160` with `shard_W = 96` is stored as
   padded `W = 192`), while the TILE output keeps the logical width. That inflated `Wt` and corrupted
   every writer page index — **whole-tensor data mismatch (Max ATOL 0.996)** on 26 reference-suite
   cells, plus OOM where the inflated split oversized the plan. `op_design.md`'s "Derived geometry"
   table (`folded_H = prod(padded_shape[:-1])`) does not say *which* tensor's padded shape, and for a
   sharded input the two differ. Fix: the tile grid comes from the output; the input's padded last dim
   is a *source row stride* only (`in_padded_row_bytes / in_page_bytes`), guarded by "the two padded
   shapes must agree on all leading dims". Reference suite 79 → 105 passing.
3. **`chunk_unit` must be `gcd(Wt, page_wt)`, not `page_wt`.** When a logical row spans several source
   pages, `chunk_wt` has to divide *both* `Wt` (for the column split) and the page width in tiles (so a
   chunk never straddles a page). Using `page_wt` alone broke the `chunk_wt | Wt` invariant on e.g.
   `Wt = 4`, `page_wt = 3`.
4. **`EXCLUSIONS` from the design turned out to be unnecessary.** `op_design.md` proposed refusing
   `{use_multicore: False} × sharded` because "sharded I/O is inherently multi-core". That is not a
   kernel-level boundary here: the generic `TensorAccessor` path addresses sharded pages from *any*
   core count, so `use_multicore=False` now routes to a 1-core generic program (Path B is gated on
   `use_multicore`) instead of being refused. The reference suite exercises exactly that cell, so
   declaring it excluded would have been a refusal the op does not need. `EXCLUSIONS` is empty.
5. **Device profiler needs three env vars in-process.** `ttnn.get_latest_programs_perf_data()` returns
   `{}` unless `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_MID_RUN_DUMP=1` and
   `TT_METAL_PROFILER_CPP_POST_PROCESS=1` are all set *before the device opens*; and a
   `ReadDeviceProfiler` after a *single* launch reliably returns an empty window, so the bench batches
   reads per round.
6. **An out-of-range L1 shard grid wedged the command queue — now refused up front.**
   `test_translated.py::test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107` builds its
   **L1** output `ShardSpec` from `device.dram_grid_size().x` (= 12 on WH B0) while the compute grid is
   8×8, so it asks for an L1 shard on core (11,0), which does not exist. `allocate_tensor_on_device`
   *accepts* it; the failure surfaced much later inside
   `TensorAccessorArgs::get_compile_time_args()` as
   `No core coordinate found at location: (8, 0, TENSIX, LOGICAL)` — thrown after the output buffer was
   allocated and the program was partway built. That left the command queue corrupted (`TT_FATAL:
   Unexpected values for event in completion queue` on the next `synchronize_device`) and then
   **segfaulted pytest** while it formatted the traceback, aborting the whole file.
   Added `_check_l1_shard_grid()`, which validates any L1 shard grid against
   `compute_with_storage_grid_size()` **before** allocating anything. Effect: that cell now fails with
   a clean `ValueError`, the device stays healthy, and the run continues — which took
   `test_translated.py` from *aborted after 11 tests* to **275 passed / 1 failed**. The remaining
   failure is a device-portability bug in the reference test itself (it reuses a DRAM-bank grid for an
   L1 shard), not an op gap; it would pass on a box whose compute grid has ≥ 12 columns.

### Advisory deviations from `op_design.md`

Binding items (algorithm, pipeline topology, work-distribution strategy, external interface) are
implemented as specified. Advisory deviations:

- **Crossover paths (design Path C rows 1-2) use the generic accessor path, not one-sided aliasing.**
  DRAM-interleaved RM → sharded TILE, and sharded RM → DRAM-interleaved TILE, are correct via
  `TensorAccessor` on both sides rather than aliasing the sharded side's CB. One-sided aliasing needs a
  per-scheme shard-index → global-tile-index mapping (HEIGHT / WIDTH / BLOCK × orientation × ND), which
  the accessor already does correctly; the design assigns these to R3b/R3c. Same-spec sharded I/O
  **does** use the zero-copy path (Path B), because it is purely shard-local and needs no such mapping.
- **Split reader (design C7, ADOPT for R3b) not implemented.** On the DRAM→sharded crossover the
  design suggests NCRISC/BRISC each taking alternate tile-row blocks. Deferred: with the generic path
  the writer is not idle, and the acceptance-test crossover shapes give one tile-row per core, so there
  is nothing to split.
- **`WT_CHUNK_MAX = 16` and `L1_CB_BUDGET_BYTES = 131072` kept at the design's initial values** (their
  sweep is R2).
- **Added `SKIP_DM` / `SKIP_COMPUTE` compile-time ablation flags** to all three kernels (the design's
  Performance-Methodology section asks for exactly these) plus the `_bench_tilize.py` harness.

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py` — perf bench (8 design-mandated regimes +
  3 Mode-C counterfactual regimes), ablation harness, A0 core-count reporting. No PCC assertions;
  underscore-prefixed so the directory run does not collect it (verified: directory run = 60 tests).
- `tests/ttnn/unit_tests/operations/tilize/probes/probe_00{1..5}.py` — API-availability and
  shard/page-geometry probes kept as the record of how the geometry bug was found.

### Ranked follow-ups for the next phase

1. **Gate depth-2 CBs on the bound class (C16, B0).** Measured neutral on the DRAM-bound interleaved
   regime while costing 2× CB L1. Concrete lever: default `use_double_buffer` to `False` when the
   planner's per-core work exceeds a threshold *and* the op is DRAM-saturated, keeping `True` for the
   latency-bound small regimes. Predicted: no perf change, −65 536 B/core L1 on wide tensors.
2. **`d_tall_narrow` is sync/dispatch-bound** (`sync_only` = 33 % of `full`, 2 tiles/core). Lever A0's
   bandwidth-knee clause applies: `dram_saturation` measures the knee at **~16 cores @ 190.9 GB/s**
   (16→64 buys +1.5 %), so capping `G` at the knee for low-work-per-core regimes should cut dispatch
   overhead. Predicted: up to ~2× on `d_tall_narrow`, no change on `a_square`.
3. **Path-B compute floor.** Sharded regimes are compute+sync-bound with a 0.69 µs sync floor out of
   1.37 µs. `ReconfigureRegisterDatatypeMode::NoReconfigure` is already selected on the no-cast path;
   the next lever is `InitUninitMode` amortisation across back-to-back calls, or dropping the reader/
   writer kernels entirely on Path B (compute-only program, as `zero_copy_fold` does with `fold=1`).
4. **One-sided aliasing for the crossover paths (R3b/R3c)** — see the advisory deviation above.
5. **B8 trid double-issue / B13 `set_state`** — untried; 32 same-shape reads per block is exactly
   B13's use case. Both are B0 per-core-overhead levers, so counterfactual them on `d_tall_narrow` and
   `f_sharded_small`, not only on `a_square`.

---

## Phase 0 — verification (2026-07-29, `incremental-verifier`)

### What was done

Code review + golden/verifier CLI run + precision baseline + extended coverage on the Phase-0
implementation above. Deliverables: `verification_report.md`, `verifier_report.json`,
`op_requirements.md` (perf-only queue: 4 measured levers + the run-closing Mode-D audit), this entry.

### SUPPORTED at Phase 0 (after the verification fix below)

```
dtype:        [bfloat16, float32, uint32, uint16, int32]
output_dtype: [bfloat16, float32, bfloat8_b, uint32, uint16, int32]
use_multicore: [False, True]      double_buffer: [False, True]
shard_api:    [none, legacy_2d, nd]
out_scheme:   [interleaved, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, nd]
buffer:       [dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram]
rank:         [2, 3, 4, 5, 6]     EXCLUSIONS: []
```

`TARGET − SUPPORTED = ∅` on every axis, so the refinement queue carries **no** capability entries.

### Golden suite / verifier CLI at Phase 0

`eval/eval_test_runner.sh eval/golden_tests/tilize/` + `python3 -m eval.verify_supported`:

| category | count |
|---|---|
| `supported_pass` | **126 / 126** |
| `invalid_skipped` | 90 |
| `xfail_expected` | 0 (nothing left to refuse) |
| `supported_fail` / `xpass_drift` / `xfail_wrong_mode` | **0 / 0 / 0** |
| `no_axes_found` | 420 (the non-registry reference/translated/regression files) |

Whole directory: 515 passed, 118 skipped, 1 failed, 2 errors, 0 hangs. The 3 non-green are all
reference-file issues outside the registry matrix (an L1 shard grid derived from
`dram_grid_size().x = 12` on an 8-column compute grid; two `use_module_device` × `device_params`
collection errors) — diagnosed in `verification_report.md`.

### Accuracy achieved (precision baseline)

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py` — 4 shapes × 5 transitions,
20/20 pass:

| Transition | PCC | max_abs | mean_abs | rel RMS | mismatching elements |
|---|---|---|---|---|---|
| bf16→bf16, fp32→fp32, bf16→fp32 | 1.0 | 0 | 0 | 0 | **0** on all 4 shapes (bit-exact) |
| fp32→bf16 | 1.0 | 1.562e-02 (1 ULP) | ≤1.8e-07 | ≤5.0e-05 | ≤3 / 196 608 (packer tie-rounding vs torch RNE) |
| bf16→bf8b | ≥0.99 | 4.688e-02 | 7.1e-03 | 9.3e-03 (≈2⁻⁷) | ~80 % (block-float quantization) |

Integer passthrough (uint32/uint16/int32) is bit-exact (`torch.equal`).

### Verification fixes (all measured / tested, no regression)

1. **Writer: per-block `noc_async_write_barrier()` → `noc_async_writes_flushed()` + one trailing
   barrier.** Recycling a CB page needs the writes to have *departed*, not completed
   (`dataflow_api.h:1802`). Measured: `c_single_core` **31 465 → 30 472 ns (−3.2 %)**,
   `x_wide_short_1core` **61 526 → 57 459 ns (−6.6 %)**; every 64-core regime inside the ±2 % noise
   floor; no regression anywhere. Verdict **KEEP** — the payoff is on the regimes where the barrier is
   not hidden behind other cores' traffic, which is exactly the B0 "gate per-core overhead on the
   smallest regime" story.
2. **Planner: the depth-2 → depth-1 L1 fallback now happens at planner step 2** (before the chunk-width
   choice) and the old post-hoc clamp — which could never fire, because `chunk_wt ≤ max_chunk_l1`
   already bounded the footprint — became an explicit invariant `assert`. Behaviour identical; the
   "auto-fall-back rather than OOM" rule is now expressed where it is claimed.
3. **`SUPPORTED["rank"]` [2,3,4] → [2,3,4,5,6]** — the fold is rank-agnostic, so a rank-5 caller was
   being refused by a contract the kernel already satisfies. Verified bit-exact (rank 5 and 6,
   single- and multi-core).
4. **`PROPERTIES["bounded_cb"]` `declared` → `verified`**, now backed by asserts (below).
5. **Bench: A0 active-core count and bounded-CB are `assert`ed, not printed**
   (`_bench_tilize.py::_assert_structural_gates`) — the design and this changelog both claimed the A0
   gate was machine-checked; it was a print column, so a height-only-split regression on the
   wide-short regime would not have failed the bench.
6. **Bench: 3 new regimes** (`g_dram_to_sharded`, `g_sharded_to_dram`, `e_square_fp32_to_bf16`) so the
   crossover paths and the narrowing-cast path have a measured baseline for the queue's Refinement 3
   and 4 gates. `g_sharded_to_dram` immediately exposes `chunk_wt = 2` ⇒ 128 B read transactions, 4×
   under the B6 512 B one-packet threshold.
7. Naming/clarity: plan key `shard_row_bytes` → `source_page_bytes` (it is a page size, and matches the
   reader's CT arg name); reader's magic `32` → `constexpr tile_height`.

Verified by probe (not assumed): **program-cache re-binding on the zero-copy path** — two consecutive
same-spec sharded calls hit the cache (entries 1 → 1) with different input/output buffer addresses,
both bit-exact, first result untouched. `apply_descriptor_runtime_args` re-patches the aliased CB
address from `cb_desc.tensor` every call (`program_descriptors.cpp:198-209`).

### Perf re-measure — cumulative bench set (carry forward; this is the non-regression baseline)

Median of 5 rounds × 10 launches, warm-up discarded, all CV ≤ 1.5 %. WH B0, 8×8, AICLK ≈ 985 MHz.
Includes verification fix #1.

| regime | shape | dtype | cores | chunk | cbB/core | ns (median) | GB/s | vs Phase 0 |
|---|---|---|---|---|---|---|---|---|
| a_square | `[1,1,2048,2048]` | bf16 | 64 | 16 | 131 072 | **85 417** | 196.4 | −0.2 % |
| b_wide_short | `[1,1,32,16384]` | bf16 | 64 | 8 | 65 536 | **13 383** | 156.7 | −0.2 % |
| c_single_core | `[1,1,512,512]` | bf16 | 1 | 16 | 131 072 | **30 472** | 34.4 | **−3.2 %** |
| d_tall_narrow | `[1,1,2048,32]` | bf16 | 64 | 1 | 8 192 | **3 658** | 71.7 | −0.2 % |
| e_square_fp32 | `[1,1,2048,2048]` | fp32 | 64 | 8 | 131 072 | **182 877** | 183.5 | +0.4 % (noise) |
| e_square_bf8b_out | `[1,1,2048,2048]` | bf16→bf8b | 64 | 16 | 100 352 | **64 373** | 199.5 | +0.2 % (noise) |
| e_square_fp32_to_bf16 | `[1,1,2048,2048]` | fp32→bf16 | 64 | 8 | 98 304 | **120 813** | 208.3 | *new* |
| f_sharded_small | `[1,1,512,64]` H-shard | bf16 | 4 | 2 | 32 768 | **1 382** | 0 DRAM | −0.4 % |
| f_sharded_large | `[1,1,2048,512]` B-shard | bf16 | 64 | 2 | 65 536 | **2 071** | 0 DRAM | 0 |
| g_dram_to_sharded | `[1,1,2048,512]` → B-shard | bf16 | 64 | 16 | 131 072 | **18 923** | 221.6* | *new* |
| g_sharded_to_dram | `[1,1,2048,512]` B-shard → | bf16 | 64 | 2 | 16 384 | **19 780** | 212.1* | *new* |
| x_square_depth1 | `[1,1,2048,2048]` d1 | bf16 | 64 | 16 | 65 536 | **85 806** | 195.5 | 0 |
| x_wide_short_1core | `[1,1,32,16384]` 1c | bf16 | 1 | 16 | 131 072 | **57 459** | 36.5 | **−6.6 %** |
| x_sharded_small_depth1 | `[1,1,512,64]` d1 | bf16 | 4 | 2 | 32 768 | **1 364** | 0 DRAM | +0.4 % |

\* the `g_*` GB/s counts bytes **moved**, not DRAM bytes — one side of a crossover lives in L1.

Achieved-vs-target, recomputed on these numbers (DRAM spec peak 288 GB/s; the operationally meaningful
ceiling for the 64-core interleaved regimes is the in-tree measured DRAM→DRAM copy, 86.6 µs @
193.8 GB/s for 16.78 MB):

| regime | binding reference | achieved | reading |
|---|---|---|---|
| a_square | in-tree 64-core copy 86.6 µs | **1.01×** | **at the ceiling** — residual is interleaved congestion |
| e_square_fp32 / bf8b_out / fp32_to_bf16 | DRAM floor | 0.64 / 0.70 / 0.72 | same DM-bound ratio ⇒ the `Fp32Mode::Lossless` slow path costs nothing here |
| b_wide_short | DRAM floor 7.3 µs | 0.54 | transaction/launch bound → queue R2 |
| c_single_core | 1-core tt-npe pin 25.2 µs | 0.83 | improved from 0.80 by fix #1 → queue R2 |
| d_tall_narrow | knee 190.9 GB/s | 0.38, 33 % sync floor | per-core overhead → queue R1 |
| g_dram_to_sharded | DRAM-side leg 7.3 µs | 0.39 | sharded side pays a NoC leg it does not need → queue R3 |
| f_sharded_small | own sync floor 0.69 µs | 0.50 | 50 % compute + 50 % sync, **0 % DM** → queue R4 |

### Issues encountered (verification)

1. The design/changelog claim that the A0 gate is "machine-checkable rather than eyeballed" was not
   true of the bench — fixed (fix #5), and A0 is now also asserted from a unit test
   (`test_tilize_plan_invariants`), so the width-split property is guarded without the profiler.
2. The `use_double_buffer` auto-fallback existed but was unreachable dead code (fix #2). No behaviour
   change, but the rule it implements is now checked rather than incidental.
3. `SUPPORTED["rank"]` under-claimed (fix #3). No XPASS signal existed because the golden matrix has no
   rank-5 cell — this was found by reading the fold, then confirmed by probe.
4. The crossover and narrowing-cast regimes had no perf baseline at all, so the queue had nothing to
   gate Refinements 3/4 on (fix #6).

### Tests added (verification)

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py` — 20 cells
  (4 shapes × 5 transitions): PCC, max/mean abs error, relative RMS, exact-element count.
- `tests/ttnn/unit_tests/operations/tilize/test_tilize_extended.py` — 13 cells: high rank (5, 6),
  awkward `Wt` (5, 7, 63) and `Wt = 256 ≫ WT_CHUNK_MAX`, host-planner invariants (`chunk_wt | Wt`, A0
  core count, bounded CB, exact grid cover), and depth-1 against an **aliased** CB (the one depth-1
  case nothing else covered — the golden harness never forwards `use_double_buffer`).
- `tests/.../tilize/probes/probe_007.py` / `probe_008.py` — the rank-5/6 capability probe and the
  zero-copy program-cache re-binding probe.

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **93 passed** (60 + 13 + 20) in **both**
default and `--dev` mode.

---

## Refinement 1 — Per-core-overhead gating for the low-work-per-core regimes (A0 knee + B0 + depth-2 default)

- **Date**: 2026-07-29
- **Outcome**: **partial (`[~]`)**. Lever 2 (C16 depth-2 default gating) landed and pays. Lever 1
  (the A0 bandwidth-knee core cap) was implemented, measured, and **refuted on its own target
  regime** — it is 2.4× *slower*, so it was not shipped. The refinement's numeric gate
  (`d_tall_narrow` ≥ 1.5× faster) is therefore **not met** and is shown to be unreachable with
  this refinement's lever set; the blocker is characterised to the ns below and handed to
  **Refinement 1b**.

### What was done

**1. A0 bandwidth-knee clause — implemented, measured, DROPPED (negative result).**

The clause proposed capping the grid at the `dram_saturation` knee (~16 cores @ 190.9 GB/s) for
low-work-per-core shapes, predicting "up to ~2×" on `d_tall_narrow`. Implemented as a
`distribution_gate` (fire only when each core would own < 4 tiles) and swept on device with the
kernels byte-identical across every point (`probes/probe_009.py`, `CORE_CAP_OVERRIDE` hook):

| forced cap | cores | blk/core | ns (median 5×10) | vs 64 cores |
|---|---|---|---|---|
| **64 (none)** | **64** | **1** | **3 623** | **1.00×** |
| 32 | 32 | 2 | 5 186 | 0.70× |
| **16 (the knee)** | **16** | **4** | **8 580** | **0.42×** |
| 8 | 8 | 8 | 14 780 | 0.25× |
| 4 | 4 | 16 | 27 950 | 0.13× |
| 1 | 1 | 64 | 107 561 | 0.03× |

Latency is ~linear in tiles-per-core (`ns ≈ 2060 + 1563 × blocks`), i.e. **capping cores is a
2.4× regression, not a 2× win**. `n_tall_narrow2` `[1,1,2048,64]` behaves the same (0.74× at the
knee). Two independently measured reasons the bandwidth knee cannot bind here:

1. **The shape cannot reach the knee's bandwidth at any core count.** A `W=32` bf16 ROW_MAJOR input
   has **64 B** DRAM pages, so the reader issues 64 B transactions. `noc_estimate` puts 64 B
   interleaved-DRAM reads at **0.68–1.41 B/cyc/core** = 45–90 GB/s aggregate over 64 cores, so the
   190.9 GB/s knee — measured on a *large-transaction* copy — is unreachable for this shape. The op
   is **read-transaction-rate bound, not DRAM-bandwidth bound**, and the knee is a bandwidth
   phenomenon. Achieved-vs-ceiling recomputed against the *right* bound: the per-core serialized NoC
   bracket (1 block ⇒ no read/write overlap) is `[1452+1440 … 3021+1440] = [2892 … 4461] ns`, and the
   measured 3 623 ns **sits inside its own bracket** — there is no core-count headroom to recover.
   The verifier's "0.38 of the knee" compared this shape to a ceiling its page size forbids.
2. **The premise "dispatch/sync cost scales with the core count" is false for this op.** The
   `sync_only` ablation scales with **blocks per core**, not cores (`probe_010.py`):
   64 c/1 blk **1 202 ns** · 32 c/2 blk 1 818 · 16 c/4 blk 3 079 · 8 c/8 blk 5 615 · 4 c/16 blk
   10 677 ≈ `590 + 612 × blocks`. Shedding cores *adds* sync cost.

Kept as the declared criterion (`a0_active_cores` = `min(grid, total_tiles, A0_KNEE_CORES)`) with
`A0_KNEE_CORES = 64` — the measured knee for tilize's transfer shapes is the whole grid. The bench
assert is **updated to that gated form** (not deleted) and the counterfactual is a permanent bench
row (`x_tall_narrow_16c`), so the verdict is re-measurable instead of a claim: 8 846 ns vs 3 609 ns
in the same run.

**2. C16 depth-2 default gating — landed, and it pays.**

`use_double_buffer` default `True` → **`None` = "the planner decides"**. `True`/`False` keep their
documented force-depth-2 / force-depth-1 meaning, so the public kwarg keeps both values and only the
*default* is gated; `SUPPORTED["double_buffer"]` is now three-valued (`[False, True, "auto"]`) and
`validate()` maps `None → "auto"`. Depth-2 buys read/write overlap across a **block boundary**, so
the gate keys on what that overlap is worth (in-run A/B, 7 rounds, CV ≤ 1.2 %):

| ncores | blk/core | depth1/depth2 | regimes | gate |
|---|---|---|---|---|
| any | **1** | 0.995 – 1.010 | b_wide_short, d_tall_narrow, g_dram_to_sharded, n_tiny | **depth-1** (structurally inert — no boundary to overlap) |
| **1** (< knee) | 16 / 32 | **1.321 / 1.360** | c_single_core, x_wide_short_1core | **depth-2** (the core's own NoC issue rate is the bound) |
| 64 (≥ knee) | **4** | 0.998 / 1.005 | a_square, e_square_bf8b_out | **depth-1** (free) |
| 64 (≥ knee) | **8** | **1.019 / 1.023 / 1.028** | e_square_fp32_to_bf16, e_square_fp32, g_sharded_to_dram | **depth-2** (~2 %, 3 independent regimes, same sign) |

⇒ `depth2_pays = blocks ≥ 2 and (ncores < 16 or blocks > 4)`. **This is narrower than the
refinement proposed** ("default off once the op is DRAM-saturated with large per-core work"):
measurement says *large* per-core work is exactly where the residual overlap still pays. The first
implementation used the proposed rule and produced a real +2.0 / +2.3 / +2.4 % regression on those
three regimes; the blocks-per-core term removes it (all now within ±0.9 %).

The gate also **pins `chunk_wt` to the depth-2 plan's**, so the gated plan differs from the ungated
one in exactly one way — the CB has half the pages. Letting the freed L1 grow the chunk instead
changed the reader's transaction size behind the caller's back and measured a **1.3 % loss** on
`e_square_fp32` (chunk 8 → 16) while saving **zero** L1. Non-regression is now structural, not just
measured (`test_gate_never_changes_the_transaction_shape`).

### Accuracy achieved

Unchanged and **bit-exact** — this refinement moves no arithmetic. Verified `torch.equal` at the
gated default and at both forced depths, on a wide and a narrow shape, single- and multi-core, and on
the aliased zero-copy path: PCC = 1.0, rtol = atol = 0, mismatching elements = 0 on
`[1,1,64,2048]`, `[1,1,2048,32]`, `[1,1,128,256]`, `[1,1,512,64]` H-sharded. bf8b / narrowing-cast
tolerances are untouched (see the Phase-0 precision baseline, still 20/20).

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged, 0 xfail / xpass / drift. Whole golden
dir minus `test_translated.py`: **240 passed, 118 skipped, 0 failed**, byte-identical to the Phase-0
non-translated baseline (`test_golden` 126 + `test_regression` 9 + `test_golden_main_tests` 105). The
2 collection ERRORs are the pre-existing `use_module_device` × `device_params` conflict inside the
reference file (Phase-0 issue 6's sibling), not an op gap. **No hangs** in either mode.

### Perf gate

Bound classification (re-ablated this phase, `d_tall_narrow`, the target regime): **DM- and
per-block-sync-bound**, no compute headroom. Decomposition of the 3 609 ns, each term measured by
subtraction (the address-gen term needed a temporary reader patch, reverted):

| term | ns | share |
|---|---|---|
| bare launch + CB handshakes (1 block) | **764** | 21 % |
| 32 × `accessor.get_noc_addr` | **437** | 12 % |
| 32 × 64 B `noc_async_read` issue + DRAM service + barrier + 1 × 2048 B write | **1 504** | 42 % |
| tilize LLK (1 tile) | **931** | 26 % |

Ceiling: per-core serialized NoC bracket `[2892 … 4461] ns` (1 block ⇒ read and write cannot
overlap); DRAM floor 910 ns. **`op_target` = [2892 … 4461], measured 3 609 ⇒ achieved 0.80 against
the optimistic end and 1.23 against the contended end — the op sits inside its own bracket.** The
1.5× gate (≤ 2 439 ns) requires removing 1 197 ns, i.e. *all* of the address-gen plus ~half the read
issue/service. No lever in this refinement's set (A0 / B0 / C16) touches either term — see
Refinement 1b.

#### Cumulative bench set — non-regression (median of 7 × 10 launches, WH B0 8×8, AICLK ≈ 985 MHz)

Noise floor: two measurements of the *identical* plan in different sessions differ by 0.85 %
(`x_square_depth1`), so ±2 % remains the threshold. All CVs ≤ 1.2 %.

| regime | shape | cores | chk | **d** | blk | Phase-0 ns | **now ns** | Δ | Phase-0 cbB | **now cbB** | **L1 saved** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a_square | `[1,1,2048,2048]` | 64 | 16 | **1** | 4 | 85 417 | **85 535** | +0.1 % | 131 072 | **65 536** | **−65 536** |
| b_wide_short | `[1,1,32,16384]` | 64 | 8 | **1** | 1 | 13 383 | **13 447** | +0.5 % | 65 536 | **32 768** | **−32 768** |
| c_single_core | `[1,1,512,512]` | 1 | 16 | 2 | 16 | 30 472 | **30 464** | −0.0 % | 131 072 | 131 072 | 0 |
| d_tall_narrow | `[1,1,2048,32]` | 64 | 1 | **1** | 1 | 3 658 | **3 609** | **−1.3 %** | 8 192 | **4 096** | **−4 096** |
| e_square_fp32 | `[1,1,2048,2048]` fp32 | 64 | 8 | 2 | 8 | 182 877 | **181 811** | −0.6 % | 131 072 | 131 072 | 0 |
| e_square_bf8b_out | `[1,1,2048,2048]`→bf8b | 64 | 16 | **1** | 4 | 64 373 | **64 577** | +0.3 % | 100 352 | **50 176** | **−50 176** |
| e_square_fp32_to_bf16 | `[1,1,2048,2048]` fp32→bf16 | 64 | 8 | 2 | 8 | 120 813 | **121 032** | +0.2 % | 98 304 | 98 304 | 0 |
| f_sharded_small | `[1,1,512,64]` H-shard | 4 | 2 | 1 | 4 | 1 382 | **1 376** | −0.4 % | 32 768 | 32 768 | 0 |
| f_sharded_large | `[1,1,2048,512]` B-shard | 64 | 2 | 1 | 8 | 2 071 | **2 073** | +0.1 % | 65 536 | 65 536 | 0 |
| g_dram_to_sharded | `[1,1,2048,512]`→B-shard | 64 | 16 | **1** | 1 | 18 923 | **19 072** | +0.8 % | 131 072 | **65 536** | **−65 536** |
| g_sharded_to_dram | `[1,1,2048,512]` B-shard→ | 64 | 2 | 2 | 8 | 19 780 | **19 615** | −0.8 % | 16 384 | 16 384 | 0 |
| x_square_depth1 | forced d1 | 64 | 16 | 1 | 4 | 85 806 | **85 656** | −0.2 % | 65 536 | 65 536 | — |
| x_wide_short_1core | 1 core | 1 | 16 | 2 | 32 | 57 459 | **57 585** | +0.2 % | 131 072 | 131 072 | — |
| x_sharded_small_depth1 | alias, forced d1 | 4 | 2 | 1 | 4 | 1 364 | **1 367** | +0.2 % | 32 768 | 32 768 | — |

**Zero regressions** — every prior bench shape is within ±1.3 %, well inside the noise floor. Net
result: **−65 536 B/core of L1 on the two widest interleaved regimes** (a_square, g_dram_to_sharded),
−50 176 on bf8b, −32 768 on b_wide_short, −4 096 on d_tall_narrow, at no measured perf cost — while
the four regimes where depth-2 was measured to pay keep it.

New counterfactual rows added to the cumulative set for future phases: `x_square_depth2`,
`x_tall_narrow_depth2` (3 635 ns — depth-1 is 0.7 % *faster* here), `x_single_core_depth1`
(40 035 ns — the +32 % that clause 2 protects), `x_square_fp32_depth2`, `x_fp32_to_bf16_depth2`,
`x_sharded_to_dram_depth2`, `x_square_bf8b_depth2`, `x_tall_narrow_16c` (8 846 ns — the A0 refutation).

#### Mode-C used-optimization ledger (this refinement's two levers)

| lever | id | predicted Δ | **measured Δ** | verdict |
|---|---|---|---|---|
| A0 bandwidth-knee core cap on low-work-per-core shapes | A0 | up to **~2×** on `d_tall_narrow` (verifier) | **0.42×** (3 623 → 8 580 ns @ 16 cores); `n_tall_narrow2` 0.74× | **DROP** — refuted. The knee is a *bandwidth* phenomenon; this op is read-transaction-rate bound at 64 B/page and cannot reach the knee's bandwidth at any core count. Kept as the declared `min(grid, total_tiles, knee)` criterion with knee = full grid; counterfactual retained as `x_tall_narrow_16c`. |
| C16 depth-2 default → gated (`use_double_buffer=None`) | C16 / B0 | "no perf change, −65 536 B/core on wide tensors" | **1 blk: 0.995–1.010 · 4 blk: 0.998/1.005 · 8 blk: 1.019–1.028 · 1 core: 1.321/1.360.** Shipped gate: 0 regressions, **−65 536 B/core** on a_square + g_dram_to_sharded, −50 176 bf8b, −32 768 b_wide_short, −4 096 d_tall_narrow | **KEEP (gated)** — but narrower than proposed: the blocks-per-core term is required, the proposed rule regressed 3 regimes by ~2 %. |
| — sub-lever: let depth-1 spend the freed L1 on a wider chunk | B5/B6 | bigger transactions ⇒ faster | **0.987×** on `e_square_fp32` (chunk 8→16) with **0 B** L1 saved | **DROP** — chunk pinned instead. |

### Issues encountered

1. **The refinement's lever 1 is wrong, and the way it is wrong is instructive.** Both of its stated
   premises are individually falsifiable on device: the target regime is not bandwidth-bound (so the
   bandwidth knee cannot apply), and the sync floor scales with blocks-per-core rather than cores (so
   shedding cores costs sync time instead of saving it). Implementing it first and measuring it is
   what surfaced that — the Mode-C counterfactual is the check that caught a lever the queue was
   confident about.
2. **The C16 gate as proposed regressed three regimes.** "Default off once the op is DRAM-saturated
   with large per-core work" is measurably backwards on the *large*-work end: at 8 block boundaries
   per core depth-2 is still worth ~2 %. Caught only because the non-regression table re-measures the
   **whole** cumulative set rather than the shape being tuned — `a_square` (4 blocks) is clean and
   would have hidden it. Resolved with the blocks-per-core term, then re-verified by in-run A/B
   pairs, since a ~2 % effect is unresolvable against the 0.85 % cross-session scatter.
3. **`ttnn-static-analyzer` cleared the depth-1 default and corrected two queued assumptions.** Zero
   correctness defects; both helper guards (`tilize_helpers_dataflow.inl:107`,
   `tilize_helpers.inl:207/209`) hold with *exact equality* at depth 1, and the reserve/pop cycle has
   no dependency cycle, on both the generic and the aliased path. It also found that
   **`InitUninitMode::InitAndUninit` already sits outside the `num_blocks` loop**
   (`tilize_helpers.inl:179-200`, `:258-272`), so **Refinement 4's "InitUninitMode amortization"
   lever has zero headroom inside a single `tilize()` call**, and that `tilize_uninit` cannot be
   dropped (`_llk_unpack_tilize_init_` leaves `tileize_mode=1` + `shift_amount` in `THCON_SEC0`,
   `llk_unpack_tilize.h:97-108`). Two advisories for the queue: `WaitMode::WaitUpfront` is now a
   **guaranteed hang** for any core with `num_blocks > 1` at depth 1, and a C7 split reader must not
   have both NCRISC and BRISC pushing `cb_rm_input` (single-producer violation).
4. **The 1.5× gate on `d_tall_narrow` is unreachable from this refinement's lever set** — see the
   decomposition above. Not silenced: the regime stays in the bench at its measured 3 609 ns and
   Refinement 1b names the exact levers with their priced prizes.

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement1.py` — **26 cells**: the A0 knee
  term is identity (and *why*, with the sweep in the docstring, so lowering the constant fails here);
  A0 active-core criterion per regime (tall-narrow / wide-short / wide); `use_multicore=False` is
  still exactly 1 core; every branch of `depth2_pays` pinned to the ratio that set it; the gate picks
  depth-1 when DRAM-saturated, depth-2 below the knee, depth-2 past 4 block boundaries, depth-1 at 1
  block/core; **the gate never changes the transaction shape** (bf16 wide/narrow + the fp32 case that
  motivated the chunk pin); bit-exactness at the gated default and both forced depths; the gate is
  inert on the zero-copy path; the axis declares `"auto"`; `validate()` accepts all three requests;
  per-core CB bytes recorded for a wide and a narrow shape.
- `_bench_tilize.py` — A0 assert updated to the **gated** criterion `min(grid, total_tiles,
  A0_KNEE_CORES)` plus a new C16-gate assert (the planner's depth must match the declared gate);
  `depth` and `blk/core` columns; a `core_cap` spec key driving the planner's sweep hook; 8 new
  counterfactual regimes (above). The bench now measures the op's **gated default**
  (`use_double_buffer=None`), i.e. what a caller actually gets.
- `test_tilize_extended.py::test_tilize_plan_invariants` — A0 assert updated to the gated form.
- `probes/probe_009.py` (core-cap knee sweep), `probes/probe_010.py` (sync-floor vs core count +
  depth A/B) — the two measurements the ledger rests on.

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **119 passed** (93 prior + 26 new) in
**both** default and `--dev` mode.

### Ranked follow-ups

1. **Refinement 1b (filed)** — `d_tall_narrow`'s remaining prize, priced: 437 ns of address-gen +
   part of 1 504 ns of read issue/service.
2. **Refinement 4's `InitUninitMode` lever has no headroom** (finding 3). Re-scope it before running:
   the only form left is chaining `InitOnly/Neither/UninitOnly` across *multiple* `tilize()` calls,
   which this kernel does not have.
3. **`DEPTH1_MAX_BLOCKS_PER_CORE = 4` is the conservative end of an unmeasured gap** (4 free, 8
   costs ~2 %; 5–7 unmeasured). Worth 3 bench points if a future phase wants the extra L1 back.

---

## Refinement 1b — Per-core-overhead gating for the low-work-per-core regimes (A0 knee + B0 + depth-2 default) (debug: fix gate violations)

- **Date**: 2026-07-29
- **Outcome**: **full (`[x]`)**. The harness completion gate now passes all three bullets, verified by
  running `eval/run_refinements.py::evaluate_completion` itself against a fresh full-suite run:
  `bullets_pass = [1, 2, 3]`, `all_pass = True`, `responsible = 126/216`, `regression = False`.

### What was done

The reported violation was:

```
Bullet 3 FAIL: golden responsible cells 126/216 below majority threshold.
```

**It was not a kernel, hang, correctness, or coverage defect — it was a registry-*declaration*
defect, and the diagnosis is worth stating precisely because the symptom pointed somewhere else.**

Three facts, each read straight off the two phases' artifacts rather than inferred:

1. **Nothing regressed.** `golden_phase0/test_results.json` vs `golden_refinement_1/test_results.json`:
   240 passing nodeids each, set-difference **∅** in both directions. `HANGS=0` in both.
2. **The ratio did not move either.** 126/216 is *byte-identical* to Phase 0's. The denominator
   includes the **90 INVALID-skipped** cells (the harness counts a cell as "responsible" when
   `feature_matrix.is_supported()` accepts its axes, and a `skipped` status is not `passed`), so
   126/216 = **0.583** is simply what this op's golden matrix yields. A perf-only refinement — one that
   unlocks no cell by construction — cannot move it at all.
3. **What moved was the threshold.** `registry_snapshot.json` diff between the two phases is *exactly*
   one line: `+"auto"` in `SUPPORTED["double_buffer"]`. That makes
   `run_refinements.py::_supported_grew()` classify the phase as a **cartesian expansion**, which
   swaps the bullet-3 bar from `GOLDEN_MAJORITY_FIX = 0.50` to `GOLDEN_MAJORITY_EXPANSION = 0.75`.
   0.583 clears 0.50 and fails 0.75.

Reproduced mechanically before changing a line — the same snapshot, with and without the value:

| `SUPPORTED["double_buffer"]` | `_supported_grew` | threshold | ratio | bullet 3 |
|---|---|---|---|---|
| `[False, True, "auto"]` (what shipped) | True | 0.75 | 0.583 | **FAIL** |
| `[False, True]` (the fix) | False | 0.50 | 0.583 | **PASS** |

The `"auto"` value was also **dead weight**: `tag_double_buffer` projects the golden scenario dict onto
a bool, so across all 216 registry cells the axis only ever takes `True` (192) or `False` (24) — never
`"auto"`. It bought zero coverage and cost the gate.

**Fix — `None` is a request shape, not a capability.**

- `SUPPORTED["double_buffer"]` back to `[False, True]`, which is exactly the golden `TARGET`.
- `validate()` now resolves the delegated request through `_double_buffer_axis_values()`: `True`/`False`
  pin the depth; `None` yields *both* depths the planner may pick, and the SUPPORTED + EXCLUSIONS
  checks run for each. So `None` stays **gated** (not waved through) without a sentinel axis value —
  and if a future refinement ever excludes a depth, the delegated request is refused automatically.
- The C16 lever itself is **untouched**. `tilize_program_descriptor.py` and all three kernels are
  **byte-identical** to the parent commit (`git diff --stat 9b85763094 HEAD` on those paths is empty),
  so the shipped depth decisions and the −65 536 B/core L1 win are unchanged by construction.

**Generalisable rule this pins** (now a test, not a comment): *only declare an axis value the op can be
asked for and a golden cell can take. Gating a **default** is never a new axis value.* A "let the
planner decide" sentinel is neither — it is the absence of a request.

### Accuracy achieved

Unchanged and **bit-exact** — no arithmetic, no kernel, and no plan changed this pass. PCC = 1.0,
rtol = atol = 0, mismatching elements = 0 at the gated default and at both forced depths on
`[1,1,64,2048]`, `[1,1,2048,32]`, `[1,1,128,256]` and `[1,1,512,64]` H-sharded (`torch.equal`).

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged. Full suite, whole directory minus
`test_translated.py`, **run to completion with no `-k` filter**:
`PASSED=240 FAILED=0 ERRORS=2 SKIPPED=118 HANGS=0 TOTAL=360` — byte-identical to both the Phase-0 and
the Refinement-1 baselines. Verifier CLI: `supported_pass=126, supported_fail=0, xfail_expected=0,
xpass_drift=0, xfail_wrong_mode=0, invalid_unexpected=0` — i.e. **removing `"auto"` introduced no
drift**, confirming the value was unobservable to the matrix. The 2 collection ERRORs are the
pre-existing `use_module_device` × `device_params` conflict inside the reference file (Phase-0 issue 6's
sibling), not an op gap.

### Perf gate

**Bound classification carries forward unchanged, and that is provable rather than assumed**: the
device program is byte-identical (planner + all three kernels unmodified), so the Refinement-1
ablation verdict still holds — `d_tall_narrow` DM- and per-block-sync-bound, `a_square` DM-bound at
1.01× the in-tree 64-core DRAM→DRAM copy, Path B compute+sync-bound with zero DM. The only code that
changed runs on the host, once per call, *before* the program is built. Re-ablating an identical
program would measure the same thing twice; the non-regression measurement below is the check that
matters, and it was run in full.

DM lever checklist (`master.md` Part 2): no data-path change this pass, so nothing new to apply or
defer. The outstanding levers remain **B13 `set_state`** and **C7 split reader** on the sub-one-packet
read path, priced to the ns and owned by **Refinement 1c**.

#### Cumulative bench set — non-regression (median of 5 × 10 launches, WH B0 8×8, AICLK ≈ 985 MHz)

Whole set re-measured, not a subset. Noise floor ±2 %; all CVs ≤ 1.1 %.

| regime | cores | chk | d | blk | cbB/core | R1 ns | **now ns** | Δ |
|---|---|---|---|---|---|---|---|---|
| a_square | 64 | 16 | 1 | 4 | 65 536 | 85 535 | **85 859** | +0.4 % |
| b_wide_short | 64 | 8 | 1 | 1 | 32 768 | 13 447 | **13 340** | −0.8 % |
| c_single_core | 1 | 16 | 2 | 16 | 131 072 | 30 464 | **30 478** | +0.0 % |
| d_tall_narrow | 64 | 1 | 1 | 1 | 4 096 | 3 609 | **3 598** | −0.3 % |
| e_square_fp32 | 64 | 8 | 2 | 8 | 131 072 | 181 811 | **183 003** | +0.7 % |
| e_square_bf8b_out | 64 | 16 | 1 | 4 | 50 176 | 64 577 | **64 601** | +0.0 % |
| e_square_fp32_to_bf16 | 64 | 8 | 2 | 8 | 98 304 | 121 032 | **120 982** | −0.0 % |
| f_sharded_small | 4 | 2 | 1 | 4 | 32 768 | 1 376 | **1 368** | −0.6 % |
| f_sharded_large | 64 | 2 | 1 | 8 | 65 536 | 2 073 | **2 073** | 0.0 % |
| g_dram_to_sharded | 64 | 16 | 1 | 1 | 65 536 | 19 072 | **19 117** | +0.2 % |
| g_sharded_to_dram | 64 | 2 | 2 | 8 | 16 384 | 19 615 | **19 721** | +0.5 % |
| x_square_depth1 | 64 | 16 | 1 | 4 | 65 536 | 85 656 | **85 813** | +0.2 % |
| x_square_depth2 | 64 | 16 | 2 | 4 | 131 072 | 85 887* | **85 887** | 0.0 % |
| x_tall_narrow_depth2 | 64 | 1 | 2 | 1 | 8 192 | 3 635 | **3 642** | +0.2 % |
| x_single_core_depth1 | 1 | 16 | 1 | 16 | 65 536 | 40 035 | **40 155** | +0.3 % |
| x_square_fp32_depth2 | 64 | 8 | 2 | 8 | 131 072 | 181 906* | **181 906** | 0.0 % |
| x_fp32_to_bf16_depth2 | 64 | 8 | 2 | 8 | 98 304 | 120 999* | **120 999** | 0.0 % |
| x_sharded_to_dram_depth2 | 64 | 2 | 2 | 8 | 16 384 | 19 740* | **19 740** | 0.0 % |
| x_square_bf8b_depth2 | 64 | 16 | 2 | 4 | 100 352 | 64 191* | **64 191** | 0.0 % |
| x_tall_narrow_16c | 16 | 1 | 1 | 4 | 4 096 | 8 846 | **8 866** | +0.2 % |
| x_wide_short_1core | 1 | 16 | 2 | 32 | 131 072 | 57 585 | **57 229** | −0.6 % |
| x_sharded_small_depth1 | 4 | 2 | 1 | 4 | 32 768 | 1 367 | **1 368** | +0.1 % |

\* the eight `x_*_depth2` counterfactual rows were added by Refinement 1 in this same session; the
value shown is this run's.

**Zero regressions** — max deviation **+0.7 %**, well inside the ±2 % noise floor, and the direction is
scatter, not drift (10 of 22 regimes are faster). Just as important, every **plan** column
(`cores`/`chk`/`d`/`blk`/`cbB/core`) matches Refinement 1's row for row: the gate still picks depth-1 on
`a_square` / `b_wide_short` / `d_tall_narrow` / `e_square_bf8b_out` / `g_dram_to_sharded` and depth-2 on
`c_single_core` / `e_square_fp32` / `e_square_fp32_to_bf16` / `g_sharded_to_dram` / `x_wide_short_1core`,
so the **−65 536 B/core L1 win survives intact**.

#### Mode-C ledger (this pass)

| lever | id | predicted Δ | **measured Δ** | verdict |
|---|---|---|---|---|
| drop the `"auto"` axis value (registry-declaration fix) | — | none — host-side gate only, device program byte-identical | **max ±0.7 % across all 22 bench regimes** (scatter, inside the ±2 % noise floor); every plan column unchanged | **KEEP** — the fix is perf-neutral by construction and confirmed by measurement |

### Issues encountered

1. **The failing symptom named the wrong subsystem.** "Golden responsible cells below majority" reads
   like a coverage or correctness regression; the artifacts said 240/240 prior-passing cells still
   passed and the ratio was unchanged from Phase 0. Diffing the two `registry_snapshot.json` files —
   a one-line diff — was what located it, and re-running `_supported_grew` + the threshold arithmetic
   on both snapshots *confirmed* it before any code changed. Worth generalising: when a gate fails but
   the numerator and denominator are both unchanged, **suspect the threshold, not the cells.**
2. **A well-intentioned "make the registry honest" instinct caused it.** Declaring `"auto"` looked like
   the conscientious thing to do — the op really did gain a third *request* value. But the axis models
   the CB **depth**, and there is no third depth; `None` selects between the two that already exist.
   The registry model's rule is now pinned by a test with the threshold arithmetic in its docstring,
   so the next implementer sees the cost rather than re-deriving it.
3. **Harness-filed `Refinement 1b` collided with the follow-up ID Refinement 1 had already filed.**
   `_next_sub_refinement_id` scans the phase list captured *before* the agent's edits, so it did not
   see the existing `Refinement 1b` and reused the letter. Two headings with one ID resolve
   *first-match* in `parse_phases` / `_set_phase_checkbox`, so the perf follow-up is renamed to
   **Refinement 1c** (and cross-references updated). The debug entry keeps `1b`, which is the ID the
   harness re-checks.
4. **A pre-commit hook (`prefer-expect-error`) rejects `pytest.raises` under `tests/`.** The new
   negative test uses the root `conftest.py::expect_error` fixture instead.

### Tests added

- `test_tilize_refinement1.py::test_double_buffer_axis_stays_two_valued` — replaces the inverted
  `test_double_buffer_axis_declares_auto`. Asserts `SUPPORTED["double_buffer"] == [False, True]` and
  that no value is a string sentinel of any spelling. Its docstring carries the full threshold
  arithmetic (0.50 vs 0.75, and why a perf-only refinement can never clear 0.75), so the next
  implementer who reaches for a sentinel fails here with the reason attached.
- `test_tilize_refinement1.py::test_validate_gates_every_depth_the_planner_may_pick` — proves the fix
  did not turn the delegated request into an *unchecked* one: with the axis monkeypatched to
  `[False]`, both `None` and `True` are refused with `UnsupportedAxisValue` while `False` still
  validates.

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **120 passed** (119 prior + 1 new; one test
inverted in place) in **both** default and `--dev` mode.

### Ranked follow-ups

1. **Refinement 1c** — unchanged and still the real remaining prize: `d_tall_narrow`'s 437 ns of
   address-gen (B13 `set_state`) plus a share of the 1 504 ns read issue/service (B13 + C7 split
   reader). Nothing in this pass touched it.
2. Refinements 2–5 unchanged.

---

## Refinement 1c — `d_tall_narrow` sub-one-packet read path: B13 `set_state` on the 32 stick reads, then C7 split reader

- **Date**: 2026-07-29
- **Outcome**: **full (`[x]`)** via the entry's second gate clause. Both levers were implemented,
  swept across **five read-transaction sizes × two block counts**, gated to exactly where each was
  measured to pay, and each carries its counterfactual number. `d_tall_narrow` improves
  **3 609 → 3 431 ns (−4.9 %, in-run A/B)**; the parent's **1.5× gate (≤ 2 439 ns) is NOT met and is
  shown to be unreachable** from these two levers, with the residual re-priced by the same
  subtraction method R1 used. The interesting result is *why*: the 437 ns address-gen term the entry
  was priced on is **real (re-priced 461 ns) but ~73 % hidden behind DRAM service latency**, so
  removing 20 of its 32 accessor calls buys only 78 ns end-to-end.

### What was done

**1. B13 — stateful bank-major reads. LANDED, gated to reads ≤ 128 B.**

New `kernel_lib` helper (so the reader keeps calling helpers rather than inlining raw NoC API):

| symbol | file | what |
|---|---|---|
| `StickReadMode{Generic,Stateful}` | `tilize_helpers_dataflow.hpp` | selects the read mode |
| `read_stick_rows_for_tilize<mode, num_splits>()` | `.hpp` / `.inl` | one tile-height band into a **caller-owned** L1 window, splittable across both DM RISC-Vs |
| `read_sticks_for_tilize<cb, granularity, mode>()` | `.inl` | gained the `mode` param; its TILE branch now calls the above (byte-identical for existing callers — both new template params default to the old behaviour) |

The mechanism: `set_state` pins the NoC **coordinate**, and for an interleaved tensor page `p` lives
in bank `p % num_banks` at bank-page `p / num_banks` (`dataflow_api_addrgen.h:19-42,236-244`), so
pages `g, g+nb, g+2nb, …` share one bank **one aligned page apart**. Visiting the block's rows
bank-major therefore lets one armed command cover ~32/12 rows with the source address as a running
increment: **12 accessor calls per block instead of 32, and no per-read coordinate write**. The bank
period comes from `NUM_DRAM_BANKS` / `NUM_L1_BANKS` selected by `DSpec::is_dram`, because the
interleaved `TensorAccessor` specialisation is an `InterleavedAddrGen` and has **no `dspec()` at
all** (`DSpec::num_banks_ct == 0`) — the first compile error of this pass.

The identity this rests on is **checked on device, not assumed**: an `ASSERT` in the inner loop
re-derives every row's address through `accessor.get_noc_addr` and compares it with the incremented
one, so any accessor/allocator change that breaks the assumption fails a `--dev` run instead of
silently transposing data.

**2. C7 — split reader. LANDED, gated to 64 B reads with one block per core.**

BRISC (the writer kernel) takes half of each block's 32 stick reads; it is otherwise parked in
`cb_wait_front` for the entire read window. NCRISC stays the **only** producer of `cb_rm_input`
(single-producer rule) and hands the reserved window over with two monotonic per-launch counting
semaphores, both in the core's own L1 so set/wait are plain local loads and stores:

```
NCRISC: reserve -> sem_reserve = blk+1 -> read its groups -> barrier
        -> wait sem_done >= blk+1 -> push
BRISC : wait sem_reserve >= blk+1 -> read its groups -> barrier -> sem_done = blk+1
```

`depth == 1` is a **structural** precondition (not a payoff question): BRISC never touches the CB
pointers, so the reserved window must be at the CB base address on every block. The band is split by
**bank group**, not by contiguous row range — under B13 a group is a whole bank's rows, so a row-range
split would halve the rows per arm in both halves instead of halving the number of arms.

**3. The two levers are mutually exclusive — by measurement.** C7 already halves the reads each
RISC-V issues, so B13's saved command programming is halved while its bank-major DRAM serialization
is not. Three in-run A/B pairs on `[1,1,2048,32]`: C7 alone **3 411.1 / 3 404.1 / 3 419.6 ns** vs
C7+B13 **3 462.4 / 3 431.6 / 3 434.2 ns** (+0.9 % mean, never negative). Since a lever that does not
move the number is a defect, the planner now ships exactly one:

| read bytes | blocks/core | ships | vs no lever |
|---|---|---|---|
| 64 B | 1 | **C7** | 0.948 – 0.956 |
| 64 B | ≥ 2 | **B13** | 0.950 – 0.957 |
| 128 B | any | **B13** | 0.950 – 0.968 |
| ≥ 256 B | any | neither | both measured **negative** |

### Issues encountered

1. **`noc_async_read_one_packet_with_state` hangs every core on a watcher build.** The first
   implementation used the `one_packet` flavour (it also saves the per-read length write). Production
   mode passed 153/153, but `--dev` hung with **all 64 cores' NCRISC at waypoint `NATW`** inside
   `noc_async_read_one_packet_with_state` (`dataflow_api.h:642`) — found from the triage report's
   callstacks, and bisected to the lever with `TILIZE_LEVER_B13=0` (60/60 pass) vs `TILIZE_LEVER_C7=0`
   (hang). The two APIs differ in **exactly one thing**: the `one_packet` sanitize macro
   `DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE` (`sanitize.h:781`) reads the
   transfer length back out of `NOC_AT_LEN_BE`, while the general API's `..._WITH_ADDR_STATE`
   (`sanitize.h:792`) takes the length as an argument. Switching to
   `noc_async_read_set_state` / `noc_async_read_with_state` made `--dev` green (153/153) at the same
   measured payoff, and the finding is recorded as a "do not optimize this back" note in the helper
   with the exact command that reproduces it. **This is a metalium-infrastructure bug, not an op bug**
   — worth filing upstream.
2. **The interleaved `TensorAccessor` has no `dspec()`.** It is a separate specialisation deriving
   from `InterleavedAddrGen` (`tensor_accessor.h:387`), so `accessor.dspec().num_banks()` does not
   compile and `DSpec::num_banks_ct` is 0. The bank count has to come from the firmware defines.
3. **B13 is a *bank-major-order* lever, and that is what turns it over.** Because `set_state` pins the
   coordinate, the lever cannot be applied in row-major order. At 64 B a bank's 2-3 reads are one
   packet each and the saved command programming dominates; from 256 B up, queueing them
   back-to-back on one DRAM endpoint costs more than it saves (+2.3 % at 256 B, **+19.1 % at 512 B**).
   The verifier's framing ("32 same-shape reads per block is exactly this lever's use case") is true
   only in the sub-one-packet corner.
4. **C7 spends the read/write overlap across the block boundary.** BRISC's read of block i+1 queues
   behind its write of block i, so the split is free only at one block per core: 0.956 at 1 block vs
   **1.145 at 4 blocks** (forced). It also doubles the read issuers per core, which only helps while
   each core reads its **own** rows — every ≥ 128 B regime here has `nt_h == 1`, i.e. all 64 cores
   read the same 32 source pages, so a second issuer just deepens an existing DRAM hot spot (+1.4 %
   at 128 B, +13.7 % at 512 B).
5. **A self-inflicted device wedge cost ~6 runs.** Deliberately corrupting the address increment to
   check that the in-helper `ASSERT` net is live produced a hang that left the device in a state where
   `--dev` failed and production passed; two clean runs cleared it. The check itself was worth it (the
   ASSERT does halt the core), but it should be done on a single-shape probe, not on a suite.

### Accuracy achieved

Unchanged and **bit-exact** — this refinement moves no arithmetic, only the order and the command
programming of the reads. `torch.equal` (PCC = 1.0, rtol = atol = 0, 0 mismatching elements) on
`[1,1,2048,32]`, `[1,1,8192,32]`, `[1,1,32,4096]`, `[1,1,32,8192]`, `[1,1,64,32]`, `[1,1,32,32]`,
`[1,1,96,96]`, `[2,3,64,64]`, at the gated default, at both forced depths, for **all four lever
combinations** (including the forced both-on path), single- and multi-core, DRAM- and
L1-interleaved, and over 10 repeated launches. Inputs are `arange` rather than `randn` on purpose:
every element is unique, so a misplaced row — the failure mode a wrong bank-major address or a wrong
split half produces — cannot cancel out. bf8b / narrowing-cast tolerances untouched (Phase-0
precision baseline still 20/20).

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged, 0 xfail / xpass / drift. Whole golden
dir minus `test_translated.py`, run to completion with no `-k` filter:
`PASSED=240 FAILED=0 ERRORS=2 SKIPPED=118 HANGS=0 TOTAL=360` — byte-identical to the Phase-0,
Refinement-1 and Refinement-1b baselines. `eval.verify_supported`: `supported_pass=126,
supported_fail=0, xfail_expected=0, xpass_drift=0, xfail_wrong_mode=0, invalid_unexpected=0`.
`SUPPORTED` is unchanged (this is a perf refinement; it unlocks no cell and declares no axis value).
**No hangs** in either mode.

### Perf gate

#### 1. Bound classification + the re-priced decomposition (ablation, 7 × 10 launches)

| variant | no levers | **shipped (C7)** | B13 only |
|---|---|---|---|
| full | 3 602.9 | **3 437.8** | 3 525.3 |
| no_compute | 2 786.5 | 3 175.0 | 2 910.5 |
| no_dm | 1 491.6 | 972.9 | 1 492.2 |
| sync_only | 1 196.7 | **735.5** | 1 191.6 |

**Re-priced terms** (same subtraction method as Refinement 1, so directly comparable to its
437 / 764 / 1 504 / 931 ns split):

| term | R1 | **now** | how |
|---|---|---|---|
| 32 × `accessor.get_noc_addr` (address-gen) | 437 | **461.2** | `sync_only`(no levers) − `sync_only`(C7): the C7 reader's `skip_dm` drops the whole read call incl. address-gen, the baseline's keeps the 32-call loop behind a `volatile` sink |
| launch + CB handshakes (1 block) | 764 | **735.5** | `sync_only`(C7), which *includes* the new semaphore handshake |
| tilize LLK, 1 tile | 931 | **237 – 295** | `no_dm` − `sync_only` (both levers off / on) |

**The headline correction: R1's four terms are not additive.** The LLK alone is ~240-295 ns, not
931 (931 was `full − no_compute`, i.e. the cost of *serializing* compute into the chain, not the
LLK's duration). And B13 removes 20 of the 32 accessor calls — ~288 ns of the 461 ns term — for a
measured **−77.6 ns** end-to-end (3 602.9 → 3 525.3), i.e. **≈ 73 % of the address-gen term is
hidden behind DRAM service latency and is not on the critical path**. That is the reason this
entry's budget (−1 197 ns) was not reachable: the only term left large enough is the DRAM 64 B
**packet rate** (32 reads × 64 cores = 2 048 requests over 12 banks), and reducing the transaction
*count* needs a row permutation the tilize LLK cannot consume — quantified this pass: coalescing a
bank's 3 contiguous pages into one 192 B read would then need 32 local L1 gather copies per block,
~10 cycles of issue each on the only two DM RISC-Vs, i.e. **more** than the DRAM reads it saves.

Ceiling: R1's per-core serialized NoC bracket `[2 892 … 4 461] ns` (1 block ⇒ no read/write
overlap). Measured **3 430.9** ⇒ **achieved 0.84** against the optimistic end (was 0.80) and 1.30
against the contended end — the regime still sits **inside its own bracket**. DRAM floor 910 ns.

#### 2. Cumulative bench set — non-regression (median of 7 × 10 launches, WH B0 8×8, AICLK ≈ 985 MHz)

Whole set re-measured, not a subset. Noise floor ±2 %; all CVs ≤ 1.8 %.

| regime | cores | chk | d | blk | B13 | C7 | R1b ns | **now ns** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| a_square | 64 | 16 | 1 | 4 | 0 | 0 | 85 859 | **85 775** | −0.1 % |
| b_wide_short | 64 | 8 | 1 | 1 | 0 | 0 | 13 340 | **13 423** | +0.6 % |
| c_single_core | 1 | 16 | 2 | 16 | 0 | 0 | 30 478 | **30 359** | −0.4 % |
| **d_tall_narrow** | 64 | 1 | 1 | 1 | 0 | **1** | 3 598 | **3 431** | **−4.6 %** |
| e_square_fp32 | 64 | 8 | 2 | 8 | 0 | 0 | 183 003 | **182 008** | −0.5 % |
| e_square_bf8b_out | 64 | 16 | 1 | 4 | 0 | 0 | 64 601 | **64 628** | +0.0 % |
| e_square_fp32_to_bf16 | 64 | 8 | 2 | 8 | 0 | 0 | 120 982 | **121 126** | +0.1 % |
| f_sharded_small | 4 | 2 | 1 | 4 | 0 | 0 | 1 368 | **1 363** | −0.4 % |
| f_sharded_large | 64 | 2 | 1 | 8 | 0 | 0 | 2 073 | **2 071** | −0.1 % |
| g_dram_to_sharded | 64 | 16 | 1 | 1 | 0 | 0 | 19 117 | **18 964** | −0.8 % |
| g_sharded_to_dram | 64 | 2 | 2 | 8 | 0 | 0 | 19 721 | **19 841** | +0.6 % |
| x_square_depth1 | 64 | 16 | 1 | 4 | 0 | 0 | 85 813 | **85 618** | −0.2 % |
| x_square_depth2 | 64 | 16 | 2 | 4 | 0 | 0 | 85 887 | **85 610** | −0.3 % |
| x_tall_narrow_depth2 | 64 | 1 | 2 | 1 | **1** | 0 | 3 642 | **3 553** | −2.5 % |
| x_single_core_depth1 | 1 | 16 | 1 | 16 | 0 | 0 | 40 155 | **40 076** | −0.2 % |
| x_square_fp32_depth2 | 64 | 8 | 2 | 8 | 0 | 0 | 181 906 | **182 074** | +0.1 % |
| x_fp32_to_bf16_depth2 | 64 | 8 | 2 | 8 | 0 | 0 | 120 999 | **121 161** | +0.1 % |
| x_sharded_to_dram_depth2 | 64 | 2 | 2 | 8 | 0 | 0 | 19 740 | **19 546** | −1.0 % |
| x_square_bf8b_depth2 | 64 | 16 | 2 | 4 | 0 | 0 | 64 191 | **64 309** | +0.2 % |
| **x_tall_narrow_16c** | 16 | 1 | 1 | 4 | **1** | 0 | 8 866 | **8 126** | **−8.3 %** |
| x_wide_short_1core | 1 | 16 | 2 | 32 | 0 | 0 | 57 229 | **57 401** | +0.3 % |
| x_sharded_small_depth1 | 4 | 2 | 1 | 4 | 0 | 0 | 1 368 | **1 362** | −0.5 % |

**Zero regressions** — max deviation **+0.6 %**, well inside the ±2 % noise floor, and 13 of 22
regimes are faster. Two real improvements: the target regime **−4.6 %** and the A0-counterfactual row
**−8.3 %** (it has 4 blocks/core at 64 B, so it picks up B13). Per-core CB bytes are **unchanged
everywhere** — neither lever costs L1; C7 adds two 4-byte semaphores.

New rows added to the cumulative set for future phases (this phase's own regimes and
counterfactuals): `x_tall_narrow_no_levers` 3 609.4 · `x_tall_narrow_b13_only` 3 506.4 ·
`x_tall_narrow_c7_only` 3 431.1 · `n_tall_narrow_4blk` 13 227.8 · `x_tall_narrow_4blk_no_levers`
13 901.4 · `x_tall_narrow_4blk_c7_forced` 15 767.1 · `m_wide_short_4k` 4 718.6 ·
`x_wide_short_4k_no_levers` 5 031.8 · `x_wide_short_4k_c7_forced` 5 101.2 · `m_wide_short_8k`
8 067.0 · `x_wide_short_8k_b13_forced` 8 310.6 · `x_wide_short_8k_c7_forced` 8 571.2 ·
`x_wide_short_b13_forced` 15 981.0 · `x_wide_short_c7_forced` 15 265.5 · `x_square_b13_forced`
86 810.4 · `x_g_to_sharded_b13_forced` 20 107.0 · `x_g_to_sharded_c7_forced` 19 901.8.

#### 3. Mode-C used-optimization ledger

| lever | id | predicted Δ | **measured Δ (lever/none)** | verdict |
|---|---|---|---|---|
| stateful bank-major reads | **B13** | "most of the 437 ns of address-gen plus a share of the per-read command programming" | **0.978** (64 B, 1 blk) · **0.957** (64 B, 4 blk) · **0.950** (128 B) · 1.023 (256 B) · **1.199** (512 B) · 1.057 (1024 B DRAM→shard) · 1.012 (1024 B square) | **KEEP, gated ≤ 128 B.** Pays 2-5 % where the read is one small packet; a **+19.9 %** regression at 512 B because `set_state` forces bank-major order. The priced 437 ns prize is real (461 ns) but 73 % hidden behind DRAM latency ⇒ only 78 ns of it is recoverable. |
| split reader | **C7** | "halve what is left of the issue cost by putting the idle BRISC to work"; `examples/split_reader` up to 1.7× | **0.948-0.956** (64 B, 1 blk) · 1.018 (128 B) · 1.056 (256 B) · **1.146** (512 B) · 1.045 (1024 B) · **1.145** (64 B, 4 blk) | **KEEP, gated to 64 B + 1 blk/core.** The single biggest term this entry recovered (−165 ns). Turns over on two independent axes: the lost read/write overlap across a block boundary, and doubling the issuers into a shared DRAM hot spot when `nt_h == 1`. |
| — sub-lever: B13 **and** C7 together | B13+C7 | additive ("neither alone is likely to be enough") | **+0.9 %** vs C7 alone, over 3 in-run A/B pairs | **DROP the combination** — mutually exclusive in the planner. |
| — sub-lever: the `one_packet` state API | B13 | + the per-read length write | correctness: **hangs all 64 cores under `--dev`** at the same measured perf | **DROP** — use the general state API (see issue 1). |

#### 4. DM lever checklist review (`master.md` Part 2)

Applied this pass: **B13** (gated). Re-confirmed still applied: B7 one barrier per block, B5/B6
width coalescing, A0 2D split, A1 `row_wise`, B9 reads NoC0 / writes NoC1, C14 alias, C16 gated
depth, D18/D19 program-cache args. **C7** (Part 1 example `split_reader`) newly applied, gated.
Deliberately **not** applied here and left to their owners: B8 trid double-issue (R2), B10 per-reader
VC (R2), A3 DRAM-bank-adjacent placement (R2), C14 one-sided aliasing (R3), B5/B6 on the sharded read
(R3). Not applicable: bigger read transactions on this regime — a `W=32` RM input has 64 B pages that
land on *different* banks, and the coalesced alternative is measured slower (see §1).

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement1c.py` — **33 cells**: both payoff
  gates pinned to the sweep that set them (raising a threshold fails here, with the numbers in the
  docstring); the mutual-exclusion invariant on 5 read sizes; plan lever selection per read size and
  block count; `use_double_buffer=True` swaps C7 for B13 rather than corrupting the window;
  levers off on the alias path and on a `row_page_stride > 1` sharded input; **bit-exactness** on 8
  shapes at the gated default, for all four lever combinations, on the forced multi-block /
  multi-chunk split path (the only cover for the per-block sequence counter and the chunk-outer ×
  block-inner handshake order), over 10 repeated launches (semaphore re-arm), single-core, and
  L1-interleaved (the in-kernel `period * 2 <= num_rows` fallback); program-cache hit with the split
  reader's two semaphores and extra runtime arg.
- `_bench_tilize.py` — a `levers` spec key (`b13`/`c7` ∈ {0 = off, 1 = gated, **2 = force past the
  gate**}) plus `B13`/`C7` report columns, and **17 new regimes**: the 4-way lever matrix on the
  64 B target, the same at 4 blocks/core, a read-size sweep at 128 B / 256 B, and forced-lever
  counterfactuals on 512 B / 1024 B / the square. Every gate threshold in the planner now has a
  re-measurable bench row behind it.
- `probes/probe_011.py`, `probe_012.py` — the first-light correctness probes for the two levers.

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **153 passed** (120 prior + 33 new) in
**both** default and `--dev` mode.

### Ranked follow-ups

1. **`d_tall_narrow` is done from the DM side.** Of its 3 431 ns, 735 ns is the launch + CB +
   handshake floor, ~240-295 ns is the 1-tile LLK, and the rest is the DRAM **64 B packet rate**
   (2 048 requests over 12 banks). Every remaining lever needs either fewer transactions (blocked by
   the layout — quantified above) or a lower per-launch floor (Refinement 4's kernel-count reduction,
   which is the same B0 lever on a different path).
2. **Refinement 2 must not re-try B13 or C7 on `b_wide_short`** — both are measured regressions there
   (+19.1 % / +13.7 %). Annotated in `op_requirements.md` under that entry. Its remaining levers (B8
   trid double-issue, B10 VC, A3 placement) are untouched by this pass.
3. **File upstream: `noc_async_read_one_packet_with_state` + watcher = guaranteed hang** (issue 1).
   Any op adopting the one-packet stateful read will hit it, and the failure mode is a hang with no
   sanitizer message.

---

## Refinement 2 — Close the read-path transaction-overhead gap on the DM-bound interleaved regimes

- **Date**: 2026-07-29
- **Outcome**: **full (`[x]`)** via the entry's second gate clause. All three levers in this entry's
  re-scoped set were implemented and measured **individually**; **B8 landed and pays 10–19 %** on four
  bench regimes, and **B10 and A3 each carry a recorded measured-no-payoff verdict with its
  counterfactual number** (B10 is a *regression* of up to 2.1×, A3 is neutral). The `b_wide_short`
  ≥ 1.2× clause is **not** met and is now shown to be **unreachable by any per-transaction lever**:
  tt-npe pins that regime at **103 % DRAM bandwidth utilisation with a 0.4 % congestion term**, i.e. it
  is already at its achievable DRAM bound and the two congestion levers had ≤ 0.4 % available *a
  priori*. The residual is decomposed and handed to **Refinement 2b**.

### What was done

**1. B8 — trid double-issue on the read path. LANDED, gated to two measured clauses.**

The reader tags each chunk-block's 32 stick reads with one of two NoC transaction ids
(`noc_async_read_set_trid`, which writes the sticky `NOC_PACKET_TAG` on NCRISC's read command buffer)
and barriers with `noc_async_read_barrier_with_trid` on the **previous** id — so block *i+1*'s reads
are already in flight while block *i* drains, instead of the NoC request queue emptying once per
block. The pipeline is flattened over the whole `(chunk, block)` sequence, preserving the
chunk-outer / tile-row-inner order the writer and compute both assume.

This is the **read-side analogue of Phase-0 verification fix #1**, which replaced the writer's
per-block `noc_async_write_barrier()` with `noc_async_writes_flushed()`. The writer can already keep
writes in flight across a block boundary because it only needs them to have *departed*; the reader
needs the bytes *present* before it pushes, and the second trid is what buys the same overlap.

It needs a **third CB window**, and the reason is a real API limitation worth recording:
`cb_reserve_back` does **not** move the FIFO write pointer, so `get_write_ptr` keeps returning the
*current* block's window until `cb_push_back` — the reader cannot ask the CB for the next block's
address before publishing the current one. The next window is therefore computed from the CB base
(`cb_base + (block % depth) * chunk_bytes`, which is exactly where the FIFO's own pointer lands after
`depth` pushes) and its freedom is guaranteed by reserving **two** windows. At depth 2 that reserve
would demand a fully drained CB and serialise compute behind the reader, hence `depth == 3`.

`chunk_wt` is **pinned** (Refinement 1's rule: a lever may not move the transaction shape behind the
caller's back), so the third window costs 1.5× the CB L1 and the counterfactual differs in exactly one
thing. B8 also only fires on the **delegated** default: `use_double_buffer=True/False` keep their
documented "depth-2, +L1" / "depth-1, minimal L1" meanings exactly.

**2. B10 — per-reader / per-writer static unicast VC. IMPLEMENTED, MEASURED, REFUTED.**

Two mechanism findings came out of implementing it, both worth carrying:

- **`noc_async_read`'s `read_req_vc` argument is dead code in `DM_DEDICATED_NOC`** — the mode
  `ReaderConfigDescriptor` selects. `ncrisc_noc_fast_read` only writes `NOC_CTRL` under
  `DM_DYNAMIC_NOC` (`noc_nonblocking_api.h:415-437`); in dedicated mode `NOC_CTRL` is programmed once
  by `noc_init` to static VC 1 and is **sticky**. So a read-side VC has to be programmed with
  `noc_async_read_one_packet_set_state<use_vc=true>` and **restored** before the kernel exits, or the
  next program on that core inherits it (the same hazard the dram-sharded matmul reader documents).
- **The write side is different**: `ncrisc_noc_fast_write` writes `NOC_CTRL` on *every* call, so
  `noc_async_write`'s `vc` argument is live and needs no restore.

Because the two halves are programmed by different mechanisms, `vc_spread` is a **bitmask** and each
half was measured separately. That is what located the cost.

**3. A3 — bank-adjacent work-unit → core assignment. IMPLEMENTED, MEASURED, no payoff.**

`get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)` (nanobind-bound on this build) gives DRAM
bank *i*'s NoC-optimal worker; those cores are taken first, in bank order, and the rest fill in A1's
row-major order. On a full grid the core **set** and the `CoreRangeSet` are byte-identical to the
default, so only the work→core *mapping* moves and the bench row measures the permutation alone.

### Accuracy achieved

Unchanged and **bit-exact** — this refinement moves no arithmetic, only the order and command
programming of the reads. `torch.equal` (PCC = 1.0, rtol = atol = 0, 0 mismatching elements) on
`[1,1,512,512]`, `[1,1,128,1024]`, `[1,1,8192,32]`, `[1,1,4096,64]`, `[1,1,4096,32]`, `[1,1,192,96]`,
`[2,3,128,64]`, `[1,1,32,16384]`, `[1,1,128,256]`, `[1,1,96,96]` — at the gated default, at both forced
depths, for **all five lever combinations** (including all-forced), single- and multi-core, DRAM- and
L1-interleaved, and over 10 repeated launches. Inputs are `arange`, not `randn`: every element is
unique, so a block written into the wrong prefetch window cannot cancel out. bf8b / narrowing-cast
tolerances untouched (Phase-0 precision baseline still 20/20).

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged, 0 xfail / xpass / drift. Whole golden
dir **run to completion with no `-k` filter**: `PASSED=515 FAILED=1 ERRORS=2 SKIPPED=118 HANGS=0
TOTAL=636`; the non-`test_translated.py` subset is **240 passed, 0 failed**, byte-identical to the
Phase-0 / R1 / R1b / R1c baselines. `eval.verify_supported`: `supported_pass=126, supported_fail=0,
xfail_expected=0, xpass_drift=0, xfail_wrong_mode=0, invalid_unexpected=0`. `SUPPORTED` is unchanged
(this is a perf refinement; it unlocks no cell and declares no axis value). The 1 failure and 2 errors
are the pre-existing reference-file issues (Phase-0 issues 6 and its `use_module_device` sibling).
**No hangs** in either mode.

### Perf gate

#### 1. Bound classification — tt-npe pins, and they overturn the entry's premise

`tools/tracy/profile_this.py --collect-noc-traces` was unusable on this box (`tracy/serve_wasm.py`
needs `websockets`), so the trace was captured directly with
`TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` + `..._RPT_PATH`, then pinned with tt-npe:

| regime | est cycles | golden cycles | pred err | **DRAM BW util** | **congestion impact** | avg / max link util | max link demand |
|---|---|---|---|---|---|---|---|
| `b_wide_short` `[1,1,32,16384]`, 64 c | 13 910 | 14 352 | −3.1 % | **103.2 %** | **0.4 %** | 30.4 % / 58.4 % | 185.9 % |
| `c_single_core` `[1,1,512,512]`, 1 c | 58 354 | 58 704 | −0.6 % | **12.6 %** | **0.2 %** | 2.9 % / 7.9 % | 9.5 % |

Congestion isolated by re-running with `--cong none`: 13 855 vs 13 910 cycles (**55 cycles = 0.4 %**)
and 58 232 vs 58 354 (**122 cycles = 0.2 %**).

Two conclusions, both decisive for this entry:

- **`b_wide_short` is at its achievable DRAM bound, not launch/transaction-rate bound.** The entry's
  premise ("0.54 of its 7 300 ns DRAM floor") divided by 288 GB/s *spec* peak; tt-npe's model says the
  DRAM endpoint is already ≥ 100 % utilised for this access pattern (512 B **partial-page** reads out
  of 32 768 B source pages). There is no DM headroom for a per-transaction lever to recover.
- **B10 and A3 are pure congestion levers and congestion is 0.4 % of this regime.** Their refutation
  below is not bad luck — it was 0.4 % of upside against a much larger downside, and tt-npe says so
  *before* the device numbers do.
- **`c_single_core` is per-core NoC-issue bound** (DRAM util 12.6 %, congestion 0.2 %) — which is
  exactly the regime B8's mechanism targets, and exactly where it paid.

Caveat recorded honestly: the NoC-event instrumentation is itself expensive on a 1-core kernel with
768 transactions, so `c_single_core`'s absolute 58 704 cycles is ~2.2× its uninstrumented 27 343 ns.
The *utilisations and ratios* are what the pin is used for; even scaled to the real duration its DRAM
util is ~28 %, nowhere near saturation, so the conclusion is unaffected.

#### 2. The B8 gate — two device sweeps, one mechanism

B8's payoff is governed by a single thing: **it pays while the plan is below DRAM saturation**, i.e.
while the binding resource is the core's own read issue/drain rather than the DRAM aggregate. Both
clauses are sweeps of that boundary from different directions (7 rounds × 10 launches, in-run A/B
pairs, CV ≤ 1.1 %).

**Clause 1 — core count at a fixed 1024 B read** (`[1,1,4096,512]`, chunk 16, `core_cap` forcing the
count so only `ncores` moves):

| cores | blk/core | no lever ns | B8 ns | B8/none | achieved GB/s |
|---|---|---|---|---|---|
| 1 | 128 | 218 946 | 190 302 | **0.869** | 38.3 |
| 2 | 64 | 112 612 | 98 316 | **0.873** | 74.5 |
| 4 | 32 | 61 084 | 52 915 | **0.866** | 137.3 |
| 8 | 16 | 45 479 | 45 484 | 1.000 | 184.5 ← DRAM saturates |
| 16 | 8 | 45 193 | 43 649 | 0.966 | 185.6 |
| 32 | 4 | 44 768 | 43 746 | 0.977 | 187.4 |
| 64 | 2 | 44 433 | 44 107 | 0.993 | 188.8 |

1–4 cores is a flat, reproducible **−13 %**; from 8 cores the wall clock stops moving with the core
count at all (~45 µs / ~186 GB/s), which is the saturation the mechanism predicts. The residual ~−3 %
at 16/32 cores is **not** monotone with +0.0 % at 8 and −0.7 % at 64, so it is scatter, not an effect
— excluded, and the numbers are kept as bench rows so a future phase can re-open it.

**Clause 2 — read transaction size at a fixed 64 cores × 2 blocks/core** (`[1,1,4096,W]`,
W = 64/128/256/512 ⇒ chunk 2/4/8/16 ⇒ 128/256/512/1024 B), plus the 64 B / 4-block row:

| read bytes | no lever ns | B8 ns | B8/none | achieved GB/s |
|---|---|---|---|---|
| 64 B | 13 967 | 11 258 | **0.806** | 75.1 |
| 128 B | 9 572 | 7 861 | **0.821** | 109.5 |
| 256 B | 13 536 | 13 709 | 1.013 | 154.9 |
| 512 B | 23 391 | 22 753 | 0.972 | 179.3 |
| 1024 B | 44 433 | 44 107 | 0.993 | 188.8 |

The same boundary from the other side: at ≤ 128 B even the full grid is transaction-rate bound
(75–110 GB/s, far under the ~190 GB/s achievable copy), so the per-block drain is still on the critical
path. 256 B is measured *negative*, so the threshold cannot be pushed to 512 B on the strength of its
0.972 alone.

⇒ `trid_prefetch_pays = blocks ≥ 2 and (ncores ≤ 4 or read_bytes ≤ 128) and depth-3 fits L1`.

**The isolation row is the important part of the attribution.** `TILIZE_LEVER_B8=3` gives the third CB
window *without* the trid pipeline: `x_single_core_b8_window_only` **30 324 ns** vs
`x_single_core_b8_off` **30 332 ns** (1.000), and `x_wide_short_1core_b8_window_only` **57 646** vs
`x_wide_short_1core_b8_off` **57 731** (0.999). So **the entire win is the reads staying in flight, not
the deeper CB** — which is what makes the 1.5× L1 cost attributable to the lever rather than to a
lucky buffer size.

B8 also **beats B13** where both could fire (64 B, 4 blk, 64 cores): no lever 13 967 →
B13 13 219 (0.946) → **B8 11 258 (0.806)**. Both own the read command programming, so the planner
ships the measured winner and B13 yields.

#### 3. Cumulative bench set — non-regression (median of 7 × 10 launches, WH B0 8×8, AICLK ≈ 985 MHz)

Whole set re-measured, not a subset. Noise floor ±2 %; all CVs ≤ 1.2 % except the B10 write-half rows
(3.7–4.7 %, itself a symptom — a saturated write VC is unstable).

| regime | cores | chk | d | blk | B13 | C7 | **B8** | R1c ns | **now ns** | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| a_square | 64 | 16 | 1 | 4 | 0 | 0 | 0 | 85 775 | **85 960** | +0.2 % |
| b_wide_short | 64 | 8 | 1 | 1 | 0 | 0 | 0 | 13 423 | **13 367** | −0.4 % |
| **c_single_core** | 1 | 16 | **3** | 16 | 0 | 0 | **1** | 30 359 | **27 343** | **−9.9 %** |
| d_tall_narrow | 64 | 1 | 1 | 1 | 0 | 1 | 0 | 3 431 | **3 472** | +1.2 % |
| e_square_fp32 | 64 | 8 | 2 | 8 | 0 | 0 | 0 | 182 008 | **182 733** | +0.4 % |
| e_square_bf8b_out | 64 | 16 | 1 | 4 | 0 | 0 | 0 | 64 628 | **64 662** | +0.1 % |
| e_square_fp32_to_bf16 | 64 | 8 | 2 | 8 | 0 | 0 | 0 | 121 126 | **121 078** | −0.0 % |
| f_sharded_small | 4 | 2 | 1 | 4 | — | — | — | 1 363 | **1 372** | +0.7 % |
| f_sharded_large | 64 | 2 | 1 | 8 | — | — | — | 2 071 | **2 070** | −0.0 % |
| g_dram_to_sharded | 64 | 16 | 1 | 1 | 0 | 0 | 0 | 18 964 | **18 988** | +0.1 % |
| g_sharded_to_dram | 64 | 2 | 2 | 8 | 0 | 0 | 0 | 19 841 | **19 753** | −0.4 % |
| x_square_depth1 | 64 | 16 | 1 | 4 | 0 | 0 | 0 | 85 618 | **85 795** | +0.2 % |
| x_square_depth2 | 64 | 16 | 2 | 4 | 0 | 0 | 0 | 85 610 | **85 445** | −0.2 % |
| x_tall_narrow_depth2 | 64 | 1 | 2 | 1 | 1 | 0 | 0 | 3 553 | **3 558** | +0.1 % |
| x_single_core_depth1 | 1 | 16 | 1 | 16 | 0 | 0 | 0 | 40 076 | **40 066** | −0.0 % |
| x_square_fp32_depth2 | 64 | 8 | 2 | 8 | 0 | 0 | 0 | 182 074 | **181 602** | −0.3 % |
| x_fp32_to_bf16_depth2 | 64 | 8 | 2 | 8 | 0 | 0 | 0 | 121 161 | **121 162** | 0.0 % |
| x_sharded_to_dram_depth2 | 64 | 2 | 2 | 8 | 0 | 0 | 0 | 19 546 | **19 815** | +1.4 % |
| x_square_bf8b_depth2 | 64 | 16 | 2 | 4 | 0 | 0 | 0 | 64 309 | **64 205** | −0.2 % |
| **x_tall_narrow_16c** | 16 | 1 | **3** | 4 | 0 | 0 | **1** | 8 126 | **7 550** | **−7.1 %** |
| **x_wide_short_1core** | 1 | 16 | **3** | 32 | 0 | 0 | **1** | 57 401 | **50 892** | **−11.3 %** |
| x_tall_narrow_no_levers | 64 | 1 | 1 | 1 | 0 | 0 | 0 | 3 609 | **3 620** | +0.3 % |
| x_tall_narrow_b13_only | 64 | 1 | 1 | 1 | 1 | 0 | 0 | 3 506 | **3 530** | +0.7 % |
| x_tall_narrow_c7_only | 64 | 1 | 1 | 1 | 0 | 1 | 0 | 3 431 | **3 448** | +0.5 % |
| **n_tall_narrow_4blk** | 64 | 1 | **3** | 4 | 0 | 0 | **1** | 13 228 | **11 199** | **−15.3 %** |
| x_tall_narrow_4blk_no_levers | 64 | 1 | 1 | 4 | 0 | 0 | 0 | 13 901 | **13 967** | +0.5 % |
| x_tall_narrow_4blk_c7_forced | 64 | 1 | 1 | 4 | 0 | 1 | 0 | 15 767 | **15 884** | +0.7 % |
| m_wide_short_4k | 64 | 2 | 1 | 1 | 1 | 0 | 0 | 4 719 | **4 769** | +1.1 % |
| x_wide_short_4k_no_levers | 64 | 2 | 1 | 1 | 0 | 0 | 0 | 5 032 | **5 006** | −0.5 % |
| x_wide_short_4k_c7_forced | 64 | 2 | 1 | 1 | 0 | 1 | 0 | 5 101 | **5 077** | −0.5 % |
| m_wide_short_8k | 64 | 4 | 1 | 1 | 0 | 0 | 0 | 8 067 | **8 124** | +0.7 % |
| x_wide_short_8k_b13_forced | 64 | 4 | 1 | 1 | 1 | 0 | 0 | 8 311 | **8 300** | −0.1 % |
| x_wide_short_8k_c7_forced | 64 | 4 | 1 | 1 | 0 | 1 | 0 | 8 571 | **8 572** | 0.0 % |
| x_wide_short_b13_forced | 64 | 8 | 1 | 1 | 1 | 0 | 0 | 15 981 | **15 978** | −0.0 % |
| x_wide_short_c7_forced | 64 | 8 | 1 | 1 | 0 | 1 | 0 | 15 265 | **15 270** | +0.0 % |
| x_square_b13_forced | 64 | 16 | 1 | 4 | 1 | 0 | 0 | 86 810 | **86 857** | +0.1 % |
| x_g_to_sharded_b13_forced | 64 | 16 | 1 | 1 | 1 | 0 | 0 | 20 107 | **20 085** | −0.1 % |
| x_g_to_sharded_c7_forced | 64 | 16 | 1 | 1 | 0 | 1 | 0 | 19 902 | **19 831** | −0.4 % |
| x_sharded_small_depth1 | 4 | 2 | 1 | 4 | — | — | — | 1 362 | **1 361** | −0.1 % |

**Zero regressions** — max deviation **+1.4 %**, inside the ±2 % noise floor, direction is scatter (16
of 39 carried regimes are faster). **Four real improvements**, all from B8: `c_single_core` −9.9 %,
`x_wide_short_1core` −11.3 %, `n_tall_narrow_4blk` −15.3 %, `x_tall_narrow_16c` −7.1 %. The
`x_tall_narrow_16c` gain is worth noting: it is the *A0-refutation* counterfactual row, and B8 fires on
it (64 B, 4 blk) — the A0 verdict is unchanged (7 550 ns at 16 cores vs 3 472 ns at 64).

Per-core CB L1 cost of B8, recorded: `c_single_core` / `x_wide_short_1core` 131 072 → **196 608 B**
(+65 536); `n_tall_narrow_4blk` / `x_tall_narrow_16c` 4 096 → **12 288 B** (+8 192);
`m_2blk_128B` 8 192 → **24 576 B** (+16 384). Still a constant in `W`
(`PROPERTIES["bounded_cb"]` re-checked at W = 32 / 256 / 2048 / 16384 by
`test_per_core_cb_bytes_stay_bounded_by_a_constant_in_w`).

New rows added to the cumulative set for future phases: `m_2blk_128B` 7 865 · `p_2blk_128B` 9 572 ·
`x_2blk_128B_b8` 7 861 · `p_2blk_256B` 13 536 · `x_2blk_256B_b8` 13 709 · `p_2blk_512B` 23 391 ·
`x_2blk_512B_b8` 22 753 · `p_2blk_1024B` 44 433 · `x_2blk_1024B_b8` 44 107 ·
`p_1024B_{1,2,4,8,16,32}c` + `x_1024B_*c_b8` (12 rows) · `x_single_core_b8_off` 30 332 ·
`x_single_core_b8_window_only` 30 324 · `x_wide_short_1core_b8_off` 57 731 ·
`x_wide_short_1core_b8_window_only` 57 646 · `x_square_b8_forced` 86 252 ·
`x_tall_narrow_4blk_b8_forced` 11 258 · `x_tall_narrow_4blk_b13_only` 13 219 · `x_fp32_b8_forced`
179 993 · `x_wide_short_b10_{forced,read_only,write_only}` 26 610 / 14 764 / 23 793 ·
`x_square_b10_{forced,read_only,write_only}` 184 129 / 93 033 / 162 675 · `x_tall_narrow_b10_forced`
3 451 · `x_single_core_b10_forced` 27 652 · `x_g_to_sharded_b10_forced` 20 007 ·
`x_wide_short_a3_forced` 13 596 · `x_square_a3_forced` 86 224 · `x_tall_narrow_a3_forced` 3 456 ·
`x_wide_short_b10_a3_forced` 34 552.

#### 4. Mode-C used-optimization ledger

| lever | id | predicted Δ | **measured Δ (lever/none)** | verdict |
|---|---|---|---|---|
| trid double-issue on the read path | **B8** | NPE read-group counterfactual: 0.861 at 1 core, 1.004 at 64 cores (32 → 64 outstanding reads) | **0.869 / 0.873 / 0.866** (1/2/4 cores) · 1.000 / 0.966 / 0.977 / 0.993 (8/16/32/64) · **0.806 / 0.821** (64 B / 128 B at 64 c) · 1.013 / 0.972 / 0.993 (256/512/1024 B) | **KEEP, gated** to `blocks ≥ 2 ∧ (ncores ≤ 4 ∨ read ≤ 128 B)`. The prediction was right about both the sign and the mechanism. −9.9 % to −15.3 % on four shipped regimes at +8 192…+65 536 B/core. |
| — sub-lever: the third CB window **alone** (no trid pipeline) | C16-ish | "the deeper CB is what pays" | **1.000** (30 324 vs 30 332) and **0.999** (57 646 vs 57 731) | **DROP as an independent lever** — it buys nothing on its own; the whole B8 win is attributable to the reads staying in flight. |
| — sub-lever: B8 **and** B13 together | B8+B13 | additive | B13 alone 0.946 vs **B8 alone 0.806** on the same row; both own the read command programming | **B8 wins, B13 yields** (mutually exclusive in the planner, `static_assert`-guarded in the reader for the C7 pair). |
| per-reader / per-writer static unicast VC | **B10** | "break first-come-first-serve serialization when readers share a route" | reads-only **1.083 / 1.105**; writes-only **1.780 / 1.893**; both **1.991 / 2.142** (a_square / b_wide_short). Inert at 1 core (1.011) and at 1 tile/core (1.006) | **DROP — refuted.** The firmware picks one static VC deliberately (VC 1); rotating over VCs 0/2/3 *splits* the DRAM-endpoint queue depth per stream instead of pooling it. tt-npe: congestion is only **0.4 %** of the target regime, so there was ≤ 0.4 % to win. |
| reader adjacent to its DRAM bank / one reader ↔ one bank | **A3** | `get_optimal_dram_bank_to_logical_worker_assignment` "one NoC hop, disjoint routes" | **1.017 / 1.003 / 1.002** (b_wide_short / a_square / d_tall_narrow) — at or inside the noise floor | **DROP — no payoff, and structurally inapplicable.** A tilize block needs 32 **consecutive** source pages and page `p` lives in bank `p % 12`, so **every** core necessarily touches all 12 banks. There is no core↔bank affinity to exploit; the permutation can only move average hop count on a grid A0 already fills. |
| — bundle: B10 + A3 (both congestion levers at once) | — | "each alone may be too small to see" | **2.585** (34 552 vs 13 367) | **DROP** — the bundle is the sum of two costs, not a hidden win. Confirms the 0.4 % congestion ceiling rather than contradicting it. |
| — writer keeps a **compile-time** VC when B10 is off | — | a refuted lever must cost nothing when unused | passing a runtime `vc` unconditionally stopped the compiler folding `NOC_CMD_STATIC_VC(vc)` into the constant `NOC_CTRL` word; `m_wide_short_4k` +2.8 % → **+1.1 %** after splitting the call on `if constexpr` | **KEEP the split** — this is the "keep the code for re-measurability" invariant made real. |

#### 5. DM lever checklist review (`master.md` Part 2)

Applied this pass: **B8** (gated). Re-confirmed still applied: B7 one barrier per block, B5/B6 width
coalescing, A0 2D height-first split, A1 `row_wise`, B9 reads NoC0 / writes NoC1, B13 (gated ≤ 128 B,
1 block), C7 (gated 64 B, 1 block), C14 alias, C16 gated depth, D18/D19 program-cache args.
**Measured-no-payoff this pass: B10, A3** (rows above). Deliberately **not** applied and left to their
owners: C14 one-sided aliasing (R3), B5/B6 on the sharded read (R3), C7 on DRAM→sharded (R3),
kernel-count reduction on Path B (R4). **Not applicable, and now measured rather than argued:** bigger
read transactions on `b_wide_short` — see the finding below.

### Issues encountered

1. **The entry's headline premise was wrong, and tt-npe is what showed it.** "`b_wide_short` … 0.54 of
   its 7 300 ns DRAM floor — it is launch/transaction-rate bound, not bandwidth bound" divides by
   288 GB/s *spec* peak. tt-npe says the DRAM endpoint is at **103 %** utilisation for this access
   pattern and congestion is **0.4 %**: the regime is bandwidth-bound after all, at the bandwidth a
   **512 B partial-page** read stream can actually get. That single number explains why all three of
   this entry's levers had to fail there, and it is why the ≥ 1.2× clause is unreachable rather than
   merely unmet.
2. **The residual on `b_wide_short` is the 64-way partial-page fan-in, and the bench already prices
   it.** Compare two rows with **identical traffic (2.10 MB) and identical core count (64) and
   identical 512 B reads**: `b_wide_short` 13 367 ns @ 156.9 GB/s, where all 64 cores read slices of
   the **same 32** source pages; `p_2blk_512B` 23 391 ns for 4.19 MB @ **179.3 GB/s**, where each core
   reads its **own** 64 consecutive pages. Same transaction size, +14 % bandwidth — the difference is
   *which* pages the 64 readers hit. And splitting `b_wide_short` finer does not help either:
   `p_2blk_256B` moves the same 2.10 MB in **13 536 ns** (2 blocks × 256 B) vs 13 367 ns (1 block ×
   512 B), i.e. **1.3 % slower**, so the "give it 2 blocks so depth-2 and B8 can apply" idea is refuted
   without writing it. The lever that remains is a *different algorithm* — read whole 32 KB source
   pages on a subset of cores and redistribute the slices through L1 — which is Refinement 2b, not a
   per-transaction knob.
3. **`noc_async_read`'s `read_req_vc` argument is dead in `DM_DEDICATED_NOC`.** Implementing B10 the
   obvious way (pass the VC per call, as the write side allows) compiles, runs, and does **nothing** —
   `ncrisc_noc_fast_read` only writes `NOC_CTRL` under `DM_DYNAMIC_NOC`. The read VC has to go through
   the sticky register and be restored. Anyone reaching for a per-reader VC on a
   `ReaderConfigDescriptor` kernel will hit this; the API's signature actively misleads.
4. **Four Refinement-1/1c gate tests failed the moment B8 landed — as designed.** B8 supersedes B13 at
   64 B / 4 blocks and takes the single-core depth to 3, so `test_plan_lever_selection_per_read_size`,
   `test_stateful_read_offered_to_every_interleaved_generic_plan`,
   `test_bit_exact_with_l1_interleaved_input` and `test_gated_default_keeps_depth2_when_latency_bound`
   all tripped. That is the pins doing their job; each was updated to the **new measured** selection
   with the reason inline, and the depth one was rewritten to assert the *property* (the C16 gate must
   not pick depth-1 here) rather than the literal `depth == 2`, so the next deeper-window lever does
   not break it again.
5. **`tools/tracy/profile_this.py --collect-noc-traces` cannot run on this box** —
   `tools/tracy/serve_wasm.py` imports `websockets`, which is not installed, and the wrapper resolves
   `tools/tracy` from a *different* checkout. Setting
   `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` + `TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=<dir>/.logs`
   around a plain `tt-probe.sh` run produces the same `noc_trace_dev0_ID*.json` files that tt-npe
   consumes. Worth knowing — the documented path is a dead end here.

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement2.py` — **40 cells**: both B8 gate
  clauses pinned to the sweep that set them (with the tables in the docstrings, so raising a threshold
  fails here with the numbers attached); the structural min-blocks clause; the L1-budget refusal
  (declining rather than shrinking `chunk_wt`); trid values distinct and non-zero; plan selection on 6
  regimes; B8 supersedes B13; B8 never fires when the caller pinned the depth; B8 ⊕ C7; off on the
  alias path and on a multi-page-row sharded input; **bit-exactness** on 7 shapes (including the
  chunk × block flatten, the minimum 2-block pipeline, awkward `Wt = 3`, and a rank-4 fold), over 10
  repeated launches, on L1-interleaved, and for all 5 lever combinations; program-cache hit preserved.
  For the **refuted** levers: gates asserted identity-false with the measured regression in the
  docstring, the B10 bitmask halves independently addressable, the VC rotation inside the 0-3 unicast
  range with read ≠ write per core, and A3 verified to be a genuine **permutation** (same core set,
  same `CoreRangeSet`, exact tensor cover) rather than a work-dropping reorder.
- `_bench_tilize.py` — a `b8`/`b10`/`a3` lever key (`0` off, `1` gated, `2` force past the gate, and
  for B8 `3` = the third window **without** the trid pipeline, the isolation row the ledger rests on),
  `B8`/`VC`/`A3` report columns, the depth-3 CB-budget assert, a B8 structural assert, and **29 new
  regimes**: the two gate sweeps (7 core counts × 2, 4 read sizes × 2), the 3-way isolation on both
  1-core regimes, the B10 read/write half-splits, and forced-lever counterfactuals for B10 and A3 on
  every regime with real traffic.
- `probes/probe_014.py` — the first-light 6 shapes × 5 lever combinations bit-exactness probe.

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **193 passed** (153 prior + 40 new) in
**both** default and `--dev` mode.

### Ranked follow-ups

1. **Refinement 2b (filed)** — `b_wide_short`'s residual, priced: it is at 156.9 GB/s while the same
   512 B transaction reaching *private* source pages gets 179.3 GB/s (`p_2blk_512B`) and a 1024 B
   transaction gets 188.8 (`p_2blk_1024B`). The gap is the 64-way partial-page fan-in on 32 source
   pages, and closing it needs a different algorithm (whole-page reads on a subset of cores +
   L1 redistribution), not a per-transaction lever. Every per-transaction lever is now measured on this
   regime: B13 +19.5 %, C7 +14.2 %, B10 +99 %, A3 +1.7 %, B8 structurally inapplicable, finer chunking
   +1.3 %.
2. **B8's 16/32-core rows are an unresolved ~3 %** (0.966 / 0.977, non-monotone with 8 c and 64 c).
   Worth 4 bench points if a future phase wants it; the rows already exist.
3. **`TRID_PREFETCH_MAX_ROW_BYTES = 128` sits below a measured 0.972 at 512 B.** 256 B is 1.013, so the
   sequence is not monotone and the conservative threshold is right today — but a phase that
   understands the 256 B dip could reclaim ~3 % on the 512 B / multi-block family.
4. Refinements 3–5 unchanged. Refinement 4's `InitUninitMode` clause still has zero headroom
   (Refinement 1 finding 3).
