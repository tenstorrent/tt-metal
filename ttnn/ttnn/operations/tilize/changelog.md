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

### Static-analysis pass on the B8 pipeline (`ttnn-static-analyzer`, this entry)

B8 is a hand-rolled software pipeline over a circular buffer, so it was reviewed with a fresh context
*after* the tests were green — the class of bug it could carry (a window computed instead of asked for)
is exactly the class that passes a bit-exactness suite and fails later under different timing.

**Zero structural findings.** All four premises the review was pointed at turn out to be *guaranteed*
rather than observed, and the proofs are worth recording because they are the reason the scheme is
allowed to exist:

1. `cb_base + (b % depth) * window_bytes` — `circular_buffer.cpp:112-136` keeps `page_size` verbatim
   (no alignment round-up), `cb_push_back` advances `fifo_wr_ptr += num_pages * fifo_page_size` and
   wraps only on landing exactly on `fifo_limit`, and `cb_addr_shift == 0` on the DM RISC-Vs — so with
   `num_pages == chunk_wt` the pointer visits exactly `base + k*window_bytes` and returns to base.
   `get_tile_size(cb)` is the same `tt::tile_size(dataformat)` the host used for `page_size`.
   **Across launches** it holds because the firmware re-runs `setup_local_cb_read_write_interfaces()`
   at the top of every launch (`ncrisc.cc:157`), so a cached program's `fifo_wr_ptr` is re-zeroed
   regardless of the previous launch's push count.
2. `cb_reserve_back(2 * chunk_wt)` is **exact, with zero margin**: at depth 3 it guarantees blocks
   `0 … b-1` are popped, and the window block `b+1` targets last held block `b-2`. So
   `static_assert(cb_depth >= 3)` is load-bearing (and generalises correctly to any depth > 3).
3. Issuing block `b+1`'s reads before block `b`'s barrier is safe, and the "popped ⇒ overwritable"
   link is real hardware ordering, not an assumption: `llk_pop_tiles` does
   `TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK)` before storing the acked counter
   (`llk_io_unpack.h:94-97`), so the counter the reader polls is only visible after the unpacker has
   drained that window.
4. Two trids suffice: block `b+2` shares block `b`'s parity but is issued at iteration `b+1`, strictly
   after iteration `b`'s barrier drove that trid's outstanding count to zero — so at most one block
   carries a given trid at any instant. `ncrisc_noc_fast_read` never rewrites `NOC_PACKET_TAG`, and
   `ncrisc_noc_set_transaction_id` waits for `noc_cmd_buf_ready` before retagging, so a retag cannot
   reach an already-latched request.

Four non-severity observations, all acted on:

| # | observation | action |
|---|---|---|
| O1 | `skip_dm` on the B8 path dropped **address generation** too (the whole helper call is inside the guard), while the non-prefetched fallback deliberately keeps its 32 accessor calls behind a `volatile` sink. Any future `TILIZE_SKIP_DM` A/B between B8 and its counterfactual would have been biased in B8's favour by the entire address-gen term (~437 ns of 3 609 on `d_tall_narrow`, per Refinement 1). | **Fixed** — the B8 branch now keeps the 32 accessor calls behind a `volatile` sink under `skip_dm`, matching the fallback. (Production numbers unaffected; this is an ablation-fidelity fix.) |
| O2 | The gate keys on the **busiest** core (`blocks_per_core = max(...)`) while `_split_contiguous` gives `total % parts` cores one extra unit, so an **uneven** split runs B8 on cores with `total_blocks == 1`. Traced safe (the prologue reserve covers the single push, the barrier parity matches the trid the prologue set, the tag is restored) but **untested** — all four B8 test shapes divide evenly. | **Test added** — `test_b8_is_bit_exact_on_an_uneven_split` on `[1,1,3200,32]` (nt_h 100 over 64 cores ⇒ 36 cores × 2 blocks + **28 × 1**), asserting the heterogeneity is really there before checking bit-exactness. Header comment corrected. |
| O3 | No `static_assert` paired `prefetch_blocks == 2` with `row_page_stride == 1`; a future host change could size the CB to 3 windows and then silently take the raw strided fallback — correct output, lever quietly lost, no diagnostic. | **Fixed** — `static_assert` added, plus one recording that the depth-3 bound is exact. |
| O4 | `get_write_ptr` is called before the first `cb_reserve_back`, against the documented contract (value is correct per proof 1; in-tree precedent in the dram prefetcher's reader). | **Comment added** naming the per-launch CB re-init that makes it valid. |

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement2.py` — **41 cells**: both B8 gate
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

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **194 passed** (153 prior + 41 new) in
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

---

## Refinement 2b — `b_wide_short`'s 64-way partial-page fan-in: whole-page reads + L1 redistribution

- **Date**: 2026-07-29
- **Outcome**: **full (`[x]`)** via the entry's second gate clause. The entry's named
  algorithm was implemented **completely** (and bit-exactly) and is **refuted with its
  counterfactual number and a re-pinned tt-npe DRAM-util figure**: 18 468 ns vs
  13 416 = **1.377× SLOWER**, and its own read-side ceiling probe shows the whole-page
  read buys **zero** DRAM time — the entry's premise is false on this hardware. The
  decomposition that produced that verdict also produced a **different** lever, which
  landed and pays: a per-core transaction-order **stagger** takes `b_wide_short`
  **13 367 → 12 554 ns (1.065×)** and `m_wide_short_8k` **8 124 → 7 208 (1.127×)** at
  **zero L1 cost**. The ≥ 1.14× clause is not met and is shown unreachable: tt-npe
  pins the regime at **116.5 % DRAM bandwidth utilisation with 0.2 % congestion**
  after the stagger. Zero regressions across all **81** carried bench regimes.

### What was done

**1. The entry's algorithm — whole-page staged read + L1 redistribution. IMPLEMENTED IN
FULL, MEASURED, REFUTED.**

`b_wide_short` `[1,1,32,16384]` has `nt_h == 1`, so all 64 cores own tile-*columns* of
one tile-row and every core reads a 512 B slice of each of the **same 32** source pages
(a bf16 `W=16384` ROW_MAJOR row *is* one 32 768 B DRAM page). The entry's lever breaks
the coupling between "which bytes a core reads" and "which tiles a core owns":

| phase | what happens |
|---|---|
| 1 | each core issues **ONE** contiguous `piece_bytes = 32 × chunk_row_bytes` read of ONE source page into `cb_stage` — **32× fewer, 32× bigger** DRAM transactions for exactly the same bytes. Cores form groups of `FANIN_GROUP_ROWS = 32` (one per source row); group *g* stages source piece *g*. |
| 2 | all-to-all ready handshake inside the group: one **posted** `noc_semaphore_inc` per group-mate *including self* (a local `+=` would race the 31 inbound atomics and drop increments), then `noc_semaphore_wait_min(sem, 32)`. |
| 3 | each core **PULLS** its own `chunk_row_bytes` slice out of every group-mate's staging buffer into its own `cb_rm_input` window, stick *r* at offset `r × chunk_row_bytes`. Pull, not push, keeps every core the sole writer of its own CB memory. |

Group-mate *r* is logical core `(grp_x[r % grp_w], grp_y[r / grp_w])`; the host passes the
group's **physical coordinate axes** (`grid.x + group_rows/grid.x` words instead of
`2 × group_size`), and asserts that its own work→core order agrees with the kernel's
rectangle indexing, so a future `grid_to_cores` change cannot silently transpose the
exchange. It is **bit-exact on first light** and on every shape in the family, which is
what makes the refutation a *perf* verdict on a working implementation rather than an
implementation failure.

**2. A one-sided DM ablation, which is what located the real residual.** `TILIZE_SKIP_DM`
gained two new values — **2 = drop the READ leg only**, **3 = drop the WRITE leg only** —
so a serialized (depth-1, one-block) regime can be split into its two DM legs. This is
the measurement the entry was missing.

**3. The lever that actually paid — per-core transaction-order `stagger`.** An interleaved
tensor puts page *p* in bank `p % NUM_DRAM_BANKS` (12 on WH B0), and every core issues its
transactions in the *same* page order. With `nt_h == 1` all 64 cores read the same 32
pages, so at issue step *r* every core hits **ONE** bank while the other 11 idle: the
requests are spread across banks *in aggregate* but **clustered in time**. The write side
has the same shape for a different reason — a core writes `chunk_wt` *consecutive* output
pages, so with `chunk_wt = 8` over 12 banks the 64 cores only ever start on **3** distinct
banks. `stagger` rotates each work unit's issue order (`row_rot = index % TILE_HW`,
`col_rot = index % chunk_wt`). It is a **pure index permutation** — same transactions,
same count, same size, same L1 destinations, **zero extra L1 and zero extra state** — and
the read half is expressed as two `read_stick_rows_for_tilize` calls over the two row
runs, so the helper still owns the address generation.

### Accuracy achieved

Unchanged and **bit-exact** — neither lever moves any arithmetic; one reorders NoC issue,
the other moves bytes through an extra L1 hop. `torch.equal` (PCC = 1.0, rtol = atol = 0,
0 mismatching elements) on `[1,1,32,4096]`, `[1,1,32,8192]`, `[1,1,32,16384]`,
`[1,1,32,32768]`, `[1,1,64,16384]`, `[1,1,2048,2048]`, `[1,1,2048,32]`, `[1,1,96,96]`,
`[2,3,128,64]` — for the **forced fan-in path**, for **all five `stagger` bitmask values**
(off / gated / both / read-only / write-only), for **both rotation moduli**, on
DRAM- and L1-interleaved, and over 5 repeated launches. Inputs are `arange`, not `randn`:
every element is unique, so a stick pulled from the wrong group-mate — or a rotation that
permuted the *data* instead of the *issue order* — cannot cancel out. bf8b / narrowing-cast
tolerances untouched (Phase-0 precision baseline still 20/20).

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged, 0 xfail / xpass / drift.
`SUPPORTED` is **unchanged** (this is a perf refinement; it unlocks no cell and declares
no axis value). Whole golden dir minus `test_translated.py`, **run to completion with no
`-k` filter**: **240 passed, 118 skipped, 0 failed**, byte-identical to the Phase-0 / R1 /
R1b / R1c / R2 baselines. The 2 collection ERRORs are the pre-existing `use_module_device`
× `device_params` conflict inside the reference file (Phase-0 issue 6's sibling), present
in every prior phase. **No hangs** in either mode.

### Perf gate

#### 1. The one-sided DM decomposition — this entry's key measurement

`b_wide_short`, 7 rounds × 10 launches, CV ≤ 1.6 %. `full − no_read` / `full − no_write`
are marginal costs; `no_read − no_dm` / `no_write − no_dm` are the legs alone:

| variant | ns | leg alone |
|---|---|---|
| full | **13 461** | — |
| `no_read` (read payload dropped) | 9 977 | **write leg = 7 751 ns → 135 GB/s** |
| `no_write` (write payload dropped) | 8 192 | **read leg = 5 966 ns → 176 GB/s** |
| `no_dm` (both dropped) | 2 226 | compute + sync |

Two findings, both of which overturn the entry's framing:

- **The WRITE leg is the weaker one**, not the read — 135 GB/s vs 176 GB/s, on
  already-whole-page 2 048 B writes. The entry (and R2) looked only at the read.
- **The legs overlap by only 2 482 of a possible ~5 966 ns**, because `nt_h == 1` gives
  every core exactly ONE chunk-block and a single block has no successor to overlap with.

#### 2. The entry's algorithm, priced by its own read-side ceiling probe

`fanin_mode == 2` is a **measurement probe**: phase 1 only, straight into the CB, no
exchange. It moves the same bytes with the same cores in **1 transaction instead of 32**,
so `probe / off` is *the most this algorithm can ever buy* and `forced / probe` is what
the L1 hop plus barrier costs. All rows stagger-free so the comparison is clean:

| shape | read | off ns | **probe** ns | probe/off | **forced** ns | forced/off |
|---|---|---|---|---|---|---|
| `[1,1,32,16384]` | 512 B | 13 416 | **12 627** | **0.941** | **18 468** | **1.377** |
| `[1,1,32,8192]` | 256 B | 8 148 | 6 909 | 0.848 | 10 208 | 1.253 |
| `[1,1,32,4096]` | 128 B | 5 003 | 3 645 | 0.729 | 6 578 | 1.315 |

And the ablation says *where* the probe's gain comes from — **not** from DRAM:

| leg | off | probe | verdict |
|---|---|---|---|
| read alone | 5 966 ns | **5 985 ns** | **IDENTICAL** — a 32× bigger transaction moves the same bytes in the same time |
| write alone | 7 785 ns | 7 765 ns | untouched, as expected |
| compute + sync | 2 226 ns | 1 684 ns | the whole probe gain is the 32 read **ISSUES** |

**So the entry's premise is false on this hardware: 512 B slices of a shared 32 768 B page
cost exactly what 512 B whole pages cost.** The 5.9 % ceiling the probe does show on the
target shape is issue overhead — already below the entry's 14 % gate — and the
redistribution's own L1 exchange leg (**+4 676 ns**) plus its 32-core barrier (**+1 217 ns**
of `no_dm`) spend it three times over. The shipped stagger delivers **7.0 %**, i.e. it
**beats the refuted algorithm's own theoretical ceiling** while costing no L1.

#### 3. Re-blocking, also measured and also refuted

The other way to create the missing read/write overlap is more blocks per core. Forced via
the planner's new `chunk_cap` sweep hook, at the same 64 cores (`b_wide_short`):

| chunk | read B | blk/core | ns | vs chunk 8 | `no_dm` |
|---|---|---|---|---|---|
| **8** | 512 B | 1 | **13 344** | **1.000** | 2 228 |
| 4 | 256 B | 2 | 13 594 | 1.019 | 2 622 |
| 2 | 128 B | 4 | 15 390 | 1.153 | 3 881 |
| 1 | 64 B | 8 | 25 249 | 1.892 | 6 005 |

The per-block sync floor grows **~400–500 ns per block** (R1 measured `590 + 612 × blocks`),
which swamps the overlap a second block buys. Stacked on top of the stagger it is still a
wash (`chunk4 + stg` 12 548 vs `chunk8 + stg` 12 549), so the shipped plan keeps chunk 8.
NB R2 recorded "finer chunking to 2 blocks × 256 B: +1.3 % (`p_2blk_256B` 13 536 vs
13 367)" — but `p_2blk_256B` is `[1,1,4096,128]`, a **different shape** with whole-page
256 B reads, not `b_wide_short` chunked. That comparison is now done properly on the
regime itself.

#### 4. The stagger gate — two device sweeps

Both halves ship **together**, because they are measured **superadditive** and neither is
worth much alone (in-run A/B, 7 rounds × 10 launches):

| shape | chunk | read only | write only | **BOTH** |
|---|---|---|---|---|
| `[1,1,32,16384]` | 8 | 0.992 | 0.985 | **0.929** |
| `[1,1,32,8192]` | 4 | 0.993 | 0.924 | **0.897** |

Mechanism, and it explains the interaction: the instantaneous demand on a bank is read
demand **plus** write demand. Spreading only the reads leaves the writes piled on a few
banks, so the *busiest* bank — which is what sets the time — barely moves. Only when both
streams are spread does the per-bank load flatten. (Returning a single half from the gate
would ship ~1 % of a 7 % win.)

Where it pays — the clause is `nt_h == 1`, i.e. **every** core reads the **same** 32 source
pages, plus a wide enough chunk:

| shape | nt_h | n_w | chunk | read B | off ns | stg ns | ratio |
|---|---|---|---|---|---|---|---|
| `[1,1,32,4096]` | 1 | 64 | 2 | 128 B | 4 989 | 4 972 | 0.997 |
| `[1,1,32,8192]` | 1 | 64 | 4 | 256 B | 8 046 | **7 194** | **0.894** |
| `[1,1,32,16384]` | 1 | 64 | 8 | 512 B | 13 433 | **12 543** | **0.934** |
| `[1,1,32,32768]` | 1 | 64 | 16 | 1024 B | 25 394 | **23 820** | **0.938** |
| `[1,1,64,16384]` | **2** | 32 | 16 | 1024 B | 24 669 | 24 447 | 0.991 |
| `a_square` | 64 | **1** | 16 | 1024 B | 86 058 | 86 591 | 1.006 |
| `d_tall_narrow` | 64 | **1** | 1 | 64 B | 3 609 | 3 627 | 1.005 |
| `g_dram_to_sharded` | 64 | **1** | 16 | 1024 B | 19 049 | 19 402 | 1.019 |
| `e_square_fp32` | 64 | **1** | 8 | 1024 B | 182 908 | 183 178 | 1.001 |

At `nt_h == 2` only half the grid shares each page set — which already halves the
clustering — and the win is gone. At `n_w == 1` each core reads its **own** rows, so there
is nothing to de-cluster. ⇒ `stagger_pays = ncores > 1 ∧ nt_h == 1 ∧ chunk_wt ≥ 4`.

Also swept and **rejected**: rotating by `NUM_DRAM_BANKS` (12), which makes the per-core
*starting* bank perfectly uniform instead of uniform-mod-32 — **12 766 vs 12 505 (+2.1 %)**
and **7 173 vs 7 176 (0.0 %)**. The row-loop period wins; the sweep hook stays (and is
covered by a bit-exactness test) so the verdict is re-measurable.

#### 5. Ceiling vs measured, and why ≥ 1.14× is unreachable

| | before | **after** |
|---|---|---|
| `b_wide_short` measured | 13 367 ns @ 156.9 GB/s | **12 554 ns @ 167.1 GB/s** |
| vs the in-tree measured 64-core DRAM→DRAM copy (193.8 GB/s) | 0.810 | **0.862** |
| **tt-npe golden cycles** | 14 477 | **12 714** |
| **tt-npe DRAM BW util** | 102.3 % | **116.5 %** |
| **tt-npe congestion impact** | 0.7 % | **0.2 %** |
| tt-npe avg / max link util | 30.0 % / 59.5 % | 35.2 % / 63.7 % |
| tt-npe max link demand | 183.2 % | 322.1 % |

Traces captured with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` +
`..._RPT_PATH` (the documented `profile_this.py --collect-noc-traces` path is still a dead
end on this box — R2 issue 5); congestion isolated with `--cong none` (12 695 vs 12 724 =
0.2 %). Prediction error −2.4 % / +0.1 %.

**The gate is unreachable, and the arithmetic is short:** 11 700 ns would require the DM
to run at `2.10 MB / (11 700 − 2 250) = 222 GB/s`, i.e. **1.15× the measured achievable
64-core DRAM copy rate** — on a regime tt-npe already places at **116.5 %** of its modeled
DRAM bandwidth with **0.2 %** congestion left. The residual `0.862 → 1.0` is not DM at
all: it is the launch + tilize-LLK floor that a 2.10 MB, one-block-per-core kernel cannot
amortize (`no_dm` alone is 2 250 ns = **17.9 %** of the runtime), and R1c already priced
the bare launch at 735–764 ns of that. That floor is Refinement 4's territory
(kernel-count reduction), not a data-movement lever's.

#### 6. Cumulative bench set — non-regression (median of 7 × 10 launches, WH B0 8×8)

Whole set re-measured — **81 carried regimes**, not a subset. Headline rows:

| regime | shape | cores | chk | blk | STG | R2 ns | **now ns** | Δ |
|---|---|---|---|---|---|---|---|---|
| a_square | `[1,1,2048,2048]` | 64 | 16 | 4 | 0 | 85 960 | **85 501** | −0.5 % |
| **b_wide_short** | `[1,1,32,16384]` | 64 | 8 | 1 | **3** | 13 367 | **12 554** | **−6.1 %** |
| c_single_core | `[1,1,512,512]` | 1 | 16 | 16 | 0 | 27 343 | **27 364** | +0.1 % |
| d_tall_narrow | `[1,1,2048,32]` | 64 | 1 | 1 | 0 | 3 472 | **3 401** | −2.0 % |
| e_square_fp32 | `[1,1,2048,2048]` fp32 | 64 | 8 | 8 | 0 | 182 733 | **182 564** | −0.1 % |
| e_square_bf8b_out | `[1,1,2048,2048]`→bf8b | 64 | 16 | 4 | 0 | 64 662 | **64 606** | −0.1 % |
| e_square_fp32_to_bf16 | `[1,1,2048,2048]` fp32→bf16 | 64 | 8 | 8 | 0 | 121 078 | **121 082** | +0.0 % |
| f_sharded_small | `[1,1,512,64]` H-shard | 4 | 2 | 4 | — | 1 372 | **1 356** | −1.1 % |
| f_sharded_large | `[1,1,2048,512]` B-shard | 64 | 2 | 8 | — | 2 070 | **2 073** | +0.1 % |
| g_dram_to_sharded | `[1,1,2048,512]`→B-shard | 64 | 16 | 1 | 0 | 18 988 | **18 939** | −0.3 % |
| g_sharded_to_dram | `[1,1,2048,512]` B-shard→ | 64 | 2 | 8 | 0 | 19 753 | **19 741** | −0.1 % |
| **m_wide_short_8k** | `[1,1,32,8192]` | 64 | 4 | 1 | **3** | 8 124 | **7 208** | **−11.3 %** |
| m_wide_short_4k | `[1,1,32,4096]` | 64 | 2 | 1 | 0 | 4 769 | **4 808** | +0.8 % |
| n_tall_narrow_4blk | `[1,1,8192,32]` | 64 | 1 | 4 | 0 | 11 199 | **11 242** | +0.4 % |
| x_wide_short_1core | `[1,1,32,16384]` 1c | 1 | 16 | 32 | 0 | 50 892 | **50 750** | −0.3 % |
| x_square_depth1 / depth2 | — | 64 | 16 | 4 | 0 | 85 795 / 85 445 | **85 813 / 85 828** | +0.0 / +0.4 % |
| x_tall_narrow_16c | `[1,1,2048,32]` 16c | 16 | 1 | 4 | 0 | 7 550 | **7 581** | +0.4 % |
| x_sharded_small_depth1 | alias, forced d1 | 4 | 2 | 4 | — | 1 361 | **1 362** | +0.0 % |

**Zero regressions.** Over all 81 carried regimes the spread is **−11.3 % … +2.0 %** with
exactly one row beyond +2 %: `x_square_b10_write_only` at **+4.2 %**, a *refuted*-lever
counterfactual (B10's write-VC half). R2 itself recorded that row family at **CV 3.7–4.7 %,
"itself a symptom — a saturated write VC is unstable"**, and three consecutive
re-measurements of the identical plan in one session gave **167 253 / 170 689 / 167 209 ns
(CV 2.2–3.5 %, ±2.1 % spread)**, so the cross-session delta is inside its own scatter. The
decisive independent check that the writer's loop rewrite cost nothing is that **every
shipped write-heavy regime is flat or faster** (a_square −0.5 %, x_square_depth1 +0.0 %,
e_square_bf8b_out −0.1 %, e_square_fp32 −0.1 %, g_dram_to_sharded −0.3 %,
p_2blk_1024B −0.6 %).

Per-core CB L1: **unchanged on every regime** — the stagger is a pure index permutation and
costs 0 B (`test_stagger_costs_no_l1`). The refuted fan-in path would add `piece_bytes`
(32 768 B on `b_wide_short`, 49 152 B total), still a constant in `W`
(≤ `TILE_HW × WT_CHUNK_MAX × TILE_HW × 4` = 65 536 B), which is why
`PROPERTIES["bounded_cb"]` survives on that path too — asserted at four widths.

New rows added to the cumulative set for future phases: `p_wide_short_r2b_off` 13 416 ·
`x_wide_short_r2b_probe` 12 627 · `x_wide_short_r2b_forced` 18 468 ·
`p_wide_short_8k_r2b_off` 8 148 · `x_wide_short_8k_r2b_probe` 6 909 ·
`x_wide_short_8k_r2b_forced` 10 208 · `p_wide_short_4k_r2b_off` 5 003 ·
`x_wide_short_4k_r2b_probe` 3 645 · `x_wide_short_4k_r2b_forced` 6 578 ·
`p_wide_short_stg_off` 13 453 · `x_wide_short_stg` 12 505 ·
`x_wide_short_stg_read_only` 13 459 · `x_wide_short_stg_write_only` 13 272 ·
`x_wide_short_stg_mod12` 12 766 · `p_wide_short_8k_stg_off` 8 111 · `x_wide_short_8k_stg`
7 176 · `x_wide_short_8k_stg_read_only` 7 988 · `x_wide_short_8k_stg_write_only` 7 329 ·
`x_wide_short_8k_stg_mod12` 7 173 · `p_wide_short_4k_stg_off` 4 971 · `x_wide_short_4k_stg`
4 979 · `p_wide_short_32k_stg_off` 25 633 · **`n_wide_short_32k` 23 690** ·
`p_wide_short_2row_stg_off` 24 704 · `n_wide_short_2row` 24 535 · `p_square_stg_off`
85 779 · `x_square_stg` 86 644 · `x_square_stg_read_only` 85 242 ·
`x_square_stg_write_only` 87 014 · `p_tall_narrow_stg_off` 3 618 · `x_tall_narrow_stg`
3 625 · `p_g_to_sharded_stg_off` 19 006 · `x_g_to_sharded_stg` 19 438 ·
`p_square_fp32_stg_off` 182 051 · `x_square_fp32_stg` 183 476 · `p_wide_short_chunk8`
12 356 · `x_wide_short_chunk4` 12 550 · `x_wide_short_chunk2` 15 224 · `x_wide_short_chunk1`
25 337 · `x_wide_short_chunk4_d2` / `chunk2_d2` / `chunk1_d2` · `x_wide_short_chunk2_gated` /
`chunk1_gated` · `x_wide_short_chunk4_stg` 12 548 · `x_wide_short_chunk2_stg` 15 310.

#### 7. Mode-C used-optimization ledger

| lever | id | predicted Δ | **measured Δ (lever/none)** | verdict |
|---|---|---|---|---|
| whole-page staged read + L1 redistribution | **R2b algorithm** | NPE Mode A: full-contention bracket 21 598 → 15 621 ns (0.72) — read `32×512 B` 14 564 → `1×16 384 B` 7 589 (0.52), + 998 ns L1 leg | **1.377 / 1.253 / 1.315** (16k / 8k / 4k). Its own read-side probe: **0.941 / 0.848 / 0.729**, and the ablation shows the read leg is **5 966 → 5 985 ns, i.e. unchanged** | **DROP — refuted.** The NPE bracket's 0.52 read factor does not exist on silicon: partial-page reads of a shared page cost what whole pages cost, so the algorithm's *entire* upside is the 32 saved read issues (≤ 5.9 % on the target shape) and its L1 leg (+4 676 ns) plus 32-core barrier (+1 217 ns of sync) spend that three times over. Gate identity-false; code + 9 bench rows retained. |
| — sub-lever: the whole-page read **alone** (no exchange) | R2b probe | "the bigger transaction is what pays" | **0.941** on the target shape — and the *shipped stagger beats it at 0.930 for 0 B of L1* | **DROP as a path** — it cannot exist without the exchange (a tile needs all 32 rows, so a core that reads whole pages reads bytes it does not own). Kept as the ceiling row. |
| — sub-lever: re-blocking to create read/write overlap | B0 / C16 | "a second block lets the write of block *i* overlap the read of *i+1*" | **1.019 / 1.153 / 1.892** (chunk 4 / 2 / 1); with the stagger 12 548 vs 12 549 = **1.000** | **DROP** — the per-block sync floor (~400–500 ns/block) exceeds the overlap it buys. `chunk_cap` sweep hook retained. |
| **per-core transaction-order rotation** | **A3′ / B5-adjacent** (new) | de-cluster the instantaneous per-bank demand: with `nt_h == 1` all 64 cores hit one bank per issue step | **0.929 / 0.894 / 0.938** (16k / 8k / 32k) · halves alone **0.992 / 0.985** and **0.993 / 0.924** · **0.997** at chunk 2 · **0.991** at `nt_h = 2` · **1.001–1.019** at `n_w == 1` | **KEEP, gated** to `ncores > 1 ∧ nt_h == 1 ∧ chunk_wt ≥ 4`. −6.1 % on `b_wide_short`, −11.3 % on `m_wide_short_8k`, −7.6 % on the 32k member, at **0 B/core** of extra L1. |
| — sub-lever: rotate by `NUM_DRAM_BANKS` instead of `TILE_HW` | — | "a perfectly uniform starting bank should beat uniform-mod-32" | **+2.1 %** (12 766 vs 12 505) and **0.0 %** (7 173 vs 7 176) | **DROP** — the row-loop period wins. Sweep hook retained. |

#### 8. DM lever checklist review (`master.md` Part 2)

Applied this pass: the **per-core issue-order rotation** — a lever the checklist does not
yet carry. It is adjacent to **A3** (which is about *which core* talks to which bank, and
is refuted here) and to **B5/B6** (which are about transaction *size*, also refuted here),
but it is a third axis: the **temporal order** in which a fixed set of transactions is
issued, at fixed size and fixed placement. Worth promoting to the catalog: it costs
nothing, it is a pure permutation, and it is worth 7–11 % wherever many cores stream the
same interleaved pages in the same order (`nt_h == 1` here, but the shape is generic).

Re-confirmed still applied: B7 one barrier per block, B5/B6 width coalescing, A0 2D
height-first split, A1 `row_wise`, B8 (gated), B9 reads NoC0 / writes NoC1, B13 (gated
≤ 128 B), C7 (gated 64 B), C14 alias, C16 gated depth, D18/D19 program-cache args.
**Measured-no-payoff, cumulative: B10, A3** (R2) and now **the whole-page + L1
redistribution algorithm** and **re-blocking** on the wide-short regime. Deliberately not
applied and left to their owners: C14 one-sided aliasing (R3), B5/B6 on the sharded read
(R3), C7 on DRAM→sharded (R3), kernel-count reduction on Path B (R4).

### Issues encountered

1. **The entry's premise is falsifiable on device, and the probe is what falsified it.**
   "A 64-way *partial-page* fan-in costs DRAM bandwidth" — measured, it does not: 512 B
   slices of a shared 32 768 B page cost **exactly** what 512 B whole pages cost
   (read leg 5 966 → 5 985 ns for a 32× bigger transaction). The 156.9 vs 179.3 GB/s
   comparison the entry rests on is between two shapes with different
   bytes-per-fixed-overhead ratios (2.10 MB / 1 block vs 4.19 MB / 2 blocks), not between
   two page-access patterns. Implementing the read-side **ceiling probe before the full
   algorithm** is what surfaced that for ~30 lines of kernel; the full 3-phase
   implementation then only confirmed it.
2. **The residual was on the side nobody had measured.** R2 spent its whole budget on the
   read path; the one-sided ablation says the **write** leg is the slower one (135 vs
   176 GB/s). It also says the two legs barely overlap (2 482 of 5 966 ns) — which is what
   pointed at the issue *order* rather than the issue *size*.
3. **The two rotation halves are superadditive, and measuring only one would have killed
   the lever.** Read-only is 0.992 and write-only 0.985 — both inside the noise floor a
   reviewer would dismiss — while together they are 0.929. The reason is that the busiest
   bank sees read *plus* write demand, so spreading one stream alone barely moves the
   maximum. A per-lever audit that tests halves independently and drops anything under the
   noise floor would have discarded a 7 % win.
4. **R2's re-blocking refutation compared the wrong thing.** `p_2blk_256B` is
   `[1,1,4096,128]`, a different shape with *whole-page* 256 B reads — not `b_wide_short`
   chunked. Re-blocking is still refuted on the regime itself (1.019 / 1.153 / 1.892), but
   it needed the `chunk_cap` hook to be measured properly, and the hook is now permanent.
5. **`ttnn-static-analyzer` found one real UB and three diagnostic gaps** — see below.

### Static-analysis pass on both new paths (`ttnn-static-analyzer`, this entry)

The fan-in path is a hand-rolled cross-core L1 gather with a semaphore barrier, and the
stagger rewrites the shipped write loop for *every* regime — both are the class of change
that passes a bit-exactness suite and fails later under different timing, so both were
reviewed with a fresh context after the tests were green.

**One finding, acted on:**

| # | finding | action |
|---|---|---|
| F1 | **UNDEFINED_BEHAVIOR (bench-reachable).** Both `fanin_mode` exits `return` early and skipped B10's `NOC_CTRL` **sticky** read-VC restore, so `TILIZE_LEVER_B10=2\|3` together with `R2B=2\|3` leaks this core's custom static read VC into the next, *unrelated* program on that core — the exact hazard the other two exits restore against. Nothing on the host pairs the two gates (both are identity-false, so it is bench-only, which is precisely where the counterfactuals are re-measured). | **Fixed** on both exits, plus an 8-cell regression test (forced `R2B × B10`, asserting the *next* default call is still bit-exact — the leak is cross-program, so the second call is the observable). |

**Three advisories, all applied:**

| # | advisory | action |
|---|---|---|
| O1 | `blocks_per_core == 1` is a **host** gate and cannot be a `static_assert`, but the runtime args that encode it are in the kernel. Without a check, a future second block would surface as a `cb_wait_front` hang preceded by wrong data — and if someone also added the outer loop, the single un-flow-controlled staging window becomes a genuine **silent-corruption** source (a mate overwrites its `cb_stage` with block 2's piece while this core is still pulling block 1). | **`ASSERT(chunk_count == 1 && num_rows == tile_height)`** added inside the fan-in block (watcher builds), matching how B13 guards its own bank identity. |
| O2 | No `static_assert(!stagger \|\| row_page_stride == 1)`, so a multi-page source row would make the host report the lever ON while the kernel silently took the raw strided fallback — correct output, lever quietly lost, no diagnostic. This is the exact gap B8 guards. | **`static_assert` added.** |
| O3 | `tilize_writer.cpp` carried **no** `static_assert` at all, so the write rotation had no compile-time tripwire against `split_read` / `stateful_read` (harmless today — disjoint CBs and semaphores — but unguarded). | **`static_assert(!stagger \|\| !split_read)` added.** |

**Zero structural findings on everything else.** Four premises the review was pointed at
turn out to be *guaranteed*, and the proofs are worth recording:

1. **No launch-skew hazard on the fan-in gather.** Mate *r* pulls only after its own
   `noc_semaphore_wait_min` returns, which requires this core's increment, which is issued
   only after its phase-1 `noc_async_read_barrier()`. Phase 2 uses the *write* atomic
   command buffer and phase 3 the *read* one, so there is no in-order cmd-buf aliasing.
2. **A next-launch overwrite of `cb_stage` while a lagging mate still pulls is impossible.**
   `process_go_signal_mcast_cmd` does not multicast program *N+1*'s go signal until **all**
   workers of programs 1..N have reported done, and N+1's config writes (including
   semaphore initial values) precede it in the dispatch stream — so no core can be a launch
   ahead, and `sem = 0` has landed everywhere before any increment can be issued.
3. **The posted self-increment through the NoC is necessary, not sloppy.** A local
   `*sem += 1` is a non-atomic RISC-V read-modify-write racing 31 inbound NoC atomics; it
   can drop increments and hang the whole group. Corollary worth knowing: because posted
   atomics are not counted, `noc_async_atomic_barrier()` here would be a no-op — the wait
   *is* the barrier, so the usual "always barrier after `noc_semaphore_inc`" rule does not
   apply on this path.
4. **The stagger's manual CB bookkeeping is an exact match for `read_sticks_for_tilize`**
   (`width_in_tiles == chunk_wt` and `padded_row_bytes == chunk_row_bytes` identically, one
   barrier and one push of `chunk_wt` per block, `div_up` and `/` agree because `num_rows`
   is always a multiple of 32), and the two rotated helper calls reproduce the *same*
   (source page → L1 offset) pairs — only the issue order moves. The writer's
   `k = (i + col_rot) mod chunk_wt` is a bijection on `[0, chunk_wt)` reusing the same
   `(l1_addr + k·tile_bytes → base_page + k)` pairing, so `noc_async_writes_flushed()` plus
   the single trailing barrier is still sufficient.

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement2b.py` — **59 cells**.
  For the **refuted** fan-in path: the identity-false gate with the measurement in the
  assert message (so re-enabling it fails here with the numbers attached); the plan never
  selects it by default and leaves **no staging CB and no semaphore** behind when off; its
  structure when forced (one group per source piece, piece exactly `FANIN_GROUP_ROWS`
  chunks wide, the pieces tiling the source row exactly, group axes consistent, bounded
  CB); its **bit-exactness** on three shapes over repeated launches, so the refutation
  reads as a perf verdict on a working implementation; the garbage-producing probe mode is
  unreachable by default; the structural preconditions hold even when forced; mutual
  exclusion with B13/C7/B8/stagger; and **F1** — the sticky read-VC restore across all 8
  forced `R2B × B10` combinations. For the **shipped** stagger: both gate clauses pinned to
  the sweep table in the docstring; the superadditivity that makes it all-or-nothing; the
  `chunk_wt == 1` write-half mask; off on the zero-copy path; **zero L1 and an unchanged
  transaction shape**; never coexisting with a lever that owns the row loop (all levers
  forced at once, the adversarial case); bit-exactness on 7 shapes × 5 bitmask values ×
  both rotation moduli × L1-interleaved × 5 repeated launches; the program-cache hit; the
  rotation being a genuine permutation that still covers the tensor exactly; and A0 +
  bounded-CB at four widths for both levers.
- `_bench_tilize.py` — an `r2b` lever key (`0` off, `1` gated, `2` force, **`3` = the
  read-side ceiling probe**) and an `stg` key (`0`/`1`/`2` plus `3` read-only, `4`
  write-only, mirroring B10's convention); `R2B`/`STG` report columns; a fan-in CB-budget
  assert and a **stagger-gate assert** (narrowing either gate now fails the bench);
  `chunk_cap` and `stagger_mod` sweep hooks; the `TILIZE_BENCH_SPLIT_DM=1` variant set
  (`no_read` / `no_write`, the one-sided DM ablation); and **41 new regimes**.
- `probes/probe_016.py` (fan-in first-light plan + bit-exactness across the family),
  `probes/probe_017.py` (stagger bit-exactness on 6 shapes × 2 lever values).

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **253 passed** (194 prior + 59
new) in **both** default and `--dev` mode.

### Ranked follow-ups

1. **`b_wide_short` is now at its ceiling; the remaining 0.862 → 1.0 is the launch +
   compute floor, which is Refinement 4's lever, not a DM one.** `no_dm` is 2 250 ns =
   17.9 % of the 12 554 ns runtime on a one-block-per-core kernel, and R1c priced the bare
   launch at 735–764 ns of that. Refinement 4's kernel-count reduction is the only entry in
   the queue that attacks it. Note its scope says "Path B (same-spec sharded)" — the same
   lever applies to any one-block-per-core interleaved regime, which is worth widening.
2. **Promote the per-core issue-order rotation to `master.md` Part 2.** It is a third axis
   next to A3 (placement) and B5/B6 (size) — the temporal *order* of a fixed transaction
   set — costs nothing, and is worth 7–11 % wherever many cores stream the same interleaved
   pages in the same order. Refinement 5's Mode-D audit should carry it.
3. **The write leg is the weaker one on every interleaved regime** (135–152 GB/s vs
   176–224 GB/s for reads), and nothing in the queue targets it. Consecutive output tile
   pages land on *different* banks so they cannot be coalesced — but pages `p` and
   `p + NUM_DRAM_BANKS` are *contiguous inside one bank*, so a bank-aware page grouping
   could write several tiles in one transaction. It inverts the read-side coalescing, so it
   is a real trade to measure, not an obvious win.
4. **The stagger's `chunk_wt >= 4` clause has an unexplored corner.** At chunk 2 the
   rotation is neutral (0.997) while the read-side probe shows a **0.729** ceiling on the
   same shape — i.e. `[1,1,32,4096]` has 27 % of headroom that neither lever reaches, and
   it is issue-count bound (32 × 128 B reads). B13 already ships there (0.968); a phase
   that understands the 128 B issue floor could reclaim more.
5. Refinements 3–5 unchanged. Refinement 4's `InitUninitMode` clause still has zero
   headroom (Refinement 1 finding 3).

---

## Refinement 3 — Crossover paths: one-sided zero-copy + bigger sharded-side read transactions

- **Date**: 2026-07-29
- **Outcome**: **partial (`[~]`)**. Both named levers landed on **both** crossover
  directions and a third (a new one) was implemented, measured and refuted. One of the
  two numeric gate clauses is **met** — `g_sharded_to_dram` **19 780 → 15 112 ns
  (1.309×)** against a ≥ 1.2× bar — and the other is **not**: `g_dram_to_sharded`
  **19 158 → 16 006 ns (1.197×)** against a ≥ 1.4× bar. The residual on that regime is
  decomposed to the ns by a one-sided ablation and handed to **Refinement 3b** with the
  one measurement this pass did not take. Both directions now show **zero traffic on
  the sharded side** in three independent ways (kernel CT arg, device ablation, tt-npe
  per-NoC demand). Two HANGS were found and fixed on the way — one of them reachable
  from a plain public call. Zero regressions across all **155** carried bench rows;
  golden 240 / 240 (126 / 126 registry cells), `test_translated.py` 275 passed, unit
  suite **321 / 321** in both default and `--dev` mode.

### What was done

**1. C14 one-sided CB aliasing — LANDED, both directions.**

Phase 0 routed both sides of a crossover through the generic `TensorAccessor`, so the
sharded side paid a full NoC leg it does not need. The sharded side's CB is now built
with `cb_descriptor_from_sharded_tensor` and **the work split is the shard map**: each
core owns exactly its own shard's tiles, so CB page *k* IS shard tile *k*.

| path | direction | who is aliased | what disappears |
|---|---|---|---|
| `alias_out` | interleaved RM → sharded TILE | the OUTPUT CB | the tilize LLK packs straight into the shard: **no write leg** |
| `alias_in` | sharded RM → interleaved TILE | the INPUT CB | the unpacker reads the shard in place: **no read leg** |

The whole difficulty is the map, and it is *derived*, not guessed: legacy 2D shard
specs take their core list from `corerange_to_cores(grid, num_cores, row_wise =
orientation == ROW_MAJOR)` (`buffer.cpp:271`) and `core_to_host_pages`
(`buffer.cpp:119-180`) walks shards column-inner / row-outer while paging each shard
row-major — so shard `(sh, sw)` is linear index `sh*n_sw + sw` and its pages are its own
tiles in row-major order. Two consequences the kernels had to absorb:

- a shard wider than one chunk forces the reader to iterate tile-row-**outer** /
  chunk-**inner** (`blocks_row_major`), the opposite of the generic path's order, or the
  aliased CB is filled transposed with every CB count still balanced;
- `alias_in` cannot chunk at all (`chunk_wt == shard_wt`), because the unpacker reads 32
  whole shard rows in place — so it declines (to the generic path) when that block does
  not fit the L1 budget, instead of OOMing.

`use_multicore=False`, cross-spec reshards, DRAM-sharded sides and genuinely-ND specs
all keep the generic path. ND turned out to be a non-issue *in practice*: the allocator
**normalises** every 2D-representable ND request to a legacy layout (measured,
`probes/probe_024.py` — including the dangerous-looking case where a leading dim is
split while the row dim is whole), so those cells alias correctly; what is left as ND
is e.g. a round-robin with more shards than cores, which the one-shard-per-core clause
declines anyway.

**2. B5/B6 coalesced sharded read — LANDED.** A ROW_MAJOR-*sharded* source stores one
page per logical row and exactly ONE page column per shard, so a chunk-block's 32 pages
are 32 **consecutive** pages inside a single core's L1. When the chunk covers the whole
source page the L1 destination is contiguous too, so the whole block is **one read of
`32 × page_bytes`** instead of 32 reads. Measured on its own (alias forced off), it
takes the L1 read leg of `g_sharded_to_dram` from **12 749 → 5 553 ns (2.30×)** and the
regime from 19 639 → 17 337 ns (1.133×). It is the fallback for every sharded-RM input
the alias declines (wide shards, ND, `use_multicore=False`, cross-spec reshards).

**3. C7 generalised to depth ≥ 2 — machinery LANDED, lever REFUTED.** C7 was
depth-1-only because BRISC read the reserved window out of `get_write_ptr`; it now
**derives** it as `cb_base + (block % depth) * window_bytes` (the identity lever B8
already relies on), and the writer's *alias* branch grew its own copy of the hand-off
(with the output aliased, BRISC has no writes left, so its whole job is the read half).
That is what made the lever measurable at the depth the alias actually wants — and the
measurement refutes it (below). The generalisation is kept and tested, because it is
also what a future depth-2 C7 user needs.

**4. B7' one barrier per GROUP of blocks — NEW lever, implemented, REFUTED.** Reserve G
windows, issue 32·G reads, ONE barrier. Monotonically worse in G.

### Accuracy achieved

Unchanged and **bit-exact** — this refinement moves no arithmetic, it moves *which
memory the CB points at*. `torch.equal` (PCC = 1.0, rtol = atol = 0, 0 mismatching
elements) on **11 crossover geometries** — HEIGHT / WIDTH / BLOCK × ROW_MAJOR /
COL_MAJOR × both directions, plus the width-chunked (`blocks_row_major`) and widest
aliasable shards — plus 6 ND geometries, bf16 and fp32, over repeated launches. Inputs
are `arange`, not `randn`: every element is unique, so a shard map that transposes
blocks cannot cancel out (this is the class of bug that cost Phase 0 26 reference
cells). The bf8b cast through an aliased output CB: PCC 0.999.

### Golden test progress

**126 / 126** registry cells (90 INVALID-skipped) — unchanged, 0 xfail / xpass / drift.
`SUPPORTED` is **unchanged**: this is a perf refinement, it unlocks no cell and declares
no axis value. Whole golden dir minus `test_translated.py`, run to completion with no
`-k` filter: **240 passed, 118 skipped, 0 failed**, byte-identical to the Phase-0 / R1 /
R1b / R1c / R2 / R2b baselines; the 2 collection ERRORs are the pre-existing
`use_module_device` × `device_params` conflict inside the reference file.
`test_translated.py`: **275 passed, 1 failed** — the same reference-file device-
portability bug as Phase 0 (an L1 shard grid derived from `dram_grid_size().x = 12`).
The golden crossover cells (`[1,1,128,64]` DRAM→HEIGHT and HEIGHT→DRAM) now run on the
aliased paths and the cross-spec reshard cell still runs generic. **No hangs** in either
mode.

### Perf gate

#### 1. The one-sided DM ablation — the zero-traffic proof, and the decomposition

`TILIZE_BENCH_SPLIT_DM=1`, 7 rounds × 10 launches, CV ≤ 1.3 %. On an aliased side there
is no payload to drop, so the ablation for that side must come back **equal to full** —
the same instrument Phase 0 used to prove Path B:

| regime | full | no_read | no_write | no_dm | reading |
|---|---|---|---|---|---|
| `g_dram_to_sharded` (`alias_out`) | **16 006** | 5 945 | **16 098 (= full, +0.6 %)** | 5 951 | **no write payload exists**; read leg = 10 055 ns |
| `g_sharded_to_dram` (`alias_in`) | **15 112** | **15 067 (= full, −0.3 %)** | 2 152 | 2 150 | **no read payload exists**; write leg = 12 962 ns |
| `x_sharded_to_dram_coal_only` (generic + B5/B6) | 17 313 | 16 322 | 7 895 | 2 342 | read leg **5 553** (was 12 749 uncoalesced) |

Rates: `alias_out`'s read leg is 2.10 MB / 10 055 ns = **209 GB/s**, within 2.3 % of the
best DRAM read rate this op has ever measured (214 GB/s for its 1024 B reads).
`alias_in`'s write leg is 2.10 MB / 12 962 ns = **162 GB/s** (vs 135 GB/s for
`b_wide_short`'s one-block write leg).

#### 2. tt-npe pins — and one of them is a model failure worth recording

Traces captured per direction with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` +
`..._RPT_PATH`, pinned through `tt_npe_pybind` (the `tt_npe.py` CLI aborts on a
`wallclock_runtime_us` attribute mismatch on this build — use the pybind directly):

| trace | est. cycles | golden | pred. err | DRAM BW util | congestion | **NoC0 demand** | **NoC1 demand** |
|---|---|---|---|---|---|---|---|
| generic (Phase 0, both sides via the accessor) | 19 333 | 19 338 | −0.0 % | 76.6 % | 0.38 % | 54.1 | 49.5 |
| **`alias_out`** | 88 718 | 36 769 | **+141 %** | 40.3 % | 0.01 % | 78.1 | **0.0** |
| **`alias_in`** | 14 862 | 15 099 | −1.6 % | **98.1 %** | 0.39 % | **0.0** | 139.4 |

- **The gate's "zero DRAM traffic on the sharded side" clause is met in the model as
  well as on device**: reads live on NoC0 and writes on NoC1 (lever B9), and the aliased
  side's NoC demand is exactly **0.0** in each direction, where the generic path has
  both non-zero.
- `alias_in` is pinned at **98.1 % DRAM bandwidth utilisation with 0.39 % congestion**
  and predicted within −1.6 %: it is *at* its DRAM write bound.
- `alias_out`'s pin is **not usable** (+141 %). Two reasons, both mechanical: the trace
  is a single **cold** launch (the warm median is 16 006 ns, i.e. ~15 800 cycles, not
  36 769), and NoC-event tracing instruments **every** transaction — `alias_out` issues
  16 384 × 128 B reads where the generic plan issues 2 048 × 1024 B, so the
  instrumentation cost lands 8× harder on exactly the plan being measured. Recorded as
  a model/instrument limitation, with the device ablation as ground truth. (R2 hit the
  mirror-image case: the model predicted a read factor that silicon did not have.)

#### 3. Why `g_dram_to_sharded` does not reach 1.4×, priced term by term

The gate is ≤ 13 517 ns; the plan measures 16 006. The ablation splits that as **read
payload 10 055 + 5 951 of everything else**, and the "everything else" splits again by
comparing two ablation variants whose `skip_dm` path differs in whether it keeps the
address generation (`x_g_alias_d2_bare`'s 5 948 vs the C7 row's 2 133):

| term | ns | share | can it move? |
|---|---|---|---|
| DRAM read payload | **10 055** | 63 % | **no** — 209 GB/s of a 214 GB/s measured best |
| 256 × `accessor.get_noc_addr` per core | **3 815** | 24 % | mostly hidden already — see below |
| launch + per-block CB handshakes | **~2 136** | 13 % | Refinement 4's lever, not a DM one |
| tilize LLK (marginal) | 334 | 2 % | overlapped |

Every lever that could touch the top two terms was implemented and measured on this
exact operating point (128 B reads × 8 blocks × 64 cores — a NEW operating point for all
of them, since they were calibrated where the same core also writes). In-run A/B,
7 rounds × 10 launches, CV ≤ 1.5 %:

| variant | ns | vs shipped | verdict |
|---|---|---|---|
| **d2 + B13 (shipped)** | **15 866** | **1.000** | KEEP |
| d2 bare | 16 796 | 1.059 | B13 pays 5.5 %, reproduced in 3 sessions |
| d1 + B13 | 16 480 | 1.039 | depth-2 pays 3.9 % (reader ↔ LLK overlap) |
| d1 bare | 17 685 | 1.115 | — |
| **d2 + C7** | 17 699 | **1.116** | DROP |
| d1 + C7 | 18 113 | 1.141 | DROP |
| **d3 + B8** | 17 440 | **1.099** | DROP |
| **B7' group 2 / 4 / 8** | 16 187 / 17 706 / 19 750 | **1.020 / 1.116 / 1.245** | DROP (monotone) |

**One mechanism explains all four negatives: the read leg is DRAM-BANK bound, not
issue-rate bound.** Three independent measurements say so — (a) C7 halves the reads each
RISC-V issues and costs 11.6 %; (b) B8 doubles the reads in flight and costs 9.9 %;
(c) the address-generation probe, which reuses ONE address per block, collapses 12 banks
onto 1 and costs **2.80×** (46 851 vs 16 743) — i.e. bank parallelism is worth 2.8× while
issue capacity is worth nothing. B7' fails for a fourth, structural reason: grouping
delays the first push by G blocks and serializes the LLK behind the reads.

**And the address-generation term is already ~70 % hidden**, which is what bounds the
one lever left. B13 removes 20 of every 32 accessor calls (it arms once per bank group),
so if the 3 815 ns were exposed it would have saved ~2 400 ns; it saved **930**. Same
ratio R1c measured on `d_tall_narrow` ("≈73 % of the address-gen prize is hidden behind
DRAM service latency"). So a *better* address generator — 12 accessor calls per **core**
instead of per block, keeping row-major issue order — has a ceiling of roughly
**400 ns (2.5 %)** on top of what B13 already takes, not the 3 900 ns the gate needs.

**What is left unexplained, and it is the handoff:** 10 055 + 2 136 + 30 % of 3 815 +
334 = **13 670 ns**, so ~2 340 ns of the measured 16 006 is not attributed by this
pass's instruments. That is the *one* measurement Refinement 3b needs (a per-RISC Tracy
timeline on the aliased plan, which no ablation variant can substitute for), and it is
also the only place a 1.4× still lives: with the read payload irreducible at 10 055 and
the launch/CB floor at ~2 136, the arithmetic floor for this regime is **~12 200 ns
(1.57×)** — so the gate is *not* provably unreachable, it is unreachable **with this
refinement's lever set**, which is a different and more useful statement.

#### 4. Cumulative bench set — non-regression (155 rows, median of 5 × 10 launches)

Whole set re-measured, not a subset. Headline rows against R2b:

| regime | R2b ns | **now ns** | Δ |
|---|---|---|---|
| a_square | 85 501 | **85 826** | +0.4 % |
| b_wide_short | 12 554 | **12 552** | −0.0 % |
| c_single_core | 27 364 | **27 364** | 0.0 % |
| d_tall_narrow | 3 401 | **3 460** | +1.7 % (7-round re-measure; the 5-round sweep read 3 513 = its own scatter) |
| e_square_fp32 | 182 564 | **182 303** | −0.1 % |
| e_square_bf8b_out | 64 606 | **64 680** | +0.1 % |
| e_square_fp32_to_bf16 | 121 082 | **120 992** | −0.1 % |
| f_sharded_small | 1 356 | **1 361** | +0.4 % |
| f_sharded_large | 2 073 | **2 069** | −0.2 % |
| **g_dram_to_sharded** | 18 939 | **16 006** | **−15.5 %** |
| **g_sharded_to_dram** | 19 741 | **15 112** | **−23.4 %** |
| m_wide_short_8k | 7 208 | **7 188** | −0.3 % |
| m_wide_short_4k | 4 808 | **4 799** | −0.2 % |
| n_tall_narrow_4blk | 11 242 | **11 337** | +0.8 % |
| x_wide_short_1core | 50 750 | **50 822** | +0.1 % |
| x_tall_narrow_16c | 7 581 | **7 560** | −0.3 % |
| x_sharded_small_depth1 | 1 362 | **1 371** | +0.7 % |
| x_sharded_to_dram_depth2 | — | **14 977** | now `alias_in`, forced depth 2 |

**Zero regressions.** Every carried row with a prior value is inside ±2 % except the two
`g_*` regimes (which improved 15-23 %) and the `x_g_to_sharded_*` counterfactual rows,
which improved for the same reason — they are now measured on the *aliased* plan
(`x_g_to_sharded_b13_forced` 16 289, `_c7_forced` 17 707, `_b10_forced` 16 677,
`p_g_to_sharded_stg_off` 15 976, `x_g_to_sharded_stg` 16 050 — the stagger stays gated
off there, `nt_h == 64`). The B10 write-VC rows remain the only high-CV family
(3.7-3.9 %), exactly as R2 and R2b recorded.

Per-core CB L1 **falls** on both crossover regimes, because only the plain side is
allocated: `g_dram_to_sharded` 65 536 → **8 192 B/core**, `g_sharded_to_dram` 16 384 →
**8 192 B/core**. The aliased side is the tensor's own shard (reported separately as
`alias_cb_bytes`), so `PROPERTIES["bounded_cb"]` keeps meaning what it says — asserted at
four widths, and the alias declines rather than growing a block past the budget.

#### 5. Mode-C used-optimization ledger

| lever | id | predicted Δ | **measured Δ** | verdict |
|---|---|---|---|---|
| one-sided alias, DRAM→sharded | **C14** | delete the 6 786 ns write leg (35 % of the runtime) | **19 158 → 16 006 = 1.197×**; ablation confirms the write payload is gone (`no_write == full`) and tt-npe puts NoC1 demand at **0.0** | **KEEP** |
| one-sided alias, sharded→DRAM | **C14** | delete the 12 749 ns read leg (65 %) | **19 639 → 15 112 = 1.300×**; `no_read == full`, tt-npe NoC0 demand **0.0**, DRAM util **98.1 %** | **KEEP** |
| coalesced 32-page sharded read | **B5/B6** | fold 32 × 128 B into 1 × 4 096 B | read leg **12 749 → 5 553 ns (2.30×)**; regime 19 639 → 17 337 = 1.133× on its own | **KEEP** (it is what every alias-declining sharded input gets) |
| B13 stateful reads on the alias path | B13 | cheaper per-read address arithmetic | **0.945** (16 796 → 15 866), reproduced in 3 sessions | **KEEP** |
| depth-2 on the alias path | C16 | overlap the reader with the LLK (there is no writer left) | **0.962** (16 480 → 15 866) | **KEEP** (the existing C16 gate already picks it) |
| C7 split reader on the alias path | **C7** | "BRISC is idle once the write side is aliased; `examples/split_reader` measures up to 1.7×" | **1.116** at depth 2, **1.141** at depth 1 | **DROP — refuted.** The freed BRISC is genuinely idle; the read leg is bank-bound, so a second issuer wins nothing and its hand-off costs. |
| B8 trid double-issue on the alias path | B8 | its own ≤ 128 B size clause fires here | **1.099** | **DROP — refuted** (same mechanism) |
| one barrier per GROUP of blocks | **B7' (new)** | hide the per-block barrier drain; 32·G reads in flight | **1.020 / 1.116 / 1.245** at G = 2 / 4 / 8 | **DROP — refuted, monotone.** There is no drain to hide (the read is at the DRAM rate) and grouping serializes the LLK behind the reads. |
| — instrument: address-gen probe | — | price the 3 815 ns address-gen term | **2.80× SLOWER** (46 851 vs 16 743) | **instrument refuted** — reusing one address collapses 12 banks to 1, so it prices bank serialization. Kept: it is the third proof that the read is bank-bound. |

#### 6. DM lever checklist review (`master.md` Part 2)

Applied this pass: **C14** on both crossover directions (the checklist's "alias the CB
onto the shard so the LLK reads/writes its final address" — Phase 0 had it only for
same-spec sharded) and **B5/B6** on a *sharded* source, which is a case the checklist
states for interleaved sources only: contiguity there comes from the shard layout
(`core_to_host_pages` pages a shard row-major), not from the page index.
Re-confirmed still applied: B7 one barrier per block, A0 2D split, A1 `row_wise`, B8
(gated), B9 reads NoC0 / writes NoC1 — now *visible* in the tt-npe per-NoC demand,
B13 (gated), C16 gated depth, the R2b issue-order rotation (gated), D18/D19
program-cache args. **Measured-no-payoff, cumulative**: B10, A3 (R2), the whole-page +
L1 redistribution algorithm and re-blocking (R2b), and now **C7 on the alias path, B8
on the alias path, and read grouping (B7')**. Deliberately left to their owners:
kernel-count reduction on the alias paths (R4 — and see the follow-up below, it now has
a *provably safe* form on `alias_out`), the run-closing Mode-D audit (R5).

### Issues encountered

1. **`ttnn-static-analyzer` found a hang reachable from a plain public call (F1).** C7 is
   a two-party protocol keyed on `alias_mode` being the same lever on both kernels: on
   `alias_out` the reader (alias_mode 0) reserves and signals while the writer
   (alias_mode 1) does the read half. On **`alias_in` the two values are swapped**, so
   the signalling party is compiled out and BRISC waits on `sem_reserve` forever.
   Reachable from `tilize(HEIGHT-sharded RM input with a 32-WIDE shard)` →
   interleaved: that gives `chunk_row_bytes == 64` and one block per core, i.e. exactly
   `split_read_pays`. **Invisible to a 316-test suite whose sharded cells are all 64
   wide.** Fixed structurally (`not alias_in`) rather than by wiring the reader in — on
   that path the CB *is* the input shard, so a second reader would overwrite the source
   — plus a `static_assert` and the cell itself as a regression test (its observable is
   a timeout, so the test must stay).
2. **A second hang, `--dev`-only, found by re-running `--dev` after fix #1.** The grouped
   read loop pushed `group * chunk_wt` pages in ONE `cb_push_back`, and a single push
   may not **straddle** the end of the FIFO — `cb_push_back` handles only the exact-hit
   wrap ("no other wrap is legal", `dataflow_api.h:213-222`). With `depth == group + 1`
   a group straddles it every other iteration. The lightweight
   `ASSERT(fifo_wr_ptr <= fifo_limit)` turned that into an ebreak (a hang) under
   `--dev`, while the **default build silently ran the write pointer past the limit**.
   Now one push per window. Two lessons: `--dev` is not optional after a CB-arithmetic
   change, and a refuted lever's code still has to be correct because its counterfactual
   rows keep running it.
3. **The alias's *cost* is that it takes the work split's freedom away.** A BLOCK-sharded
   output on 8×8 has 64-column shards, so a core that owns its own shard reads 128 B
   rows where the generic 2D split reads 1024 B. That is why this refinement's
   prediction row (`x_g_to_sharded_chunk2`, the same transaction shape on the generic
   path) was measured **before** any kernel was written — and it is why the alias buys
   1.20× rather than the 1.4× the entry projected from the write leg alone.
4. **tt-npe's `alias_out` pin is unusable and it is instructive.** NoC-event tracing
   instruments every transaction, so a plan with 32× more transactions pays 32× more
   instrumentation: +141 % prediction error on the traced cold launch. The model is fine
   on the other two traces (−0.0 % / −1.6 %). Use the device ablation when the plan's
   transaction *count* is what changed.
5. **An ND worry that measurement dissolved.** The ND shard→core order (row-major over
   the ND grid, round-robin cores) can disagree with the flattened 2D map in principle;
   in practice the allocator normalises every 2D-representable ND request to a legacy
   layout, including the split-leading-dim case. The `nd` guard stays for the specs that
   do not normalise, but the ND crossover cells alias correctly and are tested.

### Static-analysis pass (`ttnn-static-analyzer`, this entry)

Four new code paths, all of the "passes bit-exactness today, breaks later" class, so all
four were reviewed with a fresh context after the tests were green. **Two findings, both
fixed** (F1 above, and F2: the grouped read loop and the address probe also return
without signalling C7's hand-off — latent, now each with its own `static_assert`).
Explicitly cleared, with the proofs worth keeping:

- **CB accounting on both aliased paths.** `alias_out`: compute pushes exactly
  `shard_tiles` pages into a CB with exactly `shard_tiles` pages, so `cb_reserve_back`
  never blocks and the writer's single wait/pop drains precisely what was pushed — and
  because the aliased output can absorb *every* block without the writer running,
  compute always drains `cb_rm_input` and the reader always makes progress.
  `alias_in` is the mirror image.
- **The derived C7 window is the reserved window at any depth**, including across a
  chunk boundary and on an uneven split, because `seq - 1` equals the number of pushes
  NCRISC has made and the FIFO write pointer after *k* pushes is
  `base + (k % depth) * window_bytes`. `cb_in_base` is the true base on a cached
  program's second launch (`brisc.cc:503` re-runs `setup_local_cb_read_write_interfaces`
  per launch and BRISC never advances the pointer).
- **The grouped loop's `depth == read_group + 1` is sufficient, not merely necessary**,
  and the ragged last group is safe.
- **The coalesced read's "one page column per shard" is structural, not lucky**:
  `RowMajorPageConfig::get_page_shape` returns `(1, physical_shard_width)` and RM
  sharded tensors are padded to a whole number of shard widths, so
  `shard_in_pages[1] == 1` always. The reviewer's one caveat — the accessor strides by
  the *aligned* page size, so the gate should be the allocator alignment rather than 32 —
  was **applied**: the gate is now `page % 64`, which covers L1 (16), WH DRAM (32) and
  BH DRAM (64) and is free for every supported dtype.

### Tests added

- `tests/ttnn/unit_tests/operations/tilize/test_tilize_refinement3.py` — **68 cells.**
  The shard→global-tile map on 11 geometries (HEIGHT / WIDTH / BLOCK × ROW / COL × both
  directions + width-chunked + widest aliasable), each asserted to tile the tensor
  **exactly** (no overlap, no gap, every unit's rectangle inside the tensor) and to be
  `torch.equal` on `arange`; per-dtype and cast-through-the-aliased-CB variants;
  zero-copy asserted **structurally** (the aliased side's kernel CT arg 0 == 1);
  program-cache re-binding on both one-sided paths; every decline boundary (genuinely-ND,
  cross-spec reshard, single-core, DRAM-sharded side, a shard too wide for one aliased
  block); the ND-normalisation fact the ND cells rest on; the coalesced read across four
  schemes plus both sides of its "chunk must cover a whole page" gate; each refuted
  lever pinned to its counterfactual numbers **and** still bit-exact when forced (C7 at
  depth 2 is what exercises the new derived-window arithmetic); the two analyzer
  findings, including the exact cell that hung; and the three interleaved bench plans
  asserted untouched.
- `_bench_tilize.py` — an `r3` lever key (0 = the Phase-0 generic path on both sides of
  the crossover, 1 = gated, 2 = force) and a `coal` key; a `read_group` sweep hook and a
  `GRP` report column; the A0 assert extended to the one-sided paths (asserted
  **non-tautologically**: the shards must tile the tensor exactly) and a B7' depth
  invariant; **20 new regimes** — the prediction rows measured before the kernel existed
  (`x_g_to_sharded_chunk2*`), the depth × lever sweep (`x_g_alias_d{1,2,3}_*`), the group
  sweep (`x_g_alias_g{1,2,4,8}[_b13]`), the address-gen probe, and the counterfactuals
  for both directions.
- `probes/probe_019.py` (first light on 12 crossover geometries),
  `probe_023.py`/`probe_024.py` (the ND normalisation question, answered on device),
  `probe_025.py` (the tt-npe trace capture).

Suite status: `tests/ttnn/unit_tests/operations/tilize/` = **321 passed** (253 prior + 68
new) in **both** default and `--dev` mode.

### Ranked follow-ups

1. **`g_dram_to_sharded`'s unattributed 2 340 ns is the only place a 1.4× still lives —
   and it needs a per-RISC timeline, not another lever.** See Refinement 3b: the read
   payload is irreducible (209 of 214 GB/s), the launch/CB floor is ~2 136 ns, and
   30 % of the address-gen term is ~1 145 ns, which sums to 13 670 of a measured 16 006.
   Every DM lever has been measured on this operating point; what has not been measured
   is where the residual goes.
2. **Dropping the writer kernel on `alias_out` is now provably safe** and is the one
   R4-shaped lever this pass can hand over with its precondition already verified: the
   aliased output CB has exactly `shard_tiles` pages and compute pushes exactly
   `shard_tiles`, so the CB **never needs recycling** — the writer's single
   `cb_wait_front`/`cb_pop_front` exists only to close the loop. Worth ~200-400 ns from
   R1c's bare-launch price, and it removes a whole kernel launch from every DRAM→sharded
   call.
3. **The coalesced sharded read has a wider audience than this refinement gave it.** It
   is gated to `chunk_row_bytes == source_page_bytes`; a sharded source whose 2D split
   picks a *narrower* chunk (e.g. `[1,1,512,64]` HEIGHT-sharded, chunk 1 vs a 128 B page)
   gets nothing. Reading `k` whole pages and letting the tilize block be `k` chunks wide
   would extend it, at the cost of coupling the chunk width to the page width.
4. **`alias_in` is at its DRAM write bound (98.1 % util, 0.39 % congestion, −1.6 %
   prediction error).** Nothing in the DM space is left on that direction; its residual
   is the 162 GB/s DRAM *write* rate, which is the same write-side gap R2b flagged as
   unowned across the whole op.
5. Refinements 4-5 unchanged. R4's `InitUninitMode` clause still has zero headroom
   (Refinement 1 finding 3).
