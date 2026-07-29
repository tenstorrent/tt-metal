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
