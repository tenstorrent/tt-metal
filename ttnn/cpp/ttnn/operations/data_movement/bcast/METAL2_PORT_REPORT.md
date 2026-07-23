# Metal 2.0 Port Report — `data_movement/bcast`

## Outcome

**PORTED** — 4 of 5 program factories converted to `MetalV2FactoryConcept` and verified on Wormhole:
`BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`, `BcastShardedHOptimised`.

**1 factory deferred** (stays on legacy `create_descriptor`; the op builds and runs via per-factory dispatch):
- `BcastMultiCoreHW` — cross-op donor writer (coordination; invoker-approved defer). See Handoff points.

`BcastShardedHOptimised` was initially blocked by a latent kernel buffer over-run this port surfaced (device
hang on `batch_b > 1` / wide-shard configs). That over-run was root-caused, reported, and fixed on `main` by
**PR #51056** (`e09c6aea658`, closes #50908). This branch is rebased onto that fix and merges it into the
Metal 2.0-converted kernel, so the factory now ports cleanly. See Handoff points for the full trail.

## Provenance

- **Recipe docs (this port):** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Verification

Target: **Wormhole (`wormhole_b0`)**. All runs via `scripts/run_safe_pytest.sh` (5 s dispatch-timeout hang detection + auto device reset) except the C++ gtest.

| Test | Result |
|---|---|
| C++ `build/test/tt_eager/ops/test_bcast_op` | **PASS** (`Test Passed`) — interleaved H / W / HW |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py -k test_bcast` | **45 passed** — interleaved H / W (+ legacy HW) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_bcast.py` (full) | **640 passed** — ShardedH + ShardedHOptimised (both ported); incl. `batch_b>1` and the `Wt=10` config #51056 added |
| `sweeps/eltwise/binary/bcast/{bcast.py,bcast_h_sharded.py}` | **not run** — sweep-framework files (no `pytest` test functions; 0 collected). Need the sweep runner, not plain `pytest`. See Open items. |

No-regression baseline confirmed with the invoker before relying on it.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` via `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts` for
`BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`, `BcastShardedHOptimised`. `BcastMultiCoreHW`
remains on `ProgramDescriptorFactoryConcept`. The `program_factory_t` variant is unchanged (5 alternatives);
the framework dispatches per factory, so the mixed op builds and runs.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op already used the default reflection hash).
- Pybind entry points removed: **none** (plain `bind_function<"bcast">`, no factory pybind hook).

### Open items
- `TensorParameter` matching kept **strict** everywhere (no relaxation). No `ArgConfig::RuntimeTensorShape`
  in these kernels, so no `dynamic_tensor_shape` opt-in was needed (bcast is fixed-shape; unlike the
  general `eltwise` family heads-up).

## Handoff points

### 1. `BcastShardedHOptimised` — latent kernel over-run this port surfaced (RESOLVED by PR #51056)

**Status:** RESOLVED. Root-caused during the port, fixed on `main` by PR #51056 (`e09c6aea658`, closes #50908); this branch is rebased onto that fix and merges it into the Metal 2.0 kernel. Factory now ported and passing.

- **How it surfaced:** the mechanical Metal 2.0 conversion of `BcastShardedHOptimised` hung reproducibly on `in1_batch_size==2` (→ `batch_b==2`) width-sharded configs (e.g. `misc/test_bcast.py::test_bcast[ROW_MAJOR-2-2-ADD-...-128-1280-40-...-WIDTH]`), while legacy passed the identical config. The conversion was mechanical-only (CB-id→`dfb::`, positional→`args::`, `TensorAccessorArgs`→`tensor::`; FIFO/loop logic byte-identical, arg maps re-verified), so the trigger was the new DFB layer exposing pre-existing kernel behavior.
- **Root cause (two latent over-runs, both pre-existing on `main`):**
  - *Bug 1 — compute h-block over-run (`batch_b > 1`).* `h_blk = min(Ht,8)` is independent of `Ht_per_batch_b`, so the inner `htr` loop over-runs the final partial block, indexing `c_0`/`c_16` past `num_tile_per_core` on the last batch.
  - *Bug 2 — reader w-block ring wrap (`Wt` not a multiple of `w_blk`).* the small `c_1` ring (`num_input_tiles = w_blk`) misaligns across batches when `w_blk ∤ Wt`, wrapping a contiguous chunk write past the buffer end.
  Legacy's plain borrowed CBs tolerated the L1 spill (benign); the Metal 2.0 borrowed-DFB allocation/layout does not, and it deadlocks (watcher: reader RISC stuck `W` while compute math `D`; host `Timeout waiting for physical cores`).
- **Fix (PR #51056, behavior-preserving except the correctness fix):** compute clamps each block to `min(h_blk, Ht_per_batch_b - ht)`; factory picks `w_blk` as the largest divisor of `Wt` that is `≤ 8` (a no-op for all `Wt ≤ 8`). Correctly not done inside the port itself ("do not fix the legacy kernel") — routed out, fixed on `main`, then merged in here during the rebase.
- **Verification:** full `misc/test_bcast.py` = **640 passed** (incl. the previously-hanging `batch_b>1` configs and #51056's added `Wt=10` shape).

### 2. `BcastMultiCoreHW` — cross-op shared donor writer (eltwise/unary owners + bulk-port coordination)

- **Op / factory:** `data_movement/bcast` → `BcastMultiCoreHWProgramFactory`.
- The HW factory binds `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, owned by `eltwise/unary` and **shared by ~46 factories tree-wide**. Porting HW requires either (a) forking that writer as a `_metal2` variant *outside* bcast's directory (a `writer_unary_interleaved_start_id_metal2.cpp` already exists in the `experimental/quasar/` tree), or (b) a coordinated in-place migration of all ~46 consumers. Both reach outside the op directory — the recipe's canonical stop signal.
- **Disposition:** deferred (invoker-approved). No bcast-side changes were made to the shared writer. When the shared-writer Metal 2.0 migration lands (fork or coordinated), HW ports with the borrowed/self-loop shapes the audit already cleared (`IN0_SHARDED` → borrowed `c_0`; `OUT_SHARDED` → donor-writer-drained `c_16`).

## Successes

- **Borrowed-memory DFB + self-loop (recipe / catalog).** `BcastShardedH` ports cleanly with `c_0`/`c_16` as `borrowed_from` DFBs and `c_16` self-looped on the compute (resident output, no writer). Confirmed against the `experimental/quasar/pad` sharded factory that a `borrowed_from` reference **satisfies the "every TensorParameter needs ≥1 binding" validator rule** with no separate `TensorBinding` — this was the one non-obvious spec question and the reference resolved it. Both `BcastShardedH` and `BcastShardedHOptimised` use this shape; 640/640 sharded configs pass.
- **`Table` range-constructor for defines.** `Table<std::string,std::string>(bcast_op_utils::get_defines(...))` converts the legacy `std::map` of bcast defines in one line, exactly as the migration guide describes (no `push_back`, no iterator-pair ctor).
- **Function-local resource-name constants** (per the unity-build-hygiene pattern) avoided anon-namespace symbol collisions across the four factory `.cpp`s in the same unity-build target — declaring `IN0`/`INPUT_A`/`READER` etc. inside each `create_program_artifacts` was frictionless.
- **`hw_config` diff-before-after.** Legacy `ComputeConfigDescriptor{}` maps exactly to `ComputeGen1Config{}` defaults (HiFi4 / Precise / no-32-bit-dest / double-buffer / Approximate / empty `unpack_modes`); the "read resolved values, port exact equivalents" discipline confirmed no silent perf/precision drift. DM kernels use the arch-agnostic `create_reader/writer_datamovement_config(device->arch())` (legacy defaults).

## Friction

### Gaps
- **`get_arg(args::name)` return type for mutated RTAs.** The docs show `auto x = get_arg(args::x)`; for an RTA that the kernel then mutates (`offset++`, `offset += batch_offset`) I used `uint32_t offset = get_arg(args::offset)` to be safe about mutability. A one-line note in the migration guide ("named RTAs are plain `uint32_t`; use `auto` or `uint32_t`, both mutable") would remove the guesswork.

### Confusion
- **Detecting a real hang vs. slow progress cost real time before I switched to `scripts/run_safe_pytest.sh`.** A plain `pytest` run of a hanging config stalls ~37 min (host-side dispatch timeout) with no signal, and abruptly killing it corrupts the device (requiring `tt-smi -r`) — which then makes the *next* run look hung too, compounding the confusion. **The recipe's "Run tests" section should point porters at `scripts/run_safe_pytest.sh` (5 s dispatch-layer timeout + triage + auto-reset) as the default test runner**, not plain `pytest`/gtest with manual backgrounding. It turned a 37-min stall into a 90 s definitive HANG verdict with triage, and its `--dev` watcher dump (per-core waypoints + k_id legend) is what localized the stuck kernel. This was the single biggest workflow lesson of the port.

## Open items for downstream

- **Sweep coverage not exercised.** `tests/sweep_framework/sweeps/eltwise/binary/bcast/{bcast.py,bcast_h_sharded.py}` define sweep-framework suites (no `pytest` test functions) and collect **0 tests** under plain `pytest`. They must be run via the sweep-framework runner. Functional coverage of the ported factories is otherwise strong (C++ gtest + 45 interleaved + 640 sharded pytest cases), but a follow-up should run the sweeps through the proper runner.
- **`BcastMultiCoreHW` port** — the only remaining unported factory; blocked on the shared donor-writer Metal 2.0 migration (Handoff #2).
- **No cross-op kernel files were modified or forked** by this port (the two deferred factories are exactly the ones that would have required it). Nothing to sunset.
- **Dead legacy args (audit Misc anomalies)** were not carried into the ported kernels where the kernel never read them (reader host idx 1,2,5,6,7; writer idx 1,2). Dead *kernel-side* reads (`num_tiles`, `NCHtWt` locals in the H/W readers) were kept faithfully as named args — cleaning them is a separate cosmetic pass, routed here rather than bundled into the port.
