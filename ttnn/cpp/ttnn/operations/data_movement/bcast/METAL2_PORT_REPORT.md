# Metal 2.0 Port Report — `data_movement/bcast`

## Outcome

**PORTED** — 3 of 5 program factories converted to `MetalV2FactoryConcept` and verified on Wormhole:
`BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`.

**2 factories deferred** (both stay on legacy `create_descriptor`; the op builds and runs via per-factory dispatch):
- `BcastMultiCoreHW` — cross-op donor writer (coordination; invoker-approved defer).
- `BcastShardedHOptimised` — **reverted to legacy after a confirmed port regression** (reproducible device hang on `in1_batch_size==2` width-sharded configs; root cause is a latent kernel over-run the new DFB L1 layout no longer tolerates — needs a kernel fix, out of scope). See Handoff points.

## Provenance

- **Recipe docs (this port):** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Verification

Target: **Wormhole (`wormhole_b0`)**. All runs via `scripts/run_safe_pytest.sh` (5 s dispatch-timeout hang detection + auto device reset) except the C++ gtest.

| Test | Result |
|---|---|
| C++ `build/test/tt_eager/ops/test_bcast_op` | **PASS** (`Test Passed`) — interleaved H / W / HW |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py -k test_bcast` | **45 passed** — interleaved H / W (+ legacy HW) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_bcast.py` (full) | **576 passed** — ShardedH (ported) + ShardedHOptimised (legacy) |
| `sweeps/eltwise/binary/bcast/{bcast.py,bcast_h_sharded.py}` | **not run** — sweep-framework files (no `pytest` test functions; 0 collected). Need the sweep runner, not plain `pytest`. See Open items. |

No-regression baseline confirmed with the invoker before relying on it.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` via `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts` for
`BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`. `BcastMultiCoreHW` and `BcastShardedHOptimised`
remain on `ProgramDescriptorFactoryConcept`. The `program_factory_t` variant is unchanged (5 alternatives);
the framework dispatches per factory, so the mixed op builds and runs.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op already used the default reflection hash).
- Pybind entry points removed: **none** (plain `bind_function<"bcast">`, no factory pybind hook).

### Open items
- `TensorParameter` matching kept **strict** everywhere (no relaxation). No `ArgConfig::RuntimeTensorShape`
  in these kernels, so no `dynamic_tensor_shape` opt-in was needed (bcast is fixed-shape; unlike the
  general `eltwise` family heads-up).

## Handoff points

### 1. `BcastShardedHOptimised` — port regression: device hang on `batch_b > 1` (KERNEL / FRAMEWORK owners)

**Status:** factory reverted to legacy `create_descriptor` in this PR; deferred.

- **Op / factory:** `data_movement/bcast` → `BcastShardedHOptimisedProgramFactory`.
- **Repro (Wormhole):** `misc/test_bcast.py::test_bcast[ShardOrientation.ROW_MAJOR-2-2-BcastOpMath.ADD-DataType.BFLOAT16-DataType.BFLOAT16-128-1280-40-shard_grid3-ShardStrategy.WIDTH]` — i.e. `in0_batch=2, in1_batch=2`, WIDTH-sharded. Hangs (dispatch timeout); **legacy passes the identical config in ~6 s.**
- **Scope:** only `in1_batch_size == 2` (→ `batch_b == 2`) configs hang; all `batch_b == 1` configs pass. `BcastShardedH` (same borrowed-DFB + self-loop pattern, but no batch loop) passes all its configs — so the borrowed/self-loop binding shape itself is fine.
- **What the port changed:** the kernels (`reader_bcast_h_sharded_optimised.cpp`, `bcast_h_sharded_optimised.cpp`) were converted **mechanically only** (CB-id→`dfb::`, positional args→named `args::`, `TensorAccessorArgs`→`tensor::`); their FIFO/loop logic is byte-identical to legacy (verified by diff). Runtime-arg name↔value maps were re-verified against the legacy emission order. The host spec uses borrowed-memory DFBs (`c_0`←`input_a`, `c_16`←`output` self-loop) and `c_1` as a plain 1P+1C DFB — the audit-blessed shapes.
- **Why mechanical conversion fails here:** the ShardedHOptimised **compute kernel over-runs its buffers** when `batch_b > 1`. The factory sets `h_blk = min(Ht, 8)` independent of `Ht_per_batch_b`; for the repro `h_blk = 8` but `Ht_per_batch_b = 4`, so the inner `for (htr=0; htr<h_blk; htr++)` loop, for `bn=1` (`b_offset=4`), computes `current_index = 4..11` and issues `BCAST_OP` reads of `dfb::in0` and `pack_tile` writes of `dfb::out` at tile indices up to **11**, past those DFBs' **8-tile** (`num_tile_per_core`) allocations. Legacy's plain borrowed CBs let the spill land in adjacent L1 harmlessly (the checked output tiles 0..7 still get correct final values); the Metal 2.0 DFB allocation/layout does not tolerate the out-of-bounds access and the program deadlocks (watcher: bcast worker cores' reader RISC stuck in `W` while compute math is `D`; host `Timeout waiting for physical cores to finish`).
- **What a fix needs (out of porter scope):** the kernel loop should bound the `htr` iteration by `Ht_per_batch_b` (or the factory should cap `h_blk = min(Ht_per_batch_b, 8)`), eliminating the over-run. This is a **kernel-logic change** — the recipe forbids "fixing the legacy kernel" during a port, and this over-run is legacy behavior (a latent bug the old CB layout masked). Once the kernel no longer over-runs, the mechanical Metal 2.0 conversion of this factory should complete (the borrowed/self-loop spec shape already works for `batch_b==1` and for all of `BcastShardedH`).
- **Recommendation:** fix the kernel over-run in a separate (non-port) PR, then re-port ShardedHOptimised. The constructed Metal 2.0 version is preserved in git history / can be reconstructed from `METAL2_PORT_PLAN.md`.

### 2. `BcastMultiCoreHW` — cross-op shared donor writer (eltwise/unary owners + bulk-port coordination)

- **Op / factory:** `data_movement/bcast` → `BcastMultiCoreHWProgramFactory`.
- The HW factory binds `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, owned by `eltwise/unary` and **shared by ~46 factories tree-wide**. Porting HW requires either (a) forking that writer as a `_metal2` variant *outside* bcast's directory (a `writer_unary_interleaved_start_id_metal2.cpp` already exists in the `experimental/quasar/` tree), or (b) a coordinated in-place migration of all ~46 consumers. Both reach outside the op directory — the recipe's canonical stop signal.
- **Disposition:** deferred (invoker-approved). No bcast-side changes were made to the shared writer. When the shared-writer Metal 2.0 migration lands (fork or coordinated), HW ports with the borrowed/self-loop shapes the audit already cleared (`IN0_SHARDED` → borrowed `c_0`; `OUT_SHARDED` → donor-writer-drained `c_16`).

## Successes

- **Borrowed-memory DFB + self-loop (recipe / catalog).** `BcastShardedH` ports cleanly with `c_0`/`c_16` as `borrowed_from` DFBs and `c_16` self-looped on the compute (resident output, no writer). Confirmed against the `experimental/quasar/pad` sharded factory that a `borrowed_from` reference **satisfies the "every TensorParameter needs ≥1 binding" validator rule** with no separate `TensorBinding` — this was the one non-obvious spec question and the reference resolved it. 576/576 sharded configs pass.
- **`Table` range-constructor for defines.** `Table<std::string,std::string>(bcast_op_utils::get_defines(...))` converts the legacy `std::map` of bcast defines in one line, exactly as the migration guide describes (no `push_back`, no iterator-pair ctor).
- **Function-local resource-name constants** (per the unity-build-hygiene pattern) avoided anon-namespace symbol collisions across the four factory `.cpp`s in the same unity-build target — declaring `IN0`/`INPUT_A`/`READER` etc. inside each `create_program_artifacts` was frictionless.
- **`hw_config` diff-before-after.** Legacy `ComputeConfigDescriptor{}` maps exactly to `ComputeGen1Config{}` defaults (HiFi4 / Precise / no-32-bit-dest / double-buffer / Approximate / empty `unpack_modes`); the "read resolved values, port exact equivalents" discipline confirmed no silent perf/precision drift. DM kernels use the arch-agnostic `create_reader/writer_datamovement_config(device->arch())` (legacy defaults).

## Friction

### Gaps
- **`get_arg(args::name)` return type for mutated RTAs.** The docs show `auto x = get_arg(args::x)`; for an RTA that the kernel then mutates (`offset++`, `offset += batch_offset`) I used `uint32_t offset = get_arg(args::offset)` to be safe about mutability. A one-line note in the migration guide ("named RTAs are plain `uint32_t`; use `auto` or `uint32_t`, both mutable") would remove the guesswork.

### Confusion
- **Detecting a real hang vs. slow progress cost real time before I switched to `scripts/run_safe_pytest.sh`.** A plain `pytest` run of a hanging config stalls ~37 min (host-side dispatch timeout) with no signal, and abruptly killing it corrupts the device (requiring `tt-smi -r`) — which then makes the *next* run look hung too, compounding the confusion. **The recipe's "Run tests" section should point porters at `scripts/run_safe_pytest.sh` (5 s dispatch-layer timeout + triage + auto-reset) as the default test runner**, not plain `pytest`/gtest with manual backgrounding. It turned a 37-min stall into a 90 s definitive HANG verdict with triage, and its `--dev` watcher dump (per-core waypoints + k_id legend) is what localized the stuck kernel. This was the single biggest workflow lesson of the port.

## Open items for downstream

- **Sweep coverage not exercised.** `tests/sweep_framework/sweeps/eltwise/binary/bcast/{bcast.py,bcast_h_sharded.py}` define sweep-framework suites (no `pytest` test functions) and collect **0 tests** under plain `pytest`. They must be run via the sweep-framework runner. Functional coverage of the ported factories is otherwise strong (C++ gtest + 45 interleaved + 576 sharded pytest cases), but a follow-up should run the sweeps through the proper runner.
- **`BcastShardedHOptimised` re-port** — blocked on the kernel over-run fix (Handoff #1). Reconstruct the Metal 2.0 factory from `METAL2_PORT_PLAN.md` (its spec shape is documented there) once the kernel is fixed.
- **`BcastMultiCoreHW` port** — blocked on the shared donor-writer Metal 2.0 migration (Handoff #2).
- **No cross-op kernel files were modified or forked** by this port (the two deferred factories are exactly the ones that would have required it). Nothing to sunset.
- **Dead legacy args (audit Misc anomalies)** were not carried into the ported kernels where the kernel never read them (reader host idx 1,2,5,6,7; writer idx 1,2). Dead *kernel-side* reads (`num_tiles`, `NCHtWt` locals in the H/W readers) were kept faithfully as named args — cleaning them is a separate cosmetic pass, routed here rather than bundled into the port.
