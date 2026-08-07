# Metal 2.0 Port Report — `data_movement/bcast`

## Outcome

**PORTED — all 5 of 5 program factories now on `MetalV2FactoryConcept`.** Pass 2 (this pass) ported the
final factory, `BcastMultiCoreHW`; pass 1 ported `BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`,
`BcastShardedHOptimised`. All verified on **Blackhole** (`p100a` board — the physical device on this host;
kernels JIT-compile `-mcpu=tt-bh` / `-DARCH_BLACKHOLE`).

**HW (pass 2)** was deferred in pass 1 because its borrowed cross-family writer had no reusable Metal 2.0
fork (only an out-of-bounds `experimental/quasar/` copy). It now ports cleanly under the updated recipe's
`Caution: Porting a shared kernel` rungs, resolving its two shared kernels without any out-of-op edit:
- **Donor writer** `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` → **rung 1: reused** the existing `writer_unary_interleaved_start_id_metal2.cpp` fork (read-only; not modified).
- **Compute `bcast_hw.cpp`** (lent to `rotate_half`) → **rung 2: created** `bcast_hw_metal2.cpp` beside the original; the legacy original keeps serving `rotate_half`. See Handoff points.

`BcastShardedHOptimised` (pass 1) was initially blocked by a latent kernel buffer over-run that pass surfaced
(device hang on `batch_b > 1` / wide-shard configs), root-caused and fixed on `main` by **PR #51056**
(`e09c6aea658`, closes #50908); the branch is rebased onto that fix. See Handoff points for the full trail.

## Provenance

- **Recipe docs (pass 2 / this port):** `8086bd9df7d 2026-08-07 docs(metal_2.0): add the fake-FIFO DM self-loop recipe, hardened by a cold run`
- **Audit docs (inherited, pass 2):** `8086bd9df7d 2026-08-07 docs(metal_2.0): add the fake-FIFO DM self-loop recipe, hardened by a cold run`
- **Pass 1 provenance:** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Verification

Target: **Blackhole (`p100a`)** — the physical device on this host (UMD auto-discovery reports `blackhole`; kernels JIT `-mcpu=tt-bh`). All runs via `scripts/run_safe_pytest.sh` (5 s dispatch-timeout hang detection + auto device reset) except the C++ gtest.

| Test | Result | Coverage |
|---|---|---|
| C++ `build/test/tt_eager/ops/test_bcast_op` | **PASS** (`Test Passed`) | interleaved H / W / **HW** |
| `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py -k test_bcast` | **45 passed** | interleaved H / W / **HW** (32×32, 64×64, 320×384 × ADD/SUB/MUL) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_bcast.py` (full) | **640 passed** | ShardedH + ShardedHOptimised (dim=H); incl. `batch_b>1` and #51056's `Wt=10` |
| `models/demos/vision/generative/stable_diffusion/wormhole/tests/test_cross_attention.py` (all `has_encoder_hidden_states=False`) | **16 passed** (PCC ✓) | **HEIGHT-sharded HW** — 16 distinct shapes (down/mid/up attention blocks, head_dim 40/80/160): the self-attention path reshards `attention_scores` to HEIGHT_SHARDED then `ttnn.bcast(dim=HW)` → my `IN0_SHARDED`+`OUT_SHARDED` borrowed-DFB path |
| `sweeps/eltwise/binary/bcast/{bcast.py,bcast_h_sharded.py}` | **not run** | sweep-framework files (0 collected under plain `pytest`); need the sweep runner. See Open items. |

The C++ gtest and `test_binary_bcast -k test_bcast` were re-run at `ARCH_NAME=blackhole` (both green). `misc/test_bcast.py` (640) and the SD cross-attention case ran on the same Blackhole device. No-regression baseline confirmed with the invoker (pass 1).

**Transitive coverage (independent consumers, all green on Blackhole).** The op is also exercised by other ttnn ops and models that call `ttnn::bcast` internally — independent confirmation of the ported factories:

| Consumer | bcast path | Result |
|---|---|---|
| `test_stats.py` — `std_hw`/`var_hw`/`normalize_hw`/`normalize_global` (`unary_composite_op.cpp` → `bcast(SUB, HW)`) | HW interleaved | 15 passed |
| `test_backward_prod.py` — `prod_bw` (`unary_backward.cpp` → `bcast(MUL, H/W)`) | H + W interleaved | 219 passed |
| SD `test_resnet_block_2d.py` (`bcast(·, H)`) | H | 22 passed |
| Falcon40B `test_falcon_layernorm.py` (1×1 grid; `bcast(·, H)`) | H | 1 passed |

*Not runnable on this card (not a bcast issue):* BERT-large-11 `test_ffn.py`/`test_mha.py` abort in **matmul** (`MatmulMultiCoreReuseMultiCastProgramConfig` needs a 12×9 grid > Blackhole p100a's 11×10) before any bcast op is reached — a device-grid mismatch in the model, unrelated to the port (0 bcast ops fired in those logs). `ttnn.add` and `ops_for_profiling.py` are *not* bcast exercisers (the `ttnn.bcast` call in `test_add.py` is commented out; profiling is not a correctness test).

**HW coverage — both paths verified on device.** Interleaved HW: C++ gtest + the `-k test_bcast` HW cases. **HEIGHT-sharded HW** (`IN0_SHARDED`+`OUT_SHARDED`, borrowed DFBs on the shard grid): the stable-diffusion cross-attention self-attention path exercises exactly this — `ttnn_functional_cross_attention.py:517` calls `ttnn.bcast(dim=HW, memory_config=<HEIGHT_SHARDED>)` on a resharded (line 496) tensor — and **all 16** of `test_cross_attention.py`'s `has_encoder_hidden_states=False` cases pass (verified here across down/mid/up blocks × head_dim 40/80/160; also in CI: perf-models / single-card-demo / t3k). *(Earlier draft wrongly called sharded HW "unverified"; it is production-exercised and now confirmed across 16 shapes.)*

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` via `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts` for
**all 5 factories** (HW joined this pass). The `program_factory_t` variant is unchanged in shape (5
alternatives); all now satisfy `MetalV2FactoryConcept`.

HW is a single factory spanning both interleaved and HEIGHT-sharded configs (`validate` forces in0 & output
to the same layout). DFB `borrowed_from` (`c_0`←input_a, `c_16`←output) and two conditional tensor bindings
(`tensor::src0` gated `IN0_SHARDED`, `tensor::dst` gated `OUT_SHARDED`) are toggled by the sharding flags,
matching the kernels' existing `#ifdef` gates ([conditional-binding pattern]). `WorkUnitSpec::target_nodes`
= `all_device_cores` (interleaved, idle cores zero-filled as legacy) or the **shard grid** (sharded —
required so the borrowed-DFB backing resolves per shard core; behavior-preserving, since legacy's non-shard
cores were idle no-ops). `c_16` is a genuine 2-toucher **1P+1C** (compute PRODUCER + writer CONSUMER — the
writer `wait_front`s the resident output under `OUT_SHARDED`), **not** a self-loop (HW always binds the writer,
unlike the writer-less ShardedH factories).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op already used the default reflection hash).
- Pybind entry points removed: **none** (plain `bind_function<"bcast">`, no factory pybind hook).
- **TT_FATAL delta (HW factory only, legitimate):** the legacy HW factory's lone `TT_ASSERT(dst_buffer != nullptr, ...)` guarded a raw `Buffer*` that the `OUTPUT` `TensorBinding` / `borrowed_from` replaces — subject deleted, per the whitelist's "guard on a raw `Buffer*` a `TensorBinding` replaced" legitimate loss. Device-op-class scope byte-identical.

### Open items
- `TensorParameter` matching kept **strict** everywhere (no relaxation; bcast is fixed-shape).
- **`opt_level` on the pass-1 compute specs — RESOLVED.** The HW compute spec sets `opt_level = O3`
  explicitly (legacy `ComputeConfigDescriptor` resolves to O3, Metal 2.0 defaults to O2 — recipe rule 2).
  The **four pass-1 factories had omitted this**, silently dropping their compute kernels to O2 vs. `main`'s
  O3. Fixed in a follow-up commit on this branch (`.opt_level = KernelBuildOptLevel::O3` added to all four
  pass-1 compute `KernelSpec`s), restoring O3 parity with `main`. Kept as a separate commit from the HW
  port for clean attribution.

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

### 2. `BcastMultiCoreHW` — shared-kernel rungs taken (pass 2; both resolved, no out-of-op edit)

- **Op / factory:** `data_movement/bcast` → `BcastMultiCoreHWProgramFactory`.
- **Donor writer (rung 1 — REUSE):** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (owned by `eltwise/unary`, ~32 legacy binders). A Metal 2.0 fork already exists beside it — `writer_unary_interleaved_start_id_metal2.cpp` (committed by #51771; consumers: `copy/typecast`, `experimental/unary_backward/gelu_backward`). The HW writer `KernelSpec::source` points at that fork and adopts its interface verbatim (`dfb::out` CONSUMER, `tensor::dst`, args `num_pages`/`start_id`, define `OUT_SHARDED`; page size via `dfb.get_entry_size()`). The bcast call site fit the fork exactly — **no fork edit**, no new file. The fork is read-only to this port (it already has consumers). **Remaining unmigrated consumers of the legacy original (sunset list):** ~30 other legacy binders across `data_movement/*`, `eltwise/*`, etc. — the legacy copy is deleted when the last migrates (not this port's job).
- **Compute (rung 2 — CREATE fork):** `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp` is bcast-owned but **lent** to `ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/rotate_half_program_factory.cpp`, which is on the **legacy device-op concept** (`create()` + `override_runtime_arguments()`) and far from migrating. No fork existed → this port created `bcast_hw_metal2.cpp` beside the original, converted it, pointed the HW compute at it, and added the standard pointer comment to the legacy `bcast_hw.cpp` (its only edit). **Remaining unmigrated consumer (sunset list):** `experimental/transformer/rotate_half`.
- **Disposition:** HW **ported and verified** (interleaved). Both shared kernels handled within the port's sanctioned scope (rung-1 reuse + rung-2 fork-beside-original). No out-of-op edits beyond the sanctioned fork-adjacent pointer comment (and rung 1 required not even that).

## Successes

- **Shared-kernel rungs (Caution: Porting a shared kernel) — the pass-2 unblocker.** HW's two shared kernels resolved cleanly along the recipe's two rungs: **rung 1** reused the pre-existing `writer_unary_interleaved_start_id_metal2.cpp` fork (the locational `ls`-beside-the-original check found it; the tree-wide `_metal2` grep would have wrongly surfaced the out-of-bounds quasar copy — the recipe's warning about that fired correctly), and **rung 2** created `bcast_hw_metal2.cpp` for the *lent* `bcast_hw.cpp`. The "lent" case (bcast owns the file, but `rotate_half` also binds it) is exactly the trap the catalog warns about — "being in your writeable surface does not make it yours to convert in place" — and heeding it (fork, don't convert in place) kept the legacy `rotate_half` build intact. The reused writer fork's interface fit the bcast call site with zero edits, validating the "name the bindings for the kernel, not your op" convention.
- **Conditional bindings across one interleaved+sharded factory.** HW is a single factory spanning both layouts; the [conditional-binding pattern] applied to *both* DFB `borrowed_from` and the `tensor::src0` / `tensor::dst` `TensorBinding`s (gated by the same `IN0_SHARDED` / `OUT_SHARDED` the kernels already `#ifdef` on) translated 1:1 with no framework friction. The reused writer fork's own `OUT_SHARDED` gate lined up with the host's conditional `tensor::dst` binding automatically.
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
- **All 5 factories now ported.** No unported bcast factories remain.
- **HEIGHT-sharded HW — verified via the stable-diffusion cross-attention model** (all 16 `has_encoder_hidden_states=False` cases of `test_cross_attention.py`; in CI: perf-models / single-card-demo / t3k), which run `ttnn.bcast(dim=HW)` on a HEIGHT_SHARDED tensor (the `IN0_SHARDED`+`OUT_SHARDED` borrowed-DFB path) across 16 shapes. All passed on device here. A dedicated *unit* test for sharded HW would still be worthwhile (the current coverage is incidental to a model test), but the path is well-exercised, not dark.
- **Shared-kernel forks — sunset checklist** (per [Caution: Porting a shared kernel]):
  - `bcast_hw_metal2.cpp` (created this pass, in-directory fork of the lent `bcast_hw.cpp`): remaining consumer of the legacy original = `experimental/transformer/rotate_half`. When rotate_half migrates, `bcast_hw.cpp` can be deleted and the fork can take over its name.
  - `writer_unary_interleaved_start_id_metal2.cpp` (reused, not created by this pass): the eltwise/unary owners track its ~30 remaining legacy binders; bcast is now one more consumer of the fork.
- **`opt_level = O3` on the four pass-1 compute specs — RESOLVED** (see TTNN ProgramFactory → Open items): the missing O3 was restored on all four in a follow-up commit on this branch.
- **Dead legacy args (audit Misc anomalies)** were not carried into the ported kernels where the kernel never read them. For HW specifically, the legacy reader's `src0_addr`/`src1_addr` RTAs and the writer's `dst_addr` RTA are gone (→ tensor bindings); the compute's `bcast_hw_metal2.cpp` and reader keep their faithful named args. Cleaning any dead kernel-side reads is a separate cosmetic pass, routed here rather than bundled.
