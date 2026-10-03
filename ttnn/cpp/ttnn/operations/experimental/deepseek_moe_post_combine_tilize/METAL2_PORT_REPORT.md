# Port Report — deepseek_moe_post_combine_tilize

## Outcome

**PORTED.** The op's single factory, `DeepseekMoEPostCombineTilizeProgramFactory`, is now on `ProgramSpecFactoryConcept`, and all three of its kernels are converted. On a single Wormhole n150, using a scratch harness the invoker agreed to (see *Verification*), the post-port output is **bit-identical to the pre-port output** across 3 configurations × 10 iterations, with the forced Metal 2.0 legality checks proven live and Watcher clean.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. This is from the `/localdev/edwinlee/Port_Recipe` checkout (branch `akertesz/op-porting-recipe`), and the recipe file I was handed (`/localdev/edwinlee/metal2_port.md`) is byte-identical to its `ai/port/metal2_port.md`. In the op checkout (`/localdev/edwinlee/Metal_Ports`) the provenance command prints nothing, because that tree has no `metal_2.0/` docs.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Verification

- **Build:** `./build_metal.sh --build-tests` succeeded on the first try after the port, and again after a comment-only touch-up, so the tested binary matches the committed tree.
- **Legality checks:** the forcing scaffolding (all 9 `bool skip_validation` sites) was already in the working tree at session start, so I didn't reapply it. I proved it live: the post-port run logs `METAL2_CHECKS_FORCED` from both `program_spec.cpp:3632` and `program_run_args.cpp:673`, 3 each (one per program built). The scaffolding is **not** in this commit.
- **Watcher:** `TT_METAL_WATCHER=10` for every run, and `generated/watcher/watcher.log` has 0 error/assert hits.
- **Tests (confirmed with the invoker):** the op's only direct test, `models/demos/deepseek_v3/tests/unit/test_deepseek_moe_post_combine_tilize.py`, is `requires_device(["TG"])` and auto-skips on this host (one n150). There are no gtests and nothing under `tests/ttnn`. The op is also exercised inside `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_optimized_moe_decode_block.py`, which needs multiple devices. The invoker chose an uncommitted n150 harness that mirrors the TG test:
  - `tg_rowmajor`: the TG test's exact shapes and shard config `(8,1,32,7168)` → ND shard `[32,1024]` on 7×8 cores, ROW_MAJOR, plus the downstream `ttnn.mul`.
  - `tg_colmajor`: the same, in COL_MAJOR.
  - `small_rowmajor`: `(2,1,64,2048)` → `[32,512]` on 4×4.

  Each case does 10 eager iterations (iteration 0 is a cache miss; 1–9 are cache hits with fresh tensors, which exercises the binding refresh), then the same 10 under trace capture and execute, as the TG test does.

| case | exact tilize (eager / trace) | mul PCC | post-port vs pre-port output |
|---|---|---|---|
| tg_rowmajor | 10/10 · 10/10 | 1.0 (20/20) | **bit-identical** (10/10) |
| tg_colmajor | 0/10 · 0/10, **same as pre-port** (legacy bug, below) | n/a | **bit-identical** (10/10) |
| small_rowmajor | 10/10 · 10/10 | n/a | **bit-identical** (10/10) |

  Program-cache entry count is 4 both pre- and post-port.
- **Not run:** the real TG unit test and the fused MoE decode-block test. They need TG / multi-device hardware, so they should run once on TG before merge.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`, so the framework refreshes both tensor bindings on a cache hit: `input` (a `TensorBinding`) and `output` (borrow-only, backing DFB `tilize_output`).

### Device-op-class edits
- **Exception 3, direct-descriptor → conventional factory.** The op arrived in the direct-descriptor shape, with `create_descriptor` as a static member of `DeepseekMoEPostCombineTilizeDeviceOperation` and no `program_factory_t`. In `device/deepseek_moe_post_combine_tilize_device_operation.hpp` I replaced it with a nested `DeepseekMoEPostCombineTilizeProgramFactory { static ProgramArtifacts create_program_artifacts(...); }` and `using program_factory_t = std::variant<DeepseekMoEPostCombineTilizeProgramFactory>;`. Includes went from `program_descriptors.hpp` to `ttnn/metal_v2_artifacts.hpp` plus `<variant>`, and I aligned the comment above the method to the new binding terms. Nothing else in the device-op class changed: `device_operation.cpp` is byte-identical, and the `TT_FATAL` census is unchanged at 12.
- Pybind entry points removed: none. `nanobind.cpp:40` binds only the public function.
- Custom `compute_program_hash`: none, and there was nothing to leave intact.

### Open items
- Relaxation candidates: none noted. The kernels bake shard geometry into CTAs, so strict matching is right.

## Handoff points

- **Readiness sheet out of date (owner: sheet maintainer, Diego).** I fetched it live this session through the Drive connector (file `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`). The row still shows the pre-#57409 state: `Concept = legacy device-op`, `Factory (variant) = DeepseekMoEPostCombineTilizeProgramFactory`, and `Factory definition path = …/device/deepseek_moe_post_combine_tilize_program_factory.hpp`, a file that doesn't exist. In code the op was `descriptor` (direct-descriptor) before this port. This is the spreadsheet-broken case the audit predicted (primary-column conflict plus a phantom factory row). The brief's stop conditions did **not** fire: `Known op issues` is blank, and `TensorParameter relaxation = none`. After this port the factory struct named in the sheet does exist again, though in the device-op header, not in a `_program_factory.hpp`.
- **Device-op-class edit, exception 3** (above). It's noted here as the TTNN integration doc requires: the op arrived in the direct-descriptor shape.

## Findings (legacy behavior, preserved, not fixed)

1. **COL_MAJOR output shards land in the wrong place.** `device/deepseek_moe_post_combine_tilize_program_factory.cpp` (COL_MAJOR branch of the per-core offset loop, originally `:160-161`): `intra_row_byte_offset = (i / output_num_shards_high) * …`, `row_page_offset = (i % output_num_shards_high) * TILE_HEIGHT`.
   - **Symptom:** in the pre-port baseline, COL_MAJOR fails exact tilize on every iteration. Each output shard holds one complete, correct input shard (56/56 shards match an input shard), but at the transposed position (only 2/56 are in place).
   - **Cause:** with ND sharding, the i-th core in orientation order holds tensor shard `(h = i / num_shards_wide, w = i % num_shards_wide)` in both orientations. The orientation changes only the core ordering, so the ROW_MAJOR formula is correct for both orientations.
   - **Fix (for the owners):** use the ROW_MAJOR formula unconditionally.
   - **Coverage:** no existing test covers COL_MAJOR, and the TG test is ROW_MAJOR.
   - **In the port:** reproduced bit-for-bit, per the porting invariant.
2. **Unread named CTA `input_row_page_size`** (reader; was `program_factory.cpp:105`). No kernel reads it. I **carried it over** as a Metal 2.0 named CTA with the same value. Dropping it is numerically harmless, but it would change the reader's compiled-kernel identity: today the value is baked into the binary, so readers for different input row widths compile separately. A cleanup PR can drop it.
3. **`ttnn.mul` segfaults on this op's COL_MAJOR ND-sharded output.** The crash is in `ttnn::operations::binary_ng::adjust_to_shape(ShardSpec, …)`, called from `compute_mem_config_actual` (null `ShardSpec` → `CoreRangeSet` copy at nil). It looks like `binary_ng` assumes a legacy `shard_spec()` exists on an ND-sharded input. It's unrelated to this op and to the port. It surfaced only because my harness's COL_MAJOR case ran the TG test's downstream mul. Owner: eltwise/binary_ng.
4. Already noted by the audit and still present: the unused `moreh_helper_functions.hpp` include (`device_operation.cpp:10`), and the output spec built from `padded_shape()`.

## Successes

- **The *Compiler options* section caught the absent-line trap.** The legacy compute `KernelDescriptor` has no `opt_level` and reads as "nothing to carry", but it resolves to O3. The port sets `KernelBuildOptLevel::O3` explicitly on the compute spec (`program_factory.cpp`, compute `KernelSpec`), and carries DM `O2` verbatim.
- **Re-deriving from the census (CB endpoints) was cheap and confirmed the brief.** It shows two plain 1P+1C DFBs, with no self-loop, no multi-binding flag and no dead CB.
- **The *Running builds and tests* rule that kernel sources freeze mid-run shaped the workflow.** I drafted every port file in the scratchpad while the baseline build and harness ran, and copied them in only after the baseline finished. That's what made the pre/post bitwise comparison trustworthy.
- **The rule that bugs are behavior you preserve turned a scary finding into a clean outcome.** The COL_MAJOR misplacement was found, characterized and written up (Finding 1), and the port reproduces it bit-for-bit. No "fix" leaked into the diff.

## Friction

**Gaps**
- **The recipe's *Hardware configuration* section is stale against the headers.** It names `ComputeGen1Config`, `DataMovementGen1Config`, `std::get<ComputeGen1Config>(…)` and `create_reader_datamovement_config(device->arch())`. This tree has `ComputeHardwareConfig` / `DataMovementHardwareConfig`, each a single struct with common fields plus optional `config_1xx` / `config_2xx`, and `ttnn::create_reader_datamovement_config()` takes no arch argument (`datamovement_kernel_config.hpp`). The *values* guidance (the field-transform table, defaults) still held. Only the type and function names were wrong. The headers resolved it in minutes, which validates the "headers first" advice.
- **The brief points to "the port recipe's rule for unread named args", and that rule doesn't exist.** The recipe, catalog and migration guide cover only dead *CB* CTAs. I chose to preserve the arg (Finding 2). Suggestion: add a one-liner like "an unread named non-CB CTA is carried verbatim; note it in the report."
- **No guidance for an op whose only test is gated to hardware the porter doesn't have.** Here that's TG-only, run on an n150. I handled it by asking the invoker and using a scratch harness with a pre/post bitwise comparison. *Locate and confirm the op's tests* could name this case and recommend the bitwise pre/post comparison, which is a stronger no-regression check than PCC thresholds.

**Confusion**
- **The brief's DM config note was wrong.** It flagged the reader/writer NoC/RISC assignment as "swapped from the usual convention". It's exactly the default convention: reader RISCV_1/NOC_0, writer RISCV_0/NOC_1, which is what `CreateReader/WriterDataMovementConfig()` produce. So the default helpers reproduce legacy byte-for-byte. The audit may be keying on role names rather than the recipe's value table.
- **Rule 8 (preserve comments) is in tension with the total CB→DFB sweep (stale CB comments must go).** The legacy comment on the borrowed output CB ended "…replacing the old UpdateDynamicCircularBufferAddress call in override_runtime_arguments". I kept the comment's explanation of the buffer's role and dropped only that stale history clause. Both rules are stated absolutely, so a tie-break line would help.
- **The self-audit `cb` grep hits `#include "api/compute/cb_api.h"` in the compute kernel.** That's a framework compute-API header, not a DFB name, and the whitelist doesn't sanction removing it, so it stays. The grep's "expect zero hits" claim doesn't hold for compute kernels that include it.

**Environment**
- **A concurrent port was editing the same checkout.** During this session `bcast_to` sources became modified in the shared working tree; only its `METAL2_*.md` files were present at session start. My build and JIT runs therefore included another port's in-progress changes, so I staged and committed only this op's files. The recipe assumes one port per checkout, and parallel ports in one tree weaken build/test attribution.
- **Harness note:** running several ND-sharded cases back-to-back on one device needs explicit `ttnn.deallocate` between cases. Otherwise the next program's static CB region clashes with still-live L1 tensors (`program.cpp:2525`). That's my harness's problem, not the op's.

## Open items for downstream

- **Run on TG before merge:** `models/demos/deepseek_v3/tests/unit/test_deepseek_moe_post_combine_tilize.py` and `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_optimized_moe_decode_block.py`.
- **Shared kernel touches:** none. All three kernels are op-owned and converted in place, with no `_metal2` fork.
- **Gen1-only token conversions (Quasar-uplift debt):** the compute kernel passes `dfb::tilize_input` / `dfb::tilize_output` to `compute_kernel_hw_startup`, `fast_tilize_init`, `fast_tilize_block` and `fast_tilize_uninit` through the `DFBAccessor → uint32_t` conversion, which the docs call Gen1-only.
- **Varargs:** none. Both reader RTAs are named.
- **Cleanups for owners, kept out of this diff:** fix COL_MAJOR (Finding 1), drop `input_row_page_size` (Finding 2), drop the moreh include. The reader RTAs are per-node values, not uniform, so there's no RTA→CRTA candidate.
- **Readiness sheet:** reconcile the row (Handoff points).
