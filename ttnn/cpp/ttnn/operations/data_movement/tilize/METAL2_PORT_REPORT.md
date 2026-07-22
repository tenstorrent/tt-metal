# Port Report — `data_movement/tilize`

Metal 2.0 port of the **clean subset** of `TilizeDeviceOperation`. Scope-tight, config-scoped port:
**`TilizeMultiCoreDefaultProgramFactory` and `TilizeSingleCoreProgramFactory` fully ported**;
**`TilizeMultiCoreBlockProgramFactory` deferred** (structural finding, below); **`TilizeMultiCoreShardedProgramFactory`
stays legacy** (RED-gated by the audit). The device-op variant runs mixed-concept and dispatches per-factory.

## Provenance
- **Recipe docs (this port):** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`
- **Audit docs (inherited):** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`

## TTNN ProgramFactory
### Concept realized
`MetalV2FactoryConcept` on Default + SingleCore (`create_program_artifacts` returning `ProgramArtifacts`). Block +
Sharded remain on the legacy `descriptor` concept (`create_descriptor`). Mixed-concept variant builds and dispatches
per-factory, as the recipe describes.
### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op never had one).
- Pybind entry points removed: **none** (nanobind binds `tilize` via `bind_function`; no `create_descriptor` pybind hook).
- No factory-hook parameter to drop.
`tilize_device_operation.cpp/.hpp` were **not modified** — the `get_dynamic_runtime_args` hook (live only for the sharded
factory) is left untouched and continues to early-return `{}` for the ported factories.

### Open items
- **Block factory deferred — see "Open items for downstream" for the full design + remaining work.**
- **Relaxation candidates:** none mined (strict tensor matching kept; no `ArgConfig::Runtime*` in the ported kernels).

## Verification
- **Build:** `./build_metal.sh --build-tests` clean (exit 0) after a field-order fix (see Friction / DOC BUG).
- **Anti-pattern self-audit:** clean — no `buffer()->address()`, no magic CB indices, no `TensorAccessorArgs`, no `.id`
  extraction, no `allow_instance_multi_binding`, all CTAs named, no varargs. Hardware configs diffed against legacy
  resolved values.
- **Tests (confirmed baseline, WH):** `test_tilize.py` → **142 passed, 36 skipped, 3 failed** (all 3 pre-existing, below);
  `test_tilize_untilize_2D.py` + `misc/test_tilize_test.py` → **262 passed, 0 failed**. Every case that provably selects
  the ported Default / SingleCore factories passes (incl. `use_multicore=False`, nd-sharded fallback-to-Default,
  DRAM-interleaved).
- **The 3 failures are PRE-EXISTING, not regressions** — confirmed by re-running each on the pristine pre-port tree
  (git-stash + rebuild), where they fail **byte-identically**:
  1. `test_deepseek_v3_mla_tilize_trace_mode[…interleaved…]` — `trace buffers of size 2310144B > 1671168B` (hardcoded
     `trace_region_size`). Routes to the **Block** factory (deferred, still legacy) via the wide-tensor threshold.
  2. `test_deepseek_v3_mla_tilize_trace_mode[…width_sharded…]` — `Illegal kernel placement for writer_unary_sharded`
     (the **Sharded** factory, untouched; the test deliberately tiptoes around dispatch cores under `dispatch_core_axis=COL`).
  3. `test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107` — `No core coordinate found at (8,0)` (device-grid
     mismatch; issue #49107 per the test name). DRAM input → routes to Default, but the failure is a hardcoded (8,0) core
     in the test/shard-spec, not the factory — and it reproduces identically pre-port.

## Handoff points
- **Cross-op kernel forks (bulk-port coordination signal).** This port forked two shared kernels into `_metal2` copies,
  per [Caution: Modifying a shared dataflow kernel]. Both legacy copies stay in place for their unmigrated co-borrowers;
  the forks are the sunset signal (delete legacy when the last co-borrower ports). Owners: eltwise/unary + shared-pool.
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — fork of the **42-consumer**
    `writer_unary_interleaved_start_id.cpp`. Used by tilize Default + SingleCore (and Sharded-interleaved-out, when that
    ports). 40 legacy consumers remain.
  - `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` — fork of the shared-pool `tilize.cpp` (6 consumers). Used by tilize
    Default + SingleCore. `tilize_with_val_padding` (3 factories) + tilize Sharded remain on the legacy copy.
- **No kernel-lib / LLK gaps hit.** `dfb::name → uint32_t` implicit conversion carried the DFB handles into
  `compute_kernel_lib::tilize<…>` (a `uint32_t` NTTP), `is_fp32_input_format<…>()`, and `compute_kernel_hw_startup(…)`
  with no `.id` extraction or wrappers. No `sem::`/`tensor::` boundary-assumption violations (the op has no semaphores
  and no out-of-op tensor-accessor call sites).

## Successes
- **Named-binding model fit the op cleanly.** Both Case-1 tensors collapsed to `TensorAccessor(tensor::input/output)`;
  the `Buffer*` RTA slots and all `TensorAccessorArgs` plumbing dropped exactly as the [TensorParameter] recipe predicts.
  The reader's dead CTA slot 0 and dead RTA slots 2/6/7 (audit "Misc anomalies") fell away naturally under the
  positional→named conversion — zero behavior change, not a "cleanup."
- **`dfb::name` as a `uint32_t` NTTP works** (`DFBAccessor::operator uint32_t()` is `constexpr`). This was the one thing I
  pre-checked before forking the compute kernels; the recipe's rule-2 conversion held even in template-argument position,
  which the docs only illustrate for function arguments. Worth an explicit line in the patterns catalog.
- **Hardware-config discipline paid off.** The compute path uses a raw metal `ComputeConfigDescriptor` (not a ttnn
  `ComputeKernelConfig`), so I mapped it field-by-field to `ComputeGen1Config` and verified every legacy default coincides
  with a Gen1 default — only `enable_32_bit_dest` and `unpack_modes` needed setting. (See Friction: the recipe's compute
  section assumes a ttnn `ComputeKernelConfig`.)
- **The two-consumer DFB shape (Default full+cliff compute) is a legal disjoint multi-binding, not a flag.** Re-derived
  from the census: `all_cores = core_range ⊎ cliff_core_range`, so IN has one reader-producer + two compute-consumers on
  disjoint node sets — exactly the "multiple bindings on one endpoint, non-overlapping nodes" case the DFB spec header
  blesses. No `allow_instance_multi_binding`.

## Friction
### Gaps
- **DOC BUG — migration_guide.md Example 1 KernelSpec field order.** The example lists designated initializers as
  `.compile_time_args, .runtime_arg_schema, .dfb_bindings, .tensor_bindings, .hw_config`, but the `KernelSpec` struct
  (kernel_spec.hpp) declares **bindings before args**. Copying the guide's order fails `-Werror,-Wreorder-init-list` and
  cost a build cycle. Correct order: `unique_id, source, dfb_bindings, [semaphore_bindings], tensor_bindings,
  compile_time_args, runtime_arg_schema, hw_config`. **Fix the guide example to declaration order.**
- **Compute hardware-config guidance assumes a ttnn `ComputeKernelConfig`.** The recipe's "Compute kernels" section is
  written around `to_compute_hardware_config(arch, config)`. Ops like tilize build a raw metal `ComputeConfigDescriptor`
  in the factory (no ttnn config object), so the helper doesn't apply — you map `ComputeConfigDescriptor` → `ComputeGen1Config`
  by hand. The field-mapping table still applies (`fp32_dest_acc_en`→`enable_32_bit_dest`, `unpack_to_dest_mode`→`unpack_modes`,
  etc.); a one-line note that the "source" may be a metal descriptor rather than a ttnn config would help.
- **`unpack_modes` on ≤16-bit input DFBs — a latent conflict (not hit by the confirmed tests, flagged).** Legacy sets
  `unpack_to_dest_mode[c_0]=UnpackToDestFp32` whenever `fp32_llk_acc` (which is true for FP8_E4M3 input, or BFLOAT8_B/FP8
  output, even when the *input* DFB is ≤16-bit). The faithful map is `unpack_modes={{IN,UnpackToDest}}`. The recipe warns
  the Gen1 validator rejects `UnpackToDest` on a ≤16-bit-format DFB "as a pure perf loss" — which would reject a faithful
  port of those mixed-precision configs. The confirmed test baseline is bf16→bf16 / fp32→fp32 only, so this never fired
  here; but an op whose tests exercise fp8/bf8 tilize could hit a real recipe-vs-validator conflict. Flagging for the doc
  owners to confirm the intended mapping for the ≤16-bit + `fp32_llk_acc` case.
### Confusion
- **"Delegate builds/tests to a subagent" is a footgun as literally written.** A subagent that launches the build with a
  backgrounded shell and then ends its turn lets the build process die (or orphan) — I lost a long stretch to a subagent
  that reported "still running" while its build had actually stalled at cmake-configure and gone. Running the build myself
  in a tracked background shell (output to a log I grep, never read wholesale) kept builds off my context *and* under my
  control. Suggest the recipe's subagent template note that the helper must run the command in the **foreground** and only
  return when it completes (or that the primary run it as a tracked background job instead).
- **Recipe build command vs. repo norm.** The verification step says `cmake --build build_Release --target ttnncpp
  unit_tests_ttnn`; this repo's convention (and my environment's guidance) is to use `./build_metal.sh --build-tests` and
  never hand-target the unity-build test binary. I used `build_metal.sh`. Minor, but a colleague following the recipe
  literally may hit the unity-build staleness the repo norm warns about.
- **pytest venv.** `PYTHONPATH=$(pwd)` alone (recipe/workspace_setup) is insufficient if a *different* venv is active —
  `import ttnn` half-resolves and dies on `ttnn.device`. The op's `python_env` venv must be activated first. A one-liner in
  workspace_setup ("activate `python_env` before running pytest, not just set PYTHONPATH") would save a confused cycle.

## Open items for downstream

### Block factory — deferred, with design (the biggest finding of this port)
**Why deferred.** `TilizeMultiCoreBlockProgramFactory` allocates its CBs (`c_0` input, `c_1` per-row DRAM-alignment
staging, `c_16` output) **per core-range group** — `push_cb_pair` is called once per group (`core_range`,
`cliff_col_row`, `cliff_row`, `cliff_col`) with a **different `num_tiles`** each (full/cliff sizes differ). Legacy gets
away with a single reader/writer `KernelDescriptor` on `all_cores` because legacy CBs are index-based and allocated
per-core-range. **Metal 2.0 DFBs have a scalar `entry_size`/`num_entries`**, so per-group sizes require **per-group DFB
specs**, which require **per-group reader/writer/compute KernelSpecs** (a KernelSpec binds one DFB name). This is
"preserved multiplicity" taken to its conclusion — mechanical and fully expressible, but a materially larger, higher-risk
port than Default/SingleCore, and **the confirmed test baseline barely exercises the Block path** (its one wide-tensor
case runs `use_multicore=False` → SingleCore), so a Block port could not be meaningfully verified in this pass. Per the
recipe's "one factory at a time; deferral is the expected shape for a large op," it is enumerated here.

**Design for the next pass (validated groundwork):**
- The 4 compute core-ranges **exactly partition `all_cores`** (verified in `work_split_tilize.hpp`
  `split_blocks_for_tilize_wh`: each `addCore` inserts a core into exactly one group set + `all_cores`). So per node there
  is exactly one group → exactly 3 DFBs/node (no per-node DFB explosion), and the per-node DFB invariant (1 producer + 1
  consumer) holds within each group.
- For each non-empty group `G`: build `IN_G`/`TEMP_G`/`OUT_G` DFBs (sizes from that group's block sizes), `READER_G`/
  `WRITER_G`/`COMPUTE_G` KernelSpecs, and a `WU_G` over `G`. `TEMP_G` (`c_1`) is a **self-loop** (reader bound PRODUCER +
  CONSUMER — single-toucher staging buffer, matches the audit/brief).
- **RTA routing:** keep the legacy per-core loop verbatim (it computes correct per-core block sizes); route each core's
  reader/writer RTAs to its group's KernelSpec by **CoreRangeSet membership** (`core_range.contains(core)` etc.), which is
  independent of — and consistent with — the legacy loop's value computation.
- **Kernels are already drafted** (written, then reverted/removed this pass to avoid shipping a half-converted factory):
  the Metal 2.0 forms of `tilize_wh.cpp` (own compute), `writer_unary_interleaved_start_id_wh.cpp` (→ `_wh_metal2` fork,
  2 consumers), and `reader_unary_pad_multicore_both_dims.cpp` (→ `_metal2` fork, in-family). The pad reader keeps its
  `fill_with_val` helper + `UnicastEndpoint`/`tt_memmove` logic unchanged; only args/DFB/tensor bindings change. Note the
  block reader binds **two** DFBs (`dfb::in0` = `c_0`, `dfb::in1` = `c_1` self-loop).

### Cross-op kernel touches (coordination for sibling-op ports)
| Kernel (path) | Path taken | Remaining unmigrated consumers |
|---|---|---|
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | **fork** → `writer_unary_interleaved_start_id_metal2.cpp` | ~40 factories (reduction, transformer, kv_cache, embedding, examples, …) + tilize Sharded |
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared pool) | **fork** → `tilize_metal2.cpp` | `tilize_with_val_padding` ×3, tilize Sharded |

The dominant coupling is the 42-consumer writer: its full in-place migration must land across all co-borrowers at once, so
the fork is the correct bulk-port move. When the co-borrowers migrate, collapse the fork and delete the legacy copy.

### Test coverage note
The confirmed baseline (`test_tilize.py`, `test_tilize_untilize_2D.py`, `misc/test_tilize_test.py`) exercises
Default + SingleCore + Sharded but has **thin interleaved-Block coverage** (the wide `[1,1,128,7328]` case runs
single-core). A future Block port should add an interleaved wide+tall multicore case that provably selects the Block
factory, or it will ship under-verified.
