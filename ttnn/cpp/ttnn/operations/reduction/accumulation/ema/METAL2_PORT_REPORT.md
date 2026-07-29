# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/reduction/accumulation/ema`

## Outcome

**`PORTED`.** `EmaDeviceOperation`'s only factory, `EmaProgramFactory`, is converted from
`create_descriptor` (legacy `ProgramDescriptor`) to `create_program_artifacts` (`ProgramSpec` +
`ProgramRunArgs`), together with all three kernel entry points it binds. Nothing is left for a later
pass: the op has one factory, one instantiation shape, and no runtime kernel-source selection.

Verification: the confirmed baseline test set passed **4/4 before** the port and **4/4 after**, on the
same build tree. Build is green with `./build_metal.sh --build-tests`.

## Provenance

- **Recipe docs (this port):** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`
- **Audit docs (inherited):** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## TTNN ProgramFactory

### Concept realized

`MetalV2FactoryConcept` as the audit chose it, with no op-owned tensors. In code today that concept is
spelled **`ProgramSpecFactoryConcept`** ([operation_concepts.hpp:119-121](../../../../../../api/ttnn/operation_concepts.hpp#L119-L121));
the factory satisfies it by defining `create_program_artifacts` and no longer defining
`create_descriptor`. No deviation from the audit's decision.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one, so it keeps the default
  reflection-based hash. (The `UpdateTensorArgs` `TensorSpec`-legality failure mode a surviving custom
  hash produces on the *second* dispatch did not appear; the primary test loops the op twice
  specifically to exercise a program-cache hit, and both iterations pass.)
- **Pybind entry points removed:** none. `ema_nanobind.cpp` exposes only the user-facing `ttnn::ema`
  free function, so no pybind line referenced the vanished `create_descriptor`. The op's Python surface
  is unchanged.
- The only edit outside the factory body is the factory method's own declaration in
  [ema_device_operation.hpp:23-28](device/ema_device_operation.hpp#L23-L28) (`create_descriptor` →
  `create_program_artifacts`) and the matching include swap on
  [:6-10](device/ema_device_operation.hpp#L6-L10) (`<tt-metalium/program_descriptors.hpp>` →
  `"ttnn/metal_v2_artifacts.hpp"`). Both are forced by the concept change. `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors`, `EmaParams` / `EmaInputs`, and the op-level
  `ema_device` entry point are untouched.

### Open items

- **Relaxation candidates:** none applied, and none obviously available. Tensor-arg matching stays
  strict. Worth knowing for whoever revisits this: `compute_output_specs` returns the *caller's*
  optional output spec when one is supplied ([ema_device_operation.cpp:78-80](device/ema_device_operation.cpp#L78-L80)),
  so `INPUT` and `OUTPUT` can legitimately carry different `TensorSpec`s (different memory config) on
  the same input. Strict matching handles that correctly; it just means a preallocated-output call and a
  fresh-output call take separate cache entries, as they did before.
- **Concept fit:** clean. Single program stamped across the mesh, no op-owned resources, so nothing
  about this op wanted a capability the concept lacks.

## Handoff points

1. **Framework gap — the compute-kernel `opt_level` default differs between the legacy
   `ProgramDescriptor` path and Metal 2.0, silently.** *(Owner: Metal 2.0 host-API team.)*

   The legacy `ProgramDescriptor` → `Program` conversion defaults `opt_level` **per kernel kind**:
   `O2` for reader / writer / `DataMovementConfigDescriptor`
   ([program.cpp:419](../../../../../../../tt_metal/impl/program/program.cpp#L419), [:427](../../../../../../../tt_metal/impl/program/program.cpp#L427), [:439](../../../../../../../tt_metal/impl/program/program.cpp#L439))
   but **`O3` for `ComputeConfigDescriptor`**
   ([program.cpp:455](../../../../../../../tt_metal/impl/program/program.cpp#L455)). Metal 2.0 has no
   such per-kind rule: `KernelSpec::CompilerOptions::opt_level` defaults to **`O2`** for every kernel
   ([kernel_spec.hpp:116](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp#L116))
   and `MakeGen1ComputeConfig` forwards it unchanged
   ([program_spec.cpp:2721](../../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L2721)).

   So **every compute kernel ported from a `ProgramDescriptor` factory that did not set `opt_level`
   explicitly silently drops from `-O3` to `-O2`.** It builds, it produces bit-identical numerics, and
   no test catches it — precisely the silent perf-regression shape the recipe's *Hardware configuration*
   section exists to prevent, except that `opt_level` lives on `compiler_options`, not `hw_config`, so
   that section never mentions it and its checklist item does not cover it. This port sets
   `.compiler_options = {.opt_level = KernelBuildOptLevel::O3}` on the compute `KernelSpec`
   ([ema_program_factory.cpp:197](device/ema_program_factory.cpp#L197)) to reproduce the legacy
   value.

   Suggested fix, in preference order: (a) have the Metal 2.0 spec→config conversion apply the same
   per-kind default legacy applies, so faithful ports need no action; or (b) if the `O2`-everywhere
   default is deliberate, add `opt_level` to the recipe's *Hardware configuration* section and to its
   anti-pattern checklist item "Every `hw_config` reproduces the legacy op's resolved values," since a
   porter reading only that section will not look at `compiler_options`.

   Ports already landed are worth re-checking for this: a compute kernel whose `KernelSpec` omits
   `compiler_options` is currently building one optimization level lower than its legacy self.

2. **Boundary-rule assumption violations:** none. No call site outside the op directory required a
   `sem::` or `tensor::` handle. The one out-of-directory kernel include
   (`../../../device/kernels/accumulation_common.hpp`) contributes only the `ONE_TILE` constant, has no
   resource handle in any signature, and was not modified.

3. **Kernel-lib gaps:** none. The compute LLKs that take a `uint32_t` CB id
   (`compute_kernel_hw_startup`, `transpose_init`, `transpose_tile`, `pack_tile`) all accepted
   `dfb::<name>` through the documented `DFBAccessor → uint32_t` conversion with no wrapping and no
   change to the call shapes.

## Successes

- **[Hardware configuration → Data movement kernels](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#data-movement-kernels)
  caught a silent regression I would otherwise have shipped.** The section's insistence on resolving the
  *values* rather than the role name, plus its explicit reader/writer default table, is what surfaced
  that this op **inverts** the conventional RISC assignment: its reader runs on `RISCV_0` and its writer
  on `RISCV_1` ([legacy: ema_program_factory.cpp:147-160 pre-port]), while the Metal 2.0 reader default
  is `RISCV_1`/`NOC_0` and the writer default is `RISCV_0`/`NOC_1`. Both triples therefore match
  *neither* helper. The reflex — and what a nearby already-ported op does — is
  `create_reader_datamovement_config` / `create_writer_datamovement_config`, which would have swapped
  this op's RISC assignment while still passing validation (the RISCs stay distinct) and every test.
  The port instead builds raw `DataMovementGen1Config`s
  ([ema_program_factory.cpp:158-161](device/ema_program_factory.cpp#L158-L161), [:182-185](device/ema_program_factory.cpp#L182-L185)).
  This section earned its length; keep the table.

- **"Re-derive endpoint dispositions, don't transcribe"** ([recipe, Read this first](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)
  plus the [endpoint-assignment procedure](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)).
  Running the census independently was cheap and confirmed all three CBs: `c_0` and `c_1` are two-toucher
  1P+1C with both roles already locked by FIFO ops, and `c_2` has exactly one toucher (the compute
  kernel drives `reserve_back`/`push_back` at [ema_compute.cpp:108](kernels/compute/ema_compute.cpp#L108),
  [:112](kernels/compute/ema_compute.cpp#L112) and `wait_front`/`pop_front` at
  [:115](kernels/compute/ema_compute.cpp#L115), [:119](kernels/compute/ema_compute.cpp#L119)) → self-loop.
  No disagreement with the brief. The catalog's **anti-stacking guard** ("never both self-looped and
  multi-bound") is a good cheap invariant to check at the end; `allow_instance_multi_binding` appears
  nowhere in this port.

- **"Go to the headers first."** `dataflow_buffer_spec.hpp` documents `entry_size` / `num_entries` and
  the per-node endpoint invariant at the field, which made the legacy `total_size` / `page_size` →
  `entry_size` / `num_entries` mapping unambiguous (`num_entries = total_size / page_size`, preserving
  the L1 footprint byte for byte) without needing a precedent. Reading `kernel_spec.hpp` end to end is
  also what put `compiler_options.opt_level` in front of me at all — the field the Handoff point above
  turns on. Recommending the headers over precedent-hunting is the right call.

- **`Table` is a map, not a vector.** The recipe's explicit warning meant the
  `compile_time_args` / `runtime_arg_values` shapes were right the first time, with no `push_back`
  attempt.

## Friction

### Gaps

- **The target concept's name does not exist in code.** Every doc names
  `MetalV2FactoryConcept`; `grep -rn MetalV2FactoryConcept ttnn/` returns zero hits. The concept a
  `create_program_artifacts` factory actually satisfies is **`ProgramSpecFactoryConcept`**
  ([operation_concepts.hpp:119](../../../../../../api/ttnn/operation_concepts.hpp#L119)), with
  `CustomProgramSpecFactoryConcept` as its `override_runtime_arguments`-carrying sibling
  ([:132](../../../../../../api/ttnn/operation_concepts.hpp#L132)). Nothing broke, but the doc name is
  the one a porter greps for when the `AllFactoriesValid` `static_assert` fires, and it leads nowhere.
  Either rename in the docs or note the code spelling in
  [`ttnn_factory.md`](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md).

- **`opt_level` is missing from the hardware-configuration discipline.** See
  [Handoff points 1](#handoff-points). Listing it here too because the doc-side fix is separable from
  the framework-side one: the recipe's *Hardware configuration* section and its checklist item are
  written as though `hw_config` is the whole silent-perf-settings surface, and it is not.

- **Whitelist rule 7 forces a reorder it does not mention.** Rule 7 maps
  `get_tile_size(cb_id)` → `dfb.get_tile_size()`. In this op the legacy line was
  `constexpr uint32_t src_tile_size = get_tile_size(src_cb_idx);` — a *compile-time* constant declared
  above the DFB. The DFB member getter is declared `constexpr`
  ([dataflow_buffer.h:167](../../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L167)) but
  `DataflowBuffer`'s constructor is not, so the result cannot stay `constexpr`: the line becomes
  `const uint32_t` **and** has to move below the `DataflowBuffer` construction
  ([ema_reader.cpp:28-32](kernels/dataflow/ema_reader.cpp#L28-L32)). That is the only non-mechanical
  part of an otherwise 1:1 swap, and one sentence in rule 7 ("the value is no longer a compile-time
  constant, so the declaration moves below the DFB object") would cover it.

### Confusion

- **`TT_KERNEL` reads as mandatory and is not.** The recipe says the port adds exactly one argument
  header, `experimental/kernel_args.h`. The first thing that header documents is a `TT_KERNEL` macro:
  *"marks the named-arg entry point; the JIT generates kernel_main() from its signature"*
  ([kernel_args.h:44-47](../../../../../../../tt_metal/hw/inc/experimental/kernel_args.h#L44-L47)) — and
  the JIT genuinely scans for it. Since the recipe never mentions `TT_KERNEL` and its kernel examples
  keep a plain `void kernel_main()`, the two read as contradicting each other. Resolving it took reading
  [kernel_signature_parser.hpp:32-37](../../../../../../../tt_metal/jit_build/kernel_signature_parser.hpp#L32-L37)
  to find that a source with no `TT_KERNEL` marker is *"a legacy / hand-written kernel_main() — fully
  backward compatible."* One line in the kernel-side whitelist saying a hand-written `kernel_main()`
  stays correct and `TT_KERNEL` is an optional newer form would remove the doubt for every porter.

- **No guidance for the one case where host and kernel already disagree on an argument's name.** A
  named CTA needs a single name shared by both sides. Compute CTA slot 0 is read as
  `total_batches_per_core` by the kernel ([ema_compute.cpp:72](kernels/compute/ema_compute.cpp#L72)) and
  passed as `total_batch_channel_tiles_per_core` by the host
  ([ema_program_factory.cpp:220](device/ema_program_factory.cpp#L220)) — the audit flagged the kernel
  name as off by the channel-tile factor (its anomaly 2). The recipe's two relevant instructions pull
  opposite ways: rule 4 says pick the name matching the variable it is assigned to (→ the kernel's),
  while Principle 3 says pick a name reflecting what the value is (→ the host's, the accurate one). I
  used the kernel's name, so the port renames nothing on either side and the mismatch stays visible on
  one host line for the ops team rather than being quietly resolved by the porter. A sentence in rule 4
  on this tie-break (prefer the existing name on the side you are *not* rewriting; route the
  disagreement to the report) would make it a decision instead of a judgement call.

- **DFB naming when the two sides use different words is under-documented, though the API handles it.**
  `DataflowBufferSpec::unique_id` and `DFBBinding::accessor_name` are independent, and the migration
  guide notes the accessor name is "independent of the producer's name" — but no doc says what to do
  when the legacy *host* and legacy *kernel* have different words for one buffer (here `prev` vs `trp`).
  Keeping the host's word as `unique_id` and the kernel's as `accessor_name` renames nothing and reads
  well; it took a paragraph of reasoning to be confident that was intended rather than sloppy. Worth an
  explicit line, since the two-name design makes this the natural answer.

## Open items for downstream

- **Shared kernel touches:** none. The census
  (`grep -rl <filename> ttnn/cpp/ttnn/operations/` for each of the three kernels) returns exactly one
  code consumer each, this op's own factory; no `_metal2` fork existed beside any of them and none was
  created. All three kernels were converted **in place**. Nothing to sunset, no remaining unmigrated
  consumer, no pointer comment left anywhere.

- **Sibling-op carry-over.** The parent directory `reduction/accumulation/` also holds
  `AccumulationDeviceOperation` (serving `cumsum` / `cumprod`), still on the legacy API. It shares no
  factory and no kernel `.cpp` with EMA — only the constants header
  `accumulation/device/kernels/accumulation_common.hpp` — so it ports independently and this port
  neither helps nor blocks it. Two things a porter of that op should know: its kernels live under
  `accumulation/device/kernels/` (EMA's are at the op root, `ema/kernels/`), and the accumulation branch
  named in the recipe as the first worked `create_program_artifacts` example is a port of *that* op.

- **Test coverage the verification step surfaced but did not act on.** EMA has no C++ gtests, no
  sweeps, and no nightly variants; its whole coverage is the 3-case
  `tests/ttnn/unit_tests/operations/reduce/test_ema.py` (which does loop each case twice, so the
  program-cache-hit path *is* covered) plus the `out=`-path docs example. Two gaps that matter for
  anyone changing this op again: only `bfloat16` and only interleaved memory are exercised. The dtype
  limit is enforced, so that one is fine; the memory-layout limit is **not** — see the next item.

- **Audit anomalies carried forward, untouched by this port.** All six are the ops team's calls, not the
  porter's; recorded here so they survive the audit document:
  1. `c_2`'s host-side name (`prev_cb_index`) describes a role it does not have. It stages one tile
     through SRAM for a second transpose; the previous EMA output lives in an SFPU register cleared by
     `ema_clear_previous_output()`. The kernel's `trp` is the accurate word. The port preserves both
     names in their existing places (see Friction) rather than renaming.
  2. Compute CTA slot 0 is named for batches but counts batch×channel-tile rows. The loop is correct;
     only the name is off. Now visible on one line in the ported factory.
  3. The docstring states `output_0 = input_0` ([ema_nanobind.cpp:27-29](ema_nanobind.cpp#L27-L29)) but
     the kernel zeroes the previous-output register at the start of each sequence, making the first
     output `(1−α)·input_0`. The test's golden model matches the *kernel*, not the docstring. Either the
     docstring or the first-sample handling is wrong.
  4. The docstring restricts memory support to interleaved but nothing enforces it: a sharded input or
     output is accepted and runs (the kernels go entirely through `TensorAccessor`). Unchanged by the
     port — and note there is no test in either direction, so whichever way the owner resolves it wants
     a test.
  5. `alpha` is validated for NaN only ([ema_device_operation.cpp:60](device/ema_device_operation.cpp#L60));
     infinities and values outside `[0, 1]` are accepted.
  6. `accumulation_common.hpp` defines `CB_IN` / `CB_OUT` / `CB_ACC` (= `c_0` / `c_1` / `c_2`) which the
     EMA kernels pulled in but never used, declaring their own indices instead. **The port has partly
     retired this hazard on its own:** the EMA kernels no longer declare CB indices at all (they use
     `dfb::` handles), so the duplicate-naming confusion is gone from the EMA side. The unused constants
     still arrive via the include, which the EMA kernels keep only for `ONE_TILE`.

- **Doc-evolution suggestion beyond the Gap entries.** The recipe's anti-pattern self-audit checklist is
  the natural home for a "diff the *whole* legacy kernel config, not just `hw_config`" item, since
  `compiler_options` (opt_level, defines, include_paths) is a second silent surface with the same
  properties: no build signal, no test signal, real effect. This port needed one of those three
  (`opt_level`); a port of an op that set `defines` or `compiler_include_paths` on its legacy descriptor
  would need the others.
