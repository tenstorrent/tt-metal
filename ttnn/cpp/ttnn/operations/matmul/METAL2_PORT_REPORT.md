# Port Report — `ttnn/cpp/ttnn/operations/matmul`

Post-port report for **one** factory — `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`.
The op's other seven factories are untouched and stay on their legacy concepts.

## Outcome

**`PORTED`** — `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` converted to the base
`ProgramSpecFactoryConcept`; its confirmed test set passes with results identical to the
pre-conversion baseline. Seven factories remain: `MatmulMultiCoreProgramFactory` and
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` are already ported (#55224, #55961);
`MatmulMultiCoreReuseOptimizedProgramFactory` and `MatmulMultiCoreReuseMcast2DProgramFactory`
are cleared GREEN and queued; `MatmulMultiCoreReuseMcast1DProgramFactory`,
`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` and
`SparseMatmulMultiCoreReuseMcast1DProgramFactory` are blocked upstream.

Stacked on #55961 (`iwrosz/port-mm-dram-sharded` @ `a7796e0f567`).

## Provenance

- **Recipe docs (this port):** `git log -1 … -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  prints **nothing** on this branch — the branch is stacked on `main`, which does not carry the
  metal_2.0 doc tree, so the version cannot be pinned from the checkout. The docs were read from
  `origin/akertesz/op-porting-recipe`, whose head for that path is
  `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

The audit therefore ran two doc commits behind this port. Nothing in the delta changed a decision
here; recorded because the brief's rung verdict *was* stale for a different reason (below).

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` (base), as the audit chose. `create_descriptor`
  → `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. No
  `override_runtime_arguments` added — the factory had none and the base concept has the framework
  refresh tensor bindings on cache hit.
- **Custom `compute_program_hash`:** none framework-visible; left intact. The deliberately-renamed
  `compute_descriptor_program_hash` at `device/matmul_device_operation.hpp:50` and its pybind alias
  are untouched.
- **Pybind entry points removed:** one — see Handoff points.
- **Open items:** the `_UNSUPPORTED_FACTORY` gap in `models/experimental/ops/descriptors/matmul.py`
  widens by one factory, as it did for #55224 and #55961. Left unguarded for the same reason: the
  general fix is a `main`-targeted change belonging with the decision #55224 raised, which is still
  open. No in-tree caller passes a batched DRAM-sharded config to the descriptor `matmul()`.

## Handoff points

1. **Removed pybind surface — API surface: removed entry point.**
   `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`, the
   `nb::class_<ttnn::prim::MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory>` block (16
   lines) whose only member was `create_descriptor`. It exposed the factory's legacy
   `ProgramDescriptor` builder to Python — the entry point the port makes vanish. Out-of-tree
   tooling calling
   `ttnn.MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.create_descriptor(...)` by name now
   gets an `AttributeError`; a `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig`
   routed through the still-bound `matmul_select_program_factory` fails one step earlier, with a
   nanobind `TypeError` on the unregistered variant alternative. No in-tree caller does either.
   The `nb::class_<MatmulDeviceOperation>` block is untouched.

2. **Dropped a pybind-hook-only parameter.** `create_descriptor`'s fourth argument
   `const std::optional<CoreRangeSet>& core_range_set` existed only for that hook and was ignored
   by the factory body (spelled `/*core_range_set*/`). Dropped with the method; no production
   default to inline, since nothing read it.

No capitulation. No boundary-rule assumption violations: no call site required `sem::name` or
`tensor::name` outside the op directory. No kernel-lib gaps. No framework gaps beyond the
`unpack_modes` note under Friction.

## Successes

- **The shared-kernel Caution's rung ordering, and its "run rung 1 locationally" instruction.** The
  audit brief says rung 2 — create the fork — and it was right when written. Running the rung-1
  check as the catalog specifies (`ls` the original's directory for a `_metal2` sibling) found the
  fork that #55961 created in the meantime, so this port creates nothing and the fork gains its
  second consumer instead of a duplicate. Following the brief's verdict instead of re-running the
  check would have produced a second fork and an add/add conflict at merge.

- **"A fork that already has a consumer is read-only to you", and the fit check it demands.** The
  fork's `dfb::` names (`in0`, `in1`, `bias`, `out`, `intermed0`) and its named-arg set matched
  this factory's legacy CB roles and CTA list 1:1, including the two `#ifdef`-gated branches this
  factory must leave undefined. The D1 instruction to name the first fork's bindings for the
  *kernel* rather than for the first consumer is what made that fit possible — the vocabulary
  contains nothing DRAM-sharded-specific to work around.

- **The two-aliasing-patterns table.** This factory has a host-side `alias_with` pair and a
  kernel-side same-FIFO alias on the *same* CB pair, and the host-side one is config-dependent.
  The catalog's side-by-side table is what kept them apart: modelling the kernel-side
  `constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;` with `alias_with` would have produced
  two independent FIFOs at one address, and the failure would have been silent. Nothing needed
  doing on the kernel side — the fork already expresses it as a `constexpr` handle alias — but the
  table is why that was recognised as correct rather than "re-derived" into a bug.

- **"Distrust a `(0,0)` CB" plus the one-way-safety-net framing.** The audit reported two dead CBs
  and asked for owner confirmation because nothing catches a wrongly-dropped *live* CB. Chasing the
  question rather than trusting the census produced the actual reason the two exist — see the
  dropped-CB entry under Open items — which is a better answer than either the audit or the port
  would have reached alone.

- **The `opt_level` check, run as a command rather than eyeballed.** `grep -n opt_level` on the
  legacy factory returns nothing, which is exactly the shape the recipe warns about: the defect is
  an absent line. The compute `KernelSpec` carries an explicit `O3`; both DM specs carry none.

## Friction

### Gaps

- **The Case 2 bridge is specified against a DRAM tensor; two of these bindings are L1 shards
  addressed through a NOC endpoint.** The audit flagged this as a recipe gap and asked the porter
  to confirm the bridge value fits before assuming. It does, and the confirmation is one line of
  header reading rather than an experiment:
  `TensorAccessor::get_bank_base_address()` (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:186`)
  returns the accessor's `bank_base_address` verbatim, and the legacy value it replaces was the
  same number — a `MeshTensor` pushed into a `KernelDescriptor::RTArgList`
  (`tt_metal/api/tt-metalium/program_descriptors.hpp:178`) is resolved by the framework to its
  buffer's device address. An L1 shard sits at the same offset on every core holding one, so the
  base is equally valid for `{.noc_x, .noc_y, .addr}` against a *remote* storage core, which is how
  both of these are used. Worth a sentence in the Case 2 section: the bridge returns a buffer base
  address, not a bank-relative or otherwise transformed value, so it fits any legacy site that
  consumed `buffer()->address()` — bank-relative or not.

- **"The port adds exactly two headers" does not hold for a Case 2 kernel that had no
  `TensorAccessor` before.** The kernel-side whitelist names `experimental/kernel_args.h` and
  `api/dataflow/dataflow_buffer.h`, and says `TensorAccessor` "comes from the same headers before
  *and* after, so you neither add nor touch their includes." That assumes the kernel already
  constructed one. Neither of these kernels did — the audit's own finding is "no kernel in this
  factory constructs a `TensorAccessor` at all" — so introducing the bridge requires
  `api/tensor/noc_traits.h` as well, a third added header. #55961's in1 kernel needed the same
  addition. Suggest the whitelist say "two headers, plus `api/tensor/noc_traits.h` when the port
  introduces a `TensorAccessor` the kernel did not have (every all-Case-2 kernel)."

- **`unpack_modes`: the brief's rule and the framework's requirement point in opposite
  directions.** The brief says an entry is needed "**only if** `interm0` resolves to `Float32` with
  `enable_32_bit_dest` on" — i.e. treat it as conditional. `compute_hardware_config.hpp:119-121`
  states the requirement the other way round: `UnpackToSrc` is assumed when unspecified, *but* an
  explicit mode is mandatory for a 32-bit-format DFB once `enable_32_bit_dest` is set, enforced by
  validation. Since `fp32_dest_acc_en` both sets `enable_32_bit_dest` and forces
  `interm0_data_format = Float32`, a conditional entry means the factory has to re-derive the
  framework's own trigger condition to decide whether to speak. Stating `UnpackToSrc` for every
  compute-consumed DFB unconditionally is behaviour-identical (it is the default, and the legacy
  `ComputeConfigDescriptor` left `unpack_to_dest_mode` empty) and cannot get the condition wrong.
  That is what this port does, matching #55961. Suggest the guidance say "state `UnpackToSrc`
  explicitly for every DFB the compute kernel consumes from" rather than making the porter
  reconstruct the gate.

- **Short legacy arg lists on idle cores have no expressible form, and the recipe does not name the
  workaround.** The legacy factory emits a 1-element RTA list on idle cores against 4 and 8 on
  workers; Metal 2.0 requires every schema name to have a value on every node the kernel runs on.
  The port zero-fills. #55961 hit the identical shape and reported it, so this is now the second
  occurrence in one op — it is a recurring DRAM-sharded idiom (a kernel that returns on an
  `is_worker_core` flag before reading anything else) and deserves a named pattern rather than
  each porter deciding independently that zero-filling is legitimate.

### Confusion

- **The audit brief's rung verdict was stale, and nothing in the brief says a verdict can go
  stale.** "The rung-1 check was run **locationally** … So **create it beside the original**" reads
  as an instruction, and it was correct at audit time. What makes it safe is a sentence elsewhere
  in the catalog ("Look for one before doing anything else"), not anything in the brief. A brief
  issued before a sibling port lands is *expected* to be stale on exactly this point when ports
  are sequenced through one shared kernel, and the plan for this op sequences five of them.
  Suggest the audit template mark the rung verdict explicitly as "as of audit time — re-run the
  locational check", so the brief tells the porter what the catalog currently has to.

## Open items for downstream

### Shared kernel touches

| kernel | rung taken | remaining unmigrated consumers |
|---|---|---|
| `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **rung 1 — reused the existing fork** `…_metal2.cpp` (created by #55961). No new file; the legacy original already carries #55961's pointer comment and was not touched. | `matmul_multicore_reuse_optimized_program_factory.cpp`, `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (hosts **two** factories), `matmul_multicore_reuse_mcast_2d_program_factory.cpp`, `sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp` |
| `matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | converted in place — this factory is its sole binder | none |
| `matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | converted in place — this factory is its sole binder | none |

This port removes one name from #55961's list of six consumers of the legacy compute kernel; four
remain (five factories, since the mcast-1d file hosts two).

### Fork drift, reported not fixed

The catalog requires diffing an adopted fork against its original as the first step, and #55961's
own PR description repeats it. The diff shows one divergence beyond the Metal 2.0 conversion:

- Original `bmm_large_block_zm_fused_bias_activation.cpp:554` (the `FUSE_BIAS` path) calls
  `apply_activation_from_pack<activation_type, activation_param0, activation_param1, activation_param2>(out_subblock_num_tiles)`.
- Fork `bmm_large_block_zm_fused_bias_activation_metal2.cpp:571-588` carries that helper's body
  inlined — the `PACK(TTI_SEMWAIT(...))`, `PACK(TT_SETC16(...))`, `ActivationApplyHelper<...>::apply(i)`
  loop and `PACK(TTI_STALLWAIT(...))` — with the same four template arguments.

The helper is `bmm_fused_activation.hpp:109-122` and the inlined block is byte-equivalent to its
body, so **behaviour is identical**; the other call site (original:391 / fork:408) already uses the
helper. It is a *form* divergence: the fork was cut before the extraction, and #55961's rebase sync
carried the new `activation_param2` argument into the inlined copy rather than adopting the
extracted call. Not fixed here — rung 1 makes the fork read-only to this port, and the invoker
confirmed report-only. **Ports 4 and 5 inherit it**, and it is the kind of divergence that makes the
next sync harder to reason about, so it is worth closing on the base branch or in a follow-up.

### Findings preserved, not fixed

Everything below is carried across verbatim. None is introduced by the port.

- **`FUSE_BIAS` is unreachable through the public API for this factory.**
  `ttnn/cpp/ttnn/operations/matmul/matmul.cpp:170-172` routes bias to a post-processed `ttnn::add()`
  whenever in1 is batched, and this factory's in1 is always `[1, B, K, N]` — batch-sharded across
  DRAM banks, so `B > 1` in every non-degenerate config and `is_input_batched` is always true.
  The entire bias branch — the `c_3` bias buffer, the `in3_page_size` / `in3_num_pages` /
  `in3_block_tiles` CTAs, the bias read at in1 kernel:107-118, `row_broadcast_bias`, and the fork's
  whole `FUSE_BIAS` section — is therefore dead unless the device op is invoked directly. Confirmed
  empirically: a `ttnn.linear` with a bias on this config never reaches the factory (it fails
  later, inside `binary_ng`, on the fallback add). The port converts it faithfully as a conditional
  binding, but an owner should decide whether it is a half-built feature or removable.

- **Six dead compile-time args and one dead runtime arg in the two DM kernels.** Each is declared
  and never used; verified by word-boundary occurrence count (1 = the declaration alone).
  - in0 reader: `in0_shard_size_bytes` (CTA 5).
  - in1 writer: `in1_num_pages` (CTA 1), `in1_block_w` (CTA 2), `out_shard_size_bytes` (CTA 9),
    `in3_page_size` (CTA 10), `in3_num_pages` (CTA 11), and RTA `vc` (slot 4).

    `vc` is the notable one: the legacy DRAM-sharded readers pass a virtual channel into
    `noc_async_read_tile_dram_sharded_set_state`, and this kernel's Device 2.0 `noc.async_read`
    call takes none — so the host still computes a per-worker `vc` (with the collision-avoidance
    walk over previous workers) that nothing consumes. The computation is preserved because
    removing it is a functional change; it is a candidate for deletion by an owner.

  All are kept as named args with their kernel-side declarations intact: the declaration *is* the
  read, so the binding is exercised and only the resulting constant is unused — exactly the status
  quo, which compiles warning-free today. Dropping them would change the dispatched payload for no
  port reason.

- **A provably-unreachable branch in the shared compute kernel.** The fork tests
  `mm_partials_reload_dfb_id != mm_partials_dfb_id` inside `reload_from_dfb_to_dst`, now under
  `#ifdef MM_PARTIALS_RELOAD_ALIAS`, which no factory defines. Pre-existing; the file is shared, so
  not this port's to remove.

- **Two named CTAs the compute kernel never read** (`cb_in0_intermediate` → `c_8`,
  `cb_in1_intermediate` → `c_9`) were dead on both ends — no `CBDescriptor` allocated either index
  and the kernel referenced neither under any `#ifdef`. Dropped, as #55961 dropped the same pair.
  Flagged in case they are a half-removed feature rather than leftovers.

### The two dropped buffers, and why they existed

Recorded here because it is the answer to the audit's open question, and because it is the one
outcome nothing would have caught if it were wrong.

- `c_2`, borrowed from in0, on the input storage cores —
  `matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp:247-259` pre-port.
- `c_6`, borrowed from the output, on the output storage cores — same file, `:319-331` pre-port.

Both had **zero endpoints**: neither index appeared in the factory outside its own
`CBFormatDescriptor` (`:253` and `:325`, one hit each), no named CTA on any of the three kernels
carried either, and no kernel body referenced them. Metal 2.0 rejects a buffer with no producer and
no consumer, so the drop was the only expressible outcome.

**They are vestiges of an idiom the sibling factory still uses live.** In
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` the same two borrowed buffers *are* the
mechanism for obtaining a resident shard's L1 base address: its in1 writer calls
`dfb_out_reshard.get_write_ptr()`
(`matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:219`) and its
in0 sender constructs `DataflowBuffer dfb_in2(dfb::in0_sharded)`
(`…_in0_sender_dram_sharded.cpp:61`) — sync-free one-touchers, which #55961 self-looped for exactly
that reason. BatchedHS instead takes both addresses from runtime args (in0 reader slot 3, in1
writer slot 7) and never names either buffer. So the declarations are what was left behind when the
kernels were rewritten to use an explicit address.

Because both are *borrowed* views onto already-resident tensors, dropping them allocates and frees
nothing: such a buffer points at the tensor's own L1 shard rather than taking an allocation of its
own. Confirmed with the invoker before the drop landed; the evidence above also rides the PR
description so the op owner sees it at review.

### The port widens where `out` and `intermed0` are allocated

A consequence of the concept, not a choice, and the one place the port is not literally 1:1 with
the legacy descriptors — so it is called out for reviewers.

Legacy scoped `c_4` and `c_5` to `all_worker_cores` while placing all three *kernels* on
`all_cores_in_rect_grid` (the bounding box, which also contains cores that are neither worker nor
storage). That worked because both DM kernels and the compute kernel return on their worker flag
before touching those CBs. Metal 2.0 has no per-DFB core range: a DFB's node set is derived from
its bound kernels' `WorkUnitSpec`s (`advanced_options.hpp:170-171`), and all three kernels sit in
the one work unit over the bounding box. So `out` and `intermed0` are now allocated on every core
in the bounding box rather than on the workers alone. (`in0`, `in1` and `bias` were already
rect-grid in legacy, so they are unchanged.)

**This does not raise the op's L1 high-water mark.** Worker cores are a subset of the bounding box
and already carried all five buffers, so the peak per-core footprint is unchanged; only the idle
cores in the rectangle now reserve the same L1 the workers do, and nothing else in the op uses
their L1. No OOM risk is introduced. Expressing the legacy narrowing would require splitting each
kernel into worker and idle `KernelSpec`s over two `WorkUnitSpec`s — inventing a work split the
legacy factory does not have — so the port takes the widening, as #55961 did.

### Test coverage gaps

Confirmed with the invoker before the baseline was relied on. This factory is never auto-selected —
`create_matmul_program_config` excludes the config type — and
`tests/ttnn/unit_tests/gtests/test_matmul.cpp:19-29` **deliberately excludes** it from the gtest
smoke set, pointing coverage at the deepseek pytests and noting in the file that "this factory can
still regress into main."

Its entire committed coverage is two pytests, and both run `fused_activation=None`, no bias,
default compute config and bfloat16 output. That leaves three host-side branches of the port with
**no committed test**:

| branch | covered by a committed test? | exercised during this port? |
|---|---|---|
| `interm0_data_format == output_data_format` → `out`/`intermed0` alias pair | yes | yes |
| `interm0_data_format != output_data_format` → independent buffers, `Float32` partials, mandatory explicit unpack mode | **no** | yes — scratch probe, `fp32_dest_acc_en=True` |
| `SFPU_ACTIVATION` → `activation_type` / `activation_param0..2` | **no** | yes — scratch probe, GELU |
| `FUSE_BIAS` → bias buffer, bias `TensorParameter`, `in3_*` args | **no** | **no** — unreachable, see above |

The probe was a throwaway pytest run pre- and post-conversion and is **not committed** (adding
tests is outside the port's scope). It matched on both sides. The gap itself is pre-existing and
worth an owner's attention: two of the three uncovered branches are reachable from Python today and
nothing in CI would notice if they broke.

## Verification

Blackhole p150b, `TT_METAL_WATCHER=10` on every run, each suite run once before conversion and
once after. Watcher log clean throughout (0 error / fatal / hang / assert lines). Test set
discovered by the porter and confirmed by the invoker before it was relied on.

| test | pre-conversion | post-conversion |
|---|---|---|
| `test_matmul_deepseek.py -k batched_dram_sharded` (this factory's entire committed coverage: 5 shape cases + 2 program-cache cases) | **7 passed** | **7 passed** |
| `unit_tests_ttnn --gtest_filter='*MatmulSmoke*'` (stacked base: #55961's factory + the other matmul paths) | **22 passed** | **22 passed** |
| `nightly/…/matmul/test_matmul_dram_sharded.py` + `test_matmul_activations.py` (stacked base, full files) | **373 passed** | **373 passed** |
| scratch branch probe — `fp32_dest_acc_en` off/on, fused GELU, fused bias | **3 passed, 1 xfailed** | **3 passed, 1 xfailed** |

`./build_metal.sh --build-tests`: 0 errors, 2 warnings (both pre-existing third-party Tracy), both
before and after. The ported factory compiled clean on the first attempt.

**The ported path is confirmed taken, not silently bypassed.** The `METAL2_CHECKS_FORCED` markers
are a per-`Program`-construction counter, so their arithmetic doubles as proof of which path ran:
the 7 batched tests produced exactly **7 `program_spec.cpp` + 7 `program_run_args.cpp`** markers —
one `ProgramSpec` build and one `SetProgramRunArgs` per test. The probe produced 8 across its 4
tests, so the `fp32_dest_acc_en` and fused-activation branches went through the validated Metal 2.0
path too.

The stacked base's suites are in the set on purpose: after this port **two** factories compile the
same shared `_metal2` compute kernel, so re-running #55961's baseline is what proves the second
consumer did not disturb the first. Both match it exactly.

Not runnable on this box (single-device): the model-level paths at
`models/demos/deepseek_v3/tt/mla/mla1d.py:760,822` and `models/demos/deepseek_v3/tests/test_mla.py`,
which need multiple devices. Recorded so the gap is explicit rather than implied.

## Legality checks

All 9 `skip_validation` sites across `tt_metal/impl/metal2_host_api/program_run_args.cpp` and
`program_spec.cpp` were forced false, with one `METAL2_CHECKS_FORCED` marker per file (deliberately
not in `UpdateProgramRunArgs`, which fires on every cache hit). Both markers appear in runs of this
factory's own tests against the final binary, so the spec validator and the run-args validator were
live for every result above — including the cache-hit path the program-cache test exercises, since
`UpdateTensorArgs` shares a translation unit with the marked `SetProgramRunArgs`. The scaffolding
was reverted before committing: `git diff --name-only $(git merge-base origin/main HEAD)` lists no
`tt_metal/` file, and a diff-wide grep for `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` is empty.

## Anti-pattern self-audit

Run over the 4 files this port changes, with `find ttnn/cpp/ttnn/operations/matmul -name '*.cpp' -o
-name '*.hpp' | wc -l` = **56** as the op-directory denominator. All checks pass.

| check | result |
|---|---|
| buffer address / `Buffer*` / `emplace_runtime_args` surviving in run-args | 0 hits |
| legacy `CBIndex` / `CBDescriptor` / `CBFormatDescriptor` / `ProgramDescriptor` / `KernelDescriptor` in the factory | 0 hits |
| `TensorAccessorArgs<N>()` in either ported kernel | 0 + 0 hits |
| `cb`-shaped leftovers (`[Cc][Bb]_`, `_[Cc][Bb]`, `\bCB[A-Z]`) across the 4 files | 0 hits — one comment hit found and reworded |
| `.id` extraction on a `dfb::` handle | 0 hits |
| `allow_instance_multi_binding` | 0 hits — the census fits 1P+1C everywhere, with one compute self-loop |
| positional `compile_time_args` | 0 hits — every CTA named |
| `get_vararg` / `num_runtime_varargs` | 0 hits — every argument is a distinct field read once |
| `opt_level` | 1 setting on the 1 compute `KernelSpec`; both DM specs carry none |
| CTA→RTA demotion | n/a — no work-split multiplicity to demote |
| every legacy `TT_FATAL` / `TT_ASSERT` / `TT_THROW` accounted for | per-file count diff against the merge-base: **no delta** |
| ephemeral `.md` cited from code | 0 hits over 13 files scanned |
| `hw_config` reproduces the legacy resolved values | verified field-for-field — see below |
| conditional DFB bindings follow the pattern | `bias` is conditionally bound, `FUSE_BIAS` rides `compiler_options.defines`, and both kernels `#ifdef`-gate every reference |

**`hw_config` field-for-field.** DM: `DataMovementConfigDescriptor{RISCV_1, in0_noc}` →
`DataMovementGen1Config{RISCV_1, in0_noc}` and `{RISCV_0, in1_noc}` → the same, neither setting
`noc_mode` before or after. Compute: the legacy `ComputeConfigDescriptor{math_fidelity,
fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` maps onto
`to_compute_hardware_config(arch, config)`'s `fpu_math_fidelity`, `enable_32_bit_dest`,
`double_buffer_dest = !dst_full_sync_en` and `sfpu_precision_mode`. The two fields the checklist
says the helper does not cover are both accounted for: legacy `bfp8_pack_precise` was left at its
`false` default, which is the same setting as Metal 2.0's default
`bfp_pack_precision_mode = Approximate`; and `unpack_modes` is set explicitly (above). Legacy
`enable_trisc2_rvv` was also left default and has no Gen1 counterpart to carry.
