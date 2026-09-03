# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/matmul`

**Factory ported: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`** — one of the op's eight
factories. The other seven stay on their legacy concepts; the `program_factory_t` variant dispatches
per factory, so the op builds and runs with this one on `ProgramSpecFactoryConcept` and the rest on
`ProgramDescriptorFactoryConcept`.

## Outcome

`PORTED` — the factory converted and the confirmed test set passes, matching the pre-conversion
baseline case for case.

## Provenance

- **Recipe docs (this port):** `b419a49b934 2026-09-01 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

The port ran one doc commit ahead of the audit. The single intervening change broadened the
conditional-binding pattern to tensors and semaphores, which this port uses for the bias tensor.

## Verification

Target: Blackhole. Every run with `TT_METAL_WATCHER=10`. The test set below was discovered by the
porter and **confirmed by the invoker** before it was relied on, per the recipe's
no-regression-baseline checkpoint; it was then run once before conversion and once after, on the
same box.

**Legality checks were forced and proven live.** `grep -n 'bool skip_validation'
tt_metal/impl/metal2_host_api/*.cpp` named **9** sites across two files; all 9 were forced, with one
`METAL2_CHECKS_FORCED` marker per file (`program_run_args.cpp` in `SetProgramRunArgs`,
`program_spec.cpp` in `BuildProgramFromSpec`; deliberately not in `UpdateProgramRunArgs`). Both
markers appear in a run of the ported factory's own test against the final binary — one each, which
is one Program construction — so the spec validator and the run-args validator were both live for
every result below. The scaffolding was reverted before committing and is absent from the diff.

*(Pre-conversion, the markers were **zero** on the matmul tests, correctly: no matmul path built a
Metal 2.0 spec yet. They were proven live at that stage against an already-ported op's tests
instead — `test_pad.py::test_pad_rm_sharded_stickwise`, 64 marker lines from both files — which is
what established that the force had actually reached the binary rather than just the source.)*

| test | pre-conversion | post-conversion |
|---|---|---|
| `unit_tests_ttnn --gtest_filter='*MatmulSmoke*'` (incl. `DramShardedDecodeProjection`, written against this factory) | 22 passed | **22 passed** |
| `nightly/…/matmul/test_matmul_dram_sharded.py` (all; `has_bias` × `num_workers_per_dram_bank ∈ {1,2,3}` × dtypes, incl. `…_with_program_cache`) | 84 passed | **84 passed** |
| `nightly/…/matmul/test_dram_sharded_then_1d_matmul.py` | 1 passed | **1 passed** |
| `unit_tests/…/matmul/test_matmul.py -k tiny_tile / multiple_out_block_values` | 26 passed, 10 skipped | **26 passed, 10 skipped** |
| `unit_tests/…/matmul/test_linear.py -k …bias_dram_sharded_in1` | 4 skipped | **4 skipped** |
| `unit_tests/…/matmul/test_experimental.py -k dram_sharded` | 1 passed | **1 passed** |
| `unit_tests/…/matmul/test_matmul_deepseek.py -k l1_dram_sharded / single_kblock` | 1 passed, 10 skipped | **1 passed, 10 skipped** |
| `nightly/…/matmul/test_matmul.py -k tiny_tile…rejected` | 4 passed | **4 passed** |
| `nightly/…/matmul/test_matmul_activations.py -k different_program_configs` | 10 passed | **10 passed** |
| `tt_eager/…/misc/test_sharded.py -k llama_mlp_width_sharded…` | 1 passed | **1 passed** |

150 passed, 24 skipped, 0 failed — identical on both sides. Every skip is pre-existing and
attributable to a named unrelated issue (`test_linear`'s four to #31385, tiny-tile matmul on
Blackhole; the rest to dtype/arch guards in the tests themselves).

**Build:** clean via `./build_metal.sh --build-tests` — 0 errors and 0 warnings, with the
`ttnn_op_matmul` unity objects confirmed rebuilt in the log. No deprecation warning fired for the
`SemaphoreAdvancedOptions::initial_value` use.

**Anti-pattern self-audit:** every "expect zero" sweep run over the **converted file set** (5 files
— denominator printed, non-zero) rather than the op directory, for the reason under Friction. Zero
hits for: buffer addresses / `emplace_runtime_args` / bare `Buffer*` in run-args; `CBIndex::c_*` in
CTAs; `TensorAccessorArgs`; `cb`-shaped names (host variables, spec-name strings, kernel-side
references and comments all swept); `.id` extraction on a `dfb::` handle; `allow_instance_multi_binding`;
legacy `CBDescriptor` / `CircularBuffer` / `get_local_cb_interface` / `ProgramDescriptor` /
`KernelDescriptor` references; and positional `get_compile_time_arg_val` / `get_named_compile_time_arg_val`
/ `get_arg_val<>` / `get_arg_addr` in the converted kernels.

- **TT_FATAL census:** the per-file count diff across the port is **empty** — no guard moved,
  dropped or added anywhere in the op directory.
- **No ephemeral doc cited from code:** zero `.md` references across the 9 `.cpp`/`.hpp` files the
  diff and the untracked set touch.
- **`opt_level`:** the factory builds exactly one compute `KernelSpec` and `grep -n opt_level`
  returns exactly one line, `O3`, on it. Both DM specs correctly carry none (legacy default `O2`).
- **`hw_config`:** both DM specs replicate the legacy resolved triples verbatim
  (`RISCV_1`/`in0_noc` and `RISCV_0`/`in1_noc`, `noc_mode` default `DM_DEDICATED_NOC` on both
  sides), with no helper substituted. The compute config goes through
  `to_compute_hardware_config`, which was read rather than assumed: it and the legacy
  `get_compute_kernel_config_args` read the *same* four raw fields off the same
  `ComputeKernelConfig` — neither adjusts by arch — so `fpu_math_fidelity`,
  `sfpu_precision_mode`, `enable_32_bit_dest` and the inverted `double_buffer_dest` all reproduce
  the legacy values. `bfp_pack_precision_mode` stays default, matching the legacy default
  `bfp8_pack_precise = false`; `packer_l1_acc` has no counterpart and is still read for
  `packer_l1_acc_en`.
- **Varargs:** two retained uses, both reported above and justified as indexed collections. No
  nameable argument was smuggled into either; every other argument on all three kernels is named.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` (base), as the audit chose. No
  `override_runtime_arguments` was added, so the framework owns the cache-hit tensor refresh.
- **Custom `compute_program_hash`:** none, and none added. The op's
  `compute_descriptor_program_hash` helper (`device/matmul_device_operation.hpp:50`) is deliberately
  not the framework hook and was not touched.
- **Pybind entry points removed:** one — the
  `nb::class_<ttnn::prim::MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory>` block that
  exposed `create_descriptor` (was `matmul_nanobind.cpp:1308-1323`). See Handoff points.
- **Open items:** no `TensorParameter` relaxation was needed or added. The two borrowed, sync-free
  buffers are `LocalTensorAccessor` candidates for the post-port style pass — see Open items.

## Handoff points

1. **Removed pybind surface — `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory.create_descriptor`.**
   *Tagged: API surface — removed entry point.* File: `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`
   (block deleted from line 1308). The function exposed the factory's legacy `ProgramDescriptor`
   builder to Python, including a `core_range_set` argument the factory body ignored. The port makes
   the C++ method vanish, so the binding had to go. **No in-tree Python caller exists** — a search
   across `tests/`, `models/` and `ttnn/` for the factory name found only the binding itself — but
   out-of-tree notebooks or internal tooling that called
   `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory.create_descriptor(...)` will now get
   an `AttributeError`. The `nb::class_<MatmulDeviceOperation>` block (`:1222-1237`) is untouched.

2. **The fork's conditional branches are uncompiled by this port.** File:
   `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp`. Three preprocessor
   branches exist in the fork that this factory never enables, so nothing in this port's build or
   test run compiles them:
   - `IN0_TRANSPOSE_TILE` → `dfb::in0_transposed` (this factory hardcodes the flag off)
   - `MM_PARTIALS_RELOAD_ALIAS` → `dfb::intermed0_reload_alias`, plus the `evil_set_read_ptr`
     cursor alignment inside `reload_from_dfb_to_dst` (only the mcast 1D/2D factories emit the
     legacy `MM_PARTIALS_RELOAD_ALIAS_CB`)
   - `BIAS_FULL_BLOCK` (only `matmul_multicore_reuse_optimized` emits it; unchanged by this port,
     as it carries no buffer index)

   The first two were converted because their legacy forms are exactly what Metal 2.0 forbids — a
   parse-time ternary over a possibly-unbound binding, and a CB index carried inside a preprocessor
   define. The conversions follow the documented patterns, but **the next factory to adopt the fork
   will be the first to compile them**, and should expect to be. `evil_set_read_ptr(other.get_read_ptr())`
   in particular replaces a raw `get_local_cb_interface(...).fifo_rd_ptr` copy; it is the CB→DFB
   whitelist §D composition, but the round-trip has not been exercised on hardware from here.

3. **Audit gap — `get_arg_addr` is not covered by the varargs scan.** The brief and the audit both
   state "RTA varargs: none — no kernel reads arguments in a loop, at a data-computed index, or
   through a running counter" (`METAL2_PORT_BRIEF.md`, `METAL2_PREPORT_AUDIT.md`). Both DM kernels
   do exactly that, through a construct the scan did not look for: a `tt_l1_ptr uint32_t*` obtained
   from `get_arg_addr(N)` and then indexed.
   - `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp:47-48` (pre-port) — two array bases,
     indexed by a runtime-computed `block_id` at `:210` and `:229`.
   - `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:31-33` (pre-port) — three array bases
     walked as interleaved triples with `index_offset += 3`, count from a runtime arg (`:216-241`).

   Not a blocker — runtime varargs are the sanctioned mechanism and the port used them — but the
   audit's recognition signals for this subject should add `get_arg_addr` (and `get_common_arg_addr`)
   alongside `get_arg_val` in a loop / `arg_index++`. An auditor grepping only for the latter two
   will keep reporting "varargs: none" for kernels that need them.

## Successes

- **[Caution: Porting a shared kernel] — the locational rung-1 check.** Re-running it on this branch
  (`find ttnn/cpp/ttnn/operations/matmul/device/kernels/ -name '*_metal2*'`) rather than a tree-wide
  filename grep mattered: a tree-wide grep returns `_metal2` hits from `experimental/quasar/**`,
  which the catalog explicitly disqualifies. The locational form gave an unambiguous rung 2.

- **[Kernel-side whitelist rule 6] fired correctly on `cb_in0_transposed`.** The brief flagged it and
  the reason held up on inspection: compute line 200 (pre-port) selected its in0 handle in a
  parse-time ternary, so both operands are name-looked-up whatever the condition, and no
  `CBDescriptor` ever allocated `c_10`. Dropping the named CTA would have compiled for this factory
  and left an unresolvable `dfb::` name for the next one. The pattern's `#ifdef`-gated-alias shape is
  what the file already does for `in1_transpose_tile` (its lines 260-264 pre-port), so the converted
  form reads like the surrounding code rather than like an imported idiom.

- **[Hardware configuration — Data movement kernels] "match on the values, not the role name."** Both
  DM kernels here take NOCs from `preferred_noc_for_dram_write` / `preferred_noc_for_dram_read`, so
  neither triple is the reader or writer default and `create_reader_datamovement_config` would have
  substituted a wrong NOC silently. The brief said so, and the section's warning about a single
  flipped NOC is precisely this factory's failure mode: `in1_noc` is load-bearing beyond perf, since
  multi-worker mode `TT_FATAL`s unless it is `NOC_0` (factory `:119-121`).

- **[Compiler options] is worth its length.** `grep -n opt_level` over the factory returns nothing,
  the field reads as absent and therefore irrelevant, and the compute kernel would have silently
  dropped from `O3` to `O2`. The section's insistence that this is an *absent line* rather than a
  wrong value — and that it must be checked mechanically per compute `KernelSpec` — is what made it
  a checklist item rather than something to notice.

- **The pad op's sharded factory answered the per-node vararg-length question.**
  `pad_rm_sharded_height_only_program_factory.cpp:469` declares one uniform `num_runtime_varargs`
  and `resize(n, 0u)`s each node's block up to it. That is a ready answer to a question the recipe
  does not address, and it avoids reaching for the `[[deprecated]]`
  `num_runtime_varargs_per_node` — which the header says is slated for removal once existing uses
  are refactored away, so adding a new one would have been the wrong direction.

## Friction

### Gaps

- **Per-node runtime-arg *presence* is not discussed anywhere in the recipe.** Metal 2.0 requires
  every name in a kernel's `runtime_arg_schema` to have a value on every node the kernel runs on
  (`program_run_args.hpp:57-60`). Legacy descriptor factories routinely emit a *short* arg list on
  cores whose kernel returns early — here, 1 arg on an idle core against 3 (in0 sender) or 11+ (in1
  writer) on an active one. There is no way to express that, so the port zero-fills the remaining
  names on those nodes. That is behaviour-preserving (the kernel returns before reading them) but it
  is a real change to the dispatched arg payload, and a porter meeting it for the first time has
  nothing to check their reasoning against. Worth a short pattern entry: *legacy short-arg-list
  cores → zero-fill the declared schema*, with the note that the alternative (a second `KernelSpec`
  over the complementary node set) buys nothing and costs an extra kernel binary.

- **The self-audit's op-directory-wide sweeps do not fit the port's own atomic unit.** The recipe
  defines the atomic unit as **one ProgramFactory**, and explicitly expects a multi-factory op to be
  ported one factory at a time — but the `cb`-name sweep is specified as
  `grep -rnE '…' <op-dir>` with "expect **zero** hits: post-port the op has no CBs". That premise
  only holds for the *last* factory of an op. Run against `ttnn/cpp/ttnn/operations/matmul` after
  this port, it returns thousands of legitimate hits from the seven factories still on the legacy
  API, so the check as written cannot pass and gives a porter no signal. This port ran every
  "expect zero" sweep against the converted file set instead, printing the denominator (5 files) as
  the section requires. The same applies to the `CircularBuffer` / `CBDescriptor` sweep under
  kernel-side whitelist rule 1 ("A grep … across the op directory at the end of the port should
  return zero hits in code"). Worth saying explicitly that on a partial port the scope is the
  converted set, not the op directory — otherwise the natural reading is either a failing check or
  a licence to convert the whole op.

- **The forced-scaffolding self-audit grep false-positives on the recipe docs.** The check is
  `git diff "$BASE" | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'`, expecting no output. But the
  prescribed workspace is branched off the doc branch and merged with `main`, so the recipe docs
  themselves are in the diff — and `metal2_port.md` contains both strings, six times, in the section
  that specifies the scaffolding. The check therefore reports 6 hits on a perfectly clean tree. The
  companion `git diff --name-only "$BASE" | grep -E '^tt_metal/'` half is unaffected and did the
  real work here. Scoping the second grep to code paths (`-- '*.cpp' '*.hpp' '*.h'`) makes it usable;
  it then returns nothing, as it should.

- **Nothing states that a `SemaphoreSpec` may be left unbound.** This factory allocates three
  semaphores and no kernel reads the third one's id, so the faithful port declares a `SemaphoreSpec`
  with no `SemaphoreBinding` anywhere. The DFB rules are explicit that a bindingless DFB is rejected
  and a dead CB must be dropped; the semaphore side says neither, and semaphores are placed directly
  via `target_nodes` rather than derived from bindings, so the analogy does not obviously carry. One
  sentence in the [`SemaphoreSpec`] construct step would settle it.

- **The `[[deprecated]]` semaphore `initial_value` has no sanctioned-use note.** Carrying a non-zero
  legacy initial value across is mandatory under the porting invariant, and the only field that can
  express it is deprecated. The deprecation is aimed at *new* uses, but the port has no way to say so
  in the code, and the self-audit has no entry for it. Worth stating that faithfully porting a
  non-zero initial value is the one legitimate use.

### Confusion

- **"Case 2" reads as "the kernel needs a raw pointer", which under-describes what to add.** The
  whitelist's rule 5 and the brief both frame Case 2 as extracting a base address through
  `TensorAccessor::get_bank_base_address()`, and the recipe says the port adds exactly two headers.
  But this kernel constructed no `TensorAccessor` at all before the port, so it also needs
  `api/tensor/noc_traits.h`. The "adds exactly two headers" claim is written for a kernel that
  already had an accessor; a Case 2 conversion in a kernel that had none needs a third. Naming that
  in rule 5 would save a build cycle.

- **`unpack_modes` — the required-entry rule is clear, but not whether *extra* entries are wanted.**
  The rule is that an explicit entry is *required* when a compute kernel consumes a Float32 buffer
  with `enable_32_bit_dest`; omitting an entry means `UnpackToSrc`. So for a factory whose
  intermediate format is `fp32_dest_acc_en ? Float32 : …`, whether an entry is needed depends on a
  runtime-resolved config, and the natural safe move is to state `UnpackToSrc` explicitly for every
  consumed buffer — which is what `matmul_multicore_program_factory.cpp` does and what this port
  does. That is a sound reading but the section does not say it, and it sits in tension with
  "`Default` → `UnpackToSrc`, which you normally express by *omitting* the entry."

## Open items for downstream

### Shared kernel touches

Coordination signal for the next matmul porter and the eventual sunset checklist.

| kernel | rung taken | remaining unmigrated consumers |
|---|---|---|
| `device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **2 — created the fork**: `bmm_large_block_zm_fused_bias_activation_metal2.cpp` beside the original; pointer comment landed in the original (its lines 5-9) | `device/factory/matmul_multicore_reuse_optimized_program_factory.cpp`, `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (hosts **two** factories), `device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`, `device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`, `device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp` |
| `device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | **converted in place** — sole binder is this factory | none |
| `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | **converted in place** — sole binder is this factory | none |

**Fork binding vocabulary**, which the five remaining consumers inherit and cannot rename. Ratified
with the invoker before naming, taken from the kernel's own named-CTA keys with `cb_` dropped, which
is also the vocabulary `bmm_metal2.cpp` established for `MatmulMultiCoreProgramFactory`:

- **DFB accessor names:** `in0`, `in1`, `bias` (gated on `FUSE_BIAS`), `out`, `intermed0`,
  `in0_transposed` (gated on the new `IN0_TRANSPOSE_TILE`), `intermed0_reload_alias` (gated on the
  new `MM_PARTIALS_RELOAD_ALIAS`).
- **Named CTAs:** `in0_block_w`, `in0_num_subblocks`, `in0_block_num_tiles`,
  `in0_subblock_num_tiles`, `in1_num_subblocks`, `in1_block_num_tiles`, `in1_block_w`,
  `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim`, `out_subblock_h`,
  `out_subblock_w`, `out_subblock_num_tiles`, `batch`, `out_block_num_tiles`, `untilize_out`,
  `get_batch_from_reader`, `bias_ntiles`, `last_subblock_w_valid`; `row_broadcast_bias` under
  `FUSE_BIAS`; `activation_type`, `activation_param0..2` under `SFPU_ACTIVATION`.
- **Named RTA:** `is_worker_core`, read only under `MATMUL_DRAM_SHARDED`.
- **No tensor bindings**, and no varargs.
- **Two defines replace legacy compile-time arguments**, and adopting factories must emit them
  instead: `IN0_TRANSPOSE_TILE` (was positional CTA 17) and `MM_PARTIALS_RELOAD_ALIAS` (was the
  index-carrying define `MM_PARTIALS_RELOAD_ALIAS_CB`). `BIAS_FULL_BLOCK`, `FUSE_BIAS`, `PACK_RELU`,
  `SFPU_ACTIVATION`, `PACKER_L1_ACC`, `FP32_DEST_ACC_EN`, `IN1_TRANSPOSE_TILE`, `SKIP_COMPUTE`,
  `MATMUL_DRAM_SHARDED` and the stagger/throttle defines are unchanged.

**Sunset:** the legacy copy can be deleted, and the fork renamed onto its name, once all five
factories above have ported. Nobody is tracking that today. The private copies under
`tests/tt_metal/tt_metal/perf_microbenchmark/{1_compute_mm,old/matmul}/kernels/` are *not*
consumers — they are separate files — but `1_compute_mm/kernels/…_copy.cpp` is kept in sync with the
legacy copy by hand, per the note at the top of both files, and that coupling outlives the sunset.

### Findings for the op owner (preserved, not fixed)

1. **Semaphore id 2 is dead.** The factory allocated `in0_mcast_sender_valid` with initial value
   `VALID` and passed its id as positional CTA 13 to the in0 sender, which never reads that slot; no
   other kernel uses a semaphore at all. The "sender valid" handshake is in fact implemented on
   semaphore 1 (`receiver_sem.set(VALID)`, in0 kernel `:67`). The port keeps all three specs so the
   program's semaphore allocation is unchanged, and the third is declared with no binding. If the
   owner confirms it is vestigial, dropping it removes a `SemaphoreSpec`, a `[[deprecated]]`
   `initial_value` use, and one L1 cell per core.

2. **Two more dead compile-time argument slots.** in0 sender slot 13 (above) and in1 writer slot 14
   (the literal `1` pushed only when bias is present). Neither is read by any kernel; both are
   dropped by the port. These are in addition to the dead named CTAs `cb_in0_intermediate` (`c_8`)
   and `cb_in1_intermediate` (`c_9`) that the audit already flagged.

3. **The in0 sender's noc-y array base is only correct when `num_blocks % S == 0`.** The kernel
   computes the y-array base as an offset of `num_storage_cores = num_blocks / num_blocks_per_shard`
   from the x-array base, while the host emits exactly `S` x-values followed by `S` y-values, where
   `S = input_all_storage_cores_vec.size()` and `num_blocks_per_shard = num_blocks / S`. The two
   agree only when the division is exact. Preserved verbatim in the port (the vararg indices use the
   same `num_storage_cores` expression), so the fragility is unchanged, not introduced. Whether an
   inexact case is reachable depends on validation upstream of this factory, which the port did not
   audit.

4. **`in0_mcast_sender_noc_x` / `_y` in the in1 writer name the wrong thing.** In that kernel they
   hold the *output storage* core coordinates for the write-back, not in0 mcast senders — a
   copy-paste inheritance from the in0 kernel. The port keeps the names (renaming is outside a
   syntax swap) and adds a comment saying what the block holds.

5. **Three `skip_*` flags remain hardcoded off** (`skip_compute`, `skip_in0_mcast`,
   `skip_write_back`), so `SKIP_COMPUTE`, the in0-side `SKIP_MCAST` and `SKIP_WRITE_BACK` are never
   emitted and their kernel branches are unreachable from production. Carried across unchanged; the
   audit already flagged them for an owner decision.

### Test coverage notes

- **The confirmed baseline set covers bias and multi-worker, and this box exercises both.** The
  dedicated `test_matmul_dram_sharded.py` parametrizes `has_bias` and
  `num_workers_per_dram_bank ∈ {1, 2, 3}`; values above 1 are Blackhole-only and this is a Blackhole
  box, so the `SPLIT_DRAM_BANK` path and the in1 writer's vararg walk are both live.
- **`test_linear.py::test_linear_fused_non_broadcast_bias_dram_sharded_in1` is fully skipped here**
  (4/4 cases, "TinyTile Matmul needs to be fixed on BH. Issue #31385"), pre-existing and unrelated
  to this port. It was the only non-broadcast-bias case in the confirmed set; row-broadcast bias is
  covered by `test_matmul_dram_sharded.py`, so `row_broadcast_bias == false` has no live coverage on
  this box.
- **`untilize_out == true` has no coverage in the confirmed set** that this port could identify, so
  the `untilize_out` branch — including the config in which `out` and `intermed0` are *not* aliased —
  rests on the `interm0_data_format != output_data_format` half of that condition being exercised
  instead. Worth a targeted case for whoever ports the sibling DRAM-sharded factory.

### Post-port passes

- **`sync_free_dfbs` (style).** `in0_sharded` and `out_reshard` are both sync-free *and* borrowed —
  exactly that pass's `LocalTensorAccessor` target. Each is bound as a self-loop by its single
  touching kernel and reached only through `get_read_ptr()` / `get_write_ptr()`, with no FIFO
  traffic. Not done here.
- **`gen2_hardware_configs` (semantic).** Both DM kernels carry custom Gen1 configs replicated
  verbatim from the legacy triples, with no Gen2 branch, per the recipe's Gen2-is-out-of-scope rule.
- **RTA → CRTA.** The in0 sender's vararg block (the mcast senders' noc-x then noc-y values) is
  identical on every node, so it is really a common runtime vararg. Left as a per-node runtime
  vararg here because RTA→CRTA changes dispatch semantics; noted for that later pass.
