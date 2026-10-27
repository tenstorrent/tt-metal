# Port Report — eltwise/ternary

Line numbers below are the ported tree's unless they say *legacy* (`21d0f3c7ba9`).

## Outcome

**PORTED.** `TernaryProgramFactory`, the op's only factory, now satisfies `CustomProgramSpecFactoryConcept`. All 22
kernel sources it can select, plus the op-local header `compute/ternary_sfpu_copy_init.hpp`, are converted. The
confirmed test set passes identically before and after the port.

## Verification

- **Build:** `./build_metal.sh --build-tests`, incremental, clean.
- **Tests** — the targeted subset the invoker confirmed on 2026-10-27, run before and after with Watcher on
  (`TT_METAL_WATCHER=10`), a fresh kernel cache per run, and `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`:
  - gtests (`unit_tests_ttnn`): `EltwiseSmoke.Where*`, `TestScopedGraphCapture.TernaryOp*`,
    `TestLevelizedGraphCapture.TernaryOp*`.
  - pytest, 503 cases:
    - in full: `test_ternary.py`, `test_ternary_composite.py`, `test_snake_beta.py`,
      `test_ternary_category1_bfloat16.py`, `test_where_unsupported_bcast.py`, `docs_examples/test_pointwise_ternary_examples.py`;
    - sliced to TTT and to one case per regime: `test_where.py`, `test_ternary_sharding.py`;
    - the ternary cases of `test_eltwise_mixed_rank.py` and `test_eltwise_sharded_output_empty_shards.py`.

  | | gtests | pytest |
  |---|---|---|
  | baseline (legacy build of `21d0f3c7ba9`) | 5 OK, 2 skipped | 503 passed |
  | port | 5 OK, 2 skipped | 503 passed |

  The junit results have the same 503 test ids with the same status, and neither run logged a Watcher error. The two
  skips are `TestScopedGraphCapture`'s own `GTEST_SKIP` on this machine, in both runs. The port's kernel builds are
  free of warnings. The baseline's logged 20 `-Wunused-but-set-variable` warnings for `dst_args` and `src2_args`,
  accessor args that the natively sharded variants never use. After the last edit (comments only), the rebuilt tree
  passed a smoke run: the probe's control case (below) and the gtests again. Left to CI: the rest of `test_where.py`
  and `test_ternary_sharding.py`, the nightly ternary / where / lerp / backward suites, and the sweeps.
- **Legality checks are live, proven on silicon.** The recipe's forcing edit in `tt_metal/impl/` was not applied (see
  *Friction*). The miss path validates by default: the adapter calls `MakeMeshWorkloadFromSpecs` and
  `SetProgramRunArgs` without `skip_validation`. The hit path validates when `validate_program_args` is on. A temporary
  fault injection in this factory showed both. It was never committed; the tested source was restored byte-for-byte
  and rebuilt. Two `ttnn.where` calls per run (miss, then hit), bf16 predicate, fp32 values:

  | injected fault | `validate_program_args` | result |
  |---|---|---|
  | miss path declares an unbound tensor parameter | true | call 1 rejected by spec validation (`program_spec/validation/resource/tensor.cpp:108`) |
  | override binds the output to `predicate` | true | call 2 (cache hit) rejected: `program_run_args.cpp:68`, `tensor_layout` mismatch |
  | same | false | call 2 accepted, output wrong: the hit applies the override's arguments unchecked |
  | none | true | both calls exact, one cache entry |

- **Anti-pattern self-audit** (hits / files scanned):
  - Buffer addresses in run-args (`buffer()->address()`, `emplace_runtime_args`, `Buffer*`): 0 / 1 (factory).
  - `TensorAccessorArgs`, `.get_id()`, positional `get_arg_val` / `get_compile_time_arg_val` / `get_common_arg_val`:
    0 / 23 ported kernel files.
  - `CircularBuffer` / `CBDescriptor` / `CBIndex` in op code: none outside the three untouched files (below).
  - Multi-binding flag 0; varargs 0; every CTA named.
  - Forced scaffolding in the diff: no `tt_metal/` file and no `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` line.
  - `.md` cited from code: 0 / 28 code files changed since `merge-base origin/main` (the pre-port commit's included).
  - `cb`-name regex over the op directory: 38 code files, 34 hits (positive control: 86 in the legacy factory). All 34
    are outside the port's reach:
    - 31 are in the three files the port does not touch: `compute/ternary_addc_ops_fpu.cpp` and
      `compute/ternary_addc_ops_fpu_bcast.cpp` (bound by nothing) and `dataflow/ternary_reader_nobcast_ttt.cpp`
      (lent to binary_backward).
    - 1 is `ternary_device_operation.cpp:8`, the `cb_utils.hpp` include (device-op class, off-limits).
    - 2 are the legacy local `srcB_num_tiles` ("src" + operand B, not a CB) in the two TTS/TST broadcast readers.
- **Guard census** (`TT_FATAL|TT_ASSERT|TT_THROW` per file; 5 files carry guards) against `HEAD`: unchanged
  everywhere except `ternary_program_factory.cpp`, 8 → 5. The `merge-base origin/main` comparison the recipe
  prescribes also shows `ternary_device_operation.cpp` 36 → 35. That drop is the pre-port fix commit, not the port.
  The three factory guards that left are recorded as subject-deleted:
  - legacy `:1497` `bs.slot < n`: the `Buffer*` slot re-wrap of the descriptor path, replaced by tensor bindings;
  - legacy `:1587` (override) cached-program runtime-arg width check: the Metal 2.0 override never sees the `Program`;
    see *Friction* for the recipe conflict;
  - legacy `:1633` (override) "one tensor CB per index": the CB-index re-point chain, replaced by `tensor_args`.
- **`hw_config` / `opt_level`:**
  - reader and writer resolve to the legacy triples: RISCV_1 / NOC_0 and RISCV_0 / NOC_1, both dedicated;
  - compute: HiFi4, precise SFPU, Dest double-buffered, BFP pack approximate (the lowering's default without
    `config_1xx`), and `enable_32_bit_dest = fp32_dest_acc_en`;
  - compute `unpack_modes`: an `UnpackToDest` entry exactly where the legacy vector had `UnpackToDestFp32`, which is
    every FLOAT32 slot, output included. The output's entry is inert while compute only produces `out`. When the
    output is natively sharded, `out` is a compute self-loop. The entry is then legal because a FLOAT32 output turns
    on `enable_32_bit_dest`;
  - `opt_level`: compute `O3` set explicitly (`ternary_program_factory.cpp:1626`); the two DM kernels keep the `O2`
    default, as legacy did.
- **Install coverage:** 22 / 22 bound sources (20 from `get_kernel_file_path`, 2 from
  `override_addcmul_compute_kernel`) are in the regenerated `build_Release/.../eltwise/ternary/cmake_install.cmake`.
  The op's `CMakeLists.txt` globs `device/kernels/*.cpp` and `*.hpp` into its installed `kernels` file set. The port
  adds no kernel file.
- **Performance:** not measured. The kernels change only how they obtain arguments, buffer ids and accessors.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe0 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (`git log -1 --format='%h %cs %s' -- docs/.../metal_2.0/` in the recipe worktree `/home/ubuntu/metal2-port-recipe`,
  because this branch carries no `metal_2.0/` docs. Its `metal_2.0` tree is `c3e701f18bd5`, identical to that of
  `917314f85d3`.)
- **Audit docs (inherited):** `917314f85d3 2026-10-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Relaxation analysis:** `analyses/relaxations/eltwise_ternary.md`, untracked, blob `c89ba329e906`. The audit ran
  its five validity checks on this tree, and all pass (audit *B*). The port leaves what they inspect unchanged:
  `ternary_device_operation.cpp` is byte-identical, `has_sharding` is still `shard_volumes.has_value()`
  (`ternary_program_factory.cpp:1128`), and the variant still holds one factory (`ternary_device_operation.hpp:65`).

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept`, as the audit chose. `override_runtime_arguments`
(`ternary_program_factory.cpp:1682-1718`) returns `ProgramRunArgs`, with two parts:
- the runtime args of reader, writer and compute for every core, from the same builder the miss path uses;
- a `TensorArgument` for every io `TensorParameter`: `predicate`, `value_true` (TTT, TTS), `value_false` (TTT, TST) and
  `output`.

The legacy override refreshed every tensor address: reader slots 0-2, writer slot 0, and every borrowed CB. So none is
skipped. A tensor binding's runtime-field words (the sharded shape in pages) are re-emitted together with its address.
The legacy override left those words alone. The hash pins them (`ternary_device_operation.cpp:640-643`). The
re-emitted values therefore equal the cached ones.

### Device-op-class edits
- `ternary_device_operation.hpp`: only the `TernaryProgramFactory` declaration changes (`create_descriptor` →
  `create_program_artifacts`; the override now returns `ProgramRunArgs` and takes no `Program&`). The two descriptor
  includes become `ttnn/metal_v2_artifacts.hpp`. None of the three exceptions is needed: the factory struct already
  existed.
- Pybind entry points removed: none.
- Custom `compute_program_hash`: left intact at `ternary_device_operation.cpp:592-653`, over `to_hash()` at `:264-274`.
  `ternary_device_operation.cpp` is byte-identical to `HEAD`.

### Open items
- The relaxation is the analysis doc's declaration, `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on
  all four parameters (`ternary_program_factory.cpp:1155`). No further candidate.
- Named runtime args raise the cache-hit host cost (see *Friction*).

## Handoff points

none. Nothing capitulated. No out-of-op call site needs a `sem::` or `tensor::` handle. `kernel_lib`'s
`ckl::input(DFBBindingToken, …)` overload took the tokens as they are. No framework gap was hit.

## Successes

- **The brief's watch-for list was exact and line-current.** Each flagged construct turned up as described, and each
  had a recipe answer: the hardcoded `c_0` in `ternary_sfpu_copy_init.hpp`, `.get_id()` at LLK sites, the
  `process_tile` id helpers, `kernel_lib` ids, the reader that ignores its CB-id CTAs (`tts_tst_reader_row_bcast.cpp`),
  and the unconditional `DataflowBuffer` constructions. That made the 22-source atomic conversion a single pass that
  built first time.
- **Pattern: Conditional / optional resource bindings**, in particular "the define and the binding share one
  condition". The reader's tensor bindings and the writer's `out` / output bindings are decided by reading back the
  `SRC_SHARDED_*` / `DST_SHARDED` value the kernel is built with (`ternary_program_factory.cpp:1357-1383`, `:1406`).
  The LLK staging DFBs follow `BCAST_*` (`:1580-1617`). The existing `#if` gates in the kernels did the rest; only
  `ternary_writer_nobcast.cpp` and `ternary_addc_ops_fpu_rowbcast.cpp` needed constructions moved under them.
- **The endpoint census** (*Two-toucher DFB → assign 1P+1C*, *Sync-free and single-ended → self-loop*) agreed with the
  brief on every DFB:
  - `in_a` / `in_b` / `in_c`: 1P+1C;
  - `out`: 1P+1C, or a compute self-loop when natively sharded;
  - `bcast_a` / `bcast_b` / `bcast_c`: compute self-loops.

  No multi-binding.
- **The relaxation analysis doc** could be transcribed verbatim. Its validity checks are each a short grep, which
  made re-checking them after the port quick.
- **CB→DFB whitelist §A, the `constexpr` exception:** `init_once` and `copy_init_shared_with_c0` are constant
  expressions. They keep the `unpack_*[...]` array form with the token rather than a member getter. The whitelist's
  wording ("the legacy declaration is the entire test") settled it without a build experiment.
- **`unpack_modes` validator documentation** (`hardware_config.cpp`, "producer-only binding → inert: tolerated"). It
  allowed mirroring the legacy vector exactly, the output's `UnpackToDestFp32` entry included.

## Friction

### Gaps
- **Validation forcing.** The recipe's proof edits every `tt_metal/impl/metal2_host_api/*.cpp`. This environment's
  auto-mode classifier has blocked that edit before. The port therefore used `validate_program_args` plus a temporary
  fault-injection probe inside the op's own factory (*Verification*). That proof also shows the knob is what gates the
  hit-path check. The recipe could offer it as the alternative when `tt_metal/` edits are unavailable.
- **The override-guard rule.** "An `override_runtime_arguments` is never a subject-deleted case" collides with guards
  whose subject is a legacy-API object the Metal 2.0 override cannot reach. Here those are the cached `Program`'s
  positional runtime-arg vector and the CB-index re-point chain. The port records both as subject-deleted. Part of
  their protection survives in the framework: `UpdateProgramRunArgs` fails on an unknown arg name or a node without a
  runtime-arg buffer even with validation skipped. The nearest carry for the width check would be a `TT_FATAL` in
  `add_runtime_args` that a kernel's rows hold exactly one value per arg name for every core. The rule needs a
  carve-out for this case, or guidance on what to carry.
- **A binding refresh is wider than an address refresh.** `ttnn_factory.md` says the override's address statements
  "become `tensor_args` entries". It does not mention that a `TensorArgument` also re-emits the accessor's
  runtime-field words, which a legacy override may have deliberately left alone, as this one did. Harmless here
  because the hash pins them; worth a sentence in *The cache-hit contract*.
- **The DM helper sample.** The recipe shows `create_reader_datamovement_config(device->arch())`, but the helper's
  signature is `(bool disable_dfb_implicit_sync_for_all = false)`. `tt::ARCH` is an enum class. The sample therefore
  fails to compile rather than silently setting the Gen2 opt-out. Ported ops call it with no argument.
- **Helpers that take CB ids as parameters** (`process_tile(uint32_t …_cb_id, …)` in four compute kernels). Neither the
  whitelist nor the `.get_id()` anti-pattern covers them. `dataflow_buffer.h` says a function parameter "must spell
  `DFBBindingToken<P, C>` (or take the id)". The port drops the id parameters and builds the `DataflowBuffer`s from the
  tokens inside the helper, keeping the per-call construction legacy had.
- **The `cb`-name sweep** expects zero hits over the op directory. An op directory can hold unbound or lent kernels the
  port must not touch, and an off-limits device-op include. The regex also flags `srcB_` ("src" + operand letter).
  Scoping the sweep to the files the factory binds would make "zero" meaningful.
- **The guard census `BASE`** is `merge-base origin/main`, which counts pre-port fix commits on the branch as port
  deltas. Comparing against the pre-port commit isolates the port.
- **Hit-path host cost.** `Table` is a linear-scan small vector. Named per-node runtime args therefore cost
  O(names × nodes²) inserts per cache hit: 37 names × up to 64 nodes here. The legacy hit path was a fixed-width
  copy per core. Emission is name-first, which looks each name up once. This is an API-level cost, not a port choice.

### Confusion
- **Effective operands that switch DFB by path.** In `ternary_sfpu_row_bcast_ttt.cpp` and
  `ternary_addc_ops_fpu_rowbcast.cpp`, `dfb_eff_*` is the staging DFB or the input DFB. Replacing `dfb_eff_a.get_id()`
  needed a path-dependent `constexpr` token alias (`eff_a`). *Same-FIFO aliasing*'s path-dependent variant covers it,
  but the `.get_id()` anti-pattern entry does not point there.
- **One schema, eleven readers.** "Pick names that match the variables they were going to be assigned to" meets 11
  readers that assign the same runtime-arg slot to differently named locals (`true_nD_stride`, `nD_stride_b`,
  `src_b_nD_stride`, `tensor_nD_stride`, …). The host emits one schema. Each slot therefore has one canonical name
  (`ternary_program_factory.cpp:682-724`), and the kernel locals keep their legacy names.
- **"Extract the `MeshTensor` at the top and work with it throughout."** This factory's helpers are written against
  `ttnn::Tensor` (dtype, shard specs, shapes). The port takes `.mesh_tensor()` only where Metal 2.0 needs it: the
  `TensorParameter` specs and the `TensorArgument`s. That keeps the diff to port work.

## Open items for downstream

- **Shared kernel touches:** none. No `_metal2` fork was created or reused.
  - `dataflow/ternary_reader_nobcast_ttt.cpp` lives in this directory but is bound only by
    `eltwise/binary_backward/device/binary_backward_program_factory.cpp:122` (lent). It is untouched and still legacy;
    binary_backward's port will need its fork.
  - `compute/ternary_addc_ops_fpu.cpp` and `compute/ternary_addc_ops_fpu_bcast.cpp` are bound by nothing and were left
    as they are.
- **Dropped or narrowed resources** (zero functional change):
  - c_4 / c_5 / c_6 (legacy `ternary_program_factory.cpp:1141-1169`) now exist only where a kernel uses them: TTT ∧
    ROW_BCAST ∧ an LLK row-broadcast compute kernel ∧ that operand's `BCAST_*` = 1. Before, they were allocated for
    every TTT ∧ ROW_BCAST call.
  - The writer's runtime-arg slot 10 (padding, never read; legacy `:914`) is gone.
  - The TTS/TST readers' slot 2 (a constant 0 for the absent operand, read into an unused local; legacy `:502`) is gone.
- **Quasar-uplift debt** recorded for the uplift pass:
  - token-form `constexpr` metadata reads, which whitelist §A marks Gen1-only:
    - `ternary_sfpu_copy_init.hpp:15-21` (`unpack_*[dfb_id]`, `unpack_*[dfb::in_a]`);
    - `ternary_sfpu_col_scalar_bcast_ttt.cpp:19-20`;
    - `ternary_addc_ops_sfpu_bcast.cpp:18-19`, through the helper;
  - `ternary_sfpu_row_bcast_ttt.cpp:40` (and `_b` / `_c`) copy-constructs a second `DataflowBuffer` onto one FIFO;
  - the compute helpers construct their `DataflowBuffer`s on every `process_tile` call, as legacy did. Gen2's
    destructor drains, and this needs a look there.
- **Findings, not fixed** (scope discipline):
  - The out-of-scope op issues the invoker listed. A mismatched output dtype is accepted and returns wrong results
    (`determine_output_dtype` adopts the output's dtype). An ND-sharded `input_a` with no preallocated output
    dereferences an empty 2D shard spec.
  - Stale comments:
    - `ternary_device_operation.cpp:589-590` ("re-applied via get_dynamic_runtime_args");
    - the builder's "coarsens each input tensor to its padded VOLUME" (`ternary_program_factory.cpp:743`); the key
      pins logical H / W;
    - the override's "ND sharding cannot reach this path" (`:1705-1706`). It is false since #59788, and the conclusion
      still holds because the key hashes the very shape-in-pages words;
    - "Output DFB depends on variant: c_2 … c_3 …" (`:1469`). The output is `out` / legacy c_3 in every variant;
    - the readers' "moved to end" on `src_num_tiles` (positional history).
  - Unused code:
    - `ternary_device_operation.cpp:8` includes `ttnn/operations/cb_utils.hpp`;
    - `scalar_is_not_1` in `ternary_addc_ops_fpu_rowbcast.cpp:22`;
    - binary_ng's `eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp`, included and unused by five compute kernels.
      They are kept: the whitelist adds two headers and changes nothing else.
  - When `is_llk_bcast` holds and the int addcmul kernel is chosen, the reader still gets `BCAST_LLK=1`. Per the audit
    the two conditions cannot meet on a public path.
- **Per-op carry-over:** binary_ng shares the donor headers and the reader structure. Two choices here would carry
  over: the slot-letter binding names (`in_a` / `in_b` / `out`, matching the `SRC_*_A/B` defines), and the
  path-dependent token alias for row-broadcast staging.
- **Test coverage:** `TestScopedGraphCapture.TernaryOp*` skips itself on this machine before and after. The scoped
  graph-capture path was therefore not exercised locally. Device-kernel timing (ttperf, 32×32 / 1024² / 4096²) can be
  run before vs after on request.
