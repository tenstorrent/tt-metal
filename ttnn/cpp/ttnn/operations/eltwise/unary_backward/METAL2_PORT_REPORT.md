# Port Report — `eltwise/unary_backward`

## Outcome

**PORTED.** `UnaryBackwardProgramFactory`, the op's only factory, now satisfies `ProgramSpecFactoryConcept`. All
nine kernel entry points it can select converted together: the two reused `_metal2` forks, the two row-major
kernels and the five compute kernels. The port covers the four op types that exist on this base (`SIGMOID_BW`,
`TANH_BW`, `GELU_BW`, `GELU_TANH_BW`). The same fixed test set gives the same result before and after the port
(*Verification* below).

## Provenance

- **Recipe docs (this port):** `f126189dbb6 2026-11-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `f126189dbb6 2026-11-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

The date is the rebase's committer date. The branch's `metal_2.0/` tree (`c3e701f18bd`) equals upstream
`akertesz/op-porting-recipe` at `9425f50ae93` (2026-09-03), which is two commits behind upstream `c41f7f4ef0a`.
Port recipe blob `33e573fa1b6`. As instructed, I followed the in-place files.

**Base.** `anasuya/metal2_port_eltwise_unary_bw` @ `f126189dbb6`; its code equals `main` `a1ce48e3fe8`. The base
contains `be782c33d2e` (#63334), so both conditions of the previous audit hold.

**Invoker decisions for this session (2026-11-03).**
- Port the four existing op types and their five compute kernels. Do not wait for the ~43 open PRs that add op
  types, and do not anticipate them (audit question 2). The enum still had exactly four values at build time and
  before the commit.
- `.relaxations = {}` on all three `TensorParameter`s, in every mode.
- The relaxation doc ships unchanged. Its suggested edits are recorded under *Open items*.
- Test set: the full set below (confirmed by the invoker).
- Sheet cells (audit question 1): leave `Op Classification` and `Pointer patching perf issue?` as supplied. The
  port makes their premise true (the `Buffer*` runtime args and their `BufferBinding` patching are gone).
- `Diego validation` (audit question 3): the reading "the sheet owner has reviewed this row" is confirmed. `no`
  stands until the owner reviews the merged row.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. `create_descriptor` became `create_program_artifacts` inside the
existing `UnaryBackwardProgramFactory`
(`device/unary_backward_program_factory.hpp:18-21`, `device/unary_backward_program_factory.cpp:28`). The ported-from
factory had no `override_runtime_arguments`, so on a cache hit the framework refreshes the three tensor bindings and
nothing else. That matches the legacy behaviour: the descriptor fast path re-patched only the `Buffer*` runtime args
and the aliased CB addresses. The borrowed DFBs now take their address from the same refreshed tensor arguments.

### Device-op-class edits
- Pybind entry points removed: none (no pybound `create_descriptor`).
- Custom `compute_program_hash`: left intact at `device/unary_backward_device_operation.cpp:238-260`. The file is
  byte-identical to the base.
- `program_factory_t` already existed, so exception 3 did not apply.

### Open items
- Relaxation candidates: none beyond the analysis doc, whose verdict is `none`.
- Concept fit: no friction. The strict match held on every cache hit in the test set with hit-path validation on.

## Handoff points

- **Merge-order hazard for the ~43 open PRs that add op types (owner: nkapreTT and reviewers of those PRs).** Each
  open PR adds a compute kernel written against the legacy contract: `tt::CBIndex::c_0` / `c_1` / `c_2` and
  `get_arg_val<uint32_t>(0)`. Once this port lands, the factory binds every compute kernel through Metal 2.0
  bindings. A kernel still written the legacy way then fails when it is JIT-compiled, so the failure only shows on
  the first call that selects that op type. To fit the new contract, a kernel uses `dfb::grad_out`, `dfb::input`
  and `dfb::grad_in`, reads `get_arg(args::per_core_tile_cnt)` and includes `experimental/kernel_args.h`. The five
  converted kernels show the form, for example `device/kernels/compute/eltwise_bw_tanh.cpp`. The contract comment in
  `device/unary_backward_op_utils.hpp:19-21` states it. Per the invoker, none of these PRs was absorbed.
- No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap and no removed pybind surface.

## Successes

- **Compiler options (`metal2_port.md`, "Compiler options").** The legacy factory never mentions `opt_level`.
  Reading the field as "no setting" would have left the compute kernel at Metal 2.0's O2 instead of the legacy O3.
  The recipe's rule (an unset compute `opt_level` means O3) and its mandatory grep caught it. The factory sets O3
  explicitly (`device/unary_backward_program_factory.cpp:591`).
- **Reading the validator instead of the recipe's summary.** The recipe says `UnpackToDest` on a ≤16-bit DFB is
  rejected on Gen1. That holds only with `enable_32_bit_dest = false`
  (`tt_metal/impl/metal2_host_api/program_spec/validation/hardware_config.cpp:129-131`). `SIGMOID_BW` forces fp32
  DEST with bf16 operands, so a one-for-one translation depends on the 32-bit case being accepted. Checking the
  validator before writing the spec confirmed the brief. The translation
  (`device/unary_backward_program_factory.cpp:555-560`) validates in every mode.
- **Rung 1 of "Caution: Porting a shared kernel".** Both borrowed kernels already had forks whose interfaces fit
  exactly: same named args, same defines, and `#ifdef` gates that match the aliased mode. This op is the first
  binder of `reader_binary_interleaved_start_id_metal2.cpp`. Neither fork nor original was edited.
- **"Conditional / optional resource bindings".** The forks gate `tensor::src0` / `src1` / `dst` on
  `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED`. The host omits exactly those bindings under the same condition
  (`device/unary_backward_program_factory.cpp:454-470`, `:501-512`). Legacy had always appended the writer's
  accessor. Binding it unconditionally would have compiled, but the pattern entry is why the host condition and the
  defines now share `can_alias_shards`.
- **"Sync-free and single-ended CBs → self-loop DFB".** The two row-major scratch buffers are each touched by one
  kernel, through `get_write_ptr()` only. I re-derived the census (`aligned_scratch`,
  `device/kernels/dataflow/row_major_pages.hpp:142-143`) and it agrees with the brief: a DM self-loop each, legal
  on Gen1.
- **"Locate and confirm the op's tests".** A filename search finds only part of the coverage. `bias_gelu_bw`
  reaches the device op through `binary_backward.cpp:667` and `:683`. Its nightly tests, its docs example and two
  comparison-mode cases therefore belong to the set. Three more docs examples and the ULP gtests in
  `unit_tests_ttnn` call the ops directly. The two comparison-mode cases skip under the default fast runtime mode,
  so I ran them separately with it off, before and after.

## Friction

### Gaps
- **The recipe's hardware-config API names do not exist in the headers.** `metal2_port.md` ("Hardware
  configuration", "Compute kernels") uses `ComputeGen1Config`, `DataMovementGen1Config`,
  `std::get<ComputeGen1Config>(compute_hw)` and `create_reader_datamovement_config(device->arch())`. The headers
  have `ComputeHardwareConfig` with generation-agnostic fields plus an optional `config_1xx`
  (`compute_hardware_config.hpp`), `DataMovementHardwareConfig{.config_1xx}`, and TTNN DM helpers that take an
  optional `disable_dfb_implicit_sync_for_all` flag instead of an arch
  (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`). The values carry over;
  only the spelling is stale. Following the headers settled it. The recipe should name the current types.
- **Legacy defaults for "Style B" are easy to confirm, but the recipe gives no path to the lowering.** The claim
  that unset `ComputeHardwareConfig` fields reproduce the legacy defaults (`double_buffer_dest = true`,
  `sfpu_precision_mode = Precise`, unset `config_1xx` → `bfp8_pack_precise = false`) holds. I confirmed it in
  `tt_metal/impl/metal2_host_api/program_spec/construction/kernel_lowering.cpp:160-176`. A pointer to that function
  would make the "diff before against after" step quick.
- **Validation forcing.** I did not apply the recipe's `skip_validation` scaffolding in `tt_metal/impl/`, because
  this workstation's auto-mode classifier blocks that edit. Instead:
  - The miss path validates unconditionally (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:993-996`).
  - The hit path was forced with `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` (`:830`).
    `ttnn.CONFIG.validate_program_args` printed `True` at the start of each run.

  There are no `METAL2_CHECKS_FORCED` markers to grep for. If you want the recipe's marker proof, the scaffolding
  must be applied by hand.
- **Recipe silent on newer header features.** The headers now have array arguments (`compile_time_array_args`,
  `runtime_array_args`). TTNN also has a `MeshWorkloadSpecFactoryConcept` (`operation_concepts.hpp`), while
  `ttnn_factory.md` says none existed "as of 2026-08-31". This port needed neither.

### Confusion
- **The `cb`-name sweep reaches files the port may not edit.** The self-audit grep runs over the whole op
  directory and expects zero hits. One hit stays: the comment "destination CB" at
  `device/unary_backward_device_operation.cpp:252`, in the device-operation class, which the port must leave
  byte-identical. The recipe does not say what to do when the sweep and the scope rule collide. I left the comment
  and recorded it here.
- **The `cb_*` → `dfb_*` rename has no mechanical target for a bare `cb`.** The row-major writer's local `cb` (the
  front entry's address) became `dfb_addr`. `dfb` alone would have hidden nothing, since a qualified `dfb::` lookup
  ignores variables, but it reads like the namespace. In comments, "CB page" became "DFB entry" (DFB vocabulary
  has entries, not pages).
- **"Extract `MeshTensor` and work with it throughout" versus "don't refactor adjacent code".** The legacy body
  reads geometry from `ttnn::Tensor` (`padded_shape()`, `memory_config()`, `buffer()->page_size()`,
  `buffer()->alignment()`). Moving all of it to `MeshTensor` is a refactor of code the port does not otherwise
  change. I extracted the three `MeshTensor`s at the top for the `TensorParameter` specs and `TensorArgument`s
  only (`device/unary_backward_program_factory.cpp:35-37`).

## Verification

Same base, same machine (one Wormhole n150). For both runs: one fixed list of 336 pytest node ids, then 46 gtests,
`TT_METAL_WATCHER=10`, a fresh `TT_METAL_CACHE`, and `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`.

| Run | Legacy (build 13:02, before any kernel edit) | Port (build 17:03) |
|---|---|---|
| 336 pytest node ids | 306 passed, 26 skipped, 4 xfailed | 306 passed, 26 skipped, 4 xfailed; same PASSED set (`diff` of the sorted node ids is empty) |
| 46 ULP gtests (`unit_tests_ttnn`) | 46 OK | 46 OK |
| 2 comparison-mode `bias_gelu_bw` cases, fast runtime mode off | 2 passed | 2 passed |
| Kernel sources JIT-compiled (of 9 selectable) | 9 | 9, with the same variant counts per source (the reader moved to its fork, 12 → 12) |

Test files: `unit_tests/operations/eltwise/test_unary_backward_device_op.py` (35), `test_gelu_bw_ulp.py` (63),
`test_gelu_bw_main_ulp.py` (34), `test_tanh_bw_ulp.py` (53), `test_tanh_bw_fp32_ulp.py` (17); nightly
`backward/test_backward_gelu.py` (65), `test_backward_bias_gelu.py` (46), `test_backward_tanh.py` (14),
`test_backward_sigmoid.py` (3); four docs examples; two comparison-mode cases. Gtest suites `TanhBwUlpTest`,
`GeluBwUlpTest`, `GeluBwPolyTest`, `GeluBwMainUlpTest`, `GeluBwMainPolyTest`.

The skips are environmental and the same in both runs. `test_tanh_bw_fp32_ulp.py` (17) and
`TestTanhBwNonFinite` (7) are Blackhole-only (#57509). The two comparison-mode cases need fast runtime mode off; the
separate run above covers them. The four xfails are gelu special-value cases marked in the test file.

Watcher logged no errors in either run.

**Cache-hit probe** (port only, scratch script, not committed). It ran with hit-path validation on and Watcher on,
over 20 cases: 5 program shapes × 4 op types. The shapes were addressed TILE bf16 and fp32 (DRAM interleaved),
aliased TILE (L1 height-sharded on 8 cores), RM blocked (DRAM interleaved) and aliased RM (L1 height-sharded). Each
case made three calls: a miss, a hit with freshly allocated operands (new buffer addresses, same specs), and a call
with a preallocated output. All 60 results match torch with PCC ≥ 0.99999. The cache-entry counts confirm that the
second call of every case was a hit. This checks directly that the base concept's tensor-only refresh, borrowed DFBs
included, replaces the legacy `Buffer*` patching.

**Self-audit** (hits / denominator):
- Buffer addresses in run args (`->address()`, `emplace_runtime_args`, `Buffer *`): 0 in the factory. The four
  remaining `buffer()` uses read `page_size()`, `aligned_page_size()` and `alignment()` for geometry.
- `TensorAccessorArgs`, positional `get_arg_val` / `get_compile_time_arg_val`, `.id` extraction, varargs: 0 / 9
  kernel files (positive control: 17 `get_arg_val` in the base's row-major reader).
- `cb`-shaped names: 1 / 21 files, the off-limits device-op comment above.
- Multi-binding flag: 0. Self-loops: 2 (the scratch DFBs), never combined with the flag.
- CTAs: one, named (`block_or_width_sharded`).
- `tt_metal/` files in the diff: 0. `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` outside `docs/`: 0 (positive control:
  5 in the recipe doc).
- Ephemeral `.md` citations in code: 0 / 12 changed code files.
- `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census: identical per file (factory 1, device op 16, op utils 2 + 1,
  `unary_backward.cpp` 8).
- `hw_config`: reader `(RISCV_1, NOC_0, dedicated)` and writer `(RISCV_0, NOC_1, dedicated)` equal the legacy
  `ReaderConfigDescriptor` / `WriterConfigDescriptor` values on WH and BH. Compute: fidelity from the kernel spec,
  `enable_32_bit_dest = fp32_dest_acc_en`, `unpack_modes` `UnpackToDest` on the two operand DFBs exactly when fp32
  DEST is on. All other fields keep defaults that equal the legacy defaults.
- `opt_level`: one compute `KernelSpec`, one O3 line. The DM kernels set none (O2, as in legacy).
- Install coverage: all 11 kernel files (9 bound sources plus `row_major_pages.hpp` and `operand_reconfig.hpp`)
  are in the regenerated `cmake_install.cmake` rules. The reader fork is installed by `eltwise/binary`, the writer
  fork by `eltwise/unary`.
- Pre-commit hooks pass. `clang-format` only rewrapped lines, before the build.
- After the test run, four comments in the factory were reworded (comment lines only). Pre-commit and a rebuild
  pass on the committed tree.

## Open items for downstream

- **Shared kernel touches.**
  - `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`. Rung: reused the existing fork
    `reader_binary_interleaved_start_id_metal2.cpp`; no file created. This op is the fork's first binder. Still on
    the legacy original: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`.
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. Rung: reused the existing fork
    `writer_unary_interleaved_start_id_metal2.cpp`; no file created. Still on the legacy original:
    `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp:108`;
    `examples/example/device/single_core_program_factory.cpp:83` and `multi_core_program_factory.cpp:81`;
    `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:446`, `:672`, `:811`, `:894`, `:1013`; and six Python
    generic-op tests (`kernel_lib/chain_test_lib.py:33`, `operations/debug/test_generic_op.py:98`,
    `operations/eltwise/test_clamped_silu_glu_sfpu.py:121`, `test_exp_approx_disable_sfploadmacro.py:86`,
    `test_moe_gate_sqrt_softplus_sfpu.py:126`, `test_situ_glu_sfpu.py:117`). This op no longer binds either
    original.
- **Relaxation doc, suggested edits (report only, not applied).** The doc ships unchanged at blob `ce949868d1f`.
  It has CRLF line endings (the other relaxation docs use LF). It is committed byte-for-byte, as instructed. These
  line numbers are the doc's own; I re-checked each by content.

  | Line | Now reads | Suggested |
  |---|---|---|
  | 7 | "… `TanhBwDeviceOperation`, both **deleted** — by `e8cbbd6b7d9` (#57832) and `9ad6239dc29` (#58352) …". Read in order, this pairs `GeluBw` with #57832. | "… both **deleted** — by `9ad6239dc29` (#58352) and `e8cbbd6b7d9` (#57832) respectively — …" |
  | 21 | "**This check does NOT pass on `main` at the provenance commit below** — …" | "This check failed on `main` before #63334 (`be782c33d2e`, 2026-10-30), which hashes each bound tensor's whole `tensor_spec()`. It passes on `main` from that commit on." |
  | 28 | "… The fix is draft PR #63334." | "… The fix is #63334, merged as `be782c33d2e` on 2026-10-30. Re-validated at `main` `a1ce48e3fe8` by the pre-port audit (2026-11-03)." |
  | 34 | "Relaxation verdict: `none` — conditional on a pre-port hash fix." | "Relaxation verdict: `none`. The pre-port hash fix it depended on is on `main` (`be782c33d2e`)." |
  | 38, 45, 87, 92 | present tense ("As the op stands, `none` is not safe"; "Route it to the op owner before the port"; two rows "a gap today") | past tense, anchored on `be782c33d2e`. For example, line 38: "Before #63334, `none` was not safe: …" |
  | 127 (new, from this port) | quotes the factory comment "appending one for an aliased operand would shift the other's offset" and says the writer's accessor "is appended unconditionally" | "Before the Metal 2.0 port, the factory appended no `TensorAccessorArgs` for an aliased operand, and the writer's accessor was appended but dead under `OUT_SHARDED`. After the port, the reader and writer carry no tensor binding under `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED`; the three DFBs borrow the shards." The conclusion (no accessor is compiled in, the mode cannot flip on a hit) is unchanged. |
- **Stale or misplaced comments in the device-operation class** (off-limits to the port).
  - `device/unary_backward_device_operation.cpp:250-254` says the factory bakes "a TensorAccessorArgs for its buffer
    into the writer's compile-time args" and sizes "the destination CB". After the port, the output is a
    `TensorParameter` and a DFB. The comment's conclusion still holds: key on the tensor that is really bound.
  - From the audit, still open: `:26-28` ("reads interleaved tile pages on device"), and `:187-190` (sits above the
    layout check instead of the device check, and names the deleted private `gelu_bw` op).
- **CRTA candidates (separate pass; RTA → CRTA changes dispatch).** These RTAs are the same on every node:
  `row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows` and the four paging fields of
  each tensor (both row-major kernels), and the reader fork's `block_height`, `block_width`, `num_cores_y`.
- **Dead reader args.** `block_or_width_sharded` is always 0 (`:491`), which makes `block_height`, `block_width` and
  `num_cores_y` dead. They stay, because the reused fork's interface names them.
- **Gen2 debt.** `reader_scratch` and `writer_scratch` are data-movement self-loops, which Gen2 rejects. The DM
  configs leave `disable_dfb_implicit_sync_for_all` at its default (`false`), as the recipe says. The
  `eltwise/unary` port sets it to `true` at a reviewer's request; this port had no such request. It has no effect
  on WH/BH. No `constexpr` metadata token sites were needed.
- **Test coverage notes.**
  - The fp32 tanh ULP file and the non-finite tanh cases do not run on Wormhole; Blackhole CI covers them.
  - The sweeps (`tests/sweep_framework/sweeps/eltwise/unary_backward/{sigmoid_bw, tanh_bw, tanh_bw_sharded,
    bias_gelu_bw}`) were left to CI.
