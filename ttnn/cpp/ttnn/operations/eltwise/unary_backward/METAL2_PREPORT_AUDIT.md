# Metal 2.0 Audit Findings — `eltwise/unary_backward`

- **`UnaryBackwardDeviceOperation`** (`device/unary_backward_device_operation.hpp`, `.cpp`)
  - `UnaryBackwardProgramFactory` (`device/unary_backward_program_factory.hpp`, `.cpp`). It is the only factory: `program_factory_t = std::variant<UnaryBackwardProgramFactory>` (`device/unary_backward_device_operation.hpp:28`).
- **One factory serves 40 op types.** `UnaryBackwardOpType` has 4 hand-registered entries (`SIGMOID_BW`, `TANH_BW`, `GELU_BW`, `GELU_TANH_BW`; `device/unary_backward_op_utils.cpp:15-61`) and 36 entries registered by `device/ops/*.def` files. CMake globs the `.def` files at configure time into a generated `unary_bw_ops.def.inc` (`CMakeLists.txt:39-46`). Adding one `.def` file and one compute kernel adds an op type.
- **Bound kernels: 45.** 41 compute kernels (`device/kernels/compute/eltwise_bw_*.cpp`: 5 hand-registered and 36 from `.def`), 2 own row-major dataflow kernels (`reader_unary_backward_row_major.cpp`, `writer_unary_backward_row_major.cpp`, sharing `row_major_pages.hpp`), and 2 borrowed dataflow kernels (see *Borrowed kernel files* below).
- **Unreferenced kernel files: none.** All 41 `eltwise_bw_*.cpp` files are reached through `get_kernel_spec` or a `.def` file.
- **Composite layer (not part of the port):** `unary_backward.cpp` routes the gradient entry points into this device operation. The 36 `.def` types are launched only for BF16 TILE operands on WH/BH (`generated_bf16_kernel_applies`, `unary_backward.cpp:46-79`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `32fff42bf79 2026-11-21 Add unary_backward TensorParameter relaxation analysis`
*(Verbatim output of the provenance command, re-run after the recipe commits were stripped from this branch; the audit first recorded the same commit as `0b4911ccc4f6`. That commit is this branch's relaxation-analysis commit, which touches `metal_2.0/analyses/relaxations/` and is now the only file left under `metal_2.0/`, so the line does not pin the recipe. The recipe is upstream `akertesz/op-porting-recipe` at `c41f7f4ef0a` (2026-10-29, `docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`), the newest upstream commit touching `metal_2.0/`. At audit time this branch's `metal_2.0/ai/` tree equalled that commit's (tree `3137c0f8edb5`); that can now be checked upstream with `git rev-parse c41f7f4ef0a:docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai`. `audit/metal2_audit.md` blob `6435c8fcae15`. The relaxation-analysis commit named above has since been removed from this branch. The analysis doc now lives on `akertesz/op-porting-recipe` (blob `83ff1bbed224`, see the line that starts *Doc.* below).)*

**Audit basis**

- **Tree.** Branch `anasuya/metal2_unary_backward_port` at the audit = `origin/main` `5340d88ced5` (2026-11-21) + 139 recipe commits + 1 local commit that adds the relaxation analysis (`0b4911ccc4f` then, `32fff42bf79` since the recipe commits were stripped). That commit has since been removed from the branch too, and the doc now lives on `akertesz/op-porting-recipe`. Nothing outside `docs/.../metal_2.0/` differs from `origin/main`. The op directory is identical to `origin/main` and has no uncommitted changes.
- **Upstream drift.** Upstream `main` was at `5ed70985d93` when this ran. The last commit there that touches the op directory is #63799 (`2301d23c13b`, 2026-11-19), which this tree already contains (GitHub API, 2026-11-21). The upstream recipe tip is still `c41f7f4ef0a`.
- **Method.** Static. No device run: the build on disk (`build_Release/lib/_ttnncpp.so`, 2026-11-20 10:10) is older than this checkout (2026-11-21 09:50), and no rebuild was done. The Device 2.0 closure is a static `#include` walk with no `.dephash` cross-check.
- **Readiness sheet.** Not fetched. The invoker supplied the op's row (see *Gate detail*) and told the audit to ignore three stale rows.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/unary_backward` |
| **Overall** | **GREEN** (on the invoker-supplied sheet row; see *Result*) |
| **DOps / Factories** | `UnaryBackwardDeviceOperation` → `UnaryBackwardProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — 45 bound kernels and the 17 ttnn-side support files they reach; 0 violations |
| *Prereqs* — Cross-op escapes | Ok — `kernel_lib` is already `DataflowBuffer`-object based; 2 in-family borrowed dataflow kernels |
| *Feature Support* — overall | GREEN (all three Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — no kernel reads compile-time args at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (`yes`, invoker-supplied row; primary cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD` |
| *TTNN Readiness* — Custom hash | Yes (not a gate; port leaves it intact): `device/unary_backward_device_operation.cpp:273-300` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (the sheet's `Porting Target`) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `grad_output`, `input`, `output`: Case 1 in the address configs, clean (borrowed DFB) in the aliased config. No Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `legality (see analysis)` — clears; analysis doc read, 6 of 6 validity checks pass |
| *Port work* — TensorAccessor 3rd arg | none — 6 accessor sites, all two-argument |
| *Port work* — CB endpoints | legal (`c_0`, `c_1`, `c_2` in every config) + self-loop (`c_3`, `c_4` under `rm_blocked` only) |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). Every gate clears for the whole op. No gate is scoped to one factory path. There is no subset language.

These are the two things this GREEN stands on:

1. **The sheet row was supplied by the invoker.** The live sheet has three stale rows for device operations that no longer exist and no row for `UnaryBackwardDeviceOperation`. The invoker told the audit to ignore the stale rows and to treat the supplied row as the op's row. A strict reading of the recipe's *Factory-set match* rule against the sheet as it stands would gate on this (three phantom rows plus one missing row → spreadsheet-broken). The stale rows are recorded below for the readiness-sheet owner.
2. **The relaxation verdict `legality` is carried by an analysis doc** (`analyses/relaxations/eltwise_unary_backward.md`; at the audit it was a local commit on this branch, and it now lives on `akertesz/op-porting-recipe`). The doc's six validity checks pass against this tree today. The doc calls itself perishable. The checks must be re-run if the op's files change before the port starts.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The cell reads `yes` on the invoker-supplied row. The primary-column cross-check came back clean (table below). The derived cells (`Is able to port?`, `Porting Target`) were read, not vetted.

  | Sheet column (by header name) | Supplied value | Code evidence | Result |
  |---|---|---|---|
  | `Op` | `eltwise/unary_backward` | the op directory audited | ✓ |
  | `Device operation` | `UnaryBackwardDeviceOperation` | `device/unary_backward_device_operation.hpp:23` | ✓ |
  | `Factory (variant)` | `UnaryBackwardProgramFactory` | `.hpp:28`: a one-alternative variant; zero hits for `select_program_factory`. One supplied row ↔ one factory in code | ✓ factory-set match |
  | `Concept` | `descriptor` | `create_descriptor(args, tensor_args, output)` returns `ProgramDescriptor` (`device/unary_backward_program_factory.hpp:19-20`, `.cpp:26-27`); no coordinate parameter. Zero hits for `create_workload_descriptor`, `create_mesh_workload`, `create_at`, `create_program_artifacts`, or a static `create(` | ✓ |
  | `Op Classification` | `PD Op (pointer-patching) + hash` | `Buffer*` pushed through `emplace_runtime_args` (`.cpp:440,445,459,468,470`) and CB `.buffer` set when aliased (`.cpp:252`). The adapter selects the BufferBinding fast path on exactly this (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:357-361`). Custom hash: next rows | ✓ |
  | `Execution Model` | `SPMD` | `create_descriptor` takes no coordinate; one program serves the mesh | ✓ |
  | `Custom hash (compute_program_hash)` | `yes` | `device/unary_backward_device_operation.cpp:273-300` (declared `.hpp:35`) | ✓ |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` | zero hits for `attribute_values`, `attribute_names`, `to_hash` in the op directory | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | zero hits for `get_dynamic_runtime_args`, `DynamicRuntimeArg` | ✓ |
  | `Override runtime args method? (PD only)` | `no` | zero hits for `override_runtime_arguments` | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `unary_backward_nanobind.cpp` has no `create_descriptor`, `nb::class_` or device-op reference (positive control: 114 `nb::` / `bind_registered_operation` hits in the same file) | ✓ |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | zero `->address()` / `.address()` hits in the op directory. Every tensor reaches a kernel as a `Buffer*` | ✓ |
  | `Known op issues` | *(empty)* | no code-visible counterpart | n/a |
  | `Is able to port?` | `yes` | derived cell | gate cleared |
  | `TensorParameter relaxation` | `legality (see analysis)` | see the relaxation bullet below | clears |
  | `Op-owned tensors?` | `no` | the factory returns a `ProgramDescriptor`, which cannot carry op-owned tensors | ✓ |
  | `Porting Target` | `ProgramSpecFactoryConcept` | derived cell; consistent with no override and no op-owned tensors | read |
  | `Factory definition path`, `Declared in` | the two `.hpp` paths in the invoker's row | both files exist | ✓ |

  Cross-column invariants: `get_dynamic_runtime_args == yes` and `Op-owned tensors == yes` are both `no`. Neither invariant can be violated.

  **Stale sheet rows, for the readiness-sheet owner (not gated, per the invoker's override).** The sheet carries three rows for device operations that are gone and has no row for the live one:

  | Sheet `Op` | State in code |
  |---|---|
  | `eltwise/unary_backward/gelu_bw` | phantom. No device-op or factory source for it remains under `ttnn/cpp`. Gelu now routes to `GELU_BW` / `GELU_TANH_BW` (`unary_backward.cpp:2104-2122`) |
  | `eltwise/unary_backward/tanh_bw` | phantom. Tanh now routes to `TANH_BW` / `TANH_BW_BF16` (`unary_backward.cpp:438-468`) |
  | `experimental/unary_backward/gelu_backward` | phantom. There is no `experimental/unary_backward` directory |
  | *(missing)* `eltwise/unary_backward` · `UnaryBackwardDeviceOperation` · `UnaryBackwardProgramFactory` | no row |

  The relaxation analysis doc attributes the three deletions to #58352, #57832 and #54799. The cell values on the stale rows were not re-read in this session.

- **Device 2.0 (every kernel used): GREEN.** 45 bound kernels plus the 17 ttnn-side support files they reach (the 15 `kernel_lib` headers, `operand_reconfig.hpp`, `row_major_pages.hpp`).
  - *Method.* A static `#include` walk from every bound kernel, resolved in the JIT search order (`tt_metal/jit_build/build.cpp:526-536` plus the arch includes in `wh_hal.cpp`), following every `#if` branch. It reaches 62 ttnn-side files; only standard-library headers are unresolved. The 92 `tt_metal/`-side headers it touches are LLK/HAL and are not scanned.
  - *Scan.* Over the 62 files: raw `noc_async_*`, `noc_semaphore_*`, `get_noc_addr*`, `InterleavedAddrGen*` / `ShardedAddrGen`, CB-index free functions (`cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`), `get_semaphore`, `CircularBuffer&`, and raw `tt_l1_ptr` pointers all have zero hits. The only `volatile` is a `CoreLocalMem<volatile ...>` template argument (`row_major_pages.hpp:65-66`), not a raw pointer. (Positive control: the same regex gives 6 hits in a known Device 1.0 file.)
  - *Hits that are allowed:*

    | File | Line | Call | Why it is fine |
    |---|---|---|---|
    | `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | 45, 52 | `get_tile_size(cb_id_in0/1)` | sanctioned free function |
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 27 | `get_local_cb_interface(cb_id_out).fifo_page_size` | sanctioned free function |
    | `device/kernels/dataflow/reader_unary_backward_row_major.cpp` | 77-78 | `dfb_grad.get_write_ptr()`, `dfb_input.get_write_ptr()` | methods on `DataflowBuffer` objects |
    | `device/kernels/dataflow/row_major_pages.hpp` | 142 | `scratch.get_write_ptr()` | method on a `DataflowBuffer` |
    | `device/kernels/dataflow/writer_unary_backward_row_major.cpp` | 56 | `dfb_dst.get_read_ptr()` | method on a `DataflowBuffer` |

  - Local L1 copies use `CoreLocalMem` (`row_major_pages.hpp:65-66`), which is the Device 2.0 form. The compute kernels hold `DataflowBuffer` objects and use the `compute_kernel_lib` chain, which takes each `DataflowBuffer` by reference (`kernel_lib/eltwise/api/chain.hpp:48-66`).

- **Feature compatibility: GREEN (all N/A).** Zero hits in the factory, device operation, and kernels for any Appendix A signal.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | zero hits for `GlobalCircularBuffer`, `global_circular_buffer`, `remote_cb`, `global_cb`, `remote_index` |
  | CBDescriptor `address_offset` (non-zero) | N/A | zero hits for `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor`. No `CBDescriptor` sets an offset (`.cpp:244-254`, `:279-287`) |
  | GlobalSemaphore | N/A | zero hits for `GlobalSemaphore`, `global_semaphore`; the op has no semaphores at all (zero `semaphore` hits under `device/`) |

- **CB endpoints (GATE-free).** Every CB is either a legal 1P+1C FIFO or a one-toucher self-loop. Nothing needs the multi-binding flag, and nothing is dead.
  - *Configs* come from the factory: ALIAS (`can_alias_shards`, `.cpp:82-129`), ADDRESS-TILE (not aliased, TILE), and ADDRESS-RM (`rm_blocked`, `.cpp:138`). One reader, one writer and one compute `KernelDescriptor` run on `all_cores` (`.cpp:476-478`). Cores in `core_group_1` and `core_group_2` differ only in per-core RTA values.

  | CB | ALIAS (TILE or RM) | ADDRESS-TILE | ADDRESS-RM (`rm_blocked`) |
  |---|---|---|---|
  | `c_0` grad_output | borrowed from the grad_output shard (`.cpp:256-260`). Reader produces (`reserve_back(n)` + `push_back(n)`, `reader_binary_interleaved_start_id.cpp:41-43`), compute consumes. **1P+1C, legal** | reader produces per tile (`:91,103`), compute consumes. **legal** | RM reader produces (`reader_unary_backward_row_major.cpp:75,105`), compute consumes. **legal** |
  | `c_1` input | as `c_0` (`reader_binary_interleaved_start_id.cpp:48-50`). **legal** | as `c_0` (`:96,107`). **legal** | as `c_0` (`:76,106`). **legal** |
  | `c_2` grad_input | borrowed from the output shard (`.cpp:262-266`). Compute produces (`PackTile`), writer consumes (`OUT_SHARDED` wait + pop, `writer_unary_interleaved_start_id.cpp:32-36`). **legal** | compute produces, writer consumes per page (`:51,54`). **legal** | compute produces, RM writer consumes (`writer_unary_backward_row_major.cpp:55,70`). **legal** |
  | `c_3` reader scratch | absent | absent | allocated only here (`.cpp:272-289`). Touched by the RM reader alone, through the raw pointer only (`reader_unary_backward_row_major.cpp:61-63`, `row_major_pages.hpp:140-143`). **one toucher → self-loop** |
  | `c_4` writer scratch | absent | absent | allocated only here. Touched by the RM writer alone (`writer_unary_backward_row_major.cpp:42-43`). **one toucher → self-loop** |

  - *Hidden second writer:* none. The op has no semaphores. The only raw pointer accesses on `c_0`..`c_2` are the FIFO producer's own `get_write_ptr()` after `reserve_back` (RM reader `:75-78`) and the FIFO consumer's own `get_read_ptr()` after `wait_front` (RM writer `:55-56`).
  - *Multiple readers, dual-instance work-split:* none. Each of the three kernels is bound once, from a distinct source.
  - *Dead CB:* none. `c_3` and `c_4` exist only under `rm_blocked` because the factory allocates them only there (`.cpp:272`). The host-side condition already exists.
  - *Compile-time CB numbers outside LLK calls:* the only literals are `tt::CBIndex::c_0/c_1/c_2` in the 41 compute kernels (for example `eltwise_bw_abs.cpp:21-25`) and `c_0..c_4` in the RM kernels. The shared lib headers hardcode none (comment text only: `chain.hpp:100`, `chain.inl:423`). On the host, `unpack_to_dest_mode` is indexed by CB number (`.cpp:362-366`).

- **Offset base pointers: GREEN.** Every tensor address that reaches a kernel is a bare `Buffer*`: the locals at `.cpp:236-238`, passed at `:440,445` (RM reader), `:459` (RM writer), `:468` (borrowed reader), `:470` (borrowed writer). There is no `->address()` and no `+ offset`.
  - The only offset-like values are `units_done` (a per-core start tile or block index, never an address) and the kernels' in-page `offset_bytes`, passed to `Noc::async_read` / `async_write` as an accessor argument inside one page (`row_major_pages.hpp:90,99,122,131,134`). Neither is a base.
  - Type 3: zero hits for the `address_offset` family. Type 4: zero hits for `narrow` and `MeshBuffer::create`.
  - Reconciliation: no fold, and the op is not in the 2026-07-19 triage table (blob `3d2b1be9af92`) → clean. The addresses go to TensorParameter analysis.

- **TensorAccessor 3rd argument: N/A.** No accessor in the op passes a 3rd argument. The six constructions in the bound kernels are all `TensorAccessor(args, base_addr)`: `reader_binary_interleaved_start_id.cpp:46,53`, `writer_unary_interleaved_start_id.cpp:42`, `reader_unary_backward_row_major.cpp:55-56`, `writer_unary_backward_row_major.cpp:38`. The op is not in the 2026-07-06 triage table (blob `20ad52f8fd77`).

- **TensorParameter relaxation (sheet cell `legality (see analysis)`): clears.**
  - *Doc.* `analyses/relaxations/eltwise_unary_backward.md`, found by the Op-cell convention. Blob `f0c9ad2ebc69`, 127 lines. At the audit it was the commit `32fff42bf79` on this branch (`0b4911ccc4f` before the recipe commits were stripped), a cherry-pick of the analysis commit `57ace0d883e`. That commit has since been removed from this branch. The doc is on `akertesz/op-porting-recipe` as blob `83ff1bbed224`, which differs on line 15 only (two links became relative). The doc names its authors as "Anasuya and Claude" and says it was analysed against `main` at `f0d4c1fc7d0` (2026-11-20). It covers exactly the one supplied row.
  - *Validity checks, run against this tree.* All six pass:

    | # | Check | Evidence | Result |
    |---|---|---|---|
    | 1 | The key pins each tensor exactly as the declaration matches it | `compute_program_hash` keys `input`, `grad_output` and a preallocated output with `hash_tensorspec_with_relaxation(spec, {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true})` (`unary_backward_device_operation.cpp:286-289`, `:295-299`). The framework's hash and its match share one field set (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:97-126`, hash `:155-201`, match `:203-`). For these two flags both pin dtype, page config (with `Tile`), memory config and padded shape, and free logical shape and `Alignment`. The derived output slot takes dtype and memory config from the hashed `args` (`:296`) and page config and padded shape from the input (check 3) | pass |
    | 2 | The factory never reads logical shape or layout alignment | zero hits for `logical_shape`, `get_alignment`, `tensor_layout`, `logical_2d`, `logical_volume` in `unary_backward_program_factory.cpp` (positive control: the same grep set finds `create_descriptor`). It reads only the buffer's allocator `alignment()` (`:167,178`), which the doc allows | pass |
    | 3 | The derived output's padded shape comes from the input's | `compute_output_specs` calls `tensor_layout_from_padded_shape(output_dtype, PageConfig(input.layout()), args.output_memory_config, input.logical_shape(), input.padded_shape())` (`unary_backward_device_operation.cpp:255-262`) | pass |
    | 4 | The op is still the base concept | zero hits for `override_runtime_arguments`, `get_dynamic_runtime_args`, `DynamicRuntimeArg`. Per-core RTAs are baked at the miss from padded-shape values (`num_tiles` `.cpp:43`, RM paging `:155-186`, loop `:419-474`) | pass |
    | 5 | Validation ties the slots together and runs on hits | `validate_on_program_cache_miss` ties `grad_output` and a preallocated output to the input's logical shape, padded shape and layout (`unary_backward_device_operation.cpp:178-198`, `:208-232`). There is no `validate_on_program_cache_hit` in the op directory, and the framework then runs the miss validator on every hit (`ttnn/api/ttnn/device_operation.hpp:374-378`) | pass |
    | 6 | Exactly one factory | `program_factory_t = std::variant<UnaryBackwardProgramFactory>` (`.hpp:28`); zero hits for `select_program_factory` | pass |

  - *Declaration (transcribed from the doc, §2).* On every `TensorParameter` (`grad_output`, `input`, `output`), unconditionally, in every layout, sharding regime and op type: `.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`. Not `{}`, not `match_padded_shape_only` alone, and not `dynamic_tensor_shape`, because check 4 shows a padded-shape change would leave baked arguments stale.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `grad_output`, `input` and `output` are each **Case 1** (`TensorAccessor`) in the address configs and **clean** (borrowed-DFB read) in the aliased config. No Case 2.
- **TensorParameter relaxation:** `{.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}` on all three slots, unconditionally (analysis doc §2).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_3` and `c_4` under `rm_blocked` (ROW_MAJOR, not aliased). `c_0`, `c_1`, `c_2` are legal 1P+1C in every config.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** two borrowed dataflow kernels. The binary reader has no `_metal2` fork yet; the port creates it (rung 2). The unary writer already has one; the port binds it (rung 1). Details under *Borrowed kernel files*.
- **RTA varargs:** none.
- **Anything else the porter needs:**
  - **Scope is 41 compute kernels, all in legacy form.** After the port the shared factory can launch only Metal 2.0-form kernels. All 41 convert in the one change. 36 of them are structural clones (normalized diff against `eltwise_bw_abs.cpp`: 2 differing lines each; all have `lane_width = 2`, one positional RTA, no CTAs, no `#if`).
  - **The op-type set is open-ended.** A new op type is one `.def` file plus one legacy-form kernel (`CMakeLists.txt:40`). The contract comment new authors copy, `device/unary_backward_op_utils.hpp:19-21`, still describes the legacy form (CB indices `c_0`/`c_1`/`c_2`, positional RTA 0) and goes stale at the port. See Question 1.
  - **Names that tests depend on.** 38 nightly `test_backward_*.py` files and 38 `*_bw_device_perf.py` files depend on graph-capture strings containing `UnaryBackwardDeviceOperation` and `UnaryBackwardOpType::<NAME>`, as does `tests/ttnn/unit_tests/operations/eltwise/test_unary_backward_device_op.py`. A factory-only port leaves those names alone.
  - **Install coverage.** Both donor directories and this op install kernels through `GLOB_RECURSE` (`eltwise/binary/CMakeLists.txt:16`, `eltwise/unary/CMakeLists.txt:17-19`, `unary_backward/CMakeLists.txt:16-18`). The new reader fork and the converted kernels need no CMake edit.
  - **Post-port passes, not port work.** The two RM scratch DFBs are the kind of buffer the post-port `sync_free_dfbs.md` pass converts to `Scratchpad`. `gen2_hardware_configs.md` is the pass that adds Gen2 configs. See Question 2 for one Gen2 option with a precedent on this op.

## TTNN ProgramFactory

### Concept
`ProgramSpecFactoryConcept` (the sheet's `Porting Target`). There is no `override_runtime_arguments` to translate. The framework refreshes tensor bindings on a cache hit.

### Fit
- Execution model: `SPMD`. Nothing varies per coordinate: `create_descriptor` takes no coordinate (`unary_backward_program_factory.cpp:26-27`).
- Op-owned device resources: none. There are no op-owned tensors and no semaphores of any kind.
- Tensor-arg matching: **relaxed, not strict**, as declared by the analysis doc (`legality`: `match_padded_shape_only` + `relax_tensor_layout_alignment` on all three slots). This is the doc's declaration and not a port-time call. The custom hash computes each slot's key with the framework's own `hash_tensorspec_with_relaxation` under the same flags. The hash is never looser than the declaration.
- Ported-from shape: 1:1. A `descriptor` factory without a coordinate, no op-owned tensors.

### Custom compute_program_hash
Present at `device/unary_backward_device_operation.cpp:273-300` (declared `.hpp:35`). There is no backdoor `attribute_values` / `to_hash`. It is recorded so the porter knows it is there and leaves it alone; the port never edits it.

### Stop signals
None.

## Team-only

- **Out-of-directory coupling & donor shape.**
  - *Op-level roll-up:* **✓ clean** for function-call escape. No donor shape is ⚠, ✗ or ⭐: no `CircularBuffer&`, no semaphore shapes, no `TensorAccessorArgs` / CTA-offset / old addr-gen shapes.
  - *Summary table:*

    | Op kernel(s) | Donor | Class | What is used | Shape | Status |
    |---|---|---|---|---|---|
    | 41 compute kernels, `operand_reconfig.hpp` | `ttnn/cpp/ttnn/kernel_lib/eltwise/**` and `kernel_lib/dest_helpers.hpp` (15 files) | 2, shared lib | `ckl::eltwise_chain`, `IterationShape::tiles`, `input()` / `output()` specs, `CopyTile` / `PackTile` and DEST-only elements | `uint32_t cb_id` (specs) and `DataflowBuffer&` (element constructors) | ✓ |
    | 41 compute kernels | `tt_metal` LLK API: `compute_kernel_hw_startup.h`, 36 `<name>_bw.h` headers, `eltwise_unary/eltwise_unary.h` (2 kernels) | 1, LLK / HAL | `compute_kernel_hw_startup(uint32_t, uint32_t)`, `<name>_bw_tile_init` / `<name>_bw_tile` | `uint32_t cb_id` | ✓ no concern |
    | RM reader and writer | own `row_major_pages.hpp`; `tt_metal` dataflow API | in-op, 1 | `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem` | n/a | ✓ |
    | borrowed reader and writer | `tt_metal` dataflow API only (no ttnn-side includes) | 1 | n/a | n/a | ✓ |

    Per-call detail is omitted: every row is ✓.
  - *Borrowed kernel files (file-path instantiation).* The census greps the filename, then checks the bound path of each hit.

    | Kernel | Owner | Other binders | `_metal2` fork beside it |
    |---|---|---|---|
    | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | `eltwise/binary` (in-family) | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`, a gtest on the legacy generic-op API. No other op factory | **none** → this port creates it (rung 2) |
    | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` (in-family) | of the legacy original: `test_generic_op.cpp:894,1013`, `tests/ttnn/unit_tests/operations/eltwise/test_exp_approx_disable_sfploadmacro.py:86`, `tests/ttnn/unit_tests/operations/debug/test_generic_op_host_time.py:68`. No other op factory. (`data_movement/slice`'s tile factory binds its own private copy of the same filename, which is not a binder.) 26 other op factories bind the fork | **`writer_unary_interleaved_start_id_metal2.cpp` exists** → bind it (rung 1) |

    These lists are the **sunset lists**, not authorization to convert either original in place. The writer fork fits this op: it reads `dfb::out`, `tensor::dst` and named args `num_pages`, `start_id`; this factory passes `{grad_input_buffer, units, units_done}` (`.cpp:470`) and supplies `OUT_SHARDED` under aliasing (`:345-347`); the fork's `BACKWARDS` and `ENTRIES_PER_PAGE` / `PAGE_BYTES` knobs are not used here.
  - The 41 compute kernels, the RM reader and writer, and `row_major_pages.hpp` are this op's own. No other file in the repo references `unary_backward/device/kernels`. None is lent.
- **Relaxation candidates** (noticed in the custom hash; FALLIBLE, default strict): none beyond what the analysis doc already declares. The key leaves logical shape and `Alignment` free, which is the declared relaxation.
- **TTNN factory analysis:** see the `## TTNN ProgramFactory` section above. Facts with evidence are in the cross-check table under *Gate detail*.

## Misc anomalies  *(team-only, non-gating)*

- **The 36 `.def` kernels assume BF16 TILE operands, and only the composite layer enforces it.** Each hard-codes `DataFormatReconfig::Disabled` on both inputs and the output (for example `eltwise_bw_abs.cpp:42,51,58`). The device operation does not restrict those op types: `validate_operand` accepts BF16, FP32, BFP8 and BFP4 for every op type (`unary_backward_device_operation.cpp:54-57`), and the factory builds the fp32-DEST and mixed-format settings for them (`unary_backward_program_factory.cpp:355-384`). The restriction lives in `generated_bf16_kernel_applies` (`unary_backward.cpp:46-79`). Any caller that reaches `launch_unary_backward` with a `.def` type and a non-BF16 or mixed pair would run with a wrong unpack format. There is no live bug, because every call site is gated.
- **Dead RTAs in the borrowed reader, for this op.** The factory passes `block_height = 0`, `block_width = 0` and `num_cores_y` (`unary_backward_program_factory.cpp:468`), and CTA 0 (`block_or_width_sharded`) is always `0` (`:324`). The kernel reads those three only under `if constexpr (block_or_width_sharded)` (`reader_binary_interleaved_start_id.cpp:60-87`). They are dead for this op and live for the kernel's other binder.
- **Two compute kernels name the RTA differently.** `eltwise_bw_gelu_tanh.cpp:19` and `eltwise_bw_gelu_tanh_fp32.cpp:21` read it as `num_tiles`; the other 39 read `per_core_tile_cnt`, which is the name the contract comment uses. Cosmetic today, because the RTA is positional.

## Questions for the user

1. **Op types that land while the port is in flight.** Your rule from the earlier port (2026-11-03) was to port the op types that exist, and to stop and tell you if the enum changes mid-port. Since then 36 `.def` op types have landed, each with a legacy-form compute kernel, and the glob at `CMakeLists.txt:40` lets more land without a shared-file edit. Two consequences follow. First, the ported factory can launch only Metal 2.0-form kernels. All 41 compute kernels convert in the one change; there is no partial scope. Second, a `.def` PR that merges after the port starts brings one more legacy-form kernel at the next rebase, and the contract comment new authors copy (`device/unary_backward_op_utils.hpp:19-21`) still describes the legacy form. For a kernel that arrives mid-port, should the porter (a) stop and tell you, as before, or (b) convert it under the same pattern when it is a clone of the 36, list it in the report, and stop only for a kernel that is not a clone? **Recommendation: (b).** The conversion is mechanical and the factory is shared. Option (a) would cost a round trip per landed PR. The brief currently says "ask"; it will say "convert clones" once you confirm (b).

2. **Implicit-sync opt-out on the aliased reader and writer.** On the earlier port of this op (PR #64224, round 1) you chose to apply `create_reader_datamovement_config(can_alias_shards)` and `create_writer_datamovement_config(can_alias_shards)` (commit `fd1d2dd967b`, 2026-11-05), which disables DFB implicit sync for the aliased reader and writer. It has no device effect on WH/BH. You made the same call on the unary port. The recipe says a Gen1 port does not author Gen2 configuration (`gen2_hardware_configs.md` adds it later). Should the fresh port carry the opt-out over, or follow the recipe and leave it out? **Recommendation: carry it over.** It matches your earlier decisions, and the same review comment is likely to recur. The brief leaves this undecided until you answer, and tells the porter to ask.

## Recipe notes

- **Provenance command.** `git log -1 --format='%h %cs %s' -- docs/.../metal_2.0/` prints the newest commit that touches anything under `metal_2.0/`, and `analyses/` sits under it. On this branch that is the relaxation-analysis commit, not a recipe commit. The line does not pin the recipe by itself. I pasted it verbatim and added the recipe tip beside it. Scoping the command to `ai/` would pin the guidance. The relaxation doc is also an audit input. Its blob hash should be recorded too, and I did.
- **No recipe path for a user-supplied sheet row.** *TTNN factory concept prerequisite* tells the auditor to fetch the sheet, and its *Factory-set match* rule turns phantom or missing rows into a gate. Here the invoker supplied the row and overrode the stale ones. I followed the invoker, did not fetch the sheet, and recorded that a strict reading would gate. A sentence in the recipe on an invoker-supplied row would remove the ambiguity.
- **Table-driven kernel selection.** *Scope of the audit* says to follow `KernelDescriptor::kernel_source` references. This op selects its compute kernel through `get_kernel_spec` and a configure-time-generated `.def.inc`, not a literal in the factory. I enumerated the kernels from the `.def` files and checked the count against the directory (41 of 41). The recipe has no subject for an op whose kernel set grows while the port is in flight, which is the main execution risk here. I carried it in *Heads-ups* and Question 1.
- **Self-loop wording vs the post-port Scratchpad pass.** *CB endpoints* says a one-toucher DM scratch CB becomes a self-loop. The post-port `sync_free_dfbs.md` and `dm_self_loop_dfbs.md` passes then turn such DFBs into `Scratchpad`s. The audit does not say whether the port should bind a `Scratchpad` directly for a pointer-only CB. I followed the audit's self-loop wording and named the post-port pass in the brief.
- **Per-config split.** *TensorParameter analysis* allows a per-factory split when one binding is clean in one factory and Case 1 in another. Here the split is per config inside one factory (aliased vs address). I recorded it per config by analogy.
