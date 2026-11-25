# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device`

- **`ReshapeDeviceOperation`** (`device/reshape_op.hpp:15`; `program_factory_t` at `:20`)
  - `ReshapeRMProgramFactory` (`device/reshape_rm_program_factory.cpp`), selected for a ROW_MAJOR input (`device/reshape_op.cpp:21-27`)
  - `ReshapeTileProgramFactory` (`device/reshape_tile_program_factory.cpp`), selected for a TILE input
- Kernels: three owned by the op, one borrowed. Every kernel file in the directory is referenced by a factory; there are no unreferenced ones, and no compute kernels (the op is data-movement only).
  - RM: `device/kernels/dataflow/reader_unary_reshape_stick_layout_interleaved_multi_core.cpp`, `device/kernels/dataflow/writer_unary_reshape_stick_layout_interleaved_multi_core.cpp`
  - Tile: `device/kernels/dataflow/reader_unary_reshape_interleaved.cpp`, plus the borrowed writer `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (`device/reshape_tile_program_factory.cpp:107-108`)
- Outside the device operation, not ported: the host wrapper `reshape.cpp` / `reshape.hpp` (`ttnn::reshape_on_device`) and `reshape_nanobind.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** ``547b694ae6f8 2026-11-23 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form``
(This branch's local rebase of upstream `c41f7f4ef0a`. The `metal_2.0/` tree is identical to upstream's, `01c2d3ae2182`. Audit recipe blob `6435c8fcae15`. The recipe is tracked on this branch and was read in place.)

**Audited tree:** branch `anasuya/metal2_reshape_on_device` = `origin/main` `e39e42718ca` plus the recipe commits. The op directory is identical to `origin/main`; its last upstream change is #63798 (`fa826ca2a9a`, 2026-11-19). The readiness sheet was fetched live on 2026-11-23 (477 data rows, 28 columns); the CSV stayed in session scratch, not in the repo.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device` |
| **Overall** | **GREEN** (two user-directed readings, see Result) |
| **DOps / Factories** | `ReshapeDeviceOperation` → `ReshapeRMProgramFactory`, `ReshapeTileProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes: 4 kernels, 0 violations (local volatile L1 byte copies: user-ruled compliant) |
| *Prereqs* — Cross-op escapes | Ok: no function-call escape; one borrowed kernel file (tile writer), `_metal2` fork already exists |
| *Feature Support* — overall | GREEN (all three Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (no CTA varargs) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both rows) |
| *TTNN Readiness* — Execution Model | `SPMD` (both rows) |
| *TTNN Readiness* — Custom hash | Yes (not a gate; port leaves it intact): `device/reshape_op.cpp:111-127` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (the sheet's `Porting Target`) |
| *Port work* — Offset base pointer | none: 4 address RTAs, none folds an offset |
| *Port work* — Tensor bindings (per binding) | Case 1 × 4 (RM input and output, tile input and output) |
| *TTNN Readiness* — TensorParameter relaxation | sheet cell `Padded Shape + Alignment` (verbatim, both rows) → read as `legality (see analysis)` per the user; analysis doc validated 6/6 |
| *Port work* — TensorAccessor 3rd arg | none: no accessor passes a 3rd argument |
| *Port work* — CB endpoints | legal 1P+1C (RM `src0`, tile `in0`); conditional DFB + self-loop (tile scratch CB 1, Blackhole + BFLOAT16 + DRAM input only) |

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear on this tree, and the op sits on the base concept, `ProgramSpecFactoryConcept`. No gate is factory-scoped, so there is no subset question. No open questions for the user.

Two readings were directed by the user and are recorded as such:

1. **Relaxation cell.** Both sheet rows read `Padded Shape + Alignment`. That is not in the audit's vocabulary (`none` / `dynamic (see analysis)` / `legality (see analysis)`). The user directed that it be read as `legality (see analysis)` for both rows. The analysis doc `analyses/relaxations/data_movement_reshape_on_device.md` exists (untracked on this branch, blob `a437b248cfad`) and all six of its validity checks pass here, so the relaxation verdict is CONFIRMED, not UNCONFIRMED. A strict reading of the recipe would GATE this conjunct (→ ops team) until the sheet cell changes; the value is recorded verbatim below for the readiness-sheet owner.
2. **Local volatile L1 byte copies.** The `volatile tt_l1_ptr uint8_t*` copies in `reader_unary_reshape_interleaved.cpp` (`:40`, `:87-91`) count as Device 2.0 compliant. The user ruled on this pattern on 2026-10-22. Recorded as decided.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The sheet's `Is able to port?` is exactly `yes` on both rows, `Known op issues` is blank on both, and the primary cross-check is clean.

  Live sheet cells, read by header name:

  | Column | `ReshapeRMProgramFactory` | `ReshapeTileProgramFactory` |
  |---|---|---|
  | `Is able to port?` | `yes` | `yes` |
  | `Concept` | `descriptor` | `descriptor` |
  | `Op Classification` | `PD Op (pointer-patching) + hash` | `PD Op (pointer-patching) + hash` |
  | `Execution Model` | `SPMD` | `SPMD` |
  | `Porting Target` | `ProgramSpecFactoryConcept` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `yes` | `yes` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` | `no` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | `no` |
  | `Override runtime args method? (PD only)` | `no` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | `no` |
  | `Known op issues` | (blank) | (blank) |
  | `TensorParameter relaxation` | `Padded Shape + Alignment` | `Padded Shape + Alignment` |
  | `Op-owned tensors?` / `Secretly SPMD Workload?` | (blank) | (blank) |
  | `Diego validation` | `no` | `no` |
  | `Uses llama kernels? (primary or shared)` | `no` | `yes` |
  | `ProgramFactory used in llama?` | `no` | `no` |

  `Diego validation`, `Uses llama kernels?` and `ProgramFactory used in llama?` are not columns the recipe reads. They are recorded verbatim and not interpreted.

  Cross-check of the cheaply-checkable columns against the code:

  | Sheet column | Sheet | Code evidence | Match |
  |---|---|---|---|
  | `Concept` | `descriptor` | both factories define `static create_descriptor(...)` returning `ProgramDescriptor` (`device/reshape_rm_program_factory.cpp:21`, `device/reshape_tile_program_factory.cpp:21`); no `create_workload_descriptor`, `create_program_artifacts`, `create_mesh_workload` or `create_at` anywhere in the directory | ✓ |
  | `Custom hash` | `yes` | `compute_program_hash` declared `device/reshape_op.hpp:34`, defined `device/reshape_op.cpp:111-127` | ✓ |
  | Backdoor hash | `no` | no `attribute_values` / `to_hash` in the directory | ✓ |
  | `get_dynamic_runtime_args` | `no` | no such hook in the directory | ✓ |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` in the directory | ✓ |
  | `Pybind descriptor` | `no` | `reshape_nanobind.cpp:68` binds only the user-facing `reshape_on_device` function; no `create_descriptor`, no `nb::class_` | ✓ |
  | Factory-set match | two rows | `program_factory_t = std::variant<ReshapeTileProgramFactory, ReshapeRMProgramFactory>` (`device/reshape_op.hpp:20`); both `Factory definition path` files exist; no phantom or missing row | ✓ one-to-one |
  | Cross-column invariants | | `get_dynamic_runtime_args == yes` is not claimed on any row; `Op-owned tensors?` is blank on a `descriptor` row | ✓ |
  | Pointer patching (not on the recipe's list, checked anyway) | `PD Op (pointer-patching)`, `Smuggled pointer = no` | all four buffer addresses are `Buffer*` objects in `emplace_runtime_args` (RM `:186-188`, `:205-207`; tile `:95-97`, `:116`); no `->address()` anywhere in the directory. The RM factory pushes a literal `0u` instead of the `Buffer*` on idle cores (`:180`, `:199`) | ✓ |

  **`TensorParameter relaxation` — for the readiness-sheet owner.** Both rows read `Padded Shape + Alignment`. This value is outside the vocabulary the audit accepts. The analysis doc (§1) says that once it lands the cell should read `legality (see analysis)`. The audit did not gate on it, per the user's direction. See the relaxation section below for the validity checks.

- **Device 2.0 (every kernel used):** **GREEN** — no violations, so there is no violation table. Scope of the incompleteness: none.

  | Kernel | Includes (all under `tt_metal/hw/inc/api/`) | Idioms | Result |
  |---|---|---|---|
  | RM reader | `api/dataflow/dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `api/tensor/noc_traits.h` | `Noc`, `DataflowBuffer`, `TensorAccessor`; `noc.async_read(s, dfb, size, {…}, {…})` (`:35-40`) | ✓ |
  | RM writer | same four | `noc.async_write(dfb, s, …)` (`:32-37`) | ✓ |
  | Tile reader | same four plus `api/core_local_mem.h` | `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`; `s0.get_noc_addr(...)` (`:76`) is the accessor method; `dfb.get_write_ptr()` (`:39`, `:69`) is a DFB method; local volatile L1 byte copies (`:40`, `:87-91`) | ✓ (user-ruled, see below) |
  | Borrowed unary writer | same four | `get_local_cb_interface(cb_id_out).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:27`) is a sanctioned free function | ✓ |

  **Decided, not a question:** the local volatile L1 byte copies at `reader_unary_reshape_interleaved.cpp:40` and `:87-91` count as Device 2.0 compliant. These accesses touch this core's own L1; every NoC transfer in the kernel goes through `noc.async_read` with the accessor or a `CoreLocalMem` handle. User ruling of 2026-10-22.

  **Exact check.** No kernel includes a ttnn-side header; the closure of ttnn-side files is empty. A fresh-cache run of the op's tests (below) confirms it from the compiler's own dependency lists: 12 objects, 130 distinct paths, and the only `ttnn/` paths are the four kernel sources themselves.

- **Feature compatibility:** all three Appendix A entries are N/A. Sweep covered the op directory and the borrowed writer. The patterns were positive-controlled on `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (GlobalCircularBuffer) and `llama_reduce_scatter_device_operation.cpp` (GlobalSemaphore); both fire there.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `global_circular_buffer`, `remote_cb` or `global_cb`; no `CBDescriptor` sets `.global_circular_buffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress` or `cb_descriptor_from_sharded_tensor`; the CBDescriptors (`reshape_rm_program_factory.cpp:77-85`, `reshape_tile_program_factory.cpp:50-58`, `:73-81`) set only size, cores and one format descriptor |
  | GlobalSemaphore | N/A | no semaphore of any kind in the op or the borrowed writer |

- **CB endpoints (GATE-free):** every CB is legal or carries a port-time disposition; nothing blocks. There are no semaphores, so there is no hidden semaphore-gated second writer, and no kernel touches a CB through a raw cursor-mutating call.

  | CB (index) | Config | Touchers on a node | Disposition |
  |---|---|---|---|
  | RM `src0` (0) | every RM call | RM reader locked PRODUCER (`reserve_back`/`push_back`, `:31`, `:45`); RM writer locked CONSUMER (`wait_front`/`pop_front`, `:28`, `:42`). Present on every node of the full compute grid, idle nodes included (`reshape_rm_program_factory.cpp:58-59`, `:79`) | **legal 1P+1C** |
  | Tile `in0` (0) | every tile call | tile reader locked PRODUCER (`reserve_back(1)` `:57`, `push_back(1)` `:120`); unary writer locked CONSUMER (`wait_front`/`pop_front`, `writer_unary_interleaved_start_id.cpp:51`, `:54`). Node (0,0) only | **legal 1P+1C** |
  | Tile scratch (1) | `alignment > FACE_WIDTH * element_size` (`reshape_tile_program_factory.cpp:71`): Blackhole + BFLOAT16 + input in DRAM | tile reader only: `dfb_intermed.get_write_ptr()` (`:39`) is a role-free raw peek; the NoC read into the scratch and the byte copy out of it both belong to that same kernel (`:79-91`) | **self-loop** (reader bound PRODUCER and CONSUMER); **conditional DFB** |
  | Tile scratch (1) | every other config: Wormhole (any dtype, either buffer), Blackhole in L1, Blackhole FP32 | CB not allocated by the factory; the kernel still constructs `DataflowBuffer dfb_intermed(1)` (`:37-38`) but never calls a method on it | **absent**; kernel handle must be `#ifdef`-gated (see Heads-ups) |

  Sizes the porter derives from the legacy CBs (the auditor's conclusions): RM `entry_size = new_stick_size`, `num_entries = total_size / page_size = num_pages * max_page_size / new_stick_size` (an exact integer, because `reshape_rm_program_factory.cpp:47-52` enforces `max % min == 0`); tile `in0` `entry_size = single_tile_size`, `num_entries = 2`; tile scratch `entry_size = alignment`, `num_entries = 1`.

  The scratch CB exists in exactly one family of configs, from the HAL alignments (`NOC_DRAM_READ_ALIGNMENT_BYTES` is 32 on WH and 64 on BH; `NOC_L1_*_ALIGNMENT_BYTES` is 16 on both; `noc_parameters.h`):

  | Arch | Input buffer | dtype | `alignment` | `16 * element_size` | Scratch CB 1 |
  |---|---|---|---|---|---|
  | WH | DRAM | BF16 / FP32 | 32 | 32 / 64 | absent |
  | WH | L1 | BF16 / FP32 | 16 | 32 / 64 | absent |
  | BH | DRAM | BF16 | 64 | 32 | **present** |
  | BH | DRAM | FP32 | 64 | 64 | absent |
  | BH | L1 | BF16 / FP32 | 16 | 32 / 64 | absent |

  The host condition (`reshape_tile_program_factory.cpp:71`) and the kernel's `MISALIGNED` (`reader_unary_reshape_interleaved.cpp:36`) are the same inequality, because `FACE_WIDTH` is 16 and the kernel's `SUBTILE_LINE_BYTES` is `16 * ELEMENT_SIZE`.

- **Offset base pointers:** **GREEN.** Every address RTA was resolved to its host expression; none folds an offset into the base. The op is not in the dated triage doc, and the scan above stands on its own.

  | Factory | Kernel and role | Host expression | Fold |
  |---|---|---|---|
  | RM | RM reader RTA 0 `src_addr` | `src0_buffer` as `Buffer*` (`reshape_rm_program_factory.cpp:188`); `0u` on idle cores (`:180`) | none; the stick offset travels separately as RTA 4 `start_id` (`curr_sticks_read`, a page id) |
  | RM | RM writer RTA 0 `dst_addr` | `dst_buffer` as `Buffer*` (`:207`); `0u` on idle cores (`:199`) | none; offset is RTA 4 (`curr_sticks_write`) |
  | Tile | tile reader RTA 0 `src0_addr` | `src0_buffer` as `Buffer*` (`reshape_tile_program_factory.cpp:97`) | none |
  | Tile | unary writer RTA 0 `dst_addr` | `dst_buffer` as `Buffer*` (`:116`); `start_id` is the separate literal `0u` | none |

  Type 3 (`address_offset`) and Type 4 (`narrow`) do not occur.

- **TensorAccessor 3rd argument:** **N/A.** No accessor in the op passes a 3rd argument: `reader_unary_reshape_interleaved.cpp:42`, `reader_unary_reshape_stick_layout_interleaved_multi_core.cpp:23`, `writer_unary_reshape_stick_layout_interleaved_multi_core.cpp:21` and the borrowed writer `:42` are all two-argument. The tile reader's `.offset_bytes = intra_tile_offset` (`:85`, `:98`, `:107`) is the in-page offset argument of the Device 2.0 `async_read`, not an accessor page-size override. The op is not in the 3rd-arg triage doc.

- **TensorParameter relaxation (sheet cell → analysis doc):** **CONFIRMED.** The doc `analyses/relaxations/data_movement_reshape_on_device.md` on `akertesz/op-porting-recipe` exists and its verdict is `legality`. The audit validated blob `a437b248cfad`. The copy on that branch is blob `5c6b33c12e64`, which differs on line 13 only (two links became relative). All six validity checks were run against this tree:

  | # | Check | Result | Evidence |
  |---|---|---|---|
  | 1 | The cache key pins the input exactly as the declaration matches it | ✓ | `device/reshape_op.cpp:125-126` keys the input with `hash_tensorspec_with_relaxation(spec, {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true})`. In `tensor_spec_relaxations.cpp` (`:109-112`, `:155-201`) that pins dtype, page config, memory config and padded shape, and frees logical shape and Alignment. `tensorspecs_match_with_relaxation` (`:203-246`) compares the same terms. The page config term carries the Tile: `PageConfig` hashes its variant, `TilePageConfig` is a reflectable aggregate holding `Tile`, and `Tile::attribute_values()` is tile_shape, face_shape, num_faces. The rest of the key is `logical_output_shape`, `padded_output_shape`, `output_mem_config` and `program_factory.index()` (`:121-124`). No pinned term can diverge within one entry. |
  | 2 | The output spec is a pure function of the key | ✓ | `compute_output_specs` (`:92-103`) uses the three hashed attributes plus the input's dtype and page config (pinned). `ReshapeOnDeviceInputs` holds only `input_tensor` (`reshape_device_operation_types.hpp:17-19`): no preallocated output. |
  | 3 | Neither factory nor any kernel reads the input's logical shape or Alignment | ✓ | No `logical_shape`, `logical_volume` or `get_alignment` in `device/*.cpp` or any of the four kernels. The factories read `padded_shape()`, `element_size()`, `dtype()`, `physical_volume()` and the buffer type. The tile factory's `alignment` is `hal::get_dram_alignment()` or `hal::get_l1_alignment()` by buffer type (`reshape_tile_program_factory.cpp:60-61`), passed to the reader as a CTA. |
  | 4 | Still the base concept | ✓ | no `override_runtime_arguments`, no `get_dynamic_runtime_args` (greps above). Every per-core runtime arg is baked at the miss from padded shapes. This is what rules out `dynamic_tensor_shape`. |
  | 5 | Validation reads only pinned terms and runs on cache hits | ✓ | `validate_on_program_cache_miss` (`device/reshape_op.cpp:29-90`) reads storage, allocation, dtype, layout, both memory layouts, the tile (`tensor_spec().tile()`), the input's physical volume, the output's padded volume and the padded widths. Nothing there reads logical shape or Alignment. The op has no `validate_on_program_cache_hit`, and `handle_mesh_adapter_cache_hit` (`ttnn/api/ttnn/device_operation.hpp:374-378`) calls the miss validator on every hit. |
  | 6 | Both factories are interleaved-only and chosen by the input's layout | ✓ | `TT_FATAL` on INTERLEAVED for the input and the output (`reshape_op.cpp:43-48`); `select_program_factory` switches on `input.layout()` (`:21-27`); `program_factory.index()` is hashed (`:124`). |

  Further cross-checks, beyond the six:
  - The doc's analysis base `d1a10279368` is an ancestor of HEAD. Between it and HEAD, the op directory, `tensor_spec_relaxations.cpp` / `.hpp` and `page_config.hpp` are unchanged. Two framework files did change: `ttnn/api/ttnn/device_operation.hpp` (cache-key construction refactor) and `ttnn/api/ttnn/mesh_device_operation_adapter.hpp` (#65470, an alias signature in the cache key). Neither touches tensor-spec matching or validation.
  - For an op with a custom hash, the cache key's collision-resolution part holds only the op-type prefix plus the alias signature, which is empty for a single-input op (`mesh_device_operation_adapter.hpp:1788-1816`). The custom hash alone therefore decides sharing, as it does today. The relaxed pair in #63798's test keeps sharing one entry after the port.
  - Commits cited in the doc: #64663 (`e8b68619482`) and #63798 (`fa826ca2a9a`) are in the tree. `76b17270e9da`, `77f124a3c03c` and `89abbab50e4a` are commits of #63798 itself (squash-merged), confirmed from the PR's commit list. History only, not a gate.
  - The doc's factory line citations (§3) agree with the current files.

  **Declaration, transcribed from the doc (§2):** on the **input** `TensorParameter` of both factories, `.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`. On the **output** `TensorParameter`, nothing (strict). Never `dynamic_tensor_shape`. There is no conditional.

## Verification evidence

- **Silicon, this tree** (Wormhole n150, fresh JIT cache, one pytest process per file): **32 passed, 0 failed.**
  - `tests/ttnn/unit_tests/base_functionality/test_reshape.py -k "reshape_on_device or on_device"`: 17 passed. This includes `test_reshape_on_device_tile_same_padded_shape_different_alignment`, #63798's test for the TILE regime (two inputs with the same padded shape and different logical shape and Alignment share one cache entry, and the cache-hit output equals a freshly built one).
  - `tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py -k reshape_on_device`: 2 passed.
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_reshape.py`: 13 passed (tile and row-major, BF16 and FP32, DRAM and L1, plus the padding-mismatch rejection).
  - Build freshness: `build/lib/_ttnncpp.so` (07:53) postdates the last HEAD move (07:21), and no host source is newer than it.
  - A first attempt that listed all 32 ids in one `@file` run executed nothing. Two of the files share the basename `test_reshape.py` and pytest could not resolve the tt_eager ids. It is not a result; run the files separately.
- **Exact header sweep:** the union of `*.o.dephash` for the four kernels (12 objects, 130 distinct paths). The `ttnn/` paths are the four kernel sources only, so no ttnn-side header is included. The filter is positive-controlled by those four hits.
- **Not covered here:** Blackhole; the scratch-CB configuration (BH + BFLOAT16 + DRAM input), which the n150 never builds, and with it the volatile byte-copy path (`reader_unary_reshape_interleaved.cpp:74-92`) that only that configuration runs; the wider fold and split test families.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): RM input — Case 1 (`TensorAccessor`, RM reader); RM output — Case 1 (RM writer); tile input — Case 1 (tile reader); tile output — Case 1 (borrowed writer, bound as `tensor::dst` on its fork). None is clean-borrowed-DFB and none is Case 2. All four reach the kernels today as `Buffer*` slot-0 RTAs (the framework patches them on cache hits), and each kernel feeds the value to a `TensorAccessor`.
- **TensorParameter relaxation:** `legality (see analysis)`, per the user's reading of the sheet cell. Input slot of both factories: `{.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`. Output slot: none. No conditional.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** RM `src0` legal 1P+1C; tile `in0` legal 1P+1C; tile scratch CB 1 conditional DFB with a self-loop on the tile reader (Blackhole + BFLOAT16 + DRAM input), absent otherwise.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer and no multi-reader shape.
- **Cross-op / shared kernels:** the tile writer is borrowed from `eltwise/unary` and its `_metal2` fork already exists beside it (rung 1: bind the fork, adopt its names). The op's three own kernels have no other binder in the repo, so they are converted in place. Details under Team-only.
- **RTA varargs:** none. Every kernel reads each RTA at a fixed index as a distinct field, and every CTA at a constant index.
- **Hard-coded buffer numbers (anything else):** `reader_unary_reshape_interleaved.cpp:34` (`dfb_id_in0 = 0`) and `:37` (`dfb_id_intermed = 1`); `reader_unary_reshape_stick_layout_interleaved_multi_core.cpp:21` (`tt::CBIndex::c_0`); the RM writer takes its CB index as CTA 0 (`writer_unary_reshape_stick_layout_interleaved_multi_core.cpp:17`, fed by `reshape_rm_program_factory.cpp:92`). All become binding tokens. The op has no compute kernel, so no `unpack_*` table indexes a CB number.
- **Unguarded scratch handle (anything else):** the tile reader constructs `DataflowBuffer dfb_intermed(dfb_id_intermed)` and reads it in a ternary (`:37-39`) in every config, while the factory allocates the scratch CB in only one family of configs. A token for an unbound DFB does not exist, so these lines must be gated at the preprocessor, with the define emitted from the host under the same condition as the binding (`port_patterns.md`, *Conditional / optional resource bindings*, including *Promote a CTA gate to a define*). The later `if (MISALIGNED)` (`:74`) touches only locals and stays.
- **Configuration the n150 cannot run (anything else):** the scratch CB and the volatile byte-copy loop execute only on Blackhole with BFLOAT16 input in DRAM. Local verification on this Wormhole machine leaves that path unexercised. A BH run, or a temporary scratch-only switch that forces the variant on WH, is the way to cover it (whether the forced variant is functionally valid on WH was not tried here).
- **RM idle cores (anything else):** the legacy loop sets run-args on every core of the grid (`reshape_rm_program_factory.cpp:116-216`), and gives idle cores a literal `0u` address plus zero counts. After the port the tensor binding delivers the real base to every node; the zero-trip loops never touch it. Keep the per-core arg coverage and the full-grid kernel and DFB node sets.
- **Kernels are already part-modernized (anything else):** all of them use `Noc`, `DataflowBuffer`, `TensorAccessor` and, in one, `CoreLocalMem`. The port is a binding-layer change, not an idiom rewrite.
- **Device-operation class stays untouched (anything else):** `compute_program_hash`, validation and `compute_output_specs` are not port scope. There is no pybound `create_descriptor` to delete, no pybind-only factory parameter, and `program_factory_t` already exists (`reshape_op.hpp:20`), so none of the three forced device-op-class edits applies.
- **Hit-path spec check is off by default (anything else):** `validate_program_args` defaults to `false` (`ttnn/api/ttnn/config.hpp:31`) and gates only the cache-hit re-checks (`mesh_device_operation_adapter.hpp:54-57`); misses always validate. The input relaxation is only proven live with `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`. With the knob off the hit-path check is skipped, so `test_reshape_on_device_tile_same_padded_shape_different_alignment` passing says nothing about whether the input declares the relaxation (read from the code; the ported op does not exist yet to try it).
- **Open PRs that touch the op's host code (anything else):** #65944 (`device/reshape_rm_program_factory.cpp`, six lines, `Buffer*` → `MeshTensor`); #63418 (`device/reshape_op.cpp`, `reshape.cpp`; swaps the layout helper behind `compute_output_specs` and `manual_insertion` for `internal::tensor_layout_from_padded_shape`, on a base that predates #51656); #63553 (`device/reshape_op.cpp`; its hunk keys with `{.match_padded_shape_only = true}` alone and an old header path, and predates #63798's merge); #66313 and #66844 (`reshape_nanobind.cpp` docstring); #65284 (`reshape.cpp` view path). If any lands before the port starts, re-run the six checks and re-read the factories.

## Team-only

- **Out-of-directory coupling & donor shape:** op-level roll-up **✓ clean**. No kernel `#include`s anything outside `tt_metal/hw/inc/api/`.

  | Op kernel | Donor file | Donor class | Shape | Status |
  |---|---|---|---|---|
  | all four | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | `tt_metal/*` framework (class 1) | n/a | ✓ no concern |
  | tile reader | `api/core_local_mem.h` | `tt_metal/*` framework (class 1) | n/a | ✓ no concern |

  No per-call detail is needed: every row is ✓.

  **Borrowed kernel files (file-path instantiation):**

  | Kernel file | Owner | Other binders (a **sunset list**, not authorization to convert the file in place) | `_metal2` fork beside it |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (tile writer) | `eltwise/unary` | still on the legacy copy, in `ttnn/cpp` factories on this tree: `eltwise/unary_backward/device/unary_backward_program_factory.cpp:343`, `examples/example/device/multi_core_program_factory.cpp:81`, `examples/example/device/single_core_program_factory.cpp:83`. Test and model binders of the legacy path: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp`, `tests/ttnn/unit_tests/kernel_lib/chain_test_lib.py`, `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`, `tests/ttnn/unit_tests/operations/eltwise/test_clamped_silu_glu_sfpu.py`, `test_moe_gate_sqrt_softplus_sfpu.py`, `test_situ_glu_sfpu.py`, `models/tt_dit/tests/models/minimax_h3/test_sol_prepass_minimax_h3.py`. Already on the fork: 26 factory files under `ttnn/cpp`. Sunset tracking: #52228 (open). | **yes**: `writer_unary_interleaved_start_id_metal2.cpp`, same directory (rung 1) |

  Same-named private copies were discarded from the census (`matmul/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` and the slice copy are other files). The op's own kernels: a filename grep across the repository (excluding `experimental/quasar/`) finds no binder outside this op's factories, so there is no lent kernel and no fork to create. Install coverage: `data_movement/CMakeLists.txt:43` globs `reshape_on_device/device/kernels/*.cpp` recursively, and the fork sits under `eltwise/unary`'s `device/kernels/*.cpp` glob; nothing needs a build-file edit.

- **Relaxation candidates** (from the custom hash): none new. The hash equals the declaration by construction, so there is no further looseness to mine.

- **TTNN factory analysis:**
  - Current concept: `descriptor`, `create_descriptor(const ReshapeOnDeviceParams&, const ReshapeOnDeviceInputs&, Tensor& output_tensor)` in both factories. Execution model: `SPMD`, one program across the mesh. Op-owned tensors: none. Ported-from shape: 1:1.
  - Custom hash: `device/reshape_op.cpp:111-127`, no backdoor. It stays untouched. It is not looser than the declaration (equal by construction), which is the containment test in `ttnn_factory.md`.
  - `validate_on_program_cache_hit`: none, so the miss validator runs on hits.
  - Pybind: none to remove (`reshape_nanobind.cpp:68` is the user-facing function only).
  - Target: `ProgramSpecFactoryConcept` (the sheet's `Porting Target`), consistent with the absence of `override_runtime_arguments`.
  - Project status: issue #15721 ("Remove reshape_on_device operation") is open; the port issue #65484 is open and assigned. Neither is a gate.

- **Who reaches the kernels (for the porter's verification scope):**
  - C++ callers of `ttnn::reshape_on_device`: `fold/fold.cpp:80`, `:93`; `split/split.cpp:43`, `:52` (the last two pass explicit padded shapes). `reshape_view` includes the header but only calls `prim::reshape_view`. The Python entry is `ttnn.reshape_on_device`.
  - Tests that name it or call fold or split: the three files above (32 tests, baseline green); `test_fold_op.py` in full; nightly `test_universal_input_tm_fold.py`, `test_split.py`, `test_universal_input_tm_split.py`, `pool/test_grid_sample.py`; `test_zero_divisor_guards.py`, `test_maxpool2d_shard_grid.py`, `test_comparison_mode_context.py`; the sweeps `tests/sweep_framework/sweeps/data_movement/split/split_pytorch2.py` and `.../model_traced/{split,fold}_model_traced.py`; `tests/tt_eager/python_api_testing/sweep_tests/tt_lib_ops.py`; `tests/ttnn/profiling/ops_for_profiling.py`. There are no gtests. Eleven files under `models/experimental/` call `ttnn.reshape_on_device` (legacy model code). Which of these reach the reshape kernels depends on the path each takes; enumerate by a fresh-cache kernel build rather than by name.

## Misc anomalies  *(team-only, non-gating)*

- `device/reshape_op.cpp:51-53` — stale comment: "the tile is absent from compute_program_hash". Since #63798 the key hashes the input's page config, which carries the Tile. The 32×32 rejection under it is still right. Suggested replacement, not applied:
  ```
  // ReshapeTileProgramFactory sizes its CBs and tile counts from the architectural 32x32 constants.
  // A non-standard tile would compile a mis-sized program.
  // Reject it here.
  // The cache key already separates tile shapes.
  // It hashes the input's page config, and a TILE page config carries its Tile.
  ```
- `device/reshape_op.cpp:81-86` and `device/reshape_rm_program_factory.cpp:47-52` — the RM path has no input/output padded-volume check, unlike the tile path (`:72-80`, added in #52828). If a caller passes a `padded_output_shape` whose volume differs from the input's padded volume, the reader and writer stick counts disagree. Python callers are guarded by `infer_dims_for_reshape` (`reshape_view/reshape_common.cpp:58`, logical volume); the direct C++ overload `ttnn::reshape_on_device(input, logical, padded, ...)` has no such check. Reachability beyond that was not probed here.
- `device/reshape_tile_program_factory.cpp:42` and `reshape.cpp:28-33` — `TT_ASSERT` is compiled out of release builds. In `manual_insertion` the logical-volume check at `reshape.cpp:29-33` is therefore a no-op in release. The tile factory's assert goes away with its `dst_buffer` pointer in the port.
- `reshape_nanobind.cpp:37` — the docstring says the X dimension of input and output must be the same size. `reshape.cpp:60-65` routes equal-X to the view path, so the device op runs only when X differs. The open #66313 rewrites this docstring (and documents the view-path rule); #66844 fixes only its example.
- `reshape.cpp:59` — the comment says "Will do a tensor copy" while the code returns a view (`:64`).
- `data_movement/sources.cmake:269-272` — `reshape_view/reshape_nanobind.cpp` and `reshape_on_device/reshape_nanobind.cpp` are each listed twice, back to back.

## Per-DeviceOperation attribution

One device operation. Findings differ by factory only in these respects:

| Aspect | `ReshapeRMProgramFactory` | `ReshapeTileProgramFactory` |
|---|---|---|
| Nodes | full compute grid (`reshape_rm_program_factory.cpp:54-59`) | core (0,0) (`reshape_tile_program_factory.cpp:27`) |
| Kernels | own reader and own writer | own reader and the borrowed unary writer (fork exists) |
| DFBs | `src0` (1P+1C) | `in0` (1P+1C); `scratch` (conditional, self-loop) |
| Address RTAs | reader and writer slot 0 (`0u` on idle cores) | reader slot 0, writer slot 0 |
| Relaxation | input both flags, output strict | same |
| Sheet differences | `Uses llama kernels? = no` | `Uses llama kernels? = yes` |

## Recipe notes

- **Relaxation cell outside the vocabulary, with its analysis doc present.** *TensorParameter relaxations* says "anything else → GATE → ops team", and "If the cell says `see analysis` and no such doc exists, STOP". Neither covers a cell like `Padded Shape + Alignment` whose doc exists and whose own §1 says the cell should read `legality (see analysis)`. A strict reading REDs this op; I followed the user's direction instead. It would help if the recipe said whether a present, valid analysis doc may clear a stale tag.
- **Device 2.0 gate versus local volatile access.** The gate defers to `device_api_migration_guide.md` for what compliance entails. Its *Safe Local Memory Access* section (`:474-488`) labels a raw `volatile uint32_t*` cast "Legacy API" with `CoreLocalMem` as the new API. The Metal 2.0 whitelist (`cb_dfb_api_whitelist.md:41-44`) shows a `volatile tt_l1_ptr` peek as ✅, and the kernel-side whitelist does not admit a respelling at port time. The audit does not say which wins. The user decided (2026-10-22) that the local pattern is compliant; one sentence in the gate would remove the ambiguity.
- **CB endpoints, "conditional DFB" wording.** The *Dead is per `(CB, config)`* paragraph describes a legacy factory that allocates every CB unconditionally and gates only the kernel-side use. This op is the inverse: the factory already allocates the scratch CB conditionally (`reshape_tile_program_factory.cpp:71`), and the kernel constructs its DFB handle unconditionally (`reader_unary_reshape_interleaved.cpp:37-38`). The disposition (conditional DFB) still fits, but the porter's work is the kernel-side `#ifdef`, not new host structure.
- **Provenance line on a rebased doc branch.** The recipe's `git log -1 --format='%h %cs %s'` prints the local rebase commit and its committer date (`547b694ae6f8 2026-11-23`), not the upstream commit `c41f7f4ef0a` it was rebased from. I recorded both plus the identical tree hash. A tree-hash field in the provenance command would pin the version exactly.
- **Readiness-sheet fetch step.** `ttnn_op_porting_readiness.md` step 3 says the saved file is JSON with a base64 `content` field and gives a `jq … | base64 -d` decode. In this session the connector saved a plain CSV directly, and that step would have failed on it. Checking the file's first bytes before decoding avoids the surprise.
