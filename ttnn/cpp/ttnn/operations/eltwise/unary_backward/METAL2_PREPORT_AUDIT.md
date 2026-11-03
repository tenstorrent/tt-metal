# Metal 2.0 Audit Findings — `eltwise/unary_backward`

- **`UnaryBackwardDeviceOperation`** (`device/unary_backward_device_operation.hpp:23`). One factory: `program_factory_t = std::variant<UnaryBackwardProgramFactory>` (`:28`). There is no `select_program_factory`.
  - `UnaryBackwardProgramFactory` (`device/unary_backward_program_factory.hpp:17-20`). `create_descriptor` is defined at `device/unary_backward_program_factory.cpp:26-481`.
- **One factory, four gradients.** The factory serves every `UnaryBackwardOpType` (`device/unary_backward_op_types.hpp:17-22`). The op type only picks the compute kernel (`get_kernel_spec`, `device/unary_backward_op_utils.cpp:13-64`). Every verdict below holds for all four alike.

  | Op type | Public entry point (`unary_backward.cpp`) | Compute kernel (`device/kernels/compute/`) |
  |---|---|---|
  | `SIGMOID_BW` | `sigmoid_bw` `:535`, op type `:547` | `eltwise_bw_sigmoid.cpp`. DEST is forced to fp32 (`unary_backward_op_utils.cpp:25`). |
  | `TANH_BW` | `tanh_bw` `:360`, op type `:370` | `eltwise_bw_tanh.cpp` |
  | `GELU_BW` | `gelu_bw` `:1636` with `GeluVariant::ACCURATE`, op type `:1650-1651` | `eltwise_bw_gelu_poly.cpp` |
  | `GELU_TANH_BW` | `gelu_bw` with `GeluVariant::TANH` (same lines) | `eltwise_bw_gelu_tanh.cpp`, or `eltwise_bw_gelu_tanh_fp32.cpp` when DEST is fp32 (`unary_backward_op_utils.cpp:49-61`) |

  `GeluVariant::FAST_LUT` is rejected before the device op (`unary_backward.cpp:1643-1644`). Python bindings: `unary_backward_nanobind.cpp:1119` (`gelu_bw`), `:1144` (`tanh_bw`), `:1258` (`sigmoid_bw`).
- **Three program shapes inside the one factory:**

  | Mode | When | Reader | Writer |
  |---|---|---|---|
  | **Addressed** | TILE layout and `can_alias_shards` false (`unary_backward_program_factory.cpp:82-129`) | `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` (borrowed) | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (borrowed) |
  | **Aliased** | TILE or ROW_MAJOR, `can_alias_shards` true | the same reader with `IN0_SHARDED` and `IN1_SHARDED` (`:326-328`) | the same writer with `OUT_SHARDED` (`:345-347`) |
  | **RM blocked** | ROW_MAJOR and not aliased (`rm_blocked`, `:137-138`) | `device/kernels/dataflow/reader_unary_backward_row_major.cpp` (own) | `device/kernels/dataflow/writer_unary_backward_row_major.cpp` (own). Both include `row_major_pages.hpp`. |

  Every file under `device/kernels/` is bound in some configuration. Nothing in the directory is unreferenced.
- **This is a fresh audit.** The previous one is now `METAL2_PREPORT_AUDIT.prev2.md` (blob `e66f838cad7`): GREEN on a working tree, conditional on #63334 and on the uncommitted Device 2.0 migration of the three RM kernel files. Both landed in `be782c33d2e` (#63334, 2026-10-30). Every finding below was re-derived from the code on this branch; none is carried over. The oldest audit, `METAL2_PREPORT_AUDIT.prev.md` (blob `a2c348f115e`), is unchanged.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `f126189dbb6 2026-11-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- This branch carries the recipe tracked. Its `metal_2.0/` tree (`c3e701f18bd`) equals upstream `9425f50ae93` (authored 2026-09-03). The 2026-11-03 date is the rebase's committer date. Audit blob `d0576d6d739`.
- Upstream `akertesz/op-porting-recipe` is two commits ahead, at `c41f7f4ef0a` (2026-10-29; GitHub API and the local ref agree). Per your instruction I followed the in-place file. The drift is listed in *Recipe notes*; it changes no verdict here.

**Code audited** (copy for the tracking tab):

```
anasuya/metal2_port_eltwise_unary_bw @ f126189dbb6 — code identical to main a1ce48e3fe8 (2026-11-03)
(#63334 = be782c33d2e; op directory tree d4e28cdccc5; no tracked edits; outside docs/ the branch has no diff from main)
```

**Readiness sheet:** not fetched. The claude.ai Google Drive connector is not authorized in this session (no `download_file_content` tool), and the CSV export URL returns HTTP 401. The rows used here are the corrected rows you supplied on 2026-11-03. I could not resolve the column names against a live header row; *D* compares them with the recipe's column legend instead.

**Relaxation analysis doc:** `analyses/relaxations/eltwise_unary_backward.md`, untracked, blob `ce949868d1f` (2026-10-29 17:03 UTC). Checked against this branch in *C*.

**Silicon evidence:** one Wormhole n150, run by this audit on 2026-11-03. The host build (`_ttnn.so`, 13:03 UTC) was built after this branch was checked out (12:51), and it carries main tip's new `moreh_bmm` docstring. Kernels were JIT-compiled into a fresh `TT_METAL_CACHE`.

| Run | Cases | Result |
|---|---|---|
| `tests/ttnn/unit_tests/operations/eltwise/test_unary_backward_device_op.py` | 35 | 35 passed |
| `tests/ttnn/nightly/.../backward/test_backward_gelu.py -k row_major_matches_tile` | 16 | 16 passed |
| `tests/ttnn/nightly/.../backward/test_backward_gelu.py -k sharded_matches_interleaved` | 12 | 12 passed |

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/` |
| **Overall** | **GREEN, no conditions.** The audited code is `main` `a1ce48e3fe8`. Brief issued. |
| **DOps / Factories** | `UnaryBackwardDeviceOperation` → `UnaryBackwardProgramFactory`, serving `SIGMOID_BW`, `TANH_BW`, `GELU_BW`, `GELU_TANH_BW` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** 9 bound kernels; 26 ttnn-side files in their include closure (compiler-confirmed). 0 raw NoC calls, 0 address generators, 0 `get_noc_addr`, 0 CB-index FIFO or pointer free functions. The only free functions are the sanctioned `get_tile_size` (×2) and `get_local_cb_interface` (×1). `copy_bytes`' local `volatile` access is compliant by your 2026-10-22 ruling (*A*). |
| *Prereqs* — Cross-op escapes | Ok. `kernel_lib/eltwise` only (`uint32_t cb_id` ✓, `DataflowBuffer` ✓), plus two borrowed in-family kernel files, both with `_metal2` forks. |
| *Feature Support* — overall | GREEN. All three Appendix A entries N/A. |
| *Feature Support* — Variadic-CTA | Ok. None. |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** on your replacement row (analysis-details tab), and on the tracking tab. Primary cross-check clean (*D*). |
| *TTNN Readiness* — Concept (current) | `descriptor`: `create_descriptor` returns `ProgramDescriptor` (`device/unary_backward_program_factory.hpp:19-20`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A. (Your row's `Execution Model` = `SPMD` agrees: one program, no mesh coordinate.) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; the port leaves it intact): `device/unary_backward_device_operation.cpp:238-260` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (0 hits in `unary_backward_nanobind.cpp`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`. Derived per the in-place recipe; it equals your row's `Porting Target`. |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `grad_output`, `input`, `output`: **Case 1** in the addressed and RM-blocked modes, **clean** (borrowed DFB) in the aliased mode. No Case 2. |
| *TTNN Readiness* — TensorParameter relaxation | `none (see analysis)` (your cell). Doc verdict `none`, **CONFIRMED**: all six validity checks pass on this branch (*C*). `.relaxations = {}` on all three slots. |
| *Port work* — TensorAccessor 3rd arg | none. No accessor passes a 3rd argument (6 two-argument constructions). |
| *Port work* — CB endpoints | **1P+1C** on c_0, c_1, c_2 in every mode (borrowed when aliased). **Self-loop** on c_3 (RM reader) and c_4 (RM writer), RM-blocked mode only. No multi-binding, no dead CB. |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). There are no conditions. Both of the previous audit's conditions are on `main`, and this branch's code is `main`.

- **Device 2.0 clears on the code, not on the migration's say-so.** I swept every bound kernel and every ttnn-side header they include. Two independent methods give the same 26 files, and none has a Device 1.0 call (*A*).
- **The cache key now pins every bound tensor's whole `TensorSpec`.** I traced the hash down to `Tile`, `MemoryConfig` and `Alignment`. Every field the Metal 2.0 strict match compares is in the key. The three cache tests that check this pass on silicon (*C*).
- **The relaxation verdict is CONFIRMED: `none`.** All six validity checks pass. Every citation in the doc resolves by content. Some of its text is stale now that the fix has landed (*C*).
- **The sheet reconciles, with two disagreements to settle.** Your replacement row matches the code on every primary column. Two cells rest on the premise that the op no longer pointer-patches, and the code shows it still does (*D*, *Questions* 1). Neither cell gates.

**Portable subset:** the whole op. There is one factory, and the declaration is unconditional.

## Gate detail

- **TTNN factory concept: GREEN.** `Is able to port?` = `yes` on your replacement row. Every primary column matches the code (*D*). The two old rows are phantom rows. Both device operations they name were deleted, and none of their four types exists in the tree. Per your direction, that is the sheet being stale for this op; it is not gated on.
- **Device 2.0: GREEN.** Every kernel the op binds and every donor header is compliant (*A*).

  | File | Line | Call | Status |
  |---|---|---|---|
  | `eltwise/binary/.../reader_binary_interleaved_start_id.cpp` | 50, 57 | `get_tile_size(cb_id)` | sanctioned |
  | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | 27 | `get_local_cb_interface(cb_id).fifo_page_size` | sanctioned |
  | `unary_backward/.../dataflow/row_major_pages.hpp` | 65-66 | `reinterpret_cast<volatile uint8_t*>` in `copy_bytes` | local L1 access; compliant by your 2026-10-22 ruling |

  There are no other free functions, raw NoC calls or address generators in the closure.
- **Feature compatibility: GREEN.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, `global_circular_buffer`, `remote_index` or `global_cb` in the op or its 26 closure files |
  | CBDescriptor `address_offset` (non-zero) | N/A | The `CBDescriptor`s set only `total_size`, `core_ranges`, `format_descriptors` and `buffer` (`unary_backward_program_factory.cpp:244-253`, `:279-287`) |
  | GlobalSemaphore | N/A | No semaphore of any kind. The same grep finds `GlobalSemaphore` in `llama_reduce_scatter` (positive control). |

- **CB endpoints (GATE-free):** 1P+1C on c_0, c_1 and c_2 in every mode. Self-loop on c_3 and c_4 in RM-blocked mode (*B*).
- **Offset base pointers: GREEN.** The op directory has no `->address()` (0 hits). Every tensor reaches a kernel as a bare `Buffer*` RTA (`unary_backward_program_factory.cpp:440`, `:445`, `:459`, `:468`, `:470`), which cannot carry a folded offset. Neither dated triage doc (`2026-07-19_offset_base_pointers.md`, blob `3d2b1be9af9`; `2026-07-06_tensor_accessor_3rd_arg_triage.md`, blob `20ad52f8fd7`) mentions this op or its borrowed kernels. A positive control finds their own entries.
- **TensorAccessor 3rd argument: N/A.** No accessor passes a 3rd argument. The six constructions all take two: `reader_binary_interleaved_start_id.cpp:51`, `:58`; `writer_unary_interleaved_start_id.cpp:42`; `reader_unary_backward_row_major.cpp:55`, `:56`; `writer_unary_backward_row_major.cpp:38`.

## A. Device 2.0 — the kernels and every donor header

**1. Which files.** The factory binds 9 kernel sources: 4 dataflow kernels (the 2 borrowed, the 2 RM) and 5 compute kernels. I built their transitive include closure in two ways.
- **Statically.** I followed every `#include`, in every preprocessor branch, using the JIT's search order (`tt_metal/jit_build/build.cpp:522-532`, Wormhole HAL dirs in `wh_hal.cpp:121-157`). The result is 398 files: 372 under `tt_metal/` and 26 ttnn-side. The 49 names it could not resolve are std, toolchain (`sfpi.h`), generated (`chlkc_descriptors.h`), host-only or Quasar-only headers. None is a ttnn path.
- **From the compiler.** The JIT writes each object's full dependency list to `<obj>.o.dephash`. Over the fresh cache from the silicon runs above, I took the union for every compiled variant of the 9 kernels: 11 / 16 / 8 / 8 variants of the four dataflow kernels and 16 of the compute kernels. The ttnn-side set is the same 26 files.

The two sets are identical:

| Class | Files |
|---|---|
| Op-own (9) | the 5 compute kernels, `operand_reconfig.hpp`, the RM reader, the RM writer, `row_major_pages.hpp` |
| In-family (2) | the two borrowed kernels themselves. They include only `tt_metal` headers. |
| `kernel_lib` (15) | `dest_helpers.hpp`, and in `eltwise/`: `api/chain.hpp`, `core/chain.inl`, `core/op_params.hpp`, `binary/sfpu/{basic.hpp,basic.inl,sfpu.inl}`, `generators/{fill.hpp,fill.inl}`, `unary/{activations,misc,scalar}.{hpp,inl}` |
| `ttnn/kernel/`, `kernel_helper_functions/`, cross-family | none |

**2. The sweep.** I stripped comments and swept for: raw `noc_async_*`, `noc_semaphore_*`, `noc_inline_dw_write*`; `InterleavedAddrGen*`, `ShardedAddrGen*`, `InterleavedPow2AddrGen*`; `get_noc_addr*` and `get_noc_addr_from_bank_id`; the CB-index free functions `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` / `get_write_ptr` / `get_read_ptr`; `get_semaphore`; `volatile` / `tt_l1_ptr` casts; `my_x[` / `my_y[`; `fifo_*_ptr` / `evil_set_*`; the `CircularBuffer` type.

| Scope | Result |
|---|---|
| All 26 files on HEAD | The 3 sanctioned free-function calls and the 2 `copy_bytes` casts in the table above. Nothing else. |
| **Positive control:** the RM trio at `d70d725e238` (the parent of `be782c33d2e`) | 12 raw `noc_async_*` (9 in the header, 1 in the reader, 2 in the writer), 4 `get_noc_addr`, 4 CB-index pointer calls. These are exactly the counts #63334 removed. |

**3. How the kernels drive their buffers.** Every dataflow kernel builds a `Noc` and a `DataflowBuffer` per CB. The RM kernels: reader `:58-61`, writer `:40-42`. The borrowed kernels: reader `:42-44`, writer `:29-30`. Transfers use `TensorAccessor` endpoints with `{.page_id, .offset_bytes}`, and `CoreLocalMem<uint8_t>` local endpoints (`row_major_pages.hpp:90`, `:95-100`, `:122`, `:130-134`). The compute kernels hold one `DataflowBuffer` per CB. `kernel_lib` drives every FIFO through `DataflowBuffer` methods (`kernel_lib/eltwise/core/chain.inl:2623-2644`). No `kernel_lib` file hardcodes a CB number (0 hits for `CBIndex::c_*`, `unpack_src_format[`, `pack_dst_format[`).

**4. History, for the Device 2.0 team.** #49410 (`33b63815b96`, 2026-07-09) migrated the eltwise kernels to `DataflowBuffer`. #57342 (`4b092b3a83a`, 2026-09-24) then created the three RM files with the 12 raw calls. #63334 (`be782c33d2e`, 2026-10-30) migrated them. Nothing in CI caught the regression in between (*Misc anomalies*).

## B. CB endpoints — the census

Each kernel is one `KernelDescriptor` over `all_cores` (`unary_backward_program_factory.cpp:293-301`, `:386-392`). There is no dual-instance work split. Every node sees one reader, one writer and one compute kernel.

| CB | Exists | Addressed | Aliased | RM blocked |
|---|---|---|---|---|
| c_0 (`grad_output`) | always (`:256-260`) | reader `reserve_back`/`push_back` (`reader_binary...:96`, `:108`); compute consumes → **1P+1C** | `borrowed_from` `grad_output`, one shard deep. The reader marks the shard resident with `reserve_back`/`push_back(num_tiles)` (`:46-48`); compute consumes → **1P+1C** | RM reader `reserve_back`/`push_back` plus a `get_write_ptr` peek by the same kernel (`reader_unary_backward_row_major.cpp:75`, `:77`, `:105`); compute consumes → **1P+1C** |
| c_1 (`input`) | always (`:261`) | the same (`:101`, `:112`) | `borrowed_from` `input` (`:53-55`) | the same (`:76`, `:78`, `:106`) |
| c_2 (output) | always (`:262-266`) | compute produces; writer `wait_front`/`pop_front` (`writer_unary...:51`, `:54`) → **1P+1C** | `borrowed_from` the output; writer `wait_front`/`pop_front(num_pages)` (`:35-36`) → **1P+1C** | compute produces; RM writer `wait_front`/`pop_front` plus a `get_read_ptr` peek (`writer_unary_backward_row_major.cpp:55-56`, `:70`) → **1P+1C** |
| c_3 (reader scratch) | `rm_blocked` only (`:272-289`) | — | — | RM reader only, through `aligned_scratch` → `get_write_ptr()` peek, no FIFO op (`reader...:61-63`, `row_major_pages.hpp:140-143`) → **self-loop** |
| c_4 (writer scratch) | `rm_blocked` only | — | — | RM writer only, the same way (`writer...:42-43`) → **self-loop** |

**c_3 and c_4, checked rather than assumed:**
1. **Who touches them.** The only kernels naming c_3 or c_4 are the RM reader (`:49`) and the RM writer (`:34`). The compute kernels name only c_0, c_1 and c_2, and `kernel_lib` names none. The borrowed kernels are not bound in RM-blocked mode.
2. **How.** Each scratch DFB is reached only through `get_write_ptr()`. There is no `reserve_back`, `push_back`, `wait_front`, `pop_front` or `evil_set_*`. A peek locks no role. That is one role-free toucher per node, the recipe's single-ended / sync-free row.
3. **Where.** Both are created only when `rm_blocked`, on `all_cores`, where the RM kernels run. They do not exist in the other modes. This is therefore not a conditional-DFB case. The port keeps the existing host condition.
4. **Legal per the validator.** `tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:605-612` rejects a data-movement self-loop only on Gen2. `:619-631` requires the producer set to equal the consumer set, which one kernel satisfies.
5. **Quasar debt, not Gen1 work.** Gen2 rejects these two data-movement self-loops. The validator message points to a scratchpad or `LocalTensorAccessor` instead.

Metadata reads are not extra touchers. The borrowed reader's `get_tile_size(c_0/c_1)` and the borrowed writer's `get_local_cb_interface(c_2).fifo_page_size` are made by kernels that already hold those CBs.

## C. Relaxation analysis — validity checks and citations

**Doc checked:** blob `ce949868d1f`.

| # | Check | What this branch shows | Result |
|---|---|---|---|
| 1 | The key pins every bound tensor's whole `TensorSpec`. | See the trace below. | **PASS** (it failed on `main` before `be782c33d2e`) |
| 2 | The factory never reads `logical_shape`. | 0 hits in `unary_backward_program_factory.cpp`. Positive control: 8 lines in the device op. | **PASS** |
| 3 | Base `ProgramSpecFactoryConcept`: no `override_runtime_arguments`, no `get_dynamic_runtime_args`. | 0 hits for either under `device/` | **PASS** |
| 4 | The RM-blocked path bakes padded-shape geometry into per-core args. | `row_elements = input.padded_shape()[-1]` (`:156`) and `page_elements` from `buffer()->page_size()` (`:162`), inside `if (rm_blocked)` (`:155-186`). Baked into the RTAs at `:431-463`. | **PASS** |
| 5 | Validation rejects any tile other than 32×32 on a TILE operand. | `validate_operand`, `unary_backward_device_operation.cpp:91-98`. ROW_MAJOR operands return early at `:61-69`. | **PASS** |
| 6 | Exactly one factory. | `unary_backward_device_operation.hpp:28`. 0 `select_program_factory` hits. | **PASS** |

**Check 1, traced.**
1. **The key.** `hash_operation<UnaryBackwardDeviceOperation>(args, grad_output.tensor_spec(), input.tensor_spec())` (`unary_backward_device_operation.cpp:247-248`). A preallocated output adds its own `tensor_spec()` (`:255-257`). `hash_operation` also folds in the type hash (`ttnn/api/ttnn/operation.hpp:24-25`).
2. **The computed output is pinned through its inputs.** Its spec is built from the input's logical and padded shapes and layout, plus `args.output_dtype` and `args.output_memory_config` (`:205-228`). All of them are in the key. Its alignment is derived from the two shapes. Its tile is the default 32×32, because a TILE input with any other tile is rejected.
3. **What the strict match compares.** With no relaxation, `tensorspecs_match_with_relaxation` is `a == b` (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:170-171`). That is `TensorSpec::operator== = default` over five members (`tensor_spec.hpp:40`, `:118-123`). Three of them are pure functions of the other two.
4. **The hash covers at least every field that equality compares**, at every level:

   | Type | Hash walks | Equality compares |
   |---|---|---|
   | `TensorSpec` | `logical_shape_`, `tensor_layout_` (`tensor_spec.hpp:110-111`) | those two, plus three derived caches |
   | `TensorLayout` (now a pImpl) | `dtype_`, `page_config_`, `memory_config_`, `alignment_` (`tensor_layout_impl.hpp:60-62`) | the same four, `= default` (`tensor_layout_impl.hpp:25`, forwarded at `tensor_layout.cpp:498`) |
   | `PageConfig` | active variant index + value (`tt_stl/tt_stl/reflection.hpp:1349-1354`). `RowMajorPageConfig` is empty; `TilePageConfig` holds `Tile tile` (`page_config.hpp:19-27`). Aggregates are walked field by field, and an unhashable type is a `static_assert` (`reflection.hpp:1479-1487`). | `= default` |
   | `Tile` | `tile_shape`, `face_shape`, `num_faces` (`tile.hpp:86-87`) | `tile_shape`, `face_shape` (`tile.cpp:188-190`) |
   | `MemoryConfig` | 7 attributes, including both shard specs and the creation flag (`memory_config.hpp:55-71`) | buffer type, memory layout, the two allocation flags, and the authoritative shard spec(s) (`memory_config.cpp:115-153`) |

5. **Conclusion.** Two calls that share an entry have equal `TensorSpec`s on every bound slot. The hash is tighter than equality in places, which costs only extra cache entries.
6. **On silicon.** `test_sigmoid_bw_program_cache_same_padded_shape` (`test_unary_backward_device_op.py:356`) runs 96×96, 40×40 padded to 96×96, then 96×96, and gets 2 entries. `test_unary_backward_program_cache_keys_each_tensor` (`:394`) gets 2 entries when only grad_output's dtype changes, and 2 when only the preallocated output's memory config changes. `test_sigmoid_bw_rejects_mismatched_shapes_on_warm_cache` (`:341`) shows that validation runs on a hit. All three passed today.

**Verdict: CONFIRMED — `none`.** The declaration is `.relaxations = {}` on `grad_output`, `input` and `output`, unconditionally.

**The doc's citations, re-resolved by content.** The doc has no `file:line` citations at all. Every code citation is a file plus an identifier. I resolved each one at HEAD by content.

| Doc cites | At HEAD | Holds? |
|---|---|---|
| `compute_program_hash`; `logical_shape` "matches eight times, none in the key" | key `:238-260`. `logical_shape` is on 8 lines (`:144-179`, `:221`, `:226`), all in validation and `compute_output_specs` | yes |
| `validate_operand` rejects non-32×32 tiles | `:91-98` | yes |
| the `rm_blocked` branch's `row_elements` and `page_size()` | `unary_backward_program_factory.cpp:155-186` | yes |
| `program_factory_t`, no `select_program_factory` | `.hpp:28` | yes |
| hit-validation fallback in `device_operation.hpp` (`handle_mesh_adapter_cache_hit`) and `mesh_device_operation_adapter.hpp`; the `HasValidateOnProgramCacheHit` comment | `device_operation.hpp:288-291`; `mesh_device_operation_adapter.hpp:228-233`; quoted text verbatim at `operation_concepts.hpp:171` | yes |
| #37633 (2026-02-12) | `85d0dea30b5`, 2026-02-12 | yes |
| `TensorSpec::operator== = default`, five members; `attribute_values()` | `tensor_spec.hpp:40`, `:110-111`, `:118-123` | yes |
| `hash_tensorspec_with_relaxation` asserts the same equivalence | `tensor_spec_relaxations.cpp:121-134` | yes |
| `TensorLayout`'s attributes `(dtype_, page_config_, memory_config_, alignment_)` | moved behind a pImpl into `TensorLayoutImpl` (`tensor_layout_impl.hpp:60-70`); same four | yes |
| `initialize_alignment` in `tensor_layout.cpp` | constructor `:162-165`, body `:175-191` | yes |
| `legacyShapeToAlignment` returns `Alignment{}` when logical == padded | `tensor_layout.cpp:46-53` | yes |
| `compute_physical_shape` rounds up to `alignment_` | `tensor_layout.cpp:347-371` | yes |
| `RowMajorPageConfig` carries no `Tile` | `page_config.hpp:19-21` | yes |
| every relaxation requires `tensor_layout` equality | `tensor_spec_relaxations.cpp:173` | yes |
| `dynamic_tensor_shape`, `match_padded_shape_only` exist | `tensor_spec_relaxations.hpp:78`, `:44` | yes |
| the factory comment "appending one for an aliased operand would shift the other's offset" | `unary_backward_program_factory.cpp:321-323` | yes |
| the writer's accessor is appended always but is dead under `OUT_SHARDED` | factory `:338-341`; writer `:32-42` | yes |
| `compute_output_specs` documents the `tilize_with_val_padding` 40×40-in-96×96 case | `:214-219` | yes |
| the aliasing preconditions | `:82-129`. The doc's list leaves out the two ROW_MAJOR-only conditions at `:118-127` (whole tiles per shard; `page_size == aligned_page_size`). Both depend only on hashed terms. The conclusion stands. | yes |

**What is stale in the doc's text** (report only; I have not edited it). The fix has landed. Text written before it is now out of date:

| Line | Now reads | Suggested |
|---|---|---|
| 7 | "… `TanhBwDeviceOperation`, both **deleted** — by `e8cbbd6b7d9` (#57832) and `9ad6239dc29` (#58352) …" — read in order, this pairs `GeluBw` with #57832 | "… both **deleted** — by `9ad6239dc29` (#58352) and `e8cbbd6b7d9` (#57832) respectively — …" |
| 21 | "**This check does NOT pass on `main` at the provenance commit below** — it is the pre-port fix this verdict is conditional on …" | "This check failed on `main` before #63334 (`be782c33d2e`, 2026-10-30), which hashes each bound tensor's whole `tensor_spec()`. It passes on `main` from that commit on." |
| 28 | "… The fix is draft PR #63334." | "… The fix is #63334, merged as `be782c33d2e` on 2026-10-30. Re-validated at `main` `a1ce48e3fe8` by the pre-port audit (2026-11-03)." |
| 34 | "Relaxation verdict: `none` — conditional on a pre-port hash fix." | "Relaxation verdict: `none`. The pre-port hash fix it depended on is on `main` (`be782c33d2e`)." |
| 38, 45, 87, 92 | present tense: "As the op stands, `none` is not safe"; "Route it to the op owner before the port"; two table rows reading "a gap today" | past tense, anchored on `be782c33d2e`. For example, line 38: "Before #63334, `none` was not safe: …" |

## D. Readiness sheet — reconciliation

**Retired rows: confirmed phantom.**
- `eltwise/unary_backward/tanh_bw` (`TanhBwDeviceOperation`, `TanhBwProgramFactory`). These were deleted by `e8cbbd6b7d9` (#57832, 2026-09-29): 9 and 3 references before, 0 after.
- `eltwise/unary_backward/gelu_bw` (`GeluBwDeviceOperation`, `GeluBwProgramFactory`). These were deleted by `9ad6239dc29` (#58352, 2026-10-08): 5 and 3 references before, 0 after.
- None of the four types exists at HEAD (0 hits; the positive control finds `UnaryBackwardDeviceOperation`).

With your replacement row, the factory-set match is one-to-one: one factory in the code, one row.

**Primary columns, cross-checked against the code:**

| Column (your value) | Code | Match |
|---|---|---|
| `Concept` = `descriptor` (both tabs) | `create_descriptor` returns `ProgramDescriptor` (`unary_backward_program_factory.hpp:19-20`, `.cpp:26`). There is no `create_workload_descriptor`, `create_program_artifacts`, `create()` or `create_mesh_workload`, and no `MeshCoordinate` parameter (so not per-coordinate). | yes |
| `Custom hash (compute_program_hash)` = `yes` (`:238`) | `unary_backward_device_operation.cpp:238-260`. It has existed since the device op was created (#56061). Of the deleted ops, `TanhBw` had one and `GeluBw` had none, which matches the old gelu_bw row's `no`. | yes |
| `Backdoor custom hash (attribute_values / to_hash)` = `no` | 0 hits for `attribute_values`, `attribute_names`, `to_hash` in the op | yes |
| `Runtime-args update (get_dynamic_runtime_args)` = `no` | 0 hits | yes |
| `Override runtime args method? (PD only)` = `no` | 0 hits. The positive control finds ternary's. | yes |
| `Pybind descriptor` = `no` | 0 `create_descriptor` hits in `unary_backward_nanobind.cpp` | yes |
| `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `no` | 0 `->address()` in the op. Your citations hold: `Buffer*` CB binding `:243`, `:252`; `TensorAccessorArgs(*buffer)` at `:307-308`, `:315`, `:330-331`, `:341`. The RTA sites matter most for this column: `Buffer*` at `:440`, `:445`, `:459`, `:468`, `:470`. | yes |
| `Op-owned tensors?` = blank | none (a `descriptor` op cannot carry them). Blank is consistent; `no` would also be. | consistent |
| `Factory definition path` = `.../device/unary_backward_program_factory.hpp` · `Declared in` = `.../unary_backward_device_operation.hpp` | the struct is defined in the factory `.hpp` (`:17-20`) and named by `program_factory_t` in the device-op `.hpp` (`:28`). The previous audit's derived row used `.cpp` / `.hpp` instead. Your values point at real files and supersede it. | yes |

**Cross-column invariants:** none violated. `get_dynamic_runtime_args` = `no`. `Op-owned tensors?` is not `yes` on a `descriptor` row. `Concept`, `Override runtime args method?`, `Execution Model` and `Porting Target` agree with each other.

**Derived cells (read, not vetted):** `Is able to port?` = `yes` (both tabs), `TTNN Is able to port?` = `yes`, `Is safe to port` = `yes`, `Execution Model` = `SPMD`, `Porting Target` = `ProgramSpecFactoryConcept`. Each follows from the primary columns. The in-place recipe derives the target from `Concept` plus the override column, and gets the same concept.

**Two disagreements: cells that rest on "no pointer patching" (not gates; *Questions* 1).**
- **The premise does not hold.** The op still uses the descriptor framework's pointer patching on every cache hit. The factory passes each tensor as a `Buffer*` through `emplace_runtime_args`. At the cache miss, `resolve_bindings` records those as `BufferBinding`s (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:598-604`). On every hit, `apply_resolved_bindings` patches the current buffer addresses into the cached runtime args (`:736-741`), and also the aliased CBs' addresses. The adapter's own comment names this "the BufferBinding fast path" (`:332-335`). The shim itself says it exists to "re-patch Buffer addresses … on a program-cache hit" (`tt_metal/impl/program/program_descriptor_patching.cpp:8-16`). The recipe's vocabulary agrees: `ttnn_factory.md:28` calls an op with no `override_runtime_arguments` "a pointer-patching op", and says it can carry a custom hash too. This op is both.
- **`Op Classification` = `PD Op (custom hash)`.** "PD Op" and "custom hash" are both right. The stated reason for dropping "(pointer-patching)" does not hold, though, and the recipe documents no value list for this derived column. Whether the sheet's vocabulary allows both qualifiers is the sheet owner's call.
- **`Pointer patching perf issue?` = retire "suspect perf regression (+ fixed latent bug)".** That text described the deleted `GeluBw` op. Whether the merged factory has a pointer-patching perf issue was not measured by this audit. "There is no pointer patching left to regress" is not the reason to clear it. The Metal 2.0 port does remove the shim from this op: typed `TensorParameter` bindings replace the `Buffer*` RTAs.

**`Diego validation`, settled: `no`.** I read the column as "the sheet owner has reviewed this row", since the legend names Diego as the owner of the classifications. On that reading, the merged row is new and nobody has validated it yet. gelu_bw's `yes` validated a row describing a deleted device op. It does not carry over. tanh_bw's `no` is not the reason either: the value is `no` because the row is new. It becomes `yes` when Diego reviews the row. If the column means something else, tell me and I will re-settle it.

**User-supplied cells with no code-visible counterpart** (recorded as given): `Known op issues` = empty; `Model` = `other`; `ProgramFactory used in llama?` = `no`; `Uses llama kernels?` = `yes`; `Provisional relaxation finding (Edwin)` and `Why secretly SPMD?` = blank; `Formerly custom hashed?` = `yes`. For that last one: the device op has had a custom hash since its creation, and the deleted `TanhBw` op had one too.

**`TensorParameter relaxation` = `none (see analysis)` clears.** This value is not one of the three that the relaxation subject's table and the column legend list. It is a see-analysis pointer, though, and the gate's own blocking table clears any "`… (see analysis)`". The brief template carries "the cell's `see analysis` pointer". The commit that introduced the column wording (`eb8dbef92bf`) routes any see-analysis cell to the doc. The doc exists and its verdict is confirmed. (*Recipe notes.*)

**Column-name drift.** I could not check against a live header row (see the top of this file). Against the recipe's column legend:
- **In the upstream legend only** (`c41f7f4ef0a`, not the in-place one): `Execution Model`, `Porting Target`, `Why secretly SPMD?`. This is recipe drift, not sheet drift.
- **In neither legend** (informational; the legend says to find such columns by header): `Diego validation`, `ProgramFactory used in llama?`, `Uses llama kernels?`, `Provisional relaxation finding (Edwin)`, `Pointer patching perf issue?`, `Formerly custom hashed?`. `Model` is named in the legend only as an example.
- **Parenthetical suffixes** match the legend by stem: `Custom hash (compute_program_hash)`, `Override runtime args method? (PD only)`, `Smuggled pointer (raw buffer addr in RTA/CRTA)`. The backdoor column now has its own header, `Backdoor custom hash (attribute_values / to_hash)`.
- **Tabs.** The legend says the CSV export "covers only the sheet's first tab (`Sheet1`), which today holds all the data". Your rows come from two tabs: an analysis-details tab and a tracking tab. The tracking tab carries its own `Is able to port?`, plus `TTNN Is able to port?` and `Is safe to port`. All four gate-like cells are `yes` here. Nothing turns on which one gates. (*Recipe notes.*)

**Tracking tab — suggested text for the two blank cells:**

```
2.0 Audit Notes / Branch / Commit:
GREEN 2026-11-03, no conditions. anasuya/metal2_port_eltwise_unary_bw @ f126189dbb6 (code = main a1ce48e3fe8; includes #63334 be782c33d2e). Files: ttnn/cpp/ttnn/operations/eltwise/unary_backward/METAL2_PREPORT_AUDIT.md + METAL2_PORT_BRIEF.md (untracked).

2.0 Port Branch / Notes:
(blank until the port session) Target ProgramSpecFactoryConcept; .relaxations = {}; reuse both _metal2 forks.
```

## E. In-flight work on this device op

- **43 open PRs add op types to this device op.** These are 41 "Fuse BF16 `<op>`_bw into one generated-kernel program" PRs (for example #57462, #58028), plus #61521 (accuracy tier for `sigmoid_bw`, updated 2026-11-02) and #61751 (`tanh_bw`). All are open and none has merged (GitHub search, 2026-11-03).
- **They share one shape.** #58028, #61521 and #61751 each add one `UnaryBackwardOpType` value, one compute kernel under `device/kernels/compute/`, and one `get_kernel_spec` row (`unary_backward_op_types.hpp` +1, `unary_backward_op_utils.cpp` +10), plus LLK headers and tests. None touches the factory or the dataflow kernels. Whichever lands second (the port or each PR) must bring its compute kernel in the other's form (*Questions* 2).
- **`main` has not touched the op directory since `be782c33d2e`.** `be782c33d2e..a1ce48e3fe8` is 168 commits, none in the op directory. The two borrowed kernels and both forks last changed on or before 2026-10-13.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**

  | Binding | Addressed | Aliased | RM blocked |
  |---|---|---|---|
  | `grad_output` | **Case 1**: reader RTA 0 `Buffer*` (`:468`) → `TensorAccessor(src0_args, src0_addr)` (`reader_binary_interleaved_start_id.cpp:51`) | **clean**: c_0 borrows the buffer (`:256-260`); the reader compiles out its accessor | **Case 1**: RM reader RTA 7 (`:440`) → `grad_addr` (`reader_unary_backward_row_major.cpp:33`) → `TensorAccessor` (`:55`) |
  | `input` | **Case 1**: reader RTA 1 (`:468`) → `:58` | **clean**: c_1 (`:261`) | **Case 1**: RM reader RTA 12 (`:445`) → `input_addr` (`:40`) → `:56` |
  | `output` | **Case 1**: writer RTA 0 (`:470`) → `writer_unary_interleaved_start_id.cpp:42` | **clean**: c_2 (`:262-266`); the accessor is dead under `OUT_SHARDED` | **Case 1**: RM writer RTA 7 (`:459`) → `dst_addr` (`writer_unary_backward_row_major.cpp:26`) → `:38` |

  Each address is used only to build its accessor. All access goes through the accessor, with `{.page_id, .offset_bytes}` in the RM kernels. There is no Case 2.
- **TensorParameter relaxation:** `.relaxations = {}` on all three, in every mode (doc §2).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** 1P+1C on c_0, c_1 and c_2 in every mode, borrowed when aliased. Self-loop on c_3 (RM reader) and c_4 (RM writer), created only when `rm_blocked`.
- **RTA varargs:** none. Every kernel reads a fixed set: borrowed reader 7 (`:22-28`), borrowed writer 3 (`:19-21`), RM reader 17 (a fixed `arg++` run, `:25-45`), RM writer 12 (`:18-31`), compute 1. No CRTA reads. CTAs are read at fixed offsets only (`get_compile_time_arg_val(0)` plus `TensorAccessorArgs` offsets).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels: reuse both forks (rung 1).** `reader_binary_interleaved_start_id_metal2.cpp` (blob `a834b1cd6a6`) and `writer_unary_interleaved_start_id_metal2.cpp` (blob `bfc0678d0f3`) sit beside their originals.
  - **Fit check: passes.** The factory supplies every named arg, binding and define the forks read.
  - **The reader fork has no binder.** Its only binder, the experimental `gelu_backward` op (#51771), was deleted by #54799. That does not make it editable (`port_patterns.md`, "A fork with no visible consumer is not fair game either").
  - **The writer fork has 26 factory binders.** It now also reads optional `ENTRIES_PER_PAGE` / `PAGE_BYTES` defines (`:31-39`). This op sets neither. Its default (one page per DFB entry, `dfb.get_entry_size()`) matches the legacy `fifo_page_size`.
  - Both originals already carry the pointer comment.
  - **Sunset lists** (not authorization to convert in place). Legacy reader: this op and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`. Legacy writer: this op; `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp:108`; `examples/example/device/{single_core,multi_core}_program_factory.cpp:83`/`:81`; `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:446`, `:672`, `:811`, `:894`, `:1013`; and six Python tests that bind it through generic op (`kernel_lib/chain_test_lib.py:33`, `debug/test_generic_op.py:98`, `eltwise/test_clamped_silu_glu_sfpu.py:121`, `test_exp_approx_disable_sfploadmacro.py:86`, `test_moe_gate_sqrt_softplus_sfpu.py:126`, `test_situ_glu_sfpu.py:117`). The previous audit's list missed the six Python tests and two of the gtest lines. Same-named private copies in `slice/`, `matmul/` and `programming_examples/` are not binders.
- **The op's own kernels convert in place.** Nothing outside the op binds or includes the RM kernels, `row_major_pages.hpp`, `operand_reconfig.hpp` or the five compute kernels (prose mentions only).
- **Binding-layer change, not an idiom rewrite.** Every kernel is already Device 2.0 with `DataflowBuffer`s.
- **Compute unpack modes: translate one-for-one, by DFB name.** Legacy sets `unpack_to_dest_mode[c_0]` and `[c_1]` to `UnpackToDestFp32` whenever fp32 DEST is on, whatever the operands' formats (`:362-366`). On Gen1 that flag only matters for a Float32 source (`tt_metal/jit_build/data_format.cpp:215-219`). Metal 2.0's `UnpackToDest` lowers to the same flag (`tt_metal/impl/metal2_host_api/program_spec/construction/kernel_lowering.cpp:143`). With `enable_32_bit_dest` set, the validator accepts it on any consumed DFB, and it requires an entry for every consumed Float32 DFB (`validation/hardware_config.cpp:54-81`, `:165-190`). So `UnpackToDest` on both operand DFBs exactly when fp32 DEST is on reproduces the legacy lowered vector. Do not index by CB number.
- **Aliased mode.** Set `borrowed_from` on c_0, c_1 and c_2. Omit the kernel tensor bindings that `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED` compile out (`port_patterns.md`, "Conditional / optional resource bindings"). Precedent: `data_movement/bcast/device/bcast_multi_core_hw_program_factory.cpp:126`, `:143`, `:173-184`.
- **Dead reader args.** `block_or_width_sharded` is 0 (`:324`), which leaves `block_height`, `block_width` and `num_cores_y` unused (`:468`). Pass them as today; the fork names all three.
- **`CoreLocalMem<uint8_t>(address)`** in `row_major_pages.hpp` is a raw-address local handle for CB slots and the scratch window. It is not a tensor binding. Leave it.
- **Validation runs on every cache hit.** The op has no `validate_on_program_cache_hit`. The relaxation analysis relies on this.
- **New op types may land first** (*E*). Re-read `get_kernel_spec` at port time.

## TTNN ProgramFactory

### Concept
`ProgramSpecFactoryConcept`. The ported-from factory has no `override_runtime_arguments`.

### Fit
- Single vs multi-program: single. One `ProgramSpec` is stamped across the mesh, and nothing varies per coordinate.
- Op-owned device resources: none.
- Tensor-arg matching: strict. `.relaxations = {}` on all three `TensorParameter`s, per the confirmed analysis.
- Legacy-to-Metal-2.0 shape: 1:1. `create_descriptor` becomes `create_program_artifacts`. No device-operation-class edit is forced (`ttnn_factory.md`): there is no pybound entry point, no hook parameter, and `program_factory_t` already exists.

### Custom compute_program_hash
Present at `device/unary_backward_device_operation.cpp:238-260`. It keys every bound tensor's whole `TensorSpec`, which is what a `none` declaration requires (*C*). The port leaves it alone.

### Stop signals
None.

## Team-only

### Out-of-directory coupling

**Roll-up: ✓ clean.**
- `tt_metal/` APIs (372 closure files): no concern.
- `kernel_lib/eltwise` (15 files): `uint32_t cb_id` spec fields in `ckl::input` / `ckl::output` (`api/chain.hpp:385-452`) ✓. `DataflowBuffer` element bindings in `CopyTile` / `PackTile` (`DataflowBufferFor` is `DataflowBuffer`, `tt_metal/hw/inc/api/dataflow/dfb_access.h:49-52`) ✓. 38 ported kernel files already pass `dfb::` tokens to `ckl::input` / `ckl::output`, for example `eltwise/unary/device/kernels/compute/eltwise_pow_int.cpp:37-40`.
- No `ttnn/kernel/`, `kernel_helper_functions/` or cross-family headers.

| Op kernel | Donor | Shape | Status |
|---|---|---|---|
| 5 compute kernels | `kernel_lib/eltwise` | `uint32_t cb_id`, `DataflowBuffer` | ✓ |
| 5 compute kernels | `operand_reconfig.hpp` (own) | `DataFormatReconfig` constant | ✓ |
| RM reader / writer | `row_major_pages.hpp` (own) | `Noc&`, accessor template, `Dfb&` template | ✓ |
| all | `tt_metal/hw/inc/api/...` | framework | ✓ |

**Borrowed kernel files (file-path coupling):**

| Kernel file | Owner | Other binders | `_metal2` fork beside it |
|---|---|---|---|
| `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | eltwise/binary | `test_generic_op.cpp:437` only | yes; no current binder |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary | 3 factories + gtests + 6 Python tests (sunset list above) | yes; 26 factory binders |

**Install coverage:** every bound file is covered by an existing glob. That is `unary_backward/CMakeLists.txt:16-18` (`*.cpp`, `*.hpp`), `eltwise/binary/CMakeLists.txt:16` (`*.cpp`), `eltwise/unary/CMakeLists.txt:17-19`, and `kernel_lib/CMakeLists.txt:4-9` (`*.hpp`, `*.inl`). No CMake change is needed.

### Relaxation candidates
None beyond the analysis doc.

### TTNN factory analysis
Recorded in *D* and *TTNN ProgramFactory*.

## Misc anomalies  *(team-only, non-gating; route to the ops team)*

- **Stale comment, `unary_backward_device_operation.cpp:26-28`.** It says the factory "reads interleaved tile pages on device". The factory also aliases sharded CBs and walks row-major blocks.
- **Misplaced comment, `:187-190`.** It describes the same-device rule but sits above the layout check (`:191-197`), not the device check (`:198-201`). Its last sentence, "gelu_bw enforces the same invariant for its own preallocated output", refers to the deleted private `gelu_bw` op.
- **Comment and code differ in form, `unary_backward_program_factory.cpp:352-366`.** The comment says "per operand that actually is fp32". The code sets both operands whenever fp32 DEST is on. The JIT ignores the flag for non-Float32 formats. Behaviour therefore matches the comment.
- **Dead reader RTAs 4-6** (`block_height`, `block_width`, `num_cores_y`), because CTA 0 is always 0 (`:324`, `:468`). In aliased mode the `Buffer*` RTAs are dead as well (reader RTAs 0-1, writer RTA 0). The fast path still patches them on every cache hit (`:468`, `:470`).
- **Process gap.** Nothing in CI stops new raw `noc_async_*` code in a directory already migrated to Device 2.0. That is how #57342's 12 calls landed, 11 weeks after #49410. The sweep in *A* is the only check that ran.

## Per-DeviceOperation attribution

There is one DeviceOperation. All four op types share the factory, the dataflow kernels and every verdict: **GREEN** for `SIGMOID_BW`, `TANH_BW`, `GELU_BW` and `GELU_TANH_BW`. Silicon today covers `SIGMOID_BW` (35 device-op tests), `TANH_BW` (the preallocated-output cache test), and `GELU_BW` / `GELU_TANH_BW` (28 cases: RM blocked, aliased height/block, addressed DRAM width-sharded).

## Questions for the user

1. **Two sheet cells rest on "no pointer patching left", and the code still pointer-patches (*D*).** I recommend keeping a pointer-patching qualifier in `Op Classification` if the sheet's vocabulary allows two qualifiers. For `Pointer patching perf issue?` I recommend blank (not measured) rather than retired. Your values stand if you prefer them. Neither cell gates.
2. **Port timing against the 43 open PRs (*E*). Raised in both earlier audits; still unanswered.** If the port lands first, each PR must write its compute kernel in Metal 2.0 form: `dfb::` tokens and a named RTA. If the PRs land first, the port converts their compute kernels too. Each is about 60 mechanical lines, one per op type. This is worth agreeing with nkapreTT before the port session.
3. **`Diego validation` = `no`.** That rests on reading the column as "the sheet owner has reviewed this row". Is that the column's meaning?
4. **Small suggested edits, report only:** the relaxation doc's stale lines (*C*). You decided last time, for ternary, to commit such a doc unchanged. Tell me if you want these applied.

## Recipe notes

- **The in-place recipe is two commits behind upstream.** It is `9425f50ae93` (audit blob `d0576d6d739`); upstream is `c41f7f4ef0a` (audit blob `6435c8fcae1`), which the previous audit ran on. Upstream reads `Porting Target` instead of deriving it, adds `Execution Model`, the `descriptor (per-coord)` / `legacy (MeshWorkload)` concepts, `yes (with PD step)` and `MeshWorkloadSpecFactoryConcept`. For this op every one of these resolves the same way: `descriptor`, exactly `yes`, `ProgramSpecFactoryConcept`. Your row already carries the upstream-only columns.
- **The relaxation column's clearing values are listed in two incompatible ways.** The relaxation subject's table (`metal2_audit.md:237-242`) and the legend ("Any other cell blocks") list exactly `none`, `dynamic (see analysis)` and `legality (see analysis)`. The gate's blocking table (`:169`), the status template (`:587`), the brief template and commit `eb8dbef92bf` accept any "`… (see analysis)`". This op's `none (see analysis)` falls between the two. Suggest: "any `… (see analysis)` value clears and routes to the doc."
- **The Case 2 bridge no longer exists.** `TensorAccessor::get_bank_base_address`, named 5 times in `metal2_audit.md` (in place and upstream), was removed by #61451 (`d097b50e356`, 2026-10-21). The migration guide now addresses by bank with `BankView` ("Addressing a Tensor by Bank"). This op has no Case 2. Nothing here depends on it.
- **Which tab gates.** The legend says the data is all on `Sheet1`. The sheet now has an analysis-details tab and a tracking tab, with `Is able to port?` on both plus `TTNN Is able to port?` and `Is safe to port`. The recipe does not say which cell is the gate.
- **A mechanical donor sweep.** The Device 2.0 gate says "every kernel … wherever it lives" but not how to find the donor headers. The JIT's `<obj>.o.dephash` sidecars list every compiled dependency. Unioning them over a test run, checked against a static all-branch closure, made the sweep exact here. A sentence in the recipe would make this repeatable.
- **Local `volatile` access.** The Device 2.0 sanctioned list covers only `get_tile_size` and `get_local_cb_interface`. The migration guide calls the `reinterpret_cast<volatile …>` form legacy ("Safe Local Memory Access"), while the CB/DFB whitelist shows it as ✅ for a peeked entry (`cb_dfb_api_whitelist.md:31`, `:41-44`). You ruled it compliant on 2026-10-22. The recipe does not say so.
- **Carried from the previous audit, still open:** the template's "Variadic-CTA" feature row has no Appendix A entry. `ttnn_factory.md`'s "Audit report deliverable" (the `## TTNN ProgramFactory` section) is not mentioned in `metal2_audit.md`'s template. There is no re-audit mode. And the factory-set match makes phantom rows a GATE, which you directed otherwise for this op.
