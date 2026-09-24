# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/unary`

- **`UnaryDeviceOperation`** (`device/unary_device_operation.{hpp,cpp}`)
  - `ProgramFactory` (`device/unary_program_factory.cpp`). `program_factory_t = std::variant<ProgramFactory>` (`device/unary_device_operation.hpp:59`), with no `select_program_factory`.

That is the op's only porting unit. A sweep of the directory for `struct *DeviceOperation` / `program_factory_t` / `create_descriptor` / `create_workload_descriptor` / `create_program_artifacts` hits only the lines above. `device/unary_composite_op.{cpp,hpp}` is host-side composition that calls other TTNN ops; it is not a device operation. `alias.hpp` and `unary_composite.hpp` contain declarations only.

**Kernels the factory binds (the audit scope, 11 sources):**

| Role | File (under `device/kernels/`) | Selected by |
|---|---|---|
| Reader | `dataflow/reader_unary.cpp` | always (`unary_program_factory.cpp:466`) |
| Writer | `dataflow/writer_unary.cpp` | always (`unary_program_factory.cpp:485`) |
| Compute | `compute/eltwise_sfpu.cpp` (default), `eltwise_identity_kernel.cpp`, `where_tss_kernel.cpp`, `mac_tss_kernel.cpp`, `logit_kernel.cpp`, `hardswish_kernel.cpp`, `logsigmoid_kernel.cpp`, `lgamma_fast_kernel.cpp` (LGAMMA + bf16), `lgamma_kernel.cpp` (LGAMMA, other dtypes) | `get_compute_kernel_path(ops_chain[0].type(), dtype)` (`common/unary_op_utils.cpp:1194-1212`). The path is built from two literals at `unary_program_factory.cpp:412-414`. |

**Unreferenced kernel files in the directory are out of scope, and easy to mistake for this op's.** `device/kernels/dataflow/` holds ten more kernels that this factory never binds: `reader_unary_interleaved_start_id.cpp`, `reader_unary_interleaved_col_multicore.cpp`, `reader_unary_interleaved_wh_multicore.cpp`, `reader_unary_sharded.cpp`, `writer_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id_wh.cpp`, and **four `_metal2` forks** (`reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_interleaved_wh_multicore_metal2.cpp`, `reader_unary_sharded_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`). Other ops' ports created those forks. None of them is a fork of `reader_unary.cpp` or `writer_unary.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target, within the scope of `metal2_audit.md`.

**Recipe docs:** `edf75ffab9c 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

> Pinning note: `edf75ffab9c` is a local rebase of the recipe onto `origin/main` @ `4d0ca398f8b` (2026-09-23). Its `metal_2.0/` tree (`c3e701f18bd`) is identical to `origin/akertesz/op-porting-recipe` @ `4bd4bf42bfe` (2026-09-03), and the audit recipe blob is `d0576d6d739`. Outside the docs directory the branch is byte-identical to `origin/main`, so the op code audited here is `main`'s. The op now includes the pre-port fixes from #56858 (`675737933f8`).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/unary` |
| **Overall** | **GREEN**. Brief issued, with **one scope decision open before the port** (see *Questions* 1). |
| **DOps / Factories** | `UnaryDeviceOperation` → `ProgramFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **Yes**. All 11 bound kernels plus the kernel_lib donors are clean. |
| *Prereqs*: Cross-op escapes | Ok. `✓ clean` (only `tt_metal/hw/inc/api/*` and `ttnn/cpp/ttnn/kernel_lib/`) |
| *Feature Support*: overall | **GREEN** (every Appendix A entry `N/A`) |
| *Feature Support*: Variadic-CTA | Ok. Not an Appendix A entry, and the op has no CTA varargs anyway (see *Recipe notes* 2). |
| *TTNN Readiness*: `Is able to port?` (the gate) | **Yes**. The **launching user supplied** the value (the sheet can't be fetched here; see *Gate detail*). |
| *TTNN Readiness*: Concept (current) | `descriptor`. `create_descriptor` returns a `ProgramDescriptor` (`device/unary_device_operation.hpp:44`). |
| *TTNN Readiness*: Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness*: Custom hash | **Yes** (not a gate; the port leaves it intact, but see *Questions* 1). `compute_program_hash` @ `device/unary_device_operation.cpp:284-350`, plus the backdoor `operation_attributes_t::to_hash()` @ `:113-123`. |
| *TTNN Readiness*: `get_dynamic_runtime_args` | No (grep clean across the op directory) |
| *TTNN Readiness*: `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`). `ProgramFactory::override_runtime_arguments` @ `device/unary_program_factory.cpp:570-666` |
| *TTNN Readiness*: Pybind `create_descriptor` | No (no `create_descriptor` / `ProgramDescriptor` / factory binding in `unary_nanobind.cpp`) |
| *TTNN Readiness*: Op-owned tensors | No (`descriptor` concept) |
| *TTNN Readiness*: Target concept | **`CustomProgramSpecFactoryConcept`** (no op-owned tensors) |
| *Port work*: Offset base pointer | **none**. GREEN. |
| *Port work*: Tensor bindings (per binding) | `input`: Case 1 on the accessor path / clean (borrowed DFB) on the native-L1-sharded path. `output`: same split. |
| *TTNN Readiness*: TensorParameter relaxation | `dynamic (see analysis)` (user-supplied). **Clears.** The doc is present, all 6 of its validity checks pass, and it declares `{dynamic_tensor_shape, relax_logical_rank}` on both slots. |
| *Port work*: TensorAccessor 3rd arg | **none**. No accessor in the op passes a 3rd argument, so the subject never fires. |
| *Port work*: CB endpoints | `c_0` legal (1:1) · `c_1` **self-loop**, host-conditional on LOGIT · `c_2` legal (1:1). No dead CBs, no multi-binding. |

**CB endpoints** are dispositions, not gates. Every out-of-window CB has a port-time resolution, recorded per `(CB, config)` below.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). Every gate-bearing subject clears: Device 2.0, feature compatibility, TTNN factory concept, offset base pointers, and TensorAccessor 3rd argument.

Two caveats should be read before the port session starts:

1. **The relaxation doc asks for an edit the recipe forbids.** Its §2 requires the port to swap `compute_program_hash`'s `distribution_key` source from the Buffer to the `TensorSpec`, in the same edit as the relaxation declaration. `ttnn_factory.md` ("The cache key: leave the custom hash alone") makes any port-time edit to the custom hash a scope violation. The swap cannot safely land *before* the port either (see *Gate detail* → relaxation). This is a scope call for the user: *Questions* 1. The brief carries the doc's instruction and marks it **decision-required**.
2. **Gate provenance.** `Is able to port?` = `yes`, `TensorParameter relaxation` = `dynamic (see analysis)` and a blank `Known op issues` were **supplied by the launching user** on 2026-09-24. This session could not fetch the sheet. See *Gate detail*.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet could not be fetched: the claude.ai Google Drive connector isn't present in this session (`ToolSearch` finds no `download_file_content`), and the sheet's CSV export URL returns HTTP 401. Following the precedent of earlier audits on this workstation, the launching user read the live sheet and supplied the gate-bearing cells for `eltwise/unary` · `UnaryDeviceOperation` · `ProgramFactory`:

  | Column | Value (as supplied) | Role |
  |---|---|---|
  | `Is able to port?` | `yes` | the gate: cleared |
  | `TensorParameter relaxation` | `dynamic (see analysis)` | clears via the analysis doc (below) |
  | `Known op issues` | blank | no block |

  **The primary-column cross-check is code-side only.** The user supplied only the gate cells, so I recorded the code values of the cross-checkable columns and could not compare them with the sheet: `Concept` = `descriptor` · `Custom hash` = yes · `get_dynamic_runtime_args` = no · `Override runtime args method?` = yes · `Pybind descriptor` = no · `Op-owned tensors?` = no · one factory in code, matching the one row the user described. Nothing here points to a broken sheet. If any of these columns disagrees with the sheet, that is a "spreadsheet is broken" finding for the readiness-sheet owner (see *Questions* 2).

  **Relaxation conjunct: clears.** The cell's pointer resolves to `analyses/relaxations/eltwise_unary.md`. On this branch the doc is **untracked** (blob `653172edfca`, mtime 2026-09-24; not on `main` or the recipe branch). It is a revision of the copy committed on `anasuya/metal2_port_unary` @ `142243af82e`, and the user placed it here for this audit. I ran **all six** validity checks against current code, and all pass:

  | # | Check | Evidence | Result |
  |---|---|---|---|
  | 1 | Key pins `tensor_layout` for both slots | `unary_device_operation.cpp:340-341` hashes `input_tensor.tensor_spec().tensor_layout()` and `output_spec.tensor_layout()`. `TensorLayout::attribute_values()` = `(dtype, page_config, memory_config, alignment)` (`tensor_layout.hpp:74-75`), and `PageConfig` hashes its config, which includes `Tile`. | pass |
  | 2 | Key pins sharded distribution geometry | `distribution_key` (`:323-336`) hashes `shard_shape_in_pages()` + `cores()` for both slots (`:346-347`). **Source: the Buffer's `buffer_distribution_spec()`, falling back to `spec.compute_buffer_sharding_args()` only when the buffer is null** (`:328-331`). The doc tells the reader to carry this answer into its §2. | pass (Buffer source, so §2's swap applies) |
  | 3 | Accessor compiled away on sharded slots | `reader_unary.cpp:20-65`: `TensorAccessorArgs<0, 0>()` / `TensorAccessor` sit inside the `#else` of `#if SRC_SHARDED`. `writer_unary.cpp:20-69` mirrors this. | pass |
  | 4 | `has_sharding` pinned by the key | `dst_shard_vol` is engaged iff `get_shard_specs` returned a value (`:293-300`), and both optionals are hashed (`:348-349`). | pass |
  | 5 | Exactly one factory | `std::variant<ProgramFactory>` (`unary_device_operation.hpp:59`), with no `select_program_factory` | pass |
  | 6 | TILE key omits shape; override re-applies the split | `padded_shape` is hashed only when `ROW_MAJOR` (`:345`). The override re-enumerates via the shared `enumerate_core_rt_args` (`unary_program_factory.cpp:596`). | pass |

  Verdict (quoted from the doc §1): **`dynamic`**. Declaration (doc §2): `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}` on **both** `TensorParameter`s, unconditionally. `match_page_size` and `match_padded_shape_only` are **not** set.

  I also spot-checked the claims that §2's code change rests on, and all three hold. Validation's `relaxation_fields::shard_distribution_of` reads `spec.compute_buffer_sharding_args().buffer_distribution_spec()` (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:105-109`), not the buffer. `hash_tensorspec_with_relaxation` has no production callers (it is declared and defined only). The code already carries the matching `TODO(port)` (`unary_device_operation.cpp:320-322`). The doc's reason for "same edit" also holds: before the port, the legacy descriptor bakes the **buffer's** geometry into accessor CTAs (`TensorAccessorArgs(*src_buffer)`, `unary_program_factory.cpp:462`), so the Buffer branch is the correct key *today*, and swapping it early would under-pin what legacy bakes. Hashing both sources was measured by the doc at +16% on a 64-core BLOCK_SHARDED dispatch. How this squares with `ttnn_factory.md` is *Questions* 1.

- **Device 2.0 (every kernel used): GREEN.** A sweep of all 11 bound kernels found no legacy DM idiom: no raw `noc_async_*`, no `cb_*` free-function FIFO ops, no `get_read_ptr`/`get_write_ptr`, no addr-gens, no semaphore addresses. The two DM kernels are already on `Noc` + `DataflowBuffer` + `TensorAccessor`. The compute kernels drive FIFO ops through `DataflowBuffer` (`eltwise_sfpu.cpp`, `mac_tss_kernel.cpp`) or through `compute_kernel_lib`. Inside kernel_lib, `eltwise/core/chain.inl:2414-2435` routes every wait/pop/reserve/push through `DataflowBuffer(cb)`. The same sweep over `ttnn/cpp/ttnn/kernel_lib/eltwise/**`, `dfb_helpers_compute.{hpp,inl}` and `dest_helpers.hpp` found only a comment line (positive control: 6 `DataflowBuffer` hits). The remaining CB-index free functions are all fine:
  - `get_local_cb_interface(cb).fifo_page_size` (`reader_unary.cpp:57`, `writer_unary.cpp:60`, `kernel_lib/dfb_helpers_compute.inl:83,96,102`) is **sanctioned**.
  - Compute-LLK calls taking a CB id (`compute_kernel_hw_startup`, `copy_init`, `copy_tile`, `pack_tile`) have no wrapper-method replacement, so they are not holdovers.
  - The `SFPU_OP_CHAIN_0` define expansions (`common/unary_op_utils.cpp:972-1177`) emit only DEST-register `*_tile_init()` / `*_tile(idst, …)` calls. They contain no CB access and no arg reads.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | *(no violations)* | | | |

- **Feature compatibility:** GREEN. No gate fired. Scanned the op directory, all bound kernels, and the kernel_lib donors (positive control: `GlobalSemaphore` hits 181 files under `experimental/ccl`).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `global_circular_buffer` field, `remote_index` / `remote_cb` idiom, or 4-arg `CreateCircularBuffer`. All three `CBDescriptor`s (`unary_program_factory.cpp:420-453`) set only `total_size`, `core_ranges`, `format_descriptors`, and optionally `buffer`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is never set (default 0). No `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / include. The op uses no semaphores at all. |

- **CB endpoints (GATE-free):** Census per node and per config. There are two configs, because `get_shard_specs` is non-null only under `is_native_L1_sharding` (`common/unary_utils.cpp:36-59, 61-112`), which needs **both** tensors 2D-sharded in L1, even, on one grid:
  - **(A) accessor path** (`has_sharding == false`): interleaved, plus every mixed / ND / DRAM / uneven / non-tile-aligned-RM sharded case. Sub-variants: TILE, and ROW_MAJOR (`RM_INTERLEAVED=1`).
  - **(B) native-L1-sharded path** (`has_sharding == true`): `c_0` / `c_2` borrow the input / output buffers (`unary_program_factory.cpp:428, 452`).

  | CB | Config | Touchers (per node) | Verdict |
  |---|---|---|---|
  | `c_0` (src0) | A | reader FIFO-produces (`reader_unary.cpp:43,53` / `:59,62`); compute FIFO-consumes (all 9 compute kernels) | legal 1:1 |
  | `c_0` | B | reader `reserve_back`+`push_back(num_pages)` as a readiness signal (`reader_unary.cpp:21-22`); compute consumes | legal 1:1 (borrowed from `input`) |
  | `c_1` (tmp0) | A, B, only when `ops_chain[0] == LOGIT` (`unary_program_factory.cpp:431-441`) | `logit_kernel.cpp` alone produces (`:41-45`) and consumes (`:49-56`) | **self-loop** (1 toucher, compute; legal on Gen1). Not allocated in other configs, so the DFB spec stays conditional exactly as legacy is. |
  | `c_2` (output) | A | compute FIFO-produces; writer FIFO-consumes (`writer_unary.cpp:45,55` / `:62,65`) | legal 1:1 |
  | `c_2` | B | compute produces; writer `wait_front`+`pop_front(num_pages)` handshake (`writer_unary.cpp:23-24`) | legal 1:1 (borrowed from `output`) |

  Hidden-second-writer hunt: none of the bound kernels uses `get_write_ptr`, `get_read_ptr`, `fifo_wr_ptr`, `fifo_rd_ptr` or `evil_set_*` on any CB. The only `get_local_cb_interface` reads are `.fifo_page_size`, a metadata peek by the kernel that already holds that binding. No kernel is instantiated twice. No dead CBs and no multi-binding.

- **Offset base pointers: GREEN.** The address RTAs are reader RTA0 and writer RTA0. On a miss they are delivered as bare `Buffer*` (`input.buffer()` / `output.buffer()`, `unary_program_factory.cpp:531-557`). On a hit they become `input.buffer()->address()` / `output.buffer()->address()` (`:585-586`, written to `r[0]` / `wr[0]` at `:613,616`). No host arithmetic is folded in. `eltwise/unary` is not in the triage tables (`2026-07-19_offset_base_pointers.md`), so this is "no fold, op not in tables": clean. The kernel-side `.offset_bytes = j * chunk_size` in the RM path (`reader_unary.cpp:49`, `writer_unary.cpp:52`) is kernel arithmetic on an accessor page, not a host-folded base.
- **TensorAccessor 3rd argument: N/A.** No accessor in the op passes a 3rd argument (`TensorAccessor(src_args, src_addr)` at `reader_unary.cpp:26`, `TensorAccessor(dst_args, dst_addr)` at `writer_unary.cpp:28`). The compute kernels and kernel_lib construct none. The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config):
  - `input`: **Case 1** on config A. `Buffer*`-binding form in reader RTA0 (`unary_program_factory.cpp:531,536,555`) feeds `TensorAccessor(src_args, src_addr)` (`reader_unary.cpp:25-26`); the accessor uses `ArgConfig::RuntimeTensorShape` (`unary_program_factory.cpp:462-463`). **Clean** on config B: a borrowed-memory DFB `c_0` (`.buffer = src_buffer`, `:428`), where the reader never touches tensor memory and RTA0 plus the accessor CTA/CRTA payload are dead on that path.
  - `output`: **Case 1** on config A (writer RTA0 → `TensorAccessor(dst_args, dst_addr)`, `writer_unary.cpp:27-28`). **Clean** on config B (borrowed DFB `c_2`, `:452`).
  - Not the silent-wrong hazard today (`Buffer*` bindings, and the override re-writes the addresses on a hit).
- **TensorParameter relaxation:** `dynamic (see analysis)` → `analyses/relaxations/eltwise_unary.md`. The per-slot declaration, transcribed: **both** `input` and `output` get `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}`, unconditionally, in every config. Do **not** set `match_page_size` or `match_padded_shape_only`. The doc's §2 adds the `distribution_key` Buffer→spec swap in `compute_program_hash`, **same edit, decision-required** (*Questions* 1).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_1` on the LOGIT compute kernel, keeping the DFB spec conditional on `ops_chain[0] == LOGIT`. `c_0` and `c_2` are legal 1:1 in both configs, borrowed from `input` / `output` on config B only.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** `compute/eltwise_sfpu.cpp` is **lent**. Other binders: `examples/example` `ExampleDeviceOperation::SingleCore` (`examples/example/device/single_core_program_factory.cpp:91`) and `::MultiCore` (`multi_core_program_factory.cpp:89`), `examples/example_multiple_return` `ExampleMultipleReturnDeviceOperation::SingleCore` (`single_core_program_factory.cpp:80`, still legacy `create()`), and the tests `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246` and `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`. There is **no `_metal2` fork on this branch**, so this port creates `compute/eltwise_sfpu_metal2.cpp` beside the original (rung 2). The other 10 bound kernels have **no other binder**. Census: grep by filename, then each hit disambiguated. The `tests/tt_metal/**/reader_unary.cpp` / `writer_unary.cpp` / `eltwise_sfpu.cpp` hits are same-named private copies, and every split `…/kernels/dataflow/"` literal resolves to a kernel unary doesn't bind. Those 10 convert in place. Install coverage: `eltwise/unary/CMakeLists.txt:16` (`file(GLOB_RECURSE kernels device/kernels/*.cpp)`) picks up the fork with no build edit.
- **RTA varargs:** none. Reader/writer RTAs are a fixed block of 3 (sharded) or 8 (accessor path) distinct fields (`reader_unary.cpp:11-13,30-34`). Compute RTAs are fixed at 3. The only variable-length block is the accessor's `RuntimeTensorShape` CRTA payload, which disappears into the `tensor::` binding.
- **Also needed by the porter:**
  - **Target concept `CustomProgramSpecFactoryConcept`.** Translate `override_runtime_arguments` (`unary_program_factory.cpp:570-666`). Only its per-core RTA re-writes survive as `ProgramRunArgs`: the work split, start ids, noop zero-fill, RM chunk tail, and compute scalars. Its accessor-CRTA rebuild (`:638-653`) and the CB-address `apply_descriptor_runtime_args` (`:655-665`) are the tensor-binding and `borrowed_from` refresh that Metal 2.0 performs itself. Everything goes through the shared `enumerate_core_rt_args` (`:127-333`), so miss and hit cannot drift.
  - **Compute `opt_level`: set `O3` explicitly.** The legacy compute descriptor sets none and resolves to `KernelBuildOptLevel::O3` (`tt_metal/impl/program/program.cpp:485`). A Metal 2.0 `KernelSpec` defaults to `O2` (`kernel_spec.hpp:122`). The DM kernels are `O2` on both sides.
  - **Unread trailing compute CTA.** The factory appends `static_cast<uint32_t>(cb_data_format)` to every compute kernel's CTAs (`unary_program_factory.cpp:506`: index 2 for HARDSWISH, 1 for LOGIT, 0 otherwise). No bound kernel reads it (`get_compile_time_arg_val` appears only in `hardswish_kernel.cpp:14-15` and `logit_kernel.cpp:16`). Carry it as a named CTA, because dropping it is not a sanctioned port change. Record it in the port report.
  - **`unpack_to_dest_mode` is set for `c_1` even when `c_1` doesn't exist** (`unary_program_factory.cpp:401-404`). When mapping it onto per-DFB unpack modes, key the tmp0 entry on the DFB only where the LOGIT-conditional DFB exists.
  - **Runtime-selected compute source.** All 9 compute sources must convert together, because any one of them can be bound for a given `op_chain[0]`. Seven call `compute_kernel_lib` helpers taking `uint32_t` CB ids: pass `dfb::name` directly (`port_patterns.md` "Pass DFB handles directly…"). The kernels are already part-modernized (`DataflowBuffer` / kernel_lib), so the port changes only the binding layer.
  - **Don't copy the unmerged branch.** An earlier attempt on `anasuya/metal2_port_unary` (`142243af82e`) created its own `eltwise_sfpu_metal2.cpp`. It isn't on `main` and predates #56858 and the revised relaxation doc. Don't bind or copy it as a precedent. Only reuse a fork if one has landed beside the original.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`.
  - Function-call escapes: `tt_metal/hw/inc/api/{dataflow,compute,tensor}/*` (class 1, no concern). `ttnn/cpp/ttnn/kernel_lib/` (class 2) is used by `eltwise_identity_kernel.cpp`, `hardswish_kernel.cpp`, `lgamma_fast_kernel.cpp`, `lgamma_kernel.cpp`, `logit_kernel.cpp`, `logsigmoid_kernel.cpp`, and `where_tss_kernel.cpp`.

    | Op kernel(s) | Donor | Functions called | Shape | Status |
    |---|---|---|---|---|
    | the 7 above | `kernel_lib/eltwise/api/chain.hpp` (+ `core/chain.inl`, `api/convenience.hpp`) | `input(uint32_t cb_id, …)`, `output(uint32_t cb_id, …)`, `eltwise_chain`, `copy`, `IterationShape` | `uint32_t cb_id` (constexpr / NTTP position) | ✓ |
    | same | `kernel_lib/eltwise/{unary,binary/sfpu,generators,core}/*` | DEST-slot op tags (`Hardsigmoid`, `Log`, `Where`, `FillScalar`, …) | no resource handles | ✓ |
    | (transitively) | `kernel_lib/dfb_helpers_compute.hpp`, `dest_helpers.hpp` | CB metadata via sanctioned `get_local_cb_interface` | `uint32_t cb_id` | ✓ |

    Per-call detail omitted (every row ✓).
  - **Borrowed kernel files:** none. The op owns all 11 kernels it binds. The reverse coupling (lent `eltwise_sfpu.cpp`) is in *Heads-ups*.
- **Relaxation candidates** (from the custom hash): none beyond the analysis doc's declaration, which is now authoritative for this op. FALLIBLE as always.
- **TTNN factory analysis:** `descriptor` (`unary_device_operation.hpp:44`). No op-owned tensors, no MeshWorkload. Custom hash @ `unary_device_operation.cpp:284` (+ `to_hash()` @ `:113`). No `get_dynamic_runtime_args`. `override_runtime_arguments` @ `unary_program_factory.cpp:570`. No pybound `create_descriptor` or other risky pybind of internals in `unary_nanobind.cpp`. Target: `CustomProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

1. **Dead trailing compute CTA.** `unary_program_factory.cpp:506` pushes the input `cb_data_format` as the last compute CTA. No bound kernel reads it, and none did at introduction (#55487, `7e6229d80f0`). It only perturbs the kernel build key.
2. **Non-32x32 tiles are mis-sized.** CB page sizes come from `tile_size(DataFormat)` (`unary_program_factory.cpp:355-358`, and `:171-174` in the split), while the work split reads the real tile (`:167-169`). This is known and family-wide (also in ported `typecast`); the relaxation doc §4 and the #56858 review both record it. It routes to the eltwise team.
3. **Unreachable branch in `get_shard_specs`.** `common/unary_utils.cpp:109-111` (the `adjust_to_shape` path for an unsharded input) can't run, because `is_native_L1_sharding` already required the input to be sharded (`:41-43`).
4. **Sharded output specs drop input overpadding.** Both sharded branches of `compute_output_specs` build `TensorLayout` directly, not via `fromPaddedShape` (`unary_device_operation.cpp:239-240, 257-262`). This was already recorded in the #56858 review.
5. **The sharded path carries dead DM payload.** On config B the reader and writer still receive the buffer address in RTA0 and the full `TensorAccessorArgs` CTA/CRTA block (`unary_program_factory.cpp:462-463, 481-482, 531-532`), and the kernels never read it. It is harmless, and the port removes it naturally.
6. **Only `ops_chain[0]` selects the compute kernel.** The dedicated kernels (`where_tss`, `logit`, `hardswish`, `logsigmoid`, `lgamma*`, identity) never expand `SFPU_OP_CHAIN_0`, so any later ops in a chain headed by one of them would be silently skipped. I didn't check whether callers can build such a chain.

## Questions for the user

1. **Relaxation doc §2 vs `ttnn_factory.md`: who makes the `distribution_key` swap?**
   - The doc (authoritative for the relaxation) requires the port to change `compute_program_hash`'s `distribution_key` from the Buffer's `buffer_distribution_spec()` to `spec.compute_buffer_sharding_args()`, in the same edit as the declaration (`unary_device_operation.cpp:328-331`; the code's own `TODO(port)` at `:320-322` says the same).
   - `ttnn_factory.md` ("The cache key: leave the custom hash alone") calls any port-time hash edit a scope violation. It says a hash/declaration mismatch means the pre-port vetting was wrong, and the response is to stop, not fix.
   - A pre-port ops-side change (the recipe's normal route) doesn't work here. While the legacy descriptor bakes the buffer's geometry, a spec-keyed hash would under-pin what legacy bakes. Hashing both costs +16% (per the doc).
   - **Recommendation:** sanction the swap as a named, port-time exception, recorded prominently in the port report as a device-op-class edit. The alternative, keeping the Buffer key, leaves a latent validation `TT_FATAL` if the two resolutions ever diverge (the doc found none by hand-search).
   - The brief marks this decision-required rather than deciding it.
2. **Sheet provenance.** The gate cells came from you (2026-09-24), not a fetch. If you have the sheet open, a glance at `Concept` / `Custom hash` / `Override runtime args method?` / `Pybind descriptor` / `Op-owned tensors?` against the code values in *Gate detail* would complete the primary-column cross-check.
3. **Land the relaxation doc.** `analyses/relaxations/eltwise_unary.md` is untracked here. For the sheet's `(see analysis)` pointer to resolve for anyone else (and for the port's provenance), it needs to be checked in on the recipe branch. Who owns that?

## Recipe notes

1. **No fallback when the Drive connector is missing.** `ttnn_op_porting_readiness.md` offers only "point the human to authorize". On this workstation the connector is never present, so every audit here depends on user-supplied cells, which the recipe doesn't sanction. A documented "user pastes today's rows" path, with its reduced cross-check, would help.
2. **The status-summary template has a `Feature Support — Variadic-CTA` row, but Appendix A has no such entry.** The *RTA varargs* subject says CTA varargs port via `compile_time_varargs` and "don't gate either". The row looks stale.
3. **A relaxation analysis can require a hash edit, which `ttnn_factory.md` forbids.** The recipe assumes the analysis only *declares*, and that any hash fix happens upstream, before the port. For unary the doc shows the key's correct source *changes with the port itself*, so no upstream-only sequencing exists. The recipe could name "hash edit carried by the relaxation analysis" as a sanctioned exception, or rule it out.
