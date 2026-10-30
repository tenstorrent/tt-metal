# Metal 2.0 Audit Findings — `eltwise/ternary`

- **`TernaryDeviceOperation`** (`device/ternary_device_operation.hpp:19`). It has one factory: `program_factory_t = std::variant<TernaryProgramFactory>` (`:67`), and no `select_program_factory`.
  - `TernaryProgramFactory`, declared at `ternary_device_operation.hpp:49-65`. `create_descriptor` is at `device/ternary_program_factory.cpp:997` and `override_runtime_arguments` at `:1543`.
- The one factory serves every configuration of the op: variants TTT / TTS / TST, every broadcast type, and the op types WHERE, LERP, ADDCMUL, ADDCDIV, SNAKE_BETA and MAC. Kernels are chosen per configuration by `TernaryKernelConfig` (`device/ternary_op_utils.cpp:361-379`, map at `:192-359`), `get_kernel_file_path` (`:381-422`), `override_addcmul_compute_kernel` (`:424-433`) and the LLK row-broadcast override (`ternary_program_factory.cpp:1254-1268`).
- **Reached only through `ttnn::prim::ternary`** (four overloads, `ternary_device_operation.cpp:650-826`). Its six call sites are all in `ternary.cpp`:
  - `invoke_ternary_llk` (`:78`), used by TTT `where` (`:142`) and by `mac` TTT / TTS / TST (`:273`);
  - `addcmul` (`:373`) and `addcdiv` (`:412`), both TTT;
  - `lerp` with a scalar weight (TTS, `:450`) and with a tensor weight (TTT, `:490`);
  - `snake_beta` (`:508`).
  - A repo-wide grep finds no caller outside the op directory. The only outside mention of `TernaryDeviceOperation` is a docstring in `tests/ttnn/unit_tests/operations/eltwise/test_ternary.py`.
  - `where` TTS / TST does not reach this device op: `ternary.cpp:161` / `:185` go to `binary_ng`. TSS goes to `where_tss` (`:203`).
- **Kernels the factory binds: 22, all in this op's directory, none bound by any other op.**
  - Dataflow (12), under `device/kernels/dataflow/`: `ternary_reader_nosubtilebcast_ttt.cpp`, `ternary_reader_nobcast_tst_tts.cpp`, `tst_tts_reader_outer_bcast.cpp`, `tst_tts_reader_scalar_bcast.cpp`, `ternary_reader_scalar_ttt.cpp`, `ternary_reader_colbcast_ttt.cpp`, `tts_tst_reader_col_bcast.cpp`, `ternary_reader_rowbcast_ttt.cpp`, `tts_tst_reader_row_bcast.cpp`, `ternary_reader_row_col_bcast_ttt.cpp`, `tts_tst_reader_row_col_bcast.cpp`, `ternary_writer_nobcast.cpp`.
  - Compute (10), under `device/kernels/compute/`: `ternary_sfpu_no_bcast_ttt.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, `ternary_sfpu_row_bcast_ttt.cpp`, `ternary_sfpu_no_bcast_tts_tst.cpp`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`, `ternary_addc_ops_sfpu.cpp`, `ternary_addc_ops_sfpu_bcast.cpp`, `ternary_addc_ops_fpu_rowbcast.cpp`, `ternary_addcmul_int_sfpu.cpp`, `ternary_addcmul_int_sfpu_bcast.cpp`.
  - The op's own kernel-side header `compute/ternary_sfpu_copy_init.hpp` is new since prev4 (#60073). Two compute kernels include it.
- **Donor code the kernels call:** binary_ng's `device/kernels/dataflow/fill_tile_utils.hpp` (8 readers); binary_ng's `device/kernels/compute/eltwise_utils_common.hpp` and `eltwise_utils_sfpu.hpp` (5 compute kernels include them and use nothing from them); `ttnn/cpp/ttnn/kernel_lib/eltwise` (`ternary_addcmul_int_sfpu.cpp`).
- **Kernel files in the directory that this op does not bind (not audited):** `compute/ternary_addc_ops_fpu.cpp` and `compute/ternary_addc_ops_fpu_bcast.cpp` are bound nowhere. `dataflow/ternary_reader_nobcast_ttt.cpp` is bound by `eltwise/binary_backward` (`binary_backward_program_factory.cpp:122`). A ternary port must leave it alone.
- Earlier audits sit beside this file and were not modified: `METAL2_PREPORT_AUDIT.prev4.md` (GREEN on an uncommitted working tree, 2026-10-22), `.prev3.md`, `.prev2.md` and `.prev.md` (RED). This audit used prev4 only as a checklist. Every finding below was re-derived from the code on HEAD.

**Scope:** TTNN op, Gen1 (WH/BH) target. Within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `b90203d162f 2026-10-30 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- The branch's `metal_2.0/` tree is identical to upstream `9425f50ae93`. Upstream `akertesz/op-porting-recipe` is two commits further on (`c41f7f4ef0a`, 2026-10-29). This audit follows the in-place doc, as asked. *Recipe notes* says what the two newer commits would change. Neither changes this verdict.

**Code audited:** branch `anasuya/metal2_port_eltwise_ternary` @ `b90203d162f`, on 2026-10-30.
- HEAD is `origin/main` `3f57b50ff00` (2026-10-30) plus 137 recipe-doc commits. No op code differs from `main`.
- HEAD contains #62084 (`c04ac9c6ca9`, "Metal 2.0 eltwise/ternary Pre-Port Fixes", merged 2026-10-28, ticket #60512).
- No tracked file is modified. The untracked files are the `.prev*` audits, the relaxation analysis docs, and unrelated notes.

**Readiness sheet:** not fetched (no Drive access in this session). The three gate cells below are user-supplied corrections of the stale sheet (2026-10-30). The other primary columns are the values prev4 recorded. Each one is cross-checked against HEAD's code under *Gate detail*, and each one matches.

| Column | Stale sheet | Corrected (user-supplied) |
|---|---|---|
| `Known op issues` | "Seems buggy; needs analysis" | cleared. That entry is what #62084 investigated and fixed. |
| `Is able to port?` (tab 1) | `no` | `yes` |
| `TensorParameter relaxation` | `(dynamic - pending analysis)` | `dynamic (see analysis)` |

**Relaxation analysis doc:** `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/relaxations/eltwise_ternary.md`. It is untracked on this branch, blob `c89ba329e90` (confirmed with `git hash-object`). Its line numbers belong to the pre-merge tree `21d0f3c7ba9`. They are re-resolved below.

**Silicon evidence:** one Wormhole n150, on 2026-10-30.
- The host library in `build_Release` was built today. It contains #62084's per-slot message. At HEAD, ternary's host sources differ from #62084's only by #62567, which swaps `buffer() != nullptr` for the equivalent `is_allocated()`. JIT kernels were compiled from this tree into a fresh kernel cache.
- 16 targeted tests passed. They are the #62084 regression tests plus four cache tests. Three probe calls were run from a scratch script that is not checked in. Broader suites were not run locally; CI covers them.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/ternary/` |
| **Overall** | **GREEN.** Every gate-bearing subject clears on HEAD, with no conditions. The relaxation analysis validates: all five of its checks pass on HEAD's code, and its key claims hold on silicon. |
| **DOps / Factories** | `TernaryDeviceOperation` → `TernaryProgramFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **Yes.** The 22 bound kernels and the op's own header contain no Device 1.0 call. The donor `fill_tile_utils.hpp` is compliant on `main`: #62084 moved `fill_tile_with_first_row` onto `Noc` (`:384-406`). Its 11 other helpers use local `volatile tt_l1_ptr` access only, which the user ruled compliant on 2026-10-22. |
| *Prereqs*: Cross-op escapes | Ok (✓ clean). Donor calls take raw L1 pointers, `uint32_t` CB ids or `DataflowBuffer&`. No borrowed kernel files. |
| *Feature Support*: overall | GREEN. No Appendix A entry fires. |
| *Feature Support*: Variadic-CTA | Ok. None. (Appendix A has no such entry; see *Recipe notes*.) |
| *TTNN Readiness*: `Is able to port?` (the gate) | **Yes** (user-supplied correction). The primary-column cross-check is clean. |
| *TTNN Readiness*: Concept (current) | `descriptor`. `create_descriptor` returns a `ProgramDescriptor` (`ternary_device_operation.hpp:50-53`, `ternary_program_factory.cpp:997`). |
| *TTNN Readiness*: Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness*: Custom hash | **Yes** (not a gate; the port leaves it intact). `compute_program_hash` at `ternary_device_operation.cpp:582-637`, layered over `operation_attributes_t::to_hash()` at `:261-271`. |
| *TTNN Readiness*: `get_dynamic_runtime_args` | No. Only a stale comment mentions it (`ternary_device_operation.cpp:580`). |
| *TTNN Readiness*: `override_runtime_arguments` | **Yes** (not a gate; it selects `CustomProgramSpecFactoryConcept`). Declared at `ternary_device_operation.hpp:59-64`, body at `ternary_program_factory.cpp:1543-1649`. |
| *TTNN Readiness*: Pybind `create_descriptor` | No. `ternary_nanobind.cpp` binds only the public functions. |
| *TTNN Readiness*: Op-owned tensors | No |
| *TTNN Readiness*: Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work*: Offset base pointer | none |
| *Port work*: Tensor bindings (per binding) | `predicate`, `value_true`, `value_false`, `output`: **Case 1** when the slot's `SRC_SHARDED_*` / `DST_SHARDED` define is 0, **clean** (borrowed-memory DFB) when it is 1. No Case 2. |
| *TTNN Readiness*: TensorParameter relaxation | `dynamic (see analysis)` (clears). **Verdict: VALID.** `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on every `TensorParameter`, unconditionally. |
| *Port work*: TensorAccessor 3rd arg | none. No accessor passes a 3rd argument (28 two-argument constructions). |
| *Port work*: CB endpoints | 1P+1C (c_0, c_1, c_2; c_3 when `DST_SHARDED=0`) · self-loop (c_3 when `DST_SHARDED=1`) · **conditional DFB** (c_4, c_5, c_6) |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). Portable subset: the whole op. There is one factory, and the relaxation declaration is unconditional.

In plain terms:

1. **Every condition prev4 named has landed.** prev4 was GREEN only on an uncommitted working tree. Its conditions were the three ternary correctness fixes and the `fill_tile_utils.hpp` Device 2.0 fix. Those four changes are exactly #62084, merged to `main` on 2026-10-28. This branch sits on today's `main`. No rebase is pending, and no uncommitted code is involved.
2. **Every gate was re-derived from HEAD, not carried forward.** The Device 2.0 sweep, the donor shapes, the address RTAs, the accessor constructions, the CB census and the RTA reads were all redone on this tree. Five upstream commits touched the op or its donor after prev4's base: #60073, #52360, #61721, #62084 and #62567. None of them introduces a gate. #60073 adds a kernel-side header that the brief calls out.
3. **The relaxation analysis validates on HEAD.** All five validity checks pass. The doc's line numbers come from the pre-merge tree. Only `ternary_device_operation.cpp` moved, because review trimmed three explanatory comments: by −3 from line 187, −10 from line 380 and −16 from line 641. Every citation resolves by content. The mapping is under *Relaxation analysis*.
4. **The sheet should now read as corrected.** With #62084 on `main`, the analysis's three prerequisites hold on `main` itself. The corrected cells (`Known op issues` cleared, `Is able to port?` = `yes`, `dynamic (see analysis)`) are consistent with the code.

## Evidence basis

| Verdict | Silicon (this HEAD) | Code reading only |
|---|---|---|
| Every tensor slot is TILE-only | `test_ternary_rejects_row_major` (5 cases), `test_ternary_rejects_row_major_operand` (4 cases: `input_b` and `input_c` each rejected with a TILE `input_a`) | the shape of the rule (per slot, no condition) |
| The TILE checks also run on cache hits | Probe P3. A host row-major tensor, logical `[1,1,30,30]` padded to `[1,1,32,32]`, is rejected at `input_a` after a TILE call of that shape. The TILE entry stays the only entry and still serves the next call exactly. | That P3 is a hit, not a miss: every hashed term equals the TILE call's. The framework fallback (`ttnn/api/ttnn/device_operation.hpp:284-288`). |
| Check 1: the key pins `tensor_layout` | `test_ternary_preallocated_output_memory_distinguishes_cache_entries` (2), `test_ternary_cache_miss_different_output_alignment`, `test_ternary_cache_miss_different_alignment` | the per-term trace |
| Checks 3, 4, 5 | — | yes |
| The §3 pairs share one cache entry | P1: `[4,1,32,32]` then `[1,4,32,32]`, 1 entry, both exact. P2: `[1,32,64]` then `[1,1,32,64]`, 1 entry, both exact. | — |
| The override re-derives per-core args on a hit | `test_ternary_addcmul_cache_hit_refreshes_operand_addresses`, `test_ternary_scalar_distinguishes_cache_entries` | the builder trace |
| Device 2.0, features, offset base pointers, 3rd argument, CB endpoints, RTA varargs, coupling | — | yes, with the sweeps and positive controls named under each subject |

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The cell is `yes` (user-supplied correction). The primary cross-check is clean.

  | Column | Sheet value | Code on HEAD |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returns `ProgramDescriptor` (`ternary_device_operation.hpp:50-53`, `ternary_program_factory.cpp:997`). It takes no `MeshCoordinate`. ✓ |
  | `Custom hash` | `yes` | `compute_program_hash` (`ternary_device_operation.hpp:71`, `.cpp:582-637`) ✓ |
  | `Backdoor custom hash` | `layered` | `to_hash()` (`.hpp:37`, `.cpp:261-271`). `hash_operation` folds `args` through it (`.cpp:612-613`; `tt_stl/tt_stl/reflection.hpp:1388-1392`). ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | absent. Only the stale comment at `.cpp:580`. ✓ |
  | `Override runtime args method?` | `yes` | `ternary_device_operation.hpp:59-64` / `ternary_program_factory.cpp:1543-1649` ✓ |
  | `Pybind descriptor` | `no` | no `create_descriptor` in `ternary_nanobind.cpp` ✓ |
  | `Smuggled pointer` | `no` | Every address RTA is delivered as a `Buffer*` binding on the miss path (`ternary_program_factory.cpp:713-722`, `:1483-1534`). The hit path writes the current addresses into every row (`:1599-1604`). ✓ |
  | `Op-owned tensors?` | blank | not a `WorkloadDescriptor`; no `buffers` ✓ |

  - **Factory-set match:** one row, one factory (`ternary_device_operation.hpp:67`). No phantom or missing row.
  - **Cross-column invariants:** none violated. `get_dynamic_runtime_args` and op-owned tensors are both absent.
  - **`Known op issues`:** cleared. The latent items under *Misc anomalies* are code observations, not sheet entries, and none of them gates.

- **Device 2.0 (every kernel used): GREEN.**
  - **The 22 bound kernels and `ternary_sfpu_copy_init.hpp`: no violation.**
    - Swept for free-function calls (not wrapper methods) to `noc_async_*`, `noc_semaphore*`, `noc_inline*`, `get_noc_addr*`, `InterleavedAddrGen*` / `ShardedAddrGen` / `InterleavedPow2*`, `get_semaphore`, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, `get_write_ptr` / `get_read_ptr`, `get_local_cb_interface`, `get_tile_size`, `get_dataformat`, and `get_arg_addr`. 0 hits.
    - Positive control: the same pattern finds the five raw calls in the pre-#62084 donor (`git show c04ac9c6ca9^:…/fill_tile_utils.hpp`, `:382-387`).
    - `my_x[`, `my_y[`, `volatile`, `tt_l1_ptr`: none. `reinterpret_cast`: three sites, each reinterpreting a local scalar argument as `float` in a compute kernel (`ternary_sfpu_no_bcast_tts_tst.cpp:55`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:69`, `:78`). No memory access. Not flagged.
    - Dataflow kernels use `Noc`, `TensorAccessor` endpoints and the `CircularBuffer` (8 readers) or `DataflowBuffer` (4 kernels) wrappers. Compute kernels use `DataflowBuffer` for every FIFO operation.
  - **Donor `fill_tile_utils.hpp` (binary_ng, in-family; blob `4061a08975c`, the #62084 version): compliant.**

    | Helper | Lines | Selected when the operand is | Data-movement idioms |
    |---|---|---|---|
    | `fill_tile_with_first_row` | `:374-407` | FLOAT32 / INT32 / UINT32, row broadcast | `Noc` + `PrecomposedUnicastEndpoint` + `CoreLocalMem` (`:384-406`) |
    | `fill_tile_with_first_row_bfloat16` / `_bfp8` / `_bfp4` | `:349` / `:284` / `:170` | BF16 / BFLOAT8_B / BFLOAT4_B | local `volatile tt_l1_ptr` only |
    | `fill_tile_with_first_column` / `_bfloat16` / `_bfp8` / `_bfp4` | `:436` / `:411` / `:232` / `:131` | column broadcast, by dtype | local `volatile tt_l1_ptr` only |
    | `fill_tile_with_first_element<T>` / `_bfloat16` / `_bfp8` / `_bfp4` | `:50` / `:33` / `:67` / `:96` | scalar broadcast, by dtype | local `volatile tt_l1_ptr` only |

    - The three `get_noc_addr(fN, noc.get_noc_id())` calls left in the file (`:390`, `:396`, `:404`) feed `PrecomposedUnicastEndpoint`'s `.noc_addr`. That struct is documented for "a TensorAccessor or get_noc_addr result" (`tt_metal/hw/inc/api/dataflow/endpoints.h:19-25`). This is the Device 2.0 form, not a holdover.
    - The local `volatile tt_l1_ptr` form in the 11 other helpers is recorded as the user's ruling of 2026-10-22. The CB/DFB whitelist marks the same pattern as allowed (`ai/shared/cb_dfb_api_whitelist.md:31`, example at `:41-44`).
    - Ternary reaches the helpers through the `FILL_TILE_WITH_FIRST_*` defines (`ternary_op_utils.cpp:481-585`).
  - **Compute-side donors: clean.** binary_ng's two compute headers define 30 macros and functions. None is used by the five ternary kernels that include them. The positive control is binary_ng's own `eltwise_binary_sfpu*.cpp`, which use `PREPROCESS`. `kernel_lib/eltwise` has no Device 1.0 call (same sweep, 0 hits).

- **Feature compatibility: GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No type, no `global_circular_buffer` field, no `remote_index`, no `remote_cb` / `global_cb`. The four `UpdateDynamicCircularBufferAddress(program, cb->id(), *buffer)` calls (`ternary_program_factory.cpp:1640`, `:1642`, `:1644`, `:1646`) are the `Buffer&` form, which the guard excludes. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Never set. CBs are borrowed with `.buffer = <tensor>.buffer()` only (`:1079`, `:1097`, `:1110`, `:1122`, `:1135`, `:1182`). |
  | GlobalSemaphore | N/A | No semaphore of any kind, host or kernel. |

- **CB endpoints (GATE-free):** every CB has a port-time disposition. There is no dead CB, no multi-binding, and no semaphore. The census is under *Port-work summary*.
- **Offset base pointers: GREEN.** Every address RTA is a clean base with no folded offset.
  - Host: reader slot 0 `predicate_tensor.buffer()->address()` (`ternary_program_factory.cpp:500` TTS/TST, `:864` TTT); reader slot 1, the second tensor operand (`:501` / `:865`); reader slot 2, the false tensor (TTT `:866`; a plain `0u` for TTS/TST, `:502`); writer slot 0 `output.buffer()->address()` (`:904`). The miss path re-wraps the same slots as `Buffer*` (`:713-722`).
  - Kernels: all 28 accessor bases are bare RTA reads (`get_arg_val<uint32_t>(0|1|2)` in the readers, `(0)` in the writer), sometimes renamed (`src_addr = src0_addr`). There is no arithmetic on any address variable.
  - The op is not in the `2026-07-19_offset_base_pointers.md` tables. Outcome: no fold, not listed.
- **TensorAccessor 3rd argument: N/A.** None of the 28 `TensorAccessor(args, addr)` constructions passes a page-size argument. All 28 were listed verbatim; every one is closed on its own line. The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table.

## Relaxation analysis — validity checks

Run against HEAD. The doc's own citations are re-resolved further down.

| # | Check, as the doc states it | What HEAD shows | Result |
|---|---|---|---|
| 1 | The key pins `tensor_layout` in all four slots: "no two tensors differing in dtype, page config (including `Tile`), memory config or `Alignment` may share a cache entry". | **Inputs:** `dtype`, `memory_config`, `hw()`, alignment and `tile()` per slot (`ternary_device_operation.cpp:614-628`). **Output:** `data_type()`, alignment and `tile()` (`:634-636`). The output's memory config arrives through `args`: `to_hash()` hashes `memory_config` (`:266`). That value is a preallocated output's own config (`:189-192`) or the resolved config the computed output is built from (`:194-256`, `:476-539`). **Layout is not a hash term in any slot.** Check 2 pins it, on every dispatch, as the doc says. | **PASS** |
| 2 | "The op is TILE-only. Every tensor slot is checked on its own, against no condition." | `require_tile` (`:380-387`) is applied to `input_a` (`:388`), to `input_b` / `input_c` when present (`:389-394`), and to a preallocated output (`:406`). Nothing else conditions them. There is no `validate_on_program_cache_hit` in the op. On a hit, the framework runs `validate_on_program_cache_miss` (`ttnn/api/ttnn/device_operation.hpp:284-288`; adapter `mesh_device_operation_adapter.hpp:228-233`). A computed output is TILE (`:482-484`). | **PASS** |
| 3 | `has_sharding` cannot flip on a cache hit. | `const bool has_sharding = shard_volumes.has_value()` (`ternary_program_factory.cpp:1053`), and `shard_volumes` is hashed (`ternary_device_operation.cpp:629`). Both come from `get_shard_volumes` over the same specs. The factory's `output.tensor_spec()` is `compute_output_specs`'s result, or the preallocated spec that it returns verbatim (`:478-480`). | **PASS** |
| 4 | The sharded distribution geometry is hashed for every slot. | `sharded_tensor_shape_in_pages` for `input_a`, `input_b`, `input_c` and the output (`:630-633`), plus `shard_volumes` (`:629`). Grid, shard shape and orientation ride in each slot's hashed `memory_config`. | **PASS** |
| 5 | Exactly one factory. | `std::variant<TernaryProgramFactory>` (`ternary_device_operation.hpp:67`). No `select_program_factory`. | **PASS** |

**The rest of the doc holds on HEAD.**
- §2: `match_page_size` "has no effect" outside interleaved row-major (`tensor_spec_relaxations.hpp:98-101`), and check 2 makes row-major unreachable.
- §3: `hw()` reads only `s[-2]` and `s[-1]` (`:604-607`). The rationale comment is still at `ternary_program_factory.cpp:685-695`. P1 and P2 reproduce both pairs on HEAD.
- Row 3's triggers include #61721's `has_empty_shards` (`ternary_op_utils.cpp:36-56`, used at `:90` and `:155-156`).
- §4: ND-sharded slots take the non-native path, because `is_uneven` returns true without a 2D shard spec (`:22-25`). The 1D distribution strategies take the whole grid as the core list (`buffer_distribution_spec.cpp:168-174`). The empty-optional dereference that §4 describes is still reachable (*Misc anomalies*).
- §5: no failed claim is reintroduced. The TILE rule has no condition. The SNAKE_BETA TILE check (`:430-432`) is now redundant with it.

**Verdict: VALID — `dynamic`.** The three prerequisites are on `main` (#62084). The analysis is therefore no longer conditional. The declaration may be carried into the brief as the doc states it: `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}` on `predicate`, `value_true`, `value_false` and the output, unconditionally. No `match_page_size`, no `match_padded_shape_only`.

### The doc's citations on HEAD

The doc's line numbers belong to `21d0f3c7ba9`. Between that tree and HEAD only `ternary_device_operation.cpp` differs, in three places: review trimmed the comments at the doc tree's `:187-191`, `:383-389` and `:644-649`, and #62567 rewrote `:306`. Every other cited file is byte-identical.

| Doc citation | What it points at | HEAD |
|---|---|---|
| `:187-195` (prerequisite 2) | a preallocated output's memory config wins | `:187-192` |
| `:383-404` (prerequisite 1) | the per-slot TILE rule (doc `:383-389` was a comment, now removed) | `:380-394` |
| `:398-404`, `:416` (prerequisite 1, check 2) | `require_tile` on the inputs; on the output | `:388-394`, `:406` |
| `:492-494` (check 2) | `compute_output_specs` takes `input_a`'s layout when sharded | `:482-484` |
| `:624-638` (check 1) | per-input dtype, memory config, H/W, alignment, tile | `:614-628` |
| `:644-652` (prerequisite 3, check 1) | output dtype / alignment / tile (doc `:644-649` was a comment, now removed) | `:634-636` |
| `:639` (check 3) | `shard_volumes` | `:629` |
| `:640-643` (check 4, row 3) | `sharded_tensor_shape_in_pages` for all four slots | `:630-633` |
| `:614-617` (§3) | `hw()` | `:604-607` |
| `:560` (row 3) | "Sharded common_runtime_args bake tensor_shape_in_pages …" | `:550` |
| `:198-199`, `:206-218` (§4) | the empty 2D shard-spec dereferences | `:195-196`, `:203-215` |
| `:420`, `:440-442` (§5) | the SNAKE_BETA block and its TILE check | `:410`, `:430-432` |
| unchanged files | `ternary_device_operation.hpp:67`; `ternary_program_factory.cpp:206-212`, `:319-326`, `:685-695`, `:688-694`, `:1053`, `:1279-1333`, `:1543-1604`; `ternary_op_utils.cpp:59-161`, `:961`; `mesh_device_operation_adapter.hpp:228-233`; `buffer_distribution_spec.cpp:168-174`; `tensor_spec_relaxations.hpp:98-101` | same lines, content confirmed |

### Suggested doc text (not applied)

None of these changes the declaration. They keep the doc from describing the pre-merge state.

1. **Prerequisites, first paragraph.** Replace "The derivation was run against a tree carrying three fixes that are **not on `main`**; they are the base commit of `anasuya/metal2_eltwise_ternary` (`21d0f3c7ba9`, *Pre-port fixes*)." with:
   > The derivation depends on three fixes, merged to `main` in #62084 (`c04ac9c6ca9`, 2026-10-28).

   The rest of the paragraph stands.
2. **Line numbers.** Replace the `ternary_device_operation.cpp` citations with the HEAD column above. Alternatively, keep them and add to the provenance note:
   > Re-validated on `main` `3f57b50ff00` (2026-10-30). Between `21d0f3c7ba9` and that commit only `ternary_device_operation.cpp` differs: review trimmed three explanatory comments, and #62567 replaced `buffer() != nullptr` with `is_allocated()`. Citations into that file shift by −3 from line 187, −10 from line 380 and −16 from line 641.
3. **§1, last sentence** ("the sheet's `Known op issues` cell still citing it is stale"). The corrected cell is now cleared. Drop the sentence.

## TTNN ProgramFactory

### Concept
`CustomProgramSpecFactoryConcept`. The ported-from factory has an `override_runtime_arguments`, declared at `device/ternary_device_operation.hpp:59-64` with its body at `device/ternary_program_factory.cpp:1543-1649`.

### Fit
- Single vs multi-program: single. One `ProgramDescriptor` is stamped across the mesh.
- Op-owned device resources: none.
- Tensor-arg matching: **relaxed**, per the validated analysis: `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on every `TensorParameter`. This is the doc-sanctioned exception in `ttnn_factory.md` (*Tensor-arg matching*, source 1). Strict would contradict the shape-blind key (P1, P2). Every accessor's legacy `ArgConfig::RuntimeTensorShape` (`ternary_program_factory.cpp:1279-1333`) is the Device 1.0 spelling of the same choice.
- Legacy-to-Metal-2.0 shape: 1:1 with legacy.

### Custom compute_program_hash
Present at `device/ternary_device_operation.cpp:582-637`, layered over the backdoor `operation_attributes_t::to_hash()` at `:261-271`. `to_hash` leaves out `scalar_input_a` / `scalar_input_b`. Those are re-applied on every hit through the per-core compute args (`ternary_program_factory.cpp:921`, re-applied by the override's apply loop at `:1599-1604`). The key is not looser than the declaration (validity check 1). The porter leaves both functions intact.

### Stop signals
None.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):

  | Binding | TTT | TTS | TST | Kernel use when the slot define is 0 | When it is 1 |
  |---|---|---|---|---|---|
  | `predicate` | reader RTA 0, c_0 | RTA 0, c_0 | RTA 0, c_0 | **Case 1**, e.g. `TensorAccessor(src0_args, src0_addr)` (`ternary_reader_nosubtilebcast_ttt.cpp:64`) | **clean**: c_0 is borrowed (`ternary_program_factory.cpp:1079`); the reader does one bulk `reserve_back` / `push_back` (e.g. `ternary_reader_nosubtilebcast_ttt.cpp:59-61`) |
  | `value_true` | RTA 1, c_1 | RTA 1, c_1 | scalar | **Case 1** | **clean** (c_1, `:1097` TTS / `:1122` TTT) |
  | `value_false` | RTA 2, c_2 | scalar | RTA 1, c_1 | **Case 1** | **clean** (c_2 `:1135` TTT; c_1 `:1110` TST) |
  | `output` | writer RTA 0, c_3 | same | same | **Case 1**, `TensorAccessor(dst_args, dst_addr)` (`ternary_writer_nobcast.cpp:34`) | **clean**: c_3 is borrowed (`:1182`); the writer body is compiled out (`#if !DST_SHARDED`, `:31`) |

  - The slot defines are `SRC_SHARDED_*` = slot sharded ∧ `has_sharding` (`ternary_program_factory.cpp:206-212`) and `DST_SHARDED` = output sharded ∧ `has_sharding` (`:1337`).
  - For TTT, `has_sharding` can be true while the inputs are L1-interleaved and only the output is L1-sharded (`ternary_op_utils.cpp:155-158`). The inputs are then Case 1 and the output is clean. For TTS / TST, native sharding needs a sharded predicate (`:106`, `:110`).
  - When a slot is natively sharded, the legacy factory still writes its address RTA, but the kernel never reads it. In Metal 2.0 that binding is `borrowed_from` only.
  - All address RTAs are delivered today as `Buffer*` bindings (`:713-722`, `:1483-1534`). That keeps them correct on a cache hit. The port replaces them with `TensorParameter` bindings.
- **TensorParameter relaxation:** `dynamic (see analysis)` → `eltwise_ternary.md` (blob `c89ba329e90`), validated. On each `TensorParameter` the variant declares (`predicate`, `value_true`, `value_false`, the output), unconditionally: `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}`. Do not set `match_page_size` or `match_padded_shape_only`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (census per CB, per node; every CB spans `worker_grid`):

  | CB | Allocated when | Touchers | Disposition |
  |---|---|---|---|
  | c_0 (predicate) | always (`ternary_program_factory.cpp:1070-1080`) | reader: FIFO producer. Its `FILL_*` writes go to the slot it reserved. They count as the same toucher. compute: FIFO consumer. | **1P+1C** in every config. `borrowed_from` predicate when `SRC_SHARDED_A=1`. |
  | c_1 (true for TTT / TTS, false for TST) | always (`:1086-1123`) | reader: producer. compute: consumer. | **1P+1C**. Borrowed when `SRC_SHARDED_B=1`. |
  | c_2 (false, TTT only) | TTT (`:1125-1136`) | reader: producer. compute: consumer. | **1P+1C**. Borrowed when `SRC_SHARDED_C=1`. No TTS / TST kernel references c_2. |
  | c_3 (output) | always (`:1171-1183`) | compute: FIFO producer. writer: FIFO consumer, only when `DST_SHARDED=0`. | **1P+1C** when `DST_SHARDED=0`. **Self-loop** on compute when `DST_SHARDED=1` (borrowed). |
  | c_4 / c_5 / c_6 (LLK row-broadcast staging for A / B / C) | TTT ∧ ROW_BCAST, unconditionally (`:1141-1169`) | compute only, as producer and consumer | **Conditional DFB**, below. |

  - **c_4 / c_5 / c_6 are live** (a compute self-loop) only when `is_llk_bcast` holds and that operand is row-broadcast (`BCAST_A` / `BCAST_B` / `BCAST_C` = 1, `:1397-1401`). `is_llk_bcast` requires TTT ∧ ROW_BCAST with BF16 inputs **and** a BF16 output (`:219-234`). The compute kernel is then `ternary_sfpu_row_bcast_ttt.cpp`, or `ternary_addc_ops_fpu_rowbcast.cpp` for ADDCMUL / ADDCDIV.
  - **They are dead** under every other TTT ∧ ROW_BCAST configuration: any non-BF16 input or output, or an operand that is not itself row-broadcast. In those configurations the reader fills the row itself (`ternary_reader_rowbcast_ttt.cpp:174-182`) and the compute kernel never names c_4..c_6.
  - Port move: host-conditional binding plus a preprocessor gate (`port_patterns.md`, *Conditional / optional resource bindings*). **Do not drop them.**
  - No multi-binding: no CB has a third toucher, and no CB has two kernels locked to one FIFO role.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** no borrowed kernel files and no `_metal2` fork anywhere in the op or its donor directories. No bound kernel is shared. The shared-kernel rungs do not apply. `ternary_reader_nobcast_ttt.cpp` is lent to `binary_backward` and not bound by ternary: leave it alone. `fill_tile_utils.hpp` needs no fork, because its helpers take a raw L1 pointer.
- **RTA varargs:** none. Every RTA is read once at a literal index; every CTA at a literal index or at a `constexpr` `next_compile_time_args_offset()`.
- **Override translation** (`ternary_program_factory.cpp:1543-1649`). Per `ttnn_factory.md`, the refresh set is inherited, not decided:
  - The legacy override re-runs `build_per_core_runtime_args` (`:701-960`) and copies a full row into **every** core of the worker grid: reader 27, writer 11, compute 4 words, with zero rows for noop cores (`:784-791`). A core can flip between work and noop across hits (`:697-700`). `UpdateProgramRunArgs` keeps the last value for anything left out. The Metal 2.0 override must therefore keep covering every core.
  - It re-points every tensor-backed CB at the current buffer, by CB index (`:1618-1648`).
  - It deliberately does not refresh the common RTAs that carry the `TensorAccessorArgs` words (`:1555-1570`).
  - In Metal 2.0, the address slots (reader 0-2, writer 0) and the CB re-pointing both become `tensor_args`, one entry per `TensorParameter`, on every dispatch. The other per-core values become `kernel_run_args`.
- **Conditional bindings.** `ternary_addc_ops_fpu_rowbcast.cpp:29-31` constructs the c_4..c_6 `DataflowBuffer`s unconditionally. `ternary_sfpu_row_bcast_ttt.cpp:34-54` is already gated. `ternary_writer_nobcast.cpp:29` constructs `dfb_out` outside `#if !DST_SHARDED` (`:31`). When `DST_SHARDED=1` the writer is not a toucher of c_3. Its token would then not exist.
- **Compile-time CB numbers outside LLK call sites.**
  - `ternary_sfpu_copy_init.hpp:15` hardcodes `tt::CBIndex::c_0` inside `copy_init_shared_with_c0<cb>()`, which compares the JIT-generated `unpack_*` format arrays of two buffers.
  - At file scope, `ternary_sfpu_col_scalar_bcast_ttt.cpp:18-19` and `ternary_addc_ops_sfpu_bcast.cpp:17-18` pass `tt::CBIndex::c_1` / `c_2` into that template. `ternary_sfpu_col_scalar_bcast_ttt.cpp:18` also indexes `unpack_src_format[tt::CBIndex::c_0]` directly.
  - Metal 2.0 assigns DFB device slots as the lowest free slot in creation order (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2182-2202`), not by legacy CB number. The generated arrays are indexed by slot (`tt_metal/jit_build/genfiles.cpp:916-917`).
  - Every one of these must move to the matching `dfb::` token. A leftover `c_0` could silently change which operands share the copy init.
- **DFB handles at LLK call sites.** All ten compute kernels pass `dfb.get_id()` to LLKs. Four build temporary `DataflowBuffer`s from ids inside `process_tile` helpers (`ternary_sfpu_col_scalar_bcast_ttt.cpp:34-37`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:26-28`, `ternary_addc_ops_sfpu_bcast.cpp:34-37`, `ternary_addcmul_int_sfpu_bcast.cpp:25-28`). Pass `dfb::name` directly (`port_patterns.md`, *Pass DFB handles directly to LLKs and kernel-lib helpers*).
- **Already part-modernized.** 4 dataflow kernels and all 10 compute kernels are on `DataflowBuffer`. The 8 fill-helper readers still construct `CircularBuffer` and use `get_tile_size()`. They need the CB→DFB swap.
- **Hardcoded CB ids in one reader.** `tts_tst_reader_row_bcast.cpp:47-48` uses `tt::CBIndex::c_0` / `c_1` and ignores CB-id CTAs 0 and 1. Map them to the same tokens as the other readers.
- **Unused donor includes.** Five compute kernels include binary_ng's compute utils and use none of them (`ternary_sfpu_no_bcast_ttt.cpp:7-8`, `ternary_sfpu_no_bcast_tts_tst.cpp:7-8`, `ternary_sfpu_col_scalar_bcast_ttt.cpp:11-12`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:12-13`, `ternary_sfpu_row_bcast_ttt.cpp:16-17`). `eltwise_utils_sfpu.hpp:9` pulls in `api/dataflow/circular_buffer.h`. Check the kernel-side whitelist before keeping the include lines.
- **Native sharding implies broadcast type NONE.** Native sharding needs identical input shapes (`ternary_op_utils.cpp:62`, `:114-118`). Every broadcast reader therefore compiles with `SRC_SHARDED_*=0` and `has_sharding=0`. Port their sharded branches as-is.
- **Install coverage.** `CMakeLists.txt:16-20` globs `device/kernels/*.cpp` **and** `*.hpp` recursively (#60073 added `*.hpp`). The generated install script lists all 26 kernel files on disk. A `_metal2` fork or a new header beside the originals is installed without a build-file change.

## Team-only

### Out-of-directory coupling & donor shape

**Roll-up: ✓ clean.**

| Op kernel(s) | Donor file | Class | Called | Shape | Status |
|---|---|---|---|---|---|
| `ternary_reader_colbcast_ttt.cpp`, `ternary_reader_scalar_ttt.cpp`, `ternary_reader_rowbcast_ttt.cpp`, `ternary_reader_row_col_bcast_ttt.cpp`, `tst_tts_reader_scalar_bcast.cpp`, `tts_tst_reader_col_bcast.cpp`, `tts_tst_reader_row_bcast.cpp`, `tts_tst_reader_row_col_bcast.cpp` | `eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | 5 (in-family) | the 12 `fill_tile_with_first_{row,column,element}*` helpers, through the `FILL_TILE_WITH_FIRST_*` defines | a raw `uint32_t l1_write_ptr`. The port passes `dfb.get_write_ptr()`. | ✓ |
| `ternary_sfpu_no_bcast_ttt.cpp`, `ternary_sfpu_no_bcast_tts_tst.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`, `ternary_sfpu_row_bcast_ttt.cpp` | binary_ng `compute/eltwise_utils_common.hpp`, `eltwise_utils_sfpu.hpp` | 5 (in-family) | nothing (unused includes) | — | ✓ |
| `ternary_addcmul_int_sfpu.cpp` | `kernel_lib/eltwise/api/chain.hpp`, `generators/fill.hpp`, `binary/sfpu/int.hpp` | 2 (shared library) | `eltwise_chain`, `CopyTile`, `FillInt`, `MulIntBinary`, `AddIntBinary`, `PackTile` | The CB id goes into a template position (`ckl::input(id, …)`). The buffer is bound as a non-const `DataflowBuffer&` (`InputStream`, `chain.inl:1008`). The kernel already passes named locals (#60306). | ✓ |

- **Borrowed kernel files:** none. **Lent:** `ternary_reader_nobcast_ttt.cpp` → `eltwise/binary_backward`.
- **`fill_tile_utils.hpp` consumers** (kernel files that include it): binary_ng 10, ternary 8, batch_norm 4.

### Relaxation candidates (from the custom hash)
Superseded by the validated analysis. One hash fact for the ops team: layout is not a hash term in any slot, and validation pins it on every dispatch. If a lighter `validate_on_program_cache_hit` is ever added, it must keep the per-slot TILE checks, or the key must start hashing each slot's layout.

### TTNN factory analysis
- Current concept: `descriptor`. Target: `CustomProgramSpecFactoryConcept` (override at `ternary_program_factory.cpp:1543-1649`).
- Op-owned tensors: none. MeshWorkload need: none.
- Custom hash: `ternary_device_operation.cpp:582-637`, plus `to_hash()` at `:261-271`.
- `get_dynamic_runtime_args`: absent. Pybound `create_descriptor`: absent. No other pybound internals.
- Gate conjuncts: the relaxation cell clears by value (`dynamic (see analysis)`), and the doc behind it validates on HEAD. `get_dynamic_runtime_args` is absent. The op is single-program.

## Changes since prev4

| prev4 finding or condition | On HEAD |
|---|---|
| Changes 1–3 (TILE-only, output memory-config precedence, output hash terms) uncommitted | On `main` (#62084). prev4's leftover always-true conjunct in `resolve_mem_config_actual` is gone. The "roughly 94%" comment is gone. |
| `fill_tile_utils.hpp` Device 2.0 fix uncommitted | On `main` (#62084), blob `4061a08975c` |
| Rebase onto `main` pending | None needed: the branch is `main` `3f57b50ff00` plus docs |
| Per-slot test gap (`input_b` / `input_c` checks unreached) | Closed: `test_ternary_rejects_row_major_operand` (4 cases) is in #62084 |
| `BCAST_LLK` / c_4..c_6 live on all-BF16 inputs | Also requires a BF16 output (#60282, `ternary_program_factory.cpp:219-234`) |
| Pack format not restored after the LLK row broadcast: live | Trigger closed by the BF16-output requirement. The missing restore remains, latent. |
| Divide-by-zero in `ternary_addcmul_int_sfpu_bcast.cpp`: reachable | Unreachable from the front ends, latent (*Misc anomalies*) |
| `kernel_lib` call shape: ids only | Kernel-scope `DataflowBuffer`s by reference (#60306) |
| A new kernel-side `.hpp` would not be installed | The glob now covers `*.hpp` (#60073) |
| New since prev4 | `ternary_sfpu_copy_init.hpp` and the `init_once` constants (#60073); `has_empty_shards` as a new non-native trigger (#61721); `auto&` aliases in `ternary_addc_ops_fpu_rowbcast.cpp:34-48` (#52360); `is_allocated()` (#62567) |

## Misc anomalies  *(team-only, non-gating; route to the ops team)*

- **ND-sharded inputs without a preallocated output dereference an empty optional.** Code-traced; the analysis doc §4 records it too. `resolve_mem_config_actual` calls `compute_mem_config_actual(input_a, …)` (`ternary_device_operation.cpp:195-196`, which dereferences `shard_spec()` at `ternary_op_utils.cpp:961`) and `shard_spec()->num_cores()` (`ternary_device_operation.cpp:203-215`). It does both whenever a sharded input has no 2D shard spec and no `memory_config` is passed. That is undefined behaviour.
- **Latent: pack format not restored after the LLK row broadcast.** `ternary_sfpu_row_bcast_ttt.cpp:66`, `:87`, `:108` and `ternary_addc_ops_fpu_rowbcast.cpp:60`, `:77`, `:94` retarget the packer to c_4..c_6, then pack into c_3 (`:153`, `:132`) without restoring c_3's format. `is_llk_bcast` now requires BF16 everywhere. The formats then agree, and the gap is unreachable.
- **Latent: divide-by-zero in `ternary_addcmul_int_sfpu_bcast.cpp:131-132`.** It divides by RTA 1 (`tile_freq`), which is 0 for ROW_BCAST (`ternary_program_factory.cpp:939`). The kernel is reached only through `ComputeBcastAddcOp` (SCALAR / COL / ROW_COL; `tile_freq` ≥ 1). The ROW_BCAST route would need `is_llk_bcast` (BF16 output) together with the int override (INT32 / UINT32 `dtype`). `addcmul` sets `dtype` to the output tensor's dtype (`ternary.cpp:40-43`, `:379`). The two therefore cannot both hold. The only guard is that front-end invariant. The noop-core zero rows are safe: each broadcast compute kernel returns on `num_tiles == 0` before dividing.
- **Stale comments.**
  - `ternary_device_operation.cpp:580`: "re-applied via get_dynamic_runtime_args on cache hit". The hook is gone.
  - `ternary_program_factory.cpp:688`: "coarsens each input tensor to its padded VOLUME". The key pins logical H/W.
  - `ternary_program_factory.cpp:1563-1567`: "(padded volume, H, W) -- all three hashed" and "ND sharding cannot reach this path". Padded volume is not hashed, and ND-sharded tensors do reach the sharded-accessor path. The conclusion still holds, because the key hashes `sharded_tensor_shape_in_pages` for every slot.
  - `test_ternary.py:450`, `:507`: docstrings name `TernaryDeviceOperation::get_dynamic_runtime_args()`.
- **Dead code.**
  - `ternary_device_operation.cpp:228`: `else if (memory_config.has_value())` is always true after `:199`.
  - `:430-432`: the SNAKE_BETA TILE check is redundant with `:388-394`.
  - `get_kernel_file_path`'s `ComputeRowBcastAddcOp` → `ternary_addc_ops_sfpu.cpp` branch (`ternary_op_utils.cpp:417-419`) is unreachable. `is_llk_bcast` implies all-BF16 inputs, which makes `is_fpu` true.
- **Unused RTA reads and defines.**
  - All six TTS/TST readers read RTA 2 (`src2_addr`, always `0u`) and never use it.
  - The five TTS/TST broadcast readers also read RTAs 19 and 26 without using them.
  - Writer slot 10 is padding.
  - The host emits `FILL_WITH_VALUE*` reader defines (`ternary_op_utils.cpp:481-585`) that no reader uses.
  - Unused locals: `scalar_is_true` (`ternary_sfpu_col_scalar_bcast_tts_tst.cpp:130`) and `scalar_is_not_1` (`ternary_addc_ops_fpu_rowbcast.cpp:21`).
- **CB-id CTAs ignored** by `tts_tst_reader_row_bcast.cpp:47-48`. Harmless today.
- **Latent hang in an unreachable branch.** `tst_tts_reader_outer_bcast.cpp:51-58` has no bulk `reserve_back` / `push_back` for a natively sharded input. OUTER_BCAST cannot be natively sharded.
- **Dead files:** `compute/ternary_addc_ops_fpu.cpp` and `compute/ternary_addc_ops_fpu_bcast.cpp` are bound nowhere.

## Questions for the user

None blocking.

1. **The relaxation doc's text.** Its prerequisites paragraph still says the fixes are "not on `main`", and its `ternary_device_operation.cpp` line numbers are the pre-merge tree's. Suggested text is under *Suggested doc text*. Should it be applied? It is your untracked file, and I have left it alone.

## Recipe notes

- **The in-place recipe is two commits behind upstream.** Upstream `78b23d2db4d` changes *TTNN porting shape*: the target concept is read from a new `Porting Target` column, an `Execution Model` column is added, and `Is able to port?` = `yes (with PD step)` is RED. `c41f7f4ef0a` renames `DFBAccessor` → `DFBBindingToken` in the port docs and drops the `.id` form. This audit followed the in-place doc as instructed. The verdict does not change: the cell is plain `yes`, and the concept is `descriptor` (not per-coord). The derived target matches what that column should hold for an op with an override. The `Porting Target` and `Execution Model` cells were not supplied.
- **Analysis docs and line numbers.** This analysis's citations went stale on merge, because review trimmed comments in one file. Its provenance note named the tree, which made re-resolution cheap. A line in *TensorParameter relaxations* telling auditors to re-resolve a doc's citations by content, and analysis authors to name the commit they cite, would make that the norm.
- **Compile-time CB numbers outside LLK call sites have no audit signal.** The recipe covers LLK call sites (pass `dfb::name`), but not file-scope or template uses such as `unpack_src_format[tt::CBIndex::c_0]` (*Heads-ups*). These keep compiling if left alone and can change behaviour silently, because DFB slots follow creation order.
- **Still open: is the relaxation subject a gate?** The scoping rule lists it among the seven subjects that "never gate" (`metal2_audit.md:74`). Its own section ends "Finding role: GATE … on a non-clearing value" (`:258`). It did not matter here.
- **Local `volatile tt_l1_ptr` access under Device 2.0.** The user has ruled it compliant (2026-10-22). The CB/DFB whitelist allows it. The Device 2.0 migration guide still calls the form legacy. A sentence in the audit's Device 2.0 section would spare the next auditor the question.
- **Still open: `PrecomposedUnicastEndpoint` is missing from the Device 2.0 migration guide** (0 mentions). `endpoints.h:19-25` documents it.
- **The status-summary template has a `Feature Support — Variadic-CTA` row with no matching Appendix A entry** (`metal2_audit.md:575`). I filled it "Ok. None."
