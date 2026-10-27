# Metal 2.0 Audit Findings — `eltwise/ternary`

- **`TernaryDeviceOperation`** (`device/ternary_device_operation.hpp:19`). It has one factory (`program_factory_t = std::variant<TernaryProgramFactory>`, `:67`):
  - `TernaryProgramFactory`: declared at `:49-65`. `create_descriptor` is at `device/ternary_program_factory.cpp:997` and `override_runtime_arguments` at `:1543-1649`.
- The one factory serves every configuration: variants TTT / TTS / TST, every broadcast type, and every op type (WHERE, LERP, ADDCMUL, ADDCDIV, SNAKE_BETA, MAC). Kernels are chosen by `TernaryKernelConfig` (`device/ternary_op_utils.cpp`), `get_kernel_file_path` and `override_addcmul_compute_kernel` (`:401-410`).
- **Reached only through `ttnn::prim::ternary`** (four overloads, `ternary_device_operation.cpp:666-842`). All six call sites are in `ternary.cpp`. Nothing else calls `prim::ternary` or `TernaryDeviceOperation`, and `prim::ternary` has no Python binding.
  - `where` TTS/TST goes to binary_ng (`ternary.cpp:161`, `:185`), and TSS to `where_tss` (`:203`).
- **Kernels the factory binds:** the same 22 as before, all owned by this op. 12 dataflow kernels under `device/kernels/dataflow/` (byte-identical to the previously audited tree) and 10 compute kernels under `device/kernels/compute/` (7 changed upstream). Two compute kernels now include a new op-local header, `compute/ternary_sfpu_copy_init.hpp` (#60073).
- **Donor code:** `eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` (8 readers); binary_ng's `compute/eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp` (5 compute kernels, included, unused); `ttnn/cpp/ttnn/kernel_lib/eltwise` (`ternary_addcmul_int_sfpu.cpp`).
- **Not bound, not audited:** `compute/ternary_addc_ops_fpu.cpp`, `compute/ternary_addc_ops_fpu_bcast.cpp` (bound nowhere), and `dataflow/ternary_reader_nobcast_ttt.cpp` (bound by `eltwise/binary_backward`; leave it alone).
- Earlier audits beside this file: `METAL2_PREPORT_AUDIT.prev4.md` (GREEN conditional, pre-rebase, 2026-10-22, blob `d832625195b`), `.prev3.md` (RED, 2026-10-16), `.prev2.md` (RED, 2026-10-15), `.prev.md` (RED). This report calls `.prev4.md` "the previous audit". Its line numbers, rebase table, *Differences on main*, `[main differs]` markers and binary_ng change-4 description describe the pre-rebase tree and are void here.

**Scope:** TTNN op, Gen1 (WH/BH) target. Within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `917314f85d3 2026-10-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. The `metal_2.0/` tree is identical to the one the previous audit used (`git diff a82c72bc9bd 917314f85d3` is empty).

**Code audited:** the working tree of `anasuya/metal2_eltwise_ternary` @ `917314f85d3`, rebased onto `origin/main` `26e6c5e2000b` (0 behind), on 2026-10-22.
- **Five files carry uncommitted changes. They are staged, not committed.** Nothing is unstaged on top of them. Working-tree blobs:
  - `ternary/device/ternary_device_operation.cpp` `6eadbf9140d` (HEAD `b8beded5007`, which is also the blob the previous two audits called "main");
  - `binary_ng/device/binary_ng_device_operation.cpp` `0c21ae206f6` (HEAD `533530c811d`);
  - `binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` `4061a08975c` (HEAD `5e9bd45d0db`, unchanged on main);
  - `tests/.../test_ternary.py` `c6d9d395e78` and `tests/.../test_binary_ng_program_cache.py` `bfbfe6ed152`.
- The rebase moved the base 478 commits, from `d388535aee7` to `26e6c5e2000b`. Ten of them touch in-scope files: #59788, #60282, #60306, #60507 (known to the previous audit) and #59392, #52360, #60073, #60763, #57797, #61760 (new). The factory, the op utils, seven bound compute kernels, `CMakeLists.txt` and `kernel_lib/eltwise` changed. The 12 dataflow kernels and the donor did not.

**Readiness sheet:** not fetched. The cells are the ones the invoker supplied, with three local overrides (`TensorParameter relaxation` = `dynamic (see analysis)`, `Known op issues` = empty, `Is able to port?` = `yes`). The user confirmed on 2026-10-22, before the rebase, that they still stand. For `Known op issues`, the user chose the text proposed in *C*. The live cell is the sheet owner's to update.

**Relaxation analysis doc:** `analyses/relaxations/eltwise_ternary.md`, untracked. This audit validated blob `b6a742628fb` (renumbered for the rebase by its owner). At the user's request the fixes in *B* were then applied (new blob `bc9e80c0ef2`). They correct citations and update three sentences with this audit's own findings.

**Silicon evidence:** one Wormhole n150.
- `build_Release/lib/_ttnncpp.so` was built at 10:48:19 UTC, after the rebase (10:33:54) and the last source edit (10:35:59). It contains strings from main commits dated 2026-10-21 (#61612, #60503, #60696) and the new `require_tile` message, and not HEAD's old TILE message.
- JIT kernels were compiled from the working tree into a fresh kernel cache. The pre-rebase cache was not reused.
- 176 targeted test cases were run, all pass. 58 probe calls ran from scripts that are not checked in. The calls that could crash or hang (the former divide-by-zero trigger and the ND undefined-behaviour paths) ran in separate processes, under `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/ternary/` |
| **Overall** | **GREEN on the working tree, conditional on uncommitted code.** Every gate-bearing subject clears, and the relaxation analysis validates (all five checks pass; verdict `dynamic`). The analysis is void without changes 1–3. Device 2.0 clears only with the `fill_tile_utils.hpp` fix, which is uncommitted and absent from main. On HEAD (= `origin/main` for the op) it is RED. |
| **DOps / Factories** | `TernaryDeviceOperation` → `TernaryProgramFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **Yes, conditional on the donor fix landing.** The 22 own kernels and the new header are compliant. `kernel_lib/eltwise` is clean. `fill_tile_with_first_row` is fixed in the working tree only; main keeps five raw calls (`fill_tile_utils.hpp:382-387`). The 11 other helpers use local `volatile tt_l1_ptr` access, which the user ruled compliant; the evidence is validated in *D*. |
| *Prereqs*: Cross-op escapes | Ok. Raw `uint32_t` L1 pointers into `fill_tile_utils.hpp`. `DataflowBuffer` objects into `kernel_lib` (#60306). binary_ng compute utils included but unused. |
| *Feature Support*: overall | GREEN. No Appendix A entry fires. |
| *Feature Support*: Variadic-CTA | Ok. None. |
| *TTNN Readiness*: `Is able to port?` (the gate) | **Yes** (local override, user-confirmed; the live cell is `no`). The primary-column cross-check is clean. |
| *TTNN Readiness*: Concept (current) | `descriptor` (`create_descriptor` returns `ProgramDescriptor`, `ternary_program_factory.cpp:997`). |
| *TTNN Readiness*: Secretly SPMD | N/A |
| *TTNN Readiness*: Custom hash | **Yes.** `compute_program_hash` at `ternary_device_operation.cpp:592-653`, layered over `to_hash()` at `:264-274`. Change 3 adds `:644-652`. |
| *TTNN Readiness*: `get_dynamic_runtime_args` | No. Only a stale comment (`ternary_device_operation.cpp:590`). |
| *TTNN Readiness*: `override_runtime_arguments` | **Yes** (`ternary_device_operation.hpp:59-64`, `ternary_program_factory.cpp:1543-1649`). It selects `CustomProgramSpecFactoryConcept`. |
| *TTNN Readiness*: Pybind `create_descriptor` | No. |
| *TTNN Readiness*: Op-owned tensors | No |
| *TTNN Readiness*: Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work*: Offset base pointer | none |
| *Port work*: Tensor bindings | `predicate`, `value_true`, `value_false`, `output`: **Case 1** when the slot's `SRC_SHARDED_*` / `DST_SHARDED` define is 0, **clean** (borrowed DFB) when 1. No Case 2. Every slot is TILE. |
| *TTNN Readiness*: TensorParameter relaxation | `dynamic (see analysis)`. **Verdict: VALID.** `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on all four `TensorParameter`s. Four citations need fixing, and two provenance sentences need updating (*B*). None changes the declaration. |
| *Port work*: TensorAccessor 3rd arg | none (28 two-argument constructions in the 12 bound dataflow kernels). |
| *Port work*: CB endpoints | 1P+1C (c_0, c_1, c_2; c_3 when `DST_SHARDED=0`) · self-loop (c_3 when `DST_SHARDED=1`) · **conditional DFB** (c_4, c_5, c_6) |

## Result

**GREEN on the working tree, conditional on uncommitted code. RED on HEAD.**

Answers to the five priorities, in order:

- **A. binary_ng after #59392: the conflict resolution is correct.** In both overloads the dropped output entry was redundant: once an output is supplied, `mem_config_actual` is a copy of its config and nothing reassigns it before the gate. The error path is complete: the prim gate and binary_ng's launch-time check are exact complements, so every call runs the output shape check exactly once. Both rewritten comments are accurate. All 44 program-cache tests, 57 cases of the moved-error-path test and 13 silicon probes agree.
- **B. The relaxation analysis validates.** All five checks pass on this tree, on the code and on silicon. Four citations are wrong: two off-by-one renumberings (`:561` should be `:560`, `:439-441` should be `:440-442`) and two Row 3 ranges that were not renumbered (`ternary_program_factory.cpp:316-323` → `:319-326`, `ternary_op_utils.cpp:33-134` → `:37-138`). The provenance paragraph holds in substance. Two of its sentences are out of date: ND-sharded slots are now probed, and #60282 closes triggers rather than fixing hazards. §5 is not reintroduced anywhere. The §2 stop-condition bullet stays deliberately unresolved, and no ternary claim depends on it.
- **C. #60282 closes both triggers on this tree.** Silicon confirms both: the former pack-format call is exact (it was 4085/4096 wrong), and the former divide-by-zero call completes. Both kernel hazards remain, kept unreachable only by a host-side dtype gate. The same probe found a live bug: a float-input `addcmul` with an INT32 output is accepted and returns garbage (4096/4096 wrong). A second live bug: an ND-sharded `input_a` without a preallocated output dereferences an empty 2D shard spec. Proposed `Known op issues` text is in *C*.
- **D. The Device 2.0 ruling holds and is recorded below.** The whitelist marks the exact pattern allowed, #47157 kept the same local store loop, and the 11 helpers do local stores only. One wording point: changing the cast to `CoreLocalMem` would not be port work either; the port's kernel-side whitelist admits no such change.
- **E. `METAL2_PORT_BRIEF.md` is regenerated** against this tree, at the user's request (*E*).

**Resolved since the previous audit, verified rather than assumed:**
- **Per-slot coverage gap: closed.** `test_ternary_rejects_row_major_operand` (4 cases) passes on silicon, as do the equivalent probes P6, B1, B2 and B3. Each is rejected at `input_b` or `input_c`, and each of the other rejection cases stops at `input_a` or the output. Your deletion experiment is consistent with that split; I did not repeat it.
- **The 94% figure: corrected.** It is fixed in the code comment (`ternary_device_operation.cpp:386`), both test comments and the docstring, and the docstring now says P5 was exact. (The doc says 96.6–99.4%.)
- **Formatters: clean.** black passes on both test files. clang-format 19.1.4 reports nothing on the three C++ files.
- **Leftovers: all resolved now.** The binary_ng whitespace hunk and its stale comment are gone with the rebase. The ternary one, an always-true `&& !optional_output_tensor.has_value()` at `ternary_device_operation.cpp:202`, was dropped at the user's request. That edit is **unstaged**, on top of the staged change. The incremental build is clean, and 14 ternary + 58 `test_where.py` cases pass on the rebuilt library.

**Portable subset:** the whole op, once the conditions are met. There is one factory, and the declaration is unconditional.

## Evidence basis

| Verdict | Silicon | Code reading only |
|---|---|---|
| TILE-only rule, every slot, every front end | P1–P7, B1–B4, E1–E4; 10 rejection test cases | — |
| TILE checks run on cache hits | C1, C2 rejected; C3 hits afterwards | that C1 / C2 are hits (hash terms traced equal; the framework default traced) |
| Validity checks 1, 2 | as above, plus 3 cache-key tests | the per-term trace |
| Validity check 3 (`has_sharding` cannot flip) | D3, D5 exercise both values, miss and hit | the identity itself |
| Validity check 4 (geometry hashed) | D4 (2D) and N1–N3 (true ND): leading-dims pairs share one entry, all exact | the citations |
| Validity check 5 (one factory) | — | yes |
| §3 cache-sharing pairs | D1, D2 | — |
| binary_ng redundancy and error path | A1–A12b; 44 + 57 + 2 test cases | the "nothing reassigns it" trace |
| #60282 pack-format trigger closed | K1 exact, K2–K4 exact | that the kernels still lack the restore |
| #60282 divide-by-zero trigger closed | the former trigger completes (no hang) | the mutual-exclusion trace; the latent kernel division |
| float-input / int-output `addcmul` returns garbage | 4096/4096 wrong | why (int kernel from the output dtype) |
| ND `input_a` without a preallocated output | `TT_FATAL` at `tensor_layout.cpp:171` with no `memory_config`; benign with `memory_config=DRAM` | that both are undefined behaviour (empty-optional dereference) |
| Device 2.0 (own kernels, donor, header, `kernel_lib`) | F1 (fixed helper, exact) | the sweeps (positive control: main's donor) |
| Feature compatibility, offset bases, 3rd argument, CB endpoints, RTA varargs, coupling | — | yes (re-derived on this tree) |
| Rebase dry run | — | not needed: 0 behind |

## A. binary_ng after #59392

**What #59392 did** (`22898d6c823`, 2026-10-20). It moved `prim::binary_ng`'s output-spec and shard-volume work behind a sharded gate, `any_memory_config_sharded()` (`binary_ng_utils.hpp`, null entries skipped), in both overloads. Upstream's gate lists the preallocated output's config as its own entry, with the comment "mem_config_actual falls back to it only when no explicit memory_config was given". Upstream lets the `memory_config` argument win, which makes that entry necessary there.

**The redundancy argument holds on both overloads.**
- **Tensor-tensor overload.** When an output is supplied, `mem_config_actual = output_tensor->memory_config()` (`binary_ng_device_operation.cpp:696-698`), and the other two branches (`:699-750`) are skipped. From there to the gate at `:792-793` the value is only read (`:758`, `:763`). `&mem_config_actual` therefore has the same `is_sharded()` as the dropped entry on every call.
- **Scalar overload.** `mem_config_actual` is set once, at `:858-859`, and only read until the gate at `:904` (`:872`, `:876`).
- With no output supplied, the dropped entry was `nullptr`, which `any_memory_config_sharded` skips. The predicate is therefore unchanged in every case.

**The error path is complete.** binary_ng's launch-time check is in `validate_on_program_cache_hit` (`:431-437`), which the miss validation also calls (`:378`). It runs the shape check (`compute_output_specs`, `:543-547`) when `!(a ∨ b ∨ attributes.memory_config sharded)`, and `attributes.memory_config` is `mem_config_actual` (`:758`, `:872`).
- In both overloads, the prim gate and that condition are exact complements. Every call runs the check exactly once: inside `prim::binary_ng` when something is sharded, at launch otherwise.
- One case moves relative to upstream, by design of the precedence fix: a sharded `memory_config` with an interleaved preallocated output. Upstream checked it inside `prim::binary_ng`. This tree checks it at launch, with the same message (probes A9, A11).

**Silicon.** Inputs `[1,1,64,64]` bf16 TILE in DRAM; "sharded" is L1 height-sharded, shard `(32, 64)` on 2 cores.

| Probe | Call | Result |
|---|---|---|
| A1 / A2 | `add` TT; `memory_config=DRAM` + sharded output / `memory_config=sharded` + DRAM output | correct (within 1 bf16 ULP); one entry each |
| A3 / A4 | the same, scalar overload (`add(a, 2.0, …)`) | correct; one entry each |
| A5 / A6 | `where` TTS / TST with a sharded output | exact |
| A7–A11 | wrong-shaped output: interleaved and sharded, with and without a contradicting `memory_config`, both overloads | each rejected with "Shape of Output tensor … does not match the broadcasted output shape" |
| A12a / A12b | correct call, then the same call with a wrong-shaped output | correct, then rejected |

**Tests.** `test_binary_ng_program_cache.py` 44/44 (its 6 new cases and #59392's). `test_binary_bcast_ext.py::test_binary_opt_output_invalid_bcast` 57/57, which matches the moved message. `test_binary_composite_quasar.py::test_output_tensor_with_extra_leading_dim_is_rejected` 2/2. The `test_where.py` slice 58/58, including `test_where_rejects_mismatched_preallocated_output`.

**The two rewritten comments are accurate** (`:788-791`, `:901-903`). "Its memory config takes precedence, so mem_config_actual already is that config whenever one was supplied" matches `:697` and `:858`. "compute_output_specs returns its spec verbatim" matches `:548`. "Same predicate as the tensor-tensor overload" means the same predicate without the absent operand. That wording is upstream's and still fair.

## B. Relaxation analysis — validity checks

| # | Check | What this tree shows | Result | Evidence |
|---|---|---|---|---|
| 1 | The key pins `tensor_layout` in all four slots. | Inputs: dtype, memory config, alignment and tile per slot (`ternary_device_operation.cpp:624-638`). Output: dtype, alignment, tile (`:650-652`), and the memory config via `args`, which is the output's own when preallocated (`:192-195`). Layout is not hashed. A padded row-major tensor carries TILE's alignment and maps to the TILE entry, and validation rejects it on every dispatch. | **PASS** | silicon C0–C3, 3 cache-key tests; code |
| 2 | TILE-only, per slot, against no condition, on every dispatch. | `require_tile` (`:390-397`) on `input_a` (`:398`), `input_b` / `input_c` when present (`:399-404`), the preallocated output (`:416`). There is no `validate_on_program_cache_hit`, and the adapter runs the miss validation on hits (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:228-233`). A computed output is TILE (`:492-494`). | **PASS** | silicon P1–P7, B1–B4, E1–E4, C1–C2; 10 tests |
| 3 | `has_sharding` cannot flip on a hit. | `has_sharding = shard_volumes.has_value()` (`ternary_program_factory.cpp:1053`), `shard_volumes` hashed (`ternary_device_operation.cpp:639`). | **PASS** | code; D3, D5 |
| 4 | Sharded geometry hashed for every slot. | `sharded_tensor_shape_in_pages` for all four slots (`:640-643`) and each slot's memory config. For 1D (ND) distributions the core list is the whole hashed grid (`buffer_distribution_spec.cpp:159-165`). | **PASS** | code; D4 (2D) and N1–N3 (ND) share entries, exact |
| 5 | One factory. | `std::variant<TernaryProgramFactory>` (`ternary_device_operation.hpp:67`). | **PASS** | code |

**Verdict: VALID — `dynamic`.** The declaration is `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}` on all four slots, with no `match_page_size` and no `match_padded_shape_only`.

**Silicon for the §3 claims** (fresh tensors per call; entries counted across each pair):

| Probe | Calls | Entries | Result |
|---|---|---|---|
| D1 | `[4,1,32,32]` then `[1,4,32,32]`, TILE DRAM | 1 | both exact |
| D2 | `[1,32,64]` then `[1,1,32,64]` | 1 | both exact |
| D3 | row 3 (2D-sharded inputs, DRAM output), twice | 1 | both exact |
| D4 | row 3, `[2,1,32,64]` then `[1,2,32,64]` | 1 | both exact |
| D5 | row 2 (native sharded), twice | 1 | both exact |
| N1–N3 | true ND inputs (`ND_SHARDED`, shard `[1,2,32,64]` on 4 cores, empty 2D shard spec), DRAM output: `[2,2,64,64]` twice, then `[1,4,64,64]` | 1 | all exact |

A caution for anyone repeating N1–N3: an ND shard such as `[1,1,64,64]` is turned into a 2D height-sharded config (`shard_spec` set), so it does not test the ND regime. `[1,2,32,64]` stays `ND_SHARDED`.

**Citations.** Every citation lands on the code it claims, except these four. All four are fixed in `bc9e80c0ef2`.

| Doc text | Cites | Lands on | Should be |
|---|---|---|---|
| Row 3, "every slot's shape in pages is hashed on purpose" | `ternary_device_operation.cpp:561` | the function signature | `:560` (the comment that states the purpose) |
| §5, bullet 1, SNAKE_BETA TILE check | `:439-441` | a blank line, then two lines of the check | `:440-442` |
| Row 3, `has_sharding` false cases | `ternary_program_factory.cpp:316-323` | the slot-sharded booleans; stops mid-condition | `:319-326` |
| Row 3, the same | `ternary_op_utils.cpp:33-134` | starts inside `is_uneven` | `:37-138` |

The last two were added by my previous fix 5 and were not in the renumbered set. #59788 added four lines to `is_uneven`, and the factory moved by three.

**Provenance and §4** (the updates below are applied in `bc9e80c0ef2`). The substance holds. `compute_program_hash` and `resolve_mem_config_actual` are byte-identical across all 478 commits (checked function by function, `d388535aee7` against `26e6c5e2000b`). The only commit in that range that touched the file is #59788, and it added no layout validation. Three sentences need updating:
1. "the 401 commits the rebase brought in". The rebase brought in 478. 401 is the count since `254d9489b57`, the main the previous audits compared against. Suggested: "across the 478 commits between the old base `d388535aee7` and `26e6c5e2000b`".
2. "#60282 landed, which is the fix for the pack-format and divide-by-zero suspicions" (provenance) and "Both are expected to be closed by #60282 … needs re-checking rather than trusting" (§4). Re-checked, see *C*. Suggested §4 text: "**The divide-by-zero and pack-format suspicions** are closed at the trigger by #60282, which requires a BF16 output for the LLK row broadcast; both former triggers run correctly on silicon (pre-port audit, 2026-10-22). The kernel hazards remain behind that host-side gate. Neither is a relaxation question." Adjust the provenance sentence to match.
3. "ND-sharded tensors reach row 3 on this tree (#59788) … they are not probed" (provenance) and the §4 bullet **ND-sharded slots, on silicon**. They are now probed (N1–N3). Suggested §4 text: "**ND-sharded slots.** #59788 sends them down the non-native path, which makes them row 3 slots with a live sharded accessor. Probed on silicon (true `ND_SHARDED` inputs, preallocated output): a miss, a hit, and a leading-dims pair sharing one entry, all exact. An ND-sharded `input_a` reaches row 3 reliably only with a preallocated output: without one, `resolve_mem_config_actual` dereferences an empty 2D shard spec (`ternary_device_operation.cpp:198-199`, `:206-218`; `ternary_op_utils.cpp:938`), which is an op bug, not a relaxation question." If you keep Row 3 at Medium, the reason "no probe exercised one" no longer applies.

**§2's stop-condition bullet** stays deliberately unresolved, as you intended. Its paraphrase still matches `eltwise_binary_ng.md` row 4 word for word, and the ternary half of the bullet does not depend on it. For binary_ng's analysis owner: from the code, #59392 leaves row 4's premises intact. The gate still fires whenever something is sharded, and #59392 states the cache key is unchanged. I did not re-validate `eltwise_binary_ng.md`.

**§5: nothing reintroduced.** No check is guarded by a condition. No layout claim rests on a single slot. `match_page_size` is argued from check 2. `relax_logical_rank` is kept. The text added since the previous audit (checks 1 and 2, Row 3, §4) adds no row-major claim.

## C. #60282, and what `Known op issues` should say

**Pack-format: trigger closed, hazard latent.**
- `is_llk_bcast` now requires a BF16 output (`ternary_program_factory.cpp:219-234`, used at `:1256-1267`).
- On silicon, the former trigger is exact (K1: bf16 row-broadcast inputs, fp32 output, 0/4096 wrong; the previous audits measured 4085/4096). K2–K4, the bf16-output LLK path and `addcmul`, are exact too.
- The kernels still retarget the packer to c_4..c_6 and then pack into c_3 without restoring its format. This happens in `ternary_sfpu_row_bcast_ttt.cpp` (retarget `:66/87/108`, pack `:153`) and `ternary_addc_ops_fpu_rowbcast.cpp` (retarget `:60/77/94`, pack `:132`). It is harmless only because the gate now makes every one of those formats BF16.

**Divide-by-zero: trigger closed, hazard latent.**
- A zero `freq` reaches the int broadcast kernel only through `ComputeRowBcastAddcOp`. The kernel table never selects that kernel; only the LLK rename does (`:236-247`, behind `BCAST_LLK=1`). The plain ROW_BCAST config is `ComputeNoBcastAddcOp` (`ternary_op_utils.cpp:239-240`).
- `BCAST_LLK=1` now needs a BF16 output. The int kernel needs an INT32/UINT32 attribute dtype (`:1453-1455`). On every public path that dtype equals the output's, because `addcmul` passes `determine_output_dtype(output, …)` (`ternary.cpp:379`). `prim::ternary` has no other caller and no Python binding. The two conditions therefore cannot meet.
- Every other broadcast type mapped to the int broadcast kernel gets `freq` = Wt or Ht·Wt (`:922-942`).
- On silicon, the former trigger completes. The kernel still divides by an RTA the host sets to 0 for ROW_BCAST (`ternary_addcmul_int_sfpu_bcast.cpp:131-132`; `ternary_program_factory.cpp:937-939`).

**Two live bugs, on this tree and on main:**
1. **A float-input `addcmul` with an INT32/UINT32 `output_tensor` returns garbage.** The int kernel is chosen from the output dtype (`:1453-1455`), and `ttnn.addcmul` checks only that the inputs agree (`ternary.cpp:340-353`). On silicon: bf16 inputs, INT32 output, 4096/4096 wrong. The values are float bit patterns (`1069154304` where `11` was expected).
2. **An ND-sharded `input_a` without a preallocated output dereferences an empty 2D shard spec.**
   - `resolve_mem_config_actual` calls ternary's own `compute_mem_config_actual` for any sharded `input_a` (`ternary_device_operation.cpp:198-199`). That function dereferences `*shard_spec()` unconditionally (`ternary_op_utils.cpp:932-941`; binary_ng's version has an ND early return, `binary_ng_utils.cpp:974-976`).
   - With no `memory_config`, the largest-grid loop dereferences input shard specs too (`:206-218`). This is undefined behaviour.
   - Observed: with `memory_config=DRAM` it is benign, because the result is overwritten. With neither argument, it raises an unrelated `TT_FATAL` at `tensor_layout.cpp:171` (`!shard_align_error.has_value()`).
   - Change 2's early return makes the preallocated-output case safe.

**Suggested `Known op issues` text** (true once changes 1–4 land; main also has the items in *What main lacks*):

> addcmul with float inputs and an INT32/UINT32 output_tensor returns wrong data (int kernel picked from the output dtype). An ND-sharded first input without a preallocated output reads an empty 2D shard spec (undefined behaviour; can raise an unrelated TT_FATAL). #60282 closed the pack-format and divide-by-zero triggers; both kernel hazards remain behind a host-side BF16 gate.

## D. Device 2.0: local L1 access in `fill_tile_utils.hpp` — ruling recorded

**Ruling (user, 2026-10-22):** local `volatile tt_l1_ptr` access in the 11 non-NoC helpers of `fill_tile_utils.hpp` is Device 2.0 compliant. This audit validated the evidence. Do not raise it again.
- **The whitelist marks this exact pattern allowed.** `cb_dfb_api_whitelist.md:41-44` shows a ✅ "peek current cursor for local entry data": a `get_write_ptr()` peek, then `reinterpret_cast<volatile tt_l1_ptr uint32_t*>`, then a local fill. `:31` names "local L1 access / helper interop" as the sanctioned use of the public peek.
- **#47157 kept the same shape.** `647475b475ec` (2026-06-17) migrated moreh's borrowed fill helper to Device 2.0 by adding `cb_fill_helpers.hpp`. The new helper keeps the local store loop over `cb.get_write_ptr()` (`:26-29`) and changes only the cast, to `CoreLocalMem<volatile uint32_t>`. The old shape is still visible in `moreh_common.hpp:56-61`.
- **The 11 helpers do local stores only.** `fill_tile_utils.hpp` has 25 `volatile tt_l1_ptr` casts and no NoC call or NoC address outside `fill_tile_with_first_row` (`:384-406`, the uncommitted fix).
- **The line is local access versus remote access with hand-rolled addressing.** The one helper on the wrong side of it was `fill_tile_with_first_row`, which read through `get_noc_addr` + raw `noc_async_read`. That is why the donor fix is still the Device 2.0 condition.
- **On spelling.** `device_api_migration_guide.md:471-485` labels the raw cast "Legacy API" and `CoreLocalMem` the "New API". That is a spelling preference, not a gate.
  - If "the ported spelling" means the whitelist example's `dfb.get_write_ptr()` where our callers have `CircularBuffer::get_write_ptr()`, that change is port work under kernel-side whitelist rule 1. Agreed.
  - If it means #47157's `CoreLocalMem<volatile T>`, it is not port work. The port's kernel-side whitelist (`metal2_port.md:424-…`) admits no local-memory spelling change. It would be an optional Device 2.0 cleanup on binary_ng's track.

## E. The port brief

`METAL2_PORT_BRIEF.md` (blob `72580168237`) describes the pre-rebase tree. Stale on this tree:
- **Line numbers.** Every `ternary_device_operation.cpp` line, and most factory lines (+3 / +4).
- **The "Before you start" rebase item.** It is done.
- **The install warning.** `CMakeLists.txt:16-20` now globs `device/kernels/*.hpp` as well (#60073), and the new header is installed.
- **The c_4..c_6 liveness condition.** It now includes a BF16 output; the brief marked that **[rebase]**.
- **The `.get_id()` and `kernel_lib` facts.** #60282 added `reconfig_data_format(dfb.get_id(), …)` calls, #60073 added `copy_init_unless<init_once>(dfb.get_id())`, and #60306 passes `DataflowBuffer` objects to the chain.
- **A new heads-up.** `ternary_sfpu_copy_init.hpp` hardcodes `tt::CBIndex::c_0` (*Heads-ups*).

**Regenerated, at the user's request.** The gates and the declaration are unchanged. The line numbers, the install note, the c_4..c_6 condition, the `.get_id()` / `kernel_lib` facts and the new `ternary_sfpu_copy_init.hpp` heads-up are now current. Committing changes 1–4 as they stand does not move a line.

## Gate detail

- **TTNN factory concept: GREEN on the cell.** Cross-check, current lines:

  | Column | Sheet | Code |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` → `ProgramDescriptor` (`ternary_device_operation.hpp:50-53`, `ternary_program_factory.cpp:997`) ✓ |
  | `Custom hash` | `yes` | `compute_program_hash` (`.hpp:71`, `.cpp:592-653`) ✓ |
  | `Backdoor custom hash` | `layered` | `to_hash()` (`.hpp:37`, `.cpp:264-274`) via `hash_operation(args, …)` (`.cpp:622-623`) ✓ |
  | `get_dynamic_runtime_args` | `no` | absent; stale comment `.cpp:590` ✓ |
  | `Override runtime args method?` | `yes` | `.hpp:59-64`, `ternary_program_factory.cpp:1543-1649` ✓ |
  | `Pybind descriptor` | `no` | none ✓ |
  | `Smuggled pointer` | `no` | the miss path re-wraps buffer-carrying slots as `Buffer*` (`ternary_program_factory.cpp:1480-1487`). The hit path rewrites current addresses (`:1575-1604`) and re-points borrowed CBs (`:1606-1648`) ✓ |
  | `Op-owned tensors?` | (blank) | none ✓ |

- **Device 2.0: GREEN on the working tree, conditional on the donor fix.**
  - **Own dataflow kernels (12): compliant.** They are byte-identical to the previously audited tree, and that audit's sweep applies.
  - **Own compute kernels (10) and `ternary_sfpu_copy_init.hpp`: compliant.** Re-swept on this tree for CB-index free functions, raw NoC calls, `get_noc_addr`, `get_local_cb_interface`, `volatile`, `reinterpret_cast`, `CircularBuffer`, semaphores and `my_x[` / `my_y[`. The only hits are the three `reinterpret_cast<const float*>(&scalar)` sites on a local scalar (`ternary_sfpu_col_scalar_bcast_tts_tst.cpp:69`, `:78`; `ternary_sfpu_no_bcast_tts_tst.cpp:55`). Positive control: the same pattern finds main's five raw calls in the donor.
  - **`kernel_lib/eltwise` (41 files): clean.** No hits; it syncs through `DataflowBuffer` (162 uses).
  - **Donor `fill_tile_utils.hpp`.** `fill_tile_with_first_row` is fixed in the working tree (`:384-406`; probe F1 exact). Main still has `noc_async_read(get_noc_addr(…))` at `:382-387`. The other 11 helpers: compliant by the ruling in *D*.
  - **Compute-side donors.** The binary_ng utils are included by the same five kernels and still unused.
- **Feature compatibility: GREEN.** No `GlobalCircularBuffer`, `remote_cb`, `remote_index`, `address_offset`, `GlobalSemaphore` or semaphore anywhere in the op. The four `UpdateDynamicCircularBufferAddress` calls are the `Buffer&` form (`ternary_program_factory.cpp:1640/1642/1644/1646`).
- **Offset base pointers: GREEN.** Clean bases: reader slots 0–1 `buffer()->address()` (`:500-501` TTS/TST, `:864-866` TTT), writer slot 0 (`:904`).
- **TensorAccessor 3rd argument: N/A.** 28 two-argument constructions in the 12 bound dataflow kernels, and no 3-argument form (31 matches in the directory; 3 are in the unbound `ternary_reader_nobcast_ttt.cpp`).

## The uncommitted changes

Numbered as the invoker numbered them. All five are staged.

| # | Change | Does what it claims | Defects | Tests |
|---|---|---|---|---|
| 1 | every slot TILE (`ternary_device_operation.cpp:383-404`, `:416`) | **yes** (P1–P7, B1–B4, E1–E4, C1–C2) | none | 10 rejection + 1 acceptance case, pass |
| 2 | preallocated output's memory config wins (`:187-195`) | **yes** | none (the `:202` leftover is dropped, unstaged) | 2 cases, pass |
| 3 | output dtype / alignment / tile hashed (`:644-652`) | **yes** (D1 / D2 still share) | none | 1 case, passes |
| 4 | precedence in binary_ng, both overloads (`binary_ng_device_operation.cpp:692-698`, `:856-859`), with the conflict resolution at `:788-793` and `:901-904` | **yes** (*A*) | none | 6 new + 38 existing program-cache cases, pass |
| 5 | regression tests | 15 earlier + 4 new cases pass | none | — |

The `fill_tile_utils.hpp` fix is the sixth change in the tree. It is unchanged, still absent from main, and still belongs on the Device 2.0 track.

## TTNN ProgramFactory

- **Concept:** `CustomProgramSpecFactoryConcept`. The factory has an `override_runtime_arguments` (`ternary_device_operation.hpp:59-64`, `ternary_program_factory.cpp:1543-1649`), which the port changes to return `ProgramRunArgs`.
- **Fit:** single program; no op-owned resources; tensor-arg matching relaxed per the validated analysis; 1:1 with legacy.
- **Custom hash:** `compute_program_hash` (`ternary_device_operation.cpp:592-653`) over `to_hash()` (`:264-274`). `scalar_input_a/b` are not hashed. They ride the per-core compute args (`ternary_program_factory.cpp:921`, `:943`), which the override re-applies (`:1598-1604`). The porter leaves the hash alone.
- **Stop signals:** none on the working tree.

## Port-work summary  *(mirrors what a brief would carry)*

- **Tensor bindings.** TTT `predicate` / `value_true` / `value_false` use reader RTAs 0 / 1 / 2 and c_0 / c_1 / c_2. TTS uses `predicate` / `value_true` on RTAs 0 / 1 and c_0 / c_1. TST uses `predicate` / `value_false` on RTAs 0 / 1 and c_0 / c_1. `output` uses writer RTA 0 and c_3.

  | Binding | Define is 0 (not natively sharded) | Define is 1 (natively sharded) |
  |---|---|---|
  | `predicate` | **Case 1**, `TensorAccessor(src0_args, src0_addr)` (e.g. `ternary_reader_nosubtilebcast_ttt.cpp:64`) | **clean**: c_0 borrowed (`ternary_program_factory.cpp:1079`), one bulk reserve / push (`ternary_reader_nosubtilebcast_ttt.cpp:59-61`) |
  | `value_true` | **Case 1** (TTT, TTS) | **clean**: c_1 (`:1097` TTS, `:1122` TTT) |
  | `value_false` | **Case 1** (TTT, TST) | **clean**: c_2 TTT (`:1135`), c_1 TST (`:1110`) |
  | `output` | **Case 1** (`ternary_writer_nobcast.cpp:34`) | **clean**: c_3 borrowed (`:1182`); writer body compiled out (`#if !DST_SHARDED`, `:31`) |

  The defines are `SRC_SHARDED_*` = slot sharded ∧ `has_sharding` (`:206-212`) and `DST_SHARDED` = output sharded ∧ `has_sharding` (`:1337`). ND-sharded slots always have the define at 0: `is_uneven` returns true for them (`ternary_op_utils.cpp:22-25`).
- **TensorParameter relaxation:** `dynamic (see analysis)`. `{.dynamic_tensor_shape = true, .relax_logical_rank = true}` on all four slots, unconditionally.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**

  | CB | Allocated | Disposition |
  |---|---|---|
  | c_0 | always (`ternary_program_factory.cpp:1069-1080`) | **1P+1C**; borrowed when `SRC_SHARDED_A=1` |
  | c_1 | always (`:1082-1124`) | **1P+1C**; borrowed when `SRC_SHARDED_B=1` |
  | c_2 | TTT (`:1126-1136`) | **1P+1C**; borrowed when `SRC_SHARDED_C=1` |
  | c_3 | always (`:1171-1183`) | **1P+1C** when `DST_SHARDED=0`; compute **self-loop**, borrowed, when `DST_SHARDED=1` |
  | c_4 / c_5 / c_6 | TTT ∧ ROW_BCAST, unconditionally (`:1141-1169`) | **conditional DFB**: live (compute self-loop) only under `BCAST_LLK=1` ∧ that operand's `BCAST_A/B/C` = 1 (`:1399-1401`), in `ternary_sfpu_row_bcast_ttt.cpp` or `ternary_addc_ops_fpu_rowbcast.cpp`. `BCAST_LLK=1` needs all-BF16 inputs **and** a BF16 output (`:219-234`). Dead otherwise. Do not drop them: bind conditionally and gate in the kernel. |

## Heads-ups  *(mirrors what a brief would carry)*

- **New with #60073: `ternary_sfpu_copy_init.hpp` keys format tables by hardcoded CB ids.**
  - `copy_init_shared_with_c0<cb>()` compares `unpack_*[cb]` against a built-in `tt::CBIndex::c_0` (`:12-24`).
  - The two callers pass `tt::CBIndex::c_1` / `c_2` as template arguments, and one also reads `unpack_src_format[tt::CBIndex::c_0]` directly (`ternary_addc_ops_sfpu_bcast.cpp:18`; `ternary_sfpu_col_scalar_bcast_ttt.cpp:18-19`).
  - Rule 2 covers the call sites (`dfb::` tokens, provided they work as template arguments). The `c_0` inside the helper has no parameter to receive the predicate's token. Decide before porting whether giving it one is within the kernel-side whitelist. Otherwise ask.
- **`.get_id()` at LLK call sites.** Every compute kernel passes `dfb.get_id()` (e.g. 24 sites in `ternary_sfpu_row_bcast_ttt.cpp`, 23 in `ternary_addc_ops_fpu_rowbcast.cpp`), including #60282's `reconfig_data_format` calls and #60073's `copy_init_unless`. Pass `dfb::name` directly (`port_patterns.md`).
- **Temp-wrapper helpers.** `process_tile(uint32_t …_cb_id, …)` builds `DataflowBuffer`s from ids at `ternary_sfpu_col_scalar_bcast_ttt.cpp:34-37`, `ternary_addc_ops_sfpu_bcast.cpp:34-37`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:26-28` and `ternary_addcmul_int_sfpu_bcast.cpp:25-28`.
- **`kernel_lib` call shape (#60306).** `ternary_addcmul_int_sfpu.cpp:19-26` builds kernel-scope `DataflowBuffer`s from `tt::CBIndex` ids and passes them to the `ckl` chain elements. Swap the ids for `dfb::` tokens.
- **Hardcoded CB ids in one reader.** `tts_tst_reader_row_bcast.cpp:47-48` ignores CB-id CTAs 0 and 1.
- **Unconditional `DataflowBuffer` constructions meet conditional bindings.** These are `ternary_writer_nobcast.cpp:29` (outside `#if !DST_SHARDED`, `:31`) and `ternary_addc_ops_fpu_rowbcast.cpp:29-31` (c_4..c_6). Gate both.
- **Install coverage: fine.** `CMakeLists.txt:16-20` globs `*.cpp` and `*.hpp` under `device/kernels/`. The previous audit's warning about kernel headers no longer applies.
- **Rebase: done.** It is no longer a precondition.
- **Translating `override_runtime_arguments`.** It re-runs `build_per_core_runtime_args` (`:701-…`) for every core, re-points borrowed CBs (`:1606-1648`), and leaves the accessor common RTAs alone (`:1555-1570`). That comment's "ND sharding cannot reach this path" is false since #59788. The conclusion still holds, because the key hashes the very `tensor_shape_in_pages` words, and N1–N3 confirm ND hits are exact.
- **Keep the hit-path validation.** Validity checks 1 and 2 depend on `validate_on_program_cache_miss` running on hits.
- **Op bugs are not port work** (*C*, *Misc anomalies*).

## Team-only

### Out-of-directory coupling

**Roll-up: ✓ clean.**
- `fill_tile_utils.hpp` (in-family): 12 helpers through the `FILL_TILE_WITH_FIRST_*` defines; raw L1 pointer; ✓ on the working tree.
- binary_ng compute utils: dead includes; ✓.
- `kernel_lib/eltwise` (shared lib): `eltwise_chain`, `CopyTile<input(…)>{dfb}`, `FillInt`, `MulIntBinary`, `AddIntBinary`, `PackTile<output(…)>{dfb}`; `DataflowBuffer&` arguments (#60306); ✓.
- No borrowed kernel files. **Lent:** `ternary_reader_nobcast_ttt.cpp` → binary_backward.

### TTNN factory analysis
Current concept `descriptor`. Target `CustomProgramSpecFactoryConcept`. No op-owned tensors, no MeshWorkload need. Custom hash present. `get_dynamic_runtime_args` and a pybound `create_descriptor` are absent. The relaxation cell clears by value, and the doc behind it validates on the working tree.

## What main lacks until the changes land

On HEAD, which is main for every in-scope file except the staged five:
- **Row-major is accepted with wrong output.** HEAD's validation needs TILE only for an interleaved `input_a`, and for the output only when `input_a` is interleaved. P1, P3, P4 and the mixed-layout P6 return wrong data, and P5 is exact.
- **A preallocated output's memory config loses to a `memory_config` argument**, in ternary and in binary_ng. Outputs in different memories can share an entry and write to the wrong address space.
- **The output's dtype, alignment and tile are not hashed.**
- **The relaxation analysis is void,** because checks 1 and 2 fail without changes 1–3.
- **Device 2.0 is RED** on the donor (`fill_tile_utils.hpp:382-387`).

## Misc anomalies  *(team-only, non-gating; route to the ops team)*

- **Float-input / int-output `addcmul` returns garbage.** Live on this tree and main; see *C*.
- **ND-sharded `input_a` without a preallocated output: undefined behaviour.** Live on this tree and main; see *C*. The fix is an ND early return in ternary's `compute_mem_config_actual`, like binary_ng's, plus a `shard_spec()` check in the largest-grid loop.
- **Pack-format restore missing; divide by a zero RTA.** Both are latent behind #60282's gate (*C*).
- **Stale comments.**
  - `ternary_device_operation.cpp:590` ("re-applied via get_dynamic_runtime_args").
  - `ternary_program_factory.cpp:688-694` ("padded VOLUME").
  - The override's ND remark at `:1566-1570`.
- **Dead code.** The SNAKE_BETA TILE check at `ternary_device_operation.cpp:440-442`, now redundant. The row-major branches in `compute_output_specs` (`:493-494`) and `resolve_mem_config_actual` (`:252`) can only see TILE after validation.
- **Unused RTA reads, ignored CB-id CTAs, a latent hang in the unreachable outer-bcast sharded branch, two dead kernel files.** All in dataflow kernels unchanged since the previous audit, whose notes stand.

## Questions for the user

All four were answered on 2026-10-22:
1. **The port brief:** regenerate now. Done.
2. **The relaxation doc fixes:** apply all. Done (blob `bc9e80c0ef2`). In the ND bullet of §4, the doc's existing check-4 code argument is kept and the probe results are added after it.
3. **`Known op issues`:** use the text proposed in *C*, without the main-only items.
4. **The ternary leftover:** drop it. Done, unstaged.

## Recipe notes

- **A GREEN that holds only on a working tree.** Still unnamed in the recipe. This op has now been GREEN-conditional through a rebase, and the brief went stale exactly as predicted. A sentence on when a brief is written for such a GREEN, and when it must be refreshed, would settle it.
- **Validity checks should state which dispatches run a check.** This doc now does, in check 2. The recipe note from the previous audit stands for other analyses.
- **Local `volatile tt_l1_ptr` access.** The user ruled it compliant, and the CB/DFB whitelist supports that reading. The Device 2.0 migration guide still labels the raw cast "Legacy". One line in the audit recipe's Device 2.0 Green bullet would spare the next auditor the question.
- **`PrecomposedUnicastEndpoint` is still missing from the Device 2.0 migration guide's endpoint list.**
- **ND shard specs that are 2D-expressible become 2D.** An auditor probing "ND" regimes should check `memory_config().shard_spec()` is empty. A note in the TensorParameter section would prevent false evidence.
