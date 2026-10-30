# Metal 2.0 Port Brief — `eltwise/ternary`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b90203d162f 2026-10-30 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Base:** `anasuya/metal2_port_eltwise_ternary` @ `b90203d162f`, which is `origin/main` `3f57b50ff00` plus recipe docs. The pre-port fixes (#62084) are in the base. All line numbers below are this tree's. Paths are relative to `ttnn/cpp/ttnn/operations/eltwise/ternary/` unless they start with `ttnn/` or `tt_metal/`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`. `create_descriptor` is at `device/ternary_program_factory.cpp:997`.
- **Op-owned tensors:** none.
- **Target concept:** `CustomProgramSpecFactoryConcept`. Translate `override_runtime_arguments` (declared at `device/ternary_device_operation.hpp:59-64`, body at `device/ternary_program_factory.cpp:1543-1649`) into one that returns `ProgramRunArgs`. See *Watch for*.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook; only a stale comment at `device/ternary_device_operation.cpp:580`).
- **Present, and not gates:**
  - A custom hash: `compute_program_hash` (`device/ternary_device_operation.cpp:582-637`), layered over `operation_attributes_t::to_hash()` (`:261-271`). Leave both exactly as they are.
  - The `override_runtime_arguments` above.
  - No pybound `create_descriptor`.
- **Shape:** one factory, no `select_program_factory`. It branches on variant (TTT / TTS / TST) and broadcast type, with a different CB set and kernel sources per branch (`port_patterns.md`, *Multi-variant factories*).

## Construct — to do

**Tensor bindings** (per binding). Each slot's define picks its case per configuration. The defines are `SRC_SHARDED_*` = slot sharded ∧ `has_sharding` (`device/ternary_program_factory.cpp:206-212`) and `DST_SHARDED` = output sharded ∧ `has_sharding` (`:1337`).

- `predicate`: reader RTA 0, c_0, `SRC_SHARDED_A`.
  - **Case 1** (via `TensorAccessor`) when the define is 0 → express as `TensorParameter` / `TensorBinding`; the reader uses `TensorAccessor(tensor::predicate)`.
  - **clean** when it is 1 → c_0's DFB is `borrowed_from` the predicate. The reader's bulk `reserve_back` / `push_back` stays (e.g. `ternary_reader_nosubtilebcast_ttt.cpp:59-61`).
- `value_true` (TTT, TTS): reader RTA 1, c_1, `SRC_SHARDED_B`. Same two cases.
- `value_false`: TTT uses reader RTA 2, c_2, `SRC_SHARDED_C`. TST uses reader RTA 1, c_1, `SRC_SHARDED_B`. Same two cases.
- `output`: writer RTA 0, c_3, `DST_SHARDED`.
  - **Case 1** when the define is 0 (`ternary_writer_nobcast.cpp:34`).
  - **clean** when it is 1 → c_3 is `borrowed_from` the output; the writer body is compiled out (`:31`).
- No Case 2. The address RTAs and their `TensorAccessorArgs` plumbing (`device/ternary_program_factory.cpp:1271-1335`, `:1483-1534`) go away.
  - TTS / TST pass a scalar for the third operand. Reader RTA 2 is then a plain `0u` (`:502`), not a binding.

**TensorParameter relaxation:** `dynamic (see analysis)` → `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/relaxations/eltwise_ternary.md`.
- The doc is untracked on this branch (blob `c89ba329e90`). The audit validated it on this tree: all five checks pass.
- Transcribe its §2 declaration onto every `TensorParameter` the variant declares (`predicate`, `value_true`, `value_false`, the output), unconditionally:

  ```cpp
  .relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
  ```

- Do not set `match_page_size` or `match_padded_shape_only`.
- Run the doc's validity checks before applying. Its `device/ternary_device_operation.cpp` line numbers are the pre-merge tree's. The audit maps every one of them to this tree (*The doc's citations on HEAD*).

**TensorAccessor 3rd arg:** none. All 28 constructions take two arguments.

**CB endpoints:**

| DFB (legacy CB) | Configs | Producer | Consumer | Disposition |
|---|---|---|---|---|
| c_0, predicate | all | reader | compute | 1P+1C. `borrowed_from` predicate when `SRC_SHARDED_A=1`. |
| c_1, `value_true` (TTT / TTS) or `value_false` (TST) | all | reader | compute | 1P+1C. Borrowed when `SRC_SHARDED_B=1`. |
| c_2, `value_false` | TTT only | reader | compute | 1P+1C. Borrowed when `SRC_SHARDED_C=1`. |
| c_3, output | `DST_SHARDED=0` | compute | writer | 1P+1C |
| c_3, output | `DST_SHARDED=1` | compute | compute | **self-loop** on compute, `borrowed_from` output. The writer is not bound. |
| c_4 / c_5 / c_6, LLK row-broadcast staging for A / B / C | see below | compute | compute | **self-loop**, and a **conditional DFB** |

- **Sizes:** as legacy. A borrowed DFB holds the slot's shard (tile size × shard volume). Every other DFB holds 2 tiles (`device/ternary_program_factory.cpp:1065-1183`).
- **Make c_4 / c_5 / c_6 conditional.** Each one is live only when `is_llk_bcast(...)` holds (`:219-234`: TTT ∧ ROW_BCAST, BF16 inputs **and** a BF16 output) **and** that operand is row-broadcast (`BCAST_A` / `BCAST_B` / `BCAST_C` = 1, `:1397-1401`). It is dead in every other TTT ∧ ROW_BCAST configuration. **Do not drop it.**
  - Bind it on the host under that condition. Gate the kernel side with the preprocessor (`port_patterns.md`, *Conditional / optional resource bindings*).
  - Expect new host structure: the legacy factory allocates all three whenever TTT ∧ ROW_BCAST (`:1141-1169`).
  - Kernel side: `ternary_sfpu_row_bcast_ttt.cpp:34-54` is already gated by `#if BCAST_*`. `ternary_addc_ops_fpu_rowbcast.cpp:29-31` constructs all three unconditionally and needs the gate.
- **Gate the writer's c_3 construction.** `ternary_writer_nobcast.cpp:29` constructs `dfb_out` outside `#if !DST_SHARDED` (`:31`). With `DST_SHARDED=1` the writer has no c_3 binding. Its token will then not exist.
- No dead CB. No multi-binding. No semaphores.

## Watch for

- **CB endpoints (multi-binding):** none. No CB has a third toucher.
- **Cross-op / shared kernels:** none.
  - All 22 bound kernels live in this op's directory, and no other op binds them. No `_metal2` fork exists or is needed: convert them in place.
  - **`ternary_reader_nobcast_ttt.cpp` is the exception that is not yours.** It sits in this directory, but ternary does not bind it; `eltwise/binary_backward` does (`ttnn/cpp/ttnn/operations/eltwise/binary_backward/device/binary_backward_program_factory.cpp:122`). Do not touch it.
  - Other binding ops: none. There is no sunset list.
  - Donor headers need no change. The `fill_tile_utils.hpp` helpers take a raw L1 pointer: pass `dfb.get_write_ptr()`. `kernel_lib/eltwise` takes the CB id in a template position (pass the token) and a non-const `DataflowBuffer&` (keep the named locals in `ternary_addcmul_int_sfpu.cpp:23-26`).
- **RTA varargs:** none. Name every RTA. The legacy factory pushes uniform rows (reader 27, writer 11, compute 4 words; `device/ternary_program_factory.cpp:648-650`), but each source reads its own subset:
  - TTT readers read 0–26.
  - The TTS / TST broadcast readers read 0–19, 25 and 26.
  - `ternary_reader_nobcast_tst_tts.cpp` reads 0–4.
  - The writer reads 0–9. Slot 10 is padding.
  - Compute kernels read subsets of 0–3.
  - The address slots (reader 0–2, writer 0) become bindings.
  - Dead reads to carry or drop as the recipe directs: RTA 2 in the six TTS / TST readers (always 0), and RTAs 19 and 26 in the five TTS / TST broadcast readers.
- **Translating `override_runtime_arguments`** (`device/ternary_program_factory.cpp:1543-1649`). The refresh set is inherited (`ttnn_factory.md`, *The cache-hit contract*). The legacy override does three things:
  1. It re-runs `build_per_core_runtime_args` (`:701-960`) and writes a full reader, writer and compute row into **every** core of the worker grid, with all-zero rows for noop cores (`:784-791`).
  2. It re-points every tensor-backed CB at the current buffer, by CB index (`:1606-1648`).
  3. It deliberately leaves the reader / writer common RTAs alone, which carry the `TensorAccessorArgs` words (`:1555-1570`).
  - In Metal 2.0, the address slots and the CB re-pointing both become `ProgramRunArgs::tensor_args`, one entry per `TensorParameter`, on every dispatch. The other per-core values become `kernel_run_args`.
  - **Keep covering every core.** A core can flip between work and noop across hits (`:697-700`), and `UpdateProgramRunArgs` keeps the last value for anything left out. The broadcast compute kernels rely on the zero rows: each returns on `num_tiles == 0` before dividing by `tile_freq` (e.g. `ternary_sfpu_col_scalar_bcast_ttt.cpp:126-128`).
  - The legacy override handles an output that aliases an input by updating CBs by index (`:1607-1610`). Per-`TensorParameter` `tensor_args` cover the same case.
- **Compile-time CB numbers outside LLK calls: convert them to tokens, do not keep the numbers.**
  - `ternary_sfpu_copy_init.hpp:15` hardcodes `tt::CBIndex::c_0` inside `copy_init_shared_with_c0<cb>()`, which compares the JIT-generated `unpack_*` format arrays of two buffers.
  - At file scope, `ternary_sfpu_col_scalar_bcast_ttt.cpp:18-19` and `ternary_addc_ops_sfpu_bcast.cpp:17-18` pass `tt::CBIndex::c_1` / `c_2` into it. `ternary_sfpu_col_scalar_bcast_ttt.cpp:18` also indexes `unpack_src_format[tt::CBIndex::c_0]`.
  - Metal 2.0 gives each DFB the lowest free device slot in creation order (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2182-2202`), and the format arrays are indexed by slot (`tt_metal/jit_build/genfiles.cpp:916-917`). A leftover `c_0` would compare against whichever DFB holds slot 0. It still compiles, and it can silently change which operands share the copy init.
  - The token's `uint32_t` conversion is `constexpr`. It therefore works in the template position. The header's own `c0` needs to become a parameter. This is the op's own file, created after the previous audit (#60073), and none of the docs gives a precedent for it. Confirm the change rather than swapping blind.
- **DFB handles at LLK call sites.** All ten compute kernels pass `dfb.get_id()` to LLKs. Four build temporary `DataflowBuffer`s from ids inside `process_tile` helpers (`ternary_sfpu_col_scalar_bcast_ttt.cpp:34-37`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:26-28`, `ternary_addc_ops_sfpu_bcast.cpp:34-37`, `ternary_addcmul_int_sfpu_bcast.cpp:25-28`). Pass `dfb::name` directly (`port_patterns.md`, *Pass DFB handles directly to LLKs and kernel-lib helpers* and the matching anti-pattern).
- **The kernels are already part-modernized.** The port is mostly a binding-layer change.
  - All 10 compute kernels and 4 dataflow kernels (`ternary_reader_nosubtilebcast_ttt.cpp`, `ternary_reader_nobcast_tst_tts.cpp`, `tst_tts_reader_outer_bcast.cpp`, `ternary_writer_nobcast.cpp`) already use `DataflowBuffer`, on hardcoded CB ids or CB-id CTAs.
  - The 8 fill-helper readers still construct `CircularBuffer` and call `get_tile_size()`. They need the CB→DFB swap; the DFB's own accessor is `get_entry_size()`. Confirm it against the kernel-side whitelist rather than swapping blind.
- **Hardcoded CB ids in one reader.** `tts_tst_reader_row_bcast.cpp:47-48` uses `tt::CBIndex::c_0` / `c_1` and ignores CB-id CTAs 0 and 1, which the host still passes. Map them to the same tokens as the other TTS / TST readers.
- **Unused donor includes.** Five compute kernels include binary_ng's `eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp` and use nothing from them (`ternary_sfpu_no_bcast_ttt.cpp:7-8`, `ternary_sfpu_no_bcast_tts_tst.cpp:7-8`, `ternary_sfpu_col_scalar_bcast_ttt.cpp:11-12`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:12-13`, `ternary_sfpu_row_bcast_ttt.cpp:16-17`). `eltwise_utils_sfpu.hpp:9` pulls in `api/dataflow/circular_buffer.h`. Check the kernel-side whitelist before keeping or removing those lines. Do not edit the binary_ng headers.
- **Native sharding implies broadcast type NONE.** Native sharding needs identical input shapes (`device/ternary_op_utils.cpp:62`, `:114-118`). Every broadcast reader therefore compiles with `SRC_SHARDED_*=0` and `has_sharding=0`. Port their sharded branches as they are; some are unreachable, and `tst_tts_reader_outer_bcast.cpp:51-58` would hang if it were reached.
- **Install coverage.** `CMakeLists.txt:16-20` globs `device/kernels/*.cpp` and `*.hpp` recursively. Any new file beside the originals is installed without a build-file change.
- **Exercise the relaxation on cache hits.** Hit-path validation is off by default (`validate_program_args`, `ttnn/api/ttnn/config.hpp:31`). Turn it on (`TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`) for the cache tests. The audit's probe pairs are the calls that would throw if the declaration were too narrow:
  - `[4,1,32,32]` then `[1,4,32,32]` (leading dims);
  - `[1,32,64]` then `[1,1,32,64]` (rank).
  - Each pair should give one cache entry and exact results, as it does on legacy. The #62084 regression tests in `tests/ttnn/unit_tests/operations/eltwise/test_ternary.py` are the matching before/after check.
- **The in-place recipe trails upstream by two commits.** Upstream `c41f7f4ef0a` renames `DFBAccessor` → `DFBBindingToken` in the port docs. The kernel type is `DFBBindingToken` (`tt_metal/hw/inc/api/dataflow/dfb_binding_token.h`), and `dfb::name.id` does not compile (the id is private). Pass the token itself.
- **Do not use `experimental/quasar/` copies as a reference.** No quasar copy of ternary exists today, and none of this op's kernels has a `_metal2` sibling anywhere.
