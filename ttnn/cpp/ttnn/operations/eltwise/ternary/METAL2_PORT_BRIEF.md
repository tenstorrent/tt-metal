# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/eltwise/ternary/`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `917314f85d3 2026-10-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**This brief describes** `anasuya/metal2_eltwise_ternary`, now a single commit `21d0f3c7ba9` (*Pre-port fixes*) on `origin/main` `d764c9f5ce2` (0 behind), as of 2026-10-27. Line numbers are that tree's.

It was written on 2026-10-22 against `917314f85d3` rebased onto `26e6c5e2000b`, with the fixes staged rather than committed. The fix patch is byte-identical after the rebase, so every citation into it still resolves. Two citations into `ternary_op_utils.cpp` were re-resolved by content after #61721 added 23 lines to that file; `:22-25` happened not to move. Citations into `ternary_device_operation.cpp`, `ternary_program_factory.cpp` and the dataflow kernels were not affected, because no upstream commit in that span touched them.

## Before you start

- **Land changes 1–4 first.** They are staged, not committed, and the relaxation analysis is void without changes 1–3:
  1. every tensor slot must be TILE (`ternary_device_operation.cpp:383-404`, `:416`);
  2. a preallocated output's memory config wins (`:187-195`);
  3. the output's dtype, alignment and tile are hashed (`:644-652`);
  4. the same precedence in `binary_ng_device_operation.cpp` (`:692-698`, `:856-859`).
- **The `fill_tile_utils.hpp` fix lands on the Device 2.0 track, not in the port.** The kernel-side whitelist does not let a port carry it, and Device 2.0 is GREEN only with it.
- No rebase is pending.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` returns `ProgramDescriptor`, `ternary_program_factory.cpp:997`).
- **Op-owned tensors:** none.
- **Target concept:** `CustomProgramSpecFactoryConcept`. The factory has an `override_runtime_arguments` (`ternary_device_operation.hpp:59-64`, `ternary_program_factory.cpp:1543-1649`), and per `ttnn_factory.md` that one fact selects the custom concept. Translate the method; do not delete it.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (the cell is `dynamic (see analysis)`, and the doc validates) · `get_dynamic_runtime_args` (absent; only a stale comment at `ternary_device_operation.cpp:590`).
- **Present, non-gating:** a custom hash, `compute_program_hash` (`ternary_device_operation.cpp:592-653`), layered over `to_hash()` (`:264-274`). Leave both alone (`ttnn_factory.md`, *The cache key: leave the custom hash alone*). No pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (per binding). TTT `predicate` / `value_true` / `value_false` use reader RTAs 0 / 1 / 2 and c_0 / c_1 / c_2. TTS uses `predicate` / `value_true` on RTAs 0 / 1 and c_0 / c_1. TST uses `predicate` / `value_false` on RTAs 0 / 1 and c_0 / c_1. `output` uses writer RTA 0 and c_3.

- `predicate` — **Case 1** (via `TensorAccessor`) when `SRC_SHARDED_A=0` → `TensorParameter` / `TensorBinding`; the kernel uses `TensorAccessor(tensor::predicate)` (today `TensorAccessor(src0_args, src0_addr)`, e.g. `ternary_reader_nosubtilebcast_ttt.cpp:64`). When `SRC_SHARDED_A=1` the accessor is compiled out: bind c_0 as a DFB `borrowed_from` the predicate (`ternary_program_factory.cpp:1079`). The reader then does one bulk `reserve_back` / `push_back` (`ternary_reader_nosubtilebcast_ttt.cpp:60-61`).
- `value_true` — **Case 1** (TTT, TTS). Borrowed c_1 when `SRC_SHARDED_B=1` (`:1097` TTS, `:1122` TTT).
- `value_false` — **Case 1** (TTT, TST). Borrowed when its define is 1: c_2 for TTT (`:1135`), c_1 for TST (`:1110`).
- `output` — **Case 1** when `DST_SHARDED=0` (`ternary_writer_nobcast.cpp:34`). When `DST_SHARDED=1`, c_3 is borrowed from the output (`:1182`) and the writer body is compiled out (`#if !DST_SHARDED`, `:31`).
- No Case 2.
- The defines: `SRC_SHARDED_*` = slot sharded ∧ `has_sharding` (`ternary_program_factory.cpp:206-212`); `DST_SHARDED` = output sharded ∧ `has_sharding` (`:1337`).
  - `has_sharding` can be true with L1-interleaved inputs and an L1-sharded output. Then the inputs are Case 1 and the output is borrowed.
  - An ND-sharded slot always has its define at 0, because `is_uneven` returns true for it (`ternary_op_utils.cpp:22-25`).
- The legacy factory still writes a natively sharded slot's address RTA, which the kernel never reads. In Metal 2.0 that binding is `borrowed_from` only.

**TensorParameter relaxation:** `dynamic (see analysis)` → `analyses/relaxations/eltwise_ternary.md` (validated). On `predicate`, `value_true`, `value_false` and the output, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

Do not set `match_page_size` or `match_padded_shape_only`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**
- assign **1P+1C** on c_0 (`ternary_program_factory.cpp:1069-1080`), c_1 (`:1082-1124`) and c_2 (`:1126-1136`) in every config (reader producer, compute consumer). Each is borrowed when its slot's define is 1.
- c_3 (`:1171-1183`): **1P+1C** (compute → writer) when `DST_SHARDED=0`; **self-loop** on compute when `DST_SHARDED=1` (borrowed from the output; nothing drains it).
- make c_4 / c_5 / c_6's DFB specs **conditional** (row-broadcast LLK staging for A / B / C, allocated unconditionally for TTT ∧ ROW_BCAST at `:1141-1169`).
  - Each is live only under TTT ∧ ROW_BCAST ∧ `BCAST_LLK=1` ∧ that operand's `BCAST_A` / `BCAST_B` / `BCAST_C` = 1 (`:1399-1401`), as a compute self-loop in `ternary_sfpu_row_bcast_ttt.cpp` or `ternary_addc_ops_fpu_rowbcast.cpp`.
  - `BCAST_LLK=1` needs all-BF16 inputs **and** a BF16 output (`is_llk_bcast`, `:219-234`, applied at `:1256-1267`).
  - Each is dead under every other config. Do **not** drop them. Use a host-conditional binding plus a preprocessor gate (`port_patterns.md`, *Conditional / optional resource bindings*).
- No dead CB, no multi-binding.

**Translate `override_runtime_arguments`** (`ternary_program_factory.cpp:1543-1649`) to return `ProgramRunArgs` (`ttnn_factory.md`, *The custom concept*). On a hit the legacy override:
- re-runs `build_per_core_runtime_args` (`:701-…`) for every core, including zero rows for noop cores. These become `kernel_run_args`.
- re-points the borrowed CBs (`:1606-1648`). With all four tensors in `tensor_args` on every dispatch, that becomes the borrowed bindings.
- deliberately leaves the `TensorAccessorArgs` common RTAs alone (`:1555-1570`).

Reproduce exactly what it refreshes. `scalar_input_a/b` are not in the hash. They ride the per-core compute args (`:921`, `:943`) and must be re-applied on every hit, as the override does today (`:1598-1604`).

## Watch for

- **CB endpoints (multi-binding):** none. No CB has a third toucher. Each reader's own `FILL_*` writes go to a CB it produces, which is the same toucher.
- **Cross-op / shared kernels:**
  - No `_metal2` fork exists yet.
  - `dataflow/ternary_reader_nobcast_ttt.cpp` is bound by `eltwise/binary_backward`, not by ternary. Leave it alone. It is not on a sunset list for this port.
  - `fill_tile_utils.hpp` (binary_ng, in-family) needs no fork. Its helpers take a raw L1 pointer, and the ported readers pass `dfb.get_write_ptr()`. Their local `volatile tt_l1_ptr` access is Device 2.0 compliant (user ruling). Do not respell it; the kernel-side whitelist admits no such change.
  - `kernel_lib` is outside the shared-kernel convention.
  - Install coverage is fine. `CMakeLists.txt:16-20` globs `*.cpp` and `*.hpp` under `device/kernels/`, which covers a fork beside its original and a new kernel header.
- **RTA varargs:** none. Every RTA is read once at a literal index: readers 0-26 (the TTS/TST broadcast readers skip 20-24; `ternary_reader_nobcast_tst_tts.cpp` reads 0-4), writer 0-9 (slot 10 is padding, never read), compute 0-3 (`kNumReaderArgs` / `kNumWriterArgs` / `kNumComputeArgs` = 27 / 11 / 4, `ternary_program_factory.cpp:648-650`). The CB-id CTAs, the chained `TensorAccessorArgs` blocks and the trailing `has_sharding` CTA (`:1284`, `:1295`, `:1308`, `:1335`) all become named CTAs.
- **Anything else:**
  - **`ternary_sfpu_copy_init.hpp` (#60073) keys unpack-format tables by hardcoded CB ids.**
    - `copy_init_shared_with_c0<cb>()` compares `unpack_*[cb]` with a built-in `tt::CBIndex::c_0` (`:12-24`).
    - Its callers pass `tt::CBIndex::c_1` / `c_2` as template arguments, and one also reads `unpack_src_format[tt::CBIndex::c_0]` (`ternary_addc_ops_sfpu_bcast.cpp:18`; `ternary_sfpu_col_scalar_bcast_ttt.cpp:18-19`).
    - Rule 2 covers the call sites, provided the `dfb::` tokens work as template arguments. The `c_0` inside the helper has no parameter to receive the predicate's token. Settle whether adding one is within the kernel-side whitelist before porting. If unsure, stop and ask.
  - **`.get_id()` at LLK call sites.** Every compute kernel passes `dfb.get_id()`, for example 24 sites in `ternary_sfpu_row_bcast_ttt.cpp` and 23 in `ternary_addc_ops_fpu_rowbcast.cpp`. These include the `reconfig_data_format` calls and `copy_init_unless<init_once>(dfb.get_id())`. Pass `dfb::name` directly (`port_patterns.md`, *Pass DFB handles directly to LLKs and kernel-lib helpers*).
  - **Temp-wrapper helpers.** `process_tile(uint32_t …_cb_id, …)` builds `DataflowBuffer`s from ids at `ternary_sfpu_col_scalar_bcast_ttt.cpp:34-37`, `ternary_addc_ops_sfpu_bcast.cpp:34-37`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:26-28` and `ternary_addcmul_int_sfpu_bcast.cpp:25-28`.
  - **`kernel_lib` call shape (#60306).** `ternary_addcmul_int_sfpu.cpp:19-26` builds kernel-scope `DataflowBuffer`s from `tt::CBIndex` ids and passes them by reference to the `ckl` chain elements. Swap the ids for `dfb::` tokens.
  - **The other compute kernels** use `DataflowBuffer` on hardcoded `tt::CBIndex::c_N` ids. 8 dataflow kernels still construct `CircularBuffer` and need the CB→DFB swap.
  - **Hardcoded CB ids in one reader.** `tts_tst_reader_row_bcast.cpp:47-48` uses `tt::CBIndex::c_0` / `c_1` and ignores CB-id CTAs 0 and 1. Map them to the same `dfb::` tokens the other readers get.
  - **Unconditional `DataflowBuffer` constructions meet conditional bindings.** `ternary_writer_nobcast.cpp:29` constructs `dfb_out` outside `#if !DST_SHARDED` (`:31`). `ternary_addc_ops_fpu_rowbcast.cpp:29-31` constructs c_4..c_6 unconditionally. Gate both. `ternary_sfpu_row_bcast_ttt.cpp:34-54` is already gated.
  - **Unused donor includes.** Five compute kernels include binary_ng's `eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp` and use none of it (`ternary_sfpu_no_bcast_ttt.cpp:7-8`, `ternary_sfpu_no_bcast_tts_tst.cpp:7-8`, `ternary_sfpu_col_scalar_bcast_ttt.cpp:11-12`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp:12-13`, `ternary_sfpu_row_bcast_ttt.cpp:16-17`). Check the kernel-side whitelist before keeping those lines.
  - **Native sharding implies identical input shapes** (`ternary_op_utils.cpp:62`, `:114-118`), which means broadcast type NONE. Every broadcast reader compiles with `SRC_SHARDED_*=0`. Port their sharded branches as-is.
  - **Keep the hit-path validation.** The op is TILE-only, enforced on every dispatch: there is no `validate_on_program_cache_hit`, and the framework runs `validate_on_program_cache_miss` on hits too. The relaxation's checks 1 and 2 depend on that, because layout is not a hash term.
  - **ND-sharded slots reach row 3** (live sharded accessor) since #59788, and they are exact on silicon on a miss and a hit. The override comment's "ND sharding cannot reach this path" (`:1566-1567`) is stale. Its conclusion still holds.
  - **Op bugs are not port work.** These are float-input `addcmul` with an int output, an ND-sharded first input without a preallocated output, and the latent pack-format and divide-by-zero hazards. They are recorded in the audit (*C*, *Misc anomalies*). Do not fix them in the port.
