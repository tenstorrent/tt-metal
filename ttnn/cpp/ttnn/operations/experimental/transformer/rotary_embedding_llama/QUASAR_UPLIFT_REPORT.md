# QUASAR_UPLIFT_REPORT — `ttnn.experimental.rotary_embedding_llama`

**Op directory:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_rotary_embedding_llama.py`
(decode cases → `RotaryEmbeddingLlamaMultiCoreSharded`; prefill interleaved cases →
`RotaryEmbeddingLlamaMultiCore`. The third factory, `RotaryEmbeddingLlamaMultiCorePrefillSharded`,
is not driven by this test but shares the writer kernel and was uplifted with it.)
**Date:** 2026-09-01. Static audit + uplift only — **no build or device run was performed in this
session**; test commands to run are at the end.

## Status: GREEN

The op is already Metal 2.0 on Gen1 (gate passed — see below). The Quasar-uplift audit found
exactly two statically-determinable Gen2 blockers, both with a canonical, mechanical fix
prescribed by the recipe suite, and both fixes are applied in place:

1. **DM self-loop DFB** (`ZERO_DFB`, legacy c_27): the writer kernel bound it as both PRODUCER and
   CONSUMER — a shape Gen2 rejects for data-movement kernels. Converted to a `Scratchpad` per
   `ai/post_port/semantic/dm_self_loop_dfbs.md`.
2. **Gen1-only `hw_config`**: all three factories hand-write `ComputeGen1Config` (shape 4 of
   `ai/post_port/semantic/gen2_hardware_configs.md`); a kernel whose `hw_config` carries only a
   Gen1 config cannot run on Quasar at all. Added the arch-selected `ComputeGen2Config` branch,
   copying exactly the fields the Gen1 config sets.

Everything else audited clean — details below. No §7–§8 *reactive* fix was applied (no device run,
so none of those symptoms could fire; see the gotchas table).

## Metal 2.0 gate (§1 of quasar_porting.md) — PASSED

- All three program factories are `create_program_artifacts` → `ProgramArtifacts` with
  `DataflowBufferSpec` / `TensorParameter` / named bindings. No `create_descriptor`,
  `ProgramDescriptor`, `CreateKernel`, or `CreateCircularBuffer` anywhere in the op.
- All five kernels use the device-2.0 APIs (`api/dataflow/*`, `api/compute/*`, `Noc`,
  `DataflowBuffer`, `TensorAccessor`, `get_entry_size()`, `get_arg(args::…)`), and none use the
  legacy device API (`cb_*`, `noc_async_*` free functions, `get_local_cb_interface`,
  positional `get_arg_val`, address-RTA `TensorAccessorArgs`).

## Files changed (all inside the op directory)

| File | Reason |
|---|---|
| `device/rotary_embedding_llama_metal2_common.hpp` | `ZERO_DFB` (`DFBSpecName`) → `ZERO_SCRATCH` (`ScratchpadSpecName`); added `scratchpad_spec.hpp` include. Shared-vocabulary header, so the rename is made once for both binding factories. |
| `device/rotary_embedding_llama_multi_core_program_factory.cpp` | (a) `zero_dfb` `DataflowBufferSpec` → `ScratchpadSpec zero_scratchpad` (`size_per_node = entry_size × num_entries`); writer's two `ZERO_DFB` `DFBBinding`s → one `ScratchpadBinding`; `ProgramSpec` gains `.scratchpads`. (b) Gen2 `hw_config`: `ComputeGen2Config` branch behind `device->arch() == tt::ARCH::QUASAR`. |
| `device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp` | Same two changes (the writer kernel is shared, so the multi-bound zero spec converts as a unit across both factories, per the pass doc). |
| `device/rotary_embedding_llama_sharded_program_factory.cpp` | Gen2 `hw_config` branch only (this factory has no zero DFB — the lone compute kernel self-loops all its DFBs, which is legal on Gen2). |
| `device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` | Zero-buffer fake-FIFO → `Scratchpad<volatile uint32_t>` (translation below). |

Namespace and directory unchanged; nothing copied from or modeled on `experimental/quasar/`
(which was never opened).

### The DM self-loop conversion, argued

Survey (per `dm_self_loop_dfbs.md`): in both DM factories the only kernel binding `ZERO_DFB` is
the writer, with both roles → self-loop everywhere; the writer is a DM kernel; `borrowed_from`
unset; no `dfb_run_overrides` (neither factory sets any); every kernel use of the handle
enumerated and attributed:

| Old use | Translation |
|---|---|
| `reserve_back(Wt)` / `wait_front(Wt)` | deleted (pure self-waits, no state) |
| `get_write_ptr()` → `zero_tile_at` fill loop | `zero_stage[i] = 0` over `zero_stage.size()` (same bytes: `Wt` tiles; `operator[]` is bounds-checked, and `T = volatile uint32_t` carries the old `volatile tt_l1_ptr uint32_t*` view) |
| `push_back(Wt)` / `pop_front(Wt)` | deleted — **both indices are dead**: each pointer is only ever read at its initial (base) position (the write ptr once before the only push; the read ptr always before the single trailing pop), so no stride, no wrap, no index survives |
| `get_read_ptr()` (NOC source, ternary-shared with `dfb_out.get_read_ptr()`) | `zero_stage.get_base_address()` — the address form is genuinely needed because the same variable holds either buffer's address and feeds `CoreLocalMem<uint32_t>` as the NOC source |
| `get_entry_size()` | **The survey's one off-list use.** Resolved via the pass's sanctioned constant-reuse rule (not by inventing a value): the kernel already holds `tile_bytes = dfb_out.get_entry_size()`, and in **every** binding factory `zero`'s and `out`'s `entry_size` are the *same expression*, `output_single_tile_size` — so `zero_tile_bytes = tile_bytes` is exact, and introduces no new dependency on the program-cache key |

`data_format_metadata` dropped after checking every use: the writer only takes raw addresses /
NOC-sources the region; no LLK ever touches it (it is DM-only). Unused `zero_tile_at` helper
deleted with the FIFO scaffolding. NOC barriers untouched.

## Quasar-uplift audit checklist (quasar_audit.md + §7–§12)

- **Check 1 — CB/DFB redesign debt / self-loops:** `ZERO_DFB` was the only **DM** self-loop →
  converted (above). All other self-loops (`rotated_interm`/`cos_interm`/`sin_interm` in every
  factory; every DFB in the decode-sharded factory) live on a **compute** kernel — supported on
  Gen2, left as-is per the pass doc. All remaining DM-side DFBs are genuine cross-kernel FIFOs.
- **Check 2 — non-zero-init semaphores:** the op creates **no semaphores at all**. Clean.
- **Borrowed DFBs** (decode factory: input/cos/sin/trans_mat/out over resident shards;
  prefill-sharded factory: conditional cos/sin/trans_mat borrows): a supported Metal 2.0 construct
  on both generations (§6); capacities equal the per-shard tile counts the kernels push. Not a
  blocker; left as-is.

### §7 gotchas — considered

| Gotcha | Finding |
|---|---|
| `disable_dfb_implicit_sync_*` | Not used anywhere. Nothing to remove. |
| `compute_kernel_hw_startup` exactly once | **Violated in both compute kernels** (two back-to-back startups at `main()` start, one `SrcOrder::Reverse` for the matmul, one for the binary ops). This is a pre-existing Gen1 pattern already flagged in-source as `TODO(#52395)` ("call-once API"). Merging them is a base-port/API question whose safe resolution is not statically determinable and would change WH/BH un-guarded → **deferred, not edited** (see Deferred). |
| BFD re-init on DFB-id change | Both compute kernels re-run the relevant `*_init` (`matmul_init`, `mul_init`/`mul_bcast_rows_init`, `add_init`) before each operand switch. OK by inspection. |
| Tilize pack-config / wide-tilize | No tilize in this op. N/A. |
| Int32-only / no uint16-uint32 formats | Op validates all inputs `BFLOAT16`; no dtype-specific kernel branch. N/A. |
| RM shard 16-byte alignment | All paths are TILE layout. N/A. |
| Non-zero-init semaphores | None (no semaphores). |
| `evil_set_read/write_ptr` | Not used. |
| `fifo_page_size` / `get_local_cb_interface` | Not used; all sizes via `get_entry_size()`. |
| Kernel-side `data_format_metadata` validity | Every remaining DFB spec carries a real format. OK. |

### §8 pitfalls — reactive only, none applied (no device run)

- **§8.5 `reserve_back`→`push_back` with no intervening TDMA op:** the decode-sharded compute
  kernel has three such adjacent pairs on its borrowed self-loop DFBs (`trans_mat`, `sin`/`cos`,
  `input` — e.g. lines 54–55, 60–64, 73–74 of `rotary_embedding_llama_sharded.cpp`). This is a
  known Quasar HW hazard *symptom* (unpacker trap); the recipe says fix it only when it fires.
  Flagged here as the first thing to check if the decode path traps on Quasar.
- No other §8 signature (`0x19`, `0x10000`, `PACR0_TILE_INC`, credit stalls, …) can be evaluated
  statically. Nothing pre-emptively changed, per the recipe.

### §11 NoC / multicast

No multicast, no NOC1-direction tricks, no `MEM_ZEROS_BASE`, no L2-flush calls anywhere in the op.
The one zero-fill is a RISC store loop into local L1 (now the scratchpad), not a NOC op. Clean.

## Deferred / follow-up items

1. **Double `compute_kernel_hw_startup` (issue #52395)** in
   `device/kernels/compute/rotary_embedding_llama.cpp` (lines ~60–63) and
   `device/kernels/compute/rotary_embedding_llama_sharded.cpp` (lines ~49–51). §7 requires exactly
   one startup at `main()` start, and Quasar tolerates init-state abuse worse than WH/BH. Owned by
   the call-once-API issue; do not band-aid per-op.
2. **§8.5 adjacent reserve/push pairs** in the decode-sharded compute kernel — apply the
   "intervening IDMA op" fix only if the trap reproduces on Quasar (reactive by recipe).
3. **`unpack_modes` Quasar tuning (#52269)** — the Gen2 configs carry the Gen1-faithful (empty)
   table; markers left in all three factories per `gen2_hardware_configs.md`. (All formats are
   bf16, so nothing is *required*; optimization is #52269's job.)
4. No missing-feature flags for the runtime/LLK team: no construct the op needs is absent from the
   sanctioned Quasar API (no ring rewind, no non-zero-init semaphore, no DFB redesign left open).

## WH/BH parity claim (structural — no device run this session)

- **Gen2 `hw_config`:** the only behavioral branch added, guarded by
  `device->arch() == tt::ARCH::QUASAR`; the Gen1 `ComputeGen1Config` initializers are textually
  unchanged (only a `const` dropped to allow the guarded reassignment). WH/BH resolve to the
  identical config as before.
- **Zero-buffer Scratchpad conversion:** the canonical behavior-preserving pass — the writer
  performs the same zero-fill (same byte count), the same NOC writes in the same order from the
  same-sized region, with the same barriers; the deleted FIFO calls were self-synchronization
  no-ops (single single-threaded toucher). The only observable difference is the region's L1
  placement (scratchpads allocate alongside DFBs), which nothing functional depends on — per
  `dm_self_loop_dfbs.md` this is equivalent by construction on Gen1 and is what *enables* Gen2.
- No kernel signature, CTA/RTA schema, DFB entry sizing, `opt_level`, or placement changed.

## Test commands (user runs; recipe §9 — BH → WH → Quasar)

Parity on BH, then WH (unchanged behavior expected; force JIT since kernels changed):

```bash
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_rotary_embedding_llama.py
```

(Optional broader parity: `pytest tests/sweep_framework/sweeps/model_traced/rotary_embedding_llama_model_traced.py`.)

Quasar emulator (run with LLK asserts on first, then off; purge the JIT cache between eras):

```bash
rm -rf ~/.cache/tt-metal-cache
TT_METAL_LLK_ASSERTS=1 TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_rotary_embedding_llama.py
unset TT_METAL_LLK_ASSERTS && TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_rotary_embedding_llama.py
```

Delete this report before merge.
