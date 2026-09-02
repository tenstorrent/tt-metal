# QUASAR_UPLIFT_REPORT — transpose (HC tiled interleaved path)

**Status: GREEN — no changes needed.**

The op path under uplift is already Metal 2.0 on Gen1, and the Quasar-uplift audit
(`quasar_audit.md` + `quasar_porting.md` §7–§12) surfaced **zero statically-determinable
Gen2 blockers or required fixes**. Per the recipe (§2, §7–§8 fixes are reactive), a
"no changes needed" GREEN is the expected valid outcome for an already-M2 op audited
without a device run; no changes were manufactured. **No source file was modified.**

## Scope: which path this covers

Driving test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_transpose.py` —
one case, `ttnn.transpose(t, 1, 2)` on a `[1, 1, 32, 64]` BFLOAT16 / TILE / DRAM-interleaved
tensor (40 captured calls, 1 signature).

Dispatch trace (code, not the audit MDs, as source of truth):

1. `transpose.cpp` → `transpose_impl`: rank 4, dims (1,2) → `TransposeOpDim::HC`
   (`transpose_dims[1][2]`). The identity short-circuit at
   `transpose.cpp:259` does **not** fire (it requires *both* padded dims == 1; here
   padded[1]=1 but padded[2]=32). Input is TILE and unsharded, so `detail::transpose_`
   falls through to `ttnn::prim::transpose` (no permute rewrite — that is RM-only for HC).
2. `device/transpose_device_operation.cpp::select_program_factory`: strategy
   `MULTI_CORE_HC`, not sharded, not row-major →
   **`TransposeHCTiledInterleavedProgramFactory`**.

In-scope files:

- `device/transpose_hc_tiled_interleaved_program_factory.cpp` (+ `.hpp`)
- `device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp`
- `device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`

(No compute kernel on this path — it is a two-DM-kernel data-movement program.)

For the test shape, logical C = 1 and `1 % 32 != 0` → `needs_padding` is **true**, so the
conditional `pad` DFB and the `NEEDS_PADDING`-gated blocks in both kernels ARE exercised
by this case — they were audited as in-scope (see below).

## Gate: Metal 2.0 on Gen1 — PASS

- Factory: `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts`,
  `ProgramSpec` with `DataflowBufferSpec` (named `in0`/`pad`), `TensorParameter`
  (`input`/`output`), named CTAs, `runtime_arg_schema` named RTAs,
  `AddRuntimeArgsForNode` per-node run args. No `create_descriptor`/`ProgramDescriptor`,
  no `CreateKernel`/`CreateCircularBuffer`.
- Reader (`..._metal2.cpp`): `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`,
  `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`,
  `experimental/kernel_args.h`; `get_arg(args::…)`, `TensorAccessor(tensor::input)`,
  `DataflowBuffer(dfb::cb_in0)`, `dfb.get_entry_size()`. No legacy device API.
- Writer: same device-2.0 header set plus `api/core_local_mem.h`;
  `TensorAccessor(tensor::dst)`, `DataflowBuffer(dfb::out0)` / `dfb::pad`,
  `CoreLocalMem` NOC sources. No `cb_*` free functions, no positional `get_arg_val`,
  no `get_local_cb_interface`, no address-RTA `TensorAccessorArgs`.

## Quasar-uplift audit results

### `quasar_audit.md` check 1 — device-side CB/DFB redesign debt: CLEAN
- `in0` DFB: PRODUCER = reader, CONSUMER = writer — two **distinct** DM kernels →
  normal cross-kernel FIFO. Not a DM self-loop (the Gen2-rejected shape), not sync-free,
  no `borrowed_from`, no `evil_set_read_ptr`/`evil_set_write_ptr` anywhere in the op.
- `pad` DFB (bound only when `NEEDS_PADDING`; bound for the test shape): same
  reader-PRODUCER / writer-CONSUMER cross-kernel FIFO, 1 entry, produced once,
  consumed once. No redesign debt; nothing needing Scratchpad/LocalTensorAccessor.

### `quasar_audit.md` check 2 — non-zero-init semaphores: CLEAN
No semaphores of any kind on this path (grep over the whole op directory: no
`SemaphoreSpec`, no `CreateSemaphore`).

### §7 gotchas — considered, none applicable
- **Implicit-sync disable flags**: absent (no `disable_dfb_implicit_sync_for_all` /
  `disable_implicit_sync_for` anywhere in the op). ✓
- **Explicit-sync/NOC interaction**: reader does `reserve_back` →
  `noc.async_read(s, dfb, …)` → `push_back` **without** `NocOptions::TXN_ID`, so the
  implicit-sync ISR path is not engaged (per `DataflowBuffer.md` Part A/C4, implicit
  sync triggers only on TXN_ID-tagged transfers). No static double-count exposure.
- **`compute_kernel_hw_startup` / tilize / DEST rules**: no compute kernel on this path — N/A.
- **Int32-only / no uint16/uint32 on Quasar**: the kernels have **no** dtype-format
  branch (writer works in `element_size` bytes; reader moves whole tiles). The factory's
  host-side `padding_val_packed` switch handles UINT16/UINT32 *inputs*, but per
  `quasar_porting.md` §7 an op that merely forwards a `DataType` has nothing to guard —
  the limitation lives at the format/LLK layer. Test case is BFLOAT16. Nothing to guard.
- **RM shard width 16-byte alignment**: TILE path, no RM shards — N/A.
- **Non-zero-init semaphores**: none (above).
- **`evil_set_*` ring rewind**: not used.
- **Hand-rolled Quasar-only device interfaces**: none.

### §8 pitfalls — considered, applied reactively only (no device run → none applied)
- **§8.3 manual L2 flush/invalidate leftovers (the transpose-specific note)**: checked
  explicitly per the session brief — **no** `invalidate_l2_cache_range` /
  `flush_l2_cache_range` calls exist anywhere under
  `device/kernels/` (the a00dd45/#52769 uncached-getter cleanup already landed here;
  nothing left to remove).
- **§8.3 `fifo_page_size` staleness**: reader uses `DataflowBuffer::get_entry_size()`;
  no `get_local_cb_interface` in the op. ✓
- **Writer `get_read_ptr()` → `CoreLocalMem` → `noc.async_write` source**: verified
  sanctioned on Quasar — `dfb.get_read_ptr()` returns the uncached alias on Quasar DM
  (`api/dataflow/dataflow_buffer.h:336`), and `Noc::get_src_ptr` maps any
  `MEM_L1_UNCACHED_BASE` alias back to the cached view before it reaches the NOC
  (`api/dataflow/noc.h` `l1_cached_view`, `#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)`).
  No op-side change needed or allowed here.
- All other §8.1/§8.2/§8.4/§8.5 rows are build/runtime symptoms (compute/tilize/matmul/
  pool-specific, or shared-header skew) that cannot fire statically for a two-DM-kernel
  unicast op; none of their constructs appear in the in-scope files.

### §11 NoC/multicast — CLEAN
Unicast `async_read`/`async_write` only; no multicast, no NOC1 reverse-direction
tricks, no `MEM_ZEROS_BASE`, no grid-corner arithmetic. `hw_config` comes from
`create_reader/writer_datamovement_config(device->arch())` — arch-parameterized shared
helpers, so Quasar resolves its own DM config with no op-side variant needed.

### Other factory checks
- Both DFBs carry a valid `data_format_metadata` (from the input dtype). ✓
- `opt_level` absent on both DM KernelSpecs → resolves to the O2 DM default, matching
  the legacy DM default — correct per the recipe; not an uplift edit. ✓
- No `-Werror=reorder-init-list` risk observed (designated initializers follow
  `KernelSpec` field order as in sibling landed factories).

## Files changed

**None.** (Deliberately: the audit found no statically-determinable, clearly-required
fix; §7–§8 fixes are reactive and no device run was performed in this session.)

## Deferred / follow-up items

1. **Reactive Quasar bring-up**: if the Quasar run hits a credit stall / double-count
   on `in0` or `pad`, that is an implicit-sync runtime regression per §7/§8.2 — report
   to the runtime team; do not disable implicit sync and do not edit the op.
2. **Out-of-scope sibling factories still carrying legacy idioms** (not this test's
   path, not touched): `transpose_hc_sharded_program_factory.cpp`,
   `transpose_wh_sharded_rm_program_factory.cpp` and several kernels
   (`reader/writer_unary_transpose_wh_sharded_rm.cpp`, `transpose_wh*.cpp` compute
   kernels, legacy `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`)
   still use `tt::CBIndex::c_*` tokens. They need their own Metal-2.0/Quasar passes
   before any sharded/WH transpose case can be uplifted; flagged here, not fixed.
3. No missing-feature flags for the runtime/LLK team arose from this path.

## WH/BH parity claim (argued structurally — no device run this session)

The diff for this uplift is **empty** (zero source changes), therefore WH/BH behavior
is unchanged by construction. Per `quasar_porting.md` §9, confirm with:

- **BH/WH parity (user runs, per arch):**
  - `pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py -k "hc"`
    (covers `test_transpose_hc_unit`, `test_transpose_hc`, `test_transpose_hc_padded_c`,
    HC program-cache and sharded variants), or the full file without `-k` for the
    complete transpose suite.
- **Exact captured-case check (any arch):**
  - `pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_transpose.py`
- **Quasar (simulator, per the craqsim runbook env):**
  - `pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_transpose.py`
  - Run both with `TT_METAL_LLK_ASSERTS` set and unset; use
    `TT_METAL_FORCE_JIT_COMPILE=1` if kernels were touched in the same tree, and purge
    `~/.cache/tt-metal-cache` between pre/post-change runs.

## RED-stop conditions — checked, none apply

Not-M2 (no — gate passed), missing sanctioned API (none needed), owner-decision
construct (none: no semaphores, no self-loops, no open-HW-bug construct on this path),
unguardable fix (no fix needed), stub LLK (no LLKs used — DM-only path).
