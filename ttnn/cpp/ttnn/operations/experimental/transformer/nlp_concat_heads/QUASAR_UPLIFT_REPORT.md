# QUASAR_UPLIFT_REPORT — ttnn.experimental.nlp_concat_heads

- **Op:** `ttnn::experimental::nlp_concat_heads` (device op `ttnn::experimental::prim::NLPConcatHeads*`)
- **Op directory:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/`
- **Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_concat_heads.py`
  (one captured signature: input `[1, 32, 1024, 64]` BFLOAT16 / TILE / INTERLEAVED-DRAM →
  output `[1, 1, 1024, 2048]` — the **non-sharded interleaved** factory path)
- **Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` + `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md`
- **Date:** 2026-09-02
- **Audited state:** the **post-#54782-merge** tree (branch `vsureshTT/Metal2_port_nlp_concat_heads_v2`
  merged into `vsuresh/quasar-porting-recipe`). This report **supersedes** the earlier 2026-09-01 RED
  report, which was written against the pre-merge, pre-Metal-2.0 op and is now stale.

## Status: GREEN — one `ARCH_QUASAR`-guarded fix applied; WH/BH structurally unchanged

**Metal 2.0 gate: PASS.** The merged op is genuinely Metal 2.0 on every factory/kernel of the
driving test's path (and the sharded path too):

- Host factory (`device/nlp_concat_heads_program_factory.cpp:21`):
  `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts`; `KernelSpec` /
  `DataflowBufferSpec` / `TensorParameter` / `WorkUnitSpec` with named `dfb::` / `tensor::` bindings
  and a named `runtime_arg_schema`. No `create_descriptor` / `ProgramDescriptor`, no numeric CB
  indices, no address-RTAs, no `TensorAccessorArgs` CTAs anywhere in the op.
- Non-sharded reader (`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp`):
  device-2.0 APIs only (`api/dataflow/*`, `experimental/kernel_args.h`, `Noc`,
  `DataflowBuffer(dfb::in0)`, `TensorAccessor(tensor::src)`, `get_arg(args::…)`,
  `get_entry_size()`). No `dataflow_api.h`-legacy `cb_*`/`noc_async_*` free functions,
  no `circular_buffer.h`, no positional `get_arg_val`.
- Sharded kernel (`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`):
  same device-2.0 vocabulary (`DataflowBuffer`, `UnicastEndpoint`, `get_arg(args::…)`,
  `get_entry_size()`).
- Non-sharded writer is the shared Metal 2.0 fork
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  (`dfb::out` / `tensor::dst`, DFB-as-NoC-operand — the whitelisted Class-1 form). Clean; not
  touched (outside the op directory, and needs nothing).

**Quasar-uplift audit: one real site found and fixed** (the §8.3/§12 uncached-address family, on the
driving test's exact path); everything else on the checklist is clean. Details below. No builds or
device runs were performed this session (per constraints); exact parity/Quasar commands are at the end.

## Files changed

| File | Reason |
|---|---|
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` | **Uncached-DFB-pointer-to-NoC fix (`#ifdef ARCH_QUASAR`-guarded).** The kernel passed `CoreLocalMem<uint32_t>(dfb_in0.get_write_ptr() + manual advance)` as the `noc.async_read` destination. On Quasar DM `get_write_ptr()` returns the **uncached** L1 alias (`L1_UNCACHED_OFFSET = MEM_L1_UNCACHED_BASE`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:335-337`) and "NOC APIs do not accept uncached addresses" (ibid.:383-385, the getters are private); `noc_traits_t<CoreLocalMem>` forwards the raw address unchanged. The Quasar branch passes the DFB endpoint (`noc.async_read(s0, dfb_in0, …, {.page_id = …}, {})`), whose `noc_traits_t<DataflowBuffer>` resolves the private **cached** address — behaviorally equivalent because between `reserve_back(1)` and `push_back(1)` the DFB's current write pointer equals the Gen1 branch's manually tracked address (both advance by `single_tile_size_bytes` per tile; the per-block re-capture matches the ring wrap, since `per_tensor_tiles` divides the 2× ring). Same idiom and comment wording as the accepted typecast uplift on this branch (`copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp:38-50`) and the same shape the shared metal2 writer already uses. The now-Gen1-only `l1_write_addr` declaration/capture/advance are `#ifndef ARCH_QUASAR`-guarded so the Quasar TU has no set-but-unused variable. **One site.** |

No other file was changed. The op keeps its directory and namespace; nothing tempted a move or
rename. This report is the only other artifact (uncommitted; delete before merge).

## Audit checklist — applied vs considered

**Applied:**
- **Uncached DFB pointer fed to a NoC API (§5 / §8.3 / §12):** the one site above, on the driving
  test's path. Statically determinable (the uncached alias and the NOC-API restriction are stated in
  `dataflow_buffer.h` itself), precedent-backed, and guarded.

**Considered, no change needed:**
- **DM self-loop DFBs (`dm_self_loop_dfbs.md` / quasar_audit check 1):** none. Spec-level test:
  `IN0_DFB` non-sharded — reader binds PRODUCER only, writer (the shared metal2 writer) binds
  CONSUMER only → genuine cross-kernel FIFO. Sharded — `IN0_DFB`/`OUT0_DFB` are each bound PRODUCER
  by the reader-config instance and CONSUMER by the writer-config instance (two distinct
  `KernelSpec`s); no `KernelSpec` binds both roles on either DFB. Not sites.
- **Sync-free DFBs (`sync_free_dfbs.md`):** the **sharded** path's two DFBs are sync-free raw-peekers
  (no FIFO calls; the factory comment at `nlp_concat_heads_program_factory.cpp:105-107` says the
  P/C labels only satisfy the 1P+1C invariant). Both have `borrowed_from` set (resident input/output
  shards), so a `Scratchpad` conversion is explicitly out of scope (borrowed memory is a stop-and-report
  in both post-port docs; `LocalTensorAccessor` territory, a design decision). **Off the driving
  test's path** — recorded as deferred item 1, not changed.
- **Gen1-only hardcoded hw_configs (`gen2_hardware_configs.md`):** zero-site pass. Survey grep over
  the op (`hw_config|to_compute_hardware_config|Gen1Config|std::get<|std::get_if<|holds_alternative`)
  finds only the two shape-1 helper calls `ttnn::create_reader/writer_datamovement_config(arch)`
  (factory lines 140/151/176/196), which already return `DataMovementGen2Config{}` on Quasar — the
  same idiom as `sharded_to_interleaved`'s factory on this branch. **DM-only op, no compute kernel**,
  so per the doc there is no `unpack_modes` marker anywhere and its absence is not a signal.
- **Non-zero-init semaphores (quasar_audit check 2):** the op creates no semaphores at all. Clean.
- **`fifo_page_size` reads (§5 / §8.3):** none; all three kernels use
  `DataflowBuffer::get_entry_size()`.
- **`evil_set_read_ptr`/`evil_set_write_ptr` (§7):** none.
- **Implicit-sync disables (§7):** none — the DM helpers are called without the
  `disable_dfb_implicit_sync_for_all` flag (defaults `false`); no `disable_implicit_sync_for`
  anywhere.
- **uint16/uint32 format branches (§7):** none. The op forwards the input dtype
  (`datatype_to_dataformat_converter(a.dtype())`); no format-specific kernel branch. The captured
  case is BFLOAT16. Nothing to guard.
- **`data_format_metadata` validity (§4):** both `DataflowBufferSpec`s carry the real
  `data_format` (factory lines 210/220); never `Invalid`.
- **NoC / multicast (§11):** no multicast; the non-sharded path is unicast interleaved reads/writes.
  (Sharded-path L1→L1 loopback: see deferred item 1.)
- **RM shard-width 16B alignment (§7):** TILE-layout op; N/A.
- **opt_level (§4):** base-port concern, not an uplift edit — both DM `KernelSpec`s leave
  `compiler_options` at default (absent → O2 DM), matching the legacy resolution. No compute kernel.
  Nothing to flag to the base port.
- **§7–§8 runtime-symptom fixes (hangs, `0x19`, MOP timeouts, double-count, etc.):** applied only
  reactively by design; **no device run was available this session**, so none were applied. One
  specific watch item is recorded below (deferred item 2).

## Deferred / follow-up items

1. **Sharded path (off the driving test's path) — needs a design pass before any Quasar sharded run:**
   `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (a) feeds `dfb_in0.get_read_ptr()`-derived
   addresses into a `UnicastEndpoint{.addr = …}` NoC source and `dfb_out0.get_write_ptr()`-derived
   addresses into a `CoreLocalMem` destination — the same uncached-alias problem as the fixed reader,
   but **not** a one-token DFB-operand substitution (both sides carry manual per-head strided
   arithmetic against sync-free borrowed DFBs, and there is no public cached-address getter for an
   endpoint `.addr`); (b) is a **local L1→L1 NoC loopback on the same core**, which §6 warns can spin
   on `can_post` or silently drop on the emulator — the recipe's suggested end-state is a direct RISC
   L1→L1 copy; (c) its borrowed sync-free DFBs would need the `LocalTensorAccessor`-style
   classification (`sync_free_dfbs.md` stops on `borrowed_from`). All three resolve together in one
   redesign; per this session's constraints (statically-clear fixes only, driving path only) they are
   deferred, matching the verdict the `data_movement/sharded` uplift report on this branch reached
   for the identical patterns.
2. **Implicit-sync interaction watch item (runtime-owned):** the Quasar branch of the fixed reader
   passes the DFB straight to `noc.async_read` while keeping the explicit
   `reserve_back`/`push_back` — the same mixed shape as the typecast uplift and the mainline shared
   metal2 writer (`wait_front`/`pop_front` + DFB operand). §8.2's explicit+implicit double-count
   rows are marked historical/fixed; if a credit stall or double-count fires on the first Quasar run,
   **report it to the runtime team as a regression — do not disable implicit sync** (§7).
3. **Shared writer kernel** (`eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp`) is
   outside this op's directory and needed no change (already the preferred Class-1 form); its
   consolidation with the typecast duplicate is tracked separately (#52228). No shared-file edit was
   required this session.
4. No missing-Quasar-feature flags to raise for the runtime/LLK team from this op: no LLKs (DM-only),
   no `evil_set_*` dependency, no non-zero-init semaphores.

## WH/BH parity claim (argued structurally — no device run this session)

The entire diff is inside one kernel file and **every changed line is `#ifdef ARCH_QUASAR` /
`#ifndef ARCH_QUASAR` guarded**: the Gen1 (WH/BH) preprocessed TU contains exactly the pre-merge
code — same `CoreLocalMem` NoC destination, same `l1_write_addr` capture and advance, same FIFO
calls, same barriers — so WH and BH take the original path unchanged, byte-for-byte after
preprocessing. The host factory, device operation, sharded kernel, and shared writer are untouched.
No behavior change is possible on Gen1; the guard is the proof, pending the control runs below.

### Test commands for the user (not run this session)

BH/WH parity (must pass unchanged; force JIT since a kernel changed, and purge the JIT cache if a
pre-merge baseline was ever run from this tree):

```bash
# non-sharded (the changed kernel's path) + sharded coverage
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -k nlp_concat_heads
```

Quasar (craq-sim emulator, per the sim runbook) — the driving graph-op test:

```bash
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_concat_heads.py
```

Run the Quasar test with `TT_METAL_LLK_ASSERTS` both set and unset (recipe §9). If it hangs with a
credit stall on `in0`, see deferred item 2 before touching anything.
