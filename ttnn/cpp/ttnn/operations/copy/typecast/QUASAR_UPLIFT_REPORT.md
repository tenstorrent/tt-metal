# QUASAR_UPLIFT_REPORT — ttnn.typecast (`ttnn/cpp/ttnn/operations/copy/typecast/`)

> Uncommitted review artifact per `docs/source/ttnn/ttnn/ai/quasar_porting.md` — delete before merge.
> Driven by `models/experimental/llama32_1b_quasar/tests/graph_ops/test_typecast.py`
> (`ttnn.typecast`, BFLOAT16→BFLOAT8_B and BFLOAT16→BFLOAT16, TILE, DRAM interleaved →
> `TypecastProgramFactory`). Audit covered the whole op (all four factories, all six kernels).

## Status: GREEN

The op is already Metal 2.0 on Gen1 (gate §1 passes), the Quasar-uplift audit
(`quasar_audit.md` checks 1–2 plus §7–§12 of `quasar_porting.md`) is clean on the path the model
test exercises, and the uplift needed exactly one statically-determinable fix — an
uncached-address-to-NoC bug on the ROW_MAJOR chunked path — applied `#ifdef ARCH_QUASAR`-guarded
in two kernels. No device run was possible in this session, so every §7–§8 *reactive* fix was
considered but deliberately not applied (see below); parity is argued structurally and the exact
test commands are handed to the human.

### Metal 2.0 gate (why the uplift could start)

- All factories are `create_program_artifacts` → `ProgramArtifacts` with `dfb::`/`args::`/`tensor::`
  bindings: `device/typecast_program_factory.cpp` (interleaved + subgrid),
  `device/typecast_sharded_program_factory.cpp`, `device/typecast_rm_chunked_program_factory.cpp`.
- All six kernels use the device-2.0 APIs (`api/dataflow/*`, `api/compute/*`, `DataflowBuffer`,
  `Noc`, `TensorAccessor(tensor::…)`, `get_arg(args::…)`, `get_entry_size()`); no
  legacy `cb_*`, positional `get_arg_val`, address-RTA `TensorAccessorArgs`, or
  `get_local_cb_interface` anywhere in the op (grep-verified).

## Files changed (each `#ifdef ARCH_QUASAR`-guarded; WH/BH branch byte-identical to the original)

| File | Reason |
|---|---|
| `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | Passed `CoreLocalMem(get_write_ptr())` to `noc.async_read` — on Quasar DM `get_write_ptr()` returns the UNCACHED L1 alias (`L1_UNCACHED_OFFSET = MEM_L1_UNCACHED_BASE`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:335,376`) and "NOC APIs do not accept uncached addresses" (ibid.:381); `noc_traits_t<CoreLocalMem>` forwards the raw address unchanged. Quasar branch passes the DFB endpoint (`noc.async_read(s, dfb_in, nbytes, …)`), whose `noc_traits_t<DataflowBuffer>` resolves the private cached address — the same shape this op's interleaved reader already uses and the `cb_dfb_api_whitelist.md` preferred Class-1 form. Two sites (full + partial chunk). |
| `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | Same bug on the write side: `CoreLocalMem(get_read_ptr())` → `noc.async_write`. Quasar branch passes `dfb_out` as the NoC source, matching the interleaved writer. Two sites (full + partial chunk). |

No host/factory file changed. Nothing moved or renamed; namespace `ttnn::prim` /
`ttnn::operations::copy` untouched; nothing copied from or into `experimental/quasar/`.

Note the fix uses the plain sized `async_read/write(…, dfb, nbytes, …)` overloads — **not** the
`NocOptions::TXN_ID` implicit-sync DFB overloads — so the explicit `reserve/push` / `wait/pop`
FIFO pattern in these kernels is preserved on both branches; only the address the NoC receives
changes, and only on Quasar.

## Audit findings — `quasar_audit.md` checks

1. **Device-side CB/DFB redesign (check 1 / `cb_dfb_quasar_audit_helper.md`):**
   - Interleaved, subgrid, rm_chunked factories: `in` (reader→compute) and `out` (compute→writer)
     are canonical two-kernel Class-1 FIFOs. No redesign debt, no `NEEDS-DESIGN-DECISION`.
   - Sharded factory: `in`/`out` DFBs are `borrowed_from` the input/output shards (Class 6
     borrowed, already DFB-native). `out` is bound PRODUCER **and** CONSUMER by the *compute*
     kernel — a **compute self-loop**, which §6 says is legal on both Gen1 and Gen2 (only a *DM*
     self-loop is rejected); producer set == consumer set == {compute}. No change.
   - No `evil_set_*` / cursor surgery anywhere; no `get_local_cb_interface` (GATE clean); sizes
     read via `get_entry_size()`, never `fifo_page_size`.
2. **Non-zero-init semaphores (check 2):** the op creates no semaphores at all. Clean.

## §7–§8 gotchas: applied vs. considered

**Applied (statically determinable):**
- §8.3/§12 uncached-address family ("NOC APIs can't take uncached addresses on Quasar DM") — the
  two rm_chunked kernel fixes above. This is the one place the op's own code is statically wrong
  on Quasar; everything else below was checked and needs nothing.

**Considered, not needed (and why):**
- **§7 implicit-sync disable:** `disable_dfb_implicit_sync_for_all`/`disable_implicit_sync_for`
  absent everywhere; none added.
- **§7 `compute_kernel_hw_startup` exactly once:** `eltwise_typecast.cpp` calls it once at
  `main()` start. `copy_init(dfb::in)` once; the kernel never switches the DFB it operates on, so
  no re-init-on-DFB-change concern. `TYPECAST_LLK_INIT()` per tile is the SFPU-init-by-design of
  this kernel (identical on Gen1), not a `hw_configure`.
- **§7 Int32-but-no-uint16/uint32:** typecast merely *forwards* the DataTypes into the
  `TYPECAST_LLK`/`TYPECAST_LLK_INIT` template parameters — §7 names typecast as exactly the op
  with **nothing to guard**; the limitation lives at the format/LLK layer (deferred item below).
  The host-side dtype policy branches in `typecast.cpp` (UINT8/UINT16/UINT32 handling for
  `fp32_dest_acc_en`/`preserve_fp32_precision`) are host code — `ARCH_QUASAR` is a device-compile
  define, and they select precision policy, not a device format path.
- **§7 non-zero-init semaphores / §7 `evil_set_*` ring rewind:** neither construct exists here.
- **§6 DM self-loop → Scratchpad/LTA:** no DM self-loop (see audit finding 1).
- **§8.1 build-skew signatures:** no `common.hpp` (data_movement) include, no `REDUCE_OP`, no
  int-to-pointer casts, no `MEM_ZEROS_BASE`, no raw semaphore reads in the op's kernels. Nothing
  to pre-guard; remaining §8.1 rows are reactive build fixes.
- **§8.2 hangs / §8.5 reserve→push / wait→pop TDMA hazard:** the compute kernel already has
  `copy_tile` between `wait_front`→`pop_front` and `pack_tile` between `reserve_back`→`push_back`.
  All other §8.2 rows are reactive (no device run this session); none applied.
- **§8.3 `fifo_page_size` staleness:** kernels use `DataflowBuffer::get_entry_size()` already.
- **§8.3 manual L2 flush/invalidate:** none present — already conforms to the post-`a00dd45` model.
- **§8.4 tilize / reduce face-geometry rows:** no tilize, no reduce in this op.
- **§11 NoC/multicast:** no multicast, no reverse rectangles, no NOC0/NOC1 directional tricks;
  the RM writer writes exact per-chunk byte counts (no H-only tail padding pattern).
- **§4 `unpack_modes` for Float32 consumers under `enable_32_bit_dest`:** already explicitly
  handled in all three factory files, including the self-looped output DFB in the sharded factory.
- **§4 `opt_level`:** compute kernels carry explicit `O3` (legacy compute default); DM kernels
  leave it absent (→ O2, legacy DM default). Base-port concern, correct as-is; not an uplift edit.

## Deferred / follow-up items

1. **uint16/uint32/uint8 typecasts on Quasar (format/LLK layer, not this op):** Quasar has Int32
   but no uint16/uint32 device formats. Typecast forwards its dtypes into the LLK, so any
   cast involving UINT8/UINT16/UINT32 will fail at the format/LLK layer on Quasar. Per §7 there
   is nothing to guard in the op — flagged for the format/LLK owners. The model test's cases
   (BFLOAT16→BFLOAT8_B, BFLOAT16→BFLOAT16) are unaffected.
2. **Pre-existing unused `#include "api/debug/dprint.h"`** in
   `device/kernels/dataflow/reader_unary_sharded_metal2.cpp` — harmless, pre-dates this uplift;
   left untouched to keep the diff Quasar-only (a cleanup for the base-port owner, not this PR).
3. **No public cached-address getter on Quasar DM:** `get_noc_read/write_addr()` is private by
   design; the sanctioned interop is passing the DFB endpoint to `Noc` (done here). No
   missing-feature flag needed for this op — recorded only in case another op needs a raw cached
   address the traits can't express.

## WH/BH parity claim (structural — no device run this session)

The entire diff is two kernel files, and in both every change is an
`#ifdef ARCH_QUASAR … #else … #endif` where the `#else` branch is character-identical to the
pre-uplift code (the only motion is `get_*_ptr()` acquisition dropping into the `#else`, still
executed in the same position). `ARCH_QUASAR` is defined only for Quasar device builds, so the
WH/BH JIT compiles the original token stream: **zero WH/BH behavior change by construction.**
No host file, factory, spec, or hw_config changed, so program hashes and dispatch on Gen1 are
untouched. Confirm with the parity commands below.

## Test commands (user runs all builds/tests — recipe §9)

Kernels changed → force JIT and/or purge `~/.cache/tt-metal-cache` between baseline and post-uplift runs.

**BH / WH parity (must be identical to pre-uplift baseline):**
```bash
# Op unit tests — interleaved TILE (touched path's op, untouched branch) + sharded
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_sharded.py
# ROW_MAJOR chunked path — the two edited kernels (WH/BH take the #else branch)
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_copy_ops.py -k "typecast"
```

**Quasar (emulator, per the craq-sim runbook env):**
```bash
# The driving model test (TILE interleaved -> TypecastProgramFactory)
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_typecast.py
# RM chunked path exercising the ARCH_QUASAR branches (run what the emulator env supports)
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_copy_ops.py -k "typecast_row_major"
```
Run Quasar both with `TT_METAL_LLK_ASSERTS` on and unset (§9).

## RED-stop conditions checked

Not Metal 2.0? No — fully M2. Missing sanctioned API? No — the one gap (cached-address interop)
has a sanctioned in-API answer (DFB endpoints). Owner-decision construct? No — the only self-loop
is a compute self-loop (Gen2-legal). Un-guardable fix / experimental-quasar copy? No — both fixes
guarded in place. Stub LLK? Typecast LLK is format-driven; the uint* format gap is flagged, not
stubbed. → GREEN stands.
