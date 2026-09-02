# QUASAR_UPLIFT_REPORT — nlp_concat_heads_decode

**Op:** `ttnn.experimental.nlp_concat_heads_decode`
**Directory:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_concat_heads_decode.py`
(one captured case: input `[1,1,32,64]` bf16 TILE HEIGHT_SHARDED L1 on 1 core, `num_heads=32`,
output `[1,1,32,2048]` WIDTH_SHARDED L1 on an 8×4 grid — no `sub_core_grids`, so it exercises the
**default** `NLPConcatHeadsDecodeProgramFactory` path)
**Date:** 2026-09-02 (audit-only session; no builds or device runs performed — per recipe §9 the user runs all builds/tests)

> **This audit ran on the post-#54783-merge state.** PR #54783
> (`vsureshTT/Metal2_port_nlp_concat_heads_decode_v2`, the in-place Metal 2.0 port of both
> factories + both kernels) was merged into this branch before this audit; the previous
> RED ("Not Metal 2.0 yet") report at this path described the pre-merge state and is
> **obsolete — this report replaces it.**

---

## Status: GREEN

The op passes the uplift gate (it is genuinely Metal 2.0 on the test's path), and the
Quasar-uplift audit found **one** statically-determinable uplift item, which was applied:
the sync-free, output-borrowed `q_out` DFB was converted to a plain tensor binding +
`LocalTensorAccessor`, per the canonical post-port pass (`sync_free_dfbs.md`, borrowed case).
That conversion is behaviour-preserving on WH/BH and removes the one construct that was
statically wrong on the Quasar path (an uncached DFB pointer fed to NoC APIs — see below).
No other change was needed; no RED-stop condition fired.

### Gate evidence (Metal 2.0 confirmed, post-merge)

- **Both host factories** are `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts`
  with `ProgramSpec`/`KernelSpec`/`TensorBinding`/named `runtime_arg_schema`/`WorkUnitSpec`
  (`device/nlp_concat_heads_decode_program_factory.cpp`,
  `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp`). No
  `create_descriptor`/`ProgramDescriptor`, no `CBIndex::c_*`, no positional CTAs/RTAs anywhere.
- **Both kernels** use the device-2.0 kernel APIs: `api/dataflow/{dataflow_api,noc,endpoints}.h`,
  `Noc::async_read`/`async_read_barrier`, `TensorAccessor(tensor::input)`, named
  `get_arg(args::…)` via `experimental/kernel_args.h`, `get_vararg` (per-KernelSpec
  `num_runtime_varargs`). No legacy `dataflow_api.h` free functions, `cb_*`, or
  `get_local_cb_interface`.
- `hw_config` on all four kernel specs comes from the **arch-agnostic**
  `ttnn::create_reader/writer_datamovement_config(device->arch())` helpers
  (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`), which
  already return a default `DataMovementGen2Config{}` on `tt::ARCH::QUASAR` (implicit sync **not**
  disabled) and the conventional Gen1 reader/writer placement otherwise.

## Files changed (all inside the op directory)

The single audit finding and its one prescribed fix, applied symmetrically to both factory paths
(they share the identical pattern):

1. `device/nlp_concat_heads_decode_program_factory.cpp` — removed the `Q_OUT`
   `DataflowBufferSpec` (borrowed from OUTPUT) and both cosmetic 1P/1C `DFBBinding`s; bound
   OUTPUT to both kernels as `TensorBinding{OUTPUT, "q_out"}` instead; dropped the now-unused
   `q_num_tiles` local; updated comments.
2. `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp` — same conversion; also
   dropped now-unused `tile_h`/`tile_hw` locals (they only fed `q_num_tiles`).
3. `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp` — replaced
   `DataflowBuffer dfb_q_out(dfb::q_out)` + `get_write_ptr()` with
   `LocalTensorAccessor<uint8_t> q_out(tensor::q_out)` + `get_bank_base_address()`; include swap
   `api/dataflow/dataflow_buffer.h` → `api/tensor/local_tensor_accessor.h`. All address
   arithmetic, NoC calls, and loop structure untouched.
4. `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` — same
   kernel-side conversion.

No file outside the op directory was touched. Namespace and directory unchanged (nothing tempted
a move/rename).

### Why the conversion is required (not just style)

- **Sync-free criterion (`sync_free_dfbs.md`):** `q_out` was the op's only DFB. In both kernels
  its handle is used exactly once — `get_write_ptr()` — with **zero** occurrences of the six
  credit methods (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`pages_reservable_at_back`/
  `pages_available_at_front`), and the handle is never passed to any helper/guard/template
  (verified per "way 4"). The host comment itself declared the PRODUCER/CONSUMER split "cosmetic
  1P+1C to satisfy the validator" — the fake-FIFO shape `dm_self_loop_dfbs.md` explicitly calls
  worse than a self-loop. `borrowed_from = OUTPUT` ⇒ the prescribed end-state is
  **`LocalTensorAccessor`** (not `Scratchpad`).
- **Quasar semantics (the clinching, statically-determinable bug):** both kernels fed
  `get_write_ptr()`-derived addresses into `noc.async_read`'s local destination
  (`CoreLocalMem<uint32_t>(q_write_addr)`). On Quasar DM,
  `DataflowBuffer::get_write_ptr()/get_read_ptr()` return the **uncached alias**
  (`+ MEM_L1_UNCACHED_BASE`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:333–336`, per
  #52769), and that same header states "NOC APIs do not accept uncached addresses" (the cached
  getter is private, NOC-trait-only). So on the Quasar path the pre-conversion kernels handed the
  NoC an address it cannot take. `LocalTensorAccessor::get_bank_base_address()` returns the plain
  CRTA-carried L1 address on both generations, fixing Quasar without touching Gen1 behaviour.
- **Gen1 behaviour identity (why no `ARCH_QUASAR` guard is needed):** on WH/BH the borrowed DFB's
  write pointer base *is* the OUTPUT tensor's node-local L1 shard base — the same value the
  OUTPUT `TensorBinding`'s CRTA now delivers (a borrowed DFB allocates nothing, so L1 layout is
  also unchanged). No synchronization was deleted because none existed (`sync_free_dfbs.md`
  declares this pass behaviour-preserving on Gen1). Precedent for the resulting shape (kernels
  with tensor bindings and zero DFBs) exists in mainline:
  `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.cpp`.

## Audit checklist — gotchas applied vs. considered

**Applied (1):**
- **Sync-free borrowed DFB → `LocalTensorAccessor`** (`sync_free_dfbs.md` borrowed case), which
  simultaneously clears the **uncached-DFB-pointer-into-NoC** hazard (§5/§8.3/§12) — the files
  above.

**Considered, not needed (with why):**
- **DM self-loop DFB (§6 / `dm_self_loop_dfbs.md`):** `q_out` was *not* a self-loop (reader bound
  PRODUCER, writer CONSUMER — two distinct kernels); it was the neighbouring fake-FIFO shape,
  resolved by the sync-free pass above. No other DFB exists post-conversion (the op is now
  DFB-free).
- **Gen1-only hardcoded `hw_config` (`gen2_hardware_configs.md`):** zero-site pass. All four
  kernel specs are **shape 1** (arch-agnostic DM helpers that internally branch to
  `DataMovementGen2Config{}` on Quasar — same idiom as the arch-branched configs in
  reshard/sharded_to_interleaved on this branch, already packaged in the helper). No compute
  kernel ⇒ no `to_compute_hardware_config`, no `unpack_modes`, no `std::get<…Gen1Config>` /
  `get_if` sites (grep per that doc's Step 2 came back empty). No `unpack_modes` marker is
  applicable — all sites are DM (the doc says to state this explicitly rather than leave it
  inferred).
- **Non-zero-init semaphores (`quasar_audit.md` check 2):** the op creates/uses no semaphores at
  all.
- **`fifo_page_size` reads (§5/§8.3):** none (tile size is CTA-derived: `head_size /
  head_size_num_tiles`); `get_entry_size()` is now moot since no DFB remains.
- **Implicit-sync disables (§7):** none — the DM-config helpers are called with the
  `disable_dfb_implicit_sync_for_all` parameter defaulted to `false`; nothing sets
  `disable_dfb_implicit_sync_for_all`/`disable_implicit_sync_for`.
- **uint16/uint32 device-format branches (§7):** none — the kernels are element-size-driven
  (`element_size`/`SUBTILE_LINE_BYTES` CTAs); the driving case is bf16. The op merely forwards
  the dtype, so nothing to guard at op level.
- **`data_format_metadata` validity (§4):** the only DFB carried a valid bf16 format and nothing
  ever consulted it (DM-only op, no LLKs); the field disappears with the spec.
- **`evil_set_*` / DFB ring rewind (§7/§8.3):** not used.
- **Borrowed-DFB capacity (§6):** was already correct (`num_entries` = exact per-shard tile
  count); N/A after the conversion.
- **NoC/multicast on Quasar (§11):** unicast `async_read` only, coordinates supplied by the host
  per-arch via `worker_core_from_logical_core`; no multicast, no NOC0/NOC1 directional tricks, no
  degenerate-grid mcast corners. RM 16-byte shard-width alignment: N/A (TILE layout).
- **`compute_kernel_hw_startup` / tilize / DEST rules (§7/§8.4):** N/A — no compute kernel.
- **`opt_level` (§4):** left absent on all specs (resolves to O2 for DM, matching the legacy DM
  default) — a base-port setting the uplift correctly does not touch.
- **§7–§8 runtime-symptom fixes (hangs, `0x19`, credit stalls, …):** reactive by recipe rule;
  no device run happened in this session, so none applied.

## Deferred / follow-up items

1. **Feed the uncached-pointer check back into `quasar_audit.md`:** "a `get_read_ptr()`/
   `get_write_ptr()`-derived address passed to a `Noc` API" is a Gen1-legal-but-Quasar-broken
   construct this audit hit in the wild; the scaffold ("more checks land here") should gain it.
   Doc-only change outside the op directory — not made here per session constraints.
2. **Driving-test coverage note:** the captured Quasar test exercises only the default factory.
   The subcoregrids factory/kernel received the identical (behaviour-preserving) conversion but
   is exercised only by the mainline WH/BH test's `sub_core_grids` parametrizations — include
   those in the parity run (the command below covers both paths).
3. No missing-feature flags for the runtime/LLK team: the op needs no DFB rewind, no LLKs (no
   compute kernel), and no capability absent from the sanctioned Quasar API.

## Parity claim (WH/BH)

The diff is deliberately **not** `ARCH_QUASAR`-guarded because it is behaviourally identical on
Gen1, argued structurally (recipe §9, no device run in this session):
- Host: the removed DFB was borrowed (zero L1 allocation, zero sync resources on Gen1 given no
  credit ops ever ran) and its only consumer-visible artifact — the write-pointer base — equals
  the OUTPUT shard base now delivered through the tensor CRTA.
- Kernels: byte-for-byte the same NoC reads, sizes, and destination addresses; only the source of
  the base address changed (`dfb::q_out` write ptr → `tensor::q_out` CRTA). On WH/BH
  `L1_UNCACHED_OFFSET == 0`, so even the old getter returned this same plain address.
- Everything else in the op is untouched; `git diff` for the directory shows only the four files
  above (plus this report, uncommitted, to be deleted before merge).
The user should confirm with the parity commands below.

## Test commands (for the user to run — none were run in this session)

BH / WH parity (covers both factories, incl. `sub_core_grids` parametrizations; run on each Gen1
arch, unchanged results expected):
```
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py
```

Quasar (emulator; the captured-graph per-op test that drives this uplift — default factory path):
```
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_concat_heads_decode.py
```

Notes: `TT_METAL_FORCE_JIT_COMPILE=1` because kernels changed; also purge
`~/.cache/tt-metal-cache` between the pre-conversion baseline and post-conversion runs (stale
JIT-cache era). Run Quasar both with and without `TT_METAL_LLK_ASSERTS` per recipe §9.
