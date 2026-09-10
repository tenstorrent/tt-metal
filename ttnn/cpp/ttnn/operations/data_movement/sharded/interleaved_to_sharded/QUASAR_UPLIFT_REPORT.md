# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded`

**Date:** 2026-09-10
**Branch:** `vsureshTT/quasar_uplift_round_2` (working tree shared with other op sessions; nothing committed)
**Recipe:** `quasar_porting.md` (Quasar Uplift field notes) + `metal_2.0/ai/audit/quasar_audit.md`
+ `metal_2.0/ai/audit/cb_dfb_quasar_audit_helper.md`, read against `metal_2.0/ai/audit/metal2_audit.md`,
`metal_2.0/ai/port/metal2_port.md`, `metal_2.0/ai/shared/*`, `metal_2.0/ai/post_port/*`.
Recipe copies were read from a read-only reference directory; the in-tree
`docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` holds only `analyses/`, so the
recipe version cannot be pinned with the recipe's `git log` provenance command.

---

## Status: RED — stop the uplift (no in-place edits made)

**Primary reason (RED-stop condition #1 of `quasar_porting.md` §1): the op is not Metal 2.0 on Gen1.**

The only program factory,
`device/interleaved_to_sharded_program_factory.cpp` (`InterleavedToShardedProgramFactory`), is on the
legacy **`ProgramDescriptor`** API, not `create_program_artifacts` / `ProgramArtifacts`:

| Legacy construct still present | Where |
|---|---|
| `ProgramDescriptor InterleavedToShardedProgramFactory::create_descriptor(...)` | `interleaved_to_sharded_program_factory.cpp:58`, `.hpp:14` |
| `CBDescriptor` / `CBFormatDescriptor` / `desc.cbs` (helper `push_i2s_cb_pair`) | `.cpp:30-48` |
| Magic CB indices `tt::CBIndex::c_0` / `c_1` / `c_16` passed as positional CTAs | `.cpp:143-145, 200, 207, 220` |
| Host-side `TensorAccessorArgs(*src_buffer).append_to(...)` plumbing | `.cpp:201, 208, 231` |
| `Buffer*` pushed into `KernelDescriptor::RTArgList` as the address RTA | `.cpp:291, 305, 388, 406` |
| `KernelDescriptor` + `Reader/Writer/ComputeConfigDescriptor` | `.cpp:195-246` |

And every kernel the factory binds is on the **legacy positional argument convention**
(`get_arg_val<uint32_t>(N)`, `get_compile_time_arg_val(N)`, `TensorAccessorArgs<N>()`, CB ids
constructed from a CTA), not the Metal 2.0 binding tokens (`dfb::` / `args::` / `tensor::`). The kernels
*are* on the Device 2.0 kernel API (`Noc`, `DataflowBuffer`, `TensorAccessor`) — which is the prerequisite
for a Metal 2.0 port, not the port itself.

`quasar_porting.md` §1 step 1 and the RED list are explicit for this case: *"Not Metal 2.0 on Gen1 yet —
factory still `create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first (`ai/port/metal2_port.md`);
this doc starts from an already-M2 op."* The uplift procedure therefore stops here. A Metal 2.0 port is a
separately gated job (`metal2_audit.md` GREEN → explicit user go-ahead → `metal2_port.md`, a fresh primary
session per the human README) and is **not** something this uplift session may perform or fold in.

**Secondary reason (RED-stop condition #3, "a construct needs an owner decision"), on the ROW_MAJOR path
only:** the alignment scratch CB (`c_1`) is a DM-kernel single-toucher scratch that the reader drives with
`reserve_back`/`push_back` and no consumer, sized off `get_local_cb_interface(cb_id_in1).fifo_page_size`,
and drained through a NoC *self-loopback* read. All three constructs are Gen1-legal and Quasar-hostile (details
in §3 below). On Gen2 this buffer must become a `Scratchpad` with the FIFO bookkeeping removed — a semantic
redesign (`dm_self_loop_dfbs.md`) that the recipe routes to the op owner, not the uplift.

**Both reasons are recorded and the uplift stops. Nothing was edited.** A RED here is the audit doing its
job, not a failed port.

### Two recent commits confirm the op is *staged for* Metal 2.0, not on it

- `e1426777c4d` (#55407, "Resolve I2S and Slice Issues for Metal 2.0") — pre-port hardening: hash the logical
  shape, drop the never-read `keep_l1_aligned` from the hash.
- `37fb77a975d` (#55495, "Hash Tensor Specs in I2S") — keys the op on whole `TensorSpec`s so a future
  `TensorParameter` strict match agrees with the cache key. Its PR text says outright: *"There's no issue
  today because the op builds a `ProgramDescriptor` and declares no `TensorParameter`... After the op gets
  ported though..."*
- The only Metal 2.0-shaped i2s work found is the closed draft PR #48250 (`dgomez/metal2-spec-runargs-builders`),
  which migrated the **`experimental/quasar/`** copy, not this op.

So the mainline op has had its op-owner pre-port fixes and is a natural next candidate for the
`metal2_audit.md` → `metal2_port.md` pipeline; it just has not been through it.

---

## 1. Files changed

**None.** No source file in or outside the op directory was modified. This report is the only artifact,
left uncommitted for review (delete before merge, per the recipe).

Nothing tempted a move/rename: the op stays at
`ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/`, namespace `ttnn::prim`.

## 2. WH/BH parity claim

Trivially satisfied: the diff is empty, so WH and BH execute byte-identical code. The BH/WH commands in §7 are
the *baseline* to record now, so the eventual Metal 2.0 port and Quasar uplift have a pre-change reference
(`metal2_port.md` insists the baseline be captured before any kernel edit, since kernels are JIT-compiled from
the working tree).

## 3. Quasar-uplift audit findings (run anyway, so the future uplift has them in hand)

`quasar_audit.md` has two checks. Both were run against the six kernels the factory can select, plus the
factory. Scope discovery (`cb_dfb_quasar_audit_helper.md` Steps 0–3):

| Kernel (all under `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/`) | Selected when | Other binders (sunset list, not authorization) | `_metal2` sibling |
|---|---|---|---|
| `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | TILE layout | `sharded_partial/interleaved_to_sharded_partial` | **none** |
| `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | ROW_MAJOR layout | `sharded_partial/interleaved_to_sharded_partial` | **none** |
| `dataflow/writer_unary_sharded.cpp` | dst in L1 (both layouts) | i2s_partial, tilize (2 factories), untilize nd-shard, padded_slice, nlp_kv_cache_load_slice | `writer_unary_sharded_metal2.cpp` (created by #51743) |
| `dataflow/writer_unary_sharded_blocks_start_id.cpp` | dst in DRAM, TILE | i2s_partial | **none** |
| `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | dst in DRAM, ROW_MAJOR | i2s_partial | **none** |
| `compute/eltwise_copy.cpp` | `convert_df` (dtype change; TILE only) | copy, i2s_partial, s2i_partial, untilize_with_unpadding sharded | `compute/eltwise_copy_metal2.cpp` (beside it; bound by `copy_default_tilized`) — note a second, non-sibling fork exists at `ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp`, which s2i (#52207) binds |

All six kernels are shared (borrowed from the family pool or lent to `sharded_partial`), so the future Metal 2.0
port of i2s lands on the `_metal2` fork rungs of `port_patterns.md` "Caution: Porting a shared kernel" for
every kernel: reuse two existing forks, create four new ones beside their originals.

### 3.1 Check 1 — CB/DFB classification (helper Classes 1–6)

Scans run over the six kernels (`SCAN_FILES`; no `#include` closure beyond framework headers):

| Scan | Hits |
|---|---|
| GATE `get_local_cb_interface(...).<field>` | **1** — `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:63` `.fifo_page_size` |
| `get_cb_tiles_*_ptr` / `read_tile_value` / `get_tile_address` / `get_pointer_to_cb_data` | 0 |
| `get_read_ptr` / `get_write_ptr` peeks | 1 — same kernel `:79` (`dfb_in1.get_write_ptr()` as a raw NoC source address) |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `evil_set_*` / `push_back_hold` / `pages_reservable/available` | 0 |
| Legacy device API (`noc_async_*` free fns, `circular_buffer.h`, addr-gens) | 0 |
| Semaphores / multicast / `disable_dfb_implicit_sync*` / `MEM_ZEROS_BASE` | 0 |

Buffer portability (one row per logical CB; 1xx = WH/BH, 2xx = Quasar end-state):

| CB | Class | Kernels | Config | 1xx | 2xx | Notes |
|---|---|---|---|---|---|---|
| `c_0` input/out (borrowed from `dst_buffer` when dst is L1; regular when dst is DRAM) | 1 | TILE reader (P) / RM reader (P) → `writer_unary_sharded` (C) or DRAM writers (C) | `!convert_df` | Portable (already DFB in-kernel) | Portable | Canonical explicit FIFO; reader `reserve_back` + `noc.async_read(s, dfb, ..., {.offset_bytes})` + `push_back`; writer `wait_front`/`pop_front`. Metal 2.0 port: `borrowed_from = <output TensorParameter>` when dst is L1. |
| `c_0` input (regular) | 1 | reader (P) → `eltwise_copy` (C) | `convert_df` | Portable | Portable | Same shape, compute consumes. |
| `c_16` out (borrowed from `dst_buffer` when L1) | 1 | `eltwise_copy` (P) → writer (C) | `convert_df` | Portable | Portable | Compute packer → writer; `reserve_back`→`pack_tile`→`push_back` and `wait_front`→`copy_tile`→`pop_front` each have a real PACR/UNPACR between them → **TEN-4746 bare-pair check passes**. `compute_kernel_hw_startup` once at `main()`, single `copy_init`, no DFB-id switching → §7 init rules satisfied. |
| `c_1` scratch (regular, `num_trids=4` pages of `align(unit+dram_align, dram_align)`) | **6** (sync-free alignment scratch wearing a FIFO costume) | RM reader only, `!aligned` branch | ROW_MAJOR | Portable (workaround) — `1xx_port:` self-loop DFB (one DM toucher; `reserve_back(num_trids)` … `push_back(num_trids)` with no consumer) | **NEEDS-DESIGN-DECISION → Scratchpad** | This is the helper's own flagship Class 6 example ("interleaved_to_sharded scratch CB → Scratchpad migration", priority P3). Three Quasar hazards, see §3.3. |
| `c_1` scratch | dead | (none bind it) | TILE layout, and RM `aligned` at runtime | dead-CB in TILE config | — | Allocated unconditionally (`keep_l1_aligned` is hardcoded `true`, so the `if` at `.cpp:179` is always taken) but the TILE reader never receives its index. Metal 2.0 port: **conditional DFB spec** (live under ROW_MAJOR, dead under TILE) — do *not* drop it outright. Also a Misc anomaly: L1 burned on every TILE i2s for a CB no kernel touches. |

**Rollup for Check 1:** GREEN on 1xx (everything is a canonical FIFO or a one-toucher self-loop), **RED on 2xx
for the ROW_MAJOR path** (`c_1` needs the Scratchpad redesign — an owner decision), GREEN on 2xx for the TILE
path (which is the only path the model exercises, see §5).

### 3.2 Check 2 — Non-zero-initialized semaphores

None. The op creates no semaphores at all. ✓

### 3.3 §7–§8 gotchas: applied vs. considered

Nothing was *applied* (RED stop; and the recipe says fixes are reactive — apply only when the symptom fires
on a device run, which has not happened). What was considered, and why it will matter after the port:

| Gotcha | Verdict for i2s | Detail |
|---|---|---|
| §5 / §8.3 "`fifo_page_size` is stale on Quasar → value inflation / wrong pad" | **Applies** (RM `!aligned` path) | `reader_unary_stick_layout_...cpp:63` derives every scratch slot offset from `get_local_cb_interface(cb_id_in1).fifo_page_size`. Must become `dfb_in1.get_entry_size()` (`dataflow_buffer.h:109`). This is also the helper's GATE for the kernel CB→DFB port. Note the recipe conflict recorded in §8. |
| §6 "Local self-read/self-copy (src==dst L1) on the emulator can spin on `can_post` or silently drop the read — use a direct L1→L1 RISC copy, not a NoC loopback" | **Applies** (RM `!aligned` path) | `:111-121` issues `noc.async_read<TXN_ID>(self_ep, dfb_in0, ..., {.noc_x=my_x, .noc_y=my_y, .addr=scratch_l1_base+...})` — a same-core NoC loopback from the scratch CB into the input CB. |
| §8.3 / §12 "DFB `get_read/write_ptr()` return UNCACHED addresses on Quasar DM; NOC APIs cannot take uncached addresses" | **Applies** (RM `!aligned` path) | `:79` `scratch_l1_base = dfb_in1.get_write_ptr()` is then fed to the NoC as `.addr`. `dataflow_buffer.h:326-327,367-372` confirm: on Quasar DM the public getter adds `MEM_L1_UNCACHED_BASE`, and only the *private* `get_noc_*_addr()` is NOC-safe. Once ported, this read must go through a DFB/`Scratchpad` NoC operand, not a raw address. |
| §6 / `dm_self_loop_dfbs.md` "DM self-loop rejected on Gen2 → Scratchpad" | **Applies** (RM path) | `c_1` would be a DM self-loop after a mechanical Metal 2.0 port (its only legal Gen1 shape); Quasar's validator rejects it. Because the kernel *also* reads the entry size off the handle, the `dm_self_loop_dfbs.md` pass stops on it ("any other method on the handle — `get_entry_size()` most often") — hence "owner decision", not a mechanical pass. |
| §7 TEN-4746 bare `wait_front`→`pop_front` / `reserve_back`→`push_back` in compute | Considered, **clean** | `eltwise_copy.cpp` has a real `copy_tile` / `pack_tile` between each pair on every control-flow path. DM kernels are out of scope for this check by the recipe. |
| §7 `compute_kernel_hw_startup` once; re-`*_init` on DFB-id change | Considered, **clean** | Single `copy_init(c_0)`, operands never change. |
| §7 `partials_cb_uses_output` / borrow-with-offset | N/A | The only borrow (`c_0`/`c_16` on `dst_buffer`) is at offset 0 (`CBDescriptor::address_offset` unset). Also clears `metal2_audit.md` Appendix A `address_offset`. |
| §7 Int32-not-UInt32 / no uint16 / no Bfp8 on Quasar | Nothing to guard in the op | The factory forwards `datatype_to_dataformat_converter(dtype)`; it has no format-specific code path. `convert_df` to BFLOAT8_B/4_B (allowed by `validate_inputs`) would surface at the format/LLK layer on Quasar — flag, do not edit the op. The model uses bf16→bf16 (no `convert_df`). |
| §7 RM shard width 16-byte aligned on Quasar | Already enforced | `interleaved_to_sharded_op.cpp:86-91` rejects RM shards whose row is not `hal::get_l1_alignment()`-aligned. |
| §7 non-zero-init semaphores; §11 multicast rectangle normalization; §11 mcast corner clamp; `MEM_ZEROS_BASE`; `disable_dfb_implicit_sync*` | N/A | None present. |
| §4 `opt_level` (compute default O3 vs Metal 2.0 O2) | Deferred to the base port | `eltwise_copy.cpp` is a compute kernel with no explicit `opt_level` in the descriptor → resolves to **O3** today. The Metal 2.0 port must set `compiler_options.opt_level = O3` explicitly on that `KernelSpec` (`metal2_port.md` "Compiler options"); this uplift does not touch it. |
| §4 `hw_config` carry values | Deferred to the base port | Reader/writer use `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (the defaults) → `create_reader/writer_datamovement_config(arch)` will supply the Gen2 branch for free. Compute uses `ComputeConfigDescriptor{}` (all defaults) → Style B, `to_compute_hardware_config` or a default `ComputeGen1Config`; the Gen2 branch is then `gen2_hardware_configs.md`'s shape 2/4 (no `unpack_modes`, so no marker needed). |
| §8.1 build skew (`common.hpp` ckernel pull-in, `int-to-pointer-cast`, `REDUCE_OP`, etc.) | Not evaluated | No Quasar build was run (user runs builds). The six kernels include only `api/dataflow/*`, `api/compute/*`, `api/tensor/noc_traits.h`, `tensix_types.h`; none of the §8.1 signatures is visible statically. |

### 3.4 Factory observations that already favour Quasar

`277a63eacbe` (#49415) added `is_quasar` beside `is_blackhole` at `.cpp:96,179,359,369`, so the RM alignment
logic (scratch CB allocation, `aligned` computation, `round_down` to DRAM/L1 alignment) treats Quasar like
Blackhole. The uplift needs no `ARCH_QUASAR` host branch of its own here.

## 4. How the user's note was interpreted — "i2s — try the experimental/quasar"

There is a whole-op copy at `ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/` (own
factory, own device op, six duplicated kernels). **Its contents were not read and nothing from it enters
this report**, per the hard rule shared by every recipe document consulted (`quasar_porting.md` rule 1,
`metal2_port.md` "Read this first", `metal2_audit.md`, `pass_procedure.md` Step 3, `port_patterns.md`):
that tree is a deliberately hacky bring-up fork that must never be cited, copied, forked, or used as evidence,
and an uplift must land **in place** in the production op.

Two readings of the note, and what the recipe says about each:

1. **"Use the experimental copy as the source/precedent for uplifting mainline i2s."** Forbidden by the
   recipe, and it would not help: the mainline op's blocker is that it is not Metal 2.0 at all, which is a
   `metal2_audit.md` → `metal2_port.md` job derived from the canonical recipe and the Metal 2.0 headers, not
   from any existing port.
2. **"Route the Quasar *model* to the experimental copy so the model test passes today."** That is a
   model-side decision (which Python entry point `models/experimental/llama32_1b_quasar` calls), outside the
   scope of an op uplift and outside this report's authority. The model code currently imports only mainline
   `ttnn.*` ops (grep of the model tree finds no `ttnn.experimental.quasar.*` calls). If the user takes this
   route as a stop-gap, it does not change the verdict here: mainline i2s still needs its Metal 2.0 port and
   then a re-run of this uplift.

Recommendation: treat the note as (2) — a bring-up fallback the user may choose to wire at the model level —
and drive the real fix through the Metal 2.0 pipeline in §6.

## 5. What the Quasar model test actually exercises

`models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py` (generated; do not
hand-edit) has two captured cases, both **TILE layout, bf16 → bf16, DRAM-interleaved → L1-sharded**:

| Case | Input | Output shard | Path selected in the factory | Emulator-runnable? |
|---|---|---|---|---|
| `00_1x64_bf16_int-dram` (40 calls) | `[1,1,1,64]` | HEIGHT_SHARDED, grid `[[0,0,0,0]]` (1 core), shard `[32,64]` | TILE reader + `writer_unary_sharded.cpp`, `c_0` borrowed on the output, `c_1` allocated but dead, no compute | yes (1 core) |
| `01_32x2048_bf16_int-dram` (2 calls) | `[1,1,32,2048]` | WIDTH_SHARDED, grid `[[0,0,7,3]]` (32 cores), shard `[32,64]` | same kernels, 32 cores, last-core `padded_offset` path | no — needs 32 cores; `conftest.py` leaves it unmarked and `graph_case.build_memory_config` skips it at run time on a smaller grid |

So the model path avoids every Quasar hazard found in §3.3 (all are on the ROW_MAJOR `!aligned` branch), and
the compute kernel is never launched. Once the Metal 2.0 port lands, the TILE path is expected to be a
**no-change GREEN** uplift; the ROW_MAJOR path carries the `c_1` redesign.

## 6. Deferred / follow-up items (the path forward)

Ordered as the recipe orders them. None of these may be done inside this uplift session.

1. **Metal 2.0 pre-port audit** (`metal2_audit.md`) on this op → `METAL2_PREPORT_AUDIT.md` +
   `METAL2_PORT_BRIEF.md`. Findings this report can already predict for the auditor to confirm:
   - Concept: `descriptor` (`create_descriptor` → `ProgramDescriptor`), single factory, no
     `override_runtime_arguments`, no `get_dynamic_runtime_args` → target `ProgramSpecFactoryConcept`.
   - Custom `compute_program_hash` present at `interleaved_to_sharded_op.cpp:144-162` (already keys whole
     `TensorSpec`s per #55495) — leave intact.
   - No pybound `create_descriptor` (`interleaved_to_sharded_nanobind.cpp` exposes only the op).
   - Device 2.0: all six kernels on `Noc`/`DataflowBuffer`/`TensorAccessor`. The one
     `get_local_cb_interface(...).fifo_page_size` read is *sanctioned* by `metal2_audit.md`'s Device 2.0 gate
     but is a **GATE** in `cb_dfb_quasar_audit_helper.md` and a Quasar wrong-output bug per
     `quasar_porting.md` §5 — the auditor must reconcile (see §8).
   - Appendix A features: none (no GlobalCB, no `address_offset`, no GlobalSemaphore).
   - Offset base pointers: clean — reader RTAs carry the bare `Buffer*`; the per-core column shift rides
     `offset_bytes` on each read (the #51747 fix, documented in the RM reader at `:33-35`).
   - TensorAccessor 3rd arg: none (all accessors are 2-arg).
   - Tensor bindings: `src` Case 1 in both readers and DRAM writers; output `borrowed_from` when dst is L1
     (clean), Case 1 via the DRAM writers otherwise.
   - CB endpoints: `c_0`/`c_16` legal 1P+1C; `c_1` **self-loop on the RM reader** (one DM toucher) and
     **conditional DFB** (dead under TILE) — allocate the spec only on the ROW_MAJOR path.
   - Shared kernels: all six (table in §3); reuse `writer_unary_sharded_metal2.cpp` (binding vocabulary
     `dfb::out`, `args::num_units`) and `sharded/device/kernels/compute/eltwise_copy_metal2.cpp`
     (`dfb::in`, `dfb::out`, `args::per_core_tile_cnt`); create the four missing `_metal2` forks beside
     their originals.
   - `opt_level`: compute kernel must get an explicit `O3`.
2. **Metal 2.0 port** (`metal2_port.md`), fresh primary session, after explicit go-ahead on a GREEN audit.
   Run the WH/BH baseline in §7 *before* the first kernel edit.
3. **Post-port style pass** `sync_free_dfbs.md`: zero sites expected (`c_1` *does* call `reserve_back`/
   `push_back`, so it is not sync-free by that recipe's criterion).
4. **Post-port semantic pass** `dm_self_loop_dfbs.md` on `c_1`: **will stop** on the `get_entry_size()`
   read (formerly `fifo_page_size`) and on the raw-address NoC loopback. Hand to the op owner as a redesign:
   `Scratchpad<uint8_t>` sized `num_trids * scratch_cb_page_size`, slot stride carried as a named CTA or
   re-derived from a value the kernel already has, and the scratch→dest copy expressed as a `Scratchpad`
   NoC operand (or a direct L1→L1 RISC copy per `quasar_porting.md` §6) instead of `.addr = get_write_ptr()+…`.
   Exact Quasar symptoms to expect if this is skipped: wrong/inflated RM output or bad padding
   (§8.3 `fifo_page_size`), a reader spinning on `can_post` or silently dropping the loopback read (§6),
   and a NoC transaction issued against an uncached L1 alias (`dataflow_buffer.h:326,371`).
5. **Post-port semantic pass** `gen2_hardware_configs.md`: expected zero-site (default DM helpers; compute
   config all-default) — confirm.
6. **Re-run this Quasar uplift** (`quasar_porting.md`) on the ported op. Expected: TILE path GREEN with no
   edits; ROW_MAJOR path depends on item 4.
7. **Misc anomaly for the op owner** (not port work): `c_1` is allocated on every i2s program including TILE
   layout, where no kernel binds it (`.cpp:179-192`, `keep_l1_aligned` hardcoded `true` at `.cpp:65`). Metal 2.0's
   validator will reject a bindingless DFB, forcing the conditional spec; the L1 waste exists today.
8. **Recipe-level:** the readiness sheet (`ttnn_op_porting_readiness.md`) was not fetched in this session
   (no Drive access under this session's rules); the auditor in item 1 should.

## 7. Test commands for the user (this session ran none)

Run from the repo root with the venv active. Order per `quasar_porting.md` §9: BH → WH → Quasar. Because this
session changed no source, these are the **baseline** runs to record before the Metal 2.0 port begins.

**Blackhole and Wormhole (identical commands; op-level suite + the model-shape control):**

```bash
export TT_METAL_WATCHER=10
pytest tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py -v
# model-shape control (replicated (1,1) mesh; the 32-core case needs an 8x4 grid, e.g. N150/P150)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py -v
# broader coverage that routes through i2s (optional):
pytest tests/ttnn/unit_tests/base_functionality/test_to_layout.py -v -k sharded
pytest tests/sweep_framework/sweeps/data_movement/interleaved_to_sharded/interleaved_to_sharded_e2e.py -v
# tt_metal hardcoded reader/writer kernel tests (same kernel family):
#   <data_movement gtest binary under build/test/tt_metal/> --gtest_filter='*TensixDataMovementI2S*'
```

**Quasar emulator** (emulator-runnable subset = case `00_1x64_bf16_int-dram`; case `01` self-skips on a
grid smaller than 8x4). Run twice — with `TT_METAL_LLK_ASSERTS` set and unset — per §9:

```bash
export TT_METAL_WATCHER=10
TT_METAL_LLK_ASSERTS=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py -m emulator -v
unset TT_METAL_LLK_ASSERTS
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py -m emulator -v
# shapes/placement/finiteness only, if the golden is under suspicion:
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py -m emulator -v
```

`TT_METAL_FORCE_JIT_COMPILE=1` is not needed for these runs (no kernel changed); add it after the port.

## 8. Recipe notes (friction for the recipe maintainers)

- **Conflict on `get_local_cb_interface(...).fifo_page_size`.** `metal2_audit.md` (Device 2.0 gate, Green
  bullet) names `get_local_cb_interface(cb_id)` as *sanctioned* and says the holdover cue "does not override the
  list". `cb_dfb_quasar_audit_helper.md` makes any `get_local_cb_interface(...).<field>` read a hard **GATE**
  for the kernel CB→DFB port (fix: `get_entry_size()`), and `quasar_porting.md` §5/§8.3 calls the same read a
  top Quasar wrong-output cause. The three documents agree on the *fix* but not on *which stage owns it*
  (Device 2.0 track vs. the Metal 2.0 port's whitelist rule 7 vs. the Quasar uplift). For this op the Metal 2.0
  port's rule 7 ("DFB metadata via the object") would convert it naturally; the audit docs should say so
  explicitly so the porter is not stopped by the helper's GATE wording.
- **`quasar_porting.md` sends a not-yet-M2 op straight to RED**, but the uplift agent has already gathered
  most of what the `metal2_audit.md` auditor needs. The recipe could sanction carrying a "predicted audit
  findings" section (as §6 item 1 does here) so the next session starts warm.
- The human `READ_ME_FIRST.md` prescribes one op per workspace; this session ran alongside other ops in one
  working tree under a no-build/no-test/no-git-write regime. That regime is what made a zero-edit RED report
  the correct output; it would not be safe for an uplift that needed device runs.
