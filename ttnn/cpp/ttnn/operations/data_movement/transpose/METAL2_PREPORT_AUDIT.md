# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/transpose`

One device operation, eight program factories, declared in the `program_factory_t` variant of
`device/transpose_device_operation.hpp:29-37`. **Six of the eight are already on Metal 2.0** (PR #54333,
2026-09-01: `create_program_artifacts` returning a `ProgramSpec`); this audit covers the **two that are still on
the `ProgramDescriptor` API** and reports the six as done.

- **`TransposeDeviceOperation`** (`device/transpose_device_operation.{hpp,cpp}`; no custom hash, no
  `get_dynamic_runtime_args`, no pybound `create_descriptor`)
  - **In scope (legacy `ProgramDescriptor`, `create_descriptor` + `override_runtime_arguments`):**
    - `TransposeHCShardedProgramFactory` (`device/transpose_hc_sharded_program_factory.{hpp,cpp}`;
      `create_descriptor` `:324`, `override_runtime_arguments` `:462`) — selected for HC when
      `use_sharded_hc` (`transpose_device_operation.cpp:336`, `:352`): native sharding, input and output
      height-sharded, ROW_MAJOR layout.
    - `TransposeWHShardedRMProgramFactory` (`device/transpose_wh_sharded_rm_program_factory.{hpp,cpp}`;
      `create_descriptor` `:19`, `override_runtime_arguments` `:253`) — selected for WH when
      `is_row_major && use_sharded_wh_rm` (`:335`, `:343`): native sharding with the full H×W of every batch
      inside one shard on both sides.
    - "Native" (`transpose_utils.cpp:80-100`, `side_native` `:62-76`) means L1 (non-DRAM), non-BLOCK, evenly
      sharded, and for row-major a shard whose element count is a multiple of 32×32 (`:49-58`).
  - **Already Metal 2.0 (`Concept == MetalV2`, reported as done, not audited):** `TransposeCNProgramFactory`,
    `TransposeHCRMProgramFactory`, `TransposeHCTiledInterleavedProgramFactory`, `TransposeHCTiledProgramFactory`,
    `TransposeWHProgramFactory`, `TransposeWHShardedProgramFactory` (each `create_program_artifacts` + `ProgramSpec`,
    e.g. `transpose_cn_program_factory.cpp:20`/`:65`, `transpose_wh_program_factory.cpp:22`/`:68`).

Kernels bound by the two in-scope factories (all owned by this op; no borrowed kernel file):

| Kernel | Bound by | Notes |
|---|---|---|
| `device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp` | HC sharded reader (`hc_sharded:404-406`), both paths | `USE_SPECIAL_CASE` define selects the special-case body (`:399-401`) |
| `device/kernels/dataflow/writer_unary_transpose_hc_sharded_rm.cpp` | HC sharded writer (`:439-441`), **special case only** (`:437`) | generic path has no writer kernel (`:413-415`) |
| `device/kernels/dataflow/reader_unary_transpose_wh_sharded_rm.cpp` | WH sharded RM reader (`wh_sharded_rm:182-184`) | CTAs only, no RTAs |
| `device/kernels/dataflow/writer_unary_transpose_wh_sharded_rm.cpp` | WH sharded RM writer (`:201-203`) | CTAs only; no-op when `Ht <= 8` |
| `device/kernels/compute/transpose_wh_rm.cpp` | WH sharded RM compute (`:235-236`), compiled with `SHARDED=1` (`:221-222`) | **intra-op shared**: its `_metal2` fork `transpose_wh_rm_metal2.cpp` is bound by the already-ported `TransposeWHProgramFactory` (`transpose_wh_program_factory.cpp:197`) and deliberately carries only the non-sharded path (fork header `:11-16`) — see *Heads-ups* |

Unreferenced kernel files in the directory (no factory on `main` binds them; out of audit scope per the recipe —
legacy originals left behind when PR #54333 forked or converted their users): `compute/transpose_wh.cpp`,
`compute/transpose_wh_sharded.cpp`, `dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`,
`dataflow/reader_unary_transpose_wh_interleaved.cpp`. (`create_qkv_heads` and `nlp_create_qkv_heads_boltz` bind
*same-named private copies* under their own paths, not these files.)

> `ttnn/cpp/ttnn/operations/experimental/quasar/transpose/` holds a whole-op copy with its **own duplicate
> kernels** (checked by the bound path strings only — `experimental/quasar/transpose/device/kernels/…`). Nothing
> under that tree was read for this audit; the porter must not read it either, and it is **not** a binder of this
> op's kernel files.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
(the recipe branch is an orphan history and cannot be merged into `main`; the audit ran on a verbatim local export
of its `docs/…/metal_2.0/` tree, which this branch does not carry)

**Code audited:** `origin/main` @ `0df0dde3316` (2026-10-27). Line numbers below are against that commit.
Provenance of the shape audited: ProgramDescriptor migration PR #45071 (2026-05-25, dgomezTT);
`get_dynamic_runtime_args` → `override_runtime_arguments` by PR #52566 (commit `383674438e5`, cited in
PR #54333's body); six-factory Metal 2.0 port PR #54333 (2026-09-01, vsureshTT), which left these two factories on
the descriptor path because the audit of that time was RED on the now-removed `get_dynamic_runtime_args` hook;
its METAL2_*.md artifacts removed by PR #55493 (2026-09-08).

**Readiness sheet:** fetched live this session (CSV export of *"Operations analysis"*); 8 rows for
`data_movement/transpose` (sheet rows 94-101, one per factory — factory-set match ✓ against the eight factories in
`transpose_device_operation.hpp:29-37`). The two in-scope rows are 96 (`TransposeHCShardedProgramFactory`) and
101 (`TransposeWHShardedRMProgramFactory`).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/transpose` |
| **Overall** | **GREEN** for the two in-scope factories (`TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`); the other six are already ported |
| **DOps / Factories** | `TransposeDeviceOperation` → in scope: `TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory` · done: `TransposeCN`, `TransposeHCRM`, `TransposeHCTiledInterleaved`, `TransposeHCTiled`, `TransposeWH`, `TransposeWHSharded` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all four DM kernels on `Noc` / `DataflowBuffer` / `UnicastEndpoint` / `CoreLocalMem` (stateful `set_async_read_state` / `async_read_with_state` idiom); the compute kernel is a TRISC kernel on `DataflowBuffer` wrappers (DM migration N/A). No addr-gens, no raw `noc_async_*`, no `cb_*` free functions, no raw semaphores |
| *Prereqs* — Cross-op escapes | Ok — the compute kernel calls kernel-lib `compute_kernel_lib::tilize<…>` with `uint32_t` CB ids as NTTPs and LLK free functions over CB ids; the DM kernels include only `tt_metal` API headers; no borrowed kernel file |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every compile-time-arg read is at a constexpr index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** on both in-scope rows (`Diego validation = yes`) |
| *TTNN Readiness* — Concept (current) | `descriptor` on both in-scope rows — cross-check ✓ (`create_descriptor` at `transpose_hc_sharded_program_factory.cpp:324` / `transpose_wh_sharded_rm_program_factory.cpp:19`). The six sibling rows also read `descriptor` but the code is `MetalV2` since PR #54333 — **sheet stale on already-ported rows**, routed to the sheet owner (see *Gate detail*) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (plain `ProgramDescriptor`; sheet `Execution Model = SPMD`) |
| *TTNN Readiness* — Custom hash | No — sheet `no` / `Backdoor custom hash = no`; no `compute_program_hash` / `attribute_values` / `to_hash` in the op ✓ |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`; absent from the device op ✓ (removed by PR #52566) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** on both (not a gate; selects `CustomProgramSpecFactoryConcept`): `transpose_hc_sharded_program_factory.cpp:462-474`, `transpose_wh_sharded_rm_program_factory.cpp:253-265` — both are **cb-addr-only** (re-point the two borrowed CBs via `apply_descriptor_runtime_args`; tagged `override-rebuild-ok: cb-addr-only` per `scripts/detect_override_rebuild.py`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `transpose_nanobind.cpp:40-61` binds only the two `ttnn::transpose` overloads; sheet `no` ✓ |
| *TTNN Readiness* — Op-owned tensors | No — sheet cell blank (N/A on a `descriptor` row); plain `ProgramDescriptor` ✓ |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** for both (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | **none** — neither factory passes any buffer address as an RTA/CRTA/CTA; tensors enter only through borrowed CBs |
| *Port work* — Tensor bindings (per binding) | `input`: **clean** (borrowed-memory DFB, `c_0`) · `output`: **clean** (borrowed-memory DFB, `c_16`) — in both factories; no Case 1, no Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `none` on both rows → clears |
| *Port work* — TensorAccessor 3rd arg | **none** — no `TensorAccessor` exists in either factory's kernels |
| *Port work* — CB endpoints | HC sharded: `in0` / `out` → **1P+1C** (special case: reader + writer, both role-free) · **self-loop** × 2 (generic: reader only). WH sharded RM: `in0` self-loop (reader) · `in` legal 1:1 · `tilize` self-loop (compute) · `im2` **dead-CB drop** · `out_staging` legal 1:1 (**conditional DFB**, `ht > 8`) · `out`: self-loop (writer, `ht > 8`) / 1P+1C (compute + writer, `ht <= 8`) |

**CB endpoints** are dispositions, not gates: every out-of-window CB above has its port-time resolution recorded in
*Gate detail*.

## Result

**GREEN → brief issued** for `TransposeHCShardedProgramFactory` and `TransposeWHShardedRMProgramFactory`. Device 2.0,
Appendix A feature compatibility, the readiness sheet (`Is able to port? = yes` on both rows, every cheaply-checkable
column matching the code), offset base pointers, the TensorAccessor 3rd argument and the `TensorParameter`
relaxation column (`none`) all clear. The six other factories are **already on Metal 2.0** (PR #54333) and need no
port. The port can begin on explicit go-ahead: two factories, five kernels (all owned), two borrowed-memory tensor
bindings each, no semaphores, no `TensorAccessor` — a port whose only non-mechanical decision is what to do about
the sharded path of the intra-op shared compute kernel (*Heads-ups*, *Questions for the user*).

**Subject coverage.** All twelve subjects were run in full for the two in-scope factories; nothing was skipped or
deferred. The six `MetalV2` factories were not audited (the recipe says to report them as done).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both in-scope rows — each reads `yes`, every
  cheaply-checkable primary column matches the code, cross-column invariants hold, factory-set match ✓ (8 rows ↔ 8
  factories, no phantom, no missing):

  | Factory (sheet row) | `Concept` | `Op Classification` | `Porting Target` | `Override…?` | `Custom hash` | `get_dynamic…` | `Pybind descriptor` | `Smuggled pointer` | `Known op issues` | `TensorParameter relaxation` | `Is able to port?` | Cross-check |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | `TransposeHCShardedProgramFactory` (96) | `descriptor` | `PD Op (custom)` | `CustomProgramSpecFactoryConcept` | `yes` | `no` | `no` | `no` | `no` | *(empty)* | `none` | `yes` | ✓ `create_descriptor` `hc_sharded:324`, `override_runtime_arguments` `:462`; no address RTAs |
  | `TransposeWHShardedRMProgramFactory` (101) | `descriptor` | `PD Op (custom)` | `CustomProgramSpecFactoryConcept` | `yes` | `no` | `no` | `no` | `no` | *(empty)* | `none` | `yes` | ✓ `create_descriptor` `wh_sharded_rm:19`, `override_runtime_arguments` `:253`; kernels take no RTAs at all |
  | `TransposeCN` (94) · `TransposeHCRM` (95) · `TransposeHCTiledInterleaved` (97) · `TransposeHCTiled` (98) · `TransposeWH` (99) · `TransposeWHSharded` (100) | `descriptor` | `PD Op (pointer-patching)` | `ProgramSpecFactoryConcept` | `no` | `no` | `no` | `no` | `no` | *(empty)* | `none` | `yes` | **✗ stale** — code is `MetalV2` (`create_program_artifacts` + `ProgramSpec`) since PR #54333, 2026-09-01 |

  Other cells on the two in-scope rows, quoted for the record: `Execution Model = SPMD`, `Diego validation = yes`,
  `Model = llama`, `ProgramFactory used in llama? = no`, `Uses llama kernels? = no`; `Op-owned tensors?`,
  `Secretly SPMD Workload?`, `Why secretly SPMD?`, `Provisional relaxation finding`,
  `Pointer patching perf issue?`, `Formerly custom hashed?` blank. No
  `analyses/relaxations/data_movement_transpose.md` exists and none is needed (`none`).

  **The six stale rows.** `Concept` is a cheaply-checkable primary column and it disagrees with the code on rows
  94, 95, 97, 98, 99, 100 — but those rows describe factories that are *already ported*, for which the recipe's
  disposition is "report as done" regardless of the cell. The two rows this port reads are consistent with the
  code in every checked column. I therefore treat the staleness as a **reconciliation item for the readiness-sheet
  owner** (Diego: rows 94/95/97/98/99/100 should read `Concept = MetalV2`, `Porting Target` n/a) and **not** as a
  gate on the in-scope factories; the judgment call is logged under *Recipe notes*.

- **Device 2.0 (every kernel used):** **GREEN.** Per kernel:

  | Kernel | Idioms | Verdict |
  |---|---|---|
  | HC sharded reader (`reader_unary_transpose_hc_sharded_rm.cpp`) | `Noc` (`:29`, `:119`), `DataflowBuffer` (`:30-31`, `:120-121`) with `get_read_ptr` / `get_write_ptr` peeks (`:38-39`, `:62-63`, `:124`, `:149`) and `reserve_back` / `push_back` (`:123`, `:177`); `noc.set_async_read_state<DEFAULT, NOC_MAX_BURST_SIZE>` + `async_read_with_state` (`:41-54`, `:68-81`), `noc.async_read` (`:152-157`), `UnicastEndpoint{}` to remote cores, `CoreLocalMem<uint32_t>` destinations (`:47`, `:74`, `:151`), `noc.async_read_barrier()` (`:58`, `:88`, `:176`) | ✓ |
  | HC sharded writer (`writer_unary_transpose_hc_sharded_rm.cpp`) | same idioms: `Noc` `:29`, `DataflowBuffer` `:30-31`, peeks `:38-39`, `:55-56`, `noc.async_read<…>` `:42-47`, `set_async_read_state` / `async_read_with_state` `:62-74`, barriers `:52`, `:81` | ✓ |
  | WH sharded RM reader (`reader_unary_transpose_wh_sharded_rm.cpp`) | `Noc` `:27`, `DataflowBuffer` `:28-29`, peek `:31`, `:41`, `set_async_read_state` `:33-36`, `async_read_with_state` `:45-52`, `reserve_back` / `push_back` `:40` / `:57`, `async_read_barrier` `:56` | ✓ |
  | WH sharded RM writer (`writer_unary_transpose_wh_sharded_rm.cpp`) | `Noc` `:26`, `DataflowBuffer` `:27-28`, peek `:30`, `:42`, `set_async_write_state` `:34-37`, `async_write_with_state` `:46-53`, `wait_front` / `pop_front` `:41` / `:58`, `async_writes_flushed` `:57`, `async_write_barrier` `:61` | ✓ |
  | WH sharded RM compute (`transpose_wh_rm.cpp`) | TRISC kernel: `DataflowBuffer` wrappers for every FIFO op (`:141-142`, `:161`, `:181`, helper bodies `:43-53`, `:83-96`), LLK free functions over CB ids (`compute_kernel_hw_startup` `:144`, `copy_init` `:145`, `transpose_init` `:31` / `:77`, `transpose_tile` `:36` / `:87`, `pack_untilize_dest_init` `:32` / `:81`, `pack_untilize_dest` `:45` / `:51` / `:93`, `pack_untilize_uninit` `:57` / `:100`), kernel-lib `compute_kernel_lib::tilize<Wt, cb_in, cb_tilize, …>(Ht)` (`:151-158`) | N/A — not a data-movement kernel |

  No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / `cb_*` free-function FIFO calls / raw
  semaphore addresses / CB-index free-function holdovers in any of the five kernels. **One breadcrumb, not a
  gate (same as the `attn_matmul` audit):** the WH sharded RM reader and writer address their *own* shard through
  a `UnicastEndpoint` built from the firmware coordinate globals — `my_x[noc.get_noc_id()]` /
  `my_y[noc.get_noc_id()]` (reader `:36`, `:49-50`; writer `:37`, `:51-52`) — the Device 2.0 migration guide's
  *"Avoid (Device 2.0 API, but hand-built loopback endpoint)"* form
  (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:524-551`). It is
  Device 2.0 API (none of the gate's Red idioms), so it clears; the preferred local-handle form is a transfer
  rewrite, out of port scope. Recorded for the porter under *Heads-ups* (keep as is). The HC sharded kernels address
  *other* cores' shards with per-core NoC coordinates from the RTAs (`reader:44`, `:52`, `:71`, `:79`, `:156`;
  writer `:46`, `:65`, `:73`) — a genuine remote endpoint, not the loopback form.

- **Feature compatibility:** every Appendix A entry `N/A`:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field on any `CBDescriptor` (HC: `hc_sharded:359-380`; WH RM: `wh_sharded_rm:91-166`), no `remote_cb` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | both factories set `.buffer = <tensor>.buffer()` (HC `:367`, `:379`; WH RM `:99`, `:112`) — the plain borrowed-memory pattern (mechanical `borrowed_from`) — and never `.address_offset`; `UpdateDynamicCircularBufferAddress` appears only in two comments (`wh_sharded_rm:89`, `:102`); the override's `CBDescriptor{.buffer = …}` re-point (`hc_sharded:471-472`, `wh_sharded_rm:262-263`) carries no offset |
  | GlobalSemaphore | N/A | neither factory declares any semaphore (`desc.semaphores` never touched; no `Semaphore` in any kernel) |

- **CB endpoints (GATE-free):** classified per CB, per node, per configuration. Both factories place every CB and
  every kernel on the shard grid (`all_cores = shard_spec.grid`; HC `:346`, `:361`, `:373`, `:408`, `:443`; WH RM
  `:83`, `:93`, `:106`, `:120`, `:133`, `:147`, `:160`, `:186`, `:205`, `:238`), so each node sees one instance of
  each kernel.

  **`TransposeHCShardedProgramFactory`** — two configurations selected at cache-miss time by
  `hc_rm_sharded_is_special_case(shard_height, H, C)` (`:317-320`, `:344`):

  | CB (index) | Built at | Config | Touchers | Census | Disposition |
  |---|---|---|---|---|---|
  | `in0` (`c_0`, borrowed from the input shard, `shard_height` pages of `stick_size_bytes`) | `:357-368` | **special case** (`USE_SPECIAL_CASE`) | reader: `dfb_in.get_read_ptr()` peek used as the NoC address of *remote* cores' shards (`reader:38`, `:62`, `:66`), no FIFO ops → role-free; writer: same peek (`writer:38`, `:56`, `:60`), no FIFO ops → role-free | 2 touchers, both role-free | **1P+1C** — bind reader PRODUCER, writer CONSUMER (cosmetic on Gen1) |
  | | | **generic** | reader only: `get_read_ptr()` peek (`reader:149`), no FIFO ops | 1 toucher | **self-loop** on the reader (DM self-loop, legal on Gen1) |
  | `out` (`c_16`, borrowed from the output shard) | `:370-380` | **special case** | reader: `dfb_out.get_write_ptr()` peek + NoC reads *into* it (`reader:39`, `:61`, `:47-54`, `:74-81`), no FIFO ops → role-free; writer: same (`writer:39`, `:55`, `:41-47`, `:68-74`) → role-free | 2 touchers, both role-free | **1P+1C** — bind reader PRODUCER, writer CONSUMER |
  | | | **generic** | reader only: `reserve_back(num_sticks_per_core)` `:123`, `get_write_ptr()` `:124`, NoC reads into it `:152-157`, `push_back(num_sticks_per_core)` `:177` → locked producer; nothing drains the resident output | 1 toucher | **self-loop** on the reader |

  The writer `KernelDescriptor` exists only in the special case (`:437-457`); in the generic path the kernel set is
  the reader alone (`:413-415`). No hidden second writer beyond the visible split-reader co-fill (both co-fillers are
  in the census above, no semaphore gates them); no dead CB; no multi-binding flag.

  **`TransposeWHShardedRMProgramFactory`** — one kernel set; the CB set depends on `ht > 8` (`:142`), where
  `ht = ceil(H / 32)` (`:35`):

  | CB (index) | Built at | Config | Touchers | Census | Disposition |
  |---|---|---|---|---|---|
  | `in0` (`c_0`, borrowed from the input shard) | `:90-100` | all | reader: `dfb_src.get_read_ptr()` peek (`reader:31`) feeding loopback reads (`:33-36`, `:45-52`), no FIFO ops | 1 toucher, role-free | **self-loop** on the reader |
  | `in` (`c_24`, `2*wt` tiles, double-buffered) | `:116-126` | all | reader: `reserve_back(Wt)` `:40`, `get_write_ptr()` `:41`, `push_back(Wt)` `:57` → locked PRODUCER; compute: `compute_kernel_lib::tilize<Wt, c_24, c_25, …>(Ht)` (`compute:151-158`) consumes it (`in_dfb.wait_front` / `pop_front`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:163-189`) → locked CONSUMER | 1P + 1C | **legal 1:1** |
  | `tilize` (`c_25`, `ht*wt` tiles) | `:129-139` | all | compute only: the `tilize` helper packs into it (`out_dfb.reserve_back` / `push_back`, `tilize_helpers.inl:180-188`) and the kernel drains it (`dfb_tilize_buf.wait_front(HtWt)` `compute:161`, `transpose_tile` reads `:36` / `:87`, `pop_front(HtWt)` `:181`) | 1 toucher | **self-loop** on the compute kernel |
  | `im2` (`c_26`, `ht` tiles) | `:143-153`, only when `ht > 8` | `ht > 8` | **none** — no kernel references index 26: the compute kernel's CB set is `c_24` / `c_25` / `c_27`-or-`c_16` (`compute:131-134`), the reader's `c_0` / `c_24` (`reader:22-23`), the writer's `c_27` / `c_16` (`writer:21-22`); no CTA carries the index (compute CTAs `:209-219`, reader `:169-179`, writer `:190-198` are all geometry / page counts); the tilize helper takes its CBs as NTTPs from the kernel. `grep -n 'c_26\|\b26\b'` over the three kernels and `tilize_helpers.hpp`: zero hits | 0 touchers in every config where it exists | **dead-CB drop** (`:143-153`) — a dead CB has no behavior, so removing its allocation changes none; PR #54333 dropped the analogous dead `c_25` on the non-sharded WH RM path |
  | `out_staging` (`c_27`, `2*ht` tiles) | `:156-166`, only when `ht > 8` | `ht > 8` | compute: `reserve_back(Ht)` / `pack_untilize_dest` / `push_back(Ht)` (`compute:83-96`, or `:43-53` on the narrow-row path) → locked PRODUCER; writer: `wait_front(Ht)` `:41`, `get_read_ptr()` `:42`, `pop_front(Ht)` `:58` → locked CONSUMER | 1P + 1C | **legal 1:1**, as a **conditional DFB** (exists only when `ht > 8`; see *Heads-ups* for the kernel-side `#ifdef` gating) |
  | `out` (`c_16`, borrowed from the output shard, page = `output_page_size` `:110`) | `:103-113` | `ht > 8` | writer: `dfb_dst.get_write_ptr()` peek (`writer:30`) feeding loopback NoC writes into it (`:34-37`, `:46-53`), no FIFO ops → role-free; the compute kernel packs into `c_27` instead (`compute:133-134`) | 1 toucher | **self-loop** on the writer (DM self-loop, legal on Gen1) |
  | | | `ht <= 8` | compute: `reserve_back` / `push_back` straight into `c_16` (`cb_out_idx = c_16`, `compute:134`; FIFO ops `:43-53` / `:83-96`) → locked PRODUCER; writer: constructs `dfb_dst(c_16)` and peeks `get_write_ptr()` unconditionally (`writer:22`, `:28`, `:30`) even though its transfer loop is compiled out (`if constexpr (Ht > 8)`, `:33`) → role-free toucher | 2 touchers, one locked producer | **1P+1C** — compute PRODUCER, writer CONSUMER |

  No hidden second writer (no semaphores anywhere), no dual-instance work-split (five distinct sources, one instance
  each), no multi-binding flag.

- **Offset base pointers:** **GREEN.** There is **no address RTA, CRTA or CTA in either factory**: the HC reader's
  RTAs are counts, stick offsets and NoC coordinates (generic `hc_sharded:60-62`, special case `:271-287`), the HC
  writer's likewise (`:289-308`); the WH sharded RM kernels take **no runtime args at all** (CTAs only, `:169-179`,
  `:190-198`, `:209-219`). Both tensors enter solely through borrowed CBs (`.buffer`, `:367` / `:379` and `:99` /
  `:112`). The byte offsets the kernels add to a DFB base pointer in-kernel (`read_stick_offset[core]`,
  `src_read_stick_offset`, `stick_id_in_shard * stick_size_bytes` — `reader:38`, `:66`, `:149`; `writer:38`,
  `:56`, `:60`) are kernel-side arithmetic on a framework-delivered base, not a host fold. `transpose` is not in
  the tables of `2026-07-19_offset_base_pointers.md` → the *"no fold, op not in the tables"* outcome. Type 3
  (`address_offset`) is N/A per Appendix A; Type 4 (`narrow`) is not used.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument**, because neither factory's
  kernels construct a `TensorAccessor` at all (no `TensorAccessorArgs` on the host either — `grep -n
  TensorAccessor` on both factories and the five kernels returns nothing). `2026-07-06_tensor_accessor_3rd_arg_triage.md`
  does not list `transpose`. Nothing to drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; identical in both factories):
  - `input` — **clean** (borrowed-memory DFB `c_0`, `hc_sharded:357-368` / `wh_sharded_rm:90-100`) →
    `TensorParameter` + `DataflowBufferSpec::borrowed_from`; the kernels keep reading through
    `DataflowBuffer::get_read_ptr()`.
  - `output` — **clean** (borrowed-memory DFB `c_16`, `hc_sharded:370-380` / `wh_sharded_rm:103-113`) → same.
  - No Case 1, no Case 2; no `TensorAccessor` anywhere.
- **TensorParameter relaxation:** `none` (both rows). Keep strict.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - HC sharded — special case: 1P+1C on `in0` and `out` (reader PRODUCER, writer CONSUMER); generic: self-loop on
    `in0` and `out` (reader).
  - WH sharded RM — `in0` self-loop (reader); `in` legal 1:1; `tilize` self-loop (compute); `im2` dead-CB drop
    (`wh_sharded_rm:143-153`); `out_staging` legal 1:1, conditional on `ht > 8`; `out` self-loop (writer) when
    `ht > 8`, 1P+1C (compute PRODUCER, writer CONSUMER) when `ht <= 8`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB has more than two touchers and no two kernels
  share a locked FIFO role. The two-toucher CBs are the HC special-case split-reader co-read of `in0` and co-fill
  of `out` (reader + writer, both raw, no semaphore) and the WH RM `out` under `ht <= 8`; all resolve to 1P+1C.
- **Cross-op / shared kernels — the one real decision in this port.** `device/kernels/compute/transpose_wh_rm.cpp`
  is an **intra-op shared** kernel: the legacy copy is bound by `TransposeWHShardedRMProgramFactory` with
  `SHARDED=1` (`wh_sharded_rm:221-222`, `:235-236`), and its `_metal2` fork `transpose_wh_rm_metal2.cpp` is bound
  by the already-ported `TransposeWHProgramFactory` (`transpose_wh_program_factory.cpp:197`) **without** `SHARDED`.
  The fork's header (`:11-16`) states it carries *only the non-sharded path*; the sharded variant — CTAs 3-8
  (`compute:107-113`), the `use_narrow_row` / `row_size` selection with its yolov4 PCC note (`:114-124`), the
  `transpose_with_pack_untilize_narrow_row` helper (`:18-58`), the `c_24` / `c_25` / `c_27`-or-`c_16` CB set
  (`:131-134`) — lives only in the legacy copy. So the shared-kernel caution's rung 1 (reuse) **fails the fit
  check**, rung 2 (create the fork) is pre-empted (a fork exists and *"do not fork the fork"*), and the fork
  *"already has a consumer"* so it is read-only under the rule as written. After this port the legacy copy has **no
  binder left on `main`** (the quasar copy binds its own duplicate), so the kernel is at its sunset point — but the
  fork cannot take over the legacy name while it lacks the sharded path. The recipe has no sanctioned path for this
  shape; it is raised as **Question 1** below with a recommendation (extend the fork under its own `#ifdef SHARDED`
  dialect boundary, which cannot break `TransposeWHProgramFactory`'s build since that factory never defines
  `SHARDED`). **Do not decide this unilaterally at port time.** No borrowed kernel file; nothing lent.
- **RTA varargs (HC sharded only; WH sharded RM has no RTAs):**
  - reader, special case (`reader_unary_transpose_hc_sharded_rm.cpp:21-23`): three arrays of `num_cores_read`
    entries — `read_stick_offset`, `noc_coord_x`, `noc_coord_y` — read through `get_arg_addr` pointers at RTA
    offsets `5`, `5 + n`, `5 + 2n` and indexed `[core]` inside loops (`:38`, `:44`, `:52`, `:66`, `:71`, `:79`) →
    **varargs** (`3 × num_cores_read` per node; host builds them at `hc_sharded:282-287`). The five scalars at
    offsets 0-4 (`read_single_h_block_per_core`, `num_C_blocks_per_core`, `num_sticks_per_shard_core`,
    `num_cores_read`, `read_stick_stride`; `:16-20`, host `:273-281`) are **named**.
  - reader, generic (`:110-113`): `shard_grid_x_map` (`num_cores_x` entries) and `shard_grid_y_map`
    (`num_cores_y` entries), CTA-bounded, indexed by a data-derived shard id (`:142-147`) → **varargs**
    (`num_cores_x + num_cores_y`; host `:61-62`). The five scalars at offsets 0-4 (`num_sticks_per_core`,
    `start_id`, `curr_c`, `curr_h`, `curr_n`; `:104-108`, host `:60`) are **named**.
  - writer, special case (`writer_unary_transpose_hc_sharded_rm.cpp:21-23`): the same three arrays at offsets `7`,
    `7 + n`, `7 + 2n` → **varargs**; the seven scalars at offsets 0-6 (`:14-20`, host `:292-302`) are **named**.
  - **Count per `KernelSpec`.** `KernelAdvancedOptions::num_runtime_varargs` is one count for the whole spec
    (`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:70-79`); the per-node override is
    deprecated (`:93-101`). The generic count is uniform by construction. The special-case count is
    `3 × num_non_repeat_cores`, computed per core (`hc_sharded:202-233`); on a rectangular shard grid every shard id
    passes the bbox filter (`:183`) so it is uniform too. If a configuration ever varies it across cores, zero-pad
    each node's vector to the maximum rather than reach for the deprecated per-node count — the kernels index by
    the named `num_cores_read`, never by the total.
  - No CTA varargs anywhere: all compile-time-arg reads are at constexpr indices.
- **Conditional DFBs in WH sharded RM (`ht > 8`).** Both kernels that touch `c_27` reference it from contexts that
  C++ name-lookup resolves even on the dead branch: the compute kernel's `constexpr auto cb_out_idx = (Ht > 8) ?
  c_27 : c_16` (`compute:133-134`) and the writer's unconditional `DataflowBuffer dfb_src(c_27)` (`writer:21`,
  `:27`) ahead of its `if constexpr (Ht > 8)` body (`:33-62`). Apply *Pattern: Conditional / optional resource
  bindings* — promote the host condition `ht > 8` (`wh_sharded_rm:142`) to a define on **both** the compute and
  writer specs, `#ifdef`-gate the `out_staging` alias / construction / use in both kernels, and bind the DFB only
  when the define is set. The writer's `dfb_dst(c_16)` construction and peek (`:22`, `:28-30`) run in every
  config, so `out` stays bound on the writer in every config (CONSUMER under `ht <= 8`, PRODUCER+CONSUMER
  self-loop under `ht > 8`). The dead `im2` (`c_26`) is dropped outright, not made conditional.
- **Two kernel sets in HC sharded.** Special case = reader (`USE_SPECIAL_CASE`, CTAs `{in0, out,
  stick_size_bytes}`, `:383-384`, `:398-401`) + writer (`:437-457`); generic = reader only (CTAs `{in0, out, N, H, C,
  stick_size_bytes, row_major, num_cores_x, num_cores_y}`, `:385-395`). *Pattern: Multi-variant factories* — branch
  inside `create_program_artifacts`; the writer `KernelSpec` exists only on the special-case branch; the define goes
  to `compiler_options.defines`. Per-core run-args come from the two helpers (`:23-87`, `:89-314`) and are emplaced
  in shard-orientation order (`:425-433`, `:447-455`) over `core_ranges = all_cores`; keep the same core ↔ args
  mapping (and note `num_cores_x` / `num_cores_y` are the *bounding-box* extents, `:352-355`).
- **Remote-shard reads rely on one L1 address for every core.** The HC kernels read other cores' input shards at
  `dfb_in.get_read_ptr() + offset` through the NoC (`reader:38`, `:66`, `:149`; `writer:38`, `:60`), i.e. they
  assume the borrowed input DFB sits at the same L1 address on every core. A `borrowed_from` DFB inherits the one
  sharded buffer's address, so this holds after the port; mention it in the port report rather than re-derive it.
- **Loopback endpoints in WH sharded RM (reader `:33-36`, `:45-52`; writer `:34-37`, `:46-53`) — keep as is.** The
  migration guide's *"Avoid"* form; the preferred local-handle form is a transfer rewrite, out of port scope.
  Precedent for carrying it through a port unchanged: the checked-in fork
  `data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:133-137`.
- **Compute `hw_config` (WH sharded RM) — Style B, direct `ComputeConfigDescriptor` (`:241-244`).** Carry:
  `fp32_dest_acc_en = (src0 == Float32)` (`:81`) → `enable_32_bit_dest`; `unpack_to_dest_mode[c_24]` and
  `[c_25]` = `UnpackToDestFp32` when the input is Float32 (`:224-232`) → `unpack_modes = {{in, UnpackToDest},
  {tilize, UnpackToDest}}` under the same condition — exactly the two Float32 DFBs the compute kernel consumes, so
  the validator's required-entry rule is satisfied by them and nothing else is needed (`out_staging` / `out` are
  produced, not consumed). Every other field is left at its Metal default (HiFi4, `Precise`, `double_buffer_dest =
  true`, `bfp_pack_precision_mode` default), which matches the legacy descriptor defaults. Note the live header is
  a flat `ComputeHardwareConfig` with a nested `Compute1XXConfig` (`compute_hardware_config.hpp:43-127`), not the
  recipe's `ComputeGen1Config` name. `DST_ACCUM_MODE` in `compute_num_blocks_per_col` (`compute:62`) follows
  `enable_32_bit_dest` unchanged.
- **`opt_level`:** `grep -n opt_level` on both factories returns nothing → the WH sharded RM compute spec resolves to
  **O3** today and must say `KernelBuildOptLevel::O3` explicitly (`kernel_spec.hpp:122` defaults to `O2`); all four
  DM specs resolve to O2 (nothing to do). DM configs are `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`
  (HC `:411`, `:445`; WH RM `:188`, `:207`) → the arch-agnostic reader / writer helpers.
- **Magic CB indices in the WH sharded RM kernels** (`reader:22-23` `c_0` / `c_24`; `writer:21-22` `c_27` / `c_16`;
  `compute:131-134`) and CTA-carried indices in the HC kernels (`reader:25-26`, `:93-94`; `writer:25-26`) →
  `dfb::` tokens (whitelist rule 2). The HC kernels also build `DataflowBuffer`s from CTA ids (`reader:30-31`,
  `:120-121`; `writer:30-31`) → construct from the tokens (rule 3).
- **Kernel-lib call keeps its id-only form.** `compute_kernel_lib::tilize<Wt, cb_in, cb_tilize, InitAndUninit,
  WaitBlock, NoReconfigure, Lossless>(Ht)` (`compute:151-158`) takes the CB ids as `uint32_t` NTTPs and builds its
  own `DataflowBuffer`s inside (`tilize_helpers.hpp:176-190`); pass `dfb::` tokens in that position (*Pattern: Pass
  DFB handles directly to LLKs and kernel-lib helpers*). The by-reference form is only `[[deprecated]]` under
  `ARCH_QUASAR` (`id_only_helpers.hpp:14-21`) — a Quasar-uplift item, not port work.
- **Dropped plumbing to record (WH sharded RM):** reader CTAs 7 and 8 (`wh_sharded_rm:178-179`) duplicate CTAs 2
  and 3 and are never read (the reader reads 0-6, `reader:14-20`) — drop them. Compute CTAs 5 (`pack_num_pages`)
  and 7 (`pack_num_pages_last_row`) are read (`compute:110`, `:112`) but never used — name them (the kernel reads
  them) and note them in the port report's *Dropped Plumbing / Open items*.
- **`num_hw_blocks_per_core` differs in kind between the two paths of the shared compute kernel:** a `constexpr`
  CTA under `SHARDED` (`compute:108`, host `:213`) but an RTA on the non-sharded path (`:127`; fork `:76`). If the
  fork is extended (Question 1), keep the sharded path's `constexpr auto … = get_arg(args::num_hw_blocks_per_core)`
  under `#ifdef SHARDED` — the loop bound (`:147`) is compile-time there, and demoting it would change unrolling.
- **Override translation (both factories) is tensor-args-only.** Each legacy override re-points exactly the two
  borrowed CBs and nothing else (`hc_sharded:468-473`, `wh_sharded_rm:259-264`; comment: *"only the buffer
  addresses move"*). The `CustomProgramSpecFactoryConcept` override therefore returns `ProgramRunArgs` with
  `tensor_args = {{INPUT, …}, {OUTPUT, …}}` and **no** `kernel_run_args` — mirror the set exactly (the HC
  special-case varargs and scalars are pinned by the cache key; do not add a refresh the legacy override did not
  do). The two `TT_ASSERT`s live in `create_descriptor` (`hc_sharded:328-329`, `wh_sharded_rm:23-24`), not the
  override; carry them into `create_program_artifacts`.
- **Tests (no-regression baseline, confirm with the invoker):**
  `tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py` —
  WH sharded RM: `test_transpose_hw_sharded_rm` (`:473`; `skip_for_blackhole`, #12349, `:468`),
  `test_transpose_hw_sharded_rm_with_program_cache` (`:544`), `test_transpose_wh_sharded_rm_override_addr_change`
  (`:816`); HC sharded: `test_transpose_hc_sharded_with_program_cache` (`:665`),
  `test_transpose_hc_sharded_cache_hit_new_input` (`:721`; ids `generic` / `special_case_single_h_block` /
  `special_case_multi_h_block` cover both kernel sets and both special-case branches),
  `test_transpose_hc_sharded_override_addr_change` (`:807`); both: the row-major height/height cases of
  `test_transpose_sharded` (`:1561`) and `test_transpose_sharded_rm_width` (`:1629`). The `ht > 8` and `ht <= 8`
  WH RM configs are both exercised (`h=16` → `ht=1`; `h=128` → `ht=4`; `test_transpose_sharded` with `H=256` →
  `ht=8`; check that at least one case lands on `ht > 8`, e.g. by adding `H=512` locally — none of the listed tests
  obviously does).

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** (no `uint32_t sem_id` / sem-address /
  `TensorAccessorArgs<N>` / `CircularBuffer&` / addr-gen shapes in any called donor function).

  | Op kernel | Donor file | Class | Functions called | Shape |
  |---|---|---|---|---|
  | HC reader / HC writer / WH RM reader / WH RM writer | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/*` | `Noc`, `DataflowBuffer`, `UnicastEndpoint`, `CoreLocalMem`, `my_x` / `my_y` (WH RM only), `get_arg_val` / `get_arg_addr` / `get_compile_time_arg_val` | ✓ framework API, no op donor |
  | WH RM compute | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | kernel-lib (class 2) | `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, InitUninitMode, WaitMode, ReconfigureRegisterDatatypeMode, Fp32Mode>(uint32_t num_blocks)` (`tilize_helpers.hpp:176-190`, id-only form) — CB ids as `uint32_t` NTTPs | ✓ `uint32_t cb_id` |
  | WH RM compute | `api/compute/eltwise_unary/eltwise_unary.h`, `api/compute/transpose.h`, `api/compute/tilize.h`, `api/compute/pack_untilize.h`, `api/dataflow/dataflow_buffer.h` | `tt_metal/*` (LLK) | `compute_kernel_hw_startup(uint32_t, uint32_t)`, `copy_init`, `transpose_init`, `transpose_tile`, `pack_untilize_dest_init<…>`, `pack_untilize_dest<…>`, `pack_untilize_uninit`, `tile_regs_*` | ✓ `uint32_t cb_id` |

  Borrowed kernel files: **none** — all five in-scope kernels live in and are owned by this op. **Lent:** none —
  the filename census (`grep -rl <file> ttnn/cpp/ttnn/operations/`) finds only this op's factories plus the
  quasar copy's *own duplicates* (bound paths under `experimental/quasar/transpose/device/kernels/`). **Intra-op
  shared:** `transpose_wh_rm.cpp` ↔ `transpose_wh_rm_metal2.cpp` as described under *Heads-ups*; after this port
  the legacy copy's binder set on `main` is empty, so its sunset is due once the sharded path has a Metal 2.0
  home.
- **Relaxation candidates (FALLIBLE — candidates to verify, default strict):** no custom hash to mine. The
  default reflection hash keys the full `TensorSpec` (shape, shard spec, dtype), which is also what both factories
  bake into CTAs (`H`, `C`, `N`, `stick_size_bytes`, grid extents; `ht`, `wt`, page counts) and CB capacities, so
  the sheet's `none` is consistent; no candidate.
- **TTNN factory analysis:** single-program SPMD; no op-owned tensors; no custom hash (default reflection hash);
  no `get_dynamic_runtime_args` (removed by PR #52566); no pybound `create_descriptor`
  (`transpose_nanobind.cpp:40-61` binds only the public op); `override_runtime_arguments` present on both in-scope
  factories → **`CustomProgramSpecFactoryConcept`**. Tensors reachable from `tensor_args` (`input`) and
  `tensor_return_value` (the output).

  ### TTNN ProgramFactory
  #### Concept
  `CustomProgramSpecFactoryConcept` for both in-scope factories (ported-from factories have an
  `override_runtime_arguments`: `transpose_hc_sharded_program_factory.cpp:462-474`,
  `transpose_wh_sharded_rm_program_factory.cpp:253-265`; both cb-addr-only).
  #### Fit
  - Single vs multi-program: single — one `ProgramDescriptor` today, one `ProgramSpec` stamped across the mesh.
  - Op-owned device resources: none.
  - Tensor-arg matching: strict (sheet `none`).
  - Legacy-to-Metal-2.0 shape: 1:1 with the ported-from factories (the HC factory's two kernel sets become two
    branches of one `create_program_artifacts`).
  #### Custom `compute_program_hash`
  none — default reflection-based hash.
  #### Stop signals
  None.

## Misc anomalies  *(team-only, non-gating)*

- `transpose_wh_sharded_rm_program_factory.cpp:178-179` — reader CTAs 7-8 duplicate CTAs 2-3 (`H_per_tile`,
  `H_per_tile_last`) and are never read by the reader (`reader:14-20` reads 0-6).
- `transpose_wh_sharded_rm_program_factory.cpp:143-153` — `c_26` (`im2`, the "untilize cb") is allocated for
  `ht > 8` and referenced by no kernel (dead CB; the port drops it — see *Gate detail*).
- `transpose_wh_rm.cpp:110`, `:112` — `pack_num_pages` (CTA 5) and `pack_num_pages_last_row` (CTA 7) are read
  under `SHARDED` but never used; only `pack_num_pages_last_col` / `pack_num_pages_last_row_col` reach the narrow-row
  helper (`:171-172`).
- `transpose_wh_rm.cpp:144-145` vs `transpose_wh_rm_metal2.cpp:85` — fork drift: the legacy copy initialises with
  `compute_kernel_hw_startup(cb_in, cb_out_idx); copy_init(cb_in);`, the fork with
  `unary_op_init_common(dfb_in, dfb_out_idx)` (`api/compute/eltwise_unary/eltwise_unary.h:51`), although the fork
  header claims the compute logic is unchanged (`:6`). Whoever extends the fork for the sharded path should decide
  which init pair the sharded path keeps (the legacy one, to stay faithful) and whether the non-sharded path's
  divergence was intentional.
- `transpose_wh_sharded_rm_program_factory.cpp:89`, `:102` — comments still describe the imperative
  `UpdateDynamicCircularBufferAddress`; the descriptor path re-points through `apply_descriptor_runtime_args`.
- `transpose_wh_sharded_rm_program_factory.cpp:84` — `num_cores` is `[[maybe_unused]]` and only logged.
- `writer_unary_transpose_wh_sharded_rm.cpp:21`, `:27` — constructs a `DataflowBuffer` on `c_27` even when that CB
  is not allocated (`ht <= 8`); harmless on legacy (it just reads an unconfigured CB interface), but it is the
  construct a naive port trips over (see *Heads-ups*).
- `transpose_hc_sharded_program_factory.cpp:105-122` — `height` and `cores` are computed and never used;
  `:238-243` — a loop whose body only `break`s (no effect); `:236-237` — a TODO for multi-batch shards that the
  `shard_height <= C*H` guard in `hc_rm_sharded_is_special_case` (`:319`) currently keeps out of the special case.
- `transpose_hc_sharded_program_factory.cpp:336-337` use `logical_shape()` for `N/C/H/W` while both runtime-arg
  helpers use `padded_shape()` (`:25`, `:91`); consistent only for an unpadded row-major tensor.
- `reader_unary_transpose_hc_sharded_rm.cpp:95` (CTA `N`, generic path) and `:108` / `:166` (RTA `curr_n`,
  incremented, never read) are dead plumbing; the port names them (the kernel reads them) and reports them.

## Per-DeviceOperation attribution  *(single DeviceOperation; per-factory split)*

| Factory | Status | Target / note |
|---|---|---|
| `TransposeHCShardedProgramFactory` | **GREEN** — audited, brief issued | `CustomProgramSpecFactoryConcept`; two kernel sets (special / generic); varargs on the HC reader / writer |
| `TransposeWHShardedRMProgramFactory` | **GREEN** — audited, brief issued | `CustomProgramSpecFactoryConcept`; conditional `out_staging` DFB, dead `im2` DFB, shared compute kernel (Question 1) |
| `TransposeCNProgramFactory`, `TransposeHCRMProgramFactory`, `TransposeHCTiledInterleavedProgramFactory`, `TransposeHCTiledProgramFactory`, `TransposeWHProgramFactory`, `TransposeWHShardedProgramFactory` | **done** (`MetalV2`, PR #54333) | not audited; sheet rows 94/95/97/98/99/100 stale (`Concept = descriptor`) |

## Questions for the user

1. **How should the sharded path of `transpose_wh_rm.cpp` reach Metal 2.0?** The legacy copy
   (`device/kernels/compute/transpose_wh_rm.cpp`) is the only Metal-2.0-less binder left after this port; its fork
   `transpose_wh_rm_metal2.cpp` (bound by the already-ported `TransposeWHProgramFactory`) deliberately omits the
   `SHARDED` path (fork `:11-16`). The shared-kernel caution forbids editing a fork that has a consumer, forbids a
   fork of the fork, and closes the `#ifdef` escape — leaving no sanctioned path for the second variant of a
   `#ifdef`-split kernel. Options, in my order of preference:
   - **(a) Extend the fork under `#ifdef SHARDED`** (recommended): add the sharded path — CTAs 3-8 as named args,
     the narrow-row helper and its yolov4 note (`compute:114-124`), the `c_24` / `c_25` / conditional `c_27`
     bindings — behind the same `SHARDED` define the legacy file already uses as its dialect boundary. It cannot
     affect `TransposeWHProgramFactory`'s build (that factory never defines `SHARDED`), the non-sharded names and
     args stay untouched, and it leaves the kernel at its natural sunset state (delete the legacy copy, rename the
     fork). This needs your explicit go-ahead because the recipe's rule says otherwise.
   - **(b) A second fork, e.g. `transpose_wh_rm_sharded_metal2.cpp`,** sharded path only. Keeps the existing fork
     read-only but splits one kernel across two Metal 2.0 files permanently, with a name outside the `<stem>_metal2`
     convention.
   - **(c) Convert the legacy copy in place** (rung 3; its binder set is now exactly this factory). Ends with two
     Metal 2.0 files whose names invert their roles (`transpose_wh_rm.cpp` sharded-only, `…_metal2.cpp`
     non-sharded-only). Not recommended.

## Recipe notes

1. **Sheet staleness on sibling rows that are already ported.** The TTNN gate's cross-check has no wording for a
   primary-column conflict (`Concept`) on rows describing factories *outside* the port's scope. Here six of eight
   rows say `descriptor` while the code has been `MetalV2` since 2026-09-01; the two in-scope rows are accurate. I
   treated it as a reconciliation item for the sheet owner rather than a *"spreadsheet broken"* gate on the two
   factories being audited — the recipe should say which it intends, since a literal reading gates a port on rows
   that no longer matter to it.
2. **`#ifdef`-variant kernels and the shared-kernel caution.** *Caution: Porting a shared kernel* assumes a fork
   either fits or doesn't; it does not anticipate a fork that carries *one variant* of a `#ifdef`-split kernel while
   the other variant is still needed by a later port. Its three rules (fork read-only, no fork-of-fork, `#ifdef`
   escape closed) together leave no path. Suggested addition: when the define is already the legacy file's own
   dialect boundary, extending the fork behind that same define is sanctioned (it cannot break a consumer that does
   not set the define). See Question 1.
3. **Degenerate `override_runtime_arguments`.** Both overrides here are cb-addr-only (`apply_descriptor_runtime_args`
   with the two borrowed CBs), which is exactly the refresh the base concept performs automatically. The recipe
   still routes them to `CustomProgramSpecFactoryConcept` and asks the porter to write an override that returns only
   `tensor_args`. That is consistent, but a sentence acknowledging the case — and pointing at
   `scripts/detect_override_rebuild.py`'s `cb-addr-only` tag as the recognition signal — would spare the porter a
   double-take.
4. **Vararg count is per `KernelSpec`.** The RTA varargs subject could tell the auditor to check whether the
   vararg count is uniform across nodes, since `num_runtime_varargs` is one number per spec and the per-node
   override is deprecated (`advanced_options.hpp:93-101`); a split-reader op like this one computes the count per
   core.
5. **Stale type names in the port recipe (already noted by the `attn_matmul` audit):** `ComputeGen1Config` /
   `to_compute_hardware_config(arch, cfg)` — the live header is a flat `ComputeHardwareConfig` with nested
   `Compute1XXConfig` (`compute_hardware_config.hpp:43-127`).
6. **The Device 2.0 gate has no wording for the migration guide's "Avoid" forms** (already raised by the
   `attn_matmul` audit); the WH sharded RM reader / writer use the hand-built loopback endpoint and I again treated
   it as compliant-but-discouraged.
7. **Workspace recipe cannot be followed literally** (already raised): `akertesz/op-porting-recipe` is an orphan
   history; this audit branched off `main` and read the recipe docs from a local export.
