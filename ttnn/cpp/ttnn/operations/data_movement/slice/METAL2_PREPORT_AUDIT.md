# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

One DeviceOperation, five program factories (single bundled report; findings are attributed per factory
throughout — see *Per-DeviceOperation / per-factory attribution*):

- **`ttnn::prim::SliceDeviceOperation`** (`device/slice_device_operation.hpp:31`)
  - `SliceRmProgramFactory` (`device/slice_program_factory_rm.cpp`)
  - `SliceRmShardedProgramFactory` (`device/slice_program_factory_rm_sharded.cpp`)
  - `SliceRmStrideProgramFactory` (`device/slice_program_factory_rm_stride.cpp`)
  - `SliceTileProgramFactory` (`device/slice_program_factory_tile.cpp`)
  - `SliceTileTensorArgsProgramFactory` (`device/slice_program_factory_tile_tensor_args.cpp`)

**Kernels in scope** (every kernel a factory `kernel_source`s, own + donor):

| Kernel | Factory | Owner |
|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | Rm | slice |
| `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | Rm | slice |
| `slice_reader_unary_unpad_dims_rm_sharded.cpp` | RmSharded | slice |
| `reader_multicore_slice_4d.cpp` / `writer_multicore_slice_4d.cpp` | RmStride (rank ≤ 4) | slice |
| `reader_multicore_slice_nd.cpp` / `writer_multicore_slice_nd.cpp` | RmStride (rank > 4) | slice |
| `reader_unary_unpad_dims_interleaved_start_id.cpp` | Tile | slice |
| `writer_unary_interleaved_start_id.cpp` (slice's own copy) | Tile | slice |
| `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | TileTensorArgs | slice |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | TileTensorArgs | **eltwise/unary (cross-family donor)** |

**Unreferenced kernel files in the op directory** (out of scope, contents not audited — listed only so a
reader does not mistake them for live code): `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp`,
`device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. No factory names either file.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** *not pinnable* — `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
prints nothing in this checkout (that path does not exist here; `.../host_apis/` holds only
`buffers/ device_management/ kernels/ profiler/ program/ runtime_args/`). The audit ran against
`/localdev/edwinlee/metal2_audit.md`, an out-of-tree copy, so the guidance version cannot be recorded.

**Reference data:** readiness sheet *"TTNN Operations analysis"* fetched 2026-09-01 (Drive
`1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner `dgomez@`, modifiedTime `2026-09-01T05:26:10Z`).
Five slice rows, one per factory. The two dated triage analyses the recipe names as priors —
`2026-07-19_offset_base_pointers.md` and `2026-07-06_tensor_accessor_3rd_arg_triage.md` — are **not present
in this checkout and were not available**; both subjects below were therefore resolved from the code alone,
per each subject's "your own scan is the source of truth" rule. The offset-base-pointer finding is
independently corroborated by the sheet's `Known op issues` cell; the 3rd-argument finding has **no
external corroboration** and should be read as a fresh classification (see *Questions*).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice` |
| **Overall** | **RED** — at op level, **no portable subset** (all five factories blocked) |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes — GREEN.** All nine referenced kernels (slice-owned + the eltwise/unary donor) are structurally Device 2.0: `Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor`. No CB-index-keyed free-function holdovers. |
| *Prereqs* — Cross-op escapes | **Ok** (kernel-side ✓ clean) — **but see the host-side issue**: `ccl/mesh_partition` drives all five slice factories and `patch_slice_program_addresses` directly. Not a gate; a bundled-port coordination requirement. |
| *Feature Support* — overall | **GREEN** — all three Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no kernel reads a compile-time arg at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No** — `no` on **all five** factory rows. Attributed: `TensorParameter relaxation` = `(legality - pending analysis)` (all 5) **and** `Known op issues` = `offset base ptr to TensorAccessor` (`SliceRmProgramFactory`) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 5) — cross-check ✓ (`create_descriptor` returning `ProgramDescriptor` on each factory) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `device/slice_device_operation.cpp:302-354`. Cross-check ✓ |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** ✓ — no such hook on the DeviceOperation. Cross-check ✓ (comments still reference one — see *Misc anomalies*) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`) — all five factories: `slice_program_factory_rm.cpp:435`, `slice_program_factory_rm_sharded.cpp:418`, `slice_program_factory_rm_stride.cpp:178`, `slice_program_factory_tile.cpp:189`, `slice_program_factory_tile_tensor_args.cpp:195`. Cross-check ✓ |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** (not a gate; port deletes the binding): `slice_nanobind.cpp:164-176` (`SliceTileProgramFactory.create_descriptor`). Sheet value is the literal string `PR` on all five rows — see *Gate detail* |
| *TTNN Readiness* — Op-owned tensors | No (sheet blank; consistent with `descriptor`) ✓ |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all five) — matches the sheet's `Porting Target` |
| *Port work* — Offset base pointer | **GATE** → ops team + framework/Audrey (**Type 2 — accessor-fed offset arg**), `SliceRmProgramFactory` only. The other four factories are clean on this gate. |
| *Port work* — Tensor bindings (per binding) | ⚠ port work — 9 Case 1, 2 clean (borrowed-DFB), 1 gated upstream (the Type-2 base). No Case 2. |
| *TTNN Readiness* — TensorParameter relaxation | `(legality - pending analysis)` on **all five** rows → **GATE** → ops team *(the known item the requester asked to set aside)* |
| *Port work* — TensorAccessor 3rd arg | **Two sites, both in `SliceRmProgramFactory`'s kernels.** Class 2 (drop) under interleaved and HEIGHT_SHARDED; **Class 3 / Special → GATE** under BLOCK/WIDTH_SHARDED. |
| *Port work* — CB endpoints | legal (4 CBs, plain 1:1) + **self-loop** ×3 (`RmSharded` c_0, `RmSharded` c_16, `TileTensorArgs` c_1). No dead CB, no multi-binding, no conditional DFB. |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves with a **self-loop**
(one toucher). No CB needs the multi-binding advanced option, and none is dead.

---

## Result

**RED at op level; no portable subset.** All five factory rows read `Is able to port? = no`, so there is no
clean factory subset to carve out and **no porter brief is issued**.

Four distinct gates are open. Two are the items the requester already knows about; **two more sit
downstream of them**, and one of those is new information:

| # | Gate | Scope | Owner | Known? |
|---|---|---|---|---|
| 1 | `TensorParameter relaxation` = `(legality - pending analysis)` — the hash/relaxation legality analysis | all 5 factories | ops team | **known** |
| 2 | **Offset base pointer, Type 2** — `input.buffer()->address() + begins_bytes - misalignment` fed to a `TensorAccessor` as its base | `SliceRmProgramFactory` | ops team **+ framework/Audrey, flag early** | on the sheet as `Known op issues`; characterised here |
| 3 | **`TensorAccessor` 3rd argument, Class 3 / Special** — a load-bearing per-shard page-size override that the Metal 2.0 binding model cannot express, and that conflates page *payload* with page *stride* | `SliceRmProgramFactory` (reader + writer) | ops team | **NEW — not on the sheet, no triage-doc coverage available** |
| 4 | TTNN factory-concept gate (`Is able to port? = no`) — the roll-up of #1 and #2 | all 5 factories | TTNN / readiness-sheet owner (verdict); ops team (causes) | **known** |

**Answer to "what else is downstream after the hashing/relaxation work?"** — three things, in descending
order of risk:

1. **Gate #3, the `TensorAccessor` 3rd-argument override** (below). This is the one item no existing
   record covers. It is *not* the usual redundant page-size arg that drops mechanically: on
   BLOCK/WIDTH-sharded row-major tensors the override is deliberately set to the per-shard payload width
   so a shared helper can re-index shard-relative, and Metal 2.0 supplies `aligned_page_size` implicitly
   with **no override API**. Resolving it also exposes what looks like a pre-existing payload-vs-stride
   defect in `noc_async_{read,write}_sharded`.
2. **`ccl/mesh_partition` is a host-side consumer of slice's factory API** — it calls all five
   `create_descriptor`s and `patch_slice_program_addresses` directly and stores
   `SliceDeviceOperation::program_factory_t` in its own `shared_variables_t`. The port changes the
   factory entry point and deletes the patching shim, so mesh_partition breaks unless it is ported or
   adapted in the same change. Not a gate under this recipe (which inventories *kernel* escapes only),
   but it is a scheduling dependency, and mesh_partition is itself `legacy (MeshWorkload)` /
   `Is able to port? = no` on the sheet.
3. **The port is a `CustomProgramSpecFactoryConcept` translation of an unusually large patcher.**
   `patch_slice_program_addresses` is a 60-line five-way `std::visit` shared across two ops, mixing
   `GetRuntimeArgs` slot-poking, `apply_descriptor_runtime_args` on CB descriptors, and
   `apply_dynamic_runtime_args`; it must become one `ProgramRunArgs`-returning method per factory.
   Additionally, **every reader kernel reads variable-count RTA/CRTA blocks through
   `get_arg_addr`/`get_common_arg_addr` pointers** (six vararg sites), and one of those blocks is
   **mutated in place by the kernel** — which is also where a latent cache-hit bug lives (see
   *Misc anomalies*).

Everything else clears: **Device 2.0 is GREEN** for all nine kernels, **all three Appendix A features are
absent**, **CB endpoint legality is clean** (three self-loops, nothing needing the multi-binding flag),
and the kernel-side donor coupling is `✓ clean`.

**Path forward.** None of the four gates is a missing Metal 2.0 feature. #1 and #2 are ops-team fixes on
their own tracks; #3 needs an ops-team decision on the page-size override (and, if the alignment concern
below is confirmed, a fix to the shared `noc_async_*_sharded` helpers); #4 lifts when #1–#3 do. The op is
then re-audited.

**Scoping-rule disclosure.** Both blocking causes are cleared on the **op-code side** (a relaxation/hash
fix and an offset split both rewrite this code), so the recipe's *Red* scoping rule would have me **skip**
the seven purely-informational subjects on a whole-op RED with no portable subset. I ran them **in full
anyway**, because the requester explicitly asked what lies downstream of the known hash/relaxation work.
Read the informational sections with that caveat: the RM reader/writer detail in particular will change
when the offset split lands, so re-audit rather than porting from this document. Nothing was skipped.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **RED**, all five factories

Sheet cell `Is able to port?` = `no` on every slice row. Two blocking columns explain it:

| Blocking column | Value (verbatim) | Rows | Route |
|---|---|---|---|
| `TensorParameter relaxation` | `(legality - pending analysis)` | all 5 | **ops team** |
| `Known op issues` | `offset base ptr to TensorAccessor` | `SliceRmProgramFactory` | **ops team + framework** |

The `Provisional relaxation finding (Edwin)` column reads `needs fix, then none` on the
`SliceRmProgramFactory` and `SliceTileProgramFactory` rows and is blank on the other three — recorded
verbatim, not interpreted. This is an *attributed* `no`, not an unattributed one; the readiness-sheet
owner needs no question on the verdict itself.

**Lightweight cross-check — clean, with one observation.** Verified against the code:

| Column | Sheet | Code | ✓ |
|---|---|---|---|
| `Concept` | `descriptor` | `create_descriptor` → `ProgramDescriptor` on all five factory `.hpp`s | ✓ |
| `Custom hash` | `yes` | `SliceDeviceOperation::compute_program_hash`, `slice_device_operation.cpp:302` | ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on the DeviceOperation (grep clean; the adapter `static_assert`s it cannot coexist with `override_runtime_arguments`, `mesh_device_operation_adapter.hpp:509`) | ✓ |
| `Override runtime args method?` | `yes` | all five factories define it (sites in the status summary) | ✓ |
| `Pybind descriptor` | `PR` | `slice_nanobind.cpp:167` binds `SliceTileProgramFactory::create_descriptor` | see note |
| `Op-owned tensors?` | blank | no `WorkloadDescriptor`, no `buffers` vector | ✓ |
| Factory-set match | 5 rows | 5 factories in `program_factory_t` (`slice_device_operation.hpp:36-41`), names match one-for-one | ✓ |

Cross-column invariants hold: `get_dynamic_runtime_args = no` on a `descriptor` concept ✓; no op-owned
tensors on a `descriptor` row ✓; `Porting Target = CustomProgramSpecFactoryConcept` consistent with
`Override runtime args method? = yes` ✓.

Two observations that are **not** conflict claims — neither is in the recipe's cross-check set, and
neither is offered as evidence the sheet is broken:

- `Pybind descriptor` carries the literal string **`PR`** (not `yes`/`no`) on all five rows. The code has
  exactly one pybound `create_descriptor`, on the Tile factory (`slice_nanobind.cpp:164-176`); the
  nanobind file additionally exposes `SliceParams`, `SliceInputs` and `SliceDeviceOperation`'s
  `create_output_tensors` / `compute_output_specs` (`slice_nanobind.cpp:134-163`). Recorded so the porter
  knows which sites the deletion touches. Question for the sheet owner below.
- `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `no`, while `SliceRmProgramFactory` does put a
  `buffer()->address()`-derived value on reader RTA 0. This **reconciles**: the column tracks the
  *stale-pointer* hazard, and because the factory defines `override_runtime_arguments` the value is
  re-emitted on every cache hit (`slice_program_factory_rm.cpp:389-390` → `:423`), so it is never stale.
  The offset that rides along is what gates, and it is captured under `Known op issues`.

### Device 2.0 (every kernel used) — **GREEN**

Every kernel the op instantiates is structurally Device 2.0. No `InterleavedAddrGen` /
`ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no bare `noc_async_read` /
`noc_async_write`, no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, no raw
semaphore addresses, no `evil_set_*_ptr`. Every `get_write_ptr()` / `get_read_ptr()` in the op is a
`DataflowBuffer` **method** on an in-scope wrapper (14 sites across 9 files), never the CB-index free
function.

Cross-op donor, also clear: `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
constructs a `DataflowBuffer` and a `Noc`, and reads its page size via
`get_local_cb_interface(cb_id_out).fifo_page_size` (`:27`). `get_local_cb_interface(cb_id)` is on the
audit's **sanctioned** free-function list, so this is **not** a holdover and does not knock the op out of
Green — even though a wrapper is in scope and a wrapper method (`get_entry_size()`) exists. (Slice's own
copy of that kernel already uses `dfb_out.get_entry_size()`,
`device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:26`.)

No violations table — there are no violations.

### Feature compatibility — **GREEN** (all entries `N/A`)

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `global_circular_buffer.hpp` include, `CBDescriptor::global_circular_buffer` field, `remote_index(`, `remote_cb_*` identifier, `UpdateDynamicCircularBufferAddress`, or `num_global_cb_receivers` anywhere in the op. The two Buffer-backed CBs (`slice_program_factory_rm_sharded.cpp:290,302`) set only `.buffer` — the plain borrowed-memory pattern, which is a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, not this entry. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor` in the op. The two `CBDescriptor`s that set `.buffer` leave `address_offset` defaulted to 0. (Note: the op's offset problem is a *host-folded pointer* — Type 2 below — **not** a CB `address_offset`; the two are different mechanisms and only the latter is this entry.) |
| GlobalSemaphore | N/A | The op declares **no semaphores at all** — no `SemaphoreDescriptor`, no `CreateSemaphore`, no `GlobalSemaphore`, no `global_semaphore.hpp`. |

Variadic CTA: no kernel calls `get_compile_time_arg_val` at a varying index — every CTA read is at a
literal offset or a `TensorAccessorArgs<N>` constexpr chain.

### Offset base pointers — **RED (Type 2 — accessor-fed offset arg)**, `SliceRmProgramFactory`

**The fold** (`device/slice_program_factory_rm.cpp:43-50`):

```cpp
inline uint32_t slice_rm_reader_base_address(const Tensor& input, const ttnn::Shape& slice_start) {
    const uint32_t begins_bytes = slice_start[-1] * input.element_size();
    const auto src_buffer_alignment = input.buffer()->buffer_type() == BufferType::DRAM
                                          ? ::hal::get_dram_alignment() : ::hal::get_l1_alignment();
    const uint32_t misalignment = begins_bytes % src_buffer_alignment;
    return input.buffer()->address() + begins_bytes - misalignment;   // :49  ← the fold
}
```

Emitted as **reader RTA 0** at `slice_program_factory_rm.cpp:107`, and re-emitted on every cache hit at
`:423` via `slice_rm_reader_dynamic_args` → `apply_dynamic_runtime_args` (`:389-390`).

**The consumption — this is what makes it Type 2, not Type 1**
(`device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:14,33,38`):

```cpp
const uint32_t src_addr = get_arg_val<uint32_t>(0);       // the base+offset value
constexpr auto src_args = TensorAccessorArgs<0>();
const auto s0 = TensorAccessor(src_args, src_addr, padded_stick_size);   // ← offset IS the accessor base
```

The offset address is the `TensorAccessor`'s **base**, not a relocatable trailing `+` on a NoC address, so
the mechanical Type-1 arg split does not apply. Metal 2.0's `TensorAccessor(tensor::name)` ctor takes the
base implicitly from the binding's CRTA word and accepts **no** base override
(`tt_metal/hw/inc/api/tensor/tensor_accessor.h:96-107` for the sharded specialisation, `:409-418` for the
interleaved one — both delegate with the address only). There is no seam.

- **Arg:** `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` — **reader RTA 0, role `src_addr`**.
- **Offset expression:** `begins_bytes - misalignment`, where
  `begins_bytes = slice_start[-1] * input.element_size()` and
  `misalignment = begins_bytes % (DRAM ? dram_alignment : l1_alignment)`.
  Deterministic from cache-miss inputs (`slice_start`, dtype, buffer type are all hashed), so this is a
  real addressing pattern, not a one-off constant.
- **Why the fold exists:** the kernel needs an *aligned* start below the requested byte offset, then
  fixes up the residue on-device with a whole-stick `tt_memmove`
  (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:97-101`), passing `misalignment` separately
  as reader RTA 5. So the residue is *already* split out; only the aligned-multiple part is folded.
- **Route:** **ops team** to refactor before the port, **and framework/Audrey — flag early.** The likely
  shape (unsettled) is a plain `TensorAccessor` on the clean base plus kernel-side pointer manipulation,
  weighed against a first-class tensor-view binding. The affected variant is row-major, matching the
  recipe's characterisation of the wall as an RM phenomenon.

**Other factories scanned — clean, and the reason each is clean matters:**

| Factory | Address args | Verdict |
|---|---|---|
| `SliceRmShardedProgramFactory` | none — both tensors reach the kernel as borrowed-memory CBs (`.buffer = input.buffer()` / `output.buffer()`, `slice_program_factory_rm_sharded.cpp:290,302`). The width offset travels as a **separate CTA** (`begins_bytes`, `:310`) and is added kernel-side (`slice_reader_unary_unpad_dims_rm_sharded.cpp:69`). | ✓ clean — offset already split out |
| `SliceRmStrideProgramFactory` | reader RTA 0 = `input_buffer`, writer RTA 0 = `output_buffer` — bare `Buffer*` bindings (`slice_program_factory_rm_stride.cpp:128,136,147,160`) | ✓ clean base |
| `SliceTileProgramFactory` | reader CRTA 0 = `src0_buffer`, writer RTA 0 = `dst_buffer` (`slice_program_factory_tile.cpp:143,180`). The tile start offset rides a **clean tile-index scalar** (`start_offset` folded into `reader_args[0] = start_id`, `:97,119,125`) | ✓ clean base — the tiled variant the recipe predicts is unaffected |
| `SliceTileTensorArgsProgramFactory` | reader CRTA 0/1/2 = `src_buffer` / `start_buffer` / `end_buffer`, writer RTA 0 = `dst_buffer` (`slice_program_factory_tile_tensor_args.cpp:182-184,151,168`). Start offset computed **on device** from the start tensor (`reader_..._tensor_args.cpp:85-112`) | ✓ clean base |

No Type 1, no Type 3 (`address_offset`), no Type 4 (`ttnn::narrow` / interior-base `MeshBuffer::create`).
Reconciliation against the checked-in triage doc was not possible (doc absent); this is a from-code scan
of **every** address arg in all five factories, not a table lookup.

### TensorAccessor 3rd argument — **RED (Class 3 / Special)** under B/W-sharding, Class 2 otherwise

Only **two** accessors in the whole op pass a 3rd argument, and both belong to `SliceRmProgramFactory`.
The other eleven `TensorAccessor` constructions in the op's kernels pass two arguments and are not this
subject.

| # | Site | Value | Host origin |
|---|---|---|---|
| A | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:38` (`padded_stick_size`, reader RTA 1) | `per_shard_page_size_bytes(input, padded_row_size_bytes)` | `slice_program_factory_rm.cpp:102`, emitted `:108` |
| B | `slice_writer_unary_stick_layout_interleaved_start_id.cpp:32` (`page_size_override`, writer RTA 7) | `per_shard_page_size_bytes(output, unpadded_row_size_bytes)` | `slice_program_factory_rm.cpp:171`, emitted `:181` |

**Question 1 — sharded or interleaved?** Both. `SliceRmProgramFactory` is the catch-all row-major factory
(`slice_device_operation.cpp:279-296`), so the same accessor is interleaved on some configs and sharded on
others. The two specialisations read the argument differently, so the site must be classified per config.

**Question 2 — magnitude.** `per_shard_page_size_bytes` (`data_movement/common/common.cpp:782-793`)
returns three different things:

| Input/output memory layout | Returned value | True `aligned_page_size` | Accessor specialisation | Class |
|---|---|---|---|---|
| Interleaved (DRAM or L1) | `row_bytes` = `padded_shape[-1] * E` = `buffer->page_size()` | `round_up(page_size, alignment)` | interleaved — realigns internally via `InterleavedAddrGen::aligned_page_size = align_power_of_2(page_size, allocator_alignment)` (`dataflow_api_addrgen.h:278-279`) | **2 — inert, drop** |
| HEIGHT_SHARDED | `buffer()->aligned_page_size()` — *exactly* the default | same | sharded — verbatim | **2 — redundant, drop** |
| **BLOCK_SHARDED / WIDTH_SHARDED** | `shard_spec.shape[1] * E` = `buffer->page_size()`, **unaligned** | `round_up(page_size, 16)` (L1) / DRAM alignment | **sharded — verbatim, no realignment net** | **3 / Special — GATE** |

For row-major B/W-sharded tensors the buffer's page **is** the shard row
(`page_config.cpp:111-118`: page width comes from the physical shard width; `tensor_layout.cpp:321-327`:
the physical shard shape is the shard spec's shape unrounded), and the allocator strides pages at
`align(page_size, alignment)` (`buffer.cpp:764`). The sharded `TensorAccessor` uses the passed value
**verbatim** as that stride — `bank_start + bank_base_address + page_mapping.bank_page_offset *
aligned_page_size + offset` (`tensor_accessor.h:307`). So whenever `shard_W * element_size` is not
alignment-aligned, the accessor strides short and mis-addresses.

**Why it is Special and not just a wrong value.** The override is *load-bearing by design*, and both
kernels say so in comments (`..._rm_interleaved_start_id.cpp:36-37`,
`..._stick_layout_...:20-21`): the value is consumed a second time inside the shared helpers
`noc_async_read_sharded` / `noc_async_write_sharded`
(`data_movement/common/kernels/common.hpp:375-409` / `:325-360`), which call
`tensor.get_aligned_page_size()` (`:389` / `:340`) to split one logical row across shards:

```cpp
const uint32_t page_size = tensor.get_aligned_page_size();
uint32_t sharded_src_id  = src_id * pages_per_row + offset / page_size;
uint32_t sharded_offset  = offset % page_size;
uint32_t read_size       = std::min(remaining, page_size - sharded_offset);   // ← wants the PAYLOAD size
```

The split arithmetic wants the page **payload** (`shard_W * E`); the accessor's addressing wants the page
**stride** (`aligned_page_size`). One 3rd argument is being asked to be both. They coincide exactly when
`shard_W * E % alignment == 0` — which is presumably why this has never been caught — and diverge
otherwise. Metal 2.0 removes the override entirely (the binding-token ctors above supply
`aligned_page_size` implicitly and offer no override), so the port cannot preserve today's behaviour on
the B/W-sharded path even if that behaviour were correct.

- **Class 2 (mechanical drop)** for interleaved and HEIGHT_SHARDED configs — both sites.
- **Class 3 → GATE** for BLOCK_SHARDED / WIDTH_SHARDED configs, escalating to **Special** because the
  intent (shard-relative payload paging) is not expressible through the binding model. Latent rather than
  live: it is masked wherever the shard width happens to be alignment-aligned. I could not establish from
  the code whether any shipped/tested config has `shard_W * element_size % 16 != 0`, so per the recipe's
  "possibly-wrong-magnitude value gates" rule the site is **gated conservatively**.
- **Route: ops team**, for both sites. The decision is theirs: confirm the override is redundant for
  every reachable config (→ drops to Class 2 and the gate lifts), or fix the payload-vs-stride
  conflation in the two shared `noc_async_*_sharded` helpers so the accessor can carry the true stride.
  The helper fix is **outside the op** and would touch every data_movement caller of those helpers.

No triage-doc cross-check was possible; this is a fresh classification. See *Questions*.

### CB endpoints — **GATE-free**, all dispositions determined

Device 2.0 is GREEN, so the census ran on intact Device-2.0 idioms — no deferral. Seven CB instances
across the five factories; census per `(CB, config)`, per node:

| Factory | CB | Touchers on a node | Verdict | Port-time resolution |
|---|---|---|---|---|
| `Rm` | `src0_cb_index = 0` (`slice_program_factory_rm.cpp:338-346`) | reader: `reserve_back`/`push_back` + own `get_write_ptr` peek → **1 locked producer**; writer: `wait_front`/`pop_front` + own `get_read_ptr` peek → **1 locked consumer** | **plain 1:1** | none — bind PRODUCER/CONSUMER as-is |
| `RmSharded` | `c_0`, **borrowed from `input.buffer()`** (`:282-291`) | the single reader kernel, `get_write_ptr()` only (`slice_reader_..._rm_sharded.cpp:41`) — no FIFO ops → **1 role-free toucher** | **single-ended / sync-free** | **self-loop** — bind the reader PRODUCER **and** CONSUMER; plus `borrowed_from` the input |
| `RmSharded` | `c_16`, **borrowed from `output.buffer()`** (`:294-303`) | the same single reader kernel: `reserve_back` / `get_write_ptr` / `push_back`, nothing drains (`:40-42,89`) → **1 locked producer** | **single-ended** | **self-loop**; plus `borrowed_from` the output |
| `RmStride` | `in_cb = 0` (`slice_program_factory_rm_stride.cpp:69-77`) | reader (4d or nd) produces; writer (4d or nd) consumes | **plain 1:1** | none |
| `Tile` | cb 0 (`slice_program_factory_tile.cpp:53-60`) | reader produces; writer consumes. Index reaches both kernels via `named_compile_time_args` `dfb_id_in` / `dfb_id_out` (`:139,161`) | **plain 1:1** | none |
| `TileTensorArgs` | `src0_cb_index = 0` (`slice_program_factory_tile_tensor_args.cpp:56-64`) | reader produces; the **eltwise/unary donor** writer consumes | **plain 1:1** | none |
| `TileTensorArgs` | `tensor_cb_index = 1` (`:65-73`) | the reader only, running a full `reserve_back`→`push_back`→`wait_front`→`pop_front` cycle twice as staging for the start/end tensors (`reader_..._tensor_args.cpp:52-83`) → **1 toucher, locked to both roles** | **single-ended (self-loop already in code)** | **self-loop** — bind the reader PRODUCER **and** CONSUMER |

**Nothing here gates**, and the harder shapes are all absent:

- **No dead CB.** Every allocated `buffer_index` is referenced by a bound kernel in every config;
  each index was traced through its CTA / `named_compile_time_args` carrier to a real access.
- **No hidden second writer.** Actively scanned every kernel that touches each CB for a
  `get_write_ptr()` / `fifo_wr_ptr` write by a kernel that is not the CB's FIFO producer, gated by a
  semaphore pair. There are **no semaphores anywhere in this op**, so the coordination mechanism that
  face requires does not exist here.
- **No multiple-readers face and no dual-instance work-split.** No factory pushes the same
  `kernel_source` into two `KernelDescriptor`s; every kernel instance is unique per factory. No CB has
  read sites in 2+ co-resident kernels beyond the ordinary producer/consumer pair.
- **No config-dependent flip.** Each CB's census is the same across that factory's configs; no
  conditional DFB is needed.

One non-obvious `RmSharded` detail the porter should carry forward: `dfb_in.get_write_ptr()` (`:41`) is
used as a **remote** L1 address — it is combined with *another core's* `noc_x`/`noc_y` (`:65,76`) to read
that core's shard of the borrowed input buffer. That is legal precisely because the CB is
`borrowed_from` the input buffer, so the local address is the same on every core the buffer occupies. It
is a peek, not a second endpoint, and the census above already accounts for it.

---

## Port-work summary  *(would mirror the brief — no brief is issued)*

- **Tensor bindings** (per binding, per factory):
  - `Rm` / `input` — **gated upstream** as the Type-2 offset base; *not* a Case 1/2 item. Do not classify
    it as Case 1 or the offset silently vanishes.
  - `Rm` / `output` — **Case 1** (writer RTA 0 `Buffer*` → `TensorAccessor(dst_args, dst_addr, ...)`).
  - `RmSharded` / `input` — **clean** (borrowed-memory DFB read; `borrowed_from` the input buffer).
  - `RmSharded` / `output` — **clean** (borrowed-memory DFB write; `borrowed_from` the output buffer).
  - `RmStride` / `input`, `output` — **Case 1** each.
  - `Tile` / `input`, `output` — **Case 1** each.
  - `TileTensorArgs` / `input`, `start_tensor`, `end_tensor`, `output` — **Case 1** each (four bindings).
  - **No Case 2 anywhere** — no kernel does hand-rolled arithmetic on a tensor base pointer, so no
    `get_bank_base_address` bridge is needed.
  - Delivery mechanism note: except for the `Rm` reader base, every address arrives as a `Buffer*`
    binding pushed into an `RTArgList` / common-arg list. Because each factory defines
    `override_runtime_arguments`, the adapter **bypasses** `resolve_bindings`
    (`mesh_device_operation_adapter.hpp:449-455`), so those `Buffer*` entries only place the initial
    value at cache-miss time — `patch_slice_program_addresses` is what actually re-points them on a hit.
    Both mechanisms must be accounted for when the bindings become typed.
- **TensorParameter relaxation:** `(legality - pending analysis)` on all five factories — blocked, ops team.
- **TensorAccessor 3rd arg:** two sites (`Rm` reader `:38`, `Rm` writer `:32`) — Class 2 drop for
  interleaved / HEIGHT_SHARDED, **blocked** (Class 3/Special) for BLOCK/WIDTH_SHARDED.
- **CB endpoints:** self-loop `(RmSharded c_0, all configs)`, `(RmSharded c_16, all configs)`,
  `(TileTensorArgs c_1, all configs)`; the remaining four CBs are plain 1:1. No dead-CB drop, no
  multi-binding flag, no conditional DFB.
- **`override_runtime_arguments` → `ProgramRunArgs` translation** (the `CustomProgramSpecFactoryConcept`
  work), five methods funnelling into one shared 63-line patcher
  (`slice_program_factory_rm_sharded.cpp:354-416`) that mixes three distinct patching mechanisms:
  `apply_descriptor_runtime_args` on CB-address-only descriptors (`:363-367`), a raw
  `GetRuntimeArgs` slot-0 poke with a "skip if the slot holds 0" heuristic (`:372-381`), and
  `apply_dynamic_runtime_args` (`:390,404,412`). `slice_tile_dynamic_args`
  (`slice_program_factory_tile.cpp:198-281`) and `slice_rm_reader_dynamic_args`
  (`slice_program_factory_rm.cpp:405-433`) each **re-derive the work split** to stay in lockstep with
  `create_descriptor` — duplication the `ProgramRunArgs` form should be able to collapse.
  `<tt-metalium/experimental/program_descriptor_patching.hpp>` is included by all five factory `.cpp`s
  and is self-described as the shim that Metal 2.0 deletes (`:8-19` of that header), so the port removes
  every one of these calls.

---

## Heads-ups  *(would mirror the brief — no brief is issued)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in this op needs the multi-binding
  advanced option; no hidden second writer exists (there are no semaphores to coordinate one). The three
  out-of-window CBs are all one-toucher self-loops, listed in Port-work.
- **Cross-op / shared kernels:**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    — instantiated by `SliceTileTensorArgsProgramFactory`
    (`slice_program_factory_tile_tensor_args.cpp:133`). **A `_metal2` fork already exists beside it** at
    `.../writer_unary_interleaved_start_id_metal2.cpp` — **bind that fork, do not create a second one.**
    The legacy file's header comment names the sunset plan (issue **#52228**) and the consumer list.
  - Broadly shared: ~25 other program factories instantiate the same legacy file (concat, tilize ×5,
    transpose ×2, reshape_on_device, reduction ×4, matmul, embeddings, attn_matmul, nlp_concat_heads ×2,
    gelu_bw, tanh_bw, examples ×2, plus tests and a programming example). That set is a
    **sunset / coordination list, not authorization to convert the kernel in place.**
  - Slice keeps its **own** copy of that kernel for the Tile factory
    (`device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`) — same stem, different file,
    slice-owned, no fork needed. Do not confuse the two; the Tile and TileTensorArgs factories point at
    different files.
  - **`ttnn/cpp/ttnn/operations/experimental/quasar/slice/` exists and is out of bounds.** It contains a
    shortcut pre-port copy of this op (including a `slice_program_factory_rm_stride.cpp` that names the
    same kernels). It is not a precedent, not a naming source, and not evidence that any construct here
    is portable. Do not read it.
- **RTA / CRTA varargs** — six genuine vararg sites; **prefer named args everywhere else**:
  | Kernel | Site | Shape |
  |---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:29-31` — three `num_dims`-length blocks via `get_arg_addr(13)`, `num_dims` from RTA 4 | RTA vararg |
  | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` — four blocks at **runtime-computed** offsets (`get_arg_addr(1 + num_cores_read * 2)`, `* 3`), variable count | RTA vararg (both recognition shapes) |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` — `2 * num_dims` CRTA block via `get_common_arg_addr(1)` | **CRTA** vararg (CTA-bounded still varies per instantiation) |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:23` — `num_dims`-length `id_per_dim` block via `get_arg_addr(2)` | RTA vararg |
  | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26,31,91-92` — `3 * num_dims` CRTA span, one read at the runtime offset `get_common_arg_addr(3 + 2 * num_dims)`, plus the `id_per_dim` RTA block | CRTA + RTA vararg |
  | `reader_multicore_slice_nd.cpp` / `writer_multicore_slice_nd.cpp` | reader `:73-88` (five `tensor_rank`-length blocks, running offset), writer `:73` (one block) — `tensor_rank` is an **RTA** | RTA vararg |
  The two `*_4d.cpp` kernels are the counter-case: a fixed `rt_args_idx++` run over a constant field set
  (`reader_multicore_slice_4d.cpp:52-77`, `writer_multicore_slice_4d.cpp:52-61`) → **name every one of
  those**, they are not varargs.
- **A vararg block is mutated in place by the kernel.** `id_per_dim` is written back into the runtime-arg
  region as the reader walks dimensions — `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:76-78`,
  `reader_unary_unpad_dims_interleaved_start_id.cpp:45-48`,
  `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:124-127`, and the analogous `coords[]`
  copy in `reader_multicore_slice_nd.cpp:103-105`. The arg region is being used as writable scratch that
  must be re-initialised every dispatch. Confirm the vararg mechanism preserves that (and see the
  related latent bug in *Misc anomalies* — do **not** carry today's behaviour forward as intended).
- **`ccl/mesh_partition` is a second consumer of this op's factory API** — host-side, and the recipe's
  coupling buckets do not cover it. `mesh_partition_program_factory.cpp:123-136` calls
  `SliceOp::validate_on_program_cache_miss`, `SliceOp::select_program_factory` and then
  `Factory::create_descriptor` for whichever slice factory is selected; `:149-156` calls
  `ttnn::prim::patch_slice_program_addresses`; and `mesh_partition_device_operation.hpp:47-52` stores a
  `prim::SliceDeviceOperation::program_factory_t` in its own `shared_variables_t`. Changing
  `create_descriptor` → the spec entry point and deleting the patcher **breaks mesh_partition** unless it
  is ported or adapted in the same change. mesh_partition is itself `legacy (MeshWorkload)` with
  `Is able to port? = no` on the sheet, so the two cannot simply be ported together today.
- **`const` vs `constexpr` at two CTA reads** — `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:15-16`
  declares `tile_width` / `tile_height` as `const uint32_t` (not `constexpr`) from
  `get_compile_time_arg_val`. That distinction decides token-form vs member-getter for the port; confirm
  rather than swapping blind.
- **`named_compile_time_args` is already in use** for CB indices on the Tile factory
  (`slice_program_factory_tile.cpp:139,161` → `get_named_compile_time_arg_val("dfb_id_in"/"dfb_id_out")`),
  with a stated fusion-remapping motive. Two of the op's kernels are therefore already part-modernised:
  the port there is a binding-layer change, not an idiom rewrite. The other three factories still pass CB
  indices positionally.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up (function-call escape): `✓ clean.`** Exactly one out-of-directory header is included by
the op's kernels, it is in-family, and every function it exposes to slice takes Device 2.0-native
handles.

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
| `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | same | 5 | ✓ |
| `reader_multicore_slice_4d.cpp` | same | 5 | ✓ |
| `writer_multicore_slice_4d.cpp` | same | 5 | ✓ |
| `reader_multicore_slice_nd.cpp` | same | 5 | ✓ |
| `writer_multicore_slice_nd.cpp` | same | 5 | ✓ |
| `slice_reader_unary_unpad_dims_rm_sharded.cpp` | *(none — `tt_metal/*` only)* | 1 | ✓ |
| `reader_unary_unpad_dims_interleaved_start_id.cpp` | *(none)* | 1 | ✓ |
| `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | *(none)* | 1 | ✓ |
| `writer_unary_interleaved_start_id.cpp` (slice copy) | *(none)* | 1 | ✓ |
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (donor) | *(none)* | 1 | ✓ |

Every other include across all nine kernels resolves under `tt_metal/hw/inc/api/*`
(`dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `core_local_mem.h`, `endpoints.h`,
`tensor/noc_traits.h`) — bucket 1, no concern.

**Per-call detail** (`data_movement/common/kernels/common.hpp`), three functions called:

| Function | Signature shape | Status |
|---|---|---|
| `noc_async_read_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` (`:375`) | **Shape 1** — `TensorAccessor<DSpec>` by value, plus a `Noc` | ✓ excellent — porter constructs `TensorAccessor(tensor::name)` and passes it |
| `noc_async_write_sharded(...)` (`:325`) | **Shape 1** — same | ✓ excellent |
| `tt_memmove<...>(Noc, uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes)` (`:143`) | plain L1 addresses + `Noc`; no CB / semaphore / accessor handle | ✓ no bridge needed |

No `uint32_t sem_id`, no `uint32_t`/`uint64_t` sem address, no `TensorAccessorArgs<N>` parameter, no CTA
offset as NTTP, no old-style addr-gen, no `CircularBuffer&`, no `DataflowBuffer&` parameters. Nothing
starred. The deprecated no-`Noc` overloads of both `noc_async_*_sharded` (`:363`, `:411`) exist in the
donor but slice calls the `Noc`-first form everywhere.

Note for the framework/ops discussion in *TensorAccessor 3rd argument*: these two helpers are the reason
the 3rd-arg override exists, and their internal use of `get_aligned_page_size()` for **both** the page
stride and the per-page payload size is where that gate's root cause sits. Fixing it there is a
data_movement-wide change, not a slice change.

**Borrowed kernel files (file-path instantiation):**

| Kernel file | Owner | Also instantiated by | `_metal2` fork beside it? |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary (cross-family) | **broadly shared** — concat, tilize (×5), transpose (×2), reshape_on_device, reduction (×4, incl. welford), matmul multicore, embeddings fused, attn_matmul, nlp_concat_heads (×2), gelu_bw, tanh_bw, examples (×2), plus `tests/ttnn/.../test_generic_op.{cpp,py}`, `test_situ_glu_sfpu.py` and `tt_metal/programming_examples/matmul/matmul_multi_core` | **Yes** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory. Bind it. |

The other eight kernels are slice-owned and instantiated only by slice. Two further `_metal2` files with
the same stem exist elsewhere and are **not** forks to reuse: one is typecast's own copy
(`copy/typecast/device/kernels/dataflow/`), and two live under `experimental/quasar/**` (out of bounds by
the locational test).

**Host-side coupling (no recipe bucket — see *Recipe notes*):** `ccl/mesh_partition` consumes slice's
factory API directly, detailed under *Heads-ups*. Separately, four ops call slice's pure host-side offset
helpers `get_rm_start_offset` / `get_tiled_start_offset` (`slice_device_operation.hpp:23-25`):
`experimental/padded_slice` (2 factories), `experimental/slice_write` (3 factories),
`experimental/transformer/nlp_kv_cache_load_slice`. Those are scalar arithmetic functions untouched by the
port — recorded for completeness, not a coupling risk.

### Relaxation candidates (FALLIBLE — candidates to verify; the ops team owns the real analysis)

Mined from `compute_program_hash` (`slice_device_operation.cpp:302-354`), which hashes the **full**
input spec, the full output spec, the start-tensor spec when present, all slice params, `sub_core_grids`
and `factory.index()`. Default strict is the safe reading; these are only observations:

- The hash includes `input.memory_config()` and `output_spec.memory_config()` in full, and the factory's
  correctness genuinely depends on the sharded-vs-interleaved distinction (the 3rd-arg finding above and
  the `select_program_factory` height-sharded branch both key on it). So there is **no obvious
  memory-config relaxation candidate** here.
- Both `padded_shape` and `logical_shape` are hashed for input and output. The RM factory's CB sizing,
  chunking and stride math derive from `padded_shape[-1]` and `slice_start[-1]` only
  (`slice_program_factory_rm.cpp:205-303`), and the Tile factory's per-dim args derive from
  `padded_shape` only — a `match_padded_shape`-style relaxation might be viable for the Tile factory, but
  the RM factory also folds `element_size()` (i.e. dtype) into `begins_bytes`, and `sub_core_grids`
  changes the work split. Fallible; the sheet's `(legality - pending analysis)` cell is the real analysis.
- The comment at `:304-307` states the custom hash exists to *strengthen* distribution against
  false cache hits ("weak distribution for small-integer shape sequences ... same pattern as concat fix
  in PR #45144 (issue #47602)"), not to relax anything. If that reading holds, the relaxation column's
  `needs fix, then none` provisional finding is consistent: this is a hash-correctness item, not a
  genuine `TensorSpec` relaxation.

### TTNN factory analysis (sheet-derived facts, with `file:line` evidence)

- **Concept (current):** `descriptor`, all five factories. Each `.hpp` declares
  `static ProgramDescriptor create_descriptor(const SliceParams&, const SliceInputs&, Tensor&)`.
- **Op-owned tensors:** none — no `create_workload_descriptor`, no `WorkloadDescriptor`, no `buffers`
  vector anywhere in the op.
- **MeshWorkload need:** none. `select_program_factory` (`slice_device_operation.cpp:268-300`) picks one
  single-program factory; `slice()` dispatches through `ttnn::device_operation::launch`
  (`:381-384`). *(Note: `ccl/mesh_partition` wraps these same factories in a MeshWorkload of its own —
  that need belongs to mesh_partition, not to slice.)*
- **Custom hash:** `SliceDeviceOperation::compute_program_hash`, `slice_device_operation.cpp:302-354`.
  **Not a gate**; the port leaves it exactly as it is. No backdoor `attribute_values` / `to_hash`
  (sheet `no`, ✓ confirmed — `SliceParams` / `SliceInputs`, `slice_device_operation_types.hpp:13-28`,
  declare neither).
- **`get_dynamic_runtime_args`:** absent. The adapter's `static_assert`
  (`mesh_device_operation_adapter.hpp:509-513`) makes it impossible to combine with the
  `override_runtime_arguments` the op does define, so it cannot quietly return.
- **`override_runtime_arguments`:** present on all five factories → target
  **`CustomProgramSpecFactoryConcept`**. Sites: `slice_program_factory_rm.cpp:435`,
  `slice_program_factory_rm_sharded.cpp:418`, `slice_program_factory_rm_stride.cpp:178`,
  `slice_program_factory_tile.cpp:189`, `slice_program_factory_tile_tensor_args.cpp:195` — each a
  one-line delegation to the shared `patch_slice_program_addresses`
  (`slice_program_factory_rm_sharded.cpp:354-416`).
- **Pybind `create_descriptor`:** `slice_nanobind.cpp:164-176` (`SliceTileProgramFactory`). Removing it
  is a user-visible API change and gets its own entry in the eventual port report. Other risky pybinds in
  the same function (`bind_slice_descriptor`, `:134-163`): `nb::class_` of `SliceParams` with
  read-write access to every field, `nb::class_` of `SliceInputs`, and `nb::class_` of
  `SliceDeviceOperation` exposing `create_output_tensors` / `compute_output_specs`.
- **Target concept:** `CustomProgramSpecFactoryConcept`, no op-owned tensors — matches the sheet's
  `Porting Target` on all five rows.
- **Execution model:** `SPMD` on all five rows; consistent with the single-program `descriptor` shape.

---

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **Latent cache-hit bug: the RM reader's `id_per_dim` block is never re-initialised.** The reader
   increments `id_per_dim[]` in place inside the runtime-arg region
   (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:31,74-82,104-112`), so after a dispatch the
   block holds the walked-to values, not the per-core start values `create_descriptor` wrote
   (`slice_program_factory_rm.cpp:169`). Because `SliceRmProgramFactory` defines
   `override_runtime_arguments`, the adapter uses **neither** `resolve_bindings` **nor**
   `get_dynamic_runtime_args` and the factory owns the whole re-derivation
   (`mesh_device_operation_adapter.hpp:449-455`) — but `slice_rm_reader_dynamic_args`
   (`slice_program_factory_rm.cpp:405-433`) re-emits **only reader arg 0**, and
   `patch_slice_program_addresses` re-emits only writer slot 0 for this factory. Nothing restores
   `id_per_dim`. A second dispatch that hits the cached program should therefore start its dimension
   walk from stale indices and read the wrong sticks. The sibling factories do **not** have this hole:
   `slice_tile_dynamic_args` explicitly re-emits reader slots `2 .. 2+num_dims`
   (`slice_program_factory_tile.cpp:268-271`), covering both Tile and TileTensorArgs, and
   `slice_reader_unary_unpad_dims_rm_sharded.cpp` never mutates its args. Worth a targeted
   repeat-dispatch test on the RM path (rank ≥ 2, ≥ 2 dims walked, two invocations with the same shapes).
2. **Stale comments describe a hook that does not exist.** Five comments name
   `SliceDeviceOperation::get_dynamic_runtime_args` as the mechanism that re-emits the RM reader base —
   `slice_program_factory_rm.cpp:41,42,105,106` and `slice_program_factory_rm.hpp:40`. There is no such
   hook; the mechanism is `override_runtime_arguments` → `slice_rm_reader_dynamic_args` →
   `apply_dynamic_runtime_args`. Adding the hook the comments describe would now trip the adapter's
   `static_assert`. Also at `slice_program_factory_rm_sharded.cpp:278-279` a comment says "the framework
   copies runtime args and patches dynamic CB addresses" on a cache hit — which is the
   `resolve_bindings` path this op opts out of by defining `override_runtime_arguments`. Both readings
   are actively misleading for anyone reasoning about item 1.
3. **Dead CTA in all four `*_multicore_slice_*` kernels.** `compile_time_element_size =
   get_compile_time_arg_val(1)` is declared and never used —
   `reader_multicore_slice_4d.cpp:81`, `writer_multicore_slice_4d.cpp:65`,
   `reader_multicore_slice_nd.cpp:66`, `writer_multicore_slice_nd.cpp:65`. The host passes
   `element_size` as CTA 1 (`slice_program_factory_rm_stride.cpp:79,82`) **and** again as an RTA
   (`:132,142,149,161`); only the RTA is read.
4. **Dead locals.** `output_bytes_per_row`, `output_h`, `output_d`, `output_n` are computed/read and never
   used in `reader_multicore_slice_4d.cpp:56-62,86`. `old_src_tile_id` is assigned and never used in
   `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:115`. `writer_multicore_slice_nd.cpp:73`
   uses `get_arg_addr(rt_args_idx++)` where the post-increment is pointless (no further reads).
5. **The `end_tensor` binding feeds nothing.** `SliceTileTensorArgsProgramFactory` requires an
   `end_tensor` (`slice_program_factory_tile_tensor_args.cpp:27,46`), allocates a `TensorAccessor` for it
   and reads a full tile from it on device
   (`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:45,69-83`), but `end_indices` is
   declared `[[maybe_unused]]` (`:49`) and never read after the copy loop (`:80-82`). The output extent
   comes from host-computed `num_unpadded_tiles_per_dim` instead. A whole tensor binding, accessor,
   staging CB round-trip and NOC read exist to populate a dead array. If that is intended (e.g. reserved
   for a future dynamic-extent path) it deserves a comment; if not, the binding and the second staging
   read can go.
6. **Dead `#ifdef` paths in slice's own copy of the unary writer.**
   `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:29,38` branch on `OUT_SHARDED` and
   `BACKWARDS`. No slice factory sets `defines` on that `KernelDescriptor`
   (`slice_program_factory_tile.cpp:154-162`), so `OUT_SHARDED` is never taken and `BACKWARDS` never
   compiled. Carried over from the eltwise original.
7. **The `patch_slot0` "skip if zero" heuristic is undocumented as a contract.**
   `slice_program_factory_rm_sharded.cpp:370-381` writes the new address into arg slot 0 of every core's
   arg vector *except* where the slot currently holds 0, on the premise that a zero slot marks a core
   `create_descriptor` left zero-filled. That premise holds today (the no-op-core paths write
   `{0u, 0u, 0u}`, e.g. `slice_program_factory_tile.cpp:176`,
   `slice_program_factory_tile_tensor_args.cpp:151`), but it is a value-based sentinel standing in for
   structural information the descriptor already has, applied uniformly across four factories with
   different slot layouts. It is the kind of shim that quietly mis-fires if a factory ever writes a
   non-zero placeholder or a legitimately-zero address.
8. **Two unreferenced kernel files** in `device/kernels/dataflow/`:
   `strided_slice_reader_rm_interleaved_nd.cpp` and `strided_slice_writer_rm_interleaved.cpp`. No factory
   names either (the strided path uses the `*_multicore_slice_{4d,nd}.cpp` pair instead). Likely
   superseded; candidates for deletion.

---

## Per-DeviceOperation / per-factory attribution

One DeviceOperation, so the bundling is per factory. Findings differ materially between them:

| Factory | `Is able to port?` | Gates open | 3rd arg | Offset base ptr | CB dispositions | Bindings |
|---|---|---|---|---|---|---|
| `SliceRmProgramFactory` | no | relaxation · **offset Type 2** · **3rd arg Class 3/S** | 2 sites | **Type 2 GATE** | 1× plain 1:1 | 1 Case 1, 1 gated |
| `SliceRmShardedProgramFactory` | no | relaxation only | none | clean | 2× **self-loop** (both borrowed) | 2 clean |
| `SliceRmStrideProgramFactory` | no | relaxation only | none | clean | 1× plain 1:1 | 2 Case 1 |
| `SliceTileProgramFactory` | no | relaxation only | none | clean | 1× plain 1:1 | 2 Case 1 |
| `SliceTileTensorArgsProgramFactory` | no | relaxation only | none | clean | 1× plain 1:1, 1× **self-loop** | 4 Case 1 |

Shared across all five: `descriptor` → `CustomProgramSpecFactoryConcept`, the custom hash, the
`override_runtime_arguments` / `patch_slice_program_addresses` translation, and the `ccl/mesh_partition`
host-side coupling. If the relaxation gate clears **without** the offset and 3rd-arg gates clearing, the
four non-`Rm` factories become a viable clean subset — that is the most likely next re-audit outcome and
worth planning for.

---

## Questions for the user

1. **Is the `TensorAccessor` 3rd-argument override on the RM path known to anyone?**
   `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:38` and
   `slice_writer_unary_stick_layout_interleaved_start_id.cpp:32` pass a per-shard page size that, on
   BLOCK/WIDTH-sharded row-major tensors, is the page **payload** (`shard_W * element_size`) where the
   sharded accessor uses the value verbatim as the page **stride** (`aligned_page_size`). It is not on
   the readiness sheet (`Known op issues` names only the offset base pointer) and I could not consult
   `2026-07-06_tensor_accessor_3rd_arg_triage.md`. Two things would settle the class: (a) does any
   shipped or tested slice config have `shard_W * element_size % 16 != 0` on a B/W-sharded row-major
   input **or** output? (b) is the payload-vs-stride conflation in
   `data_movement/common/kernels/common.hpp:340,389` a known issue with an owner? If (a) is "no" for
   every reachable config, both sites drop to Class 2 and this gate lifts.
2. **`Pybind descriptor` = `PR` on all five slice rows.** The sheet uses the literal string `PR` where
   other rows carry `yes`/`no`. The code has one pybound `create_descriptor`, on the Tile factory only
   (`slice_nanobind.cpp:167`). Is `PR` "a PR is in flight to remove it", and does it apply to the whole
   op or just the Tile row? Routed to the readiness-sheet owner as a question, not a defect claim —
   `Pybind descriptor` is in the cross-check set, but a non-boolean value is a vocabulary question rather
   than a code/sheet conflict.
3. **Who owns adapting `ccl/mesh_partition`?** It calls all five slice `create_descriptor`s and
   `patch_slice_program_addresses` directly and is itself `legacy (MeshWorkload)` /
   `Is able to port? = no`. The slice port cannot land without either porting mesh_partition in the same
   change or giving it an adapter, and mesh_partition is not portable today. This ordering constraint
   should be settled before the slice port is scheduled, not discovered during it.

---

## Recipe notes

1. **No bucket for a host-side factory-API borrower.** *Out-of-directory coupling* is defined entirely in
   terms of kernels: `#include` escapes (the six donor classes) and file-path kernel instantiation. It has
   nowhere to record that **another op family calls this op's `create_descriptor` and its cache-hit
   patcher from host code** (`ccl/mesh_partition/device/mesh_partition_program_factory.cpp:123-156`,
   `mesh_partition_device_operation.hpp:47-52`). That is arguably the single highest-impact coupling
   finding in this audit — the port changes the exact API mesh_partition consumes — yet the subject's
   report format (roll-up / summary table / per-call detail / borrowed kernel files) has no row for it and
   its gating carve-out (a donor kernel on pre-Device-2.0 idioms) does not apply. I surfaced it under
   *Heads-ups* and *Team-only* by analogy. Suggestion: add a fourth escape type — *host-side factory-API
   consumer* — with an explicit "does the consumer's own readiness allow a bundled port?" question, since
   the answer here (`no`) is a real scheduling constraint.
2. **The 3rd-argument magnitude rule is written for interleaved accessors and misclassifies sharded ones.**
   *Question 2* lists "`buffer->page_size()`, `tt::tile_size` / `get_tile_size(cb)` and
   `aligned_page_size()`" together as **correct magnitude**. But the *Class 2* row admits only
   "`== aligned_page_size`, **or** a correct-magnitude value on an *interleaved* accessor", and the
   load-bearing-subtlety paragraph says sharded uses the value verbatim with no safety net. On a sharded
   accessor `buffer->page_size()` and `aligned_page_size()` differ by the alignment round-up, and that
   difference **does** mis-address — so following Question 2 literally yields Class 2 while following the
   table yields Class 3. I resolved it toward the table (and the conservative-gating rule), but the two
   passages contradict each other. Suggestion: split Question 2's correct-magnitude list by
   specialisation, or state explicitly that on a sharded accessor `page_size()` is *not* interchangeable
   with `aligned_page_size()`.
3. **The taxonomy has no clean slot for "load-bearing override that is also arguably wrong."** The site
   here is simultaneously Special (the intent — shard-relative payload paging — is inexpressible in the
   binding model) and Class 3 (the value looks wrong for the accessor's own addressing). Both route to
   the ops team, so the verdict is unaffected, but the class label had to be reported as "3 / Special"
   rather than picked. A note that the classes are not mutually exclusive, and that reporting both is
   fine when they route identically, would save the next auditor the same hesitation.
4. **The *Red* scoping rule's "which side does it clear on?" test is hard to apply to a relaxation
   verdict.** `(legality - pending analysis)` means the ops team will *analyse* and *possibly* fix. If
   the analysis concludes "no relaxation needed, the hash was merely strong", the cell flips with the op's
   code untouched (→ run the seven subjects); if it concludes "needs fix", the code changes (→ skip). The
   provisional column here says `needs fix, then none`, which I took as op-code side. The rule assumes
   the clearing side is knowable at audit time; for a pending-analysis cell it is knowable only
   probabilistically. Suggestion: name the pending-relaxation case explicitly and say which way to lean
   (running the subjects and disclosing the staleness risk seems cheaper than a second full pass).
5. **Provenance is unpinnable when the recipe is consumed out-of-tree.** The `git log` command targets
   `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`, which does not exist in this checkout;
   the recipe I followed lives at an absolute path outside the repo. The fallback instruction ("record
   that instead") works, but a version stamp inside the recipe document itself would survive being copied
   out of the doc branch — which appears to be the normal way these audits are actually run.
