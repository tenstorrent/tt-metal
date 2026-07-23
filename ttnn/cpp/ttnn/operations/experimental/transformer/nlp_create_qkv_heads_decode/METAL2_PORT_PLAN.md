# Port Plan — `experimental/transformer/nlp_create_qkv_heads_decode`

Port plan for `nlp_create_qkv_heads_decode`, ported from the legacy `ProgramDescriptor` (`descriptor`)
concept to Metal 2.0 (`MetalV2FactoryConcept`). One `DeviceOperation`, three factories selected by input
layout (`device/nlp_create_qkv_heads_decode_device_operation.cpp:12`): **Interleaved** (non-sharded),
**Sharded** (width-sharded), **ShardedSubcoregrid** (width-sharded, sub-core-grids). Pure data-movement op
(no compute kernels).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (each factory: `create_descriptor(...)` → `ProgramDescriptor`).
- Variants: three factories under one device-op `program_factory_t` variant.
- Custom `compute_program_hash`: none (default reflection hash).

### Kernels (all owned by the op, Device 2.0 native)
- Interleaved: `reader_interleaved_tm_tile_layout_...decode.cpp`, instantiated reader (phase 1) + writer
  (phase 2) over `q_cores`. CTAs 0..12 (element_size, sub_tile_line_bytes, q/k/v cb, head_size,
  num_q_heads, num_kv_heads, head_tiles, phase, use_aligned_path, dram_align, scratch cb) + TensorAccessorArgs.
  RTAs: in_tile_offset_by_batch, q_start_addr(=in_buffer).
- Sharded: `reader_tm_tile_layout_...decode.cpp`. q reader+writer over q_cores; +k reader+writer over
  k_cores when `!overlap`. CTAs incl. num_x/num_y, process_qv/process_k, use_batch_offset, index_stick_size,
  batch cb. RTAs: q_start_addr(Case 2), batch_offset_addr(Case 1), index_in_cores, + variable noc_x/noc_y.
- Subcoregrid: `..._on_subcoregrids.cpp`. Same but one `in_num_cores` count and single-index coords;
  writer switches batch cb to c_14 correctly.

### CBs
- c_16/c_17/c_18: q/k/v output (borrowed `output[N].buffer()`), all factories.
- c_0/c_1: interleaved reader/writer scratch (aligned path only).
- c_15/c_14: reader/writer batch-offset scratch (sharded/subcoregrid, batch_offset present).
No GlobalCircularBuffer, no aliased CBs, no `address_offset`. No semaphores.

### Work split
No `split_work_to_cores`. Per-core work driven by output shard grids; the reader/writer dual instance is a
tile-phase split over the same grid (not disjoint nodes).

### Cross-op kernels
- Interleaved kernel calls `tt::data_movement::common::tt_memmove(noc, …)` (shared util, Device 2.0
  native, plain uint32_t L1 addresses) — not modified.

## TTNN ProgramFactory
- **Concept (from audit):** `MetalV2FactoryConcept` (all three).
- **Custom `compute_program_hash`:** none.
- **Device-op-class edits:** none forced (nanobind uses `bind_function`, no `create_descriptor` pybind).

## Planned Spec Shape (as realized)

- **Interleaved:** reader/writer KernelSpecs over q_cores; q/k/v **borrowed output DFBs** (1P+1C);
  conditional scratch DFBs (self-loop, aligned path); input TensorParameter (Case 1). Single WorkUnitSpec.
- **Sharded / Subcoregrid:** q reader/writer (+ k reader/writer when `!overlap`); q/k/v outputs as
  **TensorParameters** (Case-2 `get_bank_base_address`, **not** borrowed DFBs — see Deferred/Flagged);
  input TensorParameter (Case 2); conditional batch_offset TensorParameter (Case 1) + self-loop batch
  scratch DFBs. One WorkUnitSpec (overlap) or two (`!overlap`, disjoint q/k grids).

## Dropped Plumbing

| legacy | Metal 2.0 replacement |
|---|---|
| output cb magic indices / borrowed `.buffer()` RTA | interleaved: `DFBBinding` borrowed_from; sharded/subcoregrid: `TensorParameter` + `get_bank_base_address` |
| input buffer-address RTA + `TensorAccessorArgs` | `TensorParameter` (Case 1 accessor, or Case 2 `get_bank_base_address`) |
| batch-offset address RTA + `TensorAccessorArgs` | `TensorParameter` (Case 1), conditional |
| positional noc_x/noc_y RTA block (data-indexed) | runtime varargs |
| `index_in_cores` positional RTA | named RTA |
| positional CTAs | named CTAs; `use_aligned_path`/`process_qv`/`process_k`/`use_batch_offset` → `#define` gates |

## Applied Patterns
- Two-toucher DFB → 1P+1C (interleaved outputs).
- Sync-free/single-ended → self-loop DFB (interleaved scratch; batch-offset scratch).
- Conditional/optional DFB & tensor bindings via `#ifdef` (USE_ALIGNED_PATH / USE_BATCH_OFFSET / PROCESS_QV / PROCESS_K).
- Case-2 raw base via `get_bank_base_address` (sharded/subcoregrid input; and — as the framework-bug
  workaround — the outputs).
- Multi-variant factory (three factories in the device-op variant).
- Unity-build hygiene (per-factory prefixed anon-namespace constants).

## Deferred / Flagged (outcome)

- **All three factories PORTED** to `MetalV2FactoryConcept`; suite passes 205/39 skipped/0 failed.
- **Framework bug worked around (Sharded + Subcoregrid).** The audit's "clean borrowed-DFB output"
  disposition is blocked by a Metal 2.0 framework bug: borrowed DFBs get a corrupted device-side base in
  multi-work-unit programs, which the `!overlap` layout requires. Outputs re-expressed as Case-2
  `TensorParameter`s (`get_bank_base_address`). Interleaved keeps borrowed DFBs (single WU). Revert to
  borrowed DFBs once the framework bug is fixed. Full detail + caveats (Gen2, modeling inconsistency) in
  `METAL2_PORT_REPORT.md`.
- **Sharded batch-offset writer-CB one-line fix applied** (audit Question #1 — confirmed a bug).
