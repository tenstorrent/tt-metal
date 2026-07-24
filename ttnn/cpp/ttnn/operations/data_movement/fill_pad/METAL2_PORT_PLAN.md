# Port Plan — fill_pad (`fill_implicit_tile_padding`)

Port plan for `data_movement/fill_pad`, ported from the `ProgramDescriptor` factory concept
(`create_descriptor`) to Metal 2.0 (`MetalV2FactoryConcept`, `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Both factories under one `FillPadDeviceOperation` are ported together (they share the compute
kernel `fill_pad_compute.cpp` and the mask helpers `fill_pad_dataflow_common.hpp`), per the brief.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories expose `create_descriptor()` returning `ProgramDescriptor`).
- Variants: two program factories in one device-op variant —
  - `FillPadProgramFactory` — DRAM interleaved (+ rare DRAM-sharded).
  - `FillPadL1ShardedProgramFactory` — all L1-sharded (HEIGHT / WIDTH / BLOCK).
- Custom `compute_program_hash`: none — already the default reflection-based hash (audit confirmed, grep clean).
- **In-place op:** `create_output_tensors` returns the input tensor; `compute_output_specs` returns `input.tensor_spec()`. There is a *single* io tensor (input == output); reader reads it and writer writes it back.

*(Target Metal 2.0 concept `MetalV2FactoryConcept`, chosen in the audit — carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### DRAM factory (`FillPadProgramFactory`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | dataflow/fill_pad_reader.cpp | all_cores | [0]W_tiles [1]H_tiles [2]N_slices(unused) [3]has_right_pad [4]has_bottom_pad [5]W_mod32(unused) [6]H_mod32(unused) [7]elem_size(dead) [8]fill_bits(unused) [9]cb_data_in_idx=0 [10+]TensorAccessorArgs | [0]buf_addr [1]start_right [2]num_right [3]start_bottom [4]num_bottom [5]start_corner [6]num_corner | MASK_ELEM_UINT, MASK_VALUE, FILL_PAD_DATA_FMT, FILL_PAD_FILL_DATA_FMT?, FILL_PAD_FILL_FN, FILL_PAD_FILL_ARG | ReaderConfigDescriptor{} |
| writer | dataflow/fill_pad_writer.cpp | all_cores | [0]W_tiles [1]H_tiles [2]N_slices(unused) [3]has_right_pad [4]has_bottom_pad [5]W_mod32 [6]H_mod32 [7]cb_right_mask=1 [8]cb_bot_mask=2 [9]cb_data_out=16 [10+]TensorAccessorArgs | same 7 as reader | same defines | WriterConfigDescriptor{} |
| compute | compute/fill_pad_compute.cpp | all_cores | [0]W_tiles(dead) [1]H_tiles(dead) [2]has_right_pad [3]has_bottom_pad [4]elem_size(dead) [5]fill_bits [6]cb_data_in=0 [7]cb_right_mask=1 [8]cb_bot_mask=2 [9]cb_data_out=16 | [0]num_right [1]num_bottom [2]num_corner | same defines | ComputeConfigDescriptor{.fp32_dest_acc_en=need_fp32_dest_acc, .unpack_to_dest_mode=…} |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 cb_data_in | tile_bytes*2 | all_cores | cb_data_format | tile_bytes | (unset) |
| c_1 cb_right_mask (if has_right_pad) | tile_bytes | all_cores | cb_data_format | tile_bytes | (unset) |
| c_2 cb_bot_mask (if has_bottom_pad) | tile_bytes | all_cores | cb_data_format | tile_bytes | (unset) |
| c_16 cb_data_out | tile_bytes*2 | all_cores | cb_data_format | tile_bytes | (unset) |

#### Semaphores
none — op uses no semaphores.

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(src_args, buf_addr, tile_bytes)` (`fill_pad_reader.cpp:87`) | input (Case 1) | slot 0 `tens_buffer` |
| writer `TensorAccessor(dst_args, buf_addr, tile_bytes)` (`fill_pad_writer.cpp:81`) | input (Case 1) | slot 0 `tens_buffer` |

Both carry a redundant 3rd (page-size) arg `tile_bytes = get_tile_size(...)` → **drop** (Class 2, per brief).

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, total_work)` where `total_work = T_right + T_bottom + T_corner` (unified border-tile index space).
- num_cores / all_cores / core_group_1 / core_group_2 / counts per group — the split produces per-core work ranges; per-core (start,num) phase triples are derived and passed as RTAs. **Single kernel binary per role** (one reader/writer/compute KernelDescriptor over `all_cores`); the per-group counts differ only in RTA values, NOT CTAs → no CTA work-split multiplicity.

### L1-sharded factory (`FillPadL1ShardedProgramFactory`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| sharded_reader (×2, per rp_idx) | dataflow/fill_pad_sharded_reader.cpp | rw_ranges[rp] | [0]W_tiles(shard) [1]has_right_pad(=rp) [2]elem_size(dead) [3]cb_data_in=0 | [0]shard_l1_base [1]shard_H_tiles [2]has_bottom_pad_core [3]num_work [4]local_right_col | ReaderConfigDescriptor{} |
| sharded_writer (×2, per rp_idx) | dataflow/fill_pad_sharded_writer.cpp | rw_ranges[rp] | [0]W_tiles [1]has_right_pad(=rp) [2]W_mod32 [3]H_mod32 [4]cb_right_mask=1 [5]cb_bot_mask=2 [6]cb_data_out=16 | same 5 as reader | WriterConfigDescriptor{} |
| compute (×N, per ComputeKey) | compute/fill_pad_compute.cpp | compute_ranges[key] | [0]W_tiles(=effective_W) [1]H_tiles(=key.H) [2]has_right_pad [3]has_bottom_pad [4]elem_size(dead) [5]fill_bits [6..9]cb ids | [0]num_right [1]num_bottom [2]num_corner | ComputeConfigDescriptor{…} |

- `ComputeKey = (has_right_pad, has_bottom_pad, H, effective_W)`; N distinct keys among active cores.
- `has_bottom_pad_core` is a **per-core RTA** on the legacy reader/writer (runtime `if`), but a **per-ComputeKey CTA** on compute (compile-time `if constexpr`). This asymmetry drives a port decision (see Planned Spec Shape → sharded writer split).

#### CBs
Same 4 indices (c_0/c_1/c_2/c_16), created over `all_active_set`, same formats as DRAM.

#### Semaphores / Tensor accessors
- Semaphores: none.
- Tensor accessors: **none** — the sharded reader/writer read/write local L1 via `UnicastEndpoint{}` with a raw base address (`shard_l1_base = get_arg_val<uint32_t>(0)`, RTA slot 0 = `tens_buffer`). This is the **Case 2** (raw pointer) binding.

#### Work split
- Per-core, no balancing: each active shard core processes its own border tiles. `active` = shard cores intersecting the valid tile range; `rw_ranges[rp]` groups by has_right_pad; `compute_ranges[key]` groups by ComputeKey.

### Cross-op kernels
none — all 5 kernel sources are owned by the op directory. `fill_pad_compute.cpp` and `fill_pad_dataflow_common.hpp` are shared *in-op* by both factories (port once, keep both consistent).

### Flags
- Dead / unused CTAs (audit "Misc anomalies", team-only, non-gating): reader `elem_size`[7], `N_slices`[2], `fill_bits`[8]; compute `elem_size`[4] (and `W_tiles`/`H_tiles` appear unused in the compute body); sharded reader `elem_size`[2]. **Preserved as named args in the port** (removing them is an ops-team cleanup, not port scope) — see report.
- The compute kernel's `CB_DATA_IN`/`CB_DATA_PADDING`/`CB_MASK`/`CB_OUT` constants inside `process_masked_tile`/`process_corner_tile` are **DST register indices**, not CB ids (despite the name). Left unchanged.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none (nothing to delete).
- **Pybind**: no `create_descriptor` binding (nanobind binds only the host function `fill_implicit_tile_padding`) → no pybind cleanup.
- **Implementation notes**:
  - Both factory structs get `static ProgramArtifacts create_program_artifacts(attrs, tensor_args, tensor_return_value)` replacing `create_descriptor(...)`.
  - Device-op class (`select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, the `fill_pad(...)` launcher) is unchanged.
  - Single io tensor (in-place): bind `tensor_args.input.mesh_tensor()` to the one `TensorParameter INPUT`; both reader and writer add a `TensorBinding` to it.

## Planned Spec Shape

Default 1:1 with legacy, except the sharded writer split (justified below).

### DataflowBufferSpecs (both factories)
One per legacy CB index. `entry_size = tile_bytes`, `data_format_metadata = cb_data_format`, `tile_format_metadata` unset (legacy `.tile` unset → default 32×32):
- `DATA_IN` ("data_in", c_0): num_entries=2. Reader PRODUCER, compute CONSUMER.
- `RIGHT_MASK` ("right_mask", c_1): num_entries=1. **Conditional** (has_right_pad). Writer PRODUCER, compute CONSUMER.
- `BOT_MASK` ("bot_mask", c_2): num_entries=1. **Conditional** (has_bottom_pad / has_bottom_pad_core). Writer PRODUCER, compute CONSUMER.
- `DATA_OUT` ("data_out", c_16): num_entries=2. Compute PRODUCER, writer CONSUMER.

All plain 1P+1C FIFOs (audit: no self-loop, no multi-binding, no dead CB). Placement derived from bindings.

### TensorParameters
- `INPUT` ("input"), `spec = input.tensor_spec()`. Bound by reader (accessor "input"/"src") and writer (accessor "input"/"dst") — one param, two `TensorBinding`s (in-place op → same tensor read and written).
- DRAM factory: Case 1 — kernels build `TensorAccessor(tensor::input)`.
- Sharded factory: Case 2 — kernels build `TensorAccessor(tensor::input)` and pull the raw shard base via `get_bank_base_address()`; the `UnicastEndpoint` arithmetic is left unchanged.

### SemaphoreSpecs
none.

### KernelSpecs / WorkUnitSpecs

**DRAM factory** — single instance each over `all_cores`:
- KernelSpecs: reader, writer, compute (1 each). One `WorkUnitSpec` over `all_cores` (or the used-core subset `grid_to_cores(num_cores,…)`).
- `has_right_pad` / `has_bottom_pad` are *global* compile-time flags.

**Sharded factory**:
- reader KernelSpecs: **one per rp_idx** (≤2), matching legacy grouping. `has_right_pad` is a named CTA; `has_bottom_pad_core` stays a runtime RTA (reader binds no conditional DFB — no split needed).
- writer KernelSpecs: **one per (rp_idx, has_bottom_pad_core)** (≤4) — see split rationale below.
- compute KernelSpecs: **one per ComputeKey** (N), matching legacy grouping.
- WorkUnitSpecs: **one per active ComputeKey group**, containing `{reader[rp], writer[rp,hbp], compute[key]}` over that group's node ranges. reader[rp] / writer[rp,hbp] appear in multiple WUs across their (disjoint) node sets — legal per the DFB multi-KernelSpec-per-endpoint invariant (`dataflow_buffer_spec.hpp`).

### Sharded writer split — rationale (new structural decision)
The mask DFBs are **conditionally bound**. `RIGHT_MASK` is gated on `has_right_pad` (= the reader/writer `rp_idx` key and the compute `ComputeKey.has_right_pad`), so its producer (writer) and consumer (compute) already agree per node. `BOT_MASK`, however, is gated on `has_bottom_pad_core`, which is **per-ComputeKey compile-time on the consumer (compute)** but **per-core runtime on the legacy writer** (one writer KernelSpec per rp spans cores with mixed `has_bottom_pad_core`). With derived placement + the per-node "≥1 PRODUCER and ≥1 CONSUMER" rule, a single writer spec binding `BOT_MASK` would put a PRODUCER on cores whose compute spec (`has_bottom_pad=0`) binds no CONSUMER → validator failure. **Fix:** split the sharded writer by `has_bottom_pad_core` so the bot-mask binding is per-node consistent with the compute consumer. This promotes `has_bottom_pad_core` from a writer RTA to a per-writer-spec compile flag (a `#define`), which is *promotion* (safe; enables the same compile-time gating compute already uses), not the forbidden CTA→RTA demotion. The reader is **not** split (binds no conditional DFB).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| DRAM: 1 reader / 1 writer / 1 compute | 1 / 1 / 1 | 1 WU over used cores | DATA_IN(reader P, compute C); DATA_OUT(compute P, writer C); RIGHT_MASK/BOT_MASK(writer P, compute C) |
| Sharded reader ×rp | reader ×rp | in each WU of that rp | DATA_IN (P) |
| Sharded writer ×rp (legacy) | writer ×(rp,hbp) **[split, see rationale]** | in each WU of that (rp,hbp) | DATA_OUT (C); RIGHT_MASK (P, rp=1); BOT_MASK (P, hbp=1) |
| Sharded compute ×ComputeKey | compute ×ComputeKey | one WU each | DATA_IN (C); DATA_OUT (P); RIGHT_MASK (C, rp=1); BOT_MASK (C, hbp=1) |

No legacy multi-`KernelDescriptor` **CTA** work-split of a single role over one grid (the DRAM per-core variation is RTA-only). The sharded per-config specs are legacy-preserved (compute) or newly split (writer, above).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| DRAM reader CTA[9] `cb_data_in_idx` | magic CB index | `DFBBinding` DATA_IN (PRODUCER) |
| DRAM reader CTA[10+] `TensorAccessorArgs<10>()` | accessor plumbing + RTA[0] `buf_addr` | `TensorBinding` INPUT → `TensorAccessor(tensor::input)` |
| DRAM reader `TensorAccessor(...,tile_bytes)` 3rd arg | redundant page size | dropped (Class 2) |
| DRAM writer CTA[7,8,9] `cb_right_mask/bot_mask/data_out` | magic CB indices | `DFBBinding`s RIGHT_MASK / BOT_MASK / DATA_OUT |
| DRAM writer CTA[10+] + RTA[0] | accessor plumbing + buf_addr | `TensorBinding` INPUT |
| DRAM writer `TensorAccessor(...,tile_bytes)` 3rd arg | redundant page size | dropped (Class 2) |
| DRAM compute CTA[6..9] cb ids | magic CB indices | `DFBBinding`s DATA_IN / RIGHT_MASK / BOT_MASK / DATA_OUT |
| DRAM compute CTA[2] has_right_pad, CTA[3] has_bottom_pad | CTA gate over conditional CB | `#define` HAS_RIGHT_PAD / HAS_BOTTOM_PAD (compiler_options.defines) |
| Sharded reader CTA[3] cb id; RTA[0] buf_addr | magic CB idx; raw base RTA | `DFBBinding` DATA_IN; `TensorBinding` INPUT (Case 2, `get_bank_base_address()`) |
| Sharded writer CTA[4,5,6] cb ids; CTA[1] has_right_pad; RTA[0] buf_addr; RTA[2] has_bottom_pad_core | magic CB idxs; CTA gate; raw base; runtime gate | `DFBBinding`s; `#define` HAS_RIGHT_PAD; `TensorBinding` INPUT (Case 2); `#define` HAS_BOTTOM_PAD |
| Sharded compute CTA[6..9] cb ids; CTA[2,3] gates | magic CB idxs; CTA gates | `DFBBinding`s; `#define` HAS_RIGHT_PAD / HAS_BOTTOM_PAD |
| All positional CTAs / RTAs | `get_compile_time_arg_val(N)` / `get_arg_val<uint32_t>(N)` | named `get_arg(args::name)` |

All surviving scalar CTAs/RTAs (incl. the dead ones per Flags) become **named** args (fixed-count distinct fields — no varargs).

## Applied Patterns
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): RIGHT_MASK / BOT_MASK bound conditionally; host emits `HAS_RIGHT_PAD` / `HAS_BOTTOM_PAD` defines; kernels `#ifdef`-gate the `dfb::right_mask`/`dfb::bot_mask` construction and every reference. Promotes the legacy `if constexpr(has_*_pad)` CTA gates (and the sharded writer's runtime `if(has_bottom_pad_core)`) to `#ifdef`.
- [Multi-variant / per-config KernelSpecs](.../port_patterns.md): sharded reader/writer/compute keyed by rp / (rp,hbp) / ComputeKey, each with its own WorkUnitSpec.
- [Pass DFB handles directly to LLKs](.../port_patterns.md): compute passes DFB ids to `copy_tile` / `pack_tile` via the existing `DataflowBuffer::get_id()` on the constructed objects (Device-2.0 method, kept — objects are genuine, not temporaries).
- Case 2 (raw pointer) binding: sharded reader/writer pull the shard base via `TensorAccessor(tensor::input).get_bank_base_address()`; raw `UnicastEndpoint` arithmetic unchanged.

## Deferred / Flagged
- **Derived placement narrows mask-CB footprint**: legacy allocated cb_right_mask/cb_bot_mask over `all_active_set` (all active shard cores) even where unused; Metal 2.0 derived placement puts them only on rp=1 / hbp=1 nodes. Zero functional change (unused CBs on other cores were never touched); a natural consequence of the binding model, not a deliberate refactor. Noted for the report.
- No new blocking findings during planning. `get_bank_base_address` confirmed present on the kernel-side `TensorAccessor` (`tt_metal/hw/inc/api/tensor/tensor_accessor.h`).
