# Port Plan — experimental/plusone

Port plan for `ttnn/cpp/ttnn/operations/experimental/plusone`, ported from the
`descriptor` (`ProgramDescriptor`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `PlusOneProgramFactory::create_descriptor()` returns `ProgramDescriptor` (`device/plusone_program_factory.cpp:22`).
- Variants: single (`program_factory_t = std::variant<PlusOneProgramFactory>`).
- Custom `compute_program_hash`: none — device op defines only `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors` (`device/plusone_device_operation.cpp`). Default reflection hash.

*(Target Metal 2.0 concept chosen by audit: `MetalV2FactoryConcept`. Carried forward below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_plusone_interleaved.cpp` (op-owned) | `all_cores` (default `{0,0}`; or `sub_core_grids`) | 0:`src0_cb_index`, 1:`src_is_dram`, 2:`aligned_input_page_size`, 3:`W`, 4:`H`, 5:`skip_negative_entries`, then `TensorAccessorArgs(*src_buffer)` (fixed-count) | none | slot 0: `src_buffer` (Buffer*, clean base — framework-patched form) | none | none | `ReaderConfigDescriptor{}` (reader default: RISCV_1 / NOC_0 / DEDICATED) |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| `c_0` | `aligned_input_page_size` | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `aligned_input_page_size` | none | `input.is_sharded() ? src_buffer : nullptr` |

`c_0` is used by the reader purely as an **address source** — `cb_in0.get_write_ptr()` (`reader...cpp:31`) → raw `stick[]` pointer. **No FIFO ops** (no reserve/push/wait/pop). One toucher, sync-free.

### Semaphores
none.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessor(s0_args, src_addr)` (`reader...cpp:26`) — used only when `src0_is_dram` (`reader...cpp:36,52`) | input (io) | slot 0 (`src_buffer`) |

### Work split
n/a — single work unit over `all_cores`. Each node runs the identical kernel over the full W×H (no per-node slicing; legacy passes the same `src_buffer` RTA and same CTAs to every core). Preserved verbatim.

### Cross-op kernels
none — the sole kernel is op-owned and file-path-instantiated from `device/kernels/`.

### Flags
- **Interleaved-in-L1 anomaly (preserve exactly):** when input is neither DRAM nor sharded, `c_0` is plain scratch and `src0_is_dram` is false, so the kernel increments **uninitialized scratch** and never touches the input. Pre-existing behavior; zero functional change (audit Misc anomalies). Port must preserve.
- `src_addr` RTA is dead on the non-DRAM path (constructed into `s0`, never used). Dropped entirely by the port (→ `TensorBinding`).

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** The op's single kernel serves three host-known configs from one source. All three are decided at factory (host) time — there is **no runtime kernel-source selection**, so this is a single `KernelSpec` whose bindings/defines vary by config:
  - **DRAM (interleaved):** accessor path active. `TensorParameter INPUT` + reader `TensorBinding`; DFB `c_0` = plain scratch; define `SRC0_IS_DRAM`.
  - **Sharded (L1):** accessor path skipped. `TensorParameter INPUT`; DFB `c_0` `borrowed_from = INPUT` (backing input shard); **no** `TensorBinding` (borrowed_from is the reference that satisfies the TensorParameter-needs-a-binding validator rule, as in `quasar/fold`); no `SRC0_IS_DRAM`.
  - **L1 interleaved (anomaly):** accessor path skipped. No `TensorParameter`; DFB `c_0` = plain scratch; no `SRC0_IS_DRAM`. Kernel increments uninitialized scratch (preserved).

## Planned Spec Shape
- **KernelSpecs:** 1 — `reader` (DM). `hw_config = create_reader_datamovement_config(arch)` (reader default). `compiler_options.defines` carries `SRC0_IS_DRAM` iff input is DRAM. Conditional `tensor_bindings` (INPUT) iff DRAM.
- **DataflowBufferSpecs:** 1 — `IN0` (`c_0`). `entry_size = aligned_input_page_size`, `num_entries = 1`, `data_format_metadata = input_cb_data_format`. `borrowed_from = INPUT` iff sharded. **Self-loop** (reader bound PRODUCER **and** CONSUMER, shared accessor name `in0`).
- **SemaphoreSpecs:** none.
- **TensorParameters:** `INPUT` present iff (DRAM || sharded); `spec = input.tensor_spec()`. (Absent in the L1-interleaved anomaly path.)
- **WorkUnitSpecs:** 1 — `{reader}` over `all_cores`.

## Preserved Multiplicity
none — no work-split multiplicity in legacy (single kernel descriptor).

## Dropped Plumbing
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `reader_desc.emplace_runtime_args(core, {src_buffer})` (`plusone_program_factory.cpp:85`); kernel `src_addr = get_arg_val<uint32_t>(0)` (`reader...cpp:16`) | Buffer* address RTA | `TensorParameter INPUT` + `TensorBinding` (DRAM path) / `borrowed_from` (sharded path); auto-injected base address |
| CTA slot 0 `src0_cb_index` (`plusone_program_factory.cpp:68`); kernel `get_compile_time_arg_val(0)` (`reader...cpp:18`) | magic CB index | `DFBBinding` (`dfb::in0`) |
| CTA slot 1 `src_is_dram` (`plusone_program_factory.cpp:69`); kernel `get_compile_time_arg_val(1)` (`reader...cpp:19`) | bool CTA selecting the accessor path | `KernelSpec::compiler_options.defines` `SRC0_IS_DRAM` (conditional-binding pattern, rule 6) |
| `TensorAccessorArgs(*src_buffer).append_to(...)` (`plusone_program_factory.cpp:70`); kernel `TensorAccessorArgs<6>()` (`reader...cpp:25`) | accessor CTA plumbing | binding mechanism (`TensorAccessor(tensor::input)`) |
| CTA slots 2-5 (positional) | positional CTAs | named CTAs `stick_size`, `W`, `H`, `skip_negative_entries` |

## Applied Patterns
- [Sync-free CB → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): `c_0`/`IN0` — one toucher (reader raw-peeks `get_write_ptr`), no FIFO ops → reader bound PRODUCER + CONSUMER. DM self-loop, legal on Gen1.
- [Borrowed-memory DFB](../shared/migration_guide.md#dataflowbufferspec): `IN0.borrowed_from = INPUT` on the sharded config (legacy `.buffer = src_buffer`).
- [Conditional / optional binding](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings): the `TensorBinding` (INPUT) and the accessor code are gated by `SRC0_IS_DRAM`; the condition moves from a CTA to a host-emitted `#define`, and the kernel `#ifdef`-gates the `TensorAccessor` construction and both NoC transfer blocks.

## Deferred / Flagged
- New findings during planning: none beyond the audit's already-noted L1-interleaved anomaly (preserved, not fixed).
