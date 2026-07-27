# Metal 2.0 Audit Findings — `data_movement/sharded_partial/sharded_to_interleaved_partial`

- **`ShardedToInterleavedPartialDeviceOperation`**
  - `ShardedToInterleavedPartialProgramFactory` (`device/sharded_to_interleaved_partial_program_factory.cpp`)

Single DeviceOperation, single program factory (`create_descriptor` returning a `ProgramDescriptor`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ShardedToInterleavedPartialDeviceOperation` → `ShardedToInterleavedPartialProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all referenced kernels object-based (`Noc` / `DataflowBuffer` / `TensorAccessor`); no DM free-function holdovers |
| *Prereqs* — Cross-op escapes | Ok — file-path kernel borrows only (no function-call escapes); all donor kernels Device 2.0 compliant |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at constexpr offsets; `tensor_args_t` is a fixed 2-tensor struct |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (cross-checked: no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (cross-checked: no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (cross-checked: nanobind binds only the free function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none — clean base delivered as `Buffer*`; no host-side fold |
| *Port work* — Tensor bindings (per binding) | input: clean (borrowed-memory DFB) · output/cache: Case 1 (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | all legal 1:1 (both convert-df configs) |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 (all four referenced kernels are object-based and free of data-movement holdovers), Feature compatibility (no Appendix A feature present), TTNN factory concept (`Is able to port? = yes`, `descriptor`, cross-check clean), Offset base pointers (no host-folded offset), TensorAccessor 3rd argument (no site). The port is a clean single-`descriptor`-factory port to `MetalV2FactoryConcept`. Port work is confined to the two tensor bindings (one borrowed-memory input DFB, one Case-1 output/accessor binding).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet row (op `data_movement/sharded_partial/sharded_to_interleaved_partial`, `ShardedToInterleavedPartialDeviceOperation` / `ShardedToInterleavedPartialProgramFactory`): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`. Cross-check against code confirms every cheaply-checkable column: factory exposes `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor` (`descriptor`); grep of the op tree finds no `compute_program_hash`, no `get_dynamic_runtime_args`, no `override_runtime_arguments`; the nanobind file binds only `&ttnn::sharded_to_interleaved_partial` (no `create_descriptor` pybind). No cross-column invariant violated. Sheet and code agree.
- **Device 2.0 (every kernel used):** GREEN. Four kernels are file-path-instantiated by the factory; the op owns none of them:

  | Kernel (role) | Path | Device 2.0 status |
  |---|---|---|
  | reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | ✓ `DataflowBuffer dfb; dfb.push_back()` — object-based, no NoC |
  | writer (TILE, live path) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | ✓ `Noc`, `DataflowBuffer`, `TensorAccessor`; `get_tile_size(cb_id_out)` is a **sanctioned** free function |
  | writer (RM, validate-blocked path) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | ✓ `Noc`, `DataflowBuffer`, `TensorAccessor` |
  | compute (only when `convert_df`) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | ✓ standard compute API (`cb_wait_front`/`cb_reserve_back`/`copy_tile`/`pack_tile`/`tile_regs_*`) — no data-movement idioms to migrate |

  Grep for `get_read_ptr(`/`get_write_ptr(`/`noc_async_*`/`InterleavedAddrGen`/`ShardedAddrGen`/`get_semaphore`/`get_noc_addr_from_bank` across the three DM kernels: **none**. See Recipe notes for the one interpretive call (compute-kernel CB-sync free functions).

- **Feature compatibility:** every Appendix A entry scanned; all absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Factory uses plain `CBDescriptor` (`push_s2i_partial_cb_pair`) with `.buffer` set for borrowed-memory rebinding; no `GlobalCircularBuffer`, no `.global_circular_buffer` field, no `.remote_index(`, no `remote_cb_*` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | `CBDescriptor` sets `total_size` / `core_ranges` / `format_descriptors` / `buffer` only — `address_offset` unset (default 0); no `set_address_offset`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Kernels read CTAs at constexpr offsets (`get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>`); `tensor_args_t = {input_tensor, cache_tensor}` is a fixed 2-tensor struct, not a `std::vector<Tensor>` |

- **CB endpoints (GATE-free):** all legal 1:1 in both configs — see Port-work summary.
- **Offset base pointers:** GREEN. Not listed in the dated triage doc `2026-07-19_offset_base_pointers.md`; independent scan of every address-bearing RTA confirms no host-side fold. The only tensor base delivered to a kernel is the destination (cache/output) buffer, pushed as a raw `Buffer*` (`writer_rt.push_back(dst_buffer)` — `_program_factory.cpp:243` TILE, `:294` RM), i.e. the framework-patched `Buffer*`-binding form delivering a **clean base**. No `buffer()->address() + <offset>` fold anywhere. The RM writer's kernel-side `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (`writer_unary_stick_layout_...:22`) adds a *separate clean offset RTA* on the device side — the already-split-out pattern (GREEN), not a host fold — and that path is validate-blocked regardless (see Misc anomalies).
- **TensorAccessor 3rd argument:** GREEN. Not listed in `2026-07-06_tensor_accessor_3rd_arg_triage.md`; scan of every `TensorAccessor(...)` construction shows two-argument form only — TILE writer `TensorAccessor(dst_args, dst_addr)` (`writer_unary_sharded_blocks_interleaved_start_id.cpp:28`) and RM writer `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (`writer_unary_stick_layout_...:22`). No explicit page-size 3rd argument at any site.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input tensor (sharded)** — **clean** (borrowed-memory DFB). CB `c_0` (`src0_cb_index`) is a globally-allocated CB bound to `input.buffer()` (`_program_factory.cpp:140-147`, `cb.buffer = bound_buffer`). The reader kernel does not read tensor memory through a `TensorAccessor` — it only advances the FIFO (`dfb.push_back(num_tiles_per_core)`), because the sharded input data already resides in the borrowed L1. Causal-link gate applies → clean; port via `DataflowBufferSpec::borrowed_from` the input `TensorParameter`.
  - **output / cache tensor (interleaved)** — **Case 1** (via `TensorAccessor`). Delivered as `Buffer*` in `writer_rt[0]`; the writer kernel feeds the base into `TensorAccessor(dst_args, dst_addr)` and does all writes through the accessor (`noc.async_write(dfb_out, s, ...)`). Port: express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`, and the `Buffer*` RTA + `TensorAccessorArgs` CTA plumbing both disappear.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal 1:1 — classified per `(CB, config)`:
  - **`convert_df == false`** (input dtype == output dtype; `out_cb_index == src0_cb_index == c_0`): CB `c_0` — reader `push_back` (locked producer) + writer `wait_front`/`pop_front` (locked consumer) → **1P+1C, legal**.
  - **`convert_df == true`** (dtype conversion; `out_cb_index == c_16`): CB `c_0` — reader producer + compute `cb_wait_front`/`cb_pop_front` consumer → **1P+1C legal**; CB `c_16` — compute `cb_reserve_back`/`cb_push_back` producer + writer `wait_front`/`pop_front` consumer → **1P+1C legal**.
  - No self-loop, no multi-binding flag, no dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no multi-toucher CB in any config.
- **Cross-op / shared kernels:** the op owns **no** kernels; all four are file-path borrows (see Team-only for the full inventory and port-together sets). The Metal 2.0 rewrite of each shared kernel must be adopted by every co-borrower in one change.
- **RTA varargs:** none — every RTA is read as a fixed distinct field at a constant index; no counted loop, no data-selected read.

## Team-only

- **Out-of-directory coupling & donor shape:** **file-path kernel instantiation only — no function-call escape.** The op's factory `CreateKernel`s four kernels it does not own; none of the op's (borrowed) kernels `#include` another op's helper header (their includes are all `tt_metal/*`: `api/dataflow/*`, `api/tensor/*`, `api/compute/*`, `api/debug/dprint.h` — LLK/HAL, no concern). Function-call-escape roll-up: **✓ clean.** Borrowed kernel files and their port-together sets:

  | Kernel file | Owning pool / family | Co-borrowers (approx.) |
  |---|---|---|
  | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | cross-family (`eltwise/unary`) | broadly shared — ~18 factory files reference it |
  | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | in-family (`data_movement/sharded`) | ~3 factory files (incl. `sharded/interleaved_to_sharded`, `sharded_partial`) |
  | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | in-family (`data_movement/sharded`) | shared within the sharded family (RM path) |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | shared kernel pool (`ttnn/cpp/ttnn/kernel/`) | ~4 factory files |

  Each shared kernel's Metal 2.0 rewrite (CB→DFB, named-token bindings) is a single change all co-borrowers must adopt together. `reader_unary_sharded.cpp` is the widest coupling (~18 co-borrowers) and should be sequenced as its own port-together unit. All donor kernels are already Device 2.0 compliant, so none induces a Device 2.0 gate here.
- **Relaxation candidates (from a custom hash):** none — no custom hash on this op.
- **TTNN factory analysis:** `Concept = descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; `Is safe to port? = yes`, `Smuggled pointer = no`. Target concept: `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead RM writer path.** `validate_on_program_cache_miss` hard-asserts `input_tensor.layout() == Layout::TILE` (`device_operation.cpp:24`), yet the factory selects the RM writer kernel and RM per-core RTA branch on `input.layout() != TILE` (`_program_factory.cpp:182-186`, `:259-308`). Given the validate constraint, the RM writer kernel (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`) and the RM RTA branch are unreachable at runtime. They are audited here (kernel is Device 2.0 clean, no offset/3rd-arg gate) but the porter/ops team may consider removing the dead RM branch. Note the RM writer also pushes `num_units_per_row` at `writer_rt[1]` (`_program_factory.cpp:295`) which the RM kernel never reads (it reads arg indices 0,2,3,4,5,6, skipping index 1) — a dead RTA in the dead path.
- **`is_l1_aligned` hardcoded true.** `const bool is_l1_aligned = true;` (`_program_factory.cpp:55`) makes the surrounding conditional (`if (is_blackhole or is_l1_aligned) { if (!dst_is_dram or is_l1_aligned) ... }`, RM path, `:287-291`) always take the L1-alignment branch, so `is_blackhole` and `dst_is_dram` are effectively dead there. In the dead RM path regardless; flagged only for the ops team's awareness.

## Recipe notes

- **Device 2.0 gate vs. compute-kernel CB-sync free functions.** The gate text says "every kernel the op uses" must be Device 2.0 compliant, and its holdover examples (`get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`) are framed around data-movement CB pointer access. The compute kernel `eltwise_copy.cpp` uses `cb_wait_front`/`cb_reserve_back`/`cb_pop_front`/`cb_push_back` free functions (compute-engine CB FIFO sync), which superficially match the "CB-index-keyed free function" holdover shape but have no data-movement content to migrate — they are the standard compute API, and the DM migration guide covers only NoC / CB-for-DM / semaphore / memory-access. I read this as **not** a Device 2.0 violation (compute kernels legitimately keep these free functions; the reader/writer here are object-based while the compute kernel is not, which appears to be the intended steady state). Flagging because the gate's Green/holdover bullets don't explicitly carve out compute-side CB-sync free functions, and a stricter reader could gate on the shape. A one-line note in the Device 2.0 subject ("compute-side `cb_*` FIFO-sync free functions are not DM holdovers") would remove the ambiguity.
