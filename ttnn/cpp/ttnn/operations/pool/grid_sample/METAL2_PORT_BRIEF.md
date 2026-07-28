# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/grid_sample`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `1b475de4782 2026-07-27 docs(metal_2.0): make the shared-kernel _metal2 fork a reusable checked-in artifact` *(carry this line into the port report's Provenance section)*

## Scope

One device operation, two program factories, five kernel sources:

- **`GridSampleOperation`** (`device/grid_sample_device_operation.hpp`)
  - `GridSampleBilinearProgramFactory` (`device/grid_sample_bilinear_program_factory.cpp`) — binds `reader_grid_sample_sharded.cpp` (1 or 2 instances), `reader_grid_sample_interleaved_start_id.cpp`, `writer_grid_sample_interleaved.cpp`, and the borrowed `pool/generic/device/kernels/compute/compute_pool_2d.cpp`
  - `GridSampleNearestProgramFactory` (`device/grid_sample_nearest_program_factory.cpp`) — binds `writer_grid_sample_nearest_sharded.cpp` (always 2 instances)

Five configurations, referenced by tag throughout this brief:

| Tag | Factory | Grid memory layout | Split reader |
|---|---|---|---|
| **B-INT** | bilinear | interleaved | off (forced — `device/grid_sample_utils.cpp:19-21`) |
| **B-SH** | bilinear | height-sharded | off |
| **B-SH-SR** | bilinear | height-sharded | on |
| **N-SH** | nearest | height-sharded | on (always — `device/grid_sample_utils.cpp:15-17`) |
| **N-INT** | nearest | interleaved | on (always) |

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced as a `safe` warning. All `no` on this op.

## Construct — to do

### Tensor bindings

Three tensors. Every non-clean binding is **Case 1**; there is no Case 2 anywhere in this op, and no kernel does hand-rolled address arithmetic on a base pointer.

| Binding | B-INT | B-SH | B-SH-SR | N-SH | N-INT |
|---|---|---|---|---|---|
| `input` | Case 1 | Case 1 | Case 1 | Case 1 | Case 1 |
| `grid` | Case 1 | clean | clean | clean | Case 1 |
| `output` | Case 1 | clean | clean | clean | clean |

- **`input` — Case 1, every configuration.** Express as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::input)`. Both the legacy `Buffer*` runtime arg (slot 0 of every dataflow kernel) and the `TensorAccessorArgs` compile-time plumbing disappear. Kernel sites: `reader_grid_sample_interleaved_start_id.cpp:19` → `:45`; `reader_grid_sample_sharded.cpp:64` → `:90`; `writer_grid_sample_nearest_sharded.cpp:167`/`:170` → `:178`.
- **`grid` — Case 1 in B-INT and N-INT.** Same mechanical conversion. Kernel sites: `reader_grid_sample_interleaved_start_id.cpp:20` → `:44`; `writer_grid_sample_nearest_sharded.cpp:171` → `:181`.
- **`grid` — clean in B-SH, B-SH-SR, N-SH.** The grid CB is borrowed-memory (`device/grid_sample_bilinear_program_factory.cpp:110`, `device/grid_sample_nearest_program_factory.cpp:96`) and the kernel reads the tensor straight out of it by pointer. Port via `DataflowBufferSpec::borrowed_from` — no accessor, no binding case.
- **`output` — Case 1 in B-INT only.** `device/grid_sample_bilinear_program_factory.cpp:445` → `writer_grid_sample_interleaved.cpp:11` → `:21`.
- **`output` — clean in every other configuration.** Borrowed-memory output CB: `device/grid_sample_bilinear_program_factory.cpp:215` (bilinear sharded — note there is **no writer kernel** in those configs, `:380`) and `device/grid_sample_nearest_program_factory.cpp:139` (nearest — unconditional; nearest mode always produces a sharded output). Port via `DataflowBufferSpec::borrowed_from`.

### TensorParameter relaxation

None.

### TensorAccessor 3rd arg

None — every `TensorAccessor(...)` construction in scope is already two-argument. Nothing to drop.

### CB endpoints

No CB in this op needs the multi-binding advanced option, and no CB is dead. Apply these dispositions:

**Self-loop** (one toucher — bind that kernel PRODUCER *and* CONSUMER):

- `grid_cb` (`c_0`), **B-INT** — reader0 only (`reader_grid_sample_interleaved_start_id.cpp:74`, `:79`)
- `grid_cb` (`c_0`, borrowed), **B-SH** — reader0 only (`reader_grid_sample_sharded.cpp:104`)
- `output_cb` (borrowed), **B-SH and B-SH-SR** — compute only; it FIFO-produces into the borrowed output shard and nothing drains it (there is no writer kernel in these configs). Compute self-loops are legal on Gen1.
- `grid_cb_0` and `grid_cb_1`, **N-INT** — writer0 touches only `grid_cb_0`, writer1 only `grid_cb_1` (`device/grid_sample_nearest_program_factory.cpp:190` selects per instance)

**Assign 1P+1C** (two touchers, both sync-free — bind one PRODUCER, one CONSUMER; cosmetic on Gen1):

- `grid_cb` (`c_0`, borrowed), **B-SH-SR** — reader0 and reader1 both read it. Reader1's compile-time overrides touch only CTAs 0, 2 and 17 (`device/grid_sample_bilinear_program_factory.cpp:276-278`), so CTA 1 (`grid_cb_index`) is shared between the two instances. This one is easy to miss.
- `grid_cb_0` (`c_0`, borrowed), **N-SH** — both writer instances bind `grid_cb_index0` and read via `get_write_ptr()` (`writer_grid_sample_nearest_sharded.cpp:192`)
- `fill_cb`, **N-SH and N-INT** — both writer instances call `get_write_ptr()` and `zero_out_page` (`writer_grid_sample_nearest_sharded.cpp:198-199`)
- `output_cb` (borrowed), **N-SH and N-INT** — both writer instances raw-write into it at disjoint offsets (`writer_grid_sample_nearest_sharded.cpp:97-102`, `:104-111`, `:256-261`)

**Already legal 1:1** (one locked producer + one locked consumer — no action): `input_cb_0`, `input_cb_1`, `scalar_cb_0`, `scalar_cb_1` in every bilinear configuration, and `output_cb` in B-INT.

## Watch for

- **CB endpoints (multi-binding):** none — every multi-toucher CB here has exactly two touchers and takes a 1P+1C assignment. You still have to *find* all of them: in Metal 2.0 an unbound toucher is a kernel that cannot legally access the DFB. There is no hidden second writer to hunt — the op declares **no semaphores at all**, so the semaphore-gated raw co-fill shape cannot occur.

  Two census traps worth naming:
  - Every two-toucher CB above is the **dual-instance work-split** shape (same `kernel_source` in two `KernelDescriptor`s differing only by processor/NOC and a `reader_id` CTA, both over one `core_ranges` — `device/grid_sample_bilinear_program_factory.cpp:280-287`, `device/grid_sample_nearest_program_factory.cpp:193-201`). Both instances hit every node, so each node genuinely has two touchers. This is 1P+1C, not multi-binding.
  - The bilinear factory can emit **two compute `KernelDescriptor`s** (`device/grid_sample_bilinear_program_factory.cpp:369-376`) when `split_work_to_cores` yields a second core group — but over **disjoint** core ranges. Each node sees one compute instance, so this does *not* add a toucher. Don't count it as one.

- **Cross-op / shared kernels — two sources, in opposite directions, neither forked yet.** This is the port's main coordination cost. Read `port_patterns.md` → *Caution: Porting a shared kernel* before touching either.

  1. **`ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp` — borrowed.** Bound at `device/grid_sample_bilinear_program_factory.cpp:354`. **No `_metal2` fork exists** → rung 2: create `compute_pool_2d_metal2.cpp` beside the original in `pool/generic`'s directory, convert the copy, point your `KernelSpec::source` at it, and add the pointer comment to the original. The copy under `experimental/quasar/pool_generic/` is a whole-op pre-port copy and does **not** count as a fork to reuse. Other binders — **sunset list, not authorization to convert in place**: `pool/generic` (`pool_multi_core_program_factory.cpp:920`), `pool/rotate` (`rotate_bilinear_program_factory.cpp:293`).

  2. **`device/kernels/dataflow/writer_grid_sample_interleaved.cpp` — lent.** It lives in grid_sample's own directory, so nothing about its path warns you — but `pool/rotate` binds it at `rotate_bilinear_program_factory.cpp:327`. **No `_metal2` fork exists** → rung 2 again: fork it in place beside the original, in your own directory. Converting it in place would break rotate's build. Other binders — **sunset list, not authorization**: `pool/rotate`.

  Also note that `pool/rotate` consumes two functions from grid_sample's own header `device/kernels/grid_sample_reader_common.hpp` — `read_four_corner_inputs_with_fill` (`reader_rotate_bilinear_interleaved.cpp:123`) and `fill_four_val` (`:139`) — and its host factory includes `device/grid_sample_utils.hpp` (`rotate_bilinear_program_factory.cpp:8`). Neither function's signature needs to change under Metal 2.0 (both take `Noc` / `DataflowBuffer` / raw L1 addresses), but do not alter them. `read_four_corner_inputs_with_fill` has no caller inside grid_sample at all — rotate is its only consumer.

- **The borrowed compute kernel constructs `DataflowBuffer`s over CB indices grid_sample never allocates.** `compute_pool_2d.cpp:104-110` unconditionally constructs seven `DataflowBuffer` objects. Grid_sample passes the sentinel index `32` for four of them: `input_cb_index_1` and `scalar_cb_index_1` when split reader is off (`device/grid_sample_bilinear_program_factory.cpp:156`, `:184`, both `DUMMY_CB_ID` = 32), `pre_tilize_cb_id` (`:301-302`, a bare literal `32`), and `fast_tilize_cb_id` (`:350`, `DUMMY_CB_ID`). Those code paths are compile-time dead for grid_sample (`is_output_tiled = false`, and the split-reader paths are gated on `ct_arg[2]`), so no access ever occurs at runtime — but in Metal 2.0 each `DataflowBuffer` needs a `dfb::` token to construct from, and there is no binding for an unallocated index. Resolve this when you fork the compute kernel. It is not a dead CB (nothing is allocated) and it does not appear in the endpoint census (no toucher), so it will not surface from either of those checks.

- **RTA varargs:** none. Every kernel reads its runtime args a fixed number of times at distinct literal indices — `reader_grid_sample_sharded.cpp:64-65`; `reader_grid_sample_interleaved_start_id.cpp:19-22`; `writer_grid_sample_interleaved.cpp:11-13`; `writer_grid_sample_nearest_sharded.cpp:167-172` (indices 0, 1, 3, selected by a `constexpr` branch). Name every one of them; reach for no vararg mechanism.

  One thing to know while naming: **runtime-arg slot 2 of the nearest writer's interleaved path is dead** (`device/grid_sample_nearest_program_factory.cpp:239`, `:250` emit `grid_sticks`; the kernel reads 0, 1 and 3, never 2). It is recorded as an anomaly for the ops team in `METAL2_PREPORT_AUDIT.md` — it is **not** yours to remove in the port diff. Same for the dead compile-time-arg slot 5 (`input_batch`) in the bilinear *interleaved* reader, which the *sharded* reader does read.
