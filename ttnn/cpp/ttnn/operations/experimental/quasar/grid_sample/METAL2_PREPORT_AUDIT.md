# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample`

- **`GridSampleOperation`** (`device/grid_sample_device_operation.hpp:33`, namespace `ttnn::prim::qsr`)
  - `GridSampleBilinearProgramFactory` (`device/grid_sample_bilinear_program_factory.cpp`) — selected when `mode == "bilinear"` (`device/grid_sample_device_operation.cpp:18`)
  - `GridSampleNearestProgramFactory` (`device/grid_sample_nearest_program_factory.cpp`) — every other mode

**Audit target, by user instruction:** the **`experimental/quasar/` copy** of `pool/grid_sample`. The user plans to port only this copy; the original under `pool/grid_sample` stays as it is. At audit time the copy matches the original apart from include paths, the namespace (`ttnn::prim::qsr`), kernel source paths, Python binding names, and one dropped host-side `verify_numerical_configuration` call (`pool/grid_sample/device/grid_sample_device_operation.cpp:317`, missing from the copy). None of these differences changes any finding below. (See *Recipe notes*: the recipe normally treats `experimental/quasar/` as off limits.)

**Kernels referenced by the factories:**

| Kernel | Owner | Used by |
|---|---|---|
| `experimental/quasar/grid_sample/device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp` | this op | Bilinear, interleaved grid |
| `experimental/quasar/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp` | this op | Bilinear, sharded grid (×1 or ×2 for split reader) |
| `experimental/quasar/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | this op | Bilinear, interleaved grid |
| `experimental/quasar/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp` | this op | Nearest (always ×2: split reader is always on for nearest, `device/grid_sample_utils.cpp:15-17`) |
| `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | **`pool/generic` (main tree, borrowed)** | Bilinear, all configs |

Op-owned shared header: `device/kernels/grid_sample_reader_common.hpp`. Donor headers: `pool/device/kernels/pool_kernels_common.hpp`, `pool/device/kernels/experimental_device_api.hpp` (main tree).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md` (apart from the quasar-directory point above).

**Recipe docs:** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes` (first run against `c41f7f4ef0a`; re-checked against this revision)

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GridSampleOperation` → `GridSampleBilinearProgramFactory`, `GridSampleNearestProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all five kernels and both donor headers use `Noc` / `DataflowBuffer` / `TensorAccessor`; no legacy idioms |
| *Prereqs* — Cross-op escapes | Ok — donor calls are all ✓ shapes; one borrowed compute kernel (see Heads-ups) |
| *Feature Support* — overall | GREEN (all Appendix A rows N/A) |
| *Feature Support* — Variadic-CTA | Ok — none |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both rows; read from the `pool/grid_sample` rows as a proxy — see Gate detail) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD` |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — TensorParameter Case 2 | none (GREEN) |
| *Port work* — Tensor bindings (per binding) | Case 1 (input everywhere; grid and output when interleaved) / clean (grid and output when backed by a sharded tensor) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal / self-loop / 1P+1C (no multi-binding, no dead CB) |

## Result

**Correction (re-check against `9df4aa1e011`, which added a "who borrows from this op" check):** the first version of this audit said grid_sample's kernels were bound only by grid_sample. That was wrong. `experimental/quasar/rotate` binds `device/kernels/dataflow/writer_grid_sample_interleaved.cpp` (`experimental/quasar/rotate/device/rotate_bilinear_program_factory.cpp:327-329`) and its reader includes `device/kernels/grid_sample_reader_common.hpp`. Per user decision the two ops are ported together in one change, so these files convert in place for both.

**GREEN → brief issued.** Every gate clears. The bilinear factory borrows the main-tree compute kernel `pool/generic/device/kernels/compute/compute_pool_2d.cpp`. **User decision (2026-11-27): nothing may be added to or changed in the main tree.** So the port copies that kernel into `experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp` (bound only by this op) and converts the copy there. The main-tree original stays untouched and no `_metal2` copy is made beside it.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The readiness sheet has no rows for the quasar copy. I used the two `pool/grid_sample` rows, because the copy's host code matches the original (diff above). Both rows read `Is able to port?` = `yes`, `Concept` = `descriptor`, `Porting Target` = `ProgramSpecFactoryConcept`, `TensorParameter relaxation` = `none`, `Known op issues` = blank, `Smuggled pointer` = `no`.
  Cross-check against the quasar copy (all clean):
  - `Concept`: both factories define `create_descriptor(...)` returning `ProgramDescriptor`, with no mesh coordinate argument (`device/grid_sample_device_operation.hpp:24,29`) → `descriptor` ✓
  - `Custom hash`: no `compute_program_hash` in the device op ✓. `GridSampleParams` has the usual `attribute_names` / `attribute_values` reflection (`device/grid_sample_device_operation_types.hpp:21-38`); the sheet's backdoor column reads `(complete)`.
  - `get_dynamic_runtime_args`: absent ✓
  - `Override runtime args method?`: absent ✓
  - `Pybind descriptor`: `grid_sample_nanobind.cpp` binds only the user function (`bind_function<"grid_sample", "ttnn.experimental.quasar.">`); no `create_descriptor` binding ✓
  - Factory set: two factories in code, two sheet rows, matching one to one ✓
  - Invariants: `Op-owned tensors?` is blank on a `descriptor` row; `get_dynamic_runtime_args` = `no` ✓
- **Device 2.0 (every kernel used):** GREEN. All dataflow kernels use `Noc`, `DataflowBuffer`, `TensorAccessor`, and `UnicastEndpoint`. There are no raw `noc_async_*` calls, no `cb_*` free functions, and no address generators. `compute_pool_2d.cpp` handles FIFO sync through `DataflowBuffer` (`:104-110`); its LLK calls take buffer IDs, which is normal. Free functions that take a buffer index:
  - `get_tile_size(cb_id)` in `pool/device/kernels/pool_kernels_common.hpp:75` (inside `zero_out_tiles`) is on the allowed list.
  - `get_dataformat(scalar_cb_index)` at `device/kernels/grid_sample_reader_common.hpp:33` (inside `fill_four_val`) is **not** an old-style leftover. At that call site no `DataflowBuffer` object is in scope: the helper receives only a raw address and the index as a template parameter. The value is also used in an `if constexpr`. Both Red-bullet conditions for a leftover ("wrapper in scope" and "replacement exists") therefore fail. See Recipe notes.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Only the header `<tt-metalium/memory/cross_program/global_circular_buffer.hpp>` is included (`device/grid_sample_device_operation.hpp:14`). No type use, no `.global_circular_buffer` field, no `remote_index`. A header on its own is not definitive (Appendix A). Unused include → Misc anomalies. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`. |
  | GlobalSemaphore in a compute kernel | N/A | No semaphores of any kind. |
  | Caller GlobalSemaphore not re-applied on a cache hit | N/A | |
  | Fabric mux | N/A | |
  | RPCM with per-route or auto-selected links | N/A | |
  | MeshSocket | N/A | |
  | Fused CCL op signalers | N/A | |

- **CB endpoints (GATE-free):** every CB is legal, self-loop, or 1P+1C. No multi-binding flag and no dead CB. Full table under *Port-work summary*.
- **Offset base pointers:** GREEN. Every tensor address reaches a kernel as a bare `Buffer*` runtime arg (`emplace_runtime_args(core, {tensor.buffer(), ...})`), with no host arithmetic: bilinear `:409,416,435,436,445`, nearest `:208,215,231,232,242,243`. No accessor base is offset. grid_sample does not appear in the dated triage doc (`2026-07-19_offset_base_pointers.md`), which agrees with this scan.
- **TensorParameter analysis (Case 2):** GREEN — no binding is Case 2 (every tensor address feeds a `TensorAccessor`; sharded grid/output are borrowed DFBs).
- **TensorAccessor 3rd argument:** N/A. Every `TensorAccessor(...)` in the op's kernels uses the 2-argument form (`reader_grid_sample_interleaved_start_id.cpp:44-45`, `reader_grid_sample_sharded.cpp:90`, `writer_grid_sample_interleaved.cpp:21`, `writer_grid_sample_nearest_sharded.cpp:179,182`). grid_sample is not in the 3rd-arg triage doc either.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config). Every address arrives as a `Buffer*` runtime arg, which the framework patches correctly on cache hits today (not the stale-pointer hazard). Classified by how the kernel uses it:

  | Factory / config | `input` | `grid` | `output` |
  |---|---|---|---|
  | Bilinear, interleaved grid | Case 1 (reader RTA0 → `TensorAccessor`) | Case 1 (reader RTA1 → `TensorAccessor`) | Case 1 (writer RTA0 → `TensorAccessor`) |
  | Bilinear, sharded grid | Case 1 (reader(s) RTA0) | clean — CB backed by the grid buffer (`:110`) | clean — CB backed by the output buffer (`:215`) |
  | Nearest, interleaved grid (output is always sharded here) | Case 1 (both writers RTA0) | Case 1 (both writers RTA1) | clean — CB backed by the output buffer (`nearest:134`) |
  | Nearest, sharded grid | Case 1 (both writers RTA0) | clean — CB backed by the grid buffer (`nearest:91`) | clean — CB backed by the output buffer |

  No Case 2 (raw pointer) bindings.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per node, per config):

  | Factory | CB | Config | Touchers | Disposition |
  |---|---|---|---|---|
  | Bilinear | grid (`c_0`) | interleaved | reader (`noc.async_read` into it + `get_write_ptr`, no FIFO ops) | **self-loop** |
  | Bilinear | grid (`c_0`, backed by grid buffer) | sharded, no split | reader0 (`get_read_ptr`) | **self-loop** |
  | Bilinear | grid (`c_0`, backed by grid buffer) | sharded, split | reader0 + reader1 (both `get_read_ptr`, no FIFO ops) | **1P+1C** |
  | Bilinear | input0 / scalar0 | all | reader0 produces → compute consumes | legal |
  | Bilinear | input1 / scalar1 | sharded + split only (allocated only then, `:157,185`) | reader1 produces → compute consumes | legal |
  | Bilinear | output | interleaved | compute produces → writer consumes | legal |
  | Bilinear | output (backed by output buffer) | sharded | compute produces, nothing consumes | **self-loop** |
  | Nearest | grid0 (`grid_cb_index0`) | interleaved | writer0 only (writer1 uses grid1) | **self-loop** |
  | Nearest | grid1 (`grid_cb_index1`) | interleaved only (`nearest:96`) | writer1 only | **self-loop** |
  | Nearest | grid0 (backed by grid buffer) | sharded | writer0 + writer1 (both read via `get_write_ptr`, no FIFO ops) | **1P+1C** |
  | Nearest | fill | all | writer0 + writer1 (both `zero_out_page` + `get_write_ptr`, no FIFO ops) | **1P+1C** |
  | Nearest | output (backed by output buffer) | all | writer0 + writer1 (raw writes at different offsets; same source, RISCV_0/RISCV_1) | **1P+1C** |

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The nearest factory runs two copies of one writer that split the work (`nearest:169-197`), and the sharded bilinear split reader works the same way. Each co-used CB has exactly two touchers that never call FIFO functions, so assign 1P+1C and don't set the multi-binding flag.
- **Cross-op / shared kernels:** `compute_pool_2d.cpp` is borrowed from `pool/generic/device/kernels/compute/` in the main tree. **Per user decision, the port makes a grid_sample-local copy at `experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp` and converts that; it does not create a `_metal2` copy in the main tree, and does not edit the original.** Don't reuse `experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp`: it was reworked for Quasar pool and doesn't fit grid_sample (split-reader path removed; `PACK_TO_SCRATCH 1` packs to a scratch buffer that needs a reader kernel to drain it; requires every channel block to be the same width (`static_assert`), whereas grid_sample needs a narrower last block; host-chosen `max_tiles_per_reduction` replaces compile-time arg 16). Other factories still bound to the main-tree original (left on legacy, for information only): `pool/generic/device/pool_multi_core_program_factory.cpp:939`, `pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp`, `pool/rotate/device/rotate_bilinear_program_factory.cpp`, `experimental/quasar/rotate/device/rotate_bilinear_program_factory.cpp:294`.
- **Indexed-collection args:** none. Every runtime arg and compile-time arg is read at a fixed index.
- **Two compute kernels on separate core groups (bilinear, interleaved):** `:367-375` creates one compute descriptor for `core_group_1` and one for `core_group_2`. They differ only in compile-time arg 3 (`total_interpolations`). Keep two KernelSpecs. Do not move that value into a runtime arg (see the "demoting per-group CTA to RTA" anti-pattern in `port_patterns.md`).
- **`compute_pool_2d.cpp` creates buffer objects for CBs that grid_sample never allocates:** `in_dfb_1` / `in_scalar_dfb_1` (index `DUMMY_CB_ID` = 32 unless sharded with split reader), `pre_tilize_dfb` (index 32, `:300`), and `fast_tilize_dfb` (compile-time arg 38 = 32) (`compute_pool_2d.cpp:104-110`). Its Metal 2.0 version therefore needs those bindings to be optional (the "Conditional / optional resource bindings" pattern in `port_patterns.md`). The same kernel also reads runtime arg 0 when compile-time arg 3 is 0 (`:129`); grid_sample always passes a non-zero count, so grid_sample has no compute runtime args.
- **Nearest writer, sharded config: an unused grid accessor.** `writer_grid_sample_nearest_sharded.cpp:181-182` always builds `grid_tensor_accessor` with `grid_addr = 0`. When sharded, the host fills that second accessor slot with the *input's* accessor args (`nearest:165-167`), and the accessor is only used under `if constexpr (!is_sharded)` (`:227-231`). A literal `TensorAccessor(tensor::grid)` rewrite would bind the grid tensor in a config where the kernel never reads it as a tensor. Check how the binding behaves in the sharded config before rewriting this line.
- **Index-based buffer metadata used at compile time:** `get_dataformat(scalar_cb_index)` in an `if constexpr` (`grid_sample_reader_common.hpp:33`), and `get_tile_size(cb_id)` as `constexpr` in the donor `zero_out_tiles<cb_id>` (`pool_kernels_common.hpp:75`). Per kernel-side whitelist rule 7, `constexpr` sites keep the free-function form with the binding token passed as the template parameter. Don't turn them into member getters or demote them to `const`.
- **Donor header carries a stale include:** `pool/device/kernels/experimental_device_api.hpp:11` includes `api/dataflow/circular_buffer.h` and defines `using CB = CircularBuffer`. grid_sample's kernels include this header (via `grid_sample_reader_common.hpp:11` and `pool_kernels_common.hpp:12`) but never use `CB`. It is a donor file, outside the port's scope. Don't copy the include into ported kernels.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean** (function-call escapes); one borrowed kernel file.
  - Roll-up: only `tt_metal/*` headers plus two main-tree pool donor headers. Every called donor function takes `Noc` + a `DataflowBuffer`-type template parameter, sometimes with a `uint32_t cb_id` template parameter (✓).

  | Op kernel | Donor file | Class | Functions called → shape | Status |
  |---|---|---|---|---|
  | `reader_grid_sample_interleaved_start_id.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | cross-family (pool shared) | `zero_out_tiles<uint32_t cb_id, DFB>(Noc, DFB)` → `uint32_t cb_id` + `DataflowBuffer` | ✓ |
  | `reader_grid_sample_sharded.cpp` | `pool_kernels_common.hpp` | cross-family (pool shared) | `zero_out_tiles<cb_id>(Noc, DFB)`, `zero_out_page(Noc, DFB)` | ✓ |
  | `writer_grid_sample_nearest_sharded.cpp` | `pool_kernels_common.hpp` | cross-family (pool shared) | `zero_out_page(Noc, DFB)` | ✓ |
  | all grid_sample dataflow kernels (via `grid_sample_reader_common.hpp`) | `pool/device/kernels/experimental_device_api.hpp` | cross-family (pool shared) | `experimental::local_addr(uint32_t addr, uint8_t noc_id)` → plain L1 address, no resource handle | ✓ |
  | `compute_pool_2d.cpp` (borrowed) | `experimental_device_api.hpp` | cross-family | nothing called (include only) | ✓ |

  - **Who borrows from grid_sample:** quasar rotate — `writer_grid_sample_interleaved.cpp` (bilinear writer) and `grid_sample_reader_common.hpp` (bilinear reader include). Joint port.
  - **Borrowed kernel files:** `pool/generic/device/kernels/compute/compute_pool_2d.cpp`, owned by `pool/generic`. Widely shared: binders are listed in Heads-ups. No `_metal2` copy sits beside it. `experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp` is a separate duplicate with Metal 2.0 and Quasar-specific changes that don't fit grid_sample (see Heads-ups). Resolution: a grid_sample-local copy (user decision).
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:** concept `descriptor` (`device/grid_sample_device_operation.hpp:24,29`); SPMD; no op-owned tensors; no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`; no pybound `create_descriptor`. Target `ProgramSpecFactoryConcept`.

## Misc anomalies

- **Unused runtime arg (nearest, interleaved):** the host sends `grid_sticks` as runtime arg 2 (`device/grid_sample_nearest_program_factory.cpp:233,244`), but `writer_grid_sample_nearest_sharded.cpp:171-173` reads only args 0, 1, and 3. The kernel takes the stick count from compile-time arg 13 instead.
- **Unused include:** `device/grid_sample_device_operation.hpp:14` includes `global_circular_buffer.hpp`, which nothing uses. It makes the op look like a GlobalCircularBuffer user to text-search tools.
- **Duplicate accessor args (nearest, sharded):** `nearest:165-167` appends the input tensor's `TensorAccessorArgs` twice, only to fill the slot the kernel's unused `grid_tensor_accessor` expects.
- **Both nearest writers zero the same fill CB** (`writer_grid_sample_nearest_sharded.cpp:199-200`). Harmless (both write zeros, and each waits for its own zeroing to finish before reading), but redundant.
- **`read_four_corner_inputs_with_fill`** (`device/kernels/grid_sample_reader_common.hpp:210-283`) is not called by grid_sample's own kernels; quasar rotate's bilinear reader uses it.
- **Quasar copy drift:** the copy drops `ttnn::verify_numerical_configuration(...)` from `grid_sample` (present at `pool/grid_sample/device/grid_sample_device_operation.cpp:317`). This doesn't affect the port, but the copy is not quite identical to the original.

## Questions for the user

1. ~~Where should the Metal 2.0 version of `compute_pool_2d.cpp` live?~~ **Resolved (user, 2026-11-27):** nothing goes in the main tree. The port copies the kernel into `experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp` and converts it there. The existing quasar `pool_generic` copy was checked and doesn't fit (see Heads-ups).
2. **Readiness-sheet proxy:** the sheet tracks only `pool/grid_sample`. I accepted its rows for the quasar copy because the code matches. If the copy starts to drift during porting, a re-audit can no longer lean on those rows.

## Recipe notes

- **Quasar directory conflict.** The recipe says "`ttnn/cpp/ttnn/operations/experimental/quasar/` is out of bounds — for the audit as much as for the port … Don't read them." This audit targets exactly that directory, at the user's explicit request (the plan is to port only the quasar copy). The recipe doesn't cover this case. It affects three places: (1) the readiness sheet has no row for a quasar copy, so the "missing row → spreadsheet broken → GATE" rule would fire mechanically; I judged the original's rows a valid proxy because the code matches. (2) The shared-kernel fork rule (rung 2: put the fork beside the original, in the main tree) conflicts with a quasar-only port. The user settled it with a local copy inside the op's quasar directory, which is none of the recipe's three rungs. The recipe could name this as the rule for whole-op quasar copies. (3) The "quasar `_metal2` files don't count" rule may not fit when the port itself lives under quasar.
- **`get_dataformat(cb_id)` is not covered.** The Device 2.0 Green bullet allows exactly `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`. `get_dataformat(cb_id)` is the same kind of index-based metadata lookup, and port whitelist rule 7 already names it next to `get_tile_size`. Here it was not a leftover anyway, because no wrapper object was in scope. A future case with a wrapper in scope would be forced RED for a lookup that seems to be treated as acceptable in practice. Consider adding it to the allowed list or saying explicitly why it isn't on it.
