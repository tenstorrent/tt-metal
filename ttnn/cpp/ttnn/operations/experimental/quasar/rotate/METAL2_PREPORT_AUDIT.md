# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/quasar/rotate`

- **`RotateDeviceOperation`** (`device/rotate_device_operation.hpp:15`, namespace `ttnn::operations::experimental::quasar::rotate`)
  - `NearestProgramFactory` (`device/rotate_nearest_program_factory.cpp`) — every mode except `bilinear` (`device/rotate_device_operation.cpp:15-21`)
  - `BilinearProgramFactory` (`device/rotate_bilinear_program_factory.cpp`) — `interpolation_mode == "bilinear"`

**Audit target, by user instruction:** the `experimental/quasar/` copy of `pool/rotate`, ported **together with `experimental/quasar/grid_sample`** in one change (the two share kernels; see Out-of-directory coupling). Nothing under `pool/` changes. The copy matches `pool/rotate` apart from include paths, namespaces, kernel source paths (its bilinear writer points at the quasar grid_sample writer) and Python binding names.

**Kernels referenced by the factories:**

| Kernel | Owner | Used by |
|---|---|---|
| `experimental/quasar/rotate/device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp` | this op | Nearest (reader) |
| `experimental/quasar/rotate/device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp` | this op | Nearest (writer) |
| `experimental/quasar/rotate/device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp` | this op | Bilinear (reader; includes quasar grid_sample's `grid_sample_reader_common.hpp`) |
| `experimental/quasar/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | **quasar grid_sample (borrowed)** | Bilinear writer, interleaved output only |
| `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | **pool/generic, main tree (borrowed)** | Bilinear compute ×1 (sharded) or ×2 (interleaved core groups) |

Donor headers (include-only): `pool/device/kernels/pool_kernels_common.hpp`, `pool/device/kernels/fixed_point_arithmetic.hpp`, `pool/device/kernels/experimental_device_api.hpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md` (apart from the quasar-directory point, as for grid_sample).

**Recipe docs:** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/quasar/rotate` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotateDeviceOperation` → `NearestProgramFactory`, `BilinearProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — `Noc` / `DataflowBuffer` / `TensorAccessor` throughout; only sanctioned index-keyed calls |
| *Prereqs* — Cross-op escapes | Ok — all donor calls are ✓ shapes; borrows two kernel files (co-ported) |
| *Feature Support* — overall | GREEN (all Appendix A rows N/A) |
| *Feature Support* — Variadic-CTA | Ok — none |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both rows; `pool/rotate` rows read as proxy) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD` |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings | Case 1 (input, output); clean (output when sharded, backing the borrowed output DFB) |
| *Port work* — TensorParameter Case 2 | none (GREEN) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal / self-loop / **dead-CB drop** (nearest sharded input CB) |

## Result

**GREEN → brief issued.** Every gate clears. Port it in the same change as quasar grid_sample: its bilinear factory runs grid_sample's writer and includes grid_sample's reader header, both of which that port converts in place.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. No sheet rows for the quasar copy; the two `pool/rotate` rows (`BilinearProgramFactory`, `NearestProgramFactory`) are used as a proxy because the host code matches. Both read `Is able to port?` = `yes`, `Concept` = `descriptor`, `Porting Target` = `ProgramSpecFactoryConcept`, `TensorParameter relaxation` = `none`, `Known op issues` blank, `Smuggled pointer` = `no` (sheet fetched this session). Cross-check: both factories define `create_descriptor` returning `ProgramDescriptor`, no mesh coordinate (`device/rotate_device_operation.hpp:32-44`) ✓; no `compute_program_hash`, and `operation_attributes_t` has no hand-written `attribute_values` (default reflection hash) ✓; no `get_dynamic_runtime_args` ✓; no `override_runtime_arguments` ✓; `rotate_nanobind.cpp` binds only the user function ✓; two factories, two rows ✓.
- **Device 2.0 (every kernel used):** GREEN. All three rotate kernels and the two borrowed kernels use `Noc`, `DataflowBuffer`, `TensorAccessor`, `UnicastEndpoint`; no raw `noc_async_*`, no `cb_*` free functions, no address generators. Index-keyed calls: `get_tile_size(cb_id)` (donor `zero_out_tiles`, allowed list) is not reached by rotate; `fill_four_val<scalar_cb_id>` → `get_dataformat(scalar_dfb_id)` in grid_sample's header is the same no-wrapper-in-scope, `if constexpr` case recorded in the grid_sample audit.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | not referenced |
  | CBDescriptor `address_offset` (non-zero) | N/A | never set |
  | GlobalSemaphore in a compute kernel | N/A | no semaphores |
  | Caller GlobalSemaphore not re-applied on a cache hit | N/A | no semaphores |
  | Fabric mux | N/A | |
  | RPCM with per-route or auto-selected links | N/A | |
  | MeshSocket | N/A | |
  | Fused CCL op signalers | N/A | |

- **Offset base pointers:** GREEN. Every address is a bare `Buffer*` RTA (`rotate_bilinear_program_factory.cpp:369,383`; `rotate_nearest_program_factory.cpp:271,284,299,312`).
- **TensorParameter analysis (Case 2):** GREEN — every tensor address feeds a `TensorAccessor` (`reader_rotate_bilinear_interleaved.cpp:43`, `reader_rotate_nearest_interleaved.cpp:35`, `writer_rotate_nearest_interleaved.cpp:23`, borrowed writer `writer_grid_sample_interleaved.cpp`).
- **TensorAccessor 3rd argument:** N/A — every accessor is 2-argument.
- **CB endpoints (GATE-free):** see Port-work summary.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:** `input` Case 1 (both readers, RTA 0). `output` Case 1 (nearest writer, RTA 0, every config; bilinear borrowed writer, interleaved output). `output` also backs a borrowed DFB when sharded (bilinear `:214`, nearest `:195`).
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**

  | Factory | CB | Config | Touchers | Disposition |
  |---|---|---|---|---|
  | Nearest | fill | all | reader (`zero_out_page` / raw fill + `get_write_ptr`; no FIFO ops) | self-loop |
  | Nearest | input (backed by input buffer, `:170-182`) | sharded (incl. ND-sharded) only | **none** — its index reaches no kernel (reader CTAs `:202-214`, writer CTAs `:220-226`) | **dead-CB drop** (dead in every config where it exists; no CTA carries its index) |
  | Nearest | output | interleaved | reader produces → writer consumes | legal |
  | Nearest | output (backed by output buffer) | sharded | reader produces → writer consumes | legal |
  | Bilinear | fill | all | reader only | self-loop |
  | Bilinear | input / scalar | all | reader produces → compute consumes | legal |
  | Bilinear | output | interleaved | compute produces → writer consumes | legal |
  | Bilinear | output (backed by output buffer) | any input or output sharded | compute produces, nothing consumes | self-loop |

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** borrows quasar grid_sample's `writer_grid_sample_interleaved.cpp` and `grid_sample_reader_common.hpp` (both converted by the joint port), and the main-tree `compute_pool_2d.cpp` (per user decision, both ops bind the grid_sample-local copy `experimental/quasar/grid_sample/device/kernels/compute/compute_pool_2d.cpp`; nothing in the main tree changes). Rotate passes that kernel split_reader = 0, force_4_tile_chunk = 0 and the same placeholder indices grid_sample does, so the copy's interface fits.
- **Indexed-collection args:** none.
- **Two compute KernelSpecs (bilinear, interleaved):** one per core group, differing only in compile-time arg 3 (`rotate_bilinear_program_factory.cpp:307-316`). Keep both.
- **Args read but never used — name them anyway** (recipe rule 4): nearest reader `input_batch` (CTA 2) and `num_cb_pages` (CTA 6); nearest writer `num_cb_pages` (CTA 2).
- **Compute config is Style B:** `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en=false, math_approx_mode=false}` (`rotate_bilinear_program_factory.cpp:299-303`), opt_level unset → O3.

## Team-only

- **Out-of-directory coupling:** ✓ clean. `reader_rotate_bilinear_interleaved.cpp` → `grid_sample_reader_common.hpp` (`read_four_corner_inputs_with_fill(Noc, TensorAccessor, …, DFB, …)` ✓, `fill_four_val<uint32_t>` ✓), `fixed_point_arithmetic.hpp` (scalar math only ✓), `pool_kernels_common.hpp` (`zero_out_page(Noc, DFB)` ✓). Nearest kernels → `pool_kernels_common.hpp` / `experimental_device_api.hpp` (`zero_out_page`, `experimental::local_addr`) ✓. **Who borrows from rotate:** nothing outside this directory references its kernels or factories.
- **TTNN factory analysis:** `descriptor`, SPMD, no op-owned tensors, default hash, no override, no pybound descriptor; target `ProgramSpecFactoryConcept`.

## Misc anomalies

- **Dead CB:** nearest sharded input CB (`rotate_nearest_program_factory.cpp:169-182`) is never referenced by a kernel (the reader reads input through its `TensorAccessor`).
- **Unused reads:** nearest reader `input_batch`, `num_cb_pages`; nearest writer `num_cb_pages`.
- **Sharded input with interleaved output (bilinear):** `any_sharded` makes the output CB borrow `output_tensor.buffer()` even when only the input is sharded (`:136,214`); nothing in `validate_inputs` stops a caller asking for an interleaved output from a sharded input. Worth an owner look.

## Recipe notes

- Same quasar-directory and readiness-proxy notes as the grid_sample audit.
