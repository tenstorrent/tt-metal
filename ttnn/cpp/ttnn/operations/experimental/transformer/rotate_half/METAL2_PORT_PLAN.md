# Port Plan — rotate_half

Port plan for `experimental/transformer/rotate_half`, ported from the `ProgramDescriptor` API (direct
`create_descriptor` on the device op) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, **direct-descriptor shape**. `create_descriptor` is a static method on
  `RotateHalfDeviceOperation` itself (`device/rotate_half_device_operation.hpp:24`, defined at
  `device/rotate_half_program_factory.cpp:16`). There is no `program_factory_t` and no factory struct, so
  [exception 3](ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory) applies.
- Variants: single.
- Custom `compute_program_hash`: none. The default reflection hash covers `RotateHalfParams{output_mem_config}` plus the
  input tensor. There is no `attribute_values` / `to_hash` backdoor.

*(The Metal 2.0 factory concept was chosen during the audit and is carried forward below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_rotate_half_interleaved_start_id.cpp` (own) | `{(0,0)}` | `[0] src_no_mul_cb_index=2`, `[1] src_mul_cb_index=0`, `[2] src_scalar_cb_index=1`, `[3] bf16(-1.0f) bits`, `[4..] TensorAccessorArgs(*src_buffer)` | — | `[0] src_buffer (Buffer*)`, `[1] num_rows`, `[2] half_row_size`, `[3] 0u (start_id)` | — | — | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp` (own) | `{(0,0)}` | `[0] output_no_mul_cb_index=2`, `[1] output_mul_cb_index=16`, `[2..] TensorAccessorArgs(*dst_buffer)` | — | `[0] dst_buffer (Buffer*)`, `[1] num_rows`, `[2] half_row_size`, `[3] 0u (start_id)` | — | — | O2 (unset, DM) | `WriterConfigDescriptor{}` |
| compute | `data_movement/bcast/device/kernels/compute/bcast_hw.cpp` (**borrowed**) | `{(0,0)}` | — | — | `[0] B=1`, `[1] Ht=1`, `[2] Wt=num_tiles/2` | — | `CHAIN_BCAST_OP=compute_kernel_lib::BinaryFpuOp::Mul`, `CHAIN_BCAST_DIM=compute_kernel_lib::BroadcastDim::Scalar`, `BCAST_SCALAR=1` | **O3** (unset, compute) | `ComputeConfigDescriptor{}` (all defaults) |

`grep -n opt_level device/rotate_half_program_factory.cpp` prints nothing, so every level is the resolved default.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`src_mul`, `:40`) | 2 × `single_tile_size` | `{(0,0)}` | `cb_data_format` (input dtype) | `single_tile_size` | — |
| 1 (`src_scalar`, `:53`) | 1 × `scalar_single_tile_size` (Float16_b) | `{(0,0)}` | `cb_data_format` (input dtype) | `scalar_single_tile_size` | — |
| 2 (`src_no_mul` = `output_no_mul`, `:66,88`) | 2 × `single_tile_size` | `{(0,0)}` | `cb_data_format` | `single_tile_size` | — |
| 16 (`output_mul`, `:77`) | 2 × `single_tile_size` | `{(0,0)}` | `cb_data_format` | `single_tile_size` | — |

No GlobalCircularBuffer, no `address_offset`, no aliasing, no borrowed memory.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `rotate_half_program_factory.cpp:97` (`TensorAccessorArgs(*src_buffer)`) → reader `:24,30` | `input` | reader RTA 0 (`src_buffer`, `:109`) |
| `rotate_half_program_factory.cpp:99` (`TensorAccessorArgs(*dst_buffer)`) → writer `:22,28` | `output` (`tensor_return_value`) | writer RTA 0 (`dst_buffer`, `:119`) |

### Work split
n/a. Single core `(0,0)`.

### Shared kernels
- `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp` (**borrowed**). The `_metal2` fork
  `bcast_hw_metal2.cpp` already exists beside it (from the bcast port), so this is **rung 1: reuse; no new file, no edits**.
  Fork binding vocabulary (the constraint the compute `KernelSpec` is built against):
  - DFBs: `dfb::in0` (CONSUMER), `dfb::in1` (CONSUMER), `dfb::out` (PRODUCER)
  - named RTAs: `B`, `Ht`, `Wt`
  - defines: `BCAST_OP`, `BCAST_LLKOP`, `BCAST_DIM` (required), `BCAST_SCALAR` (`#ifdef`: rhs waited once, never popped)
  - no tensor bindings, no CTAs
  - Binders of the legacy original: rotate_half only (`grep -rl bcast_hw.cpp ttnn/cpp/ttnn/operations/`, after
    discarding the fork and the bcast factory, which binds the fork).
- Reader / writer: in this directory, bound only by this op (`grep -rl` finds only this factory), so they convert in place.

### Flags
- No unreferenced kernel files in the op directory.
- Pre-existing anomaly, preserved: CB 1 declares `data_format = cb_data_format` (input dtype) with a `Float16_b`-sized
  page (audit Misc anomalies). The port carries it verbatim into the scalar DFB spec.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none (default hash, left as-is).
- **Implementation notes**: exception 3. Add a nested `RotateHalfProgramFactory` with a static
  `create_program_artifacts(attrs, input, output)` and `using program_factory_t = std::variant<RotateHalfProgramFactory>`,
  and remove `create_descriptor`. The existing cache-hit comment above `create_descriptor` moves with it. Includes:
  drop `program_descriptors.hpp`, add `<variant>` and `ttnn/metal_v2_artifacts.hpp`. No pybind surface is affected:
  `rotate_half_nanobind.cpp` binds only the user function.

## Planned Spec Shape

- **KernelSpecs** (3, 1:1 with legacy):
  - `reader`: own source, converted in place. DFB bindings `in_no_mul` (PRODUCER of `no_mul`), `in_mul` (PRODUCER of
    `src_mul`), `in_scalar` (PRODUCER of `src_scalar`). Tensor binding `src` → `input`. Named CTA `scalar_value`.
    RTAs `num_rows`, `half_row_size`, `start_id`. `hw_config = create_reader_datamovement_config()`; default O2.
  - `writer`: own source, converted in place. DFB bindings `out_no_mul` (CONSUMER of `no_mul`), `out_mul` (CONSUMER of
    `output_mul`). Tensor binding `dst` → `output`. RTAs `num_rows`, `half_row_width`, `start_id`.
    `hw_config = create_writer_datamovement_config()`; default O2.
  - `compute`: `bcast_hw_metal2.cpp` (fork). DFB bindings `in0` (CONSUMER of `src_mul`), `in1` (CONSUMER of
    `src_scalar`), `out` (PRODUCER of `output_mul`). RTAs `B`, `Ht`, `Wt`. Defines
    `bcast_op_utils::get_defines(BcastOpDim::HW, BcastOpMath::MUL)` + `BCAST_SCALAR=1`. `hw_config = ComputeHardwareConfig{}`
    (Style B: legacy `ComputeConfigDescriptor{}` sets nothing, and the defaults coincide). `opt_level = O3`.
- **DataflowBufferSpecs** (4, 1:1 with legacy CBs; all 1P+1C):
  - `src_mul` ← CB 0: `entry_size = single_tile_size`, `num_entries = 2`, `data_format_metadata = dfb_data_format`
  - `src_scalar` ← CB 1: `entry_size = scalar_single_tile_size` (Float16_b), `num_entries = 1`,
    `data_format_metadata = dfb_data_format` (input dtype; mismatch preserved)
  - `no_mul` ← CB 2 (one spec for the two host names `src_no_mul` / `output_no_mul`): `single_tile_size`, 2
  - `output_mul` ← CB 16: `single_tile_size`, 2
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` (`input.mesh_tensor().tensor_spec()`), `output` (`output.mesh_tensor().tensor_spec()`).
- **WorkUnitSpecs**: one, `rotate_half` = {reader, writer, compute} on `{(0,0)}`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none. There is no work-split multiplicity in legacy (single core, one descriptor per kernel).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:109` / reader `:15` RTA 0 | `src_buffer` (`Buffer*`) | `TensorBinding{input, "src"}` → `TensorAccessor(tensor::src)` |
| factory `:119` / writer `:15` RTA 0 | `dst_buffer` (`Buffer*`) | `TensorBinding{output, "dst"}` → `TensorAccessor(tensor::dst)` |
| factory `:97` / reader `:24` | `TensorAccessorArgs(*src_buffer).append_to(cta)` / `TensorAccessorArgs<4>()` | binding token |
| factory `:99` / writer `:22` | `TensorAccessorArgs(*dst_buffer).append_to(cta)` / `TensorAccessorArgs<2>()` | binding token |
| factory `:96` / reader `:20` CTA 0 | `src_no_mul_cb_index = 2` | `DFBBinding{no_mul, "in_no_mul", PRODUCER}` |
| factory `:96` / reader `:21` CTA 1 | `src_mul_cb_index = 0` | `DFBBinding{src_mul, "in_mul", PRODUCER}` |
| factory `:96` / reader `:22` CTA 2 | `src_scalar_cb_index = 1` | `DFBBinding{src_scalar, "in_scalar", PRODUCER}` |
| factory `:98` / writer `:20` CTA 0 | `output_no_mul_cb_index = 2` | `DFBBinding{no_mul, "out_no_mul", CONSUMER}` |
| factory `:98` / writer `:21` CTA 1 | `output_mul_cb_index = 16` | `DFBBinding{output_mul, "out_mul", CONSUMER}` |
| compute (`bcast_hw.cpp`) hardcoded `c_0` / `c_1` / `c_16` | magic CB indices | fork's `dfb::in0` / `in1` / `out` |
| factory `:96` / reader `:23` CTA 3 | positional `scalar_value` | named CTA `scalar_value` |
| reader `:16-18` RTAs 1-3 | positional | named `num_rows`, `half_row_size`, `start_id` |
| writer `:16-18` RTAs 1-3 | positional | named `num_rows`, `half_row_width`, `start_id` |
| compute RTAs 0-2 | positional | named `B`, `Ht`, `Wt` (fork vocabulary) |
| reader `:27`, writer `:25` | `get_tile_size(cb_id)` (non-`constexpr` `const`) | `dfb_in_no_mul.get_tile_size()` / `dfb_out_no_mul.get_tile_size()` (rule 7) |

No page-size 3rd-arg plumbing (none existed). No semaphore plumbing.

## Applied Patterns

- [Caution: Porting a shared kernel](port_patterns.md#caution-porting-a-shared-kernel), rung 1: bind the existing
  `bcast_hw_metal2.cpp` fork, adopting its vocabulary. Translate the legacy `CHAIN_*` defines to the fork's
  `BCAST_OP` / `BCAST_LLKOP` / `BCAST_DIM` (the fork lags its original's `eltwise_chain` migration, #55584).
- [Pattern: Pass DFB handles directly to LLKs](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  the fork already does it. Nothing new here.
- Exception 3 (direct-descriptor op → nested program factory), `ttnn_factory.md`.
- A DFB consumed with `wait_front` and never popped (`src_scalar` under `BCAST_SCALAR`) is intentional fill-once reuse
  and is left as-is.

## Deferred / Flagged

- The compute body changes code path (legacy `eltwise_chain` → the fork's hand-rolled `bcast_init` / `mul_tiles_bcast`
  loop). It is the same math. The test compares with `torch.equal`, so a numerics drift would show up as a failure.
- `src_scalar` DFB format/size mismatch for non-bf16 inputs (pre-existing; preserved; untested).
- No new findings beyond the audit.
