# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/upsample`

**Audited unit:** `UpsampleOperation::UpsampleBilinearProgramFactory` only. The op's directory bundles **one** DeviceOperation (`UpsampleOperation`) with four program factories; the other three are **already Metal 2.0** (`create_program_artifacts` → `ProgramArtifacts`) and are out of scope for this audit per the request.

- **`UpsampleOperation`**
  - `UpsampleBilinearProgramFactory` (`upsample_bilinear_program_factory_multicore.cpp`) — **audited here** (`descriptor` concept)
  - `UpsampleMultiCoreInterleavedProgramFactory` (`upsample_program_factory_multicore_interleaved.cpp`) — already MetalV2, not audited
  - `UpsampleMultiCoreShardedProgramFactory` (`upsample_program_factory_multicore_sharded.cpp`) — already MetalV2, not audited
  - `UpsampleNearestFloatProgramFactory` (`upsample_nearest_float_program_factory.cpp`) — already MetalV2, not audited

**Kernels referenced by the bilinear factory:**
- `device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp` — instantiated **twice** (reader + writer config), same source, branched on the `is_reader` CTA (dual-instance work-split).
- `device/kernels/compute/bilinear.cpp`
- Header deps: `device/kernels/dataflow/bilinear_weights_lut.hpp` (local), plus in-family shared helpers `pool/device/kernels/experimental_device_api.hpp` and `pool/device/kernels/fixed_point_arithmetic.hpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `23861284522 2026-09-08 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/upsample` |
| **Overall** | **GREEN** (for `UpsampleBilinearProgramFactory`) |
| **DOps / Factories** | `UpsampleOperation` → `UpsampleBilinearProgramFactory` (the 3 sibling factories are already MetalV2) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all kernels on Device 2.0 idioms (`DataflowBuffer`, `Noc`, `UnicastEndpoint`, `experimental::local_addr`); `get_local_cb_interface` is sanctioned |
| *Prereqs* — Cross-op escapes | Ok — in-family shared helpers only, all Device 2.0 native |
| *Feature Support* — overall | **GREEN** (all N/A) |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet), cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (concept is `descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (no `->address()` RTA at all) |
| *Port work* — Tensor bindings (per binding) | clean (both borrowed-memory DFB): `input`/halo, `output` |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | N/A — no `TensorAccessor` in the bilinear kernels |
| *Port work* — CB endpoints | self-loop ×1 · 1P+1C ×1 · legal 1:1 ×4 (no multi-binding, no dead CB) |

## Result

**GREEN → brief issued.** `UpsampleBilinearProgramFactory` clears every gate and is ready for a Metal 2.0 port:
- Device 2.0 ✓ · Features ✓ · TTNN factory concept (`Is able to port? = yes`) ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A).

The port is a **binding-layer rewrite** of an already-modernized op: the kernels are already on Device-2.0/Metal-2.0 kernel idioms (`DataflowBuffer`, `Noc`, `experimental::local_addr`); the host `create_descriptor` factory is the piece that migrates to a `ProgramSpecFactoryConcept` spec, and CB IDs currently passed as CTAs become `dfb::name` tokens.

(The three sibling factories are already MetalV2 and were not audited.)

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Sheet cell `Is able to port? = yes`. Cross-check against code all clean:
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `ProgramDescriptor` (`upsample_bilinear_program_factory_multicore.cpp:30`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override in the device-op.
  - `get_dynamic_runtime_args = no` ✓ — hook absent.
  - `Override runtime args method? = no` ✓ — no `override_runtime_arguments` → base `ProgramSpecFactoryConcept`.
  - `Pybind descriptor = no` ✓ — `upsample_nanobind.cpp` binds only the top-level `upsample` function (`upsample_nanobind.cpp:68`), no `create_descriptor` binding.
  - `Smuggled pointer = no` ✓ — no buffer-address RTA (see Offset base pointers / Tensor bindings).
  - `Op-owned tensors = <empty>` ✓ — the bilinear factory declares none (op-owned tensors live only in the sibling `UpsampleMultiCoreShardedProgramFactory`, which is already MetalV2).
  - Factory-set match ✓ — 4 sheet rows ↔ 4 code factories, one-to-one, no phantom/missing rows.
- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the factory instantiates is on Device 2.0 idioms:
  - `reader_bilinear_multi_core_sharded.cpp` — `DataflowBuffer` objects, `Noc noc;`, `UnicastEndpoint`, `noc.async_read(...)` / `noc.async_read_barrier()`, `experimental::local_addr(...)`, `dfb.reserve_back/push_back/get_read_ptr/get_write_ptr`. Includes the Metal-2.0 `api/dataflow/dataflow_buffer.h` (not the stale legacy CB header).
  - `bilinear.cpp` — `DataflowBuffer` objects, `dfb.wait_front/pop_front`, `tile_regs_*`, `pack_untilize_dest`. The one CB-index free function it uses — `get_local_cb_interface(operand)` in `llk_push_pages_bilinear` (`bilinear.cpp:18–24`) — is **sanctioned** by the Green bullet (Device 2.0 keeps it as a free function); not a holdover.
  - In-family shared helpers (`experimental_device_api.hpp`, `fixed_point_arithmetic.hpp`) are Device 2.0 native (`Noc`, `UnicastEndpoint`, `CoreLocalMem`, pure fixed-point math). No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read` / raw sem addresses anywhere.

- **Feature compatibility:** all entries N/A — no gate feature is present.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `.global_circular_buffer` field, no `remote_index`, no `CreateGlobalCircularBuffer`, no remote-CB idiom. All CBs are plain `CBDescriptor`, two of them borrowed-memory (`.buffer` set). The bare `#include <tt-metalium/global_circular_buffer.hpp>` in `upsample_device_operation.hpp:17` is header-presence-only (suggestive, not definitive) and pairs with no other signal in this factory. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` set on any `CBDescriptor`; borrowed CBs use base (offset 0). |
  | GlobalSemaphore | N/A | The bilinear factory uses **no** semaphores at all. |

- **CB endpoints (GATE-free):** every CB has a legal disposition; no multi-binding, no dead CB, single config (`HEIGHT_SHARDED` bf16, integer scale — the factory's only supported config, enforced at `upsample_device_operation.cpp:56–68`). Census per node:

  | CB | Index | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `halo_cb` (borrowed input, `.buffer = halo_in.buffer()`) | c_0 | reader (raw read `halo_dfb.get_read_ptr()`), writer (same, other instance) — both role-free; compute doesn't touch | 2 role-free | **1P+1C** (bind one instance PRODUCER, the other CONSUMER; cosmetic on Gen1) |
  | `tilize_reduce_cb_0` | c_1 | reader FIFO-produces, compute FIFO-consumes | 1 locked producer + 1 locked consumer | **legal 1:1** |
  | `tilize_reduce_cb_1` | c_2 | writer FIFO-produces, compute FIFO-consumes | 1 locked producer + 1 locked consumer | **legal 1:1** |
  | `in_scalar_cb_id1` | c_3 | reader produces (`get_write_ptr`+`push_back`), compute consumes | 1P + 1C | **legal 1:1** |
  | `in_scalar_cb_id2` | c_4 | writer produces, compute consumes | 1P + 1C | **legal 1:1** |
  | `out_cb` (borrowed output, `.buffer = output.buffer()`) | c_5 | compute only (`pack_untilize_dest` + `llk_push_pages_bilinear` raw `fifo_wr_ptr` advance = write-cursor driver, locked producer); nothing consumes | 1 toucher | **self-loop** (bind compute PRODUCER **and** CONSUMER; legal on Gen1 for compute) |

  Note: reader/writer are the same kernel source instantiated over the same core range (dual-instance work-split, face (c)). The halo CB is the classic two-role-free-toucher case → 1P+1C, **not** multi-binding. There is no hidden second writer: the `in_scalar`/`tilize_reduce` CBs each have exactly one FIFO producer and one FIFO consumer, split cleanly between the reader-instance CBs (c_1/c_3) and writer-instance CBs (c_2/c_4).

- **Offset base pointers:** **GREEN — cleared.** No address RTA at all: the factory's runtime args are `{start_output_idx, min_input_offset, out_sticks_this_core}` (reader/writer) and `{nsticks_per_core}` (compute) — all shape/index/count scalars, no `buffer()->address()`. `min_input_offset` is a *stick* offset into the halo layout (used kernel-side to derive `halo_starting_row/col_offset` from the L1 base `halo_dfb.get_read_ptr()`, `reader_bilinear_multi_core_sharded.cpp:117–120`), **not** a `base + offset` device-pointer fold. Both input and output reach the kernel through borrowed-memory CBs, so no base is smuggled through an RTA. Not in the offset-base triage tables; scan confirms clean.

- **TensorAccessor 3rd argument:** **N/A** — no `TensorAccessor` is constructed anywhere in the bilinear kernels (the borrowed-memory CBs carry both input and output; addressing is raw L1 offset arithmetic). The subject never fires. (The `TensorAccessor(tensor::...)` sites elsewhere in `device/kernels/dataflow/` belong to the already-MetalV2 sibling factories, out of scope.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): both **clean** (borrowed-memory DFB) —
  - `input` (halo) → borrowed-memory DFB `c_0`, `.buffer = halo_in.buffer()`, read raw via `halo_dfb.get_read_ptr()`; port via `DataflowBufferSpec::borrowed_from`.
  - `output` → borrowed-memory DFB `c_5`, `.buffer = output.buffer()`, written by compute; port via `DataflowBufferSpec::borrowed_from`.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none — no accessor in scope.
- **CB endpoints:** self-loop `out_cb (c_5)` · 1P+1C `halo_cb (c_0)` · legal 1:1 `tilize_reduce_cb_0/1 (c_1/c_2)`, `in_scalar_cb_id1/2 (c_3/c_4)`. Single config, no per-config flips.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB needs the multi-binding flag. The dual-instance work-split touches only `halo_cb` co-resident (both instances raw-read it) → resolves to 1P+1C, not a flag. Confirmed there is no hidden semaphore-gated co-fill (the factory declares no semaphores).
- **Cross-op / shared kernels:** the bilinear kernels (`reader_bilinear_multi_core_sharded.cpp`, `compute/bilinear.cpp`) are **owned by this op** and instantiated by **no other op** — no `_metal2` fork exists or is needed, no sunset list. The kernels do `#include` in-family shared *header* helpers (`experimental_device_api.hpp`, `fixed_point_arithmetic.hpp`) — headers, not file-path kernel instantiation; all Device 2.0 native, no donor-shape concern.
- **RTA varargs:** none — every RTA/CRTA is a fixed named scalar read at a constant index (`reader_bilinear_multi_core_sharded.cpp:263–265`, `bilinear.cpp:64`); all CTAs fixed-index. Port names each; no vararg mechanism needed.
- **Kernel already modernized:** the port is a *binding-layer* change, not an idiom rewrite. The kernels already use `DataflowBuffer`/`Noc`. The main porter work is: CB-ID CTAs → `dfb::name` tokens, the two borrowed CBs → `borrowed_from` a `TensorParameter`, and the per-core runtime-arg loop → a `ProgramRunArgs` schedule.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No cross-family donor kernels. In-family header includes only:

  | Op kernel | Donor header | Class | Shape |
  |---|---|---|---|
  | reader, compute | `pool/device/kernels/experimental_device_api.hpp` | in-family shared pool | Device 2.0 aliases/wrappers (`local_addr`, `Noc` helpers); free functions, no CB/sem-handle signatures. `using CB = CircularBuffer` alias present but the bilinear kernels use `DataflowBuffer` directly. ✓ |
  | reader, compute, LUT | `pool/device/kernels/fixed_point_arithmetic.hpp` | in-family shared pool | pure fixed-point math, no resource handles. ✓ |

  No file-path kernel borrow (both `.cpp` kernels are op-owned).
- **TTNN factory analysis:** current concept `descriptor`; target `ProgramSpecFactoryConcept` (no op-owned tensors; `Override runtime args method? = no`). No custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor`. Sheet `Porting Target` column independently confirms `ProgramSpecFactoryConcept`.
- **Relaxation candidates:** none noticed (no custom hash to mine).

## Misc anomalies  *(team-only, non-gating, not porter work)*

- **Dead compute CTAs.** `bilinear.cpp` reads CTA[6] `in_ntiles_hwc` (`= 1 * in_ntiles_c`) and CTA[7] `window_size_hw` (`= 4`) into `constexpr` locals that are **never used** (`bilinear.cpp:73–74`; the factory still passes them at `upsample_bilinear_program_factory_multicore.cpp:258–259`). Also `num_output_tiles` (`bilinear.cpp:86`) is computed but unused. Zero runtime cost (compile-time constants), but dead. The port may carry them across unchanged; flagging for the ops team to prune independently.
- **`MAX_TILES_PER_REDUCTION = 8` duplicated** as a literal in both the factory (`upsample_bilinear_program_factory_multicore.cpp:83`) and the compute kernel (`bilinear.cpp:79`). Not a bug; a single source of truth would be cleaner.

## Recipe notes  *(none)*

The recipe covered every construct cleanly. Worth recording for the next auditor of this directory: the `ide_selection` context this session opened with showed `metal2::AddRuntimeArgsForNode(...)` calls in `upsample_bilinear_program_factory_multicore.cpp` — those do **not** exist in the current file (it is still on `create_descriptor` + `KernelDescriptor::runtime_args`), so that selection was stale/from a different working state and had no bearing on the audit.
