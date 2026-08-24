# Metal 2.0 Port Report — reshape_view

## Outcome

**PORTED** — both factories (`ReshapeViewRMProgramFactory`, `ReshapeViewTiledProgramFactory`) fully
converted to `ProgramSpecFactoryConcept`; all three kernels converted. The whole op is on Metal 2.0
(no factory left behind). Verified with the Metal 2.0 host-side legality checks forced on and proven
live (`METAL2_CHECKS_FORCED` present in three of the four test logs — 278 / 510 / 308 markers). All
confirmed tests pass with counts **identical** to the pre-port baseline:

| Test file | Pre-port | Post-port |
|---|---|---|
| `test_tm_reshape.py` | 49 passed | 49 passed |
| `test_universal_input_tm_reshape.py` | 344 passed | 344 passed |
| `base_functionality/test_reshape.py` | 333 passed, 8 skipped, 7 xfailed | 333 passed, 8 skipped, 7 xfailed |
| `misc/test_reshape.py` | 3 passed | 3 passed (host path — 0 markers, does not route through the device op) |

## Provenance

- **Recipe docs (this port):** `c9b98ecf065 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `355760227dd 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for **both** factories, exactly as the audit chose. Each factory struct
(`ReshapeViewRMProgramFactory`, `ReshapeViewTiledProgramFactory`) now defines `create_program_artifacts`
returning `ProgramArtifacts`; both already lived in the device-op's `program_factory_t` variant, so no
direct-descriptor conversion (exception 3) was needed. No `override_runtime_arguments` on either → base
concept, no cache-hit override to translate.

- **RM factory**: single-program, no op-owned tensors. `create_descriptor` → `create_program_artifacts`.
- **Tiled factory**: single-program with **one op-owned tensor** (the page-mapping tensor). The legacy
  `WorkloadDescriptor`/`tensor_coords` replication ("secretly SPMD") collapsed to a single `ProgramSpec`
  stamped across the mesh by the framework adapter — the per-coord replication loop was dropped. The
  mapping tensor's owning `MeshTensor` is moved out via `device_storage().release_mesh_tensor()` into
  `ProgramArtifacts::op_owned_tensors` and bound as a `TensorParameter` (`map`).

### Device-op-class edits
- Pybind entry points removed: none (no pybound `create_descriptor`).
- Custom `compute_program_hash`: left intact at `reshape_device_operation.cpp:48-63` — untouched.
- `select_program_factory` / `validate` / `compute_output_specs` / device-op class: untouched.

### Open items
- Relaxation candidates: none noticed. The custom hash pins the whole `TensorSpec` (no relaxation), and
  the audit declared `TensorParameter relaxation = none`; the strict default is kept.

## Handoff points

None. The port stayed entirely within the op directory; no shared-kernel fork, no out-of-op edit, no
capitulation. (The temporary `skip_validation` forcing in `tt_metal/impl/metal2_host_api/` is
working-tree scaffolding only and is reverted before commit — never part of the diff.)

## Successes

- **[Sync-free / single-ended CB → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**
  fired correctly for every scratch buffer: RM `src0`–`src3` (each a single-toucher L1 staging ring,
  `rm_reshape_interleaved.cpp:82-87`) and tiled `working` (writer-only scratch page,
  `writer_reshape_tiled.cpp:32-33,78`). The one-toucher census made these unambiguous self-loops.
- **RM dual-instance work-split with *disjoint* CBs**: the brief flagged this as the shape to watch, and
  it steered the port right — because the reader/writer instances touch disjoint DFBs (src0/1 vs src2/3),
  there is no shared DFB and no 1P+1C / multi-binding question; each DFB self-loops. The
  [two-toucher pattern](../shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)'s
  "distinguish from disjoint-node" note confirmed this is neither the CTA-demotion case nor the
  shared-grid two-toucher case.
- **Conditional DFB binding without kernel `#ifdef`**: RM's `src2`/`src3` + writer KernelSpec are gated
  on `can_use_dual_kernel` purely host-side. Because both RM instances compile the *same* source with
  accessor names `in0`/`in1` (each mapped to a distinct DFB spec per instance), the kernel is identical
  and unconditional — no preprocessor gating needed. This is a cleaner shape than the general
  conditional-binding pattern (which needs `#ifdef`) because the conditionality lives in *which
  KernelSpec exists*, not in a within-kernel code path.
- **Op-owned tensor path** (first real exercise per the recipe): `release_mesh_tensor()` +
  `op_owned_tensors` + bind-against-the-vector-element worked as documented; `reserve(1)` up front,
  bound `op_owned.back()` immediately.

## Friction

### Gaps
- **Dropping `TensorAccessorArgs` loses the `is_dram` compile-time constant, and the docs don't say how
  to recover it.** The RM kernel branches on `src_args.is_dram` inside `if constexpr`
  (`rm_reshape_interleaved.cpp:108,114`). `src_args` (a `TensorAccessorArgs<N>`) is dropped in the port,
  but `TensorAccessor` exposes the same value as a `static constexpr bool is_dram`
  (`tensor_accessor.h:58`). The whitelist / migration guide cover `TensorAccessor(tensor::name)`
  construction but not "how do I read the accessor's compile-time layout flags after the args object is
  gone." Resolved by using `decltype(s)::is_dram` (a `TensorAccessor` instance's type), which is
  unambiguously a constant expression. Worth a one-line note in the TensorAccessor section of the
  migration guide: *compile-time accessor properties (`is_dram`, …) move from `args.is_dram` to
  `decltype(accessor)::is_dram`.*

### Confusion
- **Two mapping concepts share the word "mapping" in the tiled factory** — the op-owned *mapping tensor*
  (`tensor::map`, a `TensorParameter`) and the *mapping DFB* (`dfb::mapping`, a staging FIFO) are
  distinct resources. The audit/brief name both "mapping"; it was briefly easy to conflate them. Named
  them distinctly in the port (`MAPPING_T` / `map` for the tensor, `MAPPING` / `mapping` for the DFB).

## Open items for downstream

- **RM `write_start_offset` (RTA slot 6) is a dead-valued RTA** — the factory always passes `0u`
  (`reshape_rm_program_factory.cpp:266,270,281`) yet the kernel still reads it and folds it into
  `writable` / `write_offset` (`rm_reshape_interleaved.cpp:66,95,106`). Kept as a named RTA in the port
  (faithful, zero behavior change). Candidate for removal in a separate cleanup.
- **RM `source_read_size_bytes` (RTA slot 2) is identical on every node** — a CRTA candidate. Kept as a
  named RTA (RTA→CRTA changes dispatch semantics; out of scope for the port).
- **`recreate_mapping_tensor` op attribute is accepted but unused** (`reshape_device_operation_types.hpp:18`,
  ignored at `reshape_tiled_program_factory.cpp:463-466`, excluded from the hash). Pre-existing; not
  touched by the port. Route to ops team.
- **Shared kernel touches**: none — all three kernels are owned by `reshape_view`; no fork created or reused.
- **Stale comment in `reshape.cpp:441`** (op entry point, outside the factory body → off-limits): the
  comment "TILED factories use TensorAccessorArgs for transparent sharded I/O" now misdescribes the
  mechanism (the tiled factory uses `TensorParameter`/`TensorBinding`, not `TensorAccessorArgs`). Left
  unchanged per host-side scope discipline; flag for a follow-up touch by the op owner.

## Verification notes

- **TT_FATAL/TT_ASSERT census**: the tiled factory dropped 2 guards (16 → 14), both the canonical
  legitimate subject-deleted case — `TT_ASSERT(input_buffer != nullptr)` / `TT_ASSERT(output_buffer !=
  nullptr)` on the raw `Buffer*`s that `TensorParameter`/`TensorBinding` replaced. Every other guard
  (the `detail::` mapping-compute asserts, the volume/rank/num_cores checks) is preserved verbatim. RM
  factory census unchanged — its `dst_buffer` null-assert stayed (the `Buffer*` is still used for
  `is_dram()`, not as a binding).
- **hw_config**: all four DM kernels reproduce the legacy resolved config — RM/tiled reader =
  `create_reader_datamovement_config` (reader default RISCV_1/NOC_0/DM_DEDICATED), RM/tiled writer =
  `create_writer_datamovement_config` (writer default RISCV_0/NOC_1/DM_DEDICATED). No compute kernels,
  so no `unpack_modes` / `bfp_pack_precision_mode` / compute-config concerns.
- **opt_level**: not set on any KernelSpec — correct. All kernels are DM (legacy `opt_level` unset →
  `O2`, which is Metal 2.0's default). No compute kernel exists, so the compute-`O3` rule does not apply.
